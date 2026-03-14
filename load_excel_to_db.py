"""Ingestion Excel -> SQLite pour les données NBA.

Schema relationnel cible:
- players(player_id PK, ...)
- matches(match_id PK, team_code UNIQUE, ...)
- stats(stat_id PK, player_id FK -> players, match_id FK -> matches)
- reports(report_id PK)

Ce script lit l'Excel principal, valide les lignes
avec Pydantic, recrée les tables et charge les données en base.
La table `reports` est alimentée depuis les PDF Reddit (`inputs/Reddit *.pdf`).
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from PyPDF2 import PdfReader
from pydantic import BaseModel, Field, ValidationError, field_validator

from utils.config import DATABASE_DIR

EXCEL_CANDIDATES = [
    Path("inputs/regular NBA.xlsx"),
    Path("matchs/regular+NBA.xlsx"),
]
PDF_GLOB_PATTERNS = [
    "Reddit *.pdf",
    "reddit *.pdf",
    "Reddit*.pdf",
    "reddit*.pdf",
]
DB_PATH = Path(DATABASE_DIR) / "nba_data.db"


class PlayerRow(BaseModel):
    player_name: str = Field(min_length=1)
    team_code: str = Field(min_length=2, max_length=4)
    age: int | None = Field(default=None, ge=15, le=60)
    gp: int | None = Field(default=None, ge=0)
    wins: int | None = Field(default=None, ge=0)
    losses: int | None = Field(default=None, ge=0)
    minutes: float | None = Field(default=None, ge=0)
    points_total: float | None = Field(default=None, ge=0)
    fg_pct: float | None = Field(default=None, ge=0, le=100)
    three_pt_pct: float | None = Field(default=None, ge=0, le=100)
    ft_pct: float | None = Field(default=None, ge=0, le=100)
    rebounds: float | None = Field(default=None, ge=0)
    assists: float | None = Field(default=None, ge=0)
    steals: float | None = Field(default=None, ge=0)
    blocks: float | None = Field(default=None, ge=0)
    turnovers: float | None = Field(default=None, ge=0)
    off_rating: float | None = None
    def_rating: float | None = None
    net_rating: float | None = None
    usage_pct: float | None = None
    pie: float | None = None

    @field_validator("player_name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        return value.strip()

    @field_validator("team_code")
    @classmethod
    def _clean_code(cls, value: str) -> str:
        return value.strip().upper()


class MatchRow(BaseModel):
    team_code: str = Field(min_length=2, max_length=4)
    team_name: str = Field(min_length=1)
    players_count: int | None = Field(default=None, ge=0)
    team_points_total: float | None = Field(default=None, ge=0)
    team_games_played: int | None = Field(default=None, ge=0)
    team_wins: int | None = Field(default=None, ge=0)
    team_losses: int | None = Field(default=None, ge=0)

    @field_validator("team_code")
    @classmethod
    def _normalize_team_code(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("team_name")
    @classmethod
    def _normalize_team_name(cls, value: str) -> str:
        return value.strip()


class StatRow(BaseModel):
    player_id: int = Field(ge=1)
    match_id: int | None = Field(default=None, ge=1)
    stat_key: str = Field(min_length=1)
    stat_value: float | None = None
    unit: str | None = None
    source_sheet: str = Field(default="Données NBA", min_length=1)


class ReportRow(BaseModel):
    report_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_sheet: str = Field(min_length=1)
    row_order: int = Field(ge=0)


STAT_COLUMNS: list[tuple[str, str, str | None]] = [
    ("points_total", "points_total", "points"),
    ("fg_pct", "fg_pct", "percent"),
    ("three_pt_pct", "three_pt_pct", "percent"),
    ("ft_pct", "ft_pct", "percent"),
    ("rebounds", "rebounds", "count"),
    ("assists", "assists", "count"),
    ("steals", "steals", "count"),
    ("blocks", "blocks", "count"),
    ("turnovers", "turnovers", "count"),
    ("off_rating", "off_rating", None),
    ("def_rating", "def_rating", None),
    ("net_rating", "net_rating", None),
    ("usage_pct", "usage_pct", "percent"),
    ("pie", "pie", "percent"),
]


def _resolve_excel_path() -> Path:
    for candidate in EXCEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Aucun fichier Excel trouve. Attendu: "
        + ", ".join(str(path) for path in EXCEL_CANDIDATES)
    )


def _resolve_reddit_pdf_paths() -> list[Path]:
    inputs_dir = Path("inputs")
    if not inputs_dir.exists():
        return []

    pdf_paths: list[Path] = []
    for pattern in PDF_GLOB_PATTERNS:
        pdf_paths.extend(sorted(inputs_dir.glob(pattern)))

    # Déduplication stable
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in pdf_paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _to_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL UNIQUE,
            team_code TEXT NOT NULL,
            age INTEGER,
            gp INTEGER,
            wins INTEGER,
            losses INTEGER,
            minutes REAL,
            points_total REAL,
            fg_pct REAL,
            three_pt_pct REAL,
            ft_pct REAL,
            rebounds REAL,
            assists REAL,
            steals REAL,
            blocks REAL,
            turnovers REAL,
            off_rating REAL,
            def_rating REAL,
            net_rating REAL,
            usage_pct REAL,
            pie REAL
        );

        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_code TEXT NOT NULL UNIQUE,
            team_name TEXT NOT NULL,
            players_count INTEGER,
            team_points_total REAL,
            team_games_played INTEGER,
            team_wins INTEGER,
            team_losses INTEGER,
            source_sheet TEXT NOT NULL DEFAULT 'Analyse'
        );

        CREATE TABLE IF NOT EXISTS stats (
            stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            match_id INTEGER,
            stat_key TEXT NOT NULL,
            stat_value REAL,
            unit TEXT,
            source_sheet TEXT NOT NULL,
            FOREIGN KEY(player_id) REFERENCES players(player_id) ON DELETE CASCADE,
            FOREIGN KEY(match_id) REFERENCES matches(match_id) ON DELETE SET NULL,
            UNIQUE(player_id, match_id, stat_key)
        );

        CREATE TABLE IF NOT EXISTS reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source_sheet TEXT NOT NULL,
            row_order INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_code);
        CREATE INDEX IF NOT EXISTS idx_matches_team ON matches(team_code);
        CREATE INDEX IF NOT EXISTS idx_stats_player ON stats(player_id);
        CREATE INDEX IF NOT EXISTS idx_stats_match ON stats(match_id);
        """
    )


def _clear_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DELETE FROM stats;
        DELETE FROM reports;
        DELETE FROM players;
        DELETE FROM matches;
        """
    )


def _load_players(excel_path: Path) -> list[PlayerRow]:
    df = pd.read_excel(excel_path, sheet_name="Données NBA", header=1)
    df = df.dropna(subset=["Player", "Team"]).copy()

    players: list[PlayerRow] = []
    for _, row in df.iterrows():
        payload = {
            "player_name": row.get("Player"),
            "team_code": row.get("Team"),
            "age": _to_int(row.get("Age")),
            "gp": _to_int(row.get("GP")),
            "wins": _to_int(row.get("W")),
            "losses": _to_int(row.get("L")),
            "minutes": _to_float(row.get("Min")),
            "points_total": _to_float(row.get("PTS")),
            "fg_pct": _to_float(row.get("FG%")),
            "three_pt_pct": _to_float(row.get("3P%")),
            "ft_pct": _to_float(row.get("FT%")),
            "rebounds": _to_float(row.get("REB")),
            "assists": _to_float(row.get("AST")),
            "steals": _to_float(row.get("STL")),
            "blocks": _to_float(row.get("BLK")),
            "turnovers": _to_float(row.get("TOV")),
            "off_rating": _to_float(row.get("OFFRTG")),
            "def_rating": _to_float(row.get("DEFRTG")),
            "net_rating": _to_float(row.get("NETRTG")),
            "usage_pct": _to_float(row.get("USG%")),
            "pie": _to_float(row.get("PIE")),
        }
        try:
            players.append(PlayerRow.model_validate(payload))
        except ValidationError:
            continue
    return players


def _group_team_outcomes(players: list[PlayerRow]) -> dict[str, dict[str, int | None]]:
    outcomes: dict[str, dict[str, int | None]] = {}
    by_team: dict[str, list[PlayerRow]] = {}
    for player in players:
        by_team.setdefault(player.team_code, []).append(player)

    for code, team_players in by_team.items():
        gp_values = [p.gp for p in team_players if p.gp is not None]
        win_values = [p.wins for p in team_players if p.wins is not None]
        loss_values = [p.losses for p in team_players if p.losses is not None]
        outcomes[code] = {
            "team_games_played": max(gp_values) if gp_values else None,
            "team_wins": max(win_values) if win_values else None,
            "team_losses": max(loss_values) if loss_values else None,
        }
    return outcomes


def _load_matches(excel_path: Path, players: list[PlayerRow]) -> list[MatchRow]:
    raw = pd.read_excel(excel_path, sheet_name="Analyse", header=None)
    team_outcomes = _group_team_outcomes(players)

    header_idx = None
    for i, value in raw.iloc[:, 0].items():
        if str(value).strip().lower() == "code":
            header_idx = i
            break
    if header_idx is None:
        return []

    matches: list[MatchRow] = []
    for idx in range(header_idx + 1, len(raw)):
        code = str(raw.iat[idx, 0]).strip()
        if not re.fullmatch(r"[A-Z]{2,4}", code):
            break
        team_name = str(raw.iat[idx, 1]).strip()
        players_count = _to_int(raw.iat[idx, 2])
        team_points_total = _to_float(raw.iat[idx, 3])
        inferred = team_outcomes.get(code, {})

        payload = {
            "team_code": code,
            "team_name": team_name,
            "players_count": players_count,
            "team_points_total": team_points_total,
            "team_games_played": inferred.get("team_games_played"),
            "team_wins": inferred.get("team_wins"),
            "team_losses": inferred.get("team_losses"),
        }
        try:
            matches.append(MatchRow.model_validate(payload))
        except ValidationError:
            continue
    return matches


def _load_reports() -> list[ReportRow]:
    """Charge les reports depuis les PDF Reddit fournis dans `inputs/`."""
    reports: list[ReportRow] = []
    pdf_paths = _resolve_reddit_pdf_paths()

    row_order = 0
    for pdf_path in pdf_paths:
        file_had_text = False
        try:
            reader = PdfReader(str(pdf_path))
        except Exception:
            reader = None

        if reader is not None:
            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    content = (page.extract_text() or "").strip()
                except Exception:
                    content = ""
                if not content:
                    continue

                file_had_text = True
                normalized = re.sub(r"\s+", " ", content).strip()
                reports.append(
                    ReportRow(
                        report_type="reddit_pdf",
                        title=f"{pdf_path.stem} - page {page_idx}",
                        content=normalized,
                        source_sheet=pdf_path.name,
                        row_order=row_order,
                    )
                )
                row_order += 1

        # Fallback OCR si PDF image/scanné (cas fréquent sur les exports Reddit).
        if file_had_text:
            continue
        try:
            from utils.data_loader import extract_text_from_pdf

            ocr_text = (extract_text_from_pdf(str(pdf_path)) or "").strip()
        except Exception:
            ocr_text = ""
        if not ocr_text:
            page_count = len(reader.pages) if reader is not None else 1
            for page_idx in range(1, page_count + 1):
                reports.append(
                    ReportRow(
                        report_type="reddit_pdf_unreadable",
                        title=f"{pdf_path.stem} - page {page_idx}",
                        content="Aucun texte extractible (PDF image/scanné, OCR indisponible).",
                        source_sheet=pdf_path.name,
                        row_order=row_order,
                    )
                )
                row_order += 1
            continue

        normalized = re.sub(r"\s+", " ", ocr_text).strip()
        reports.append(
            ReportRow(
                report_type="reddit_pdf_ocr",
                title=f"{pdf_path.stem} - OCR complet",
                content=normalized,
                source_sheet=pdf_path.name,
                row_order=row_order,
            )
        )
        row_order += 1

    return reports


def _insert_matches(conn: sqlite3.Connection, matches: list[MatchRow]) -> None:
    conn.executemany(
        """
        INSERT INTO matches (
            team_code, team_name, players_count, team_points_total,
            team_games_played, team_wins, team_losses, source_sheet
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'Analyse')
        ON CONFLICT(team_code) DO UPDATE SET
            team_name = excluded.team_name,
            players_count = excluded.players_count,
            team_points_total = excluded.team_points_total,
            team_games_played = excluded.team_games_played,
            team_wins = excluded.team_wins,
            team_losses = excluded.team_losses;
        """,
        [
            (
                row.team_code,
                row.team_name,
                row.players_count,
                row.team_points_total,
                row.team_games_played,
                row.team_wins,
                row.team_losses,
            )
            for row in matches
        ],
    )


def _insert_players(conn: sqlite3.Connection, players: list[PlayerRow]) -> None:
    conn.executemany(
        """
        INSERT INTO players (
            player_name, team_code, age, gp, wins, losses, minutes, points_total,
            fg_pct, three_pt_pct, ft_pct, rebounds, assists, steals, blocks,
            turnovers, off_rating, def_rating, net_rating, usage_pct, pie
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_name) DO UPDATE SET
            team_code = excluded.team_code,
            age = excluded.age,
            gp = excluded.gp,
            wins = excluded.wins,
            losses = excluded.losses,
            minutes = excluded.minutes,
            points_total = excluded.points_total,
            fg_pct = excluded.fg_pct,
            three_pt_pct = excluded.three_pt_pct,
            ft_pct = excluded.ft_pct,
            rebounds = excluded.rebounds,
            assists = excluded.assists,
            steals = excluded.steals,
            blocks = excluded.blocks,
            turnovers = excluded.turnovers,
            off_rating = excluded.off_rating,
            def_rating = excluded.def_rating,
            net_rating = excluded.net_rating,
            usage_pct = excluded.usage_pct,
            pie = excluded.pie;
        """,
        [
            (
                row.player_name,
                row.team_code,
                row.age,
                row.gp,
                row.wins,
                row.losses,
                row.minutes,
                row.points_total,
                row.fg_pct,
                row.three_pt_pct,
                row.ft_pct,
                row.rebounds,
                row.assists,
                row.steals,
                row.blocks,
                row.turnovers,
                row.off_rating,
                row.def_rating,
                row.net_rating,
                row.usage_pct,
                row.pie,
            )
            for row in players
        ],
    )


def _insert_stats(conn: sqlite3.Connection, players: list[PlayerRow]) -> int:
    player_map = {
        row[1]: row[0]
        for row in conn.execute("SELECT player_id, player_name FROM players").fetchall()
    }
    match_map = {
        row[1]: row[0]
        for row in conn.execute("SELECT match_id, team_code FROM matches").fetchall()
    }

    rows: list[StatRow] = []
    for player in players:
        player_id = player_map.get(player.player_name)
        if not player_id:
            continue
        match_id = match_map.get(player.team_code)
        for attr_name, stat_key, unit in STAT_COLUMNS:
            value = getattr(player, attr_name)
            if value is None:
                continue
            rows.append(
                StatRow(
                    player_id=player_id,
                    match_id=match_id,
                    stat_key=stat_key,
                    stat_value=float(value),
                    unit=unit,
                    source_sheet="Données NBA",
                )
            )

    conn.executemany(
        """
        INSERT INTO stats (player_id, match_id, stat_key, stat_value, unit, source_sheet)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id, match_id, stat_key) DO UPDATE SET
            stat_value = excluded.stat_value,
            unit = excluded.unit,
            source_sheet = excluded.source_sheet;
        """,
        [
            (
                row.player_id,
                row.match_id,
                row.stat_key,
                row.stat_value,
                row.unit,
                row.source_sheet,
            )
            for row in rows
        ],
    )
    return len(rows)


def _insert_reports(conn: sqlite3.Connection, reports: list[ReportRow]) -> None:
    conn.executemany(
        """
        INSERT INTO reports (report_type, title, content, source_sheet, row_order)
        VALUES (?, ?, ?, ?, ?);
        """,
        [
            (
                report.report_type,
                report.title,
                report.content,
                report.source_sheet,
                report.row_order,
            )
            for report in reports
        ],
    )


def main() -> None:
    excel_path = _resolve_excel_path()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    players = _load_players(excel_path)
    matches = _load_matches(excel_path, players)
    reports = _load_reports()

    with sqlite3.connect(DB_PATH) as conn:
        _init_db(conn)
        _clear_db(conn)
        _insert_matches(conn, matches)
        _insert_players(conn, players)
        stats_count = _insert_stats(conn, players)
        _insert_reports(conn, reports)
        conn.commit()

    print(f"Excel charge: {excel_path}")
    print(f"Base SQLite: {DB_PATH}")
    print(f"players: {len(players)}")
    print(f"matches: {len(matches)}")
    print(f"stats: {stats_count}")
    print(f"reports: {len(reports)}")


if __name__ == "__main__":
    main()

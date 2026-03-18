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

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from utils.config import get_settings

LOGGER = logging.getLogger(__name__)

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
DB_PATH = Path(get_settings().database_file)


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

    @field_validator("player_name", mode="before")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        return str(value).strip()

    @field_validator("team_code", mode="before")
    @classmethod
    def _clean_code(cls, value: str) -> str:
        return str(value).strip().upper()


class MatchRow(BaseModel):
    team_code: str = Field(min_length=2, max_length=4)
    team_name: str = Field(min_length=1)
    players_count: int | None = Field(default=None, ge=0)
    team_points_total: float | None = Field(default=None, ge=0)
    team_games_played: int | None = Field(default=None, ge=0)
    team_wins: int | None = Field(default=None, ge=0)
    team_losses: int | None = Field(default=None, ge=0)

    @field_validator("team_code", mode="before")
    @classmethod
    def _normalize_team_code(cls, value: str) -> str:
        return str(value).strip().upper()

    @field_validator("team_name", mode="before")
    @classmethod
    def _normalize_team_name(cls, value: str) -> str:
        return str(value).strip()


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


PLAYER_FIELD_MAP = (
    ("player_name", "Player", None),
    ("team_code", "Team", None),
    *((field, column, _to_int) for field, column in (("age", "Age"), ("gp", "GP"), ("wins", "W"), ("losses", "L"))),
    *(
        (field, column, _to_float)
        for field, column in (
            ("minutes", "Min"),
            ("points_total", "PTS"),
            ("fg_pct", "FG%"),
            ("three_pt_pct", "3P%"),
            ("ft_pct", "FT%"),
            ("rebounds", "REB"),
            ("assists", "AST"),
            ("steals", "STL"),
            ("blocks", "BLK"),
            ("turnovers", "TOV"),
            ("off_rating", "OFFRTG"),
            ("def_rating", "DEFRTG"),
            ("net_rating", "NETRTG"),
            ("usage_pct", "USG%"),
            ("pie", "PIE"),
        )
    ),
)
PLAYER_COLUMNS = tuple(field for field, _, _ in PLAYER_FIELD_MAP)
MATCH_COLUMNS = tuple("team_code team_name players_count team_points_total team_games_played team_wins team_losses source_sheet".split())
STAT_INSERT_COLUMNS = tuple("player_id match_id stat_key stat_value unit source_sheet".split())
REPORT_COLUMNS = tuple("report_type title content source_sheet row_order".split())


def _build_payload(row: Any, field_map: tuple[tuple[str, str, Any], ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    getter = row.get if hasattr(row, "get") else None
    for field_name, source_name, converter in field_map:
        raw_value = getter(source_name) if getter else row[source_name]
        payload[field_name] = converter(raw_value) if callable(converter) else raw_value
    return payload


def _validate_model(model_cls: type[BaseModel], payload: dict[str, Any]) -> BaseModel | None:
    try:
        return model_cls.model_validate(payload)
    except ValidationError:
        return None


def _record_tuples(
    rows: list[BaseModel],
    columns: tuple[str, ...],
    extra_values: dict[str, Any] | None = None,
) -> list[tuple[Any, ...]]:
    extras = extra_values or {}
    return [
        tuple(extras[column] if column in extras else getattr(row, column) for column in columns)
        for row in rows
    ]


def _execute_upsert(
    conn: sqlite3.Connection,
    *,
    table: str,
    columns: tuple[str, ...],
    rows: list[tuple[Any, ...]],
    conflict_target: str | None = None,
    update_columns: tuple[str, ...] = (),
) -> None:
    if not rows:
        return

    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    if conflict_target and update_columns:
        updates = ", ".join(f"{column} = excluded.{column}" for column in update_columns)
        sql += f" ON CONFLICT({conflict_target}) DO UPDATE SET {updates}"
    conn.executemany(f"{sql};", rows)


def _lookup_map(conn: sqlite3.Connection, table: str, id_column: str, key_column: str) -> dict[str, int]:
    query = f"SELECT {id_column}, {key_column} FROM {table}"
    return {row[1]: row[0] for row in conn.execute(query).fetchall()}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _append_report(
    reports: list[ReportRow],
    row_order: int,
    *,
    report_type: str,
    title: str,
    content: str,
    source_sheet: str,
) -> int:
    reports.append(
        ReportRow(
            report_type=report_type,
            title=title,
            content=content,
            source_sheet=source_sheet,
            row_order=row_order,
        )
    )
    return row_order + 1


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
        player = _validate_model(PlayerRow, _build_payload(row, PLAYER_FIELD_MAP))
        if player is not None:
            players.append(player)
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
    try:
        from PyPDF2 import PdfReader
    except Exception:
        PdfReader = None

    reports: list[ReportRow] = []
    pdf_paths = _resolve_reddit_pdf_paths()

    row_order = 0
    for pdf_path in pdf_paths:
        file_had_text = False
        if PdfReader is None:
            reader = None
        else:
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
                row_order = _append_report(
                    reports,
                    row_order,
                    report_type="reddit_pdf",
                    title=f"{pdf_path.stem} - page {page_idx}",
                    content=_normalize_text(content),
                    source_sheet=pdf_path.name,
                )

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
                row_order = _append_report(
                    reports,
                    row_order,
                    report_type="reddit_pdf_unreadable",
                    title=f"{pdf_path.stem} - page {page_idx}",
                    content="Aucun texte extractible (PDF image/scanné, OCR indisponible).",
                    source_sheet=pdf_path.name,
                )
            continue

        row_order = _append_report(
            reports,
            row_order,
            report_type="reddit_pdf_ocr",
            title=f"{pdf_path.stem} - OCR complet",
            content=_normalize_text(ocr_text),
            source_sheet=pdf_path.name,
        )

    return reports


def _insert_matches(conn: sqlite3.Connection, matches: list[MatchRow]) -> None:
    _execute_upsert(
        conn,
        table="matches",
        columns=MATCH_COLUMNS,
        rows=_record_tuples(matches, MATCH_COLUMNS, {"source_sheet": "Analyse"}),
        conflict_target="team_code",
        update_columns=MATCH_COLUMNS[1:],
    )


def _insert_players(conn: sqlite3.Connection, players: list[PlayerRow]) -> None:
    _execute_upsert(
        conn,
        table="players",
        columns=PLAYER_COLUMNS,
        rows=_record_tuples(players, PLAYER_COLUMNS),
        conflict_target="player_name",
        update_columns=PLAYER_COLUMNS[1:],
    )


def _insert_stats(conn: sqlite3.Connection, players: list[PlayerRow]) -> int:
    player_map = _lookup_map(conn, "players", "player_id", "player_name")
    match_map = _lookup_map(conn, "matches", "match_id", "team_code")

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

    _execute_upsert(
        conn,
        table="stats",
        columns=STAT_INSERT_COLUMNS,
        rows=_record_tuples(rows, STAT_INSERT_COLUMNS),
        conflict_target="player_id, match_id, stat_key",
        update_columns=("stat_value", "unit", "source_sheet"),
    )
    return len(rows)


def _insert_reports(conn: sqlite3.Connection, reports: list[ReportRow]) -> None:
    _execute_upsert(
        conn,
        table="reports",
        columns=REPORT_COLUMNS,
        rows=_record_tuples(reports, REPORT_COLUMNS),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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

    LOGGER.info("Excel charge: %s", excel_path)
    LOGGER.info("Base SQLite: %s", DB_PATH)
    LOGGER.info("players: %s", len(players))
    LOGGER.info("matches: %s", len(matches))
    LOGGER.info("stats: %s", stats_count)
    LOGGER.info("reports: %s", len(reports))


if __name__ == "__main__":
    main()

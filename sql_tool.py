"""Tool SQL minimal pour questions chiffrées NBA."""

from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from utils.config import get_settings

SETTINGS = get_settings()
DB_PATH = Path(SETTINGS.database_file)
MAX_ROWS = 30

SQL_FEW_SHOTS = """
Q: Donne le top 5 des joueurs avec le plus de points totaux.
SQL:
SELECT player_name, team_code, points_total
FROM players
WHERE points_total IS NOT NULL
ORDER BY points_total DESC
LIMIT 5;

Q: Entre OKC et MIA, quelle equipe a le plus de points totaux ?
SQL:
SELECT team_code, team_name, team_points_total
FROM matches
WHERE team_code IN ('OKC','MIA')
ORDER BY team_points_total DESC;

Q: Compare les rebonds domicile vs extérieur.
SQL:
SELECT 'Limite: la base ne contient pas de split domicile/extérieur.' AS message;
""".strip()


def _tool_result(
    status: str,
    message: str,
    *,
    sql: str | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {"status": status, "sql": sql, "rows": rows or [], "message": message}


def is_likely_quant_question(question: str) -> bool:
    lowered = question.lower()
    keywords = [
        "combien",
        "moyenne",
        "total",
        "pourcentage",
        "points",
        "difference",
        "diff",
        "compare",
        "top",
        "minimum",
        "maximum",
        "classement",
    ]
    return any(token in lowered for token in keywords) or bool(re.search(r"\d", question))


def _extract_sql(text: str) -> str:
    block_match = re.search(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = block_match.group(1).strip() if block_match else text.strip()
    candidate = candidate.strip("`").strip()
    if not candidate.endswith(";"):
        candidate += ";"
    return candidate


def _is_safe_sql(sql: str) -> bool:
    normalized = " ".join(sql.lower().split())
    if not (normalized.startswith("select") or normalized.startswith("with")):
        return False
    forbidden = [" insert ", " update ", " delete ", " drop ", " alter ", " pragma ", " attach ", " vacuum "]
    if any(keyword in f" {normalized} " for keyword in forbidden):
        return False
    if normalized.count(";") > 1:
        return False
    return True


def _schema_as_text(conn: sqlite3.Connection) -> str:
    table_names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('players','matches','stats','reports') ORDER BY name"
        ).fetchall()
    ]
    sections: list[str] = []
    for table_name in table_names:
        columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        col_text = ", ".join(f"{col[1]} {col[2]}" for col in columns)
        sections.append(f"{table_name}({col_text})")
    return "\n".join(sections)


class NBASQLTool:
    def __init__(self) -> None:
        if not SETTINGS.mistral_api_key:
            raise EnvironmentError("MISTRAL_API_KEY est requis pour le Tool SQL.")
        from mistralai import Mistral

        self.client = Mistral(api_key=SETTINGS.mistral_api_key)

    def _build_prompt(self, question: str, schema: str) -> str:
        return (
            "Tu es un expert SQL SQLite.\n"
            "Objectif: produire UNE requête SQL valide et en lecture seule.\n"
            "Contraintes:\n"
            "- Autorise uniquement SELECT / WITH ... SELECT\n"
            "- Utilise uniquement les tables et colonnes du schéma\n"
            "- Si la question demande une dimension absente, renvoie un SELECT avec message explicite\n"
            "- Réponds uniquement avec SQL\n\n"
            f"Schema:\n{schema}\n\n"
            f"Few-shot:\n{SQL_FEW_SHOTS}\n\n"
            f"Question:\n{question}\n\n"
            "SQL:"
        )

    def _generate_sql(self, question: str, schema: str) -> str:
        prompt = self._build_prompt(question=question, schema=schema)
        response = self.client.chat.complete(
            model=SETTINGS.model_name,
            messages=[
                {"role": "system", "content": "Tu écris uniquement des requêtes SQL SQLite."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        sql = _extract_sql(response.choices[0].message.content or "")
        if not _is_safe_sql(sql):
            raise ValueError(f"SQL non sécurisé ou invalide: {sql}")
        return sql

    def answer(self, question: str) -> dict[str, Any]:
        if not DB_PATH.exists():
            return _tool_result("error", f"Base absente: {DB_PATH}. Lance d'abord load_excel_to_db.py.")

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            schema = _schema_as_text(conn)
            sql = self._generate_sql(question=question, schema=schema)
            rows = [dict(row) for row in conn.execute(sql).fetchmany(MAX_ROWS)]

        return _tool_result("ok", "Résultats SQL disponibles.", sql=sql, rows=rows)

    def run(self, question: str) -> str:
        return json.dumps(self.answer(question), ensure_ascii=False)


@lru_cache(maxsize=1)
def _get_agent() -> NBASQLTool:
    return NBASQLTool()


@lru_cache(maxsize=1)
def _get_routing_llm() -> ChatMistralAI:
    from langchain_mistralai import ChatMistralAI

    if not SETTINGS.mistral_api_key:
        raise EnvironmentError("MISTRAL_API_KEY est requis pour le routage SQL.")
    try:
        return ChatMistralAI(
            model=SETTINGS.model_name,
            temperature=0.0,
            api_key=SETTINGS.mistral_api_key,
        )
    except TypeError:
        return ChatMistralAI(
            model=SETTINGS.model_name,
            temperature=0.0,
            mistral_api_key=SETTINGS.mistral_api_key,
        )


@lru_cache(maxsize=1)
def _get_sql_tool() -> StructuredTool:
    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        func=_get_agent().run,
        name="nba_sql_tool",
        description=(
            "Interroge la base SQL NBA (players, matches, stats, reports) "
            "pour les questions chiffrées (totaux, moyennes, comparaisons)."
        ),
    )


def answer_question_sql_via_langchain(question: str) -> dict[str, Any]:
    """Flux public unique: route et exécute le Tool SQL."""
    trimmed = question.strip()
    if not trimmed:
        return _tool_result("no_tool", "Question vide, aucun appel SQL.")

    if not SETTINGS.mistral_api_key:
        return _tool_result("no_tool", "MISTRAL_API_KEY manquant, SQL tool indisponible.")

    if not is_likely_quant_question(trimmed):
        return _tool_result("no_tool", "Aucun appel SQL jugé utile.")

    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_routing_llm().bind_tools([_get_sql_tool()])
    messages = [
        SystemMessage(
            content=(
                "Tu réponds aux questions NBA. "
                "Appelle `nba_sql_tool` uniquement pour les besoins chiffrés."
            )
        ),
        HumanMessage(content=trimmed),
    ]

    try:
        ai_message = llm.invoke(messages)
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if not tool_calls:
            return _tool_result("no_tool", "Le routeur n'a pas déclenché le SQL tool.")

        first_call = tool_calls[0]
        args = first_call.get("args", {})
        raw_result = _get_sql_tool().invoke(args if isinstance(args, dict) else {"question": trimmed})
        if isinstance(raw_result, str):
            result = json.loads(raw_result)
        elif isinstance(raw_result, dict):
            result = raw_result
        else:
            return _tool_result("error", f"Réponse tool inattendue: {type(raw_result).__name__}")

        return _tool_result(
            result.get("status", "ok"),
            result.get("message", "Résultats SQL disponibles."),
            sql=result.get("sql"),
            rows=result.get("rows"),
        )
    except Exception as exc:
        return _tool_result("error", f"Echec du flux SQL: {exc}")

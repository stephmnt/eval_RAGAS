"""Tool SQL minimal pour questions chiffrées NBA.

Fonctions principales:
- génération dynamique SQL (LLM Mistral + few-shot)
- exécution SQL en lecture seule
- exposition d'un Tool LangChain et d'une fonction simple pour l'app
"""

from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_mistralai import ChatMistralAI
from mistralai import Mistral

from utils.config import DATABASE_FILE, MISTRAL_API_KEY, MODEL_NAME

DB_PATH = Path(DATABASE_FILE)
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

Q: Quelle est la moyenne des assists des joueurs de BOS ?
SQL:
SELECT AVG(assists) AS avg_assists
FROM players
WHERE team_code = 'BOS';
""".strip()


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


def llm_should_use_sql(question: str) -> bool:
    """Décision minimale par LLM: faut-il router vers le Tool SQL ?"""
    if not MISTRAL_API_KEY:
        return is_likely_quant_question(question)
    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un routeur. Reponds uniquement par YES ou NO. "
                        "YES si la question demande une réponse chiffrée ou une aggrégation SQL."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )
        verdict = (response.choices[0].message.content or "").strip().upper()
        return verdict.startswith("YES")
    except Exception:
        return is_likely_quant_question(question)


def _extract_sql(text: str) -> str:
    block_match = re.search(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if block_match:
        candidate = block_match.group(1).strip()
    else:
        candidate = text.strip()
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
        if not MISTRAL_API_KEY:
            raise EnvironmentError("MISTRAL_API_KEY est requis pour le Tool SQL.")
        self.client = Mistral(api_key=MISTRAL_API_KEY)

    def _build_prompt(self, question: str, schema: str) -> str:
        return (
            "Tu es un expert SQL SQLite.\n"
            "Objectif: produire UNE requête SQL valide et en lecture seule.\n"
            "Contraintes:\n"
            "- Autorise uniquement SELECT / WITH ... SELECT\n"
            "- Utilise uniquement les tables et colonnes du schéma\n"
            "- Si la question demande une dimension absente (ex: domicile/extérieur), renvoie une requête SELECT avec message explicite\n"
            "- Réponds uniquement avec SQL\n\n"
            f"Schema:\n{schema}\n\n"
            f"Few-shot:\n{SQL_FEW_SHOTS}\n\n"
            f"Question:\n{question}\n\n"
            "SQL:"
        )

    def _generate_sql(self, question: str, schema: str) -> str:
        prompt = self._build_prompt(question=question, schema=schema)
        response = self.client.chat.complete(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Tu ecris uniquement des requetes SQL SQLite."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        sql = _extract_sql(content)
        if not _is_safe_sql(sql):
            raise ValueError(f"SQL non securisé ou invalide: {sql}")
        return sql

    def _execute_sql(self, conn: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchmany(MAX_ROWS)
        return [dict(row) for row in rows]

    def answer(self, question: str) -> dict[str, Any]:
        if not DB_PATH.exists():
            return {
                "status": "error",
                "sql": None,
                "rows": [],
                "message": f"Base absente: {DB_PATH}. Lance d'abord load_excel_to_db.py.",
            }

        with sqlite3.connect(DB_PATH) as conn:
            schema = _schema_as_text(conn)
            sql = self._generate_sql(question=question, schema=schema)
            rows = self._execute_sql(conn, sql)

        return {
            "status": "ok",
            "sql": sql,
            "rows": rows,
            "message": "Résultats SQL disponibles.",
        }

    def run(self, question: str) -> str:
        result = self.answer(question)
        return json.dumps(result, ensure_ascii=False)


_SQL_AGENT: NBASQLTool | None = None


def _get_agent() -> NBASQLTool:
    global _SQL_AGENT
    if _SQL_AGENT is None:
        _SQL_AGENT = NBASQLTool()
    return _SQL_AGENT


def answer_question_sql(question: str) -> dict[str, Any]:
    return _get_agent().answer(question)


@lru_cache(maxsize=1)
def _get_routing_llm() -> ChatMistralAI:
    if not MISTRAL_API_KEY:
        raise EnvironmentError("MISTRAL_API_KEY est requis pour le routage LangChain.")
    try:
        return ChatMistralAI(
            model=MODEL_NAME,
            temperature=0.0,
            api_key=MISTRAL_API_KEY,
        )
    except TypeError:
        return ChatMistralAI(
            model=MODEL_NAME,
            temperature=0.0,
            mistral_api_key=MISTRAL_API_KEY,
        )


@lru_cache(maxsize=1)
def _get_structured_sql_tool() -> StructuredTool:
    return build_sql_tool()


def answer_question_sql_via_langchain(question: str) -> dict[str, Any]:
    """Route et exécute le tool SQL via un flux LangChain tool-calling.

    Returns:
        dict[str, Any]: Payload unifié contenant status/sql/rows/message.
            status peut valoir `ok`, `no_tool` ou `error`.
    """
    trimmed = question.strip()
    if not trimmed:
        return {
            "status": "no_tool",
            "sql": None,
            "rows": [],
            "message": "Question vide, aucun appel SQL.",
        }

    # Fallback minimal si les dépendances LangChain/clé API ne sont pas disponibles.
    if not MISTRAL_API_KEY:
        if not is_likely_quant_question(trimmed):
            return {
                "status": "no_tool",
                "sql": None,
                "rows": [],
                "message": "Aucun appel SQL nécessaire.",
            }
        return answer_question_sql(trimmed)

    tool = _get_structured_sql_tool()
    llm = _get_routing_llm().bind_tools([tool])
    messages = [
        SystemMessage(
            content=(
                "Tu réponds aux questions NBA. "
                "Appelle l'outil `nba_sql_tool` uniquement si la question "
                "nécessite une réponse chiffrée, une comparaison numérique, "
                "un classement ou une agrégation."
            )
        ),
        HumanMessage(content=trimmed),
    ]

    try:
        ai_message = llm.invoke(messages)
        tool_calls = getattr(ai_message, "tool_calls", None) or []

        if not tool_calls:
            return {
                "status": "no_tool",
                "sql": None,
                "rows": [],
                "message": "Aucun appel SQL jugé utile par le routeur outillé.",
            }

        first_call = tool_calls[0]
        args = first_call.get("args", {})
        raw_result = tool.invoke(args if isinstance(args, dict) else {"question": trimmed})

        if isinstance(raw_result, str):
            result = json.loads(raw_result)
        elif isinstance(raw_result, dict):
            result = raw_result
        else:
            result = {
                "status": "error",
                "sql": None,
                "rows": [],
                "message": f"Réponse tool inattendue: {type(raw_result).__name__}",
            }

        result.setdefault("status", "ok")
        result.setdefault("sql", None)
        result.setdefault("rows", [])
        result.setdefault("message", "Résultats SQL disponibles.")
        return result
    except Exception as exc:
        return {
            "status": "error",
            "sql": None,
            "rows": [],
            "message": f"Echec du flux LangChain SQL: {exc}",
        }


def build_sql_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_get_agent().run,
        name="nba_sql_tool",
        description=(
            "Interroge la base SQL NBA (players, matches, stats, reports) "
            "pour repondre aux questions chiffrées (totaux, moyennes, comparaisons)."
        ),
    )

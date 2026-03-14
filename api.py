"""API REST minimale pour exposer le système RAG + SQL.

Endpoints:
- GET /health
- POST /ask
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field
from mistralai import Mistral

from sql_tool import answer_question_sql_via_langchain
from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="RAG NBA API",
    version="1.0.0",
    description="API REST pour interroger le système RAG enrichi par le Tool SQL.",
)
router_v1 = APIRouter(prefix="/api/v1", tags=["v1"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, description="Question utilisateur.")
    k: int = Field(default=SEARCH_K, ge=1, le=20, description="Nombre de chunks RAG.")


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieval_count: int
    contexts: list[dict[str, Any]]
    sql_status: str
    sql_query: str | None
    sql_rows: list[dict[str, Any]]
    latency_retrieval_s: float
    latency_generation_s: float
    latency_total_s: float


@lru_cache(maxsize=1)
def _get_mistral_client() -> Mistral:
    if not MISTRAL_API_KEY:
        raise EnvironmentError("MISTRAL_API_KEY manquant.")
    return Mistral(api_key=MISTRAL_API_KEY)


@lru_cache(maxsize=1)
def _get_vector_store() -> VectorStoreManager:
    manager = VectorStoreManager()
    if manager.index is None or not manager.document_chunks:
        raise RuntimeError("Index vectoriel indisponible. Exécute d'abord `python indexer.py`.")
    return manager


def _build_prompt(question: str, contexts: list[dict[str, Any]], sql_context: str) -> str:
    if contexts:
        context_str = "\n\n---\n\n".join(
            f"Source: {ctx.get('metadata', {}).get('source', 'Inconnue')} | "
            f"Score: {ctx.get('score', 0):.1f}%\n"
            f"Contenu: {ctx.get('text', '')}"
            for ctx in contexts
        )
    else:
        context_str = "Aucun contexte pertinent trouvé dans la base documentaire."

    return (
        "Tu es un assistant NBA orienté faits.\n"
        "Utilise en priorité le CONTEXTE RAG, puis les DONNEES SQL si elles sont disponibles.\n"
        "Si une information est absente, réponds explicitement que la donnée est indisponible.\n\n"
        f"CONTEXTE RAG:\n{context_str}\n\n"
        f"DONNEES SQL:\n{sql_context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "RÉPONSE:"
    )


def _health_payload() -> dict[str, Any]:
    """Construit le statut de santé de l'API."""
    issues: list[str] = []
    if not MISTRAL_API_KEY:
        issues.append("MISTRAL_API_KEY manquant")
    try:
        _get_vector_store()
    except Exception as exc:
        issues.append(str(exc))

    return {
        "status": "ok" if not issues else "degraded",
        "issues": issues,
        "model": MODEL_NAME,
    }


def _ask_impl(payload: AskRequest) -> AskResponse:
    """Implémentation interne de l'endpoint /ask."""
    total_start = time.perf_counter()

    try:
        retriever = _get_vector_store()
        client = _get_mistral_client()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    retrieval_start = time.perf_counter()
    try:
        search_results = retriever.search(payload.question, k=payload.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur retrieval: {exc}") from exc
    retrieval_latency_s = round(time.perf_counter() - retrieval_start, 6)

    sql_result = answer_question_sql_via_langchain(payload.question)
    if sql_result.get("status") == "ok":
        sql_context = (
            f"SQL: {sql_result.get('sql')}\n"
            f"Rows (max 10): {(sql_result.get('rows') or [])[:10]}"
        )
    else:
        sql_context = sql_result.get("message", "Aucune donnée SQL.")

    generation_start = time.perf_counter()
    prompt = _build_prompt(
        question=payload.question,
        contexts=search_results,
        sql_context=sql_context,
    )
    try:
        completion = client.chat.complete(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Tu réponds uniquement à partir des données fournies."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        answer = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Erreur génération: {exc}") from exc
    generation_latency_s = round(time.perf_counter() - generation_start, 6)
    total_latency_s = round(time.perf_counter() - total_start, 6)

    return AskResponse(
        question=payload.question,
        answer=answer,
        retrieval_count=len(search_results),
        contexts=search_results,
        sql_status=str(sql_result.get("status", "unknown")),
        sql_query=sql_result.get("sql"),
        sql_rows=(sql_result.get("rows") or [])[:10],
        latency_retrieval_s=retrieval_latency_s,
        latency_generation_s=generation_latency_s,
        latency_total_s=total_latency_s,
    )


@router_v1.get("/health")
def health_v1() -> dict[str, Any]:
    """Vérifie la disponibilité des composants principaux (v1)."""
    return _health_payload()


@router_v1.post("/ask", response_model=AskResponse)
def ask_v1(payload: AskRequest) -> AskResponse:
    """Interroge le pipeline RAG + SQL (v1)."""
    return _ask_impl(payload)


app.include_router(router_v1)


# Compatibilité avec les appels historiques non versionnés.
@app.get("/health", deprecated=True, tags=["legacy"])
def health_legacy() -> dict[str, Any]:
    return _health_payload()


@app.post("/ask", response_model=AskResponse, deprecated=True, tags=["legacy"])
def ask_legacy(payload: AskRequest) -> AskResponse:
    return _ask_impl(payload)

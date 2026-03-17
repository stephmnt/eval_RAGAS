"""API REST minimale et service métier RAG+SQL."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from sql_tool import answer_question_sql_via_langchain
from utils.config import (
    AskRequest,
    AskResponse,
    GeneratedAnswer,
    RetrievedContext,
    SQLToolResult,
    configure_logfire,
    get_settings,
    logfire_event,
    logfire_span,
)

LOGGER = logging.getLogger(__name__)
SETTINGS = get_settings()


class RAGService:
    """Exécute le flux métier RAG + SQL partagé par API et Streamlit."""

    def __init__(self) -> None:
        if not SETTINGS.mistral_api_key:
            raise EnvironmentError("MISTRAL_API_KEY manquant.")

        from mistralai import Mistral
        from utils.vector_store import VectorStoreManager

        self._client = Mistral(api_key=SETTINGS.mistral_api_key)
        self._model_name = SETTINGS.model_name
        self._default_k = SETTINGS.search_k
        self._retriever = VectorStoreManager()

        if self._retriever.index is None or not self._retriever.document_chunks:
            raise RuntimeError("Index vectoriel indisponible. Exécute d'abord `python indexer.py`.")

    def health(self) -> dict[str, object]:
        issues: list[str] = []
        if not SETTINGS.mistral_api_key:
            issues.append("MISTRAL_API_KEY manquant")
        if self._retriever.index is None or not self._retriever.document_chunks:
            issues.append("Index vectoriel indisponible")
        return {
            "status": "ok" if not issues else "degraded",
            "issues": issues,
            "model": self._model_name,
        }

    @staticmethod
    def _build_prompt(question: str, contexts: list[RetrievedContext], sql_context: str) -> str:
        if contexts:
            context_block = "\n\n---\n\n".join(
                f"Source: {ctx.metadata.get('source', 'Inconnue')} | Score: {ctx.score:.1f}%\nContenu: {ctx.text}"
                for ctx in contexts
            )
        else:
            context_block = "Aucun contexte pertinent trouvé dans la base documentaire."

        return (
            "Tu es un assistant NBA orienté faits.\n"
            "Utilise en priorité le CONTEXTE RAG, puis les DONNEES SQL si disponibles.\n"
            "Si une information est absente, réponds explicitement que la donnée est indisponible.\n\n"
            f"CONTEXTE RAG:\n{context_block}\n\n"
            f"DONNEES SQL:\n{sql_context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "RÉPONSE:"
        )

    def ask(self, question: str, k: int | None = None) -> AskResponse:
        effective_k = k or self._default_k
        total_start = time.perf_counter()

        with logfire_span("rag_service_ask", question=question, k=effective_k):
            retrieval_start = time.perf_counter()
            raw_contexts = self._retriever.search(question, k=effective_k)
            contexts = [RetrievedContext.model_validate(item) for item in raw_contexts]
            retrieval_latency_s = round(time.perf_counter() - retrieval_start, 6)

            sql_result = SQLToolResult.model_validate(answer_question_sql_via_langchain(question))
            if sql_result.status == "ok":
                sql_context = f"SQL: {sql_result.sql}\nRows (max 10): {sql_result.rows[:10]}"
            else:
                sql_context = sql_result.message or "Aucune donnée SQL."

            generation_start = time.perf_counter()
            prompt = self._build_prompt(question=question, contexts=contexts, sql_context=sql_context)
            completion = self._client.chat.complete(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": "Tu réponds uniquement à partir des données fournies."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )

            answer = GeneratedAnswer.model_validate(
                {"content": completion.choices[0].message.content or ""}
            ).content
            generation_latency_s = round(time.perf_counter() - generation_start, 6)
            total_latency_s = round(time.perf_counter() - total_start, 6)

            payload = AskResponse(
                question=question,
                answer=answer,
                retrieval_count=len(contexts),
                contexts=contexts,
                sql_status=sql_result.status,
                sql_query=sql_result.sql if sql_result.status == "ok" else None,
                sql_rows=sql_result.rows[:10] if sql_result.status == "ok" else [],
                latency_retrieval_s=retrieval_latency_s,
                latency_generation_s=generation_latency_s,
                latency_total_s=total_latency_s,
            )

            logfire_event(
                "info",
                "rag_service_ask_completed",
                retrieval_count=payload.retrieval_count,
                sql_status=payload.sql_status,
                latency_total_s=payload.latency_total_s,
            )
            return payload


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService()


app = FastAPI(
    title="RAG NBA API",
    version="1.0.0",
    description="API REST pour interroger le système RAG enrichi par SQL.",
)
router_v1 = APIRouter(prefix="/api/v1", tags=["v1"])


@router_v1.get("/health")
def health_v1() -> dict[str, Any]:
    try:
        return get_rag_service().health()
    except Exception as exc:
        return {"status": "degraded", "issues": [str(exc)]}


@router_v1.post("/ask", response_model=AskResponse)
def ask_v1(payload: AskRequest) -> AskResponse:
    try:
        return get_rag_service().ask(payload.question, payload.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.include_router(router_v1)

configure_logfire(
    enabled=SETTINGS.logfire_enabled,
    send_to_logfire=SETTINGS.logfire_send_to_logfire,
    service_name=SETTINGS.logfire_service_name,
    logger=LOGGER,
    instrument_pydantic=True,
    instrument_fastapi_app=app,
)

"""API REST minimale pour exposer le système RAG + SQL.

Endpoints:
- GET /health
- POST /ask
"""

from __future__ import annotations

import contextlib
import logging
import time
from functools import lru_cache
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException
from mistralai import Mistral
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from sql_tool import answer_question_sql_via_langchain
from utils.config import (
    LOGFIRE_ENABLED,
    LOGFIRE_SEND_TO_LOGFIRE,
    LOGFIRE_SERVICE_NAME,
    MISTRAL_API_KEY,
    MODEL_NAME,
    SEARCH_K,
)
from utils.vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)
_LOGFIRE: Any | None = None

app = FastAPI(
    title="RAG NBA API",
    version="1.0.0",
    description="API REST pour interroger le système RAG enrichi par le Tool SQL.",
)
router_v1 = APIRouter(prefix="/api/v1", tags=["v1"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, description="Question utilisateur.")
    k: int = Field(default=SEARCH_K, ge=1, le=20, description="Nombre de chunks RAG.")


class RetrievedContext(BaseModel):
    text: str = Field(min_length=1)
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("Le contexte est vide.")
        return cleaned

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, value: Any) -> float:
        if value is None:
            return 0.0
        return float(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        raise ValueError("metadata doit être un objet clé/valeur.")


class SQLToolResult(BaseModel):
    status: Literal["ok", "no_tool", "error", "unknown"]
    sql: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    message: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "SQLToolResult":
        if self.status == "ok":
            if not self.sql or not self.sql.strip():
                raise ValueError("status=ok exige une requête SQL.")
        else:
            if self.sql is not None and str(self.sql).strip():
                raise ValueError("status!=ok ne doit pas exposer de requête SQL.")
            if self.rows:
                raise ValueError("status!=ok ne doit pas exposer de lignes SQL.")
        return self


class GeneratedAnswer(BaseModel):
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _normalize_answer(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("Réponse générée vide.")
        return cleaned


class AskResponse(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    retrieval_count: int = Field(ge=0)
    contexts: list[RetrievedContext]
    sql_status: Literal["ok", "no_tool", "error", "unknown"]
    sql_query: str | None
    sql_rows: list[dict[str, Any]]
    latency_retrieval_s: float = Field(ge=0.0)
    latency_generation_s: float = Field(ge=0.0)
    latency_total_s: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _check_sql_consistency(self) -> "AskResponse":
        if self.sql_status == "ok":
            if not self.sql_query or not self.sql_query.strip():
                raise ValueError("sql_status=ok exige sql_query.")
        else:
            if self.sql_query is not None and str(self.sql_query).strip():
                raise ValueError("sql_status!=ok ne doit pas exposer sql_query.")
            if self.sql_rows:
                raise ValueError("sql_status!=ok ne doit pas exposer sql_rows.")
        return self


def _configure_logfire() -> None:
    global _LOGFIRE
    if not LOGFIRE_ENABLED:
        LOGGER.info("Logfire désactivé (LOGFIRE_ENABLED=false).")
        _LOGFIRE = None
        return
    try:
        import logfire
    except Exception:
        LOGGER.info("Logfire non installé : instrumentation API désactivée.")
        _LOGFIRE = None
        return
    try:
        try:
            logfire.configure(
                service_name=LOGFIRE_SERVICE_NAME,
                send_to_logfire=LOGFIRE_SEND_TO_LOGFIRE,
            )
        except TypeError:
            logfire.configure()
        try:
            logfire.instrument_fastapi(app)
        except Exception:
            LOGGER.warning("Logfire actif, mais l'instrumentation FastAPI a échoué.")
        try:
            logfire.instrument_pydantic()
        except Exception:
            LOGGER.warning("Logfire actif, mais l'instrumentation Pydantic a échoué.")
        _LOGFIRE = logfire
        LOGGER.info("Logfire activé pour l'API.")
    except Exception as exc:
        LOGGER.warning("Impossible d'initialiser Logfire sur API: %s", exc)
        _LOGFIRE = None


def _logfire_span(name: str, **attrs: Any) -> contextlib.AbstractContextManager[Any]:
    if _LOGFIRE is None:
        return contextlib.nullcontext()
    try:
        return _LOGFIRE.span(name, **attrs)
    except Exception:
        return contextlib.nullcontext()


def _logfire_event(level: str, message: str, **attrs: Any) -> None:
    if _LOGFIRE is None:
        return
    try:
        fn = getattr(_LOGFIRE, level, None)
        if callable(fn):
            fn(message, **attrs)
    except Exception:
        return


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


def _build_prompt(question: str, contexts: list[RetrievedContext], sql_context: str) -> str:
    if contexts:
        context_str = "\n\n---\n\n".join(
            f"Source: {ctx.metadata.get('source', 'Inconnue')} | "
            f"Score: {ctx.score:.1f}%\n"
            f"Contenu: {ctx.text}"
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

    with _logfire_span("api_ask", question=payload.question, k=payload.k):
        retrieval_start = time.perf_counter()
        try:
            raw_search_results = retriever.search(payload.question, k=payload.k)
            search_results = [RetrievedContext.model_validate(item) for item in raw_search_results]
        except ValidationError as exc:
            _logfire_event("error", "retrieval_context_validation_failed", errors=exc.errors())
            raise HTTPException(status_code=500, detail="Contexte retrieval invalide.") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Erreur retrieval: {exc}") from exc
        retrieval_latency_s = round(time.perf_counter() - retrieval_start, 6)

        try:
            sql_result = SQLToolResult.model_validate(answer_question_sql_via_langchain(payload.question))
        except ValidationError as exc:
            _logfire_event("error", "sql_result_validation_failed", errors=exc.errors())
            raise HTTPException(status_code=500, detail="Résultat SQL invalide.") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Erreur SQL tool: {exc}") from exc

        if sql_result.status == "ok":
            sql_context = (
                f"SQL: {sql_result.sql}\n"
                f"Rows (max 10): {sql_result.rows[:10]}"
            )
        else:
            sql_context = sql_result.message or "Aucune donnée SQL."

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
            answer = GeneratedAnswer.model_validate(
                {"content": completion.choices[0].message.content or ""}
            ).content
        except ValidationError as exc:
            _logfire_event("error", "generated_answer_validation_failed", errors=exc.errors())
            raise HTTPException(status_code=502, detail="Réponse générée invalide.") from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Erreur génération: {exc}") from exc
        generation_latency_s = round(time.perf_counter() - generation_start, 6)
        total_latency_s = round(time.perf_counter() - total_start, 6)

        response = AskResponse(
            question=payload.question,
            answer=answer,
            retrieval_count=len(search_results),
            contexts=search_results,
            sql_status=sql_result.status,
            sql_query=sql_result.sql if sql_result.status == "ok" else None,
            sql_rows=sql_result.rows[:10] if sql_result.status == "ok" else [],
            latency_retrieval_s=retrieval_latency_s,
            latency_generation_s=generation_latency_s,
            latency_total_s=total_latency_s,
        )
        _logfire_event(
            "info",
            "api_ask_completed",
            retrieval_count=response.retrieval_count,
            sql_status=response.sql_status,
            latency_total_s=response.latency_total_s,
        )
        return response


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


_configure_logfire()

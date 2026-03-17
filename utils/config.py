"""Configuration, observabilité et schémas partagés du projet."""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

_LOGFIRE: Any | None = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    mistral_api_key: str | None
    embedding_model: str
    model_name: str
    input_dir: str
    vector_db_dir: str
    faiss_index_file: str
    document_chunks_file: str
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    search_k: int
    database_dir: str
    database_file: str
    app_title: str
    app_name: str
    logfire_enabled: bool
    logfire_send_to_logfire: bool
    logfire_service_name: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    vector_db_dir = "vector_db"
    database_dir = "database"

    return Settings(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        embedding_model="mistral-embed",
        model_name="mistral-small-latest",
        input_dir="inputs",
        vector_db_dir=vector_db_dir,
        faiss_index_file=os.path.join(vector_db_dir, "faiss_index.idx"),
        document_chunks_file=os.path.join(vector_db_dir, "document_chunks.pkl"),
        chunk_size=1500,
        chunk_overlap=150,
        embedding_batch_size=32,
        search_k=5,
        database_dir=database_dir,
        database_file=os.path.join(database_dir, "nba_data.db"),
        app_title="NBA Analyst AI",
        app_name="NBA",
        logfire_enabled=_env_bool("LOGFIRE_ENABLED", default=False),
        logfire_send_to_logfire=_env_bool("LOGFIRE_SEND_TO_LOGFIRE", default=False),
        logfire_service_name=os.getenv("LOGFIRE_SERVICE_NAME", "rag-nba"),
    )


def configure_logfire(
    *,
    enabled: bool,
    send_to_logfire: bool,
    service_name: str,
    logger: logging.Logger,
    instrument_pydantic: bool = False,
    instrument_fastapi_app: Any | None = None,
) -> None:
    """Configure Logfire si activé, sinon no-op."""
    global _LOGFIRE
    if not enabled:
        logger.info("Logfire désactivé.")
        _LOGFIRE = None
        return

    try:
        import logfire
    except Exception:
        logger.info("Logfire non installé, instrumentation ignorée.")
        _LOGFIRE = None
        return

    try:
        try:
            logfire.configure(service_name=service_name, send_to_logfire=send_to_logfire)
        except TypeError:
            logfire.configure()

        if instrument_fastapi_app is not None:
            try:
                logfire.instrument_fastapi(instrument_fastapi_app)
            except Exception as exc:
                logger.warning("Instrumentation FastAPI Logfire échouée: %s", exc)

        if instrument_pydantic:
            try:
                logfire.instrument_pydantic()
            except Exception as exc:
                logger.warning("Instrumentation Pydantic Logfire échouée: %s", exc)

        _LOGFIRE = logfire
        logger.info("Logfire activé.")
    except Exception as exc:
        logger.warning("Initialisation Logfire impossible: %s", exc)
        _LOGFIRE = None


def logfire_span(name: str, **attrs: Any) -> contextlib.AbstractContextManager[Any]:
    """Retourne un span Logfire si disponible, sinon un context manager no-op."""
    if _LOGFIRE is None:
        return contextlib.nullcontext()
    try:
        return _LOGFIRE.span(name, **attrs)
    except Exception:
        return contextlib.nullcontext()


def logfire_event(level: str, message: str, **attrs: Any) -> None:
    """Émet un événement Logfire si disponible."""
    if _LOGFIRE is None:
        return
    try:
        fn = getattr(_LOGFIRE, level, None)
        if callable(fn):
            fn(message, **attrs)
    except Exception:
        return


class AskRequest(BaseModel):
    question: str = Field(min_length=3, description="Question utilisateur.")
    k: int | None = Field(default=None, ge=1, le=20, description="Nombre de chunks RAG.")


class RetrievedContext(BaseModel):
    text: str = Field(min_length=1)
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("Le contexte est vide.")
        return cleaned

    @field_validator("score", mode="before")
    @classmethod
    def normalize_score(cls, value: Any) -> float:
        if value is None:
            return 0.0
        return float(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def normalize_metadata(cls, value: Any) -> dict[str, Any]:
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
    def check_consistency(self) -> "SQLToolResult":
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
    def normalize_content(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("Réponse générée vide.")
        return cleaned


class GeneratedAnswerUsage(BaseModel):
    answer: str = Field(min_length=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @field_validator("answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("Réponse générée vide.")
        return cleaned

    @model_validator(mode="after")
    def check_tokens_consistency(self) -> "GeneratedAnswerUsage":
        if (
            self.total_tokens is not None
            and self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens < self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens incohérent avec input_tokens + output_tokens.")
        return self


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
    def check_sql_consistency(self) -> "AskResponse":
        if self.sql_status == "ok":
            if not self.sql_query or not self.sql_query.strip():
                raise ValueError("sql_status=ok exige sql_query.")
        else:
            if self.sql_query is not None and str(self.sql_query).strip():
                raise ValueError("sql_status!=ok ne doit pas exposer sql_query.")
            if self.sql_rows:
                raise ValueError("sql_status!=ok ne doit pas exposer sql_rows.")
        return self


class EvaluationSample(BaseModel):
    sample_index: int = Field(ge=0)
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str = Field(min_length=1)
    retrieval_keywords: list[str] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    sql_used: bool
    sql_query: str | None = None
    retrieval_latency_s: float = Field(ge=0.0)
    generation_latency_s: float = Field(ge=0.0)
    total_latency_s: float = Field(ge=0.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @field_validator("id", "category", "question", "answer", "ground_truth")
    @classmethod
    def normalize_text_fields(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("Champ textuel obligatoire vide.")
        return cleaned

    @field_validator("contexts", mode="before")
    @classmethod
    def normalize_contexts(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("contexts doit être une liste.")
        return [str(item).strip() for item in value if str(item).strip()]


class RagasRunOutput(BaseModel):
    sample_count: int = Field(ge=0)
    summary: dict[str, Any]
    details_rows: int = Field(ge=0)

    @model_validator(mode="after")
    def check_counts(self) -> "RagasRunOutput":
        if self.details_rows and self.sample_count and self.details_rows != self.sample_count:
            raise ValueError("Le nombre de lignes détaillées doit correspondre au nombre d'échantillons.")
        return self

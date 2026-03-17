"""Evaluation RAG minimale en mode core (RAGAS + métriques retrieval).

Ce script:
1. génère des échantillons (question, contextes, réponse, métadonnées),
2. exécute RAGAS en profil core,
3. calcule des métriques retrieval/latence/tokens,
4. sauvegarde les artefacts JSON/CSV.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from mistralai import Mistral

from sql_tool import answer_question_sql_via_langchain
from utils.config import (
    EvaluationSample,
    GeneratedAnswerUsage,
    RagasRunOutput,
    RetrievedContext,
    SQLToolResult,
    configure_logfire,
    get_settings,
    logfire_event,
    logfire_span,
)
from utils.vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)
SETTINGS = get_settings()

OUTPUT_DIR = Path("outputs/evaluations")
EVAL_K = SETTINGS.search_k
EVAL_MIN_SCORE: float | None = None
INCLUDE_CONTEXT_RECALL = True

REQUEST_DELAY_SECONDS = 1
RETRIEVAL_OVERLAP_THRESHOLD = 0.2
RAGAS_TIMEOUT_SECONDS = 240
RAGAS_MAX_RETRIES = 12
RAGAS_MAX_WAIT_SECONDS = 120
RAGAS_MAX_WORKERS = 1
RAGAS_BATCH_SIZE = 1
STRICT_RAGAS_ERRORS = False

FRENCH_STOPWORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "et",
    "eux",
    "il",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "lui",
    "ma",
    "mais",
    "me",
    "meme",
    "mes",
    "moi",
    "mon",
    "ne",
    "nos",
    "notre",
    "nous",
    "on",
    "ou",
    "par",
    "pas",
    "pour",
    "qu",
    "que",
    "qui",
    "sa",
    "se",
    "ses",
    "son",
    "sur",
    "ta",
    "te",
    "tes",
    "toi",
    "ton",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
}

DEFAULT_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "q1",
        "category": "simple",
        "question": "Quel est le nom complet de l'équipe codée OKC ?",
        "ground_truth": "OKC correspond à Oklahoma City Thunder.",
        "retrieval_keywords": ["OKC", "Oklahoma City Thunder"],
    },
    {
        "id": "q2",
        "category": "simple",
        "question": "Selon le tableau des équipes, combien de points totaux a OKC ?",
        "ground_truth": "OKC totalise 9880 points.",
        "retrieval_keywords": ["OKC", "9880"],
    },
    {
        "id": "q3",
        "category": "complex",
        "question": "Parmi MIA, OKC, LAC et BKN, quelle équipe a le plus de points totaux ?",
        "ground_truth": "Parmi ces quatre équipes, OKC est premier avec 9880 points.",
        "retrieval_keywords": ["MIA", "OKC", "LAC", "BKN", "9880"],
    },
    {
        "id": "q4",
        "category": "complex",
        "question": "Quelle est la différence de points totaux entre OKC (9880) et MIA (9828) ?",
        "ground_truth": "La différence est de 52 points.",
        "retrieval_keywords": ["OKC", "9880", "MIA", "9828", "52"],
    },
    {
        "id": "q5",
        "category": "simple",
        "question": "Combien de joueurs compte l'équipe Brooklyn Nets (BKN) ?",
        "ground_truth": "BKN compte 20 joueurs.",
        "retrieval_keywords": ["Brooklyn Nets", "BKN", "20"],
    },
    {
        "id": "q6",
        "category": "simple",
        "question": "Dans le top 15 des joueurs par points, combien de points totaux a Shai Gilgeous-Alexander ?",
        "ground_truth": "Shai Gilgeous-Alexander affiche 2485 points totaux.",
        "retrieval_keywords": ["Shai Gilgeous-Alexander", "2485"],
    },
    {
        "id": "q7",
        "category": "simple",
        "question": "Quel est le pourcentage à 3 points (3P%) de Shai Gilgeous-Alexander dans ce tableau ?",
        "ground_truth": "Le 3P% de Shai Gilgeous-Alexander est de 37.5.",
        "retrieval_keywords": ["Shai Gilgeous-Alexander", "3P%", "37.5"],
    },
    {
        "id": "q8",
        "category": "complex",
        "question": "Entre Anthony Edwards (2180) et Nikola Jokic (2072), qui a le plus de points totaux ?",
        "ground_truth": "Anthony Edwards a le total le plus élevé avec 2180 points (contre 2072).",
        "retrieval_keywords": ["Anthony Edwards", "2180", "Nikola Jokic", "2072"],
    },
    {
        "id": "q9",
        "category": "complex",
        "question": "Entre Detroit Pistons (10292) et Cleveland Cavaliers (10180), quelle équipe a le plus de points totaux ?",
        "ground_truth": "Detroit Pistons est devant avec 10292 points (contre 10180).",
        "retrieval_keywords": ["Detroit Pistons", "10292", "Cleveland Cavaliers", "10180"],
    },
    {
        "id": "q10",
        "category": "noisy",
        "question": "code MIA -> équipe + points ??? réponse rapide",
        "ground_truth": "MIA correspond à Miami Heat et le total affiché est de 9828 points.",
        "retrieval_keywords": ["MIA", "Miami Heat", "9828"],
    },
    {
        "id": "q11",
        "category": "hybrid",
        "question": "Donne le nom complet de l'équipe devant entre OKC et MIA, puis l'écart de points exact.",
        "ground_truth": "Oklahoma City Thunder est devant Miami Heat avec 52 points d'écart.",
        "retrieval_keywords": ["OKC", "MIA", "Oklahoma City Thunder", "Miami Heat", "9880", "9828", "52"],
    },
    {
        "id": "q12",
        "category": "hybrid",
        "question": "Entre Tyler Herro (1840) et Trae Young (1839), qui est devant et de combien ?",
        "ground_truth": "Tyler Herro est devant Trae Young d'un point (1840 contre 1839).",
        "retrieval_keywords": ["Tyler Herro", "1840", "Trae Young", "1839", "1"],
    },
    {
        "id": "q13",
        "category": "hybrid_noisy",
        "question": "code BOS -> nom équipe + points totaux + nb joueurs ?",
        "ground_truth": "BOS correspond à Boston Celtics avec 9551 points totaux et 17 joueurs.",
        "retrieval_keywords": ["BOS", "Boston Celtics", "9551", "17"],
    },
    {
        "id": "q14",
        "category": "robustness_limit",
        "question": "Compare les rebonds domicile vs extérieur pour MIA et donne la différence exacte.",
        "ground_truth": "Les données domicile/extérieur ne sont pas disponibles dans le corpus fourni.",
        "retrieval_keywords": ["MIA", "Miami Heat", "rebonds", "domicile", "extérieur"],
    },
]


def _sleep_between_requests() -> None:
    if REQUEST_DELAY_SECONDS > 0:
        time.sleep(REQUEST_DELAY_SECONDS)


async def _async_sleep_between_requests() -> None:
    if REQUEST_DELAY_SECONDS > 0:
        await asyncio.sleep(REQUEST_DELAY_SECONDS)


def _truncate_context(text: str, max_chars: int = 1200) -> str:
    clean = " ".join(str(text).split())
    clean = re.sub(r"\bNaN\b", "", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean[:max_chars]


def _normalize_text_for_match(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _derive_keywords_from_reference(reference: str, max_keywords: int = 12) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", reference.lower())
    keywords: list[str] = []
    for token in tokens:
        if token in FRENCH_STOPWORDS:
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        keywords.append(token)
    return _dedupe_keep_order(keywords)[:max_keywords]


def _resolve_retrieval_keywords(sample: dict[str, Any]) -> list[str]:
    raw = sample.get("retrieval_keywords")
    if isinstance(raw, list):
        explicit = _dedupe_keep_order([str(item).strip() for item in raw if str(item).strip()])
        if explicit:
            return explicit
    return _derive_keywords_from_reference(str(sample.get("ground_truth", "")))


def _build_retrieval_queries(question: str, retrieval_keywords: list[str], max_queries: int = 3) -> list[str]:
    queries = [question.strip()]
    keywords = _dedupe_keep_order(retrieval_keywords)
    if keywords:
        queries.append(" ".join(keywords[:6]))
        queries.append(f"{question.strip()} {' '.join(keywords[:4])}".strip())
    return _dedupe_keep_order([q for q in queries if q])[:max_queries]


def _merge_retrieval_results(batches: list[list[dict[str, Any]]], k: int) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for batch in batches:
        for item in batch:
            text = str(item.get("text", "")).strip()
            metadata = item.get("metadata", {})
            key = f"{text}||{json.dumps(metadata, sort_keys=True, ensure_ascii=False, default=str)}"
            score = float(item.get("score", 0.0))
            current = by_key.get(key)
            if current is None or score > float(current.get("score", 0.0)):
                by_key[key] = item
    merged = sorted(by_key.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:k]


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value))
    except Exception:
        return None


def _extract_usage_tokens(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    if hasattr(usage, "model_dump"):
        usage_dict = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage_dict = usage.dict()
    elif isinstance(usage, dict):
        usage_dict = usage
    else:
        usage_dict = {}

    input_tokens = _safe_int(usage_dict.get("prompt_tokens", usage_dict.get("input_tokens")))
    output_tokens = _safe_int(usage_dict.get("completion_tokens", usage_dict.get("output_tokens")))
    total_tokens = _safe_int(usage_dict.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _validate_retrieved_contexts(
    raw_contexts: list[dict[str, Any]],
    *,
    sample_id: str,
    retrieval_query: str,
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, context in enumerate(raw_contexts):
        try:
            parsed = RetrievedContext.model_validate(context)
            validated.append(parsed.model_dump())
        except Exception as exc:
            LOGGER.warning(
                "Contexte invalide ignoré (sample=%s, query=%s, idx=%s): %s",
                sample_id,
                retrieval_query,
                index,
                exc,
            )
            logfire_event(
                "warning",
                "retrieved_context_invalid",
                sample_id=sample_id,
                retrieval_query=retrieval_query,
                index=index,
                error=str(exc),
            )
    return validated


def _validate_generated_answer(
    *,
    answer: str,
    usage: dict[str, int | None],
    sample_id: str,
) -> tuple[str, dict[str, int | None]]:
    validated = GeneratedAnswerUsage.model_validate(
        {
            "answer": answer,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    )
    logfire_event("info", "generated_answer_validated", sample_id=sample_id)
    return validated.answer, {
        "input_tokens": validated.input_tokens,
        "output_tokens": validated.output_tokens,
        "total_tokens": validated.total_tokens,
    }


def _validate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return EvaluationSample.model_validate(sample).model_dump()


def _load_questions() -> list[dict[str, Any]]:
    raw = DEFAULT_QUESTIONS
    if isinstance(raw, dict) and "questions" in raw:
        raw = raw["questions"]
    if not isinstance(raw, list):
        raise ValueError("DEFAULT_QUESTIONS doit être une liste (ou {'questions': [...]})")

    normalized: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for index, row in enumerate(raw, start=1):
        if not isinstance(row, dict):
            row = {"question": str(row)}

        raw_id = str(row.get("id", f"q{index}")).strip() or f"q{index}"
        final_id = raw_id
        suffix = 2
        while final_id in used_ids:
            final_id = f"{raw_id}_{suffix}"
            suffix += 1
        used_ids.add(final_id)

        normalized.append(
            {
                "id": final_id,
                "category": str(row.get("category", "non_categorise")).strip() or "non_categorise",
                "question": str(row.get("question", "")).strip(),
                "ground_truth": str(row.get("ground_truth", "")).strip(),
                "retrieval_keywords": _dedupe_keep_order(
                    [str(item).strip() for item in row.get("retrieval_keywords", []) if str(item).strip()]
                ),
            }
        )

    return normalized


def _build_prompt(question: str, contexts: list[str], sql_context: str) -> str:
    context_block = "\n\n".join([f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)])
    return (
        "Réponds uniquement avec les informations présentes dans le CONTEXTE et le SQL_CONTEXT. "
        "Si l'information est absente, réponds exactement: Information non disponible dans le contexte.\n\n"
        f"CONTEXTE:\n{context_block}\n\n"
        f"SQL_CONTEXT:\n{sql_context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "RÉPONSE FINALE :"
    )


def _build_sql_context(question: str) -> tuple[str, bool, str | None]:
    try:
        result = SQLToolResult.model_validate(answer_question_sql_via_langchain(question))
    except Exception as exc:
        logfire_event("error", "sql_tool_invalid", question=question, error=str(exc))
        return f"Echec du tool SQL: {exc}", False, None

    if result.status == "no_tool":
        return result.message or "Aucun appel SQL jugé nécessaire.", False, None
    if result.status != "ok":
        return f"Tool SQL indisponible: {result.message}", False, None

    return (
        f"SQL: {result.sql}\nRows (max 10): {result.rows[:10]}",
        True,
        result.sql,
    )


def _generate_answer(
    *,
    client: Mistral,
    model: str,
    question: str,
    contexts: list[str],
    sql_context: str,
    sample_id: str,
) -> tuple[str, dict[str, int | None]]:
    if not contexts:
        return (
            "Contexte insuffisant dans le vector store pour répondre à cette question.",
            {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        )

    try:
        with logfire_span("generate_answer", sample_id=sample_id, model=model):
            response = client.chat.complete(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un assistant NBA. Utilise uniquement le contexte fourni.",
                    },
                    {"role": "user", "content": _build_prompt(question, contexts, sql_context)},
                ],
                temperature=0.0,
            )
            answer = (response.choices[0].message.content or "").strip()
            usage = _extract_usage_tokens(response)
            return _validate_generated_answer(answer=answer, usage=usage, sample_id=sample_id)
    except Exception as exc:
        LOGGER.exception("Echec de génération (sample=%s)", sample_id)
        logfire_event("error", "generate_answer_failed", sample_id=sample_id, error=str(exc))
        return (
            f"Erreur de génération : {exc}",
            {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        )


def _build_samples(
    *,
    questions: list[dict[str, Any]],
    retriever: VectorStoreManager,
    client: Mistral,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for sample_index, question_row in enumerate(questions):
        sample_id = question_row["id"]
        with logfire_span("build_sample", sample_id=sample_id):
            total_start = time.perf_counter()

            retrieval_start = time.perf_counter()
            retrieval_queries = _build_retrieval_queries(
                question=question_row["question"],
                retrieval_keywords=_resolve_retrieval_keywords(question_row),
            )
            batches: list[list[dict[str, Any]]] = []
            for query in retrieval_queries:
                _sleep_between_requests()
                raw_batch = retriever.search(query, k=EVAL_K, min_score=EVAL_MIN_SCORE)
                if not raw_batch:
                    continue
                validated_batch = _validate_retrieved_contexts(
                    raw_batch,
                    sample_id=sample_id,
                    retrieval_query=query,
                )
                if validated_batch:
                    batches.append(validated_batch)
            search_results = _merge_retrieval_results(batches, k=EVAL_K)
            retrieval_latency = round(time.perf_counter() - retrieval_start, 6)

            contexts = [_truncate_context(item.get("text", "")) for item in search_results]
            contexts = [ctx for ctx in contexts if len(ctx.strip()) > 30]

            sql_context, sql_used, sql_query = _build_sql_context(question_row["question"])

            generation_start = time.perf_counter()
            _sleep_between_requests()
            answer, usage = _generate_answer(
                client=client,
                model=SETTINGS.model_name,
                question=question_row["question"],
                contexts=contexts,
                sql_context=sql_context,
                sample_id=sample_id,
            )
            generation_latency = round(time.perf_counter() - generation_start, 6)
            total_latency = round(time.perf_counter() - total_start, 6)

            sample = _validate_sample(
                {
                    "sample_index": sample_index,
                    "id": sample_id,
                    "category": question_row["category"],
                    "question": question_row["question"],
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": question_row["ground_truth"],
                    "retrieval_keywords": question_row.get("retrieval_keywords", []),
                    "retrieval_queries": retrieval_queries,
                    "sql_used": sql_used,
                    "sql_query": sql_query,
                    "retrieval_latency_s": retrieval_latency,
                    "generation_latency_s": generation_latency,
                    "total_latency_s": total_latency,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["total_tokens"],
                }
            )
            samples.append(sample)

        LOGGER.info(
            "Echantillon genere %s (%s) - retrieval_queries=%s, contexts=%s",
            sample_id,
            question_row["category"],
            len(retrieval_queries),
            len(contexts),
        )
        _sleep_between_requests()

    if len(samples) != len(questions):
        raise RuntimeError(
            f"Nombre d'échantillons incohérent ({len(samples)}) pour {len(questions)} questions."
        )

    return samples


def _compute_retrieval_metrics_for_sample(sample: dict[str, Any]) -> dict[str, float | int | None]:
    contexts = [str(ctx) for ctx in sample.get("contexts", []) if str(ctx).strip()]
    keywords = _resolve_retrieval_keywords(sample)

    if not contexts or not keywords:
        return {
            "retrieval_precision_at_k": None,
            "retrieval_recall_at_k": None,
            "retrieval_mrr": None,
            "retrieval_ndcg_at_k": None,
            "retrieval_keyword_coverage": None,
            "retrieval_keywords_count": len(keywords),
        }

    normalized_contexts = [_normalize_text_for_match(ctx) for ctx in contexts]
    normalized_keywords = [_normalize_text_for_match(kw) for kw in keywords]
    normalized_keywords = [kw for kw in normalized_keywords if kw]

    if not normalized_keywords:
        return {
            "retrieval_precision_at_k": None,
            "retrieval_recall_at_k": None,
            "retrieval_mrr": None,
            "retrieval_ndcg_at_k": None,
            "retrieval_keyword_coverage": None,
            "retrieval_keywords_count": 0,
        }

    per_context_relevance: list[float] = []
    covered_keywords: set[str] = set()
    for context in normalized_contexts:
        hits = [kw for kw in normalized_keywords if kw in context]
        per_context_relevance.append(len(hits) / len(normalized_keywords))
        covered_keywords.update(hits)

    relevant_flags = [score >= RETRIEVAL_OVERLAP_THRESHOLD for score in per_context_relevance]
    relevant_count = sum(1 for flag in relevant_flags if flag)

    precision_at_k = relevant_count / len(normalized_contexts)
    recall_at_k = len(covered_keywords) / len(normalized_keywords)

    mrr = 0.0
    for rank, is_relevant in enumerate(relevant_flags, start=1):
        if is_relevant:
            mrr = 1.0 / rank
            break

    dcg = sum(((2**rel) - 1) / math.log2(rank + 2) for rank, rel in enumerate(per_context_relevance))
    ideal = sorted(per_context_relevance, reverse=True)
    idcg = sum(((2**rel) - 1) / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "retrieval_precision_at_k": round(precision_at_k, 6),
        "retrieval_recall_at_k": round(recall_at_k, 6),
        "retrieval_mrr": round(mrr, 6),
        "retrieval_ndcg_at_k": round(ndcg, 6),
        "retrieval_keyword_coverage": round(recall_at_k, 6),
        "retrieval_keywords_count": len(normalized_keywords),
    }


def _build_additional_metrics_dataframe(samples: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.append(
            {
                "sample_index": sample["sample_index"],
                "id": sample["id"],
                "latency_retrieval_s": sample.get("retrieval_latency_s"),
                "latency_generation_s": sample.get("generation_latency_s"),
                "latency_total_s": sample.get("total_latency_s"),
                "input_tokens": sample.get("input_tokens"),
                "output_tokens": sample.get("output_tokens"),
                "total_tokens": sample.get("total_tokens"),
                "sql_used": 1.0 if sample.get("sql_used") else 0.0,
                **_compute_retrieval_metrics_for_sample(sample),
            }
        )
    return pd.DataFrame(rows)


def _resolve_ragas_models() -> tuple[Any, Any]:
    try:
        try:
            from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
        except Exception:
            from langchain_mistralai.chat_models import ChatMistralAI
            from langchain_mistralai.embeddings import MistralAIEmbeddings

        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except Exception as exc:
        raise RuntimeError(
            "Impossible de charger les wrappers RAGAS Mistral. "
            "Installe les dépendances `ragas` et `langchain-mistralai`."
        ) from exc

    class SafeChatMistralAI(ChatMistralAI):
        def _combine_llm_outputs(self, llm_outputs: list[dict | None]) -> dict:
            overall_token_usage: dict[str, Any] = {}
            for output in llm_outputs:
                if not output:
                    continue
                token_usage = output.get("token_usage")
                if not token_usage:
                    continue
                for key, value in token_usage.items():
                    if isinstance(value, (int, float)):
                        overall_token_usage[key] = overall_token_usage.get(key, 0) + value
                        continue
                    if isinstance(value, dict):
                        previous = overall_token_usage.get(key, {})
                        if not isinstance(previous, dict):
                            previous = {}
                        merged = dict(previous)
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                merged[sub_key] = merged.get(sub_key, 0) + sub_value
                            else:
                                merged[sub_key] = sub_value
                        overall_token_usage[key] = merged
                        continue
                    overall_token_usage[key] = value
            return {"token_usage": overall_token_usage, "model_name": self.model}

    class ThrottledChatMistralAI(SafeChatMistralAI):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            _sleep_between_requests()
            return super()._generate(*args, **kwargs)

        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            await _async_sleep_between_requests()
            return await super()._agenerate(*args, **kwargs)

    class ThrottledMistralAIEmbeddings(MistralAIEmbeddings):
        def embed_query(self, text: str) -> list[float]:
            _sleep_between_requests()
            return super().embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            _sleep_between_requests()
            return super().embed_documents(texts)

        async def aembed_query(self, text: str) -> list[float]:
            await _async_sleep_between_requests()
            return await super().aembed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            await _async_sleep_between_requests()
            return await super().aembed_documents(texts)

    try:
        llm_model = ThrottledChatMistralAI(
            model=SETTINGS.model_name,
            temperature=0.0,
            api_key=SETTINGS.mistral_api_key,
        )
    except TypeError:
        llm_model = ThrottledChatMistralAI(
            model=SETTINGS.model_name,
            temperature=0.0,
            mistral_api_key=SETTINGS.mistral_api_key,
        )

    try:
        embed_model = ThrottledMistralAIEmbeddings(
            model=SETTINGS.embedding_model,
            api_key=SETTINGS.mistral_api_key,
        )
    except TypeError:
        embed_model = ThrottledMistralAIEmbeddings(
            model=SETTINGS.embedding_model,
            mistral_api_key=SETTINGS.mistral_api_key,
        )

    try:
        sanity_vec = embed_model.embed_query("hello")
        if not isinstance(sanity_vec, list) or len(sanity_vec) == 0:
            raise RuntimeError("Sanity check embeddings invalide.")
    except Exception as exc:
        raise RuntimeError("Echec du sanity check embeddings.") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        llm = LangchainLLMWrapper(llm_model)
        embeddings = LangchainEmbeddingsWrapper(embed_model)

    return llm, embeddings


def _resolve_ragas_metrics(llm: Any, embeddings: Any) -> tuple[Any, list[Any], list[str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas import evaluate
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

        metrics: list[Any] = [
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            Faithfulness(llm=llm),
            ContextPrecision(llm=llm),
        ]
        if INCLUDE_CONTEXT_RECALL:
            metrics.append(ContextRecall(llm=llm))

    metric_names = [getattr(metric, "name", metric.__class__.__name__) for metric in metrics]
    return evaluate, metrics, metric_names


def _run_ragas(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    from datasets import Dataset
    from ragas.run_config import RunConfig

    dataset = Dataset.from_dict(
        {
            "question": [sample["question"] for sample in samples],
            "answer": [sample["answer"] for sample in samples],
            "contexts": [sample["contexts"] for sample in samples],
            "ground_truth": [sample["ground_truth"] for sample in samples],
        }
    )

    llm, embeddings = _resolve_ragas_models()
    evaluate_fn, metrics, metric_names = _resolve_ragas_metrics(llm, embeddings)

    run_config = RunConfig(
        timeout=RAGAS_TIMEOUT_SECONDS,
        max_retries=RAGAS_MAX_RETRIES,
        max_wait=RAGAS_MAX_WAIT_SECONDS,
        max_workers=RAGAS_MAX_WORKERS,
    )

    with logfire_span("run_ragas", sample_count=len(samples), metrics=metric_names):
        result = evaluate_fn(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            batch_size=RAGAS_BATCH_SIZE,
            raise_exceptions=STRICT_RAGAS_ERRORS,
            show_progress=False,
        )

    if hasattr(result, "to_dict"):
        summary: dict[str, Any] = result.to_dict()
    elif isinstance(result, dict):
        summary = result
    else:
        summary = {"result_repr": str(result)}

    details = result.to_pandas() if hasattr(result, "to_pandas") else pd.DataFrame()

    sample_indexes = [sample["sample_index"] for sample in samples]
    ids = [sample["id"] for sample in samples]
    categories = [sample["category"] for sample in samples]

    if details.empty:
        details = pd.DataFrame({"sample_index": sample_indexes, "id": ids, "category": categories})
    elif len(details) == len(samples):
        if "sample_index" not in details.columns:
            details.insert(0, "sample_index", sample_indexes)
        if "id" not in details.columns:
            details.insert(1, "id", ids)
        if "category" not in details.columns:
            details.insert(2, "category", categories)
    else:
        raise RuntimeError(
            f"Nombre de lignes détaillées incohérent ({len(details)}) pour {len(samples)} questions."
        )

    additional_df = _build_additional_metrics_dataframe(samples)
    if not additional_df.empty:
        details = details.merge(
            additional_df,
            on=["sample_index", "id"],
            how="left",
            validate="one_to_one",
        )
        for column in additional_df.columns:
            if column in {"sample_index", "id"}:
                continue
            if pd.api.types.is_numeric_dtype(additional_df[column]):
                values = additional_df[column].dropna()
                summary[f"mean_{column}"] = float(values.mean()) if not values.empty else None

    summary["activated_ragas_metrics"] = metric_names
    summary["metrics_profile"] = "core"
    summary["ragas_run_config"] = {
        "max_workers": RAGAS_MAX_WORKERS,
        "max_retries": RAGAS_MAX_RETRIES,
        "max_wait": RAGAS_MAX_WAIT_SECONDS,
        "timeout": RAGAS_TIMEOUT_SECONDS,
        "batch_size": RAGAS_BATCH_SIZE,
        "strict_ragas_errors": STRICT_RAGAS_ERRORS,
    }

    RagasRunOutput.model_validate(
        {
            "sample_count": len(samples),
            "summary": summary,
            "details_rows": len(details),
        }
    )

    return summary, details


def _save_outputs(*, samples: list[dict[str, Any]], summary: dict[str, Any] | None, details: pd.DataFrame | None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    samples_path = OUTPUT_DIR / f"samples_{timestamp}.json"
    samples_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")

    if summary is not None:
        summary_path = OUTPUT_DIR / f"ragas_summary_{timestamp}.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    if details is not None and not details.empty:
        details_path = OUTPUT_DIR / f"ragas_details_{timestamp}.csv"
        details.to_csv(details_path, index=False)


def _configure_observability() -> None:
    configure_logfire(
        enabled=SETTINGS.logfire_enabled,
        send_to_logfire=SETTINGS.logfire_send_to_logfire,
        service_name=SETTINGS.logfire_service_name,
        logger=LOGGER,
        instrument_pydantic=True,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _configure_observability()

    if not SETTINGS.mistral_api_key:
        raise EnvironmentError("MISTRAL_API_KEY est absent de l'environnement.")

    retriever = VectorStoreManager()
    if retriever.index is None or not retriever.document_chunks:
        raise RuntimeError("Index vectoriel introuvable. Lance d'abord `python indexer.py`.")

    questions = _load_questions()
    LOGGER.info("Questions chargées : %s", len(questions))

    client = Mistral(api_key=SETTINGS.mistral_api_key)
    samples = _build_samples(questions=questions, retriever=retriever, client=client)
    LOGGER.info("Echantillons générés : %s", len(samples))

    try:
        LOGGER.info(
            "Lancement RAGAS core: workers=%s, batch_size=%s, strict_errors=%s",
            RAGAS_MAX_WORKERS,
            RAGAS_BATCH_SIZE,
            STRICT_RAGAS_ERRORS,
        )
        summary, details = _run_ragas(samples)
    except Exception as exc:
        _save_outputs(samples=samples, summary=None, details=None)
        LOGGER.error("Echec de l'évaluation RAGAS : %s", exc)
        LOGGER.error("Les échantillons ont été sauvegardés pour diagnostic.")
        raise

    _save_outputs(samples=samples, summary=summary, details=details)

    LOGGER.info("Résumé RAGAS : %s", summary)
    if not details.empty:
        metric_cols = [
            "answer_relevancy",
            "faithfulness",
            "context_precision",
            "context_recall",
            "retrieval_precision_at_k",
            "retrieval_recall_at_k",
            "retrieval_mrr",
            "retrieval_ndcg_at_k",
            "latency_retrieval_s",
            "latency_generation_s",
            "latency_total_s",
        ]
        metric_cols = [column for column in metric_cols if column in details.columns]
        if metric_cols:
            LOGGER.info("Taux de valeurs manquantes : %s", details[metric_cols].isna().mean().to_dict())
        LOGGER.info("Nombre de lignes détaillées : %s", len(details))


if __name__ == "__main__":
    main()

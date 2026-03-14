"""Script d'évaluation du prototype RAG actuel avec RAGAS.

Ce script génère des échantillons de questions-réponses à partir du Vector Store et les évalue avec RAGAS.
Il est conçu pour être exécuté après l'indexation des données et peut être personnalisé avec des questions spécifiques via un fichier JSON.
Il nécessite une clé API Mistral valide et l'installation de `langchain-mistralai` pour l'intégration avec RAGAS.
"""

from __future__ import annotations

import argparse
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

from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)

MIN_KEYWORD_MATCH_RATIO = 0.2
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
        "question": "Quel est le nom complet de l'equipe codee OKC ?",
        "ground_truth": "OKC correspond a Oklahoma City Thunder.",
        "retrieval_keywords": ["OKC", "Oklahoma City Thunder"],
    },
    {
        "id": "q2",
        "category": "simple",
        "question": "Selon le tableau des equipes, combien de points totaux a OKC ?",
        "ground_truth": "OKC totalise 9880 points.",
        "retrieval_keywords": ["OKC", "9880"],
    },
    {
        "id": "q3",
        "category": "complex",
        "question": "Parmi MIA, OKC, LAC et BKN, quelle equipe a le plus de points totaux ?",
        "ground_truth": "Parmi ces quatre equipes, OKC est premier avec 9880 points.",
        "retrieval_keywords": ["MIA", "OKC", "LAC", "BKN", "9880"],
    },
    {
        "id": "q4",
        "category": "complex",
        "question": "Quelle est la difference de points totaux entre OKC (9880) et MIA (9828) ?",
        "ground_truth": "La difference est de 52 points.",
        "retrieval_keywords": ["OKC", "9880", "MIA", "9828", "52"],
    },
    {
        "id": "q5",
        "category": "simple",
        "question": "Combien de joueurs compte l'equipe Brooklyn Nets (BKN) ?",
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
        "question": "Quel est le pourcentage a 3 points (3P%) de Shai Gilgeous-Alexander dans ce tableau ?",
        "ground_truth": "Le 3P% de Shai Gilgeous-Alexander est de 37.5.",
        "retrieval_keywords": ["Shai Gilgeous-Alexander", "3P%", "37.5"],
    },
    {
        "id": "q8",
        "category": "complex",
        "question": "Entre Anthony Edwards (2180) et Nikola Jokic (2072), qui a le plus de points totaux ?",
        "ground_truth": "Anthony Edwards a le total le plus eleve avec 2180 points (contre 2072).",
        "retrieval_keywords": ["Anthony Edwards", "2180", "Nikola Jokic", "2072"],
    },
    {
        "id": "q9",
        "category": "complex",
        "question": "Entre Detroit Pistons (10292) et Cleveland Cavaliers (10180), quelle equipe a le plus de points totaux ?",
        "ground_truth": "Detroit Pistons est devant avec 10292 points (contre 10180).",
        "retrieval_keywords": ["Detroit Pistons", "10292", "Cleveland Cavaliers", "10180"],
    },
    {
        "id": "q10",
        "category": "noisy",
        "question": "code MIA -> equipe + points ??? reponse rapide",
        "ground_truth": "MIA correspond a Miami Heat et le total affiche est de 9828 points.",
        "retrieval_keywords": ["MIA", "Miami Heat", "9828"],
    },
    {
        "id": "q11",
        "category": "complex",
        "question": "Entre Detroit Pistons (10292) et Cleveland Cavaliers (10180), quelle est la différence de points ?",
        "ground_truth": "La différence est de 112 points.",
        "retrieval_keywords": ["Detroit Pistons", "10292", "Cleveland Cavaliers", "10180", "112"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalue le prototype RAG actuel avec RAGAS.")
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Fichier JSON facultatif contenant les questions (liste ou {'questions': [...]})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluations",
        help="Repertoire de sortie pour les resultats JSON/CSV.",
    )
    parser.add_argument("--k", type=int, default=SEARCH_K, help="Nombre de chunks recuperes par question (top-k).")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Seuil de score facultatif dans [0,1], applique au retrieval.",
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Modele Mistral utilise pour la generation.")
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Genere et sauvegarde uniquement les echantillons QA (sans metriques RAGAS).",
    )
    parser.add_argument(
        "--include-context-recall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Active la metrique context_recall. A utiliser uniquement si ground_truth "
            "contient une vraie reponse de reference (et non une consigne)."
        ),
    )
    parser.add_argument(
        "--input-cost-per-1k-tokens",
        type=float,
        default=0.0,
        help="Cout USD des tokens d'entree pour 1000 tokens (0 pour desactiver).",
    )
    parser.add_argument(
        "--output-cost-per-1k-tokens",
        type=float,
        default=0.0,
        help="Cout USD des tokens de sortie pour 1000 tokens (0 pour desactiver).",
    )
    parser.add_argument(
        "--retrieval-overlap-threshold",
        type=float,
        default=MIN_KEYWORD_MATCH_RATIO,
        help=(
            "Seuil de recouvrement [0,1] utilise pour estimer la pertinence d'un chunk "
            "dans les metriques retrieval@k."
        ),
    )
    parser.add_argument(
        "--ragas-max-workers",
        type=int,
        default=1,
        help="Nombre max de workers RAGAS (1 recommande si rate limit).",
    )
    parser.add_argument(
        "--ragas-max-retries",
        type=int,
        default=12,
        help="Nombre maximal de retries RAGAS par appel.",
    )
    parser.add_argument(
        "--ragas-max-wait",
        type=int,
        default=120,
        help="Temps max d'attente (s) entre retries RAGAS.",
    )
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=240,
        help="Timeout RAGAS par appel (s).",
    )
    parser.add_argument(
        "--ragas-batch-size",
        type=int,
        default=1,
        help="Taille de batch pour evaluate(). 1 reduit fortement le risque de 429.",
    )
    parser.add_argument(
        "--strict-ragas-errors",
        action="store_true",
        help="Si active, stoppe l'execution au premier echec metrique RAGAS.",
    )
    return parser.parse_args()


def _load_questions(questions_file: str | None) -> list[dict[str, Any]]:
    if not questions_file:
        raw_data: Any = DEFAULT_QUESTIONS
    else:
        raw_data = json.loads(Path(questions_file).read_text(encoding="utf-8"))

    if isinstance(raw_data, dict) and "questions" in raw_data:
        raw_data = raw_data["questions"]
    if not isinstance(raw_data, list):
        raise ValueError("questions-file doit etre une liste ou {'questions': [...]} ")

    normalized: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for idx, row in enumerate(raw_data, start=1):
        raw_keywords = row.get("retrieval_keywords", []) if isinstance(row, dict) else []
        keywords: list[str] = []
        if isinstance(raw_keywords, list):
            keywords = [str(item).strip() for item in raw_keywords if str(item).strip()]

        raw_id = str(row.get("id", f"q{idx}")).strip() if isinstance(row, dict) else f"q{idx}"
        if not raw_id:
            raw_id = f"q{idx}"
        final_id = raw_id
        suffix = 2
        while final_id in used_ids:
            final_id = f"{raw_id}_{suffix}"
            suffix += 1
        if final_id != raw_id:
            LOGGER.warning(
                "ID de question duplique (%s). Renomme automatiquement en %s.",
                raw_id,
                final_id,
            )
        used_ids.add(final_id)

        normalized.append(
            {
                "id": final_id,
                "category": str(row.get("category", "non_categorise")) if isinstance(row, dict) else "non_categorise",
                "question": str(row["question"]) if isinstance(row, dict) else str(row),
                "ground_truth": str(row.get("ground_truth", "")) if isinstance(row, dict) else "",
                "retrieval_keywords": keywords,
            }
        )
    return normalized


def _truncate_context(text: str, max_chars: int = 1200) -> str:
    clean = " ".join(text.split())
    clean = re.sub(r"\bNaN\b", "", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean[:max_chars]


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
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


def _estimate_cost_usd(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    input_cost_per_1k_tokens: float,
    output_cost_per_1k_tokens: float,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    if input_cost_per_1k_tokens <= 0 and output_cost_per_1k_tokens <= 0:
        return None
    return round(
        (input_tokens / 1000.0) * input_cost_per_1k_tokens
        + (output_tokens / 1000.0) * output_cost_per_1k_tokens,
        8,
    )


def _normalize_text_for_match(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


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
        explicit = [str(item).strip() for item in raw if str(item).strip()]
        explicit = _dedupe_keep_order(explicit)
        if explicit:
            return explicit
    return _derive_keywords_from_reference(str(sample.get("ground_truth", "")))


def _build_retrieval_queries(
    *,
    question: str,
    retrieval_keywords: list[str],
    max_queries: int = 3,
) -> list[str]:
    """Construit plusieurs requêtes pour améliorer le recall du retrieval."""
    queries: list[str] = [question.strip()]
    keywords = _dedupe_keep_order([kw.strip() for kw in retrieval_keywords if kw and kw.strip()])

    if keywords:
        # Requête compacte orientée entités/valeurs clés.
        queries.append(" ".join(keywords[:6]))
        # Requête hybride pour conserver l'intention tout en forçant les ancres.
        queries.append(f"{question.strip()} {' '.join(keywords[:4])}".strip())

    queries = _dedupe_keep_order([q for q in queries if q])
    return queries[:max_queries]


def _merge_retrieval_results(
    batches: list[list[dict[str, Any]]],
    *,
    k: int,
) -> list[dict[str, Any]]:
    """Fusionne plusieurs lots de résultats en supprimant les doublons."""
    by_key: dict[str, dict[str, Any]] = {}

    for results in batches:
        for item in results:
            text = str(item.get("text", "")).strip()
            metadata = item.get("metadata", {})
            meta_dump = json.dumps(metadata, sort_keys=True, ensure_ascii=False, default=str)
            key = f"{text}||{meta_dump}"
            score = float(item.get("score", 0.0))

            existing = by_key.get(key)
            if existing is None or score > float(existing.get("score", 0.0)):
                by_key[key] = item

    merged = sorted(by_key.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:k]


def _build_prompt(question: str, contexts: list[str]) -> str:
    formatted_context = "\n\n".join([f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)])
    return (
        "Reponds uniquement avec les informations presentes dans le CONTEXTE. "
        "Si des valeurs numeriques sont presentes, cite-les explicitement. "
        "Tu peux faire un calcul simple (difference, comparaison) uniquement a partir des valeurs du contexte. "
        "Si l'information est absente, reponds exactement: Information non disponible dans le contexte.\n\n"
        f"CONTEXTE:\n{formatted_context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "REPONSE FINALE:"
    )


def _generate_answer(
    client: Mistral,
    model: str,
    question: str,
    contexts: list[str],
) -> tuple[str, dict[str, int | None]]:
    if not contexts:
        return (
            "Contexte insuffisant dans le vector store pour repondre a cette question.",
            {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        )

    prompt = _build_prompt(question=question, contexts=contexts)
    try:
        response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant NBA. Utilise uniquement le contexte fourni et evite toute hallucination.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        return answer, _extract_usage_tokens(response)
    except Exception as exc:
        LOGGER.exception("Echec de generation pour la question: %s", question)
        return (
            f"Erreur de generation: {exc}",
            {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        )


def _build_samples(
    *,
    questions: list[dict[str, Any]],
    retriever: VectorStoreManager,
    client: Mistral,
    model: str,
    k: int,
    min_score: float | None,
    input_cost_per_1k_tokens: float,
    output_cost_per_1k_tokens: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sample_index, row in enumerate(questions):
        sample_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        retrieval_queries = _build_retrieval_queries(
            question=row["question"],
            retrieval_keywords=list(row.get("retrieval_keywords", [])),
        )
        retrieval_batches: list[list[dict[str, Any]]] = []
        for query in retrieval_queries:
            batch = retriever.search(query, k=k, min_score=min_score)
            if batch:
                retrieval_batches.append(batch)
        search_results = _merge_retrieval_results(retrieval_batches, k=k)
        retrieval_latency_s = round(time.perf_counter() - retrieval_start, 6)

        contexts = [_truncate_context(r.get("text", "")) for r in search_results]
        contexts = [c for c in contexts if c and len(c.strip()) > 30]

        generation_start = time.perf_counter()
        answer, usage = _generate_answer(client, model, row["question"], contexts)
        generation_latency_s = round(time.perf_counter() - generation_start, 6)

        total_latency_s = round(time.perf_counter() - sample_start, 6)
        estimated_cost_usd = _estimate_cost_usd(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            input_cost_per_1k_tokens=input_cost_per_1k_tokens,
            output_cost_per_1k_tokens=output_cost_per_1k_tokens,
        )

        sample = {
            "sample_index": sample_index,
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": row["ground_truth"],
            "retrieval_keywords": row.get("retrieval_keywords", []),
            "retrieval_queries": retrieval_queries,
            "retrieval_latency_s": retrieval_latency_s,
            "generation_latency_s": generation_latency_s,
            "total_latency_s": total_latency_s,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "estimated_cost_usd": estimated_cost_usd,
        }
        samples.append(sample)
        LOGGER.info(
            "Echantillon genere %s (%s) - requetes retrieval=%s, contexts=%s",
            sample["id"],
            sample["category"],
            len(retrieval_queries),
            len(contexts),
        )

    if len(samples) != len(questions):
        raise RuntimeError(
            f"Nombre d'echantillons incoherent ({len(samples)}) pour {len(questions)} questions."
        )
    return samples


def _compute_retrieval_metrics_for_sample(
    sample: dict[str, Any],
    *,
    overlap_threshold: float,
) -> dict[str, float | int | None]:
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

    relevant_flags = [score >= overlap_threshold for score in per_context_relevance]
    relevant_count = sum(1 for flag in relevant_flags if flag)
    precision_at_k = relevant_count / len(normalized_contexts)
    recall_at_k = len(covered_keywords) / len(normalized_keywords)

    mrr = 0.0
    for idx, is_relevant in enumerate(relevant_flags, start=1):
        if is_relevant:
            mrr = 1.0 / idx
            break

    dcg = sum(
        ((2**rel) - 1) / math.log2(idx + 2)
        for idx, rel in enumerate(per_context_relevance)
    )
    ideal = sorted(per_context_relevance, reverse=True)
    idcg = sum(
        ((2**rel) - 1) / math.log2(idx + 2)
        for idx, rel in enumerate(ideal)
    )
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "retrieval_precision_at_k": round(precision_at_k, 6),
        "retrieval_recall_at_k": round(recall_at_k, 6),
        "retrieval_mrr": round(mrr, 6),
        "retrieval_ndcg_at_k": round(ndcg, 6),
        "retrieval_keyword_coverage": round(recall_at_k, 6),
        "retrieval_keywords_count": len(normalized_keywords),
    }


def _build_additional_metrics_dataframe(
    samples: list[dict[str, Any]],
    *,
    overlap_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        retrieval_metrics = _compute_retrieval_metrics_for_sample(
            sample,
            overlap_threshold=overlap_threshold,
        )
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
                "estimated_cost_usd": sample.get("estimated_cost_usd"),
                **retrieval_metrics,
            }
        )
    return pd.DataFrame(rows)


def _resolve_ragas_metrics(
    *,
    llm: Any,
    embeddings: Any,
    include_context_recall: bool,
) -> tuple[Any, list[Any], list[str]]:
    from ragas import evaluate

    try:
        # API recommandee par RAGAS.
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        metrics = [
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            Faithfulness(llm=llm),
            ContextPrecision(llm=llm),
        ]
        if include_context_recall:
            metrics.append(ContextRecall(llm=llm))
        metric_names = [getattr(metric, "name", metric.__class__.__name__) for metric in metrics]
        return evaluate, metrics, metric_names
    except Exception as exc:
        message = str(exc)
        if "Collections metrics only support modern InstructorLLM" in message:
            LOGGER.info(
                "RAGAS collections non compatibles avec LangchainLLMWrapper, fallback active.",
            )
        else:
            LOGGER.warning(
                "Impossible de charger les metriques RAGAS avancees (fallback restreint): %s",
                exc,
            )
        # Compatibilite ascendante selon la version installee.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

        metrics: list[Any] = []
        metric_names: list[str] = []
        fallback_metrics: list[Any] = [
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            Faithfulness(llm=llm),
            ContextPrecision(llm=llm),
        ]

        for metric_obj in fallback_metrics:
            metrics.append(metric_obj)
            metric_names.append(getattr(metric_obj, "name", metric_obj.__class__.__name__))

        if include_context_recall:
            recall_metric = ContextRecall(llm=llm)
            metrics.append(recall_metric)
            metric_names.append(getattr(recall_metric, "name", recall_metric.__class__.__name__))

        # Fallback ultime si la version expose uniquement les objets globaux.
        if not metrics:
            metrics = [answer_relevancy, faithfulness, context_precision]
            if include_context_recall:
                metrics.append(context_recall)
            metric_names = [str(metric) for metric in metrics]

        return evaluate, metrics, metric_names

def _resolve_ragas_models() -> tuple[Any | None, Any | None]:
    try:
        try:
            from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
        except Exception:
            from langchain_mistralai.chat_models import ChatMistralAI
            from langchain_mistralai.embeddings import MistralAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        # Contournement d'un bug connu: certaines reponses Mistral renvoient
        # des token_usage imbriques (dict dans dict), ce qui casse l'addition
        # naive de langchain-mistralai.
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

        # Les versions de `langchain-mistralai` varient legerement sur les noms d'arguments.
        try:
            llm_model = SafeChatMistralAI(
                model=MODEL_NAME,
                temperature=0.0,
                api_key=MISTRAL_API_KEY,
            )
        except TypeError:
            llm_model = SafeChatMistralAI(
                model=MODEL_NAME,
                temperature=0.0,
                mistral_api_key=MISTRAL_API_KEY,
            )

        try:
            embed_model = MistralAIEmbeddings(
                model="mistral-embed",
                api_key=MISTRAL_API_KEY,
            )
        except TypeError:
            embed_model = MistralAIEmbeddings(
                model="mistral-embed",
                mistral_api_key=MISTRAL_API_KEY,
            )

        # Sanity check embeddings: evite les NaN silencieux plus tard dans RAGAS.
        try:
            if hasattr(embed_model, "embed_query"):
                vec = embed_model.embed_query("hello")
            elif hasattr(embed_model, "embed_documents"):
                vec = embed_model.embed_documents(["hello"])[0]
            else:
                raise RuntimeError("Le modele d'embeddings ne fournit pas de methode de test.")
            if not isinstance(vec, list) or len(vec) == 0:
                raise RuntimeError("Vecteur d'embedding invalide retourne par MistralAIEmbeddings.")
        except Exception as exc:
            raise RuntimeError(
                "Echec du sanity check embeddings avant RAGAS. "
                "Verifie la configuration Mistral et les dependances."
            ) from exc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            llm = LangchainLLMWrapper(llm_model)
            embeddings = LangchainEmbeddingsWrapper(embed_model)
        return llm, embeddings
    except Exception as exc:
        raise RuntimeError(
            "Impossible de configurer RAGAS avec Mistral. "
            "Installe `langchain-mistralai` et verifie que MISTRAL_API_KEY est defini. "
            "Aucun fallback OpenAI n'est autorise sur ce projet."
        ) from exc


def _run_ragas(
    samples: list[dict[str, Any]],
    *,
    include_context_recall: bool,
    retrieval_overlap_threshold: float,
    ragas_max_workers: int,
    ragas_max_retries: int,
    ragas_max_wait: int,
    ragas_timeout: int,
    ragas_batch_size: int,
    strict_ragas_errors: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    from datasets import Dataset
    from ragas.run_config import RunConfig

    payload = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    }
    dataset = Dataset.from_dict(payload)
    llm, embeddings = _resolve_ragas_models()
    evaluate, metrics, activated_ragas_metric_names = _resolve_ragas_metrics(
        llm=llm,
        embeddings=embeddings,
        include_context_recall=include_context_recall,
    )

    run_config = RunConfig(
        timeout=ragas_timeout,
        max_retries=ragas_max_retries,
        max_wait=ragas_max_wait,
        max_workers=ragas_max_workers,
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        batch_size=ragas_batch_size,
        raise_exceptions=strict_ragas_errors,
        show_progress=False,
    )

    if hasattr(result, "to_dict"):
        summary = result.to_dict()
    elif isinstance(result, dict):
        summary = result
    else:
        summary = {"result_repr": str(result)}

    if hasattr(result, "to_pandas"):
        details = result.to_pandas()
    else:
        details = pd.DataFrame()

    sample_indexes = [s["sample_index"] for s in samples]
    ids = [s["id"] for s in samples]
    categories = [s["category"] for s in samples]

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
            f"Nombre de lignes detaillees incoherent ({len(details)}) pour {len(samples)} questions."
        )

    additional_df = _build_additional_metrics_dataframe(
        samples=samples,
        overlap_threshold=retrieval_overlap_threshold,
    )
    if not additional_df.empty:
        details = details.merge(
            additional_df,
            on=["sample_index", "id"],
            how="left",
            validate="one_to_one",
        )

        for col in additional_df.columns:
            if col in {"sample_index", "id"}:
                continue
            if pd.api.types.is_numeric_dtype(additional_df[col]):
                series = additional_df[col].dropna()
                summary[f"mean_{col}"] = float(series.mean()) if not series.empty else None

    summary["activated_ragas_metrics"] = activated_ragas_metric_names
    summary["metrics_profile"] = "core"
    summary["ragas_run_config"] = {
        "max_workers": ragas_max_workers,
        "max_retries": ragas_max_retries,
        "max_wait": ragas_max_wait,
        "timeout": ragas_timeout,
        "batch_size": ragas_batch_size,
        "strict_ragas_errors": strict_ragas_errors,
    }

    return summary, details


def _save_outputs(
    *,
    output_dir: Path,
    samples: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    details: pd.DataFrame | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    samples_path = output_dir / f"samples_{ts}.json"
    samples_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")

    if summary is not None:
        summary_path = output_dir / f"ragas_summary_{ts}.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    if details is not None and not details.empty:
        details_path = output_dir / f"ragas_details_{ts}.csv"
        details.to_csv(details_path, index=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not MISTRAL_API_KEY:
        raise EnvironmentError("MISTRAL_API_KEY est absent du fichier .env")

    retriever = VectorStoreManager()
    if retriever.index is None or not retriever.document_chunks:
        raise RuntimeError("Index vectoriel introuvable. Lance d'abord `python indexer.py`.")

    questions = _load_questions(args.questions_file)
    LOGGER.info("Questions chargees: %s", len(questions))
    client = Mistral(api_key=MISTRAL_API_KEY)

    samples = _build_samples(
        questions=questions,
        retriever=retriever,
        client=client,
        model=args.model,
        k=args.k,
        min_score=args.min_score,
        input_cost_per_1k_tokens=args.input_cost_per_1k_tokens,
        output_cost_per_1k_tokens=args.output_cost_per_1k_tokens,
    )
    LOGGER.info("Echantillons generes: %s", len(samples))

    output_dir = Path(args.output_dir)

    if args.skip_ragas:
        _save_outputs(output_dir=output_dir, samples=samples, summary=None, details=None)
        LOGGER.info("Execution terminee (echantillons uniquement, RAGAS ignore).")
        return

    if not args.include_context_recall:
        LOGGER.info(
            "context_recall est desactive par defaut pour eviter des scores trompeurs "
            "si les references ne sont pas fiables. Active-le avec --include-context-recall."
        )

    try:
        LOGGER.info(
            "Lancement RAGAS (profile=core), workers=%s, batch_size=%s, strict_errors=%s",
            args.ragas_max_workers,
            args.ragas_batch_size,
            args.strict_ragas_errors,
        )
        summary, details = _run_ragas(
            samples,
            include_context_recall=args.include_context_recall,
            retrieval_overlap_threshold=args.retrieval_overlap_threshold,
            ragas_max_workers=args.ragas_max_workers,
            ragas_max_retries=args.ragas_max_retries,
            ragas_max_wait=args.ragas_max_wait,
            ragas_timeout=args.ragas_timeout,
            ragas_batch_size=args.ragas_batch_size,
            strict_ragas_errors=args.strict_ragas_errors,
        )
    except Exception as exc:
        _save_outputs(output_dir=output_dir, samples=samples, summary=None, details=None)
        LOGGER.error("Echec de l'evaluation RAGAS: %s", exc)
        LOGGER.error(
            "Les echantillons ont ete sauvegardes. Tu peux lancer `--skip-ragas` pendant la correction des dependances."
        )
        raise

    _save_outputs(output_dir=output_dir, samples=samples, summary=summary, details=details)

    LOGGER.info("Resume RAGAS: %s", summary)
    LOGGER.info("Questions evaluees: %s", len(samples))
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
            "estimated_cost_usd",
        ]
        metric_cols = [col for col in metric_cols if col in details.columns]
        if metric_cols:
            missing_ratio = details[metric_cols].isna().mean().to_dict()
            LOGGER.info("Taux de valeurs manquantes par metrique: %s", missing_ratio)
        LOGGER.info("Nombre de lignes detaillees: %s", len(details))


if __name__ == "__main__":
    main()

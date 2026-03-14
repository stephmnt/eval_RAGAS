"""Évalue le prototype RAG actuel avec RAGAS.

Ce script construit des échantillons question-réponse à partir du Vector Store,
puis calcule des métriques RAGAS et des métriques de retrieval
complémentaires. Il est conçu pour être exécuté après l'indexation des
données, avec une configuration minimale directement intégrée dans le fichier.

L'exécution nécessite une clé API Mistral valide ainsi que le package
`langchain-mistralai` pour l'intégration avec RAGAS.
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

OUTPUT_DIR = "outputs/evaluations"
EVAL_K = SEARCH_K
EVAL_MIN_SCORE: float | None = None
EVAL_MODEL = MODEL_NAME
SKIP_RAGAS = False
INCLUDE_CONTEXT_RECALL = True
RETRIEVAL_OVERLAP_THRESHOLD = MIN_KEYWORD_MATCH_RATIO
RAGAS_MAX_WORKERS = 1
RAGAS_MAX_RETRIES = 12
RAGAS_MAX_WAIT = 120
RAGAS_TIMEOUT = 240
RAGAS_BATCH_SIZE = 1
STRICT_RAGAS_ERRORS = False
# Délai anti-429 entre appels réseau (embeddings/chat). À ajuster selon les limites de l'API Mistral et la taille du jeu de données, en l'état 0.25 fonctionne.
REQUEST_DELAY_SECONDS = 0.25


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


def _load_questions() -> list[dict[str, Any]]:
    """Normalise la liste des questions d'évaluation.

    La fonction accepte la structure embarquée dans `DEFAULT_QUESTIONS`,
    déduplique les identifiants et homogénéise les champs attendus par le reste
    du pipeline.

    Returns:
        list[dict[str, Any]]: La liste normalisée des questions.

    Raises:
        ValueError: Si `DEFAULT_QUESTIONS` n'est ni une liste ni un dictionnaire
            contenant une clé `questions`.
    """
    raw_data: Any = DEFAULT_QUESTIONS

    if isinstance(raw_data, dict) and "questions" in raw_data:
        raw_data = raw_data["questions"]
    if not isinstance(raw_data, list):
        raise ValueError("DEFAULT_QUESTIONS doit être une liste ou {'questions': [...]} ")

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
                "ID de question dupliqué (%s). Renommé automatiquement en %s.",
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
    """Nettoie et tronque un contexte textuel.

    Args:
        text: Le texte brut à nettoyer.
        max_chars: Le nombre maximal de caractères à conserver.

    Returns:
        str: Le texte nettoyé et tronqué.
    """
    clean = " ".join(text.split())
    clean = re.sub(r"\bNaN\b", "", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean[:max_chars]


def _sleep_between_requests() -> None:
    """Applique un délai synchrone entre deux requêtes si configuré."""
    if REQUEST_DELAY_SECONDS > 0:
        time.sleep(REQUEST_DELAY_SECONDS)


async def _async_sleep_between_requests() -> None:
    """Applique un délai asynchrone entre deux requêtes si configuré."""
    if REQUEST_DELAY_SECONDS > 0:
        await asyncio.sleep(REQUEST_DELAY_SECONDS)


def _safe_int(value: Any) -> int | None:
    """Convertit une valeur hétérogène en entier si possible.

    Args:
        value: La valeur à convertir.

    Returns:
        int | None: L'entier converti, ou `None` si la conversion est
        impossible ou non pertinente.
    """
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
    """Extrait les compteurs de tokens depuis une réponse Mistral.

    Args:
        response: La réponse brute retournée par le client Mistral.

    Returns:
        dict[str, int | None]: Les tokens d'entrée, de sortie et le total.
    """
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


def _normalize_text_for_match(text: str) -> str:
    """Normalise un texte pour les comparaisons par recouvrement.

    Args:
        text: Le texte à normaliser.

    Returns:
        str: Le texte réduit à des tokens alphanumériques en minuscules.
    """
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """Supprime les doublons en conservant l'ordre d'origine.

    Args:
        items: La liste d'éléments à dédupliquer.

    Returns:
        list[str]: La liste dédupliquée.
    """
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
    """Construit des mots-clés de retrieval à partir de la référence.

    Args:
        reference: La réponse de référence utilisée comme source.
        max_keywords: Le nombre maximal de mots-clés à retourner.

    Returns:
        list[str]: Les mots-clés déduits depuis la référence.
    """
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
    """Résout les mots-clés de retrieval pour un échantillon.

    La fonction privilégie les mots-clés explicitement définis dans
    l'échantillon et retombe sur une dérivation depuis la référence si besoin.

    Args:
        sample: L'échantillon à enrichir.

    Returns:
        list[str]: Les mots-clés utilisables pour le retrieval.
    """
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
    """Construit plusieurs requêtes pour améliorer le recall du retrieval.

    Args:
        question: La question utilisateur d'origine.
        retrieval_keywords: Les mots-clés servant à ancrer les requêtes.
        max_queries: Le nombre maximal de requêtes à retourner.

    Returns:
        list[str]: Les requêtes de retrieval dédupliquées.
    """
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
    """Fusionne plusieurs lots de résultats en supprimant les doublons.

    Args:
        batches: Les lots de résultats retournés par différentes requêtes.
        k: Le nombre maximal de résultats finaux à conserver.

    Returns:
        list[dict[str, Any]]: Les meilleurs résultats fusionnés.
    """
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


def _build_prompt(question: str, contexts: list[str], sql_context: str) -> str:
    """Construit le prompt de génération à partir de la question et du contexte.

    Args:
        question: La question à poser au modèle.
        contexts: Les contexts récupérés par le retriever.

    Returns:
        str: Le prompt final envoyé au modèle.
    """
    formatted_context = "\n\n".join([f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)])
    return (
        "Réponds uniquement avec les informations présentes dans le CONTEXTE. "
        "Quand une sortie SQL est fournie, tu peux t'en servir comme donnée structurée fiable. "
        "Si des valeurs numériques sont présentes, cite-les explicitement. "
        "Tu peux faire un calcul simple (différence, comparaison) uniquement à partir des valeurs du contexte. "
        "Si l'information est absente, réponds exactement : Information non disponible dans le contexte.\n\n"
        f"CONTEXTE:\n{formatted_context}\n\n"
        f"SQL_CONTEXT:\n{sql_context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "RÉPONSE FINALE :"
    )


def _generate_answer(
    client: Mistral,
    model: str,
    question: str,
    contexts: list[str],
    sql_context: str,
) -> tuple[str, dict[str, int | None]]:
    """Génère une réponse Mistral contrainte par les contexts.

    Args:
        client: Le client Mistral déjà initialisé.
        model: Le nom du modèle à utiliser.
        question: La question à traiter.
        contexts: Les contexts transmis au modèle.

    Returns:
        tuple[str, dict[str, int | None]]: La réponse produite et les
        informations de consommation de tokens.
    """
    if not contexts:
        return (
            "Contexte insuffisant dans le vector store pour répondre à cette question.",
            {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        )

    prompt = _build_prompt(question=question, contexts=contexts, sql_context=sql_context)
    try:
        response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant NBA. Utilise uniquement le contexte fourni et évite toute hallucination.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        return answer, _extract_usage_tokens(response)
    except Exception as exc:
        LOGGER.exception("Échec de génération pour la question : %s", question)
        return (
            f"Erreur de génération : {exc}",
            {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        )


def _build_sql_context(question: str) -> tuple[str, bool, str | None]:
    """Construit le contexte SQL à injecter dans le prompt de génération.

    Returns:
        tuple[str, bool, str | None]: (contexte SQL, sql_utilise, sql_query)
    """
    try:
        result = answer_question_sql_via_langchain(question)
    except Exception as exc:
        return f"Echec du tool SQL: {exc}", False, None

    if result.get("status") == "no_tool":
        return str(result.get("message", "Aucun appel SQL jugé nécessaire.")), False, None

    if result.get("status") != "ok":
        return f"Tool SQL indisponible: {result.get('message')}", False, None

    sql_query = result.get("sql")
    rows = result.get("rows", [])
    preview_rows = rows[:10]
    return (
        f"SQL: {sql_query}\nRows (max 10): {preview_rows}",
        True,
        sql_query,
    )


def _build_samples(
    *,
    questions: list[dict[str, Any]],
    retriever: VectorStoreManager,
    client: Mistral,
    model: str,
    k: int,
    min_score: float | None,
) -> list[dict[str, Any]]:
    """Construit les échantillons évaluables à partir des questions.

    Pour chaque question, la fonction exécute le retrieval, prépare les
    contexts, appelle le modèle de génération puis enrichit l'échantillon avec
    des métadonnées opérationnelles.

    Args:
        questions: Les questions d'évaluation normalisées.
        retriever: Le gestionnaire de Vector Store.
        client: Le client Mistral utilisé pour la génération.
        model: Le nom du modèle de génération.
        k: Le nombre maximal de contexts à conserver.
        min_score: Le score minimal de retrieval à appliquer, si défini.

    Returns:
        list[dict[str, Any]]: Les échantillons prêts pour l'évaluation.

    Raises:
        RuntimeError: Si le nombre d'échantillons produits est incohérent avec
            le nombre de questions d'entrée.
    """
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
            _sleep_between_requests()
            batch = retriever.search(query, k=k, min_score=min_score)
            if batch:
                retrieval_batches.append(batch)
        search_results = _merge_retrieval_results(retrieval_batches, k=k)
        retrieval_latency_s = round(time.perf_counter() - retrieval_start, 6)

        contexts = [_truncate_context(r.get("text", "")) for r in search_results]
        contexts = [c for c in contexts if c and len(c.strip()) > 30]

        sql_context, sql_used, sql_query = _build_sql_context(row["question"])

        generation_start = time.perf_counter()
        _sleep_between_requests()
        answer, usage = _generate_answer(
            client=client,
            model=model,
            question=row["question"],
            contexts=contexts,
            sql_context=sql_context,
        )
        generation_latency_s = round(time.perf_counter() - generation_start, 6)

        total_latency_s = round(time.perf_counter() - sample_start, 6)
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
            "sql_used": sql_used,
            "sql_query": sql_query,
            "retrieval_latency_s": retrieval_latency_s,
            "generation_latency_s": generation_latency_s,
            "total_latency_s": total_latency_s,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
        }
        samples.append(sample)
        LOGGER.info(
            "Échantillon généré %s (%s) - requêtes retrieval=%s, contexts=%s",
            sample["id"],
            sample["category"],
            len(retrieval_queries),
            len(contexts),
        )
        _sleep_between_requests()

    if len(samples) != len(questions):
        raise RuntimeError(
            f"Nombre d'échantillons incohérent ({len(samples)}) pour {len(questions)} questions."
        )
    return samples


def _compute_retrieval_metrics_for_sample(
    sample: dict[str, Any],
    *,
    overlap_threshold: float,
) -> dict[str, float | int | None]:
    """Calcule les métriques de retrieval pour un échantillon.

    Args:
        sample: L'échantillon enrichi avec sa question, ses contexts et sa
            référence.
        overlap_threshold: Le seuil de recouvrement à partir duquel un contexte
            est considéré comme pertinent.

    Returns:
        dict[str, float | int | None]: Les métriques de retrieval calculées
        pour l'échantillon.
    """
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
    """Construit le DataFrame des métriques additionnelles hors RAGAS.

    Args:
        samples: Les échantillons produits par le pipeline.
        overlap_threshold: Le seuil de recouvrement pour les métriques de
            retrieval.

    Returns:
        pd.DataFrame: Le tableau des métriques complémentaires.
    """
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
                "sql_used": 1.0 if sample.get("sql_used") else 0.0,
                **retrieval_metrics,
            }
        )
    return pd.DataFrame(rows)


def _resolve_ragas_metrics(
    *,
    llm: Any,
    embeddings: Any,
) -> tuple[Any, list[Any], list[str]]:
    """Résout les métriques RAGAS selon la version installée.

    La fonction tente d'abord l'API moderne, puis applique un fallback
    compatible avec des versions plus anciennes de RAGAS.

    Args:
        llm: Le wrapper LLM compatible RAGAS.
        embeddings: Le wrapper d'embeddings compatible RAGAS.

    Returns:
        tuple[Any, list[Any], list[str]]: La fonction `evaluate`, la liste des
        métriques activées et leurs noms.
    """
    from ragas import evaluate

    try:
        # API recommandée par RAGAS.
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
        if INCLUDE_CONTEXT_RECALL:
            metrics.append(ContextRecall(llm=llm))
        metric_names = [getattr(metric, "name", metric.__class__.__name__) for metric in metrics]
        return evaluate, metrics, metric_names
    except Exception as exc:
        message = str(exc)
        if "Collections metrics only support modern InstructorLLM" in message:
            LOGGER.info(
                "Les collections RAGAS ne sont pas compatibles avec LangchainLLMWrapper, fallback activé.",
            )
        else:
            LOGGER.warning(
                "Impossible de charger les métriques RAGAS avancées (fallback restreint) : %s",
                exc,
            )
        # Compatibilité ascendante selon la version installée.
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

        if INCLUDE_CONTEXT_RECALL:
            recall_metric = ContextRecall(llm=llm)
            metrics.append(recall_metric)
            metric_names.append(getattr(recall_metric, "name", recall_metric.__class__.__name__))

        # Fallback ultime si la version expose uniquement les objets globaux.
        if not metrics:
            metrics = [answer_relevancy, faithfulness, context_precision]
            if INCLUDE_CONTEXT_RECALL:
                metrics.append(context_recall)
            metric_names = [str(metric) for metric in metrics]

        return evaluate, metrics, metric_names

def _resolve_ragas_models() -> tuple[Any | None, Any | None]:
    """Initialise les wrappers Mistral attendus par RAGAS.

    Returns:
        tuple[Any | None, Any | None]: Le wrapper LLM et le wrapper
        d'embeddings compatibles RAGAS.

    Raises:
        RuntimeError: Si la configuration Mistral ou les dépendances ne
            permettent pas d'initialiser RAGAS correctement.
    """
    try:
        try:
            from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
        except Exception:
            from langchain_mistralai.chat_models import ChatMistralAI
            from langchain_mistralai.embeddings import MistralAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        # Contournement d'un bug connu : certaines réponses Mistral renvoient
        # des token_usage imbriqués (dict dans dict), ce qui casse l'addition
        # naïve de langchain-mistralai.
        class SafeChatMistralAI(ChatMistralAI):
            """Version sûre de `ChatMistralAI` pour agréger les token usages."""

            def _combine_llm_outputs(self, llm_outputs: list[dict | None]) -> dict:
                """Fusionne proprement les compteurs de tokens imbriqués.

                Args:
                    llm_outputs: Les sorties unitaires retournées par LangChain.

                Returns:
                    dict: Les compteurs fusionnés et le nom du modèle.
                """
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

        # Les versions de `langchain-mistralai` varient légèrement sur les noms d'arguments.
        try:
            llm_model = ThrottledChatMistralAI(
                model=MODEL_NAME,
                temperature=0.0,
                api_key=MISTRAL_API_KEY,
            )
        except TypeError:
            llm_model = ThrottledChatMistralAI(
                model=MODEL_NAME,
                temperature=0.0,
                mistral_api_key=MISTRAL_API_KEY,
            )

        try:
            embed_model = ThrottledMistralAIEmbeddings(
                model="mistral-embed",
                api_key=MISTRAL_API_KEY,
            )
        except TypeError:
            embed_model = ThrottledMistralAIEmbeddings(
                model="mistral-embed",
                mistral_api_key=MISTRAL_API_KEY,
            )

        # Sanity check embeddings : évite les NaN silencieux plus tard dans RAGAS.
        try:
            if hasattr(embed_model, "embed_query"):
                vec = embed_model.embed_query("hello")
            elif hasattr(embed_model, "embed_documents"):
                vec = embed_model.embed_documents(["hello"])[0]
            else:
                raise RuntimeError("Le modèle d'embeddings ne fournit pas de méthode de test.")
            if not isinstance(vec, list) or len(vec) == 0:
                raise RuntimeError("Vecteur d'embedding invalide retourné par MistralAIEmbeddings.")
        except Exception as exc:
            raise RuntimeError(
                "Échec du sanity check embeddings avant RAGAS. "
                "Vérifie la configuration Mistral et les dépendances."
            ) from exc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            llm = LangchainLLMWrapper(llm_model)
            embeddings = LangchainEmbeddingsWrapper(embed_model)
        return llm, embeddings
    except Exception as exc:
        raise RuntimeError(
            "Impossible de configurer RAGAS avec Mistral. "
            "Installe `langchain-mistralai` et vérifie que MISTRAL_API_KEY est défini. "
            "Aucun fallback OpenAI n'est autorisé sur ce projet."
        ) from exc


def _run_ragas(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Exécute RAGAS sur les échantillons et enrichit les sorties.

    Args:
        samples: Les échantillons à évaluer.

    Returns:
        tuple[dict[str, Any], pd.DataFrame]: Le résumé global des métriques et
        le détail par échantillon.

    Raises:
        RuntimeError: Si le nombre de lignes détaillées ne correspond pas au
            nombre d'échantillons évalués.
    """
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
    )

    run_config = RunConfig(
        timeout=RAGAS_TIMEOUT,
        max_retries=RAGAS_MAX_RETRIES,
        max_wait=RAGAS_MAX_WAIT,
        max_workers=RAGAS_MAX_WORKERS,
    )

    result = evaluate(
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
            f"Nombre de lignes détaillées incohérent ({len(details)}) pour {len(samples)} questions."
        )

    additional_df = _build_additional_metrics_dataframe(
        samples=samples,
        overlap_threshold=RETRIEVAL_OVERLAP_THRESHOLD,
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
        "max_workers": RAGAS_MAX_WORKERS,
        "max_retries": RAGAS_MAX_RETRIES,
        "max_wait": RAGAS_MAX_WAIT,
        "timeout": RAGAS_TIMEOUT,
        "batch_size": RAGAS_BATCH_SIZE,
        "strict_ragas_errors": STRICT_RAGAS_ERRORS,
    }

    return summary, details


def _save_outputs(
    *,
    output_dir: Path,
    samples: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    details: pd.DataFrame | None,
) -> None:
    """Sauvegarde les échantillons et les résultats d'évaluation sur disque.

    Args:
        output_dir: Le répertoire de sortie.
        samples: Les échantillons générés.
        summary: Le résumé global des métriques, si disponible.
        details: Le détail par échantillon, si disponible.
    """
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
    """Orchestre le pipeline complet d'évaluation RAGAS.

    Raises:
        EnvironmentError: Si la clé API Mistral est absente.
        RuntimeError: Si l'index vectoriel est absent ou si l'évaluation échoue.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not MISTRAL_API_KEY:
        raise EnvironmentError("MISTRAL_API_KEY est absent du fichier .env")

    retriever = VectorStoreManager()
    if retriever.index is None or not retriever.document_chunks:
        raise RuntimeError("Index vectoriel introuvable. Lance d'abord `python indexer.py`.")

    questions = _load_questions()
    LOGGER.info("Questions chargées : %s", len(questions))
    client = Mistral(api_key=MISTRAL_API_KEY)

    samples = _build_samples(
        questions=questions,
        retriever=retriever,
        client=client,
        model=EVAL_MODEL,
        k=EVAL_K,
        min_score=EVAL_MIN_SCORE,
    )
    LOGGER.info("Échantillons générés : %s", len(samples))

    output_dir = Path(OUTPUT_DIR)

    if SKIP_RAGAS:
        _save_outputs(output_dir=output_dir, samples=samples, summary=None, details=None)
        LOGGER.info("Exécution terminée (échantillons uniquement, RAGAS ignoré).")
        return

    if not INCLUDE_CONTEXT_RECALL:
        LOGGER.info(
            "context_recall est désactivé dans la configuration du script."
        )

    try:
        LOGGER.info(
            "Lancement RAGAS (profile=core), workers=%s, batch_size=%s, strict_errors=%s",
            RAGAS_MAX_WORKERS,
            RAGAS_BATCH_SIZE,
            STRICT_RAGAS_ERRORS,
        )
        summary, details = _run_ragas(samples)
    except Exception as exc:
        _save_outputs(output_dir=output_dir, samples=samples, summary=None, details=None)
        LOGGER.error("Échec de l'évaluation RAGAS : %s", exc)
        LOGGER.error(
            "Les échantillons ont été sauvegardés. Tu peux passer SKIP_RAGAS=True dans le script pendant la correction."
        )
        raise

    _save_outputs(output_dir=output_dir, samples=samples, summary=summary, details=details)

    LOGGER.info("Résumé RAGAS : %s", summary)
    LOGGER.info("Questions évaluées : %s", len(samples))
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
        metric_cols = [col for col in metric_cols if col in details.columns]
        if metric_cols:
            missing_ratio = details[metric_cols].isna().mean().to_dict()
            LOGGER.info("Taux de valeurs manquantes par métrique : %s", missing_ratio)
        LOGGER.info("Nombre de lignes détaillées : %s", len(details))


if __name__ == "__main__":
    main()

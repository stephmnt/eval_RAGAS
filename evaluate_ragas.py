"""Script d'évaluation du prototype RAG actuel avec RAGAS.

Ce script génère des échantillons de questions-réponses à partir du Vector Store et les évalue avec RAGAS.
Il est conçu pour être exécuté après l'indexation des données et peut être personnalisé avec des questions spécifiques via un fichier JSON.
Il nécessite une clé API Mistral valide et l'installation de `langchain-mistralai` pour l'intégration avec RAGAS.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from mistralai import Mistral

from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)


DEFAULT_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "q1",
        "category": "simple",
        "question": "Quelle equipe a gagne le match et quel est le score final ?",
        "ground_truth": "La reponse doit identifier le vainqueur et le score final a partir du corpus source.",
    },
    {
        "id": "q2",
        "category": "simple",
        "question": "Quel joueur a marque le plus de points dans le match ?",
        "ground_truth": "La reponse doit nommer le meilleur marqueur et son total de points.",
    },
    {
        "id": "q3",
        "category": "complex",
        "question": "Compare les rebonds a domicile et a l'exterieur, puis donne une conclusion.",
        "ground_truth": "La reponse doit comparer les rebonds domicile/exterieur et formuler une conclusion claire.",
    },
    {
        "id": "q4",
        "category": "complex",
        "question": "Quel joueur a le meilleur pourcentage a 3 points sur les cinq derniers matchs ?",
        "ground_truth": "La reponse doit citer un joueur et une affirmation etayee sur le pourcentage a 3 points sur une fenetre de cinq matchs.",
    },
    {
        "id": "q5",
        "category": "noisy",
        "question": "domicile vs exterieur rebonds diff ??? insight rapide",
        "ground_truth": "La reponse doit interpreter la demande bruitee et comparer les rebonds domicile vs exterieur.",
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
    return parser.parse_args()


def _load_questions(questions_file: str | None) -> list[dict[str, str]]:
    if not questions_file:
        return DEFAULT_QUESTIONS

    data = json.loads(Path(questions_file).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    if not isinstance(data, list):
        raise ValueError("questions-file doit etre une liste ou {'questions': [...]}")

    normalized: list[dict[str, str]] = []
    for idx, row in enumerate(data, start=1):
        normalized.append(
            {
                "id": str(row.get("id", f"q{idx}")),
                "category": str(row.get("category", "non_categorise")),
                "question": str(row["question"]),
                "ground_truth": str(row.get("ground_truth", "")),
            }
        )
    return normalized


def _truncate_context(text: str, max_chars: int = 1200) -> str:
    clean = " ".join(text.split())
    return clean[:max_chars]


def _build_prompt(question: str, contexts: list[str]) -> str:
    formatted_context = "\n\n".join([f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)])
    return (
        "Reponds a la question uniquement avec les informations presentes dans le CONTEXTE. "
        "Si le contexte est insuffisant, indique explicitement l'incertitude.\n\n"
        f"CONTEXTE:\n{formatted_context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "REPONSE FINALE:"
    )


def _generate_answer(
    client: Mistral,
    model: str,
    question: str,
    contexts: list[str],
) -> str:
    if not contexts:
        return "Contexte insuffisant dans le vector store pour repondre a cette question."

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
        return response.choices[0].message.content.strip()
    except Exception as exc:
        LOGGER.exception("Echec de generation pour la question: %s", question)
        return f"Erreur de generation: {exc}"


def _build_samples(
    *,
    questions: list[dict[str, str]],
    retriever: VectorStoreManager,
    client: Mistral,
    model: str,
    k: int,
    min_score: float | None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in questions:
        search_results = retriever.search(row["question"], k=k, min_score=min_score)
        contexts = [_truncate_context(r.get("text", "")) for r in search_results]
        answer = _generate_answer(client, model, row["question"], contexts)

        sample = {
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": row["ground_truth"],
        }
        samples.append(sample)
        LOGGER.info("Echantillon genere %s (%s)", sample["id"], sample["category"])

    return samples


def _resolve_ragas_metrics() -> tuple[Any, list[Any]]:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    return evaluate, [
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
    ]

def _resolve_ragas_models() -> tuple[Any | None, Any | None]:
    try:
        try:
            from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
        except Exception:
            from langchain_mistralai.chat_models import ChatMistralAI
            from langchain_mistralai.embeddings import MistralAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        # Les versions de `langchain-mistralai` varient legerement sur les noms d'arguments.
        try:
            llm_model = ChatMistralAI(
                model=MODEL_NAME,
                temperature=0.0,
                api_key=MISTRAL_API_KEY,
            )
        except TypeError:
            llm_model = ChatMistralAI(
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

        llm = LangchainLLMWrapper(llm_model)
        embeddings = LangchainEmbeddingsWrapper(embed_model)
        return llm, embeddings
    except Exception as exc:
        raise RuntimeError(
            "Impossible de configurer RAGAS avec Mistral. "
            "Installe `langchain-mistralai` et verifie que MISTRAL_API_KEY est defini. "
            "Aucun fallback OpenAI n'est autorise sur ce projet."
        ) from exc


def _run_ragas(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    from datasets import Dataset

    evaluate, metrics = _resolve_ragas_metrics()
    payload = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    }
    dataset = Dataset.from_dict(payload)
    llm, embeddings = _resolve_ragas_models()

    kwargs: dict[str, Any] = {"dataset": dataset, "metrics": metrics}
    if llm is not None:
        kwargs["llm"] = llm
    if embeddings is not None:
        kwargs["embeddings"] = embeddings

    try:
        result = evaluate(**kwargs)
    except TypeError:
        result = evaluate(dataset=dataset, metrics=metrics)

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

    if not details.empty and len(details) == len(samples):
        details.insert(0, "id", [s["id"] for s in samples])
        details.insert(1, "category", [s["category"] for s in samples])

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
    client = Mistral(api_key=MISTRAL_API_KEY)

    samples = _build_samples(
        questions=questions,
        retriever=retriever,
        client=client,
        model=args.model,
        k=args.k,
        min_score=args.min_score,
    )

    output_dir = Path(args.output_dir)

    if args.skip_ragas:
        _save_outputs(output_dir=output_dir, samples=samples, summary=None, details=None)
        LOGGER.info("Execution terminee (echantillons uniquement, RAGAS ignore).")
        return

    try:
        summary, details = _run_ragas(samples)
    except Exception as exc:
        _save_outputs(output_dir=output_dir, samples=samples, summary=None, details=None)
        LOGGER.error("Echec de l'evaluation RAGAS: %s", exc)
        LOGGER.error(
            "Les echantillons ont ete sauvegardes. Tu peux lancer `--skip-ragas` pendant la correction des dependances."
        )
        raise

    _save_outputs(output_dir=output_dir, samples=samples, summary=summary, details=details)

    LOGGER.info("Resume RAGAS: %s", summary)
    if not details.empty:
        LOGGER.info("Nombre de lignes detaillees: %s", len(details))


if __name__ == "__main__":
    main()

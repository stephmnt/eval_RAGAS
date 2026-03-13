"""Script d'évaluation du prototype RAG actuel avec RAGAS.

Ce script génère des échantillons de questions-réponses à partir du Vector Store et les évalue avec RAGAS.
Il est conçu pour être exécuté après l'indexation des données et peut être personnalisé avec des questions spécifiques via un fichier JSON.
Il nécessite une clé API Mistral valide et l'installation de `langchain-mistralai` pour l'intégration avec RAGAS.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
        "question": "Quel est le nom complet de l'equipe codee OKC ?",
        "ground_truth": "OKC correspond a Oklahoma City Thunder.",
    },
    {
        "id": "q2",
        "category": "simple",
        "question": "Selon le tableau des equipes, combien de points totaux a OKC ?",
        "ground_truth": "OKC totalise 9880 points.",
    },
    {
        "id": "q3",
        "category": "complex",
        "question": "Parmi MIA, OKC, LAC et BKN, quelle equipe a le plus de points totaux ?",
        "ground_truth": "Parmi ces quatre equipes, OKC est premier avec 9880 points.",
    },
    {
        "id": "q4",
        "category": "complex",
        "question": "Quelle est la difference de points totaux entre OKC (9880) et MIA (9828) ?",
        "ground_truth": "La difference est de 52 points.",
    },
    {
        "id": "q5",
        "category": "simple",
        "question": "Combien de joueurs compte l'equipe Brooklyn Nets (BKN) ?",
        "ground_truth": "BKN compte 20 joueurs.",
    },
    {
        "id": "q6",
        "category": "simple",
        "question": "Dans le top 15 des joueurs par points, combien de points totaux a Shai Gilgeous-Alexander ?",
        "ground_truth": "Shai Gilgeous-Alexander affiche 2485 points totaux.",
    },
    {
        "id": "q7",
        "category": "simple",
        "question": "Quel est le pourcentage a 3 points (3P%) de Shai Gilgeous-Alexander dans ce tableau ?",
        "ground_truth": "Le 3P% de Shai Gilgeous-Alexander est de 37.5.",
    },
    {
        "id": "q8",
        "category": "complex",
        "question": "Entre Anthony Edwards (2180) et Nikola Jokic (2072), qui a le plus de points totaux ?",
        "ground_truth": "Anthony Edwards a le total le plus eleve avec 2180 points (contre 2072).",
    },
    {
        "id": "q9",
        "category": "complex",
        "question": "Entre Detroit Pistons (10292) et Cleveland Cavaliers (10180), quelle equipe a le plus de points totaux ?",
        "ground_truth": "Detroit Pistons est devant avec 10292 points (contre 10180).",
    },
    {
        "id": "q10",
        "category": "noisy",
        "question": "code MIA -> equipe + points ??? reponse rapide",
        "ground_truth": "MIA correspond a Miami Heat et le total affiche est de 9828 points.",
    },
    {
        "id": "q11",
        "category": "complex",
        "question": "Entre Detroit Pistons (10292) et Cleveland Cavaliers (10180), quelle est la différence de points ?",
        "ground_truth": "La différence est de 112 points.",
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
        action="store_true",
        help=(
            "Active la metrique context_recall. A utiliser uniquement si ground_truth "
            "contient une vraie reponse de reference (et non une consigne)."
        ),
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
    clean = re.sub(r"\bNaN\b", "", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean[:max_chars]


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
        contexts = [c for c in contexts if c and len(c.strip()) > 30]
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


def _resolve_ragas_metrics(
    *,
    llm: Any,
    embeddings: Any,
    include_context_recall: bool,
) -> tuple[Any, list[Any]]:
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
        return evaluate, metrics
    except Exception:
        # Compatibilite ascendante selon la version installee.
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics = [answer_relevancy, faithfulness, context_precision]
        if include_context_recall:
            metrics.append(context_recall)
        return evaluate, metrics

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
) -> tuple[dict[str, Any], pd.DataFrame]:
    from datasets import Dataset

    payload = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    }
    dataset = Dataset.from_dict(payload)
    llm, embeddings = _resolve_ragas_models()
    evaluate, metrics = _resolve_ragas_metrics(
        llm=llm,
        embeddings=embeddings,
        include_context_recall=include_context_recall,
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=True,
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

    if not args.include_context_recall:
        LOGGER.info(
            "context_recall desactive par defaut car ground_truth semble etre une consigne et non une reponse de reference. "
            "Utilise --include-context-recall uniquement avec des labels de verite terrain."
        )

    try:
        summary, details = _run_ragas(
            samples,
            include_context_recall=args.include_context_recall,
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
    if not details.empty:
        metric_cols = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]
        metric_cols = [col for col in metric_cols if col in details.columns]
        if metric_cols:
            missing_ratio = details[metric_cols].isna().mean().to_dict()
            LOGGER.info("Taux de valeurs manquantes par metrique: %s", missing_ratio)
        LOGGER.info("Nombre de lignes detaillees: %s", len(details))


if __name__ == "__main__":
    main()

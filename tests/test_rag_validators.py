import pytest
from pydantic import ValidationError

from evaluate_ragas import (
    EvaluationSample,
    GeneratedAnswerPayload,
    RagasRunOutput,
    RetrievedContext,
    SQLToolResultPayload,
)


def test_retrieved_context_validation_ok() -> None:
    ctx = RetrievedContext.model_validate(
        {
            "text": "OKC totalise 9880 points.",
            "score": "0.91",
            "metadata": {"source": "unit-test"},
        }
    )
    assert ctx.score == pytest.approx(0.91)


def test_sql_result_consistency_rejected() -> None:
    with pytest.raises(ValidationError):
        SQLToolResultPayload.model_validate(
            {"status": "ok", "sql": None, "rows": []}
        )


def test_generated_answer_tokens_consistency() -> None:
    with pytest.raises(ValidationError):
        GeneratedAnswerPayload.model_validate(
            {
                "answer": "Réponse.",
                "input_tokens": 10,
                "output_tokens": 10,
                "total_tokens": 5,
            }
        )


def test_evaluation_sample_validation_ok() -> None:
    sample = EvaluationSample.model_validate(
        {
            "sample_index": 0,
            "id": "q1",
            "category": "simple",
            "question": "Question ?",
            "answer": "Réponse",
            "contexts": ["Contexte 1"],
            "ground_truth": "Référence",
            "retrieval_keywords": ["OKC"],
            "retrieval_queries": ["OKC points"],
            "sql_used": False,
            "sql_query": None,
            "retrieval_latency_s": 0.1,
            "generation_latency_s": 0.2,
            "total_latency_s": 0.3,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
    )
    assert sample.id == "q1"


def test_ragas_run_output_count_check() -> None:
    with pytest.raises(ValidationError):
        RagasRunOutput.model_validate(
            {"sample_count": 5, "summary": {"ok": True}, "details_rows": 4}
        )

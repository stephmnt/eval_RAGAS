from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api
from load_excel_to_db import MatchRow, PlayerRow
from sql_tool import _is_safe_sql
from utils.config import (
    AskResponse,
    EvaluationSample,
    GeneratedAnswerUsage,
    RagasRunOutput,
    RetrievedContext,
    SQLToolResult,
)


class _DummyService:
    def health(self) -> dict[str, object]:
        return {"status": "ok", "issues": []}

    def ask(self, question: str, k: int | None) -> AskResponse:
        assert question
        assert k is None or k >= 1
        return AskResponse.model_validate(
            {
                "question": question,
                "answer": "Réponse test",
                "retrieval_count": 1,
                "contexts": [
                    {
                        "text": "OKC totalise 9880 points.",
                        "score": 95.0,
                        "metadata": {"source": "tests"},
                    }
                ],
                "sql_status": "no_tool",
                "sql_query": None,
                "sql_rows": [],
                "latency_retrieval_s": 0.01,
                "latency_generation_s": 0.02,
                "latency_total_s": 0.03,
            }
        )


class _FailingService:
    def health(self) -> dict[str, object]:
        return {"status": "degraded", "issues": ["boom"]}

    def ask(self, question: str, k: int | None) -> AskResponse:
        raise RuntimeError("Résultat SQL invalide")


def test_api_ask_contract_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "get_rag_service", lambda: _DummyService())

    client = TestClient(api.app)
    response = client.post("/api/v1/ask", json={"question": "Qui est devant OKC ou MIA ?", "k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["question"]
    assert payload["answer"] == "Réponse test"
    assert payload["retrieval_count"] == 1
    assert payload["sql_status"] == "no_tool"
    assert payload["sql_query"] is None
    assert payload["sql_rows"] == []
    assert payload["latency_total_s"] >= 0


def test_api_ask_failure_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "get_rag_service", lambda: _FailingService())

    client = TestClient(api.app)
    response = client.post("/api/v1/ask", json={"question": "Top score ?", "k": 2})

    assert response.status_code == 500
    assert "Résultat SQL invalide" in response.text


def test_safe_sql_accepts_select_queries() -> None:
    assert _is_safe_sql("SELECT player_name FROM players LIMIT 5;")
    assert _is_safe_sql("WITH t AS (SELECT 1) SELECT * FROM t;")


def test_safe_sql_rejects_mutation_queries() -> None:
    assert not _is_safe_sql("DELETE FROM players;")
    assert not _is_safe_sql("UPDATE players SET team_code='OKC';")
    assert not _is_safe_sql("SELECT * FROM players; DROP TABLE players;")


def test_player_row_validation_ok() -> None:
    row = PlayerRow(
        player_name="  Shai Gilgeous-Alexander ",
        team_code="okc",
        age=26,
        points_total=2485,
    )
    assert row.player_name == "Shai Gilgeous-Alexander"
    assert row.team_code == "OKC"


def test_player_row_validation_rejects_invalid_age() -> None:
    with pytest.raises(ValidationError):
        PlayerRow(
            player_name="Player",
            team_code="MIA",
            age=10,
        )


def test_match_row_validation_normalizes_fields() -> None:
    row = MatchRow(team_code=" bos ", team_name=" Boston Celtics ")
    assert row.team_code == "BOS"
    assert row.team_name == "Boston Celtics"


def test_retrieved_context_validation_ok() -> None:
    context = RetrievedContext.model_validate(
        {
            "text": "OKC totalise 9880 points.",
            "score": "0.91",
            "metadata": {"source": "unit-test"},
        }
    )
    assert context.score == pytest.approx(0.91)


def test_sql_result_consistency_rejected() -> None:
    with pytest.raises(ValidationError):
        SQLToolResult.model_validate({"status": "ok", "sql": None, "rows": []})


def test_generated_answer_tokens_consistency() -> None:
    with pytest.raises(ValidationError):
        GeneratedAnswerUsage.model_validate(
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
            {
                "sample_count": 5,
                "summary": {"ok": True},
                "details_rows": 4,
            }
        )

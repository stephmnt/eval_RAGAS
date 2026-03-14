from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api


@dataclass
class _DummyMessage:
    content: str


@dataclass
class _DummyChoice:
    message: _DummyMessage


@dataclass
class _DummyCompletion:
    choices: list[_DummyChoice]


class _DummyChat:
    def complete(self, **_: Any) -> _DummyCompletion:
        return _DummyCompletion(choices=[_DummyChoice(message=_DummyMessage(content="Réponse test"))])


class _DummyClient:
    def __init__(self) -> None:
        self.chat = _DummyChat()


class _DummyRetriever:
    def search(self, question: str, k: int) -> list[dict[str, Any]]:
        assert question
        assert k >= 1
        return [
            {
                "text": "OKC totalise 9880 points.",
                "score": 0.95,
                "metadata": {"source": "tests"},
            }
        ]


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    api._get_mistral_client.cache_clear()
    api._get_vector_store.cache_clear()
    yield
    api._get_mistral_client.cache_clear()
    api._get_vector_store.cache_clear()


def test_api_ask_contract_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_get_mistral_client", lambda: _DummyClient())
    monkeypatch.setattr(api, "_get_vector_store", lambda: _DummyRetriever())
    monkeypatch.setattr(
        api,
        "answer_question_sql_via_langchain",
        lambda _: {"status": "no_tool", "sql": None, "rows": [], "message": "Aucun SQL"},
    )

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


def test_api_rejects_invalid_sql_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_get_mistral_client", lambda: _DummyClient())
    monkeypatch.setattr(api, "_get_vector_store", lambda: _DummyRetriever())
    monkeypatch.setattr(
        api,
        "answer_question_sql_via_langchain",
        lambda _: {"status": "ok", "sql": None, "rows": []},
    )

    client = TestClient(api.app)
    response = client.post("/api/v1/ask", json={"question": "Top score ?", "k": 2})
    assert response.status_code == 500
    assert "Résultat SQL invalide" in response.text

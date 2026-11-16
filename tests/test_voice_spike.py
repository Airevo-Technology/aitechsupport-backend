"""Voice V0 spike endpoint (app/api/v1/endpoints/voice.py).

The RAG/LLM path is faked and the router is mounted on a minimal app, so these tests need
no database, model or API key.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.endpoints import voice
from app.api.v1.api import api_router
from app.core import llm
from app.db.session import get_db

URL = "/api/v1/voice/spike/7/chat/completions"
AUTH = {"Authorization": "Bearer test-voice-secret"}
BODY = {"stream": True, "messages": [
    {"role": "system", "content": "agent prompt"},
    {"role": "user", "content": "What are your opening hours?"},
]}


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(voice.config_store, "get", lambda k, d="": "test-voice-secret" if k == "VOICE_SPIKE_SECRET" else d)
    bot = SimpleNamespace(id=7, organization_id=1)
    monkeypatch.setattr(voice, "_load_bot", lambda db, bid: bot if bid == 7 else None)
    monkeypatch.setattr(voice.presence, "has_available_agent", lambda db, org: False)
    seen = {}

    def fake_stream(db, b, question, online, system_suffix=""):
        seen.update(question=question, suffix=system_suffix)
        yield {"type": "delta", "text": "We open at "}
        yield {"type": "delta", "text": "nine."}
        yield {"type": "final", "text": "We open at nine.", "tokens": 42}

    monkeypatch.setattr(voice.rag, "answer_question_stream", fake_stream)
    return seen


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(voice.router, prefix="/api/v1/voice")
    app.dependency_overrides[get_db] = lambda: iter([None])  # bot lookup is faked above
    return TestClient(app)


def test_router_is_registered_in_the_api():
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    assert "/api/v1/voice/spike/{bot_id}/chat/completions" in app.openapi()["paths"]


def _sse_data(raw: str) -> list[str]:
    return [line[6:] for line in raw.splitlines() if line.startswith("data: ")]


def test_off_without_secret(client, monkeypatch):
    monkeypatch.setattr(voice.config_store, "get", lambda k, d="": d)
    assert client.post(URL, json=BODY, headers=AUTH).status_code == 404


def test_rejects_wrong_or_missing_token(client):
    assert client.post(URL, json=BODY, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post(URL, json=BODY).status_code == 401


def test_unknown_bot_and_empty_question(client):
    assert client.post("/api/v1/voice/spike/999/chat/completions", json=BODY, headers=AUTH).status_code == 404
    only_system = {"stream": True, "messages": [{"role": "system", "content": "x"}]}
    assert client.post(URL, json=only_system, headers=AUTH).status_code == 400


def test_streams_openai_chunks_with_voice_style(client, fakes):
    r = client.post(URL, json=BODY, headers=AUTH)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    data = _sse_data(r.text)
    assert data[-1] == "[DONE]"
    chunks = [json.loads(d) for d in data[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "We open at nine."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert fakes["question"] == "What are your opening hours?"
    assert "phone call" in fakes["suffix"]


def test_non_streaming_returns_completion(client):
    r = client.post(URL, json={**BODY, "stream": False}, headers=AUTH)
    assert r.status_code == 200
    j = r.json()
    assert j["choices"][0]["message"]["content"] == "We open at nine."
    assert j["usage"]["total_tokens"] == 42


def test_list_content_parts_are_flattened(client, fakes):
    body = {"stream": False, "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi there"}]}]}
    assert client.post(URL, json=body, headers=AUTH).status_code == 200
    assert fakes["question"] == "Hi there"


def test_failure_before_any_speech_becomes_spoken_apology(client, monkeypatch):
    def boom(*a, **k):
        raise llm.AllCandidatesFailed([("m", RuntimeError("down"))])
        yield  # pragma: no cover

    monkeypatch.setattr(voice.rag, "answer_question_stream", boom)
    r = client.post(URL, json=BODY, headers=AUTH)
    text = "".join(json.loads(d)["choices"][0]["delta"].get("content", "") for d in _sse_data(r.text)[:-1])
    assert text == voice.FALLBACK_SPEECH

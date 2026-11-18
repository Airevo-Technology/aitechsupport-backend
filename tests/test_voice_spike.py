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

    def fake_stream(db, b, question, online, system_suffix="", first_token_s=None):
        seen.update(question=question, suffix=system_suffix, first_token_s=first_token_s)
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


# ---- repeat requests are answered without the model -------------------------------------

@pytest.mark.parametrize("utterance", [
    "Sorry, can you say that again?", "Can you repeat that please", "Say that again", "Pardon?", "Sorry?",
    "I didn't catch that", "Could you say it one more time", "come again", "What did you say?",
])
def test_repeat_requests_detected(utterance):
    assert voice.is_repeat_request(utterance)


@pytest.mark.parametrize("utterance", [
    "Sorry, I need to change my booking", "What are your opening hours?", "Can you say whether it's open on Sunday?",
    "I want to repeat my order from last week and add a coffee", "Can I speak to someone about my order?", "",
])
def test_normal_questions_not_treated_as_repeat(utterance):
    assert not voice.is_repeat_request(utterance)


def _body(*msgs, stream=False):
    return {"stream": stream, "messages": [{"role": r, "content": c} for r, c in msgs]}


def test_repeat_replays_previous_reply_without_calling_the_model(client, fakes):
    r = client.post(URL, headers=AUTH, json=_body(
        ("user", "What are your opening hours?"), ("assistant", "We open at nine."), ("user", "Sorry, can you say that again?")))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Sure. We open at nine."
    assert r.json()["usage"]["total_tokens"] == 0
    assert "question" not in fakes  # the RAG/LLM path never ran


def test_repeat_with_no_history_does_not_invent_anything(client, fakes):
    r = client.post(URL, headers=AUTH, json=_body(("user", "Sorry, can you say that again?")))
    assert r.json()["choices"][0]["message"]["content"] == voice._NO_HISTORY_REPLY
    assert "question" not in fakes


def test_repeat_streams_too(client):
    r = client.post(URL, headers=AUTH, json=_body(("assistant", "We open at nine."), ("user", "say that again"), stream=True))
    chunks = [json.loads(d) for d in _sse_data(r.text) if d != "[DONE]"]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Sure. We open at nine."


def test_sorry_prefix_on_a_real_question_still_goes_to_the_model(client, fakes):
    client.post(URL, headers=AUTH, json=_body(("user", "Sorry, I need to change my booking")))
    assert fakes["question"] == "Sorry, I need to change my booking"


# ---- voice-only first-token budget & prompt guard ----------------------------------------

def test_voice_budget_defaults_to_one_second_and_is_passed_to_rag(client, fakes):
    client.post(URL, headers=AUTH, json=BODY)
    assert fakes["first_token_s"] == 1.0


@pytest.mark.parametrize("raw,want", [("", 1.0), ("1500", 1.5), ("0", 0.0), ("abc", 1.0), ("-5", 0.0)])
def test_voice_budget_setting(monkeypatch, raw, want):
    monkeypatch.setattr(voice.config_store, "get", lambda k, d="": raw if k == "VOICE_FIRST_TOKEN_TIMEOUT_MS" else d)
    assert voice._voice_first_token_s() == want


def test_voice_style_forbids_inventing_earlier_speech():
    assert "Never claim you said" in voice.VOICE_STYLE

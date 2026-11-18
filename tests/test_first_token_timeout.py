"""First-token deadline in the streaming fallback chain (llm.answer_stream_with_fallback)."""
import time

import pytest

from app.core import llm


def _fake_stream(delay_first: float, pieces=("hello", " world"), gap: float = 0.0):
    def gen(*_a, **_k):
        time.sleep(delay_first)
        for p in pieces:
            yield {"type": "delta", "text": p}
            time.sleep(gap)
        yield {"type": "final", "text": "".join(pieces), "tokens": 7}
    return gen


def _run(monkeypatch, plans, budget_ms):
    """plans: {model: stream-fn}. Returns (events, models_started)."""
    started: list[str] = []

    def fake_answer_stream(system_prompt, context, question, model, image_data_url=None, base_url=None):
        started.append(model)
        return plans[model]()

    monkeypatch.setattr(llm, "answer_stream", fake_answer_stream)
    monkeypatch.setattr(llm, "first_token_budget", lambda: (budget_ms / 1000.0) if budget_ms else None)
    cands = [(m, None) for m in plans]
    return list(llm.answer_stream_with_fallback("s", "c", "q", cands)), started


def text_of(events):
    return "".join(e["text"] for e in events if e["type"] == "delta")


def test_slow_first_candidate_fails_over(monkeypatch):
    t0 = time.monotonic()
    events, started = _run(monkeypatch, {"slow": lambda: _fake_stream(1.5)(), "fast": lambda: _fake_stream(0.0, ("quick",))()}, budget_ms=150)
    assert started == ["slow", "fast"]
    assert text_of(events) == "quick"
    assert time.monotonic() - t0 < 1.0  # did not wait out the slow model


def test_fast_first_candidate_is_used(monkeypatch):
    events, started = _run(monkeypatch, {"a": lambda: _fake_stream(0.0)(), "b": lambda: _fake_stream(0.0, ("nope",))()}, budget_ms=300)
    assert started == ["a"]
    assert text_of(events) == "hello world"


def test_last_candidate_is_never_cut_off(monkeypatch):
    events, started = _run(monkeypatch, {"a": lambda: _fake_stream(1.0, ("slow-a",))(), "b": lambda: _fake_stream(0.4, ("slow-b",))()}, budget_ms=100)
    assert started == ["a", "b"]
    assert text_of(events) == "slow-b"  # b took 0.4s > budget but is last, so it completes


def test_no_deadline_after_first_token(monkeypatch):
    events, started = _run(monkeypatch, {"a": lambda: _fake_stream(0.0, ("x", "y", "z"), gap=0.2)(), "b": lambda: _fake_stream(0.0, ("nope",))()}, budget_ms=100)
    assert started == ["a"]
    assert text_of(events) == "xyz"


def test_budget_off_by_default(monkeypatch):
    events, started = _run(monkeypatch, {"a": lambda: _fake_stream(0.4, ("late",))(), "b": lambda: _fake_stream(0.0, ("nope",))()}, budget_ms=0)
    assert started == ["a"]
    assert text_of(events) == "late"


def test_budget_parsing(monkeypatch):
    for raw, want in [("", None), ("0", None), ("abc", None), ("1500", 1.5)]:
        monkeypatch.setattr(llm.config_store, "get", lambda k, d="", _r=raw: _r)
        assert llm.first_token_budget() == want


def test_all_slow_raises_nothing_extra_when_last_succeeds(monkeypatch):
    with pytest.raises(llm.AllCandidatesFailed):
        def boom(*_a, **_k):
            raise RuntimeError("down")
            yield  # pragma: no cover
        _run(monkeypatch, {"a": lambda: boom(), "b": lambda: boom()}, budget_ms=100)


def test_per_call_budget_overrides_platform_setting(monkeypatch):
    plans = {"slow": lambda: _fake_stream(1.0)(), "fast": lambda: _fake_stream(0.0, ("quick",))()}
    monkeypatch.setattr(llm, "answer_stream", lambda s, c, q, model, image_data_url=None, base_url=None: plans[model]())
    monkeypatch.setattr(llm, "first_token_budget", lambda: None)  # platform: off
    t0 = time.monotonic()
    events = list(llm.answer_stream_with_fallback("s", "c", "q", [("slow", None), ("fast", None)], first_token_s=0.1))
    assert text_of(events) == "quick" and time.monotonic() - t0 < 0.8


def test_per_call_zero_disables_platform_budget(monkeypatch):
    plans = {"a": lambda: _fake_stream(0.4, ("late",))(), "b": lambda: _fake_stream(0.0, ("nope",))()}
    monkeypatch.setattr(llm, "answer_stream", lambda s, c, q, model, image_data_url=None, base_url=None: plans[model]())
    monkeypatch.setattr(llm, "first_token_budget", lambda: 0.1)  # platform: on
    events = list(llm.answer_stream_with_fallback("s", "c", "q", [("a", None), ("b", None)], first_token_s=0))
    assert text_of(events) == "late"

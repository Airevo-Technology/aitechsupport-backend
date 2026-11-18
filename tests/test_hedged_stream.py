"""Hedged streaming: race the next model when the first is silent (llm.answer_stream_with_fallback)."""
import threading
import time

import pytest

from app.core import llm


def stream(first_delay, pieces=("hello", " world"), gap=0.0, fail_after_first=False, blank=False):
    def gen():
        time.sleep(first_delay)
        if blank:
            yield {"type": "final", "text": "", "tokens": 0}
            return
        for k, p in enumerate(pieces):
            yield {"type": "delta", "text": p}
            if fail_after_first and k == 0:
                raise RuntimeError("provider died mid-stream")
            time.sleep(gap)
        yield {"type": "final", "text": "".join(pieces), "tokens": 5}
    return gen


def boom(delay=0.0):
    def gen():
        time.sleep(delay)
        raise RuntimeError("provider down")
        yield  # pragma: no cover
    return gen


def run(monkeypatch, plans, hedge, **kw):
    started, lock = [], threading.Lock()

    def fake(system_prompt, context, question, model, image_data_url=None, base_url=None):
        with lock:
            started.append(model)
        return plans[model]()

    monkeypatch.setattr(llm, "answer_stream", fake)
    monkeypatch.setattr(llm, "hedge_delay", lambda: None)
    monkeypatch.setattr(llm, "first_token_budget", lambda: None)
    cands = [(m, None) for m in plans]
    t0 = time.monotonic()
    events = list(llm.answer_stream_with_fallback("s", "c", "q", cands, hedge_s=hedge, **kw))
    return events, started, time.monotonic() - t0


def text(events):
    return "".join(e["text"] for e in events if e["type"] == "delta")


def test_slow_primary_loses_the_race(monkeypatch):
    events, started, took = run(monkeypatch, {"slow": stream(1.5, ("slow",)), "fast": stream(0.0, ("fast",))}, hedge=0.15)
    assert started == ["slow", "fast"] and text(events) == "fast"
    assert took < 0.8
    assert events[-1]["type"] == "final" and events[-1]["text"] == "fast"


def test_fast_primary_never_hedges(monkeypatch):
    events, started, _ = run(monkeypatch, {"a": stream(0.0), "b": stream(0.0, ("nope",))}, hedge=0.3)
    assert started == ["a"] and text(events) == "hello world"


def test_first_to_speak_wins_even_if_it_is_the_original(monkeypatch):
    events, started, _ = run(monkeypatch, {"a": stream(0.4, ("a-wins",)), "b": stream(1.5, ("b-late",))}, hedge=0.1)
    assert started == ["a", "b"] and text(events) == "a-wins"


def test_error_starts_next_immediately_not_after_the_timer(monkeypatch):
    events, started, took = run(monkeypatch, {"bad": boom(), "good": stream(0.0, ("ok",))}, hedge=5.0)
    assert started == ["bad", "good"] and text(events) == "ok"
    assert took < 1.0  # did not wait for the 5 s hedge timer


def test_empty_answer_counts_as_failure(monkeypatch):
    events, started, _ = run(monkeypatch, {"blank": stream(0.0, blank=True), "good": stream(0.0, ("ok",))}, hedge=5.0)
    assert started == ["blank", "good"] and text(events) == "ok"


def test_all_fail_raises(monkeypatch):
    with pytest.raises(llm.AllCandidatesFailed) as ei:
        run(monkeypatch, {"a": boom(), "b": boom()}, hedge=0.05)
    assert [m for m, _ in ei.value.errors] == ["a", "b"]


def test_at_most_two_in_flight(monkeypatch):
    events, started, _ = run(monkeypatch, {"a": stream(0.6, ("a",)), "b": stream(0.8, ("b",)), "c": stream(0.0, ("c",))}, hedge=0.05)
    # a and b are both in flight and silent, so c may not start; a speaks first at 0.6 s.
    assert started == ["a", "b"] and text(events) == "a"


def test_loser_output_never_leaks(monkeypatch):
    events, _, _ = run(monkeypatch, {"slow": stream(0.3, ("LOSER-1", "LOSER-2"), gap=0.05), "fast": stream(0.0, ("w1", "w2", "w3"), gap=0.2)}, hedge=0.1)
    out = text(events)
    assert out == "w1w2w3" and "LOSER" not in out


def test_failure_after_winner_is_reraised_not_restarted(monkeypatch):
    with pytest.raises(RuntimeError, match="mid-stream"):
        run(monkeypatch, {"a": stream(0.0, ("x", "y"), fail_after_first=True), "b": stream(0.0, ("nope",))}, hedge=0.3)


def test_single_candidate_and_zero_hedge_use_sequential_path(monkeypatch):
    events, started, _ = run(monkeypatch, {"only": stream(0.0)}, hedge=0.1)
    assert started == ["only"] and text(events) == "hello world"
    events, started, _ = run(monkeypatch, {"a": stream(0.3, ("late",)), "b": stream(0.0, ("x",))}, hedge=0)
    assert started == ["a"] and text(events) == "late"  # hedge off -> sequential, no race


def test_platform_setting_parsing(monkeypatch):
    for raw, want in [("", None), ("0", None), ("zzz", None), ("400", 0.4)]:
        monkeypatch.setattr(llm.config_store, "get", lambda k, d="", _r=raw: _r)
        assert llm.hedge_delay() == want

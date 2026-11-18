"""Conversation history: message layout and follow-up retrieval query."""
from app.core import llm, rag


def test_history_sits_between_system_and_question():
    msgs = llm._messages("sys", "ctx", "What about Sundays?", history=[
        {"role": "user", "content": "Opening hours?"},
        {"role": "assistant", "content": "Nine to five."},
    ])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"].endswith("QUESTION: What about Sundays?")


def test_no_history_keeps_old_layout():
    assert [m["role"] for m in llm._messages("sys", "ctx", "q")] == ["system", "user"]


def test_short_followup_borrows_previous_question():
    hist = [{"role": "user", "content": "What are your opening hours?"},
            {"role": "assistant", "content": "Nine to five."}]
    assert rag.followup_query("What about Sundays?", hist) == "What are your opening hours? What about Sundays?"


def test_long_or_first_questions_are_untouched():
    hist = [{"role": "user", "content": "Opening hours?"}]
    long_q = "Can you tell me how much the premium plan costs per month please"
    assert rag.followup_query(long_q, hist) == long_q
    assert rag.followup_query("What about Sundays?", None) == "What about Sundays?"
    assert rag.followup_query("What about Sundays?", []) == "What about Sundays?"

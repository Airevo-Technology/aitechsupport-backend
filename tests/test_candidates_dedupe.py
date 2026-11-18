from types import SimpleNamespace

from app.core import models_catalog as mc


def _pkg(*models):
    return SimpleNamespace(fallback_chain=[{"provider": "openrouter", "model": m} for m in models])


def test_bot_model_equal_to_first_tier_is_not_repeated(monkeypatch):
    monkeypatch.setattr(mc.config_store, "get", lambda k, d="": d)
    bot = SimpleNamespace(chat_model="mistral/x")
    assert mc.resolve_candidates(bot, _pkg("mistral/x", "qwen/y")) == [
        ("mistral/x", None), ("qwen/y", mc._REAL_OPENROUTER_BASE_URL)]


def test_same_model_on_a_different_endpoint_is_kept(monkeypatch):
    monkeypatch.setattr(mc.config_store, "get", lambda k, d="": d)
    pkg = SimpleNamespace(fallback_chain=[{"provider": "self_hosted", "base_url": "http://10.0.0.1:11434", "model": "m"}])
    assert len(mc.resolve_candidates(SimpleNamespace(chat_model="m"), pkg)) == 2


def test_no_bot_model_leaves_chain_untouched(monkeypatch):
    monkeypatch.setattr(mc.config_store, "get", lambda k, d="": d)
    assert [m for m, _ in mc.resolve_candidates(SimpleNamespace(chat_model=None), _pkg("a", "b", "c"))] == ["a", "b", "c"]

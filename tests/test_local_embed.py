"""embed_query: opt-in local path, and it must never break retrieval when it fails."""
from app.core import embeddings, local_embed


def _provider_stub(monkeypatch):
    monkeypatch.setattr(embeddings, "_provider", lambda: "openrouter")
    monkeypatch.setattr(embeddings, "_openrouter_embed", lambda texts: [[0.5] * 1024 for _ in texts])


def test_disabled_uses_provider(monkeypatch):
    _provider_stub(monkeypatch)
    monkeypatch.setattr(local_embed, "enabled", lambda: False)
    monkeypatch.setattr(local_embed, "embed_query", lambda t: (_ for _ in ()).throw(AssertionError("must not run")))
    assert embeddings.embed_query("hi")[0] == 0.5


def test_enabled_uses_local(monkeypatch):
    _provider_stub(monkeypatch)
    monkeypatch.setattr(local_embed, "enabled", lambda: True)
    monkeypatch.setattr(local_embed, "embed_query", lambda t: [0.25] * 1024)
    assert embeddings.embed_query("hi")[0] == 0.25


def test_local_failure_falls_back_to_provider(monkeypatch):
    _provider_stub(monkeypatch)
    monkeypatch.setattr(local_embed, "enabled", lambda: True)

    def boom(_t):
        raise RuntimeError("model corrupt")

    monkeypatch.setattr(local_embed, "embed_query", boom)
    assert embeddings.embed_query("hi")[0] == 0.5


def test_enabled_requires_setting_and_files(monkeypatch, tmp_path):
    monkeypatch.setattr(local_embed.config_store, "get", lambda k, d="": {"QUERY_EMBEDDINGS_LOCAL": "true", "LOCAL_EMBED_MODEL_DIR": str(tmp_path)}.get(k, d))
    assert local_embed.enabled() is False  # setting on, files missing
    (tmp_path / local_embed.MODEL_FILE).write_bytes(b"x")
    (tmp_path / local_embed.TOKENIZER_FILE).write_bytes(b"x")
    assert local_embed.enabled() is True
    monkeypatch.setattr(local_embed.config_store, "get", lambda k, d="": {"QUERY_EMBEDDINGS_LOCAL": "false", "LOCAL_EMBED_MODEL_DIR": str(tmp_path)}.get(k, d))
    assert local_embed.enabled() is False  # files present, setting off

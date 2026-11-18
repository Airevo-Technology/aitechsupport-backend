"""Pluggable text embeddings.

- provider "ollama":     self-hosted Ollama on the ai-server (e.g. bge-m3, 1024-dim,
                         multilingual). No API key; reached at settings.OLLAMA_BASE_URL.
- provider "voyage":     Voyage AI (external). Requires VOYAGE_API_KEY. Free-tier
                         rate limits (3 RPM / 10K TPM) make this unsuitable alone
                         for a commercial workload — see "openrouter" below.
- provider "openrouter": OpenRouter's embeddings endpoint. Requires OPENROUTER_API_KEY
                         plus a main model (EMBEDDING_MODEL_OPENROUTER_MAIN) and up
                         to two fallback models, tried in order until one succeeds.
- provider "huggingface": Hugging Face Inference API feature-extraction. Requires
                         HUGGINGFACE_API_KEY and a sentence-embedding model id
                         (EMBEDDING_MODEL_HUGGINGFACE, default BAAI/bge-m3).
- provider "fake":       deterministic, dependency-free hashing embedder for offline
                         dev/CI. It has NO real semantics (bag-of-words hashing) — it
                         only makes the ingest/retrieve pipeline runnable without a key.
                         Never use in production.

The active provider is admin-configurable at runtime (config_store key
EMBEDDINGS_PROVIDER), falling back to settings.EMBEDDINGS_PROVIDER (.env) when unset.

All providers return L2-normalized vectors of length settings.EMBEDDING_DIM, so
cosine distance in pgvector behaves consistently.
"""
from __future__ import annotations

import hashlib
import logging
import math

from app.core.settings import settings
from app.core import config_store, local_embed

logger = logging.getLogger(__name__)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def _fake_embed(text: str, dim: int) -> list[float]:
    """Deterministic bag-of-words hash into `dim` buckets. Dev/CI only."""
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    return _l2_normalize(vec)


def _ollama_base_url() -> str:
    # admin DB setting first, then env/settings default
    return (config_store.get("OLLAMA_BASE_URL") or settings.OLLAMA_BASE_URL).rstrip("/")


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    """Embed via the ai-server's Ollama /api/embed (batch). Model = EMBEDDING_MODEL."""
    import httpx  # lazy import so the app boots without the SDK configured

    resp = httpx.post(
        f"{_ollama_base_url()}/api/embed",
        json={"model": settings.EMBEDDING_MODEL, "input": texts},
        # Fast connect failure if the ai-server is unreachable; generous read
        # budget for a cold/CPU-offloaded embed model.
        timeout=httpx.Timeout(120.0, connect=5.0),
    )
    resp.raise_for_status()
    vectors = resp.json().get("embeddings")
    if not vectors or len(vectors) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(vectors) if vectors else 0} embeddings for "
            f"{len(texts)} inputs (model={settings.EMBEDDING_MODEL})."
        )
    dim = settings.EMBEDDING_DIM
    for v in vectors:
        if len(v) != dim:
            raise RuntimeError(
                f"Embedding dim mismatch: model '{settings.EMBEDDING_MODEL}' returned "
                f"{len(v)} but the schema expects {dim}. Pick a {dim}-dim model or migrate."
            )
    # Ollama does not guarantee normalized vectors; normalize for stable cosine search.
    return [_l2_normalize(v) for v in vectors]


def _voyage_key() -> str:
    # admin DB setting first, then env
    return config_store.get("VOYAGE_API_KEY")


def _voyage_embed(texts: list[str], input_type: str) -> list[list[float]]:
    import voyageai  # lazy import so the app boots without the SDK configured

    key = _voyage_key()
    if not key:
        raise RuntimeError("VOYAGE_API_KEY is not set (EMBEDDINGS_PROVIDER=voyage).")
    client = voyageai.Client(api_key=key)
    result = client.embed(texts, model=settings.EMBEDDING_MODEL, input_type=input_type)
    return result.embeddings


def _huggingface_key() -> str:
    return config_store.get("HUGGINGFACE_API_KEY")


def _huggingface_model() -> str:
    return config_store.get("EMBEDDING_MODEL_HUGGINGFACE") or "BAAI/bge-m3"


def _huggingface_embed(texts: list[str]) -> list[list[float]]:
    """Embed via the HF Inference API's feature-extraction pipeline.

    Sentence-embedding models (bge, gte, e5, ...) return one pooled vector per
    input already, so no manual pooling is needed here.
    """
    import httpx  # lazy import so the app boots without the SDK configured

    key = _huggingface_key()
    if not key:
        raise RuntimeError("HUGGINGFACE_API_KEY is not set (EMBEDDINGS_PROVIDER=huggingface).")
    model = _huggingface_model()
    resp = httpx.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers={"Authorization": f"Bearer {key}"},
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=httpx.Timeout(60.0, connect=5.0),
    )
    resp.raise_for_status()
    vectors = resp.json()
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise RuntimeError(
            f"Hugging Face returned {len(vectors) if isinstance(vectors, list) else 0} "
            f"embeddings for {len(texts)} inputs (model={model})."
        )
    dim = settings.EMBEDDING_DIM
    for v in vectors:
        if len(v) != dim:
            raise RuntimeError(
                f"Embedding dim mismatch: model '{model}' returned {len(v)} but the "
                f"schema expects {dim}. Pick a {dim}-dim model or migrate."
            )
    return [_l2_normalize(v) for v in vectors]


def _openrouter_key() -> str:
    return config_store.get("OPENROUTER_API_KEY")


def _openrouter_models() -> list[str]:
    """Main model then up to two fallbacks, in order; empty/unset entries dropped."""
    raw = [
        config_store.get("EMBEDDING_MODEL_OPENROUTER_MAIN"),
        config_store.get("EMBEDDING_MODEL_OPENROUTER_FALLBACK_1"),
        config_store.get("EMBEDDING_MODEL_OPENROUTER_FALLBACK_2"),
    ]
    return [m.strip() for m in raw if m and m.strip()]


def _openrouter_client():
    from openai import OpenAI  # lazy import so the app boots without the SDK configured

    key = _openrouter_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (EMBEDDINGS_PROVIDER=openrouter).")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=60.0)


def _openrouter_embed(texts: list[str]) -> list[list[float]]:
    """Embed via OpenRouter, trying the main model then each fallback in order."""
    models = _openrouter_models()
    if not models:
        raise RuntimeError(
            "No OpenRouter embedding model is configured (EMBEDDING_MODEL_OPENROUTER_MAIN)."
        )
    client = _openrouter_client()
    dim = settings.EMBEDDING_DIM
    errors: list[tuple[str, Exception]] = []
    for model in models:
        try:
            resp = client.embeddings.create(model=model, input=texts)
            vectors = [d.embedding for d in resp.data]
            # Reject a dimension mismatch here rather than letting pgvector
            # refuse the INSERT further downstream — the DB error surfaces as a
            # generic "ingest failed" with no hint about which model is wrong.
            for v in vectors:
                if len(v) != dim:
                    raise RuntimeError(
                        f"model '{model}' returned {len(v)}-dim vectors but the schema "
                        f"expects {dim}; pick a {dim}-dim model or migrate the column"
                    )
            return [_l2_normalize(v) for v in vectors]
        except Exception as e:  # noqa: BLE001 — try the next candidate
            errors.append((model, e))
    detail = "; ".join(f"{m}: {e}" for m, e in errors)
    raise RuntimeError(f"all OpenRouter embedding candidate(s) failed: {detail}")


def _provider() -> str:
    """Admin DB override first (EMBEDDINGS_PROVIDER), then env/settings default."""
    return config_store.get("EMBEDDINGS_PROVIDER") or settings.EMBEDDINGS_PROVIDER


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed knowledge-base chunks for storage.

    Batched to stay under the provider's per-request text/token limits — a large
    source can produce thousands of chunks, which a single call rejects.
    """
    if not texts:
        return []
    provider = _provider()
    if provider == "fake":
        return [_fake_embed(t, settings.EMBEDDING_DIM) for t in texts]
    batch = max(1, settings.EMBED_BATCH_SIZE)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        if provider == "ollama":
            out.extend(_ollama_embed(chunk))
        elif provider == "openrouter":
            out.extend(_openrouter_embed(chunk))
        elif provider == "huggingface":
            out.extend(_huggingface_embed(chunk))
        else:
            out.extend(_voyage_embed(chunk, "document"))
    return out


def embed_query(text: str) -> list[float]:
    """Embed a user question for retrieval."""
    provider = _provider()
    if provider == "fake":
        return _fake_embed(text, settings.EMBEDDING_DIM)
    if local_embed.enabled():
        try:
            return local_embed.embed_query(text)
        except Exception:  # noqa: BLE001 — retrieval must keep working; use the provider below
            logger.warning("local query embedding failed; falling back to %s", provider, exc_info=True)
    if provider == "ollama":
        return _ollama_embed([text])[0]
    if provider == "openrouter":
        return _openrouter_embed([text])[0]
    if provider == "huggingface":
        return _huggingface_embed([text])[0]
    return _voyage_embed([text], "query")[0]


def is_configured() -> bool:
    """True when embeddings can actually run (fake/ollama need no key)."""
    provider = _provider()
    if provider in ("fake", "ollama"):
        return True
    if provider == "openrouter":
        return bool(_openrouter_key()) and bool(_openrouter_models())
    if provider == "huggingface":
        return bool(_huggingface_key())
    return bool(_voyage_key())

"""In-process query embeddings: BAAI/bge-m3 (int8 ONNX) on the app server's CPU.

Why: an OpenRouter embedding round trip costs ~800 ms per question, which alone
blows a voice-call latency budget. The same model run locally answers in tens of
milliseconds. Vectors are 1024-dim and match OpenRouter's baai/bge-m3 closely
(cosine ~0.99, int8 quantisation noise), so chunks already stored stay valid.

Scope: QUERIES only. Document ingestion keeps using the configured provider so a
large upload can't monopolise a small shared CPU. Opt-in via the admin/env setting
QUERY_EMBEDDINGS_LOCAL=true, and only correct when the stored chunks were embedded
with bge-m3. Any failure here is reported to the caller, which falls back to the
provider path (see embeddings.embed_query).

Model files (see scripts/fetch_local_embed_model.py) live in LOCAL_EMBED_MODEL_DIR
(default ~/models/bge-m3): model_quantized.onnx + tokenizer.json. ~1.2 GB RAM.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from app.core import config_store

logger = logging.getLogger(__name__)

MODEL_FILE = "model_quantized.onnx"
TOKENIZER_FILE = "tokenizer.json"
MAX_TOKENS = 512
DIM = 1024

_lock = threading.Lock()
_runtime: tuple | None = None  # (onnx session, tokenizer)


def model_dir() -> Path:
    return Path(os.path.expanduser(config_store.get("LOCAL_EMBED_MODEL_DIR") or "~/models/bge-m3"))


def enabled() -> bool:
    """Setting is on AND the model files are present (so a missing download never breaks chat)."""
    if config_store.get("QUERY_EMBEDDINGS_LOCAL").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    d = model_dir()
    return (d / MODEL_FILE).is_file() and (d / TOKENIZER_FILE).is_file()


def _load() -> tuple:
    global _runtime
    with _lock:
        if _runtime is None:
            import onnxruntime as ort  # heavy imports stay lazy so the app boots without them
            from tokenizers import Tokenizer

            d = model_dir()
            tok = Tokenizer.from_file(str(d / TOKENIZER_FILE))
            tok.enable_truncation(MAX_TOKENS)
            tok.enable_padding()
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            sess = ort.InferenceSession(str(d / MODEL_FILE), opts, providers=["CPUExecutionProvider"])
            _runtime = (sess, tok)
            logger.info("local embeddings: loaded %s", d / MODEL_FILE)
        return _runtime


def embed_query(text: str) -> list[float]:
    """Return the L2-normalised bge-m3 (CLS-pooled) embedding of `text`. Raises on any failure."""
    import numpy as np

    sess, tok = _load()
    enc = tok.encode_batch([text])
    names = {i.name for i in sess.get_inputs()}
    feed = {
        "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
    }
    out = sess.run(None, {k: v for k, v in feed.items() if k in names})[0]
    vec = out[:, 0] if out.ndim == 3 else out  # bge-m3 pools the [CLS] token
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    result = vec[0].tolist()
    if len(result) != DIM:
        raise RuntimeError(f"local embedding has {len(result)} dims, expected {DIM}")
    return result


def preload() -> None:
    """Load the model off the request path (startup thread) so the first question isn't slow."""
    if not enabled():
        return
    try:
        _load()
        embed_query("warm up")
    except Exception:  # noqa: BLE001 — never block boot; embed_query falls back per call
        logger.exception("local embeddings: preload failed")


def preload_in_background() -> None:
    threading.Thread(target=preload, name="local-embed-preload", daemon=True).start()

"""Voice V0 spike: an OpenAI-compatible "custom LLM" endpoint for a managed voice agent.

A managed voice platform (ElevenLabs Agents) handles the phone call, speech-to-text,
turn-taking and text-to-speech, and calls this endpoint for the reply text. We answer
with the same RAG + model-fallback path the website widget uses, so the answer comes
only from the bot's own knowledge base.

This is a spike, deliberately small and off by default:
- Dormant (404) until the platform-admin/env setting VOICE_SPIKE_SECRET is set.
- Auth is one shared bearer secret; the bot is chosen in the URL. Production (phase V1)
  replaces this with a per-voice-line secret and org-scoped lookup.
- No billing/quota metering yet; token counts are only logged.
Point the voice agent's custom-LLM server URL at
`{API_BASE_URL}/api/v1/voice/spike/{bot_id}` and put the secret in its API-key field.
"""
from __future__ import annotations

import hmac
import json
import logging
import time
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.core import config_store, limits, llm, presence, rag
from app.core.settings import settings
from app.db.session import SessionLocal, get_db
from app.models.bot import Bot

logger = logging.getLogger(__name__)
router = APIRouter()

# Appended to the bot's own system prompt: the reply is spoken aloud, not read.
VOICE_STYLE = (
    "\n\nYou are speaking on a live phone call. Reply in one to three short, natural spoken sentences. "
    "Never use lists, markdown, links or emoji. Say numbers, dates and times the way a person would say them. "
    "If you need a detail from the caller, ask for one thing at a time."
)
# Spoken when nothing could be generated, so the caller never hears dead air.
FALLBACK_SPEECH = "Sorry, I'm having a little trouble right now. Let me get someone to help you."


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | list | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    messages: list[_Message]
    stream: bool = False


def _text(content: str | list | None) -> str:
    """Message content is a string or a list of OpenAI content parts; keep the text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _require_secret(authorization: str | None) -> None:
    secret = config_store.get("VOICE_SPIKE_SECRET")
    if not secret:
        raise HTTPException(status_code=404, detail="Not found")  # feature off
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), secret.encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")


def _load_bot(db: Session, bot_id: int) -> Bot | None:
    return db.query(Bot).filter(Bot.id == bot_id).first()


def _reply_events(bot_id: int, question: str) -> Iterator[dict]:
    """Yield rag 'delta' events then a 'final' event, on a fresh DB session while holding
    an inference slot (the request session closes before a streaming body finishes).
    Any failure becomes a spoken apology instead of a broken stream."""
    token = limits.acquire_slot(settings.WIDGET_MAX_CONCURRENCY)
    if token is None:
        yield {"type": "delta", "text": FALLBACK_SPEECH}
        yield {"type": "final", "text": FALLBACK_SPEECH, "tokens": 0}
        return
    db = SessionLocal()
    spoke = False
    t0 = time.monotonic()
    try:
        bot = _load_bot(db, bot_id)
        if bot is None:
            raise LookupError(f"bot {bot_id} not found")
        online = presence.has_available_agent(db, bot.organization_id)
        for ev in rag.answer_question_stream(db, bot, question, online, system_suffix=VOICE_STYLE):
            if ev["type"] == "delta":
                if not spoke:
                    logger.info("voice spike bot=%s first_token_ms=%d", bot_id, (time.monotonic() - t0) * 1000)
                spoke = True
            elif ev["type"] == "final":
                logger.info("voice spike bot=%s total_ms=%d tokens=%s", bot_id,
                            (time.monotonic() - t0) * 1000, ev.get("tokens"))
            yield ev
    except (llm.AllCandidatesFailed, LookupError, Exception) as e:  # noqa: BLE001
        logger.warning("voice spike bot=%s failed (spoke=%s): %s", bot_id, spoke, e)
        if not spoke:
            yield {"type": "delta", "text": FALLBACK_SPEECH}
            yield {"type": "final", "text": FALLBACK_SPEECH, "tokens": 0}
    finally:
        db.close()
        limits.release_slot(token)


def _chunk(cid: str, delta: dict, finish: str | None = None) -> str:
    body = {
        "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
        "model": "aichatsupport-rag",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


@router.post("/spike/{bot_id}/chat/completions")
def chat_completions(
    bot_id: int,
    payload: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    _require_secret(authorization)
    if _load_bot(db, bot_id) is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    question = next((_text(m.content).strip() for m in reversed(payload.messages) if m.role == "user"), "")
    if not question:
        raise HTTPException(status_code=400, detail="No user message")
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if not payload.stream:
        text, tokens = "", 0
        for ev in _reply_events(bot_id, question):
            if ev["type"] == "final":
                text, tokens = ev["text"], ev.get("tokens", 0)
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": int(time.time()), "model": "aichatsupport-rag",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"total_tokens": tokens},
        })

    def stream() -> Iterator[str]:
        yield _chunk(cid, {"role": "assistant", "content": ""})
        for ev in _reply_events(bot_id, question):
            if ev["type"] == "delta":
                yield _chunk(cid, {"content": ev["text"]})
        yield _chunk(cid, {}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

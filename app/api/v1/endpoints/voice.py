"""Voice V0 spike: an OpenAI-compatible "custom LLM" endpoint for a managed voice agent.

A managed voice platform (ElevenLabs Agents) handles the phone call, speech-to-text,
turn-taking and text-to-speech, and calls this endpoint for the reply text. We answer
with the same RAG + model-fallback path the website widget uses, so the answer comes
only from the bot's own knowledge base.

Two entry points share the answering code:
- `/voice/lines/{line_id}/chat/completions`: production. One per AI phone line
  (app/core/voice_lines.py), authenticated by that line's own secret, and only
  answered while the line is active and the plan has minutes left.
- `/voice/spike/{bot_id}/chat/completions`: the original test endpoint, dormant (404)
  until VOICE_SPIKE_SECRET is set; one shared secret, the bot chosen in the URL.
`/voice/webhooks/elevenlabs` receives ElevenLabs' signed post-call webhook, which logs
the call in the inbox and counts its minutes.
"""
from __future__ import annotations

import hmac
import json
import logging
import re
import time
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core import config_store, elevenlabs, limits, llm, presence, rag, voice_lines
from app.core.settings import settings
from app.db.session import SessionLocal, get_db
from app.models.bot import Bot
from app.models.voice_line import VoiceLine

logger = logging.getLogger(__name__)
router = APIRouter()

# Appended to the bot's own system prompt: the reply is spoken aloud, not read.
VOICE_STYLE = (
    "\n\nYou are speaking on a live phone call. Reply in one to three short, natural spoken sentences. "
    "Never use lists, markdown, links or emoji. Say numbers, dates and times the way a person would say them. "
    "If you need a detail from the caller, ask for one thing at a time. "
    "Only state what the context supports. Never claim you said, promised or did something earlier unless it appears "
    "in this conversation; if you do not have the information, say so briefly and offer to get someone."
)
# Spoken when nothing could be generated, so the caller never hears dead air.
FALLBACK_SPEECH = "Sorry, I'm having a little trouble right now. Let me get someone to help you."
# Spoken when the line isn't in service (plan without voice, minutes used up, line paused).
UNAVAILABLE_SPEECH = "Sorry, this phone line isn't available right now. Please contact the business another way. Goodbye."
# Said just before handing the call to a person (transfer_to_number tool).
TRANSFER_SPEECH = "Sure, I'll put you through to someone now."
TRANSFER_TOOL = "transfer_to_number"

# "Can I speak to a person" style requests, answered with a transfer when the agent offers one.
_HANDOFF_RE = re.compile(
    r"\b(speak|talk|chat)\s+(to|with)\s+(a\s+|an\s+|the\s+|your\s+)?(real\s+|actual\s+)?"
    r"(human|person|someone|somebody|staff|manager|operator|agent|representative|owner)\b"
    r"|\btransfer\s+me\b|\b(a|an)\s+(real|actual)\s+(person|human)\b|\bput\s+me\s+through\b"
)


def is_handoff_request(utterance: str) -> bool:
    return bool(_HANDOFF_RE.search((utterance or "").lower()))


# Voice replies must start fast; a stalled model call is abandoned after this long (ms) and the
# next model in the chain answers. Admin/env VOICE_FIRST_TOKEN_TIMEOUT_MS tunes it; 0 turns it off.
DEFAULT_VOICE_FIRST_TOKEN_MS = 1000.0
# Preferred over the deadline above: after this long with no first token, race the next model
# alongside the slow one and take whichever speaks first. VOICE_HEDGE_MS tunes it; 0 turns it off.
DEFAULT_VOICE_HEDGE_MS = 400.0

# "Say that again" style requests. Handled without the model: the model only ever sees the last
# user turn here, so it cannot know what was said before and used to invent it. Replaying the
# agent's previous reply is exact, instant and free.
_REPEAT_RES = [re.compile(p) for p in (
    r"\b(say|repeat)\s+(that|it|this|what you said)\b(?!')",
    r"\b(say|repeat)\b.*\b(again|once more|one more time)\b",
    r"\b(didn'?t|did not|couldn'?t|could not|can'?t|cannot) (quite |really )?(catch|hear|get) (that|what you said|you|it)\b",
    r"^(sorry|pardon|pardon me|come again|excuse me|huh|what|say again|what was that|what did you say|sorry what|sorry\?)$",
)]
_NO_HISTORY_REPLY = "I haven't said anything yet. How can I help you today?"


def is_repeat_request(utterance: str) -> bool:
    text = re.sub(r"[^a-z' ]+", " ", utterance.lower()).strip()
    text = re.sub(r"\s+", " ", text)
    if not text or len(text.split()) > 9:  # a long utterance is a real question, not "say that again"
        return False
    return any(r.search(text) for r in _REPEAT_RES)


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | list | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    messages: list[_Message]
    stream: bool = False
    # OpenAI-style tools the voice platform offers (e.g. transfer_to_number).
    tools: list | None = None


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


def _voice_first_token_s() -> float:
    raw = (config_store.get("VOICE_FIRST_TOKEN_TIMEOUT_MS") or "").strip()
    try:
        ms = float(raw) if raw else DEFAULT_VOICE_FIRST_TOKEN_MS
    except ValueError:
        ms = DEFAULT_VOICE_FIRST_TOKEN_MS
    return max(ms, 0.0) / 1000.0


def _voice_hedge_s() -> float:
    raw = (config_store.get("VOICE_HEDGE_MS") or "").strip()
    try:
        ms = float(raw) if raw else DEFAULT_VOICE_HEDGE_MS
    except ValueError:
        ms = DEFAULT_VOICE_HEDGE_MS
    return max(ms, 0.0) / 1000.0


def _repeat_reply(messages: list["_Message"], question: str) -> str | None:
    """The spoken reply for a "say that again" turn, or None if this is a normal question."""
    if not is_repeat_request(question):
        return None
    prior = next((_text(m.content).strip() for m in reversed(messages) if m.role == "assistant" and _text(m.content).strip()), "")
    return f"Sure. {prior}" if prior else _NO_HISTORY_REPLY


HISTORY_MAX_MESSAGES = 6   # last three exchanges is plenty for a phone call
HISTORY_MAX_CHARS = 600    # per message; keeps the prompt (and first-token time) small


def _history(messages: list["_Message"]) -> list[dict]:
    """Earlier user/assistant turns (everything before the final user message), trimmed.
    The caller's own system prompt is ignored — the bot's prompt is authoritative."""
    last = max((i for i, m in enumerate(messages) if m.role == "user"), default=-1)
    turns = [
        {"role": m.role, "content": _text(m.content).strip()[:HISTORY_MAX_CHARS]}
        for m in messages[:last]
        if m.role in ("user", "assistant") and _text(m.content).strip()
    ]
    return turns[-HISTORY_MAX_MESSAGES:]


def _canned_events(text: str) -> Iterator[dict]:
    yield {"type": "delta", "text": text}
    yield {"type": "final", "text": text, "tokens": 0}


def _load_bot(db: Session, bot_id: int) -> Bot | None:
    return db.query(Bot).filter(Bot.id == bot_id).first()


def _reply_events(bot_id: int, question: str, history: list[dict] | None = None) -> Iterator[dict]:
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
        for ev in rag.answer_question_stream(
            db, bot, question, online, system_suffix=VOICE_STYLE,
            first_token_s=_voice_first_token_s(), hedge_s=_voice_hedge_s(), history=history,
        ):
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


def _offered_tool(tools: list | None, name: str) -> dict | None:
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name") == name:
            return fn
    return None


def _transfer_arguments(tool: dict, transfer_number: str | None) -> str:
    """Fill the tool's declared parameters: enums take their first value, number-like
    fields the line's transfer number, anything else a short reason."""
    props = ((tool.get("parameters") or {}).get("properties") or {}) if isinstance(tool.get("parameters"), dict) else {}
    args: dict = {}
    for key, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        if spec.get("enum"):
            args[key] = spec["enum"][0]
        elif any(w in key.lower() for w in ("number", "phone", "destination")) and transfer_number:
            args[key] = transfer_number
        elif spec.get("type") in (None, "string"):
            args[key] = "The caller asked to speak to a person."
    return json.dumps(args)


def _respond(bot_id: int, payload: ChatCompletionRequest, *, transfer_number: str | None = None,
             canned: str | None = None):
    """Answer one turn as an OpenAI chat completion (streamed or not)."""
    question = next((_text(m.content).strip() for m in reversed(payload.messages) if m.role == "user"), "")
    if not question:
        raise HTTPException(status_code=400, detail="No user message")
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    transfer_tool = _offered_tool(payload.tools, TRANSFER_TOOL)
    tool_call = None
    if canned is not None:
        reply = canned
    elif transfer_tool and is_handoff_request(question):
        reply = TRANSFER_SPEECH
        tool_call = {"id": f"call_{uuid.uuid4().hex[:16]}", "type": "function",
                     "function": {"name": TRANSFER_TOOL, "arguments": _transfer_arguments(transfer_tool, transfer_number)}}
    else:
        reply = _repeat_reply(payload.messages, question)
    history = _history(payload.messages)

    def events() -> Iterator[dict]:
        return _canned_events(reply) if reply is not None else _reply_events(bot_id, question, history)

    finish = "tool_calls" if tool_call else "stop"
    if not payload.stream:
        text, tokens = "", 0
        for ev in events():
            if ev["type"] == "final":
                text, tokens = ev["text"], ev.get("tokens", 0)
        message: dict = {"role": "assistant", "content": text}
        if tool_call:
            message["tool_calls"] = [tool_call]
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": int(time.time()), "model": "aichatsupport-rag",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"total_tokens": tokens},
        })

    def stream() -> Iterator[str]:
        yield _chunk(cid, {"role": "assistant", "content": ""})
        for ev in events():
            if ev["type"] == "delta":
                yield _chunk(cid, {"content": ev["text"]})
        if tool_call:
            yield _chunk(cid, {"tool_calls": [{"index": 0, **tool_call}]})
        yield _chunk(cid, {}, finish=finish)
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/available")
def voice_available() -> dict:
    """Whether this site offers AI phone lines (shows or hides the dashboard's Phone tab)."""
    return {"enabled": voice_lines.enabled()}


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
    return _respond(bot_id, payload)


@router.post("/lines/{line_id}/chat/completions")
def line_chat_completions(
    line_id: int,
    payload: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    """Production custom-LLM endpoint for one AI phone line."""
    line = db.query(VoiceLine).filter(VoiceLine.id == line_id).first()
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if line is None or not line.llm_secret_hash or not supplied or not hmac.compare_digest(
        voice_lines.hash_secret(supplied), line.llm_secret_hash
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    canned = None if voice_lines.can_take_calls(db, line) else UNAVAILABLE_SPEECH
    return _respond(line.bot_id, payload, transfer_number=line.transfer_number, canned=canned)


@router.post("/webhooks/elevenlabs")
async def elevenlabs_webhook(request: Request, db: Session = Depends(get_db)) -> JSONResponse:
    """Signed post-call webhook: log the call in the inbox and count its minutes."""
    raw = await request.body()
    secret = config_store.get("ELEVENLABS_WEBHOOK_SECRET").strip()
    if not elevenlabs.verify_webhook(raw, request.headers.get("elevenlabs-signature", ""), secret):
        logger.warning("voice webhook: bad or missing signature")
        return JSONResponse(status_code=401, content={"detail": "Invalid signature"})
    try:
        event = json.loads(raw)
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "Body is not JSON"})
    if event.get("type") == "post_call_transcription" and isinstance(event.get("data"), dict):
        try:
            call = voice_lines.record_call(db, event["data"])
        except IntegrityError:  # a concurrent redelivery stored it first
            db.rollback()
            call = None
        if call is not None:
            logger.info("voice call logged id=%s line=%s secs=%s", call.id, call.voice_line_id, call.duration_secs)
    return JSONResponse({"ok": True})

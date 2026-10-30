"""Public, JWT-less website-widget endpoints.

This is the system's only unauthenticated surface, in front of a finite GPU shared
with another app. Every request is bounded by: a per-key+IP rate limit, a global
in-flight-inference cap, a per-bot daily message cap, and the org token quota.
Tenant identity (organization_id) comes ONLY from the resolved widget Channel.
"""
import asyncio
import html
import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db, SessionLocal
from app.core import (
    rag, embeddings, llm, billing, widget, whatsapp, limits, email, config_store, storage,
    attachments, models_catalog, handoff, presence,
)
from app.core.events import bus, conv_topic, org_topic
from app.core.settings import settings
from app.models.user import User
from app.models.organization import Organization
from app.models.conversation import Message
from app.schemas.widget import (
    PublicChatRequest,
    PublicChatResponse,
    WidgetPublicConfig,
    HandoffRequest,
    HandoffResponse,
    VisitorMessagesOut,
    ConversationMessageOut,
    ImageUploadOut,
    WhatsAppLinkResponse,
    WidgetAvailability,
)
from app.schemas.site_widget import SiteWidgetPublic

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/site-widget", response_model=SiteWidgetPublic)
def site_widget_public() -> SiteWidgetPublic:
    """What (if anything) support widget the aichatsupport.my site should load.
    Unauthenticated + non-secret: the public key is embedded on the page anyway."""
    key = config_store.get("SITE_WIDGET_PUBLIC_KEY")
    src = config_store.get("SITE_WIDGET_SRC")
    on = config_store.get("SITE_WIDGET_ENABLED") == "1" and bool(key and src)
    return SiteWidgetPublic(enabled=on, src=src if on else "", public_key=key if on else "")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _notify_tenant(org_id: int, bot_name: str, name: str, contact_email: str, message: str,
                   phone: str | None = None) -> None:
    """Best-effort email to the org's active users that a visitor wants a human."""
    db = SessionLocal()
    try:
        if not email.is_configured(db):
            logger.info("handoff email skipped (SMTP off) for org %s", org_id)
            return
        recipients = [
            u.email for u in db.query(User).filter(
                User.organization_id == org_id, User.is_active.is_(True)
            ).all()
        ]
        subject = f"New chat lead for {bot_name}"
        inner = (
            '<h2 style="margin:0 0 14px;font-size:19px;color:#0f172a;">New chat lead</h2>'
            f'<p style="margin:0 0 14px;">A website visitor asked to talk to a human on '
            f'<strong>{html.escape(bot_name)}</strong>.</p>'
            f'<p style="margin:0 0 14px;"><strong>Name:</strong> {html.escape(name)}<br>'
            f'<strong>Email:</strong> {html.escape(contact_email)}'
            + (f'<br><strong>Phone:</strong> {html.escape(phone)}' if phone else "")
            + '</p>'
            f'<p style="margin:0 0 14px;"><strong>Message</strong><br>'
            f'{html.escape(message or "(none)")}</p>'
            '<p style="margin:0;color:#64748b;">Reply to them directly, or open your dashboard inbox to respond.</p>'
        )
        body_html = email.wrap_email(inner, preheader=f"New lead from {name} on {bot_name}")
        for to in recipients:
            try:
                email.send(db, to, subject, body_html)
            except Exception:  # noqa: BLE001 — one bad address shouldn't drop the rest
                logger.warning("handoff email to %s failed", to, exc_info=True)
    finally:
        db.close()


def _client_ip(request: Request) -> str:
    """Real client IP behind nginx (first X-Forwarded-For hop), else the peer."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _origin(request: Request) -> str | None:
    """The site the widget is embedded on (browser-set Origin, or Referer)."""
    o = request.headers.get("origin") or request.headers.get("referer")
    return o.strip()[:300] if o else None


def _ensure_not_blocked(db: Session, channel, session_id: str, ip: str) -> None:
    """403 if this visitor is blocked on this bot (by session, IP, or their stored
    email). Runs at ingress, before any GPU/quota spend."""
    conv = widget.find_conversation(db, channel, session_id)
    email = conv.contact_email if conv else None
    if widget.is_blocked(db, channel, session_id, ip=ip, email=email):
        raise HTTPException(status_code=403, detail="chat_unavailable")


def _claim_image(image_key: str | None, channel, bot, session_id: str) -> tuple[str | None, str | None]:
    """Validate an image_key the visitor claims to have uploaded.

    The key encodes org/bot/session, so a guessed or copied key from another
    tenant (or another visitor) is rejected outright rather than being attached
    to this conversation and fed to the model.
    """
    if not image_key:
        return None, None
    if not attachments.owns_key(image_key, channel.organization_id, bot.id, session_id):
        raise HTTPException(status_code=403, detail="That image does not belong to this chat.")
    ext = image_key.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif"}.get(ext)
    return image_key, mime


def _visitor_message_out(db: Session, m: Message) -> ConversationMessageOut:
    """Serialize a message for the visitor, resolving any image to a short URL."""
    out = ConversationMessageOut.model_validate(m)
    out.image_url = attachments.view_url(db, m.image_key, m.image_mime)
    out.image_mime = m.image_mime
    return out


_UPLOAD_CHUNK = 64 * 1024


async def _read_capped_upload(file: UploadFile, limit: int) -> bytes:
    """Read an upload in bounded chunks, rejecting once it exceeds `limit`.

    Chunked so a huge body is refused mid-stream instead of being buffered whole.
    """
    buf = bytearray()
    while True:
        part = await file.read(_UPLOAD_CHUNK)
        if not part:
            break
        buf.extend(part)
        if len(buf) > limit:
            raise HTTPException(status_code=413, detail="Image is too large (max 5MB).")
    return bytes(buf)


@router.post("/widget/{public_key}/upload", response_model=ImageUploadOut)
async def widget_upload_image(
    public_key: str,
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(""),
    db: Session = Depends(get_db),
) -> ImageUploadOut:
    """Visitor uploads an image to attach to their next message.

    Sits under /widget/{key}/ so the per-bot CORS middleware allows it from the
    tenant's site. Runs the same ingress gates as chat (resolve, rate limit,
    block list) so it can't be used to bypass them or as free storage.
    """
    channel, bot = _resolve_and_ratelimit(public_key, request, db)
    sid = widget.clean_session_id(session_id)
    ip = _client_ip(request)
    _ensure_not_blocked(db, channel, sid, ip)

    if not storage.is_configured(db):
        raise HTTPException(status_code=503, detail="Image upload is unavailable.")

    data = await _read_capped_upload(file, settings.MAX_UPLOAD_BYTES)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    mime = attachments.sniff_image_mime(data)
    if mime is None:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG, WebP or GIF images are supported.",
        )

    key = attachments.build_key(channel.organization_id, bot.id, sid, mime)
    try:
        storage.put_bytes(db, data, key, mime)
    except Exception:
        logger.exception("widget image upload failed for bot %s", bot.id)
        raise HTTPException(status_code=502, detail="Could not store the image.")

    return ImageUploadOut(
        image_key=key,
        url=attachments.view_url(db, key, mime) or "",
        mime=mime,
        size=len(data),
        session_id=sid,
    )


@router.get("/widget/{public_key}/config", response_model=WidgetPublicConfig)
def widget_config(public_key: str, db: Session = Depends(get_db)):
    resolved = widget.resolve_widget(db, public_key)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Widget not found")
    channel, bot = resolved
    ap = {**widget.DEFAULT_APPEARANCE, **(channel.appearance or {})}
    # Appearance only supplies the look; handoff_mode comes from the bot.
    fields = {k: ap[k] for k in WidgetPublicConfig.model_fields if k in ap}
    return WidgetPublicConfig(**fields, handoff_mode=handoff.effective_mode(bot))


@router.get("/widget/{public_key}/availability", response_model=WidgetAvailability)
def widget_availability(public_key: str, request: Request, db: Session = Depends(get_db)):
    """Is a human reachable right now — and if not, what should the visitor be offered?

    The widget polls this to decide whether "Talk to a human" is honest. Offering
    it while every agent is offline is worse than not offering it at all: the
    visitor waits for a reply that nobody is there to send. When no one is on duty
    the org's configured fallback (email / WhatsApp, set on the Support team page)
    is returned instead, and the widget relabels the button accordingly.
    """
    resolved = widget.resolve_widget(db, public_key)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Widget not found")
    channel, _bot = resolved
    ip = _client_ip(request)
    if not limits.rate_limit_ok(f"{public_key}:{ip}:avail", settings.WIDGET_RATE_PER_MIN, 60):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.",
                            headers={"Retry-After": "60"})

    if presence.has_available_agent(db, channel.organization_id):
        return WidgetAvailability(agents_online=True)

    org = db.query(Organization).filter(Organization.id == channel.organization_id).first()
    number = handoff.normalize_wa_number(org.support_whatsapp) if org else None
    return WidgetAvailability(
        agents_online=False,
        email=(org.support_email or None) if org else None,
        whatsapp_url=handoff.build_link(number, "Hi! I'd like to ask something.") if number else None,
    )


@router.get("/widget/{public_key}/whatsapp-redirect", response_model=WhatsAppLinkResponse)
def widget_whatsapp_redirect(public_key: str, request: Request, db: Session = Depends(get_db)):
    """If this bot's Fonnte WhatsApp is connected AND a number is set, the AI
    answers customers directly on WhatsApp — the widget launcher redirects
    there instead of opening the in-page chatbox. Independent of handoff_mode
    (that escalates an EXISTING bot conversation to a human; this decides the
    primary channel before any conversation starts). Fetched once on load
    (not on click) for the same reason as widget_whatsapp_link: opening a
    window after an await is blocked by iOS Safari.
    """
    resolved = widget.resolve_widget(db, public_key)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Widget not found")
    channel, bot = resolved
    ip = _client_ip(request)
    if not limits.rate_limit_ok(f"{public_key}:{ip}:wa-redirect", settings.WIDGET_RATE_PER_MIN, 60):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.",
                            headers={"Retry-After": "60"})

    wa_channel = whatsapp.get_whatsapp_channel(db, bot)
    if wa_channel is None or wa_channel.connection_status != "connected":
        return WhatsAppLinkResponse(enabled=False)
    number = handoff.normalize_wa_number(bot.whatsapp_number)
    if not number:
        return WhatsAppLinkResponse(enabled=False)
    return WhatsAppLinkResponse(enabled=True, url=handoff.build_link(number, "Hi! I'd like to ask something."))


def _resolve_and_ratelimit(public_key: str, request: Request, db: Session):
    """Resolve the tenant widget + the cheapest abuse gate. Runs before any streaming."""
    resolved = widget.resolve_widget(db, public_key)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Widget not found")
    channel, bot = resolved
    ip = _client_ip(request)
    if not limits.rate_limit_ok(f"{public_key}:{ip}", settings.WIDGET_RATE_PER_MIN, 60):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.",
                            headers={"Retry-After": "60"})
    return channel, bot


def _inference_gates(public_key: str, channel, db: Session) -> None:
    """Guards that only apply when we're about to spend GPU: config, quota, daily cap."""
    if not embeddings.is_configured() or not llm.is_configured():
        raise HTTPException(status_code=503, detail="Assistant temporarily unavailable.")
    sub = billing.get_or_create_subscription(db, channel.organization_id)
    if not billing.has_quota(db, sub):
        raise HTTPException(status_code=402, detail="This assistant has reached its usage limit.")
    cap = channel.widget_daily_message_cap or settings.WIDGET_DAILY_CAP_DEFAULT
    if limits.daily_incr(public_key) > cap:
        raise HTTPException(status_code=429, detail="Daily message limit reached for this assistant.",
                            headers={"Retry-After": "3600"})


def _bot_paused(db: Session, channel, session_id: str):
    """If a human owns this session's conversation, return its status (bot is paused),
    else None. When paused, the widget's chat must NOT spend GPU or bill."""
    conv = widget.find_conversation(db, channel, session_id)
    if conv and conv.status in ("needs_human", "human"):
        return conv.status
    return None


@router.post("/widget/{public_key}/chat", response_model=PublicChatResponse)
def widget_chat(
    public_key: str,
    payload: PublicChatRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Buffered (non-streaming) answer — kept as the fallback for the streaming path."""
    channel, bot = _resolve_and_ratelimit(public_key, request, db)
    session_id = widget.clean_session_id(payload.session_id)
    ip = _client_ip(request)
    src = _origin(request)
    _ensure_not_blocked(db, channel, session_id, ip)
    image_key, image_mime = _claim_image(payload.image_key, channel, bot, session_id)

    # Bot-pause: a human owns this conversation -> store the visitor turn, no inference.
    paused = _bot_paused(db, channel, session_id)
    if paused:
        conv = widget.record_user_message(db, channel, session_id, payload.question, ip=ip, source_url=src,
                                          image_key=image_key, image_mime=image_mime)
        bus.publish(org_topic(channel.organization_id), {"type": "ping", "conv_id": conv.id})
        return PublicChatResponse(answer="", session_id=session_id, status=paused, handoff=True)

    # Image with AI vision off: don't answer blind — store the picture and put the
    # conversation straight in the agent queue for a human to look at.
    if image_key and not models_catalog.vision_enabled():
        conv = widget.record_turn(db, channel, session_id, payload.question,
                                  widget.IMAGE_HANDOFF_MESSAGE, 0, ip=ip, source_url=src,
                                  image_key=image_key, image_mime=image_mime)
        widget.flag_needs_human(db, channel, session_id)
        bus.publish(org_topic(channel.organization_id), {"type": "ping", "conv_id": conv.id})
        return PublicChatResponse(answer=widget.IMAGE_HANDOFF_MESSAGE, session_id=session_id,
                                  status="needs_human", handoff=True)

    _inference_gates(public_key, channel, db)

    # Generate under the global in-flight-inference cap so the widget can never
    # monopolize the shared GPU.
    agents_online = presence.has_available_agent(db, channel.organization_id)
    try:
        with limits.inference_slot(settings.WIDGET_MAX_CONCURRENCY):
            answer, tokens = rag.answer_question(
                db, bot, payload.question, agents_online, image_key=image_key, image_mime=image_mime
            )
    except limits.AtCapacity:
        raise HTTPException(status_code=429, detail="Busy right now — please retry shortly.",
                            headers={"Retry-After": "5"})
    except Exception:  # noqa: BLE001 — upstream embedding/LLM/network failure
        # 503 (not 502) so the widget shows its "assistant unavailable" copy rather
        # than a generic "something went wrong" — e.g. every chat model rate-limited.
        logger.exception("widget chat failed for bot %s", bot.id)
        raise HTTPException(status_code=503,
                            detail="The assistant is temporarily unavailable. Please try again.",
                            headers={"Retry-After": "30"})

    if tokens:
        billing.record_usage(db, channel.organization_id, tokens)
    conv = widget.record_turn(db, channel, session_id, payload.question, answer, tokens, ip=ip, source_url=src,
                              image_key=image_key, image_mime=image_mime)
    # tokens == 0 means the no-context fallback fired -> suggest a human.
    return PublicChatResponse(answer=answer, session_id=session_id, status=conv.status,
                              handoff=(tokens == 0))


@router.post("/widget/{public_key}/chat/stream")
def widget_chat_stream(
    public_key: str,
    payload: PublicChatRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """SSE token streaming. Pre-flight gates use normal HTTP status; once the stream
    starts, everything (busy / errors / done) is a terminal in-band SSE event, since
    the HTTP status is committed the moment the first byte flushes."""
    # Gates run on the request session BEFORE we commit to a 200 stream.
    channel, _bot = _resolve_and_ratelimit(public_key, request, db)
    session_id = widget.clean_session_id(payload.session_id)
    question = payload.question
    ip = _client_ip(request)
    src = _origin(request)
    _ensure_not_blocked(db, channel, session_id, ip)
    image_key, image_mime = _claim_image(payload.image_key, channel, _bot, session_id)

    # Bot-pause: a human owns this conversation -> store the visitor turn + a terminal
    # 'done' event (no inference), so the widget flips to polling for agent replies.
    paused = _bot_paused(db, channel, session_id)
    if paused:
        pconv = widget.record_user_message(db, channel, session_id, question, ip=ip, source_url=src,
                                           image_key=image_key, image_mime=image_mime)
        bus.publish(org_topic(channel.organization_id), {"type": "ping", "conv_id": pconv.id})

        def paused_stream():
            yield _sse({"type": "done", "session_id": session_id, "status": paused, "handoff": True})

        return StreamingResponse(paused_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # Image with AI vision off -> straight to the human queue (see widget_chat).
    if image_key and not models_catalog.vision_enabled():
        iconv = widget.record_turn(db, channel, session_id, question,
                                   widget.IMAGE_HANDOFF_MESSAGE, 0, ip=ip, source_url=src,
                                   image_key=image_key, image_mime=image_mime)
        widget.flag_needs_human(db, channel, session_id)
        bus.publish(org_topic(channel.organization_id), {"type": "ping", "conv_id": iconv.id})

        def image_stream():
            yield _sse({"type": "delta", "text": widget.IMAGE_HANDOFF_MESSAGE})
            yield _sse({"type": "done", "session_id": session_id, "status": "needs_human",
                        "handoff": True})

        return StreamingResponse(image_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    _inference_gates(public_key, channel, db)

    def event_stream():
        # Hold one inference slot for the stream's whole lifetime (in-band 'busy' if none).
        token = limits.acquire_slot(settings.WIDGET_MAX_CONCURRENCY)
        if token is None:
            yield _sse({"type": "error", "detail": "busy"})
            return
        # Own DB session: the request's get_db session is closed before a streaming
        # body finishes, so retrieval + persistence must run on a fresh session.
        sdb = SessionLocal()
        org_id = None
        full_text = ""
        tokens = 0
        recorded = False
        try:
            resolved = widget.resolve_widget(sdb, public_key)
            if resolved is None:
                yield _sse({"type": "error", "detail": "not_found"})
                return
            channel, bot = resolved
            org_id = channel.organization_id
            agents_online = presence.has_available_agent(sdb, org_id)
            yield _sse({"type": "start", "session_id": session_id})
            for ev in rag.answer_question_stream(sdb, bot, question, agents_online,
                                                 image_key=image_key, image_mime=image_mime):
                if ev["type"] == "delta":
                    yield _sse({"type": "delta", "text": ev["text"]})
                elif ev["type"] == "final":
                    full_text = ev["text"]
                    tokens = ev["tokens"]
            if tokens:
                billing.record_usage(sdb, org_id, tokens)
            conv = widget.record_turn(sdb, channel, session_id, question, full_text, tokens, ip=ip, source_url=src,
                                      image_key=image_key, image_mime=image_mime)
            recorded = True
            yield _sse({"type": "done", "session_id": session_id, "status": conv.status,
                        "handoff": (tokens == 0)})
        except GeneratorExit:
            # Client disconnected mid-stream — fall through to finally (no more yields).
            raise
        except Exception:  # noqa: BLE001
            logger.exception("widget stream failed for key %s", public_key)
            yield _sse({"type": "error", "detail": "unavailable"})
        finally:
            # Meter + persist whatever was produced, even on disconnect, exactly once.
            if not recorded and full_text and org_id is not None:
                try:
                    billing.record_usage(sdb, org_id, tokens or max(1, len(full_text) // 4))
                    again = widget.resolve_widget(sdb, public_key)
                    if again is not None:
                        widget.record_turn(sdb, again[0], session_id, question, full_text, tokens, ip=ip, source_url=src,
                                           image_key=image_key, image_mime=image_mime)
                except Exception:  # noqa: BLE001
                    pass
            sdb.close()
            limits.release_slot(token)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/widget/{public_key}/whatsapp", response_model=WhatsAppLinkResponse)
def widget_whatsapp_link(
    public_key: str,
    request: Request,
    session_id: str | None = None,
    db: Session = Depends(get_db),
):
    """The wa.me link for this conversation, or enabled=false if not configured.

    Deliberately side-effect free and fetched BEFORE the visitor clicks, so the
    widget can render a real <a href>. Building the URL on click instead would
    mean opening a window after an await, which iOS Safari blocks — precisely
    the platform where a WhatsApp handoff matters most.
    """
    resolved = widget.resolve_widget(db, public_key)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Widget not found")
    channel, bot = resolved

    ip = _client_ip(request)
    if not limits.rate_limit_ok(f"{public_key}:{ip}:wa", settings.WIDGET_RATE_PER_MIN, 60):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.",
                            headers={"Retry-After": "60"})

    if handoff.effective_mode(bot) not in ("whatsapp", "both"):
        return WhatsAppLinkResponse(enabled=False)
    number = handoff.normalize_wa_number(bot.whatsapp_number)
    if not number:
        return WhatsAppLinkResponse(enabled=False)

    sid = widget.clean_session_id(session_id)
    if widget.is_blocked(db, channel, sid, ip=ip):
        raise HTTPException(status_code=403, detail="chat_unavailable")

    conv = widget.find_conversation(db, channel, sid)
    msgs = []
    if conv is not None:
        msgs = (
            db.query(Message)
            .filter(Message.conversation_id == conv.id)
            .order_by(Message.id.asc())
            .all()
        )
    text = handoff.build_message(msgs, conv.id if conv else 0, bot.name)
    return WhatsAppLinkResponse(enabled=True, url=handoff.build_link(number, text))


@router.post("/widget/{public_key}/whatsapp/opened", response_model=HandoffResponse)
def widget_whatsapp_opened(
    public_key: str,
    request: Request,
    session_id: str | None = None,
    db: Session = Depends(get_db),
):
    """Beacon: the visitor actually tapped through to WhatsApp.

    Split from the link endpoint so merely *offering* WhatsApp doesn't fill the
    inbox with conversations nobody escalated. Flagging it here keeps the thread
    visible to agents instead of the visitor silently leaving the site.

    The session comes in the QUERY STRING, not a JSON body, and that is load
    bearing: the widget sends this with navigator.sendBeacon as the page is being
    replaced by WhatsApp, and a JSON content type would force a CORS preflight
    that sendBeacon cannot perform — cross-origin it silently sends nothing while
    still returning true. A bodyless POST is a CORS-simple request.
    """
    resolved = widget.resolve_widget(db, public_key)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Widget not found")
    channel, _bot = resolved

    ip = _client_ip(request)
    if not limits.rate_limit_ok(f"{public_key}:{ip}:wa-open", settings.WIDGET_RATE_PER_MIN, 60):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.",
                            headers={"Retry-After": "60"})

    sid = widget.clean_session_id(session_id)
    conv = widget.flag_needs_human(db, channel, sid)
    if conv is None:
        return HandoffResponse(ok=True, status="bot")
    bus.publish(org_topic(channel.organization_id), {"type": "ping", "conv_id": conv.id})
    return HandoffResponse(ok=True, status=conv.status)


@router.post("/widget/{public_key}/handoff", response_model=HandoffResponse)
def widget_handoff(
    public_key: str,
    payload: HandoffRequest,
    request: Request,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Lead-capture: a visitor asks for a human. Flags their conversation needs_human,
    stores their contact, and emails the tenant (best-effort, off the request path)."""
    resolved = widget.resolve_widget(db, public_key)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Widget not found")
    channel, bot = resolved

    ip = _client_ip(request)
    if not limits.rate_limit_ok(f"{public_key}:{ip}:handoff", settings.WIDGET_RATE_PER_MIN, 60):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.",
                            headers={"Retry-After": "60"})

    session_id = widget.clean_session_id(payload.session_id)
    # Block by session/IP, and also the email they're submitting right now.
    if widget.is_blocked(db, channel, session_id, ip=ip, email=payload.email):
        raise HTTPException(status_code=403, detail="chat_unavailable")
    conv = widget.mark_handoff(db, channel, session_id, payload.name, payload.email, payload.message,
                               phone=payload.phone, ip=ip, source_url=_origin(request))
    bus.publish(org_topic(channel.organization_id), {"type": "ping", "conv_id": conv.id})
    background.add_task(_notify_tenant, channel.organization_id, bot.name,
                        payload.name, payload.email, payload.message or "", phone=payload.phone)
    return HandoffResponse(ok=True, status=conv.status)


@router.get("/widget/{public_key}/conversation/{session_id}/messages", response_model=VisitorMessagesOut)
def widget_conversation_messages(
    public_key: str,
    session_id: str,
    request: Request,
    after: int = 0,
    db: Session = Depends(get_db),
):
    """Visitor poll for live human replies: agent messages (role='agent') with id>after,
    plus the conversation status. Scoped STRICTLY to this widget's channel + this session —
    never a client-supplied conversation id — so one visitor can't read another's thread."""
    resolved = widget.resolve_widget(db, public_key)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Widget not found")
    channel, _bot = resolved
    ip = _client_ip(request)
    if not limits.rate_limit_ok(f"{public_key}:{ip}:poll", settings.WIDGET_RATE_PER_MIN * 6, 60):
        raise HTTPException(status_code=429, detail="Too many requests.", headers={"Retry-After": "10"})
    _ensure_not_blocked(db, channel, widget.clean_session_id(session_id), ip)
    conv = widget.find_conversation(db, channel, widget.clean_session_id(session_id))
    if conv is None:
        return VisitorMessagesOut(status="bot", messages=[])
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id, Message.role == "agent", Message.id > max(0, after))
        .order_by(Message.id.asc())
        .limit(100)
        .all()
    )
    return VisitorMessagesOut(
        status=conv.status,
        messages=[_visitor_message_out(db, m) for m in msgs],
    )


def _resolve_conv_for_stream(public_key: str, session_id: str, ip: str):
    """Sync (threadpool): resolve widget + conversation. Returns (conv_id|None, status),
    the sentinel "blocked", or None when the widget doesn't resolve."""
    db = SessionLocal()
    try:
        resolved = widget.resolve_widget(db, public_key)
        if resolved is None:
            return None
        channel, _bot = resolved
        sid = widget.clean_session_id(session_id)
        conv = widget.find_conversation(db, channel, sid)
        email = conv.contact_email if conv else None
        if widget.is_blocked(db, channel, sid, ip=ip, email=email):
            return "blocked"
        if conv is None:
            return (None, "bot")
        return (conv.id, conv.status)
    finally:
        db.close()


def _agent_messages_after(conv_id: int, after: int) -> list[dict]:
    db = SessionLocal()
    try:
        rows = (
            db.query(Message)
            .filter(Message.conversation_id == conv_id, Message.role == "agent", Message.id > after)
            .order_by(Message.id.asc())
            .limit(100)
            .all()
        )
        return [
            {
                "id": m.id,
                "role": "agent",
                "content": m.content,
                "sender_name": m.sender_name,
                "image_url": attachments.view_url(db, m.image_key, m.image_mime),
            }
            for m in rows
        ]
    finally:
        db.close()


@router.get("/widget/{public_key}/conversation/{session_id}/stream")
async def widget_conversation_stream(
    public_key: str,
    session_id: str,
    request: Request,
    after: int = 0,
):
    """Visitor SSE — live agent replies + status. Replays messages with id>after from
    the DB (self-healing on reconnect), then streams live off the in-process bus.
    Scoped strictly by widget key + session (no client conversation id)."""
    ip = _client_ip(request)
    resolved = await run_in_threadpool(_resolve_conv_for_stream, public_key, session_id, ip)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Widget not found")
    if resolved == "blocked":
        raise HTTPException(status_code=403, detail="chat_unavailable")
    conv_id, status0 = resolved

    async def gen():
        if conv_id is None:
            yield _sse({"type": "status", "status": status0})
            return
        q = await bus.subscribe(conv_topic(conv_id))
        last = max(0, after)
        try:
            for m in await run_in_threadpool(_agent_messages_after, conv_id, last):
                last = m["id"]
                yield _sse({"type": "message", "message": m})
            yield _sse({"type": "status", "status": status0})
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=20)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if data.get("type") == "message":
                    mid = data["message"]["id"]
                    if mid <= last:
                        continue
                    last = mid
                yield _sse(data)
                if data.get("type") == "status" and data.get("status") == "resolved":
                    break
        finally:
            bus.unsubscribe(conv_topic(conv_id), q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy import func, or_, and_
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.api.deps import get_current_user, get_agent_user
from app.core import (
    rag, embeddings, llm, models_catalog, billing, widget, whatsapp, fonnte, evolution, wa_provider, crypto,
    analytics, attachments, storage,
)
from app.core.events import bus, conv_topic, org_topic, user_topic
from app.core.settings import settings
from app.models.user import User
from app.models.bot import Bot
from app.models.channel import Channel
from app.models.conversation import Conversation, Message
from app.models.blocked_visitor import BlockedVisitor
from app.schemas.bot import BotCreate, BotOut, BotUpdate, ChatRequest, ChatResponse
from app.schemas.setting import ModelOption
from app.schemas.whatsapp import WhatsAppStatusOut, WhatsAppConnectOut
from app.schemas.widget import (
    WidgetConfigOut,
    WidgetConfigUpdate,
    ConversationOut,
    ConversationMessageOut,
    ConversationStatusUpdate,
    AgentReplyRequest,
    WidgetAnalyticsOut,
    BlockRequest,
    ConversationTransferRequest,
    ImageUploadOut,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/models", response_model=list[ModelOption])
def list_models(user: User = Depends(get_current_user)):
    """Suggested chat models for the admin Packages model picker (any authenticated
    user — actual write access is still gated by get_platform_admin on that endpoint)."""
    return models_catalog.CHAT_MODELS


def _bot_out(bot: Bot, package=None) -> BotOut:
    model, _base_url = models_catalog.resolve_candidates(bot, package)[0]
    return BotOut(
        id=bot.id,
        organization_id=bot.organization_id,
        name=bot.name,
        system_prompt=bot.system_prompt,
        fallback_message=bot.fallback_message,
        effective_chat_model=model,
        is_active=bot.is_active,
        handoff_mode=bot.handoff_mode,
        whatsapp_number=bot.whatsapp_number,
    )


def _org_package(db: Session, organization_id: int):
    sub = billing.get_or_create_subscription(db, organization_id)
    return billing.package_for(db, sub.plan), sub


@router.get("", response_model=list[BotOut])
def list_bots(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bots = db.query(Bot).filter(Bot.organization_id == user.organization_id).all()
    pkg, _sub = _org_package(db, user.organization_id)
    return [_bot_out(b, pkg) for b in bots]


@router.post("", response_model=BotOut, status_code=201)
def create_bot(payload: BotCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    # Enforce the plan's bot limit. A missing package row is treated as the most
    # restrictive (1 bot) rather than fail-open.
    pkg, sub = _org_package(db, user.organization_id)
    max_bots = pkg.max_bots if pkg is not None else 1
    current = db.query(Bot).filter(Bot.organization_id == user.organization_id).count()
    if current >= max_bots:
        plan_name = pkg.name if pkg is not None else sub.plan
        raise HTTPException(
            status_code=402,
            detail=f"Your {plan_name} plan allows {max_bots} bot(s). Upgrade to add more.",
        )
    bot = Bot(organization_id=user.organization_id, **payload.model_dump(exclude_none=True))
    db.add(bot)
    db.commit()
    db.refresh(bot)
    return _bot_out(bot, pkg)


def _message_out(db: Session, m: Message) -> ConversationMessageOut:
    """Serialize a message, resolving an attached image to a short-lived URL."""
    out = ConversationMessageOut.model_validate(m)
    out.image_url = attachments.view_url(db, m.image_key, m.image_mime)
    out.image_mime = m.image_mime
    return out


def _get_owned_bot(bot_id: int, db: Session, user: User) -> Bot:
    bot = (
        db.query(Bot)
        .filter(Bot.id == bot_id, Bot.organization_id == user.organization_id)
        .first()
    )
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    return bot


@router.patch("/{bot_id}", response_model=BotOut)
def update_bot(
    bot_id: int,
    payload: BotUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Bot:
    """Rename a bot / tweak its settings. Only the supplied fields change."""
    bot = _get_owned_bot(bot_id, db, user)
    changes = payload.model_dump(exclude_unset=True)
    if "name" in changes:
        name = (changes["name"] or "").strip()
        if not name:
            raise HTTPException(status_code=422, detail="Bot name cannot be empty.")
        changes["name"] = name
    for field, value in changes.items():
        setattr(bot, field, value)
    db.commit()
    db.refresh(bot)
    pkg, _sub = _org_package(db, user.organization_id)
    return _bot_out(bot, pkg)


@router.post("/{bot_id}/chat", response_model=ChatResponse)
def chat(
    bot_id: int,
    payload: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Dashboard test harness — ask the bot a question through the RAG engine."""
    bot = _get_owned_bot(bot_id, db, user)
    # RAG needs embeddings (retrieve) AND a chat model via OpenRouter (generate).
    # Surface a clear 503 instead of a 500 when either isn't configured yet.
    if not embeddings.is_configured() or not llm.is_configured():
        raise HTTPException(
            status_code=503,
            detail="LLM not configured: set OPENROUTER_API_KEY and VOYAGE_API_KEY "
            "(or EMBEDDINGS_PROVIDER=fake for dev).",
        )
    # Enforce the monthly token quota before spending more.
    sub = billing.get_or_create_subscription(db, user.organization_id)
    if not billing.has_quota(db, sub):
        raise HTTPException(
            status_code=402,
            detail="Monthly token quota reached. Upgrade your plan to keep chatting.",
        )
    try:
        answer, tokens = rag.answer_question(db, bot, payload.question)
    except Exception:  # noqa: BLE001 — upstream embedding/LLM/network failure
        logger.exception("chat failed for bot %s", bot_id)
        raise HTTPException(
            status_code=502,
            detail="The assistant is temporarily unavailable. Please try again.",
        )
    if tokens:
        billing.record_usage(db, user.organization_id, tokens)
    return ChatResponse(answer=answer)


# ===== Website widget config (JWT, org-scoped) =====

def _widget_out(ch: Channel) -> WidgetConfigOut:
    src = settings.WIDGET_SRC_URL
    snippet = f'<script src="{src}" data-public-key="{ch.public_key}" defer></script>'
    return WidgetConfigOut(
        enabled=bool(ch.is_active),
        public_key=ch.public_key,
        allowed_origins=list(ch.allowed_origins or []),
        appearance={**widget.DEFAULT_APPEARANCE, **(ch.appearance or {})},
        daily_message_cap=ch.widget_daily_message_cap,
        src_url=src,
        snippet=snippet,
    )


@router.get("/{bot_id}/widget", response_model=WidgetConfigOut)
def get_widget(bot_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Widget config for this bot (find-or-creates a disabled widget channel)."""
    bot = _get_owned_bot(bot_id, db, user)
    return _widget_out(widget.get_or_create_widget_channel(db, bot))


@router.put("/{bot_id}/widget", response_model=WidgetConfigOut)
def update_widget(
    bot_id: int,
    payload: WidgetConfigUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Update widget config. Origins are normalized/validated; unknown appearance keys dropped."""
    bot = _get_owned_bot(bot_id, db, user)
    ch = widget.get_or_create_widget_channel(db, bot)
    if payload.enabled is not None:
        ch.is_active = payload.enabled
    if payload.allowed_origins is not None:
        ch.allowed_origins = widget.validate_origins(payload.allowed_origins)
    if payload.appearance is not None:
        merged = {**widget.DEFAULT_APPEARANCE, **(ch.appearance or {})}
        for k, v in payload.appearance.items():
            if k in widget.DEFAULT_APPEARANCE and isinstance(v, str):
                merged[k] = v[:200]
        ch.appearance = merged
    if payload.daily_message_cap is not None:
        ch.widget_daily_message_cap = max(0, payload.daily_message_cap) or None
    db.commit()
    db.refresh(ch)
    return _widget_out(ch)


@router.post("/{bot_id}/widget/rotate-key", response_model=WidgetConfigOut)
def rotate_widget_key(bot_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Mint a new public_key (invalidates already-pasted snippets)."""
    bot = _get_owned_bot(bot_id, db, user)
    ch = widget.get_or_create_widget_channel(db, bot)
    ch.public_key = widget.gen_public_key()
    db.commit()
    db.refresh(ch)
    return _widget_out(ch)


# ===== WhatsApp (Fonnte, JWT, org-scoped) =====

def _whatsapp_webhook_urls(channel: Channel) -> tuple[str, str]:
    base = f"{settings.API_BASE_URL}{settings.API_V1_STR}/webhooks/whatsapp/{channel.provider}/{channel.id}/{channel.webhook_secret}"
    return f"{base}/message", f"{base}/status"


def _whatsapp_status_out(ch: Channel, pkg) -> WhatsAppStatusOut:
    return WhatsAppStatusOut(
        connection_status=ch.connection_status,
        is_active=bool(ch.is_active),
        plan_allows_whatsapp=bool(pkg.whatsapp_enabled) if pkg is not None else True,
    )


@router.get("/{bot_id}/whatsapp", response_model=WhatsAppStatusOut)
def get_whatsapp(bot_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """WhatsApp connection status for this bot (find-or-creates an unlinked channel)."""
    bot = _get_owned_bot(bot_id, db, user)
    pkg, _sub = _org_package(db, user.organization_id)
    return _whatsapp_status_out(whatsapp.get_or_create_whatsapp_channel(db, bot), pkg)


@router.get("/{bot_id}/whatsapp/status", response_model=WhatsAppStatusOut)
def whatsapp_status(bot_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Cheap poll target for the frontend while a QR is displayed."""
    bot = _get_owned_bot(bot_id, db, user)
    pkg, _sub = _org_package(db, user.organization_id)
    return _whatsapp_status_out(whatsapp.get_or_create_whatsapp_channel(db, bot), pkg)


@router.post("/{bot_id}/whatsapp/connect", response_model=WhatsAppConnectOut)
def connect_whatsapp(bot_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Provision (if needed) this bot's Fonnte device and return a QR to scan.
    Safe to call again while pending — re-fetches the QR for the same device."""
    bot = _get_owned_bot(bot_id, db, user)
    pkg, _sub = _org_package(db, user.organization_id)
    if pkg is not None and not pkg.whatsapp_enabled:
        raise HTTPException(
            status_code=402,
            detail=f"WhatsApp isn't available on your {pkg.name} plan. Upgrade to connect a number.",
        )
    ch = whatsapp.get_or_create_whatsapp_channel(db, bot)

    if ch.connection_status == "connected":
        return WhatsAppConnectOut(connection_status=ch.connection_status, qr_base64=None)

    if not ch.access_token and ch.provider != wa_provider.default_provider():
        # Never linked yet — adopt the platform's current default provider.
        ch.provider = wa_provider.default_provider()
        db.commit()
    if ch.provider == "evolution":
        return _connect_evolution(db, ch)
    return _connect_fonnte(db, ch, bot)


def _connect_evolution(db: Session, ch: Channel) -> WhatsAppConnectOut:
    name = wa_provider.instance_name(ch)
    webhook_url, _status_url = _whatsapp_webhook_urls(ch)
    try:
        if not ch.access_token:
            created = evolution.create_instance(name, webhook_url)
            ch.phone_number_id = name
            token = created.get("hash")
            token = token.get("apikey") if isinstance(token, dict) else token
            ch.access_token = crypto.encrypt(token or name)
            db.commit()
            qr_b64 = (created.get("qrcode") or {}).get("base64")
        else:
            evolution.set_webhook(name, webhook_url)  # idempotent re-assert
            qr_b64 = None
        if not qr_b64:
            qr_b64 = evolution.get_qr(name)["base64"]
    except evolution.AlreadyConnected:
        ch.connection_status = "connected"
        ch.is_active = True
        db.commit()
        return WhatsAppConnectOut(connection_status="connected", qr_base64=None)
    except evolution.EvolutionError as e:
        raise HTTPException(status_code=502, detail=f"Could not connect WhatsApp: {e}")
    ch.connection_status = "pending_qr"
    db.commit()
    return WhatsAppConnectOut(connection_status="pending_qr", qr_base64=qr_b64)


def _connect_fonnte(db: Session, ch: Channel, bot: Bot) -> WhatsAppConnectOut:
    device_name = f"bot-{bot.id}"
    if not ch.access_token:
        device_number = whatsapp.gen_device_placeholder(ch.id)
        try:
            created = fonnte.add_device(name=device_name, device_number=device_number)
        except fonnte.FonnteError as e:
            raise HTTPException(status_code=502, detail=f"Could not create WhatsApp device: {e}")
        ch.phone_number_id = device_number
        ch.access_token = crypto.encrypt(created["token"])
        db.commit()
        db.refresh(ch)

    # Reassert the webhook config every time (idempotent, cheap) rather than only
    # at first provisioning — otherwise a channel whose one-shot config attempt
    # failed (e.g. a transient error) stays permanently unwired with no retry path.
    webhook_url, webhook_connect_url = _whatsapp_webhook_urls(ch)
    try:
        fonnte.update_device(crypto.decrypt(ch.access_token), device_name, ch.phone_number_id,
                             webhook_url, webhook_connect_url)
    except fonnte.FonnteError as e:
        raise HTTPException(status_code=502, detail=f"Could not configure WhatsApp webhooks: {e}")

    try:
        qr = fonnte.get_qr(crypto.decrypt(ch.access_token))
    except fonnte.AlreadyConnected:
        ch.connection_status = "connected"
        ch.is_active = True
        db.commit()
        return WhatsAppConnectOut(connection_status="connected", qr_base64=None)
    except fonnte.FonnteError as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch WhatsApp QR: {e}")

    ch.connection_status = "pending_qr"
    db.commit()
    return WhatsAppConnectOut(connection_status="pending_qr", qr_base64=qr.get("url"))


@router.post("/{bot_id}/whatsapp/disconnect", response_model=WhatsAppStatusOut)
def disconnect_whatsapp(bot_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bot = _get_owned_bot(bot_id, db, user)
    ch = whatsapp.get_or_create_whatsapp_channel(db, bot)
    if ch.access_token:
        try:
            wa_provider.disconnect(ch)
        except (fonnte.FonnteError, evolution.EvolutionError) as e:
            raise HTTPException(status_code=502, detail=f"Could not disconnect WhatsApp: {e}")
    ch.connection_status = "disconnected"
    ch.is_active = False
    db.commit()
    db.refresh(ch)
    pkg, _sub = _org_package(db, user.organization_id)
    return _whatsapp_status_out(ch, pkg)


# ===== Conversation inbox (JWT, org + bot scoped) =====

def _conv_out(conv: Conversation, message_count: int, channel_kind: str | None,
              assignee_name: str | None = None) -> ConversationOut:
    return ConversationOut(
        id=conv.id,
        status=conv.status,
        channel_kind=channel_kind,
        contact_name=conv.contact_name,
        contact_email=conv.contact_email,
        contact_phone=conv.contact_phone,
        source_url=conv.source_url,
        needs_human_at=conv.needs_human_at,
        assigned_user_id=conv.assigned_user_id,
        assignee_name=assignee_name,
        last_message_at=conv.last_message_at,
        created_at=conv.created_at,
        message_count=message_count,
    )


def _conv_block_identifiers(conv: Conversation) -> list[tuple[str, str]]:
    """Durable identifiers to block/unblock for a conversation (session, IP, email)."""
    out: list[tuple[str, str]] = []
    if conv.external_user_id:
        out.append(("session", conv.external_user_id))
    if conv.last_ip:
        out.append(("ip", conv.last_ip))
    if conv.contact_email:
        e = conv.contact_email.strip().lower()
        if e:
            out.append(("email", e))
    return out


def _is_conv_blocked(db: Session, conv: Conversation) -> bool:
    idents = _conv_block_identifiers(conv)
    if not idents:
        return False
    conds = [and_(BlockedVisitor.identifier_type == t, BlockedVisitor.identifier == v) for t, v in idents]
    return (
        db.query(BlockedVisitor.id)
        .filter(
            BlockedVisitor.organization_id == conv.organization_id,
            BlockedVisitor.bot_id == conv.bot_id,
            or_(*conds),
        )
        .first()
        is not None
    )


def _conv_out_full(conv: Conversation, db: Session, bot: Bot) -> ConversationOut:
    """Build a ConversationOut for a single conversation (message count + assignee name)."""
    mc = db.query(func.count(Message.id)).filter(Message.conversation_id == conv.id).scalar() or 0
    kinds = {c.id: c.kind for c in db.query(Channel).filter(Channel.bot_id == bot.id).all()}
    name = None
    if conv.assigned_user_id:
        u = db.query(User).filter(User.id == conv.assigned_user_id).first()
        name = (u.full_name or u.email) if u else None
    out = _conv_out(conv, int(mc), kinds.get(conv.channel_id), name)
    out.blocked = _is_conv_blocked(db, conv)
    return out


def _owned_conversation(bot: Bot, conv_id: int, db: Session) -> Conversation:
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conv_id,
            Conversation.bot_id == bot.id,
            Conversation.organization_id == bot.organization_id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.get("/{bot_id}/conversations", response_model=list[ConversationOut])
def list_conversations(
    bot_id: int,
    status: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Conversations for a bot (newest first). Filter by status (e.g. needs_human)."""
    bot = _get_owned_bot(bot_id, db, user)
    q = db.query(Conversation).filter(
        Conversation.bot_id == bot.id,
        Conversation.organization_id == user.organization_id,
    )
    if status:
        q = q.filter(Conversation.status == status)
    convs = q.order_by(Conversation.last_message_at.desc()).limit(200).all()
    if not convs:
        return []
    ids = [c.id for c in convs]
    counts = dict(
        db.query(Message.conversation_id, func.count(Message.id))
        .filter(Message.conversation_id.in_(ids))
        .group_by(Message.conversation_id)
        .all()
    )
    kinds = {c.id: c.kind for c in db.query(Channel).filter(Channel.bot_id == bot.id).all()}
    assignee_ids = {c.assigned_user_id for c in convs if c.assigned_user_id}
    names = {}
    if assignee_ids:
        for u in db.query(User).filter(User.id.in_(assignee_ids)).all():
            names[u.id] = u.full_name or u.email
    return [_conv_out(c, counts.get(c.id, 0), kinds.get(c.channel_id), names.get(c.assigned_user_id)) for c in convs]


@router.get("/{bot_id}/conversations/{conv_id}/messages", response_model=list[ConversationMessageOut])
def conversation_messages(
    bot_id: int,
    conv_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    bot = _get_owned_bot(bot_id, db, user)
    conv = _owned_conversation(bot, conv_id, db)
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )
    return [_message_out(db, m) for m in msgs]


@router.patch("/{bot_id}/conversations/{conv_id}", response_model=ConversationOut)
def update_conversation(
    bot_id: int,
    conv_id: int,
    payload: ConversationStatusUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Update a conversation's status (e.g. mark a lead resolved / hand back to the bot)."""
    bot = _get_owned_bot(bot_id, db, user)
    conv = _owned_conversation(bot, conv_id, db)
    conv.status = payload.status
    if payload.status == "resolved":
        conv.resolved_at = datetime.utcnow()
    elif payload.status == "bot":
        conv.assigned_user_id = None  # hand back to the bot -> drop the agent assignment
    db.commit()
    db.refresh(conv)
    bus.publish(conv_topic(conv.id), {"type": "status", "status": conv.status})
    bus.publish(org_topic(conv.organization_id), {"type": "ping", "conv_id": conv.id})
    return _conv_out_full(conv, db, bot)


# ===== Live human-agent takeover (JWT, org + bot scoped) =====

@router.post("/{bot_id}/conversations/{conv_id}/claim", response_model=ConversationOut)
def claim_conversation(
    bot_id: int,
    conv_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_agent_user),
):
    """Claim an unassigned conversation. Double-claim-safe: a single conditional UPDATE
    whose WHERE clause IS the lock. 409 if another agent already owns it."""
    bot = _get_owned_bot(bot_id, db, user)
    _owned_conversation(bot, conv_id, db)  # 404 if not this org/bot
    updated = (
        db.query(Conversation)
        .filter(
            Conversation.id == conv_id,
            Conversation.bot_id == bot.id,
            Conversation.organization_id == user.organization_id,
            Conversation.assigned_user_id.is_(None),
        )
        .update(
            {
                Conversation.assigned_user_id: user.id,
                Conversation.status: "human",
                Conversation.assigned_at: datetime.utcnow(),
            },
            synchronize_session=False,
        )
    )
    db.commit()
    conv = _owned_conversation(bot, conv_id, db)
    if updated == 0 and conv.assigned_user_id != user.id:
        raise HTTPException(status_code=409, detail="Already claimed by another agent.")
    bus.publish(org_topic(user.organization_id), {"type": "ping", "conv_id": conv_id})
    return _conv_out_full(conv, db, bot)


_AGENT_UPLOAD_CHUNK = 64 * 1024


def _claim_agent_image(image_key: str | None, conv: Conversation) -> tuple[str | None, str | None]:
    """An agent may only attach an image uploaded into THIS conversation."""
    if not image_key:
        return None, None
    if not attachments.owns_agent_key(image_key, conv.organization_id, conv.bot_id, conv.id):
        raise HTTPException(status_code=403, detail="That image does not belong to this conversation.")
    ext = image_key.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif"}.get(ext)
    return image_key, mime


@router.post("/{bot_id}/conversations/{conv_id}/upload", response_model=ImageUploadOut)
async def upload_conversation_image(
    bot_id: int,
    conv_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_agent_user),
) -> ImageUploadOut:
    """Agent uploads an image to send into a conversation (e.g. an annotated screenshot)."""
    bot = _get_owned_bot(bot_id, db, user)
    conv = _owned_conversation(bot, conv_id, db)
    if not storage.is_configured(db):
        raise HTTPException(status_code=503, detail="Image upload is unavailable.")

    buf = bytearray()
    while True:
        part = await file.read(_AGENT_UPLOAD_CHUNK)
        if not part:
            break
        buf.extend(part)
        if len(buf) > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image is too large (max 5MB).")
    data = bytes(buf)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    mime = attachments.sniff_image_mime(data)
    if mime is None:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, WebP or GIF images are supported.")

    key = attachments.agent_key(conv.organization_id, bot.id, conv.id, mime)
    try:
        storage.put_bytes(db, data, key, mime)
    except Exception:
        logger.exception("agent image upload failed for conversation %s", conv.id)
        raise HTTPException(status_code=502, detail="Could not store the image.")
    return ImageUploadOut(
        image_key=key,
        url=attachments.view_url(db, key, mime) or "",
        mime=mime,
        size=len(data),
        session_id="",
    )


@router.post("/{bot_id}/conversations/{conv_id}/reply", response_model=ConversationMessageOut)
def reply_conversation(
    bot_id: int,
    conv_id: int,
    payload: AgentReplyRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_agent_user),
):
    """Post a human agent reply (role='agent'). Self-claims + flips status->human.
    Ownership-guarded so a losing racer can't reply. No billing (human labor != GPU)."""
    bot = _get_owned_bot(bot_id, db, user)
    conv = _owned_conversation(bot, conv_id, db)
    if conv.assigned_user_id not in (None, user.id) and user.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="This conversation is handled by another agent.")
    now = datetime.utcnow()
    if conv.assigned_user_id is None:
        conv.assigned_user_id = user.id
        conv.assigned_at = now
    conv.status = "human"
    conv.last_agent_at = now
    conv.last_message_at = now
    agent_name = user.full_name or user.email
    image_key, image_mime = _claim_agent_image(payload.image_key, conv)
    msg = Message(
        organization_id=conv.organization_id,
        conversation_id=conv.id,
        role="agent",
        content=payload.content,
        tokens=0,
        sender_user_id=user.id,
        sender_name=agent_name,
        image_key=image_key,
        image_mime=image_mime,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    out = _message_out(db, msg)
    # Realtime: push the reply to the visitor's stream + nudge the org's agent queue.
    bus.publish(conv_topic(conv.id),
                {"type": "message",
                 "message": {"id": msg.id, "role": "agent", "content": msg.content,
                             "sender_name": agent_name, "image_url": out.image_url}})
    bus.publish(org_topic(conv.organization_id), {"type": "ping", "conv_id": conv.id})
    _deliver_to_whatsapp_if_applicable(db, conv, payload.content)
    return out


def _deliver_to_whatsapp_if_applicable(db: Session, conv: Conversation, content: str) -> None:
    """If this conversation is on a WhatsApp channel, actually send the agent's
    reply to the customer's phone — otherwise it only ever lands in our own DB.
    Text only for now (image attachments aren't forwarded to WhatsApp yet)."""
    if not conv.channel_id or not content.strip():
        return
    channel = db.query(Channel).filter(
        Channel.id == conv.channel_id, Channel.kind == whatsapp.WHATSAPP_KIND,
    ).first()
    if channel is None or not channel.access_token:
        return
    try:
        wa_provider.send_text(channel, conv.external_user_id, content)
    except Exception:  # noqa: BLE001 — reply already saved; delivery failure shouldn't 500 the request
        logger.exception("whatsapp send failed for conversation %s", conv.id)


@router.post("/{bot_id}/conversations/{conv_id}/release", response_model=ConversationOut)
def release_conversation(
    bot_id: int,
    conv_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_agent_user),
):
    """Release a conversation back to the shared needs_human queue."""
    bot = _get_owned_bot(bot_id, db, user)
    conv = _owned_conversation(bot, conv_id, db)
    conv.assigned_user_id = None
    conv.assigned_at = None
    conv.status = "needs_human"
    db.commit()
    db.refresh(conv)
    bus.publish(conv_topic(conv.id), {"type": "status", "status": conv.status})
    bus.publish(org_topic(conv.organization_id), {"type": "ping", "conv_id": conv.id})
    return _conv_out_full(conv, db, bot)


@router.post("/{bot_id}/conversations/{conv_id}/transfer", response_model=ConversationOut)
def transfer_conversation(
    bot_id: int,
    conv_id: int,
    payload: ConversationTransferRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_agent_user),
):
    """Hand this conversation to another agent in the same org. Keeps the bot paused
    (status stays 'human'); notifies the target over their directed stream."""
    bot = _get_owned_bot(bot_id, db, user)
    conv = _owned_conversation(bot, conv_id, db)
    # Only the current owner (or an owner/admin) may transfer.
    if conv.assigned_user_id not in (None, user.id) and user.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="This conversation is handled by another agent.")
    # Target must be active staff in THIS org (404, not 403, to avoid cross-tenant id probing).
    target = (
        db.query(User)
        .filter(
            User.id == payload.target_user_id,
            User.organization_id == conv.organization_id,
            User.is_active.is_(True),
        )
        .first()
    )
    if target is None or target.role not in ("owner", "admin", "agent"):
        raise HTTPException(status_code=404, detail="Agent not found")
    if target.id == conv.assigned_user_id:
        raise HTTPException(status_code=400, detail="Already assigned to that agent.")

    now = datetime.utcnow()
    conv.assigned_user_id = target.id
    conv.assigned_at = now
    conv.status = "human"  # keep the bot paused through the handoff
    conv.last_agent_at = now
    db.commit()
    db.refresh(conv)

    note = (payload.note or "").strip()[:500] or None
    bus.publish(user_topic(target.id), {
        "type": "transfer",
        "conv_id": conv.id,
        "bot_id": bot.id,
        "from_user_id": user.id,
        "from_name": user.full_name or user.email,
        "to_user_id": target.id,
        "note": note,
    })
    bus.publish(org_topic(conv.organization_id), {"type": "ping", "conv_id": conv.id})
    return _conv_out_full(conv, db, bot)


@router.post("/{bot_id}/conversations/{conv_id}/block", response_model=ConversationOut)
def block_conversation(
    bot_id: int,
    conv_id: int,
    payload: BlockRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_agent_user),
):
    """Block this visitor from the bot's widget by every durable identifier we have
    (session token, last IP, email). Blocking ends the chat: resolve + unassign."""
    bot = _get_owned_bot(bot_id, db, user)
    conv = _owned_conversation(bot, conv_id, db)
    reason = (payload.reason or "").strip()[:500] or None
    for id_type, value in _conv_block_identifiers(conv):
        exists = (
            db.query(BlockedVisitor)
            .filter(
                BlockedVisitor.organization_id == conv.organization_id,
                BlockedVisitor.bot_id == conv.bot_id,
                BlockedVisitor.identifier_type == id_type,
                BlockedVisitor.identifier == value,
            )
            .first()
        )
        if exists is None:
            db.add(BlockedVisitor(
                organization_id=conv.organization_id,
                bot_id=conv.bot_id,
                channel_id=conv.channel_id,
                identifier_type=id_type,
                identifier=value,
                blocked_by_user_id=user.id,
                reason=reason,
            ))
    conv.status = "resolved"
    conv.resolved_at = datetime.utcnow()
    conv.assigned_user_id = None
    db.commit()
    db.refresh(conv)
    bus.publish(conv_topic(conv.id), {"type": "status", "status": "resolved"})
    bus.publish(org_topic(conv.organization_id), {"type": "ping", "conv_id": conv.id})
    return _conv_out_full(conv, db, bot)


@router.delete("/{bot_id}/conversations/{conv_id}/block", response_model=ConversationOut)
def unblock_conversation(
    bot_id: int,
    conv_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_agent_user),
):
    """Lift the block for this visitor's identifiers (session, IP, email)."""
    bot = _get_owned_bot(bot_id, db, user)
    conv = _owned_conversation(bot, conv_id, db)
    for id_type, value in _conv_block_identifiers(conv):
        db.query(BlockedVisitor).filter(
            BlockedVisitor.organization_id == conv.organization_id,
            BlockedVisitor.bot_id == conv.bot_id,
            BlockedVisitor.identifier_type == id_type,
            BlockedVisitor.identifier == value,
        ).delete(synchronize_session=False)
    db.commit()
    db.refresh(conv)
    return _conv_out_full(conv, db, bot)


@router.get("/{bot_id}/widget/analytics", response_model=WidgetAnalyticsOut)
def widget_analytics(
    bot_id: int,
    days: int = 30,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Usage analytics for a bot over the last `days`: volume, leads, fallback rate,
    token spend, a daily series, top questions, and recent unanswered questions."""
    bot = _get_owned_bot(bot_id, db, user)
    return analytics.widget_analytics(db, bot, days)


# ===== Realtime (SSE) — live agent queue/thread updates =====

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.get("/{bot_id}/events")
async def agent_events(
    bot_id: int,
    request: Request,
    user: User = Depends(get_agent_user),
):
    """SSE for the agent dashboard/app: a 'ping' (with conv_id) on any change in the
    org's conversations — new lead, new visitor message, claim, status. The client
    refetches the queue (and the open thread) on a ping. Org-scoped, agent-guarded."""
    org_id = user.organization_id

    async def gen():
        q = await bus.subscribe(org_topic(org_id))
        try:
            yield _sse({"type": "connected"})
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=20)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # keep the connection warm through proxies
                    continue
                yield _sse(data)
        finally:
            bus.unsubscribe(org_topic(org_id), q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

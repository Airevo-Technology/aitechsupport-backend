"""WhatsApp webhooks.

Two providers live here:
- Meta Business Cloud API (GET verify + POST inbound below) — a dormant stub,
  kept for a possible future direct-Meta integration. Not wired up.
- Evolution API (the `/evolution/...` routes) — self-hosted gateway, the default
  for new channels; same URL-secret auth as Fonnte. Evolution posts every event
  to one URL, so both routes share a dispatcher on the payload's `event`.
- Fonnte (the `/fonnte/...` routes) — still serves already-linked channels. An unofficial
  QR-linked WhatsApp Web gateway; one platform-owned Fonnte account provisions
  a device per client bot (see app/core/fonnte.py). Fonnte doesn't sign its
  webhook payloads, so routing/auth for these routes is entirely URL-based:
  the channel id + a random per-channel secret embedded in the path (set as
  this channel's webhook/webhookconnect URL via fonnte.update_device).

Both POST routes MUST return 200 fast — the heavy work (retrieve + LLM answer
+ outbound send) runs in a background task, per CLAUDE.md.
"""
import logging

from fastapi import APIRouter, Request, Response, BackgroundTasks, Query

from app.core.settings import settings
from app.core import billing, embeddings, llm, rag, wa_provider, whatsapp as wa_core
from app.core.events import bus, org_topic
from app.db.session import SessionLocal
from app.models.bot import Bot

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
def verify(
    mode: str = Query(alias="hub.mode", default=""),
    token: str = Query(alias="hub.verify_token", default=""),
    challenge: str = Query(alias="hub.challenge", default=""),
):
    """Meta calls this once when you subscribe the webhook."""
    if mode == "subscribe" and token == settings.WHATSAPP_VERIFY_TOKEN:
        return Response(content=challenge, media_type="text/plain")
    return Response(status_code=403)


@router.post("")
async def inbound(request: Request, background: BackgroundTasks):
    payload = await request.json()
    # TODO Phase 2 (if a direct-Meta integration is ever built): validate
    # X-Hub-Signature-256 with WHATSAPP_APP_SECRET, resolve the Channel by
    # phone_number_id, enqueue RAG answer + outbound send.
    background.add_task(_handle_inbound, payload)
    return {"status": "received"}


def _handle_inbound(payload: dict) -> None:
    # Placeholder for the (dormant) Meta-direct worker entrypoint.
    ...


# ===== Fonnte =====

@router.post("/fonnte/{channel_id}/{secret}/message")
async def fonnte_message(channel_id: int, secret: str, request: Request, background: BackgroundTasks):
    payload = await request.json()
    background.add_task(_handle_fonnte_message, channel_id, secret, payload)
    return {"status": "received"}


@router.post("/fonnte/{channel_id}/{secret}/status")
async def fonnte_status(channel_id: int, secret: str, request: Request, background: BackgroundTasks):
    payload = await request.json()
    background.add_task(_handle_fonnte_status, channel_id, secret, payload)
    return {"status": "received"}


def _handle_fonnte_status(channel_id: int, secret: str, payload: dict) -> None:
    """{device, status: "connect"|"disconnect", timestamp, reason?}"""
    db = SessionLocal()
    try:
        channel = wa_core.resolve_channel(db, channel_id, secret, "fonnte")
        if channel is None:
            logger.warning("fonnte status webhook: unknown channel %s", channel_id)
            return
        status = payload.get("status")
        if status == "connect":
            channel.connection_status = "connected"
            channel.is_active = True
        elif status == "disconnect":
            channel.connection_status = "disconnected"
        db.commit()
        bus.publish(org_topic(channel.organization_id),
                    {"type": "whatsapp_status", "channel_id": channel.id, "status": channel.connection_status})
    except Exception:  # noqa: BLE001 — never let a background task crash silently
        logger.exception("fonnte status webhook failed for channel %s", channel_id)
    finally:
        db.close()


def _handle_fonnte_message(channel_id: int, secret: str, payload: dict) -> None:
    """{device, sender, name, message, timestamp, inboxid, url?, filename?, ...}"""
    db = SessionLocal()
    try:
        channel = wa_core.resolve_channel(db, channel_id, secret, "fonnte")
        if channel is None:
            logger.warning("fonnte message webhook: unknown channel %s", channel_id)
            return
        sender = (payload.get("sender") or "").strip()
        text = (payload.get("message") or "").strip()
        _answer_inbound(db, channel, sender, text)
    except Exception:  # noqa: BLE001 — never let a background task crash silently
        logger.exception("fonnte message webhook failed for channel %s", channel_id)
    finally:
        db.close()


def _answer_inbound(db, channel, sender: str, text: str) -> None:
    """Provider-agnostic: store the inbound message, answer from the KB, send the reply."""
    bot = db.query(Bot).filter(Bot.id == channel.bot_id, Bot.is_active.is_(True)).first()
    if bot is None:
        return
    if not sender or not text:
        return  # no-caption media, poll responses, etc. — nothing to answer yet

    # A human already owns this thread -> store the message only, don't answer.
    if wa_core.is_paused(db, channel, sender):
        conv = wa_core.record_user_message(db, channel, sender, text)
        bus.publish(org_topic(channel.organization_id), {"type": "ping", "conv_id": conv.id})
        return

    if not embeddings.is_configured() or not llm.is_configured():
        logger.error("whatsapp webhook: LLM/embeddings not configured (channel %s)", channel.id)
        return
    sub = billing.get_or_create_subscription(db, channel.organization_id)
    if not billing.has_quota(db, sub):
        logger.warning("whatsapp webhook: quota exhausted for org %s", channel.organization_id)
        return

    try:
        answer, tokens = rag.answer_question(db, bot, text)
    except Exception:  # noqa: BLE001 — upstream embedding/LLM/network failure
        logger.exception("whatsapp webhook: answer failed for bot %s", bot.id)
        return

    if tokens:
        billing.record_usage(db, channel.organization_id, tokens)
    conv = wa_core.record_turn(db, channel, sender, text, answer, tokens)
    # No "talk to a human" button on WhatsApp — escalate automatically when
    # the bot has nothing to answer with, so an agent picks it up proactively.
    if tokens == 0:
        wa_core.flag_needs_human(db, channel, sender)

    if channel.access_token:
        try:
            wa_provider.send_text(channel, sender, answer)
        except Exception:  # noqa: BLE001 — reply persisted even if delivery fails
            logger.exception("whatsapp send failed for channel %s", channel.id)

    bus.publish(org_topic(channel.organization_id), {"type": "ping", "conv_id": conv.id})


# ===== Evolution API =====

@router.post("/evolution/{channel_id}/{secret}/message")
@router.post("/evolution/{channel_id}/{secret}/status")
async def evolution_event(channel_id: int, secret: str, request: Request, background: BackgroundTasks):
    payload = await request.json()
    background.add_task(_handle_evolution_event, channel_id, secret, payload)
    return {"status": "received"}


def _evolution_text(message: dict) -> str:
    return (
        message.get("conversation")
        or (message.get("extendedTextMessage") or {}).get("text")
        or (message.get("imageMessage") or {}).get("caption")
        or ""
    ).strip()


def _handle_evolution_event(channel_id: int, secret: str, payload: dict) -> None:
    """{event: "messages.upsert"|"connection.update", instance, data: {...}}"""
    db = SessionLocal()
    try:
        channel = wa_core.resolve_channel(db, channel_id, secret, "evolution")
        if channel is None:
            logger.warning("evolution webhook: unknown channel %s", channel_id)
            return
        event = str(payload.get("event") or "").lower().replace("_", ".")
        data = payload.get("data")
        if isinstance(data, list):
            data = data[0] if data else {}
        data = data or {}

        if event == "connection.update":
            state = data.get("state")
            if state == "open":
                channel.connection_status = "connected"
                channel.is_active = True
            elif state == "close":
                channel.connection_status = "disconnected"
            else:
                return
            db.commit()
            bus.publish(org_topic(channel.organization_id),
                        {"type": "whatsapp_status", "channel_id": channel.id, "status": channel.connection_status})
        elif event == "messages.upsert":
            key = data.get("key") or {}
            jid = key.get("remoteJid") or ""
            # Skip our own outbound echoes, groups and status broadcasts.
            if key.get("fromMe") or not jid.endswith("@s.whatsapp.net"):
                return
            _answer_inbound(db, channel, jid.split("@")[0], _evolution_text(data.get("message") or {}))
    except Exception:  # noqa: BLE001 — never let a background task crash silently
        logger.exception("evolution webhook failed for channel %s", channel_id)
    finally:
        db.close()

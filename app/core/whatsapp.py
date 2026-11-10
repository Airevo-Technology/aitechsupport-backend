"""WhatsApp (Fonnte) helpers: the Channel record, conversation persistence keyed
on the customer's phone number, and human-handoff transitions.

Mirrors app/core/widget.py's shape for the website widget — Conversation and
Message are channel-agnostic, so the same find-or-create-by-external_user_id
pattern applies, just keyed on a phone number instead of a session token.
"""
from __future__ import annotations

import secrets
from datetime import datetime

from sqlalchemy.orm import Session

from app.core import wa_provider
from app.models.channel import Channel
from app.models.bot import Bot
from app.models.conversation import Conversation, Message

WHATSAPP_KIND = "whatsapp"


def gen_webhook_secret() -> str:
    return secrets.token_urlsafe(32)


def gen_device_placeholder(channel_id: int) -> str:
    """Fonnte's `device` param at device-creation time is an arbitrary unique
    string (confirmed in Fonnte's docs: "not necessarily a whatsapp number"),
    since the real linked number isn't known until the client scans the QR.
    Prefixed to stay distinctive within Fonnte's platform-wide uniqueness rule.
    """
    return f"9{channel_id:010d}"


def get_whatsapp_channel(db: Session, bot: Bot) -> Channel | None:
    return (
        db.query(Channel)
        .filter(
            Channel.bot_id == bot.id,
            Channel.organization_id == bot.organization_id,
            Channel.kind == WHATSAPP_KIND,
        )
        .first()
    )


def get_or_create_whatsapp_channel(db: Session, bot: Bot) -> Channel:
    """Idempotently return the bot's single WhatsApp Channel (unlinked until connected)."""
    ch = get_whatsapp_channel(db, bot)
    if ch is None:
        ch = Channel(
            organization_id=bot.organization_id,
            bot_id=bot.id,
            kind=WHATSAPP_KIND,
            provider=wa_provider.default_provider(),
            connection_status="disconnected",
            webhook_secret=gen_webhook_secret(),
            is_active=False,
        )
        db.add(ch)
        db.commit()
        db.refresh(ch)
    return ch


def resolve_channel(db: Session, channel_id: int, secret: str, provider: str | None = None) -> Channel | None:
    """Resolve a WhatsApp Channel from a webhook URL's id+secret, else None.
    Fonnte/Evolution don't sign webhook payloads, so this URL-based check IS the
    auth. `provider`, when given, must match the channel's own provider."""
    ch = (
        db.query(Channel)
        .filter(Channel.id == channel_id, Channel.kind == WHATSAPP_KIND)
        .first()
    )
    if ch is not None and provider and ch.provider != provider:
        return None
    if ch is None or not ch.webhook_secret:
        return None
    if not secrets.compare_digest(ch.webhook_secret, secret):
        return None
    return ch


def find_conversation(db: Session, channel: Channel, phone: str) -> Conversation | None:
    return (
        db.query(Conversation)
        .filter(
            Conversation.channel_id == channel.id,
            Conversation.external_user_id == phone,
            Conversation.organization_id == channel.organization_id,
        )
        .first()
    )


def _get_or_create_conversation(db: Session, channel: Channel, phone: str) -> Conversation:
    conv = find_conversation(db, channel, phone)
    if conv is None:
        conv = Conversation(
            organization_id=channel.organization_id,
            bot_id=channel.bot_id,
            channel_id=channel.id,
            external_user_id=phone,
            status="bot",
            last_message_at=datetime.utcnow(),
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
    return conv


def record_turn(
    db: Session, channel: Channel, phone: str, question: str, answer: str, tokens: int,
) -> Conversation:
    """Find-or-create the customer's conversation and append the user + assistant messages."""
    conv = _get_or_create_conversation(db, channel, phone)
    db.add(Message(organization_id=channel.organization_id, conversation_id=conv.id,
                   role="user", content=question, tokens=0))
    db.add(Message(organization_id=channel.organization_id, conversation_id=conv.id,
                   role="assistant", content=answer, tokens=tokens))
    conv.last_message_at = datetime.utcnow()
    db.commit()
    db.refresh(conv)
    return conv


def record_user_message(db: Session, channel: Channel, phone: str, content: str) -> Conversation:
    """Store just the inbound message — used when a human already owns this
    thread, so the bot must not also generate an answer."""
    conv = _get_or_create_conversation(db, channel, phone)
    db.add(Message(organization_id=channel.organization_id, conversation_id=conv.id,
                   role="user", content=content, tokens=0))
    conv.last_message_at = datetime.utcnow()
    db.commit()
    db.refresh(conv)
    return conv


def flag_needs_human(db: Session, channel: Channel, phone: str) -> Conversation | None:
    """Put the conversation in the agent queue. Never downgrades a chat a human
    already owns. WhatsApp has no "talk to a human" button like the widget does,
    so this fires automatically whenever the bot can't answer from the KB."""
    conv = find_conversation(db, channel, phone)
    if conv is None:
        return None
    if conv.status in ("bot", "resolved"):
        conv.status = "needs_human"
        conv.needs_human_at = datetime.utcnow()
        db.commit()
        db.refresh(conv)
    return conv


def is_paused(db: Session, channel: Channel, phone: str) -> bool:
    """True if a human already owns this conversation — the bot must not answer."""
    conv = find_conversation(db, channel, phone)
    return bool(conv and conv.status in ("needs_human", "human"))

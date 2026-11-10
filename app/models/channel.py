from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, JSON

from app.db.session import Base


class Channel(Base):
    """A messaging channel a bot answers on. MVP: WhatsApp + website widget.

    Per-tenant WhatsApp credentials (phone_number_id + access token) live here,
    NOT in .env — each organization connects its own WABA via embedded signup.
    Website-widget config (kind='widget') also lives here so Conversation.channel_id
    scopes the handoff inbox uniformly across channels.
    """

    __tablename__ = "channels"

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), index=True, nullable=False)
    bot_id = Column(Integer, ForeignKey("bots.id"), index=True, nullable=False)

    # whatsapp | widget (telegram later)
    kind = Column(String, default="whatsapp", nullable=False)

    # Which WhatsApp gateway owns this channel: fonnte | evolution. Chosen at
    # first connect from the WHATSAPP_DEFAULT_PROVIDER setting, then fixed.
    provider = Column(String, default="fonnte", nullable=False, server_default="fonnte")

    # WhatsApp-specific (Fonnte gateway — see app/core/fonnte.py)
    # waba_id unused for Fonnte (no WABA concept); kept for a possible future
    # Meta-direct provider.
    waba_id = Column(String, nullable=True)
    # Fonnte's device identifier — an arbitrary unique string assigned at
    # creation, NOT necessarily the real linked WhatsApp number. Inbound webhook
    # routing does NOT depend on this (see webhook_secret below); it's here for
    # reference/display only.
    phone_number_id = Column(String, index=True, nullable=True)
    # Fonnte DEVICE token (not the platform account token), Fernet-encrypted at
    # rest — see app/core/crypto.py. Decrypt only at the point of use.
    access_token = Column(String, nullable=True)

    # disconnected | pending_qr | connected — distinct from is_active (admin on/off).
    connection_status = Column(String, default="disconnected", nullable=False)
    # Random token embedded in this channel's webhook URL path. Fonnte doesn't
    # sign webhook payloads, so routing/auth for inbound webhooks is entirely
    # URL-based: /public/whatsapp/fonnte/{channel_id}/{webhook_secret}/...
    webhook_secret = Column(String, nullable=True)

    # Website-widget-specific (kind='widget')
    # Public embed key — a selector in the tenant's page HTML, NOT a secret.
    public_key = Column(String, unique=True, index=True, nullable=True)
    # Normalized "scheme://host[:port]" origins allowed to embed (browser hygiene).
    allowed_origins = Column(JSON, nullable=True)
    # Appearance blob the widget bundle renders (title, greeting, colors, position).
    appearance = Column(JSON, nullable=True)
    # Per-day message cap for this widget (blast-radius limit); null -> platform default.
    widget_daily_message_cap = Column(Integer, nullable=True)

    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

"""Provider dispatch for WhatsApp channels (Fonnte | Evolution).

Each Channel records its own `provider`, so Fonnte numbers keep working while
new connections default to WHATSAPP_DEFAULT_PROVIDER (Evolution).
"""
from __future__ import annotations

from app.core import config_store, crypto, evolution, fonnte
from app.models.channel import Channel

PROVIDERS = ("fonnte", "evolution")


def default_provider() -> str:
    value = config_store.get("WHATSAPP_DEFAULT_PROVIDER", "fonnte").strip().lower()
    return value if value in PROVIDERS else "fonnte"


def instance_name(channel: Channel) -> str:
    """Evolution instance id for a channel (stored in phone_number_id)."""
    return channel.phone_number_id or f"ats-{channel.id}"


def normalize_target(provider: str, target: str) -> str:
    """Evolution wants bare digits; strip any JID suffix / '+'."""
    if provider == "evolution":
        return target.split("@")[0].lstrip("+")
    return target


def send_text(channel: Channel, target: str, message: str) -> None:
    """Send a WhatsApp text via the channel's own provider. Raises on failure."""
    if channel.provider == "evolution":
        evolution.send_message(instance_name(channel), normalize_target("evolution", target), message)
    else:
        fonnte.send_message(crypto.decrypt(channel.access_token), target, message)


def disconnect(channel: Channel) -> None:
    if channel.provider == "evolution":
        evolution.disconnect_instance(instance_name(channel))
    else:
        fonnte.disconnect_device(crypto.decrypt(channel.access_token))

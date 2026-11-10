"""Client for Evolution API v2 (self-hosted WhatsApp gateway — doc.evolution-api.com).

One platform-owned Evolution server; a single global API key creates an
"instance" (a QR-linked WhatsApp session) per client bot. Every call here uses
the global key, so the per-instance token Evolution returns is not needed.

Like Fonnte, inbound webhooks are unsigned: routing/auth is the channel id +
secret embedded in the webhook URL (see app/api/v1/endpoints/whatsapp.py).
"""
from __future__ import annotations

import httpx

from app.core import config_store

_TIMEOUT = 30.0
# Webhook events we need: inbound messages and link/unlink state changes.
WEBHOOK_EVENTS = ["MESSAGES_UPSERT", "CONNECTION_UPDATE"]


class EvolutionError(Exception):
    """An Evolution API call failed (HTTP error or unusable response)."""


class AlreadyConnected(EvolutionError):
    """The instance is already linked — nothing to show a QR for."""


def _base() -> str:
    url = config_store.get("EVOLUTION_BASE_URL").rstrip("/")
    if not url:
        raise EvolutionError("EVOLUTION_BASE_URL is not configured — set it in Settings.")
    return url


def _key() -> str:
    key = config_store.get("EVOLUTION_API_KEY")
    if not key:
        raise EvolutionError("EVOLUTION_API_KEY is not configured — set it in Settings.")
    return key


def is_configured() -> bool:
    return bool(config_store.get("EVOLUTION_BASE_URL") and config_store.get("EVOLUTION_API_KEY"))


def _request(method: str, path: str, json: dict | None = None) -> dict:
    try:
        resp = httpx.request(method, f"{_base()}{path}", headers={"apikey": _key()},
                             json=json, timeout=_TIMEOUT)
    except httpx.HTTPError as e:
        raise EvolutionError(f"Could not reach Evolution API: {e}") from e
    if resp.status_code >= 400:
        raise EvolutionError(f"Evolution {method} {path} failed ({resp.status_code}): {resp.text[:200]}")
    try:
        return resp.json()
    except ValueError:
        return {}


def webhook_config(url: str) -> dict:
    return {"url": url, "byEvents": False, "base64": False, "events": WEBHOOK_EVENTS}


def create_instance(name: str, webhook_url: str) -> dict:
    """Create the instance with its webhook already wired. The response carries
    a first QR under `qrcode.base64` and the instance token under `hash`."""
    return _request("POST", "/instance/create", {
        "instanceName": name,
        "integration": "WHATSAPP-BAILEYS",
        "qrcode": True,
        "webhook": webhook_config(webhook_url),
    })


def set_webhook(name: str, webhook_url: str) -> dict:
    return _request("POST", f"/webhook/set/{name}", {"webhook": webhook_config(webhook_url)})


def connection_state(name: str) -> str:
    """Returns Evolution's state: open | connecting | close."""
    body = _request("GET", f"/instance/connectionState/{name}")
    return ((body.get("instance") or {}).get("state") or body.get("state") or "close")


def get_qr(name: str) -> dict:
    """Returns {"base64": "data:image/png;base64,..."}; raises AlreadyConnected if linked."""
    if connection_state(name) == "open":
        raise AlreadyConnected("already connected")
    body = _request("GET", f"/instance/connect/{name}")
    if not body.get("base64"):
        raise EvolutionError("Evolution returned no QR code yet — try again in a few seconds.")
    return body


def send_message(name: str, target: str, message: str) -> dict:
    return _request("POST", f"/message/sendText/{name}", {"number": target, "text": message})


def disconnect_instance(name: str) -> dict:
    try:
        return _request("DELETE", f"/instance/logout/{name}")
    except EvolutionError as e:
        if "404" in str(e) or "not connected" in str(e).lower():
            return {"detail": "already disconnected"}
        raise

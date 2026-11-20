"""ElevenLabs Agents REST client: agents, workspace secrets, SIP numbers, post-call webhooks.

Each AI phone line gets its own agent whose "custom LLM" is our per-line endpoint
(/api/v1/voice/lines/{id}/chat/completions), authenticated by a per-line secret
stored as an ElevenLabs workspace secret. See app/core/voice_lines.py.
"""
from __future__ import annotations

import hashlib
import hmac
import time

import httpx

from app.core import config_store

API_BASE = "https://api.elevenlabs.io"
TIMEOUT = 20
DEFAULT_TTS_MODEL = "eleven_v4_turbo"
# How old a signed webhook may be (ElevenLabs' SDK allows 30 minutes).
WEBHOOK_TOLERANCE_SECS = 30 * 60


class ElevenLabsError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def api_key() -> str:
    return config_store.get("ELEVENLABS_API_KEY").strip()


def tts_model() -> str:
    return config_store.get("VOICE_TTS_MODEL").strip() or DEFAULT_TTS_MODEL


def _request(method: str, path: str, *, json: dict | None = None, params: dict | None = None) -> dict:
    key = api_key()
    if not key:
        raise ElevenLabsError("ElevenLabs API key is not set.")
    try:
        resp = httpx.request(method, API_BASE + path, json=json, params=params, timeout=TIMEOUT,
                             headers={"xi-api-key": key, "Accept": "application/json"})
    except httpx.HTTPError as exc:
        raise ElevenLabsError(f"Could not reach ElevenLabs ({exc.__class__.__name__}).") from exc
    if resp.status_code >= 400:
        raise ElevenLabsError(_error_text(resp), resp.status_code)
    if not resp.content:
        return {}
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _request_any(method: str, path: str) -> object:
    """Like _request, for endpoints that answer with a JSON list."""
    key = api_key()
    if not key:
        raise ElevenLabsError("ElevenLabs API key is not set.")
    try:
        resp = httpx.request(method, API_BASE + path, timeout=TIMEOUT, headers={"xi-api-key": key, "Accept": "application/json"})
    except httpx.HTTPError as exc:
        raise ElevenLabsError(f"Could not reach ElevenLabs ({exc.__class__.__name__}).") from exc
    if resp.status_code >= 400:
        raise ElevenLabsError(_error_text(resp), resp.status_code)
    try:
        return resp.json()
    except ValueError:
        return None


def _error_text(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("status")
    elif isinstance(detail, list):
        detail = "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)
    return f"ElevenLabs answered HTTP {resp.status_code}" + (f": {detail}" if detail else ".")


# ===== Secrets =====


def create_secret(name: str, value: str) -> str:
    secret_id = _request("POST", "/v1/convai/secrets", json={"type": "new", "name": name, "value": value}).get("secret_id")
    if not secret_id:
        raise ElevenLabsError("ElevenLabs did not return a secret id.")
    return secret_id


def delete_secret(secret_id: str) -> None:
    _request("DELETE", f"/v1/convai/secrets/{secret_id}")


# ===== Agents =====


def create_agent(body: dict) -> str:
    agent_id = _request("POST", "/v1/convai/agents/create", json=body).get("agent_id")
    if not agent_id:
        raise ElevenLabsError("ElevenLabs did not return an agent id.")
    return agent_id


def update_agent(agent_id: str, body: dict) -> None:
    _request("PATCH", f"/v1/convai/agents/{agent_id}", json=body)


def delete_agent(agent_id: str) -> None:
    _request("DELETE", f"/v1/convai/agents/{agent_id}")


# ===== Phone numbers (SIP trunk) =====


def import_sip_number(label: str, phone_number: str, agent_id: str | None = None) -> str:
    """Inbound from Telnyx: no digest auth (Telnyx sends none), encryption allowed, any caller."""
    body: dict = {
        "provider": "sip_trunk",
        "label": label,
        "phone_number": phone_number,
        "inbound_trunk_config": {"media_encryption": "allowed"},
    }
    if agent_id:
        body["agent_id"] = agent_id
    number_id = _request("POST", "/v1/convai/phone-numbers", json=body).get("phone_number_id")
    if not number_id:
        raise ElevenLabsError("ElevenLabs did not return a phone number id.")
    return number_id


def find_phone_number(phone_number: str) -> str | None:
    """The id of a number already imported into the workspace, if any."""
    body = _request_any("GET", "/v1/convai/phone-numbers")
    rows = body if isinstance(body, list) else (body.get("phone_numbers") or []) if isinstance(body, dict) else []
    for row in rows:
        if isinstance(row, dict) and row.get("phone_number") == phone_number:
            return row.get("phone_number_id")
    return None


def assign_agent(phone_number_id: str, agent_id: str) -> None:
    _request("PATCH", f"/v1/convai/phone-numbers/{phone_number_id}", json={"agent_id": agent_id})


def delete_phone_number(phone_number_id: str) -> None:
    _request("DELETE", f"/v1/convai/phone-numbers/{phone_number_id}")


# ===== Voices =====


def get_voice(voice_id: str) -> dict:
    """{voice_id, name, preview_url, description, labels} for a voice in the workspace library."""
    v = _request("GET", f"/v1/voices/{voice_id}")
    labels = v.get("labels") or {}
    return {
        "voice_id": v.get("voice_id") or voice_id,
        "name": v.get("name") or voice_id,
        "preview_url": v.get("preview_url") or "",
        "description": v.get("description") or ", ".join(str(x) for x in labels.values() if x),
    }


# ===== Post-call webhooks =====


def create_post_call_webhook(name: str, url: str) -> tuple[str, str]:
    """Workspace webhook with HMAC auth. Returns (webhook_id, webhook_secret)."""
    body = {"settings": {"auth_type": "hmac", "name": name, "webhook_url": url}}
    resp = _request("POST", "/v1/workspace/webhooks", json=body)
    if not resp.get("webhook_id") or not resp.get("webhook_secret"):
        raise ElevenLabsError("ElevenLabs did not return the webhook id and secret.")
    return resp["webhook_id"], resp["webhook_secret"]


def verify_webhook(raw_body: bytes, signature_header: str, secret: str, now: float | None = None) -> bool:
    """`ElevenLabs-Signature: t=<unix>,v0=<hex hmac-sha256 of "<t>.<body>">`, as ElevenLabs' SDKs check it."""
    if not secret or not signature_header:
        return False
    parts = dict(p.strip().split("=", 1) for p in signature_header.split(",") if "=" in p)
    ts, given = parts.get("t", ""), parts.get("v0", "")
    if not ts.isdigit() or not given:
        return False
    if abs((now if now is not None else time.time()) - int(ts)) > WEBHOOK_TOLERANCE_SECS:
        return False
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, given)


def check_connection() -> None:
    _request("GET", "/v1/convai/agents", params={"page_size": 1})

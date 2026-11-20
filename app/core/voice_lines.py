"""AI phone lines: a Telnyx number per bot, answered by an ElevenLabs agent that uses
the bot's knowledge base (our per-line custom-LLM endpoint) to reply.

Lifecycle (VoiceLine.status):
  draft        subscriber is filling in the details
  submitted    details + proof of address sent; waiting for a platform admin
  ordered      admin ordered a number; Telnyx is vetting the documents
  info_needed  Telnyx (or the admin) needs something else from the subscriber
  number_ready Telnyx approved; the number is ours and routed to ElevenLabs
  active       ElevenLabs agent created and attached: calls are answered
  released     number given back; failed: the order failed

Ordering costs non-refundable fees, so a person (platform admin) places each order.
Everything after Telnyx approval is automatic once the admin presses Activate.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
from datetime import datetime

from sqlalchemy.orm import Session

from app.core import billing, config_store, elevenlabs, storage, telnyx
from app.core.settings import settings
from app.models.bot import Bot
from app.models.channel import Channel
from app.models.conversation import Conversation, Message
from app.models.subscription import Subscription
from app.models.voice_line import VoiceCall, VoiceLine

logger = logging.getLogger(__name__)

DEFAULT_TIMEZONE = "Australia/Sydney"
DEFAULT_MAX_CALL_SECONDS = 600
MAX_CALL_SECONDS_LIMIT = 1800
# Lines in these states can still have their regulatory details edited by the subscriber.
EDITABLE_STATUSES = {"draft", "info_needed"}
E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")
AU_STATES = {"ACT", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"}


class VoiceLineError(Exception):
    """A refusal with a message safe to show the person who asked."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


# ===== Configuration =====


def enabled() -> bool:
    return config_store.get("VOICE_ENABLED", "false").strip().lower() == "true"


def catalog() -> list[dict]:
    """The voices subscribers may choose: [{voice_id, name, description, preview_url}]."""
    try:
        items = json.loads(config_store.get("VOICE_CATALOG") or "[]")
    except ValueError:
        return []
    return [v for v in items if isinstance(v, dict) and v.get("voice_id")] if isinstance(items, list) else []


def default_voice_id() -> str | None:
    voices = catalog()
    return voices[0]["voice_id"] if voices else None


def llm_url(line: VoiceLine) -> str:
    """ElevenLabs appends /chat/completions to the custom-LLM URL."""
    return f"{settings.API_BASE_URL.rstrip('/')}{settings.API_V1_STR}/voice/lines/{line.id}"


def webhook_url() -> str:
    return f"{settings.API_BASE_URL.rstrip('/')}{settings.API_V1_STR}/voice/webhooks/elevenlabs"


# ===== Plan allowance =====


def allowance(db: Session, organization_id: int) -> dict:
    """{plan_allows, minutes_included, seconds_used} for the org's current package."""
    sub = billing.get_or_create_subscription(db, organization_id)
    pkg = billing.package_for(db, sub.plan)
    return {
        "plan_allows": bool(pkg and pkg.voice_enabled),
        "minutes_included": int(pkg.monthly_voice_minutes or 0) if pkg else 0,
        "seconds_used": int(sub.voice_seconds_used or 0),
        "subscription_active": sub.status == "active",
    }


def can_take_calls(db: Session, line: VoiceLine) -> bool:
    """Active line, active subscription, a plan with voice, and minutes left."""
    if line.status != "active" or not enabled():
        return False
    a = allowance(db, line.organization_id)
    return a["plan_allows"] and a["subscription_active"] and a["seconds_used"] < a["minutes_included"] * 60


# ===== Lines =====


def line_for_bot(db: Session, bot: Bot) -> VoiceLine:
    line = db.query(VoiceLine).filter(VoiceLine.bot_id == bot.id, VoiceLine.organization_id == bot.organization_id).first()
    if line is None:
        line = VoiceLine(organization_id=bot.organization_id, bot_id=bot.id, status="draft",
                         timezone=DEFAULT_TIMEZONE, max_call_seconds=DEFAULT_MAX_CALL_SECONDS)
        db.add(line)
        db.commit()
        db.refresh(line)
    return line


def submit_request(db: Session, line: VoiceLine, details: dict, document: tuple[bytes, str, str] | None) -> VoiceLine:
    """Save the regulatory details (and proof of address) and hand the line to an admin."""
    if line.status not in EDITABLE_STATUSES:
        raise VoiceLineError("This line's details are with Airevo for review and can't be changed now.", 409)
    clean = {k: (v or "").strip() for k, v in details.items()}
    missing = [label for key, label in (
        ("business_name", "business name"), ("contact_name", "contact name"), ("contact_phone", "contact phone"),
        ("street_address", "street address"), ("locality", "suburb"), ("administrative_area", "state"),
        ("postal_code", "postcode"),
    ) if not clean.get(key)]
    if missing:
        raise VoiceLineError("Please fill in: " + ", ".join(missing) + ".", 422)
    if clean["administrative_area"].upper() not in AU_STATES:
        raise VoiceLineError("State must be one of " + ", ".join(sorted(AU_STATES)) + ".", 422)
    if not re.fullmatch(r"\d{4}", clean["postal_code"]):
        raise VoiceLineError("Postcode must be 4 digits.", 422)
    phone = normalize_au_phone(clean["contact_phone"])
    if not phone:
        raise VoiceLineError("Contact phone must be an Australian number, e.g. +61 4xx xxx xxx.", 422)
    if document is None and not line.document_key:
        raise VoiceLineError("Please attach a proof-of-address document.", 422)

    if document is not None:
        data, filename, mime = document
        key = f"voice-docs/{line.organization_id}/{line.id}/{secrets.token_hex(8)}-{_safe_name(filename)}"
        storage.put_bytes(db, data, key, mime)
        line.document_key, line.document_filename, line.document_mime = key, filename, mime

    line.business_name = clean["business_name"]
    line.contact_name = clean["contact_name"]
    line.contact_phone = phone
    line.street_address = clean["street_address"]
    line.locality = clean["locality"]
    line.administrative_area = clean["administrative_area"].upper()
    line.postal_code = clean["postal_code"]
    line.preferred_locality = clean.get("preferred_locality") or None
    line.status = "submitted"
    line.status_message = None
    line.submitted_at = datetime.utcnow()
    db.commit()
    db.refresh(line)
    return line


def update_settings(db: Session, line: VoiceLine, changes: dict) -> VoiceLine:
    """Caller experience. Applied to the live agent straight away when the line is active."""
    if "voice_id" in changes and changes["voice_id"] is not None:
        if changes["voice_id"] not in {v["voice_id"] for v in catalog()}:
            raise VoiceLineError("Choose one of the listed voices.", 422)
        line.voice_id = changes["voice_id"]
    if "first_message" in changes and changes["first_message"] is not None:
        line.first_message = changes["first_message"].strip()[:500] or None
    if "timezone" in changes and changes["timezone"] is not None:
        tz = changes["timezone"].strip()
        if tz and not re.fullmatch(r"[A-Za-z_]+(/[A-Za-z_]+){0,2}", tz):
            raise VoiceLineError("Timezone must look like Australia/Sydney.", 422)
        line.timezone = tz or DEFAULT_TIMEZONE
    if "transfer_number" in changes and changes["transfer_number"] is not None:
        raw = changes["transfer_number"].strip()
        number = normalize_au_phone(raw) if raw else ""
        if raw and not number:
            raise VoiceLineError("Transfer number must be a phone number like +61 2 xxxx xxxx.", 422)
        line.transfer_number = number or None
    if "max_call_seconds" in changes and changes["max_call_seconds"] is not None:
        line.max_call_seconds = max(60, min(int(changes["max_call_seconds"]), MAX_CALL_SECONDS_LIMIT))
    db.commit()
    if line.status == "active" and line.el_agent_id:
        bot = db.query(Bot).filter(Bot.id == line.bot_id).first()
        try:
            elevenlabs.update_agent(line.el_agent_id, agent_body(line, bot, line.el_secret_id))
        except elevenlabs.ElevenLabsError as exc:
            raise VoiceLineError(f"Saved, but the live line couldn't be updated: {exc.message}", 502)
    db.refresh(line)
    return line


def normalize_au_phone(raw: str) -> str:
    """E.164 for an Australian number given as +61…, 61…, or 0…; any other +E.164 kept. '' if invalid."""
    digits = re.sub(r"[^\d+]", "", raw or "")
    if digits.startswith("+"):
        number = digits
    elif digits.startswith("61"):
        number = "+" + digits
    elif digits.startswith("0"):
        number = "+61" + digits[1:]
    else:
        return ""
    return number if E164_RE.match(number) else ""


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name or "document")[-80:] or "document"


# ===== Ordering (admin) =====


def _requirement_values(db: Session, line: VoiceLine, reqs: list[dict]) -> list[dict]:
    """Answer each Telnyx requirement by its field type."""
    ref = f"line-{line.id}"
    values: list[dict] = []
    address_id = document_id = None
    for req in reqs:
        kind = (req.get("field_type") or "").lower()
        if kind in ("address", "address_id"):
            if address_id is None:
                first, _, last = (line.contact_name or "").partition(" ")
                address_id = telnyx.create_address(
                    first_name=first or line.contact_name or "", last_name=last or first or "",
                    business_name=line.business_name or "", street_address=line.street_address or "",
                    locality=line.locality or "", administrative_area=line.administrative_area or "",
                    postal_code=line.postal_code or "", country_code=line.country_code or "AU",
                    phone_number=line.contact_phone or "", customer_reference=ref,
                )
            values.append({"requirement_id": req["id"], "field_value": address_id})
        elif kind == "document":
            if document_id is None:
                if not line.document_key:
                    raise VoiceLineError("The subscriber hasn't uploaded a proof-of-address document.", 409)
                document_id = telnyx.upload_document(storage.get_bytes(db, line.document_key),
                                                     line.document_filename or "proof-of-address.pdf", ref)
            values.append({"requirement_id": req["id"], "field_value": document_id})
        elif kind == "textual":
            values.append({"requirement_id": req["id"], "field_value": contact_text(line)})
        else:
            raise VoiceLineError(f"Telnyx asks for '{req.get('name')}' ({kind}), which this form can't supply yet. "
                                 "Complete this order in the Telnyx portal.", 409)
    return values


def contact_text(line: VoiceLine) -> str:
    return f"Contact: {line.contact_name} | Business: {line.business_name} | Phone: {line.contact_phone}"


def order_number(db: Session, line: VoiceLine, phone_number: str) -> VoiceLine:
    """Buy `phone_number` (from a fresh search) with the subscriber's details attached."""
    if line.status not in ("submitted", "info_needed", "failed"):
        raise VoiceLineError(f"A line that is '{line.status}' can't be ordered.", 409)
    connection = telnyx.sip_connection_id()
    if not connection:
        raise VoiceLineError("Set the Telnyx SIP connection ID in Voice settings first.", 409)
    country = line.country_code or "AU"
    try:
        reqs = telnyx.ordering_requirements(country, "local")
        group_id = telnyx.create_requirement_group(country, "local", _requirement_values(db, line, reqs), f"line-{line.id}") if reqs else None
        order = telnyx.create_number_order(phone_number, connection_id=connection, requirement_group_id=group_id,
                                           customer_reference=f"aichatsupport-line-{line.id}")
    except telnyx.TelnyxError as exc:
        raise VoiceLineError(exc.message, 502)
    line.phone_number = phone_number
    line.telnyx_requirement_group_id = group_id
    line.telnyx_number_order_id = order.get("id")
    line.status = "ordered"
    line.status_message = None
    _apply_order(line, order)
    db.commit()
    db.refresh(line)
    return line


def adopt_number(db: Session, line: VoiceLine, phone_number: str) -> VoiceLine:
    """Use a number already on the platform's Telnyx account (bought in the portal) instead of ordering one."""
    if line.status not in ("submitted", "info_needed", "failed"):
        raise VoiceLineError(f"A line that is '{line.status}' can't take a number.", 409)
    number = normalize_au_phone(phone_number)
    if not number:
        raise VoiceLineError("Enter the number in full, e.g. +61 2 8390 2245.", 422)
    if db.query(VoiceLine).filter(VoiceLine.phone_number == number, VoiceLine.id != line.id,
                                  VoiceLine.status != "released").first():
        raise VoiceLineError("That number already belongs to another line.", 409)
    connection = telnyx.sip_connection_id()
    try:
        record = telnyx.find_phone_number(number)
        if record is None:
            raise VoiceLineError("That number isn't on the Telnyx account.", 404)
        line.telnyx_phone_number_id = str(record.get("id"))
        if connection and str(record.get("connection_id") or "") != connection:
            telnyx.assign_connection(line.telnyx_phone_number_id, connection)
    except telnyx.TelnyxError as exc:
        raise VoiceLineError(exc.message, 502)
    line.phone_number = number
    line.telnyx_number_order_id = None
    _apply_number_status(line, record.get("status"))
    db.commit()
    db.refresh(line)
    return line


def _apply_number_status(line: VoiceLine, status: str | None) -> None:
    """Map a Telnyx phone-number status onto the line (adopted numbers have no order to follow)."""
    line.telnyx_status = status
    if status == "active":
        line.status = "number_ready"
        line.status_message = None
    elif status in ("requirement-info-pending", "requirement-info-exception"):
        line.status = "info_needed"
        line.status_message = line.status_message or "The carrier needs more information about your business or address."
    elif status in ("purchase-failed", "deleted"):
        line.status = "failed"
        line.status_message = f"The number is {status} at Telnyx."
    else:
        line.status = "ordered"


def sync(db: Session, line: VoiceLine) -> VoiceLine:
    """Refresh an ordered (or adopted) line from Telnyx."""
    if line.status not in ("ordered", "info_needed"):
        return line
    if not line.telnyx_number_order_id:
        if not line.phone_number:
            return line
        try:
            record = telnyx.find_phone_number(line.phone_number)
        except telnyx.TelnyxError as exc:
            raise VoiceLineError(exc.message, 502)
        if record is not None:
            _apply_number_status(line, record.get("status"))
            line.telnyx_phone_number_id = str(record.get("id"))
        db.commit()
        db.refresh(line)
        return line
    try:
        order = telnyx.get_number_order(line.telnyx_number_order_id)
        _apply_order(line, order)
        if line.status == "number_ready" and not line.telnyx_phone_number_id:
            number = telnyx.find_phone_number(line.phone_number or "")
            if number:
                line.telnyx_phone_number_id = str(number.get("id"))
                connection = telnyx.sip_connection_id()
                if connection and str(number.get("connection_id") or "") != connection:
                    telnyx.assign_connection(line.telnyx_phone_number_id, connection)
    except telnyx.TelnyxError as exc:
        raise VoiceLineError(exc.message, 502)
    db.commit()
    db.refresh(line)
    return line


def _apply_order(line: VoiceLine, order: dict) -> None:
    numbers = order.get("phone_numbers") or []
    entry = next((n for n in numbers if n.get("phone_number") == line.phone_number), numbers[0] if numbers else {})
    line.telnyx_status = entry.get("requirements_status") or entry.get("status") or order.get("status")
    status = order.get("status")
    if status == "success" or entry.get("status") == "success":
        line.status = "number_ready"
        line.status_message = None
    elif status == "failure" or entry.get("status") == "failure":
        line.status = "failed"
        line.status_message = "Telnyx could not complete the number order."
    elif line.telnyx_status in ("requirement-info-pending", "requirement-info-exception"):
        line.status = "info_needed"
        line.status_message = line.status_message or "Telnyx needs more information about your business or address."
    else:
        line.status = "ordered"


# ===== Activation (admin) =====


def new_llm_secret() -> tuple[str, str]:
    plain = "vls_" + secrets.token_urlsafe(32)
    return plain, hash_secret(plain)


def hash_secret(plain: str) -> str:
    return hashlib.sha256(plain.encode()).hexdigest()


def agent_body(line: VoiceLine, bot: Bot | None, secret_id: str | None) -> dict:
    business = line.business_name or (bot.name if bot else "us")
    prompt: dict = {
        # Our endpoint answers from the bot's own prompt + knowledge base; this text is not used.
        "prompt": f"You answer phone calls for {business}.",
        "llm": "custom-llm",
        "custom_llm": {"url": llm_url(line), "model_id": "aichatsupport-rag"},
        "timezone": line.timezone or DEFAULT_TIMEZONE,
    }
    if secret_id:
        prompt["custom_llm"]["api_key"] = {"secret_id": secret_id}
    if line.transfer_number:
        prompt["built_in_tools"] = {"transfer_to_number": {
            "type": "system", "name": "transfer_to_number",
            "description": "Transfer the caller to a person at the business when they ask for one.",
            "params": {
                "system_tool_type": "transfer_to_number",
                "transfers": [],
                "transfer_rules": [{
                    "condition": "When the caller asks to speak to a person.",
                    "provider_configs": [{"type": "sip_trunk", "config": {
                        "type": "sip_cold_conference_transfer",
                        "transfer_destination": {"type": "phone", "destination": line.transfer_number},
                    }}],
                }],
            },
        }}
    body: dict = {
        "name": f"{business} · line {line.id}"[:100],
        "tags": ["aichatsupport", f"line-{line.id}"],
        "conversation_config": {
            "agent": {
                "first_message": line.first_message or f"Hi, thanks for calling {business}. How can I help you today?",
                "language": "en",
                "prompt": prompt,
            },
            "tts": {"voice_id": line.voice_id or default_voice_id(), "model_id": elevenlabs.tts_model()},
            "asr": {"keywords": [w for w in {business, bot.name if bot else ""} if w][:10]},
            "conversation": {"max_duration_seconds": line.max_call_seconds or DEFAULT_MAX_CALL_SECONDS},
        },
    }
    if not body["conversation_config"]["tts"]["voice_id"]:
        del body["conversation_config"]["tts"]["voice_id"]
    webhook_id = config_store.get("ELEVENLABS_WEBHOOK_ID").strip()
    if webhook_id:
        body["platform_settings"] = {"workspace_overrides": {"webhooks": {"post_call_webhook_id": webhook_id, "events": ["transcript"]}}}
    return body


def activate(db: Session, line: VoiceLine) -> VoiceLine:
    """Create (or refresh) the ElevenLabs side. Each step is saved as it completes, so a
    retry after a failure picks up where it stopped."""
    if line.status not in ("number_ready", "active"):
        raise VoiceLineError("Activate once Telnyx has approved the number.", 409)
    bot = db.query(Bot).filter(Bot.id == line.bot_id).first()
    try:
        if not line.el_secret_id:
            plain, digest = new_llm_secret()
            line.el_secret_id = elevenlabs.create_secret(f"aichatsupport-line-{line.id}", plain)
            line.llm_secret_hash = digest
            db.commit()
        if not line.el_agent_id:
            line.el_agent_id = elevenlabs.create_agent(agent_body(line, bot, line.el_secret_id))
            db.commit()
        else:
            elevenlabs.update_agent(line.el_agent_id, agent_body(line, bot, line.el_secret_id))
        if not line.el_phone_number_id:
            # A number imported by hand earlier is reused rather than imported twice.
            existing = elevenlabs.find_phone_number(line.phone_number or "")
            if existing:
                line.el_phone_number_id = existing
                db.commit()
                elevenlabs.assign_agent(existing, line.el_agent_id)
            else:
                line.el_phone_number_id = elevenlabs.import_sip_number(
                    f"{line.business_name or 'line'} ({line.id})"[:50], line.phone_number or "", line.el_agent_id)
                db.commit()
        else:
            elevenlabs.assign_agent(line.el_phone_number_id, line.el_agent_id)
    except elevenlabs.ElevenLabsError as exc:
        raise VoiceLineError(exc.message, 502)
    get_or_create_voice_channel(db, line)
    line.status = "active"
    line.status_message = None
    line.activated_at = line.activated_at or datetime.utcnow()
    db.commit()
    db.refresh(line)
    return line


def release(db: Session, line: VoiceLine, release_number: bool) -> VoiceLine:
    """Take the line down. Releasing the Telnyx number stops its monthly charge and can't be undone."""
    def quietly(fn, *args) -> None:
        try:
            fn(*args)
        except elevenlabs.ElevenLabsError as exc:
            if exc.status != 404:
                raise VoiceLineError(exc.message, 502)

    if line.el_phone_number_id:
        quietly(elevenlabs.delete_phone_number, line.el_phone_number_id)
        line.el_phone_number_id = None
    if line.el_agent_id:
        quietly(elevenlabs.delete_agent, line.el_agent_id)
        line.el_agent_id = None
    if line.el_secret_id:
        quietly(elevenlabs.delete_secret, line.el_secret_id)
        line.el_secret_id = None
        line.llm_secret_hash = None
    if release_number and line.telnyx_phone_number_id:
        try:
            telnyx.release_number(line.telnyx_phone_number_id)
        except telnyx.TelnyxError as exc:
            if exc.status != 404:
                raise VoiceLineError(exc.message, 502)
        line.telnyx_phone_number_id = None
        line.status = "released"
    else:
        line.status = "number_ready" if line.telnyx_phone_number_id else "released"
    db.commit()
    db.refresh(line)
    return line


# ===== Calls (ElevenLabs post-call webhook) =====


def get_or_create_voice_channel(db: Session, line: VoiceLine) -> Channel:
    ch = db.query(Channel).filter(Channel.bot_id == line.bot_id, Channel.kind == "voice",
                                  Channel.organization_id == line.organization_id).first()
    if ch is None:
        ch = Channel(organization_id=line.organization_id, bot_id=line.bot_id, kind="voice", provider="elevenlabs",
                     connection_status="connected", phone_number_id=line.phone_number)
        db.add(ch)
        db.commit()
        db.refresh(ch)
    return ch


def record_call(db: Session, data: dict) -> VoiceCall | None:
    """Store a finished call as a conversation and count its minutes. Idempotent on the
    ElevenLabs conversation id; calls for agents we don't know are ignored."""
    call_id = str(data.get("conversation_id") or "")
    agent_id = str(data.get("agent_id") or "")
    if not call_id or not agent_id:
        return None
    existing = db.query(VoiceCall).filter(VoiceCall.id == call_id).first()
    if existing is not None:
        return existing
    line = db.query(VoiceLine).filter(VoiceLine.el_agent_id == agent_id).first()
    if line is None:
        return None

    meta = data.get("metadata") or {}
    phone = meta.get("phone_call") or {}
    caller = phone.get("external_number") if isinstance(phone, dict) else None
    duration = max(0, int(meta.get("call_duration_secs") or 0))
    started = datetime.utcfromtimestamp(int(meta["start_time_unix_secs"])) if meta.get("start_time_unix_secs") else datetime.utcnow()

    channel = get_or_create_voice_channel(db, line)
    convo = Conversation(organization_id=line.organization_id, bot_id=line.bot_id, channel_id=channel.id,
                         external_user_id=caller or f"call-{call_id[:12]}", contact_phone=caller, status="resolved",
                         resolved_at=datetime.utcnow(), last_message_at=datetime.utcnow(), created_at=started)
    db.add(convo)
    db.flush()
    for turn in data.get("transcript") or []:
        text = (turn.get("message") or "").strip() if isinstance(turn, dict) else ""
        if not text:
            continue
        role = "user" if turn.get("role") == "user" else "assistant"
        db.add(Message(organization_id=line.organization_id, conversation_id=convo.id, role=role, content=text))
    call = VoiceCall(id=call_id, organization_id=line.organization_id, voice_line_id=line.id,
                     conversation_id=convo.id, caller_number=caller, duration_secs=duration)
    db.add(call)
    billing.get_or_create_subscription(db, line.organization_id)
    db.query(Subscription).filter(Subscription.organization_id == line.organization_id).update(
        {Subscription.voice_seconds_used: Subscription.voice_seconds_used + duration}, synchronize_session=False)
    db.commit()
    return call

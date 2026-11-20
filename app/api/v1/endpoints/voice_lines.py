"""AI phone lines: the subscriber's Phone page (`router`) and the platform admin's
Voice lines + Voice settings (`admin_router`). Rules live in app/core/voice_lines.py.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_org_admin, get_platform_admin
from app.api.v1.endpoints.bots import _get_owned_bot
from app.core import config_store, elevenlabs, storage, telnyx, voice_lines
from app.core.voice_lines import VoiceLineError
from app.db.session import get_db
from app.models.bot import Bot
from app.models.organization import Organization
from app.models.user import User
from app.models.voice_line import VoiceCall, VoiceLine
from app.schemas.voice import (
    AddVoiceRequest,
    AdminVoiceLineRow,
    AvailableNumber,
    NoteRequest,
    OrderRequest,
    ReleaseRequest,
    VoiceLineOut,
    VoiceOption,
    VoicePageOut,
    VoiceSettingsAdminUpdate,
    VoiceSettingsOut,
    VoiceSettingsUpdate,
)

router = APIRouter()
admin_router = APIRouter()

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
DOCUMENT_TYPES = {"application/pdf", "image/jpeg", "image/png"}


def _raise(exc: VoiceLineError):
    raise HTTPException(status_code=exc.status, detail=exc.message)


# ===== Subscriber =====


def _page(db: Session, bot: Bot, user: User) -> VoicePageOut:
    line = voice_lines.line_for_bot(db, bot)
    a = voice_lines.allowance(db, bot.organization_id)
    return VoicePageOut(
        site_enabled=voice_lines.enabled(),
        plan_allows=a["plan_allows"],
        minutes_included=a["minutes_included"],
        minutes_used=round(a["seconds_used"] / 60, 1),
        can_edit=user.role in ("owner", "admin"),
        line=VoiceLineOut.model_validate(line),
        voices=[VoiceOption(**v) for v in voice_lines.catalog()],
    )


def _require_available(db: Session, bot: Bot) -> None:
    if not voice_lines.enabled():
        raise HTTPException(status_code=404, detail="Phone lines aren't available on this site.")
    if not voice_lines.allowance(db, bot.organization_id)["plan_allows"]:
        raise HTTPException(status_code=402, detail="Your plan doesn't include a phone line. Upgrade to add one.")


@router.get("/bots/{bot_id}/voice", response_model=VoicePageOut)
def get_voice(bot_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bot = _get_owned_bot(bot_id, db, user)
    return _page(db, bot, user)


@router.post("/bots/{bot_id}/voice/request", response_model=VoicePageOut)
async def request_number(
    bot_id: int,
    business_name: str = Form(...),
    contact_name: str = Form(...),
    contact_phone: str = Form(...),
    street_address: str = Form(...),
    locality: str = Form(...),
    administrative_area: str = Form(...),
    postal_code: str = Form(...),
    preferred_locality: str = Form(""),
    document: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_org_admin),
):
    """Send the details Telnyx needs for an Australian number, with proof of address."""
    bot = _get_owned_bot(bot_id, db, user)
    _require_available(db, bot)
    doc = None
    if document is not None and document.filename:
        data = await document.read()
        if len(data) > MAX_DOCUMENT_BYTES:
            raise HTTPException(status_code=413, detail="The document must be 10 MB or smaller.")
        mime = (document.content_type or "").lower()
        if mime not in DOCUMENT_TYPES:
            raise HTTPException(status_code=415, detail="Upload a PDF, JPG or PNG.")
        doc = (data, document.filename, mime)
    line = voice_lines.line_for_bot(db, bot)
    try:
        voice_lines.submit_request(db, line, {
            "business_name": business_name, "contact_name": contact_name, "contact_phone": contact_phone,
            "street_address": street_address, "locality": locality, "administrative_area": administrative_area,
            "postal_code": postal_code, "preferred_locality": preferred_locality,
        }, doc)
    except VoiceLineError as exc:
        _raise(exc)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="File storage isn't set up on this site yet. Please contact support.")
    return _page(db, bot, user)


@router.put("/bots/{bot_id}/voice/settings", response_model=VoicePageOut)
def update_voice_settings(bot_id: int, payload: VoiceSettingsUpdate, db: Session = Depends(get_db),
                          user: User = Depends(get_org_admin)):
    bot = _get_owned_bot(bot_id, db, user)
    _require_available(db, bot)
    line = voice_lines.line_for_bot(db, bot)
    try:
        voice_lines.update_settings(db, line, payload.model_dump(exclude_unset=True))
    except VoiceLineError as exc:
        _raise(exc)
    return _page(db, bot, user)


# ===== Platform admin: settings =====


def _hint(value: str) -> str | None:
    return f"…{value[-4:]}" if value and len(value) >= 8 else None


def _settings_out() -> VoiceSettingsOut:
    tk, ek = telnyx.api_key(), elevenlabs.api_key()
    return VoiceSettingsOut(
        enabled=voice_lines.enabled(),
        telnyx_api_key_set=bool(tk), telnyx_api_key_hint=_hint(tk),
        telnyx_sip_connection_id=telnyx.sip_connection_id(),
        elevenlabs_api_key_set=bool(ek), elevenlabs_api_key_hint=_hint(ek),
        webhook_configured=bool(config_store.get("ELEVENLABS_WEBHOOK_ID") and config_store.get("ELEVENLABS_WEBHOOK_SECRET")),
        webhook_url=voice_lines.webhook_url(),
        tts_model=elevenlabs.tts_model(),
        voices=[VoiceOption(**v) for v in voice_lines.catalog()],
    )


@admin_router.get("/settings", response_model=VoiceSettingsOut)
def get_settings(admin: User = Depends(get_platform_admin)):
    return _settings_out()


@admin_router.put("/settings", response_model=VoiceSettingsOut)
def update_settings(payload: VoiceSettingsAdminUpdate, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    updates: dict[str, str] = {}
    if payload.enabled is not None:
        updates["VOICE_ENABLED"] = "true" if payload.enabled else "false"
    for field, key in (("telnyx_api_key", "TELNYX_API_KEY"), ("elevenlabs_api_key", "ELEVENLABS_API_KEY")):
        value = (getattr(payload, field) or "").strip()
        if value:
            updates[key] = value
    if payload.telnyx_sip_connection_id is not None:
        updates["TELNYX_SIP_CONNECTION_ID"] = payload.telnyx_sip_connection_id.strip()
    if payload.tts_model is not None:
        updates["VOICE_TTS_MODEL"] = payload.tts_model.strip()
    if payload.voices is not None:
        updates["VOICE_CATALOG"] = json.dumps([v.model_dump() for v in payload.voices])
    if updates:
        config_store.set_many(db, updates)
    return _settings_out()


@admin_router.post("/test")
def test_connections(admin: User = Depends(get_platform_admin)) -> dict:
    """Check both API keys; reports each separately."""
    result = {}
    for name, check, error in (("telnyx", telnyx.check_connection, telnyx.TelnyxError),
                               ("elevenlabs", elevenlabs.check_connection, elevenlabs.ElevenLabsError)):
        try:
            check()
            result[name] = "ok"
        except error as exc:  # type: ignore[misc]
            result[name] = exc.message
    return result


@admin_router.post("/webhook", response_model=VoiceSettingsOut)
def create_webhook(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Create the ElevenLabs post-call webhook pointing here and keep its signing secret."""
    try:
        webhook_id, secret = elevenlabs.create_post_call_webhook("AiChatSupport calls", voice_lines.webhook_url())
    except elevenlabs.ElevenLabsError as exc:
        raise HTTPException(status_code=502, detail=exc.message)
    config_store.set_many(db, {"ELEVENLABS_WEBHOOK_ID": webhook_id, "ELEVENLABS_WEBHOOK_SECRET": secret})
    return _settings_out()


@admin_router.post("/voices", response_model=VoiceSettingsOut)
def add_voice(payload: AddVoiceRequest, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Add a voice from the ElevenLabs library by id (name and sample fetched from ElevenLabs)."""
    try:
        voice = elevenlabs.get_voice(payload.voice_id.strip())
    except elevenlabs.ElevenLabsError as exc:
        raise HTTPException(status_code=502, detail=f"{exc.message} Add the voice to 'My voices' in ElevenLabs first.")
    voices = [v for v in voice_lines.catalog() if v["voice_id"] != voice["voice_id"]] + [voice]
    config_store.set_many(db, {"VOICE_CATALOG": json.dumps(voices)})
    return _settings_out()


# ===== Platform admin: lines =====


def _line_or_404(db: Session, line_id: int) -> VoiceLine:
    line = db.query(VoiceLine).filter(VoiceLine.id == line_id).first()
    if line is None:
        raise HTTPException(status_code=404, detail="Line not found")
    return line


def _row(db: Session, line: VoiceLine) -> AdminVoiceLineRow:
    org = db.query(Organization).filter(Organization.id == line.organization_id).first()
    bot = db.query(Bot).filter(Bot.id == line.bot_id).first()
    calls, secs = db.query(func.count(VoiceCall.id), func.coalesce(func.sum(VoiceCall.duration_secs), 0)).filter(
        VoiceCall.voice_line_id == line.id).one()
    row = AdminVoiceLineRow.model_validate(line, from_attributes=True)
    row.organization_name = org.name if org else None
    row.bot_name = bot.name if bot else None
    row.has_document = bool(line.document_key)
    row.calls = int(calls or 0)
    row.minutes = round(int(secs or 0) / 60, 1)
    return row


@admin_router.get("/lines", response_model=list[AdminVoiceLineRow])
def list_lines(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    lines = db.query(VoiceLine).filter(VoiceLine.status != "draft").order_by(VoiceLine.updated_at.desc()).all()
    return [_row(db, line) for line in lines]


@admin_router.get("/lines/{line_id}/document")
def document_url(line_id: int, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)) -> dict:
    line = _line_or_404(db, line_id)
    if not line.document_key:
        raise HTTPException(status_code=404, detail="No document uploaded")
    return {"url": storage.presigned_url(db, line.document_key, expiry=300, download_name=line.document_filename)}


@admin_router.get("/lines/{line_id}/numbers", response_model=list[AvailableNumber])
def available_numbers(line_id: int, locality: str = "", db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    line = _line_or_404(db, line_id)
    try:
        found = telnyx.search_local_numbers(line.country_code or "AU", locality.strip() or None)
    except telnyx.TelnyxError as exc:
        raise HTTPException(status_code=502, detail=exc.message)
    return [AvailableNumber(**n) for n in found]


def _act(db: Session, line_id: int, fn, *args) -> AdminVoiceLineRow:
    line = _line_or_404(db, line_id)
    try:
        fn(db, line, *args)
    except VoiceLineError as exc:
        _raise(exc)
    return _row(db, line)


@admin_router.post("/lines/{line_id}/order", response_model=AdminVoiceLineRow)
def order(line_id: int, payload: OrderRequest, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Buy the number (non-refundable) with the subscriber's details attached."""
    return _act(db, line_id, voice_lines.order_number, payload.phone_number.strip())


@admin_router.post("/lines/{line_id}/adopt", response_model=AdminVoiceLineRow)
def adopt(line_id: int, payload: OrderRequest, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Use a number already on the Telnyx account instead of ordering one."""
    return _act(db, line_id, voice_lines.adopt_number, payload.phone_number.strip())


@admin_router.post("/lines/{line_id}/sync", response_model=AdminVoiceLineRow)
def sync(line_id: int, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    return _act(db, line_id, voice_lines.sync)


@admin_router.post("/lines/{line_id}/activate", response_model=AdminVoiceLineRow)
def activate(line_id: int, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    return _act(db, line_id, voice_lines.activate)


@admin_router.post("/lines/{line_id}/note", response_model=AdminVoiceLineRow)
def note(line_id: int, payload: NoteRequest, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Tell the subscriber what's needed; `needs_info` reopens their form."""
    line = _line_or_404(db, line_id)
    line.status_message = payload.message.strip()
    if payload.needs_info and line.status in ("submitted", "ordered", "failed"):
        line.status = "info_needed"
    db.commit()
    return _row(db, line)


@admin_router.post("/lines/{line_id}/release", response_model=AdminVoiceLineRow)
def release(line_id: int, payload: ReleaseRequest, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    return _act(db, line_id, voice_lines.release, payload.release_number)

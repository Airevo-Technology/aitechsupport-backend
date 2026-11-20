from datetime import datetime

from pydantic import BaseModel, Field


class VoiceOption(BaseModel):
    voice_id: str
    name: str
    description: str = ""
    preview_url: str = ""


class VoiceLineOut(BaseModel):
    """What the subscriber sees. Never includes secrets or provider ids."""
    id: int
    bot_id: int
    status: str
    status_message: str | None = None
    business_name: str | None = None
    contact_name: str | None = None
    contact_phone: str | None = None
    street_address: str | None = None
    locality: str | None = None
    administrative_area: str | None = None
    postal_code: str | None = None
    preferred_locality: str | None = None
    document_filename: str | None = None
    phone_number: str | None = None
    voice_id: str | None = None
    first_message: str | None = None
    timezone: str | None = None
    transfer_number: str | None = None
    max_call_seconds: int | None = None
    submitted_at: datetime | None = None
    activated_at: datetime | None = None

    class Config:
        from_attributes = True


class VoicePageOut(BaseModel):
    """Everything the bot's Phone page needs in one call."""
    site_enabled: bool
    plan_allows: bool
    minutes_included: int
    minutes_used: float
    can_edit: bool  # owner/admin may change the line
    line: VoiceLineOut
    voices: list[VoiceOption]


class VoiceSettingsUpdate(BaseModel):
    voice_id: str | None = None
    first_message: str | None = Field(default=None, max_length=500)
    timezone: str | None = Field(default=None, max_length=64)
    transfer_number: str | None = Field(default=None, max_length=32)
    max_call_seconds: int | None = Field(default=None, ge=60, le=1800)


class AdminVoiceLineRow(VoiceLineOut):
    organization_id: int
    organization_name: str | None = None
    bot_name: str | None = None
    country_code: str = "AU"
    telnyx_status: str | None = None
    telnyx_number_order_id: str | None = None
    el_agent_id: str | None = None
    has_document: bool = False
    calls: int = 0
    minutes: float = 0


class AvailableNumber(BaseModel):
    phone_number: str
    locality: str = ""
    upfront_cost: str | None = None
    monthly_cost: str | None = None
    currency: str | None = None


class OrderRequest(BaseModel):
    phone_number: str


class NoteRequest(BaseModel):
    """Admin asks the subscriber for something (status -> info_needed) or just leaves a note."""
    message: str = Field(min_length=1, max_length=1000)
    needs_info: bool = True


class ReleaseRequest(BaseModel):
    release_number: bool = False


class VoiceSettingsOut(BaseModel):
    enabled: bool
    telnyx_api_key_set: bool
    telnyx_api_key_hint: str | None = None
    telnyx_sip_connection_id: str
    elevenlabs_api_key_set: bool
    elevenlabs_api_key_hint: str | None = None
    webhook_configured: bool
    webhook_url: str
    tts_model: str
    voices: list[VoiceOption]


class VoiceSettingsAdminUpdate(BaseModel):
    """Omit a field to keep it; a blank key keeps the saved one."""
    enabled: bool | None = None
    telnyx_api_key: str | None = None
    telnyx_sip_connection_id: str | None = None
    elevenlabs_api_key: str | None = None
    tts_model: str | None = None
    voices: list[VoiceOption] | None = None


class AddVoiceRequest(BaseModel):
    voice_id: str = Field(min_length=4, max_length=64)

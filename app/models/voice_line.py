from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from app.db.session import Base


class VoiceLine(Base):
    """A bot's phone line: a Telnyx number answered by an ElevenLabs agent that
    uses this bot's knowledge base as its custom LLM (app/core/voice_lines.py).

    One per bot. The subscriber supplies the regulator's details (business,
    contact, address, proof-of-address document); a platform admin orders the
    number, which Telnyx vets; once active, the admin activates the line, which
    creates the ElevenLabs agent and attaches the number.
    """

    __tablename__ = "voice_lines"

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), index=True, nullable=False)
    bot_id = Column(Integer, ForeignKey("bots.id"), unique=True, index=True, nullable=False)

    # draft -> submitted -> ordered -> (info_needed) -> number_ready -> active; released | failed
    status = Column(String, default="draft", nullable=False)
    # Last note for the subscriber (e.g. what Telnyx asked for, or why it failed).
    status_message = Column(Text, nullable=True)

    # Regulatory details, as Telnyx asks for them for an Australian local number.
    business_name = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    street_address = Column(String, nullable=True)
    locality = Column(String, nullable=True)          # suburb / town
    administrative_area = Column(String, nullable=True)  # state, e.g. VIC
    postal_code = Column(String, nullable=True)
    country_code = Column(String, default="AU", nullable=False)
    # Proof of address, held in object storage (never public).
    document_key = Column(String, nullable=True)
    document_filename = Column(String, nullable=True)
    document_mime = Column(String, nullable=True)
    # Where the subscriber would like the number (city), e.g. Sydney.
    preferred_locality = Column(String, nullable=True)

    # Telnyx
    phone_number = Column(String, nullable=True)  # E.164
    telnyx_number_order_id = Column(String, nullable=True)
    telnyx_phone_number_id = Column(String, nullable=True)
    telnyx_requirement_group_id = Column(String, nullable=True)
    telnyx_status = Column(String, nullable=True)  # raw requirements_status / number status

    # Caller experience (applied to the ElevenLabs agent).
    voice_id = Column(String, nullable=True)
    first_message = Column(Text, nullable=True)
    timezone = Column(String, nullable=True)
    transfer_number = Column(String, nullable=True)  # E.164; empty = no transfer
    max_call_seconds = Column(Integer, nullable=True)

    # ElevenLabs
    el_agent_id = Column(String, nullable=True)
    el_phone_number_id = Column(String, nullable=True)
    el_secret_id = Column(String, nullable=True)
    # sha256 of the bearer secret ElevenLabs sends to our custom-LLM endpoint.
    llm_secret_hash = Column(String, nullable=True)

    submitted_at = Column(DateTime, nullable=True)
    activated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class VoiceCall(Base):
    """One finished call, from ElevenLabs' post-call webhook (deduped on its id)."""

    __tablename__ = "voice_calls"

    # ElevenLabs conversation_id
    id = Column(String, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), index=True, nullable=False)
    voice_line_id = Column(Integer, ForeignKey("voice_lines.id"), index=True, nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    caller_number = Column(String, nullable=True)
    duration_secs = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

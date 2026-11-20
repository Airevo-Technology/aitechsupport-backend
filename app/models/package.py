from datetime import datetime

from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, false

from app.db.session import Base


class Package(Base):
    """A sellable AI-chatbot plan. The operator manages these in the admin panel."""

    __tablename__ = "packages"

    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    # Whole ringgit per month (0 = free).
    price_myr = Column(Integer, default=0, nullable=False)
    monthly_token_quota = Column(Integer, default=0, nullable=False)
    max_bots = Column(Integer, default=1, nullable=False)
    # List[str] of marketing bullet points.
    features = Column(JSON, default=list)
    is_active = Column(Boolean, default=True, nullable=False)
    sort_order = Column(Integer, default=0, nullable=False)

    # Whether a bot on this package can connect WhatsApp (Fonnte) at all —
    # gates the /whatsapp/connect flow, not just the model used to answer.
    whatsapp_enabled = Column(Boolean, default=True, nullable=False)

    # AI phone line (app/core/voice_lines.py): whether a bot on this package can
    # have one, and the call minutes included each month (0 = none).
    voice_enabled = Column(Boolean, default=False, server_default=false(), nullable=False)
    monthly_voice_minutes = Column(Integer, default=0, server_default="0", nullable=False)

    # Ordered chat fallback chain for bots on this package: a list of
    # {label, provider, base_url, model} tried top to bottom until one answers
    # (see models_catalog.resolve_candidates). NULL -> inherit the platform-wide
    # chain from settings, so a new package works before it's configured.
    fallback_chain = Column(JSON, nullable=True)

    # DEPRECATED — superseded by fallback_chain above. Nothing reads these; they
    # are kept only to avoid a destructive migration and should not be exposed in
    # the API or admin UI (stale values here once silently misled the operator).
    model_provider = Column(String, default="openrouter", nullable=False)
    chat_model = Column(String, nullable=True)
    self_hosted_base_url = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

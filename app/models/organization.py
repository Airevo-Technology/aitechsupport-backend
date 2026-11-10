from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey

from app.db.session import Base


class Organization(Base):
    """A tenant. Everything else is scoped by organization_id."""

    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, index=True, nullable=False)

    # Offline fallback contacts. Offered to visitors by the widget only while no
    # support agent is online, so "Talk to a human" never leads to silence.
    support_email = Column(String, nullable=True)
    support_whatsapp = Column(String, nullable=True)

    # Referral program: our shareable code, and who (if anyone) referred us.
    referral_code = Column(String, unique=True, index=True, nullable=True)
    referred_by_organization_id = Column(Integer, ForeignKey("organizations.id"), index=True, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

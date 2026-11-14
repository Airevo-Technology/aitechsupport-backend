from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.db.session import Base


class AirevoEvent(Base):
    """An airevo.com.au marketplace event we have processed (dedupe on its id)."""

    __tablename__ = "airevo_events"

    id = Column(String, primary_key=True)
    type = Column(String, nullable=False)
    airevo_subscription_id = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    received_at = Column(DateTime, default=datetime.utcnow, nullable=False)

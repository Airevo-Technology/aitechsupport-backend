from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey

from app.db.session import Base


class Payment(Base):
    """One Billplz bill for a paid-plan purchase.

    Persisted at checkout (status ``pending``) so the webhook can map the paid
    bill back to the right organization + plan and activate it idempotently —
    the bill id, not a client-supplied reference, is the source of truth.
    """

    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), index=True, nullable=False)

    # Package slug to activate once the bill is paid (see packages.slug).
    plan_slug = Column(String, nullable=False)
    amount_cents = Column(Integer, nullable=False)  # sen (RM * 100)

    # ISO currency of amount_cents (the deployment's BILLING_CURRENCY at checkout).
    currency = Column(String, default="MYR", nullable=False, server_default="MYR")

    # billplz | paypal | bank_transfer
    method = Column(String, default="billplz", nullable=False)

    # Billplz bill id (null only if the gateway call failed before we got one, or
    # this is a bank_transfer payment which has no gateway bill at all).
    billplz_bill_id = Column(String, unique=True, index=True, nullable=True)
    # PayPal order id (paypal payments only).
    paypal_order_id = Column(String, unique=True, index=True, nullable=True)
    # pending | paid | failed
    status = Column(String, default="pending", nullable=False)
    # True if created against the Billplz sandbox environment.
    sandbox = Column(Boolean, default=False, nullable=False)

    # Customer-supplied note on a bank_transfer report (e.g. "transferred RM99,
    # own bank ref ABC123") — there's no webhook for this method, so a human
    # admin confirms against this note before marking the payment paid.
    reference_note = Column(Text, nullable=True)
    # When the customer reported the transfer (bank_transfer only).
    reported_at = Column(DateTime, nullable=True)

    paid_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

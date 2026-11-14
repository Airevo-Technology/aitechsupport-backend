"""airevo.com.au marketplace integration (provider side).

Airevo is the legal supplier to Australian customers: it takes the payment,
issues the tax invoice and tells us, through signed events, when a paid plan
starts, changes, pauses or ends. This module verifies those events, applies them
to the organization's Subscription, and calls back to Airevo's provider API
(single sign-on verification).

Contract: Airevo's "Marketplace — Provider Integration Guide" (docs/
marketplace-provider-integration.md in the Airevo repo). Both directions use the
same HMAC scheme and the same key and secret.

Rules worth knowing before changing anything here:
- Airevo does not verify customer emails. A purchase whose email matches an
  existing account upgrades that account, but SSO never signs into it; only
  accounts the marketplace created get `airevo_sso_allowed`.
- Events can arrive more than once. Each event id is stored with the reply we
  gave, and a redelivery gets that same reply without touching anything.
- Airevo-billed subscriptions are left alone by direct billing (checkout,
  renewal reminders); see billing.is_airevo_billed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core import billing, config_store
from app.core import email as email_service
from app.core.security import create_reset_token, get_password_hash
from app.core.settings import settings
from app.models.airevo_event import AirevoEvent
from app.models.organization import Organization
from app.models.package import Package
from app.models.subscription import Subscription
from app.models.user import User

logger = logging.getLogger(__name__)

# Airevo refuses anything older; we hold ourselves to the same window.
MAX_SKEW_SECONDS = 300
# Set-password link in the welcome email. Longer than the 30-minute reset link,
# because a new customer may not open the email straight after paying.
WELCOME_LINK_MINUTES = 7 * 24 * 60
DEFAULT_API_BASE = "https://airevo.com.au/api/marketplace/v1"

# Event types that move the subscription. Anything else (ticket.*, new types
# Airevo adds later) is acknowledged with 200 and otherwise ignored.
SUBSCRIPTION_EVENTS = {
    "subscription.provision",
    "subscription.updated",
    "subscription.cancel_scheduled",
    "subscription.suspend",
    "subscription.resume",
    "subscription.end",
}


class AirevoError(Exception):
    """A permanent refusal (4xx): Airevo does not retry it and alerts its admin."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


# ===== Configuration =====


def enabled() -> bool:
    return config_store.get("AIREVO_ENABLED", "false").strip().lower() == "true"


def key_id() -> str:
    return config_store.get("AIREVO_KEY_ID").strip()


def _secrets() -> list[str]:
    """Current secret first, then the previous one while a rotation is running."""
    return [s for s in (config_store.get("AIREVO_SECRET").strip(), config_store.get("AIREVO_SECRET_PREVIOUS").strip()) if s]


def api_base() -> str:
    return (config_store.get("AIREVO_API_BASE") or DEFAULT_API_BASE).rstrip("/")


def listing_url() -> str:
    return config_store.get("AIREVO_LISTING_URL").strip()


def site_base() -> str:
    """https://airevo.com.au, derived from the API base so one setting covers both."""
    parsed = urlparse(api_base())
    return f"{parsed.scheme}://{parsed.netloc}"


def manage_url(sub: Subscription) -> str | None:
    """The customer's own subscription page on Airevo (change plan, card, cancel)."""
    return f"{site_base()}/portal/subscriptions/{sub.airevo_subscription_id}" if sub.airevo_subscription_id else None


def is_configured() -> bool:
    return enabled() and bool(key_id()) and bool(_secrets())


def sells_paid_plans() -> bool:
    """True when paid plans on this deployment are bought on airevo.com.au only."""
    return enabled() and bool(listing_url())


# ===== Signing (section 3 of the guide) =====


def signature(secret: str, timestamp: str, method: str, path_and_query: str, raw_body: bytes) -> str:
    canonical = "\n".join([timestamp, method.upper(), path_and_query, hashlib.sha256(raw_body).hexdigest()])
    return "v1=" + hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()


def verify(headers: dict[str, str], method: str, path_and_query: str, raw_body: bytes, now: float | None = None) -> bool:
    """Key id, then timestamp window, then a constant-time signature check.

    `headers` keys must be lowercase."""
    if not is_configured() or not hmac.compare_digest(headers.get("x-airevo-key-id", ""), key_id()):
        return False
    ts = headers.get("x-airevo-timestamp", "")
    if not ts.isdigit() or abs((now if now is not None else time.time()) - int(ts)) > MAX_SKEW_SECONDS:
        return False
    given = headers.get("x-airevo-signature", "")
    # Check every valid secret (no early exit), so a rotation can't be timed.
    matches = [hmac.compare_digest(signature(s, ts, method, path_and_query, raw_body), given) for s in _secrets()]
    return any(matches)


def signed_headers(method: str, path_and_query: str, raw_body: bytes) -> dict[str, str]:
    ts = str(int(time.time()))
    return {
        "X-Airevo-Key-Id": key_id(),
        "X-Airevo-Timestamp": ts,
        "X-Airevo-Signature": signature(_secrets()[0], ts, method, path_and_query, raw_body),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ===== Provider API: us → Airevo (section 5) =====


def _post(path: str, payload: dict) -> dict:
    url = api_base() + path
    body = json.dumps(payload, separators=(",", ":")).encode()
    signed_path = urlparse(url).path
    resp = httpx.post(url, content=body, headers=signed_headers("POST", signed_path, body), timeout=10)
    resp.raise_for_status()
    return resp.json()


def sso_verify(token: str) -> dict:
    """{"valid": bool, "subscription_id", "external_id", "customer": {...}}."""
    return _post("/sso/verify", {"token": token})


# ===== Events: Airevo → us (section 4) =====


def external_id(org_id: int) -> str:
    return f"org_{org_id}"


def _parse_time(raw: str | None) -> datetime | None:
    """Airevo sends ISO-8601 with an offset; we store naive UTC like the rest of the schema."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def package_for_plan(db: Session, plan: dict | None) -> Package:
    """Airevo plans are named after our packages ("Starter" → starter)."""
    name = ((plan or {}).get("name") or "").strip()
    pkg = (
        db.query(Package)
        .filter((Package.slug == name.lower()) | (func.lower(Package.name) == name.lower()))
        .first()
    ) if name else None
    if pkg is None or not pkg.is_active or pkg.price_myr <= 0:
        raise AirevoError(422, "unknown_plan", f"No paid AiChatSupport plan matches '{name}'.")
    return pkg


def _access(email: str, new_account: bool) -> dict:
    login = f"{settings.FRONTEND_URL.rstrip('/')}/login"
    if new_account:
        how = "Use Open app to go straight in, or check your email for a link to set a password for signing in directly."
    else:
        how = "Your existing AiChatSupport account has been upgraded. Sign in with your usual password."
    return {"login_url": login, "username": email, "instructions": how}


def _sub_by_airevo_id(db: Session, airevo_id: int) -> Subscription | None:
    return db.query(Subscription).filter(Subscription.airevo_subscription_id == airevo_id).first()


def _apply_plan(db: Session, sub: Subscription, pkg: Package) -> None:
    """Same rule as billing.subscribe: usage resets only on a real plan change."""
    if sub.plan != pkg.slug:
        sub.plan = pkg.slug
        sub.tokens_used = 0
        sub.messages_used = 0
        sub.period_start = datetime.utcnow()


def _provision(db: Session, event: dict) -> dict:
    data = event.get("subscription") or {}
    customer = event.get("customer") or {}
    airevo_id = int(data["id"])
    pkg = package_for_plan(db, data.get("plan"))

    sub = _sub_by_airevo_id(db, airevo_id)
    new_account = False
    if sub is not None:
        # A redelivery, or the customer returning within the retention window:
        # restore the same account.
        org_id = sub.organization_id
        owner = _owner(db, org_id)
        email = owner.email if owner else (customer.get("email") or "")
        new_account = bool(sub.airevo_sso_allowed)
    else:
        email = (customer.get("email") or "").strip().lower()
        if not email:
            raise AirevoError(422, "missing_email", "The customer has no email address.")
        user = db.query(User).filter(func.lower(User.email) == email).first()
        if user is not None:
            # Only an owner's purchase upgrades their organization; an agent's
            # email must not attach a plan to the organization they work for.
            if user.role != "owner":
                raise AirevoError(422, "email_in_use", "This email belongs to a team member of another AiChatSupport account.")
            org_id = user.organization_id
            sub = billing.get_or_create_subscription(db, org_id)
            if sub.billing_source == "airevo" and sub.airevo_subscription_id not in (None, airevo_id):
                raise AirevoError(422, "already_subscribed", "This account already has an active airevo.com.au subscription.")
        else:
            org_id = _create_account(db, email, customer)
            sub = billing.get_or_create_subscription(db, org_id)
            sub.airevo_sso_allowed = True
            new_account = True

    _apply_plan(db, sub, pkg)
    sub.billing_source = "airevo"
    sub.airevo_subscription_id = airevo_id
    sub.status = "active"
    sub.airevo_period_end = _parse_time(data.get("current_period_end"))
    # Airevo owns renewals: keep our own reminder cycle off this subscription.
    sub.next_billing_date = None
    db.commit()

    if new_account and sub.airevo_sso_allowed and not _welcome_sent(db, airevo_id):
        _send_welcome(db, org_id, pkg)

    return {"external_id": external_id(org_id), "access": _access(email, new_account)}


def _create_account(db: Session, email: str, customer: dict) -> int:
    # Local import: the auth endpoint module owns slug generation.
    from app.api.v1.endpoints.auth import _unique_org_slug

    name = (customer.get("business_name") or customer.get("name") or email.split("@")[0]).strip()
    org = Organization(name=name, slug=_unique_org_slug(db, name))
    db.add(org)
    db.flush()
    db.add(User(
        organization_id=org.id,
        email=email,
        # Never revealed; the customer sets their own through the welcome link or uses SSO.
        hashed_password=get_password_hash(secrets.token_urlsafe(24)),
        full_name=(customer.get("name") or "").strip() or None,
        role="owner",
        is_email_verified=False,
    ))
    db.flush()
    return org.id


def _owner(db: Session, org_id: int) -> User | None:
    return (
        db.query(User)
        .filter(User.organization_id == org_id, User.role == "owner", User.is_active.is_(True))
        .order_by(User.id)
        .first()
    )


def _welcome_sent(db: Session, airevo_id: int) -> bool:
    """A provision for this subscription was already answered (so the email went)."""
    return (
        db.query(AirevoEvent.id)
        .filter(AirevoEvent.airevo_subscription_id == airevo_id, AirevoEvent.type == "subscription.provision")
        .first()
        is not None
    )


def _send_welcome(db: Session, org_id: int, pkg: Package) -> None:
    owner = _owner(db, org_id)
    if owner is None:
        return
    token = create_reset_token(owner.id, owner.hashed_password, expires_minutes=WELCOME_LINK_MINUTES)
    verify_url = f"{settings.FRONTEND_URL.rstrip('/')}/reset-password?token={token}&verify=1"
    org = db.query(Organization).filter(Organization.id == org_id).first()
    email_service.send_template_bg(
        owner.email,
        "verify_email",
        {"name": owner.full_name or (org.name if org else owner.email), "email": owner.email, "plan": pkg.name, "verify_url": verify_url},
    )


def _existing(db: Session, event: dict) -> Subscription:
    data = event.get("subscription") or {}
    sub = _sub_by_airevo_id(db, int(data["id"]))
    if sub is None:
        raise AirevoError(404, "unknown_subscription", "No account was provisioned for this subscription.")
    return sub


def _updated(db: Session, event: dict) -> None:
    sub = _existing(db, event)
    data = event.get("subscription") or {}
    _apply_plan(db, sub, package_for_plan(db, data.get("plan")))
    sub.billing_source = "airevo"
    sub.status = "active"
    sub.airevo_period_end = _parse_time(data.get("current_period_end")) or sub.airevo_period_end
    db.commit()


def _set_status(db: Session, event: dict, status: str) -> None:
    sub = _existing(db, event)
    sub.status = status
    db.commit()


def _end(db: Session, event: dict) -> None:
    """Back to Free with the data kept. The Airevo id stays, so a customer who
    returns within the retention window gets the same account back."""
    sub = _existing(db, event)
    free = billing.package_for(db, billing.DEFAULT_PLAN)
    if free is not None:
        _apply_plan(db, sub, free)
    else:
        sub.plan = billing.DEFAULT_PLAN
    sub.billing_source = "direct"
    sub.status = "active"
    sub.airevo_period_end = None
    db.commit()


def handle(db: Session, event: dict) -> tuple[int, dict]:
    """Apply one verified event. Returns (HTTP status, JSON body)."""
    event_id = str(event.get("id") or "")
    event_type = str(event.get("type") or "")
    if not event_id or not event_type:
        raise AirevoError(422, "bad_event", "The event has no id or type.")

    seen = db.query(AirevoEvent).filter(AirevoEvent.id == event_id).first()
    if seen is not None:
        return 200, json.loads(seen.response_body or "{}")

    body: dict = {}
    airevo_id = None
    if event_type in SUBSCRIPTION_EVENTS:
        airevo_id = int((event.get("subscription") or {}).get("id") or 0) or None
        if airevo_id is None:
            raise AirevoError(422, "bad_event", "The event has no subscription id.")

        created = int(event.get("created") or 0)
        current = _sub_by_airevo_id(db, airevo_id)
        if current is not None and current.airevo_last_event_at and created and created < current.airevo_last_event_at:
            # Older than what we already applied: acknowledge without undoing newer state.
            logger.info("airevo.event_stale", extra={"event": event_id, "type": event_type})
        elif event_type == "subscription.provision":
            body = _provision(db, event)
        elif event_type == "subscription.updated":
            _updated(db, event)
        elif event_type == "subscription.suspend":
            _set_status(db, event, "past_due")
        elif event_type == "subscription.resume":
            _set_status(db, event, "active")
        elif event_type == "subscription.end":
            _end(db, event)
        # cancel_scheduled: access continues to the period end; nothing to change.

        sub = _sub_by_airevo_id(db, airevo_id)
        if sub is not None and created and (sub.airevo_last_event_at or 0) < created:
            sub.airevo_last_event_at = created

    db.add(AirevoEvent(id=event_id, type=event_type, airevo_subscription_id=airevo_id, response_body=json.dumps(body)))
    db.commit()
    return 200, body


def sso_user(db: Session, token: str) -> User | None:
    """The owner to sign in for a verified SSO token, or None to fall back to the
    normal sign-in page (invalid token, pre-existing account, suspended)."""
    result = sso_verify(token)
    if not result.get("valid"):
        return None
    sub = _sub_by_airevo_id(db, int(result.get("subscription_id") or 0))
    if sub is None or not sub.airevo_sso_allowed or sub.billing_source != "airevo" or sub.status != "active":
        return None
    if external_id(sub.organization_id) != result.get("external_id"):
        return None
    return _owner(db, sub.organization_id)

"""Subscription + token-metering helpers, plus Billplz gateway config.

Paid-plan self-serve stays gated on BILLING_ENABLED + configured Billplz
credentials; until then a platform admin assigns paid plans manually. The
Billplz connection details (API key, X-Signature key, collection, sandbox
toggle) resolve from config_store — DB settings first, `.env` as fallback — so
an admin can wire the gateway from the panel without a redeploy, same pattern as
storage/SMTP.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta

import httpx
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core import config_store, paypal
from app.core.settings import settings
from app.models.subscription import Subscription
from app.models.package import Package
from app.models.payment import Payment
from app.models.channel import Channel
from app.models.organization import Organization

logger = logging.getLogger(__name__)

PERIOD_DAYS = 30
DEFAULT_PLAN = "free"

# Billplz API roots. Sandbox is a fully separate environment with its own keys.
BILLPLZ_LIVE_BASE = "https://www.billplz.com/api"
BILLPLZ_SANDBOX_BASE = "https://www.billplz-sandbox.com/api"


def package_for(db: Session, slug: str) -> Package | None:
    return db.query(Package).filter(Package.slug == slug).first()


def get_or_create_subscription(db: Session, organization_id: int) -> Subscription:
    """Return the org's subscription, creating a free one and rolling the monthly
    usage window if the current period has elapsed."""
    sub = db.query(Subscription).filter(Subscription.organization_id == organization_id).first()
    if sub is None:
        sub = Subscription(organization_id=organization_id, plan=DEFAULT_PLAN, tokens_used=0, period_start=datetime.utcnow())
        db.add(sub)
        try:
            db.commit()
            db.refresh(sub)
        except IntegrityError:
            # Concurrent first-access created it first — use that row.
            db.rollback()
            sub = db.query(Subscription).filter(Subscription.organization_id == organization_id).first()
        return sub

    # Roll the window if the 30-day period has passed (atomic to avoid clobbering a
    # concurrent usage write).
    if sub.period_start and datetime.utcnow() - sub.period_start >= timedelta(days=PERIOD_DAYS):
        db.query(Subscription).filter(Subscription.organization_id == organization_id).update(
            {Subscription.tokens_used: 0, Subscription.messages_used: 0, Subscription.period_start: datetime.utcnow()},
            synchronize_session=False,
        )
        db.commit()
        db.refresh(sub)
    return sub


def quota_for(db: Session, sub: Subscription) -> int:
    """Monthly token quota for this subscription's package (0 = unknown/no allowance)."""
    pkg = package_for(db, sub.plan)
    return pkg.monthly_token_quota if pkg else 0


def tokens_remaining(db: Session, sub: Subscription) -> int:
    quota = quota_for(db, sub)
    return max(0, quota - (sub.tokens_used or 0))


def has_quota(db: Session, sub: Subscription) -> bool:
    """True if the org can still spend tokens (quota 0 is treated as 'no allowance')."""
    return tokens_remaining(db, sub) > 0


def record_usage(db: Session, organization_id: int, tokens: int, messages: int = 1) -> None:
    """Atomically increment usage so concurrent chats don't lose counts."""
    get_or_create_subscription(db, organization_id)  # ensure row exists + window rolled
    db.query(Subscription).filter(Subscription.organization_id == organization_id).update(
        {
            Subscription.tokens_used: Subscription.tokens_used + max(0, tokens),
            Subscription.messages_used: Subscription.messages_used + max(0, messages),
        },
        synchronize_session=False,
    )
    db.commit()


def subscribe(db: Session, organization_id: int, package: Package) -> Subscription:
    """Assign a package to the org. Only reset the usage window on a real plan change,
    so re-subscribing to the same plan can't be used to wipe usage."""
    sub = get_or_create_subscription(db, organization_id)
    changed = sub.plan != package.slug
    sub.plan = package.slug
    sub.status = "active"
    if changed:
        sub.tokens_used = 0
        sub.messages_used = 0
        sub.period_start = datetime.utcnow()
    db.commit()
    db.refresh(sub)
    return sub


def roll_billing_period(sub: Subscription) -> None:
    """Move the renewal date one period forward after a confirmed payment.

    Paying early extends from the existing due date (no lost days); paying late
    restarts from today."""
    now = datetime.utcnow()
    base = sub.next_billing_date if sub.next_billing_date and sub.next_billing_date > now else now
    sub.start_date = sub.start_date or now
    sub.next_billing_date = base + timedelta(days=PERIOD_DAYS)


# ===== Referral program =====
# Each referred organization that is on an active paid plan takes 10% off the
# referrer's price at every paid checkout/renewal (capped at 100%). The discount
# is baked into Payment.amount_cents at checkout, so gateways and webhooks charge
# and verify exactly what the client saw. At 100% no gateway is involved at all.

REFERRAL_PERCENT_EACH = 10
REFERRAL_GRACE_DAYS = 7  # a lapsed renewal stops counting after this long
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I


def ensure_referral_code(db: Session, org: Organization) -> str:
    if org.referral_code:
        return org.referral_code
    for _ in range(10):
        code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(8))
        if not db.query(Organization.id).filter(Organization.referral_code == code).first():
            org.referral_code = code
            db.commit()
            return code
    raise RuntimeError("could not generate a referral code")


def organization_by_referral_code(db: Session, code: str | None) -> Organization | None:
    code = (code or "").strip().upper()
    if not code:
        return None
    return db.query(Organization).filter(Organization.referral_code == code).first()


def _is_paying(db: Session, sub: Subscription | None) -> bool:
    """On a paid plan, active, and not lapsed beyond the grace period."""
    if sub is None or sub.status != "active":
        return False
    pkg = package_for(db, sub.plan)
    if pkg is None or pkg.price_myr <= 0:
        return False
    if sub.next_billing_date and sub.next_billing_date < datetime.utcnow() - timedelta(days=REFERRAL_GRACE_DAYS):
        return False
    return True


def referral_list(db: Session, organization_id: int) -> list[dict]:
    """Every org this one referred, newest first, with whether it counts today."""
    rows = (
        db.query(Organization)
        .filter(Organization.referred_by_organization_id == organization_id)
        .order_by(Organization.created_at.desc())
        .all()
    )
    out = []
    for o in rows:
        sub = db.query(Subscription).filter(Subscription.organization_id == o.id).first()
        out.append({"organization": o, "counts": _is_paying(db, sub), "plan": sub.plan if sub else "free"})
    return out


def referral_discount_percent(db: Session, organization_id: int) -> int:
    n = sum(1 for r in referral_list(db, organization_id) if r["counts"])
    return min(100, n * REFERRAL_PERCENT_EACH)


def net_price_cents(db: Session, organization_id: int, package: Package) -> int:
    """What this org pays for `package` now, in minor units, after its referral discount."""
    gross = int(package.price_myr) * 100
    pct = referral_discount_percent(db, organization_id)
    return gross * (100 - pct) // 100


def activate_without_payment(db: Session, organization_id: int, package: Package) -> Subscription:
    """A fully referral-discounted period: record a zero-value paid Payment (audit
    trail) and activate/roll the plan with no gateway."""
    payment = Payment(
        organization_id=organization_id,
        plan_slug=package.slug,
        amount_cents=0,
        currency=billing_currency(),
        method="referral",
        status="paid",
        sandbox=False,
        paid_at=datetime.utcnow(),
    )
    db.add(payment)
    sub = subscribe(db, organization_id, package)
    roll_billing_period(sub)
    db.commit()
    db.refresh(sub)
    logger.info("referral-covered period: plan %s for org %s", package.slug, organization_id)
    return sub


# ===== Billplz payment gateway =====

def _as_bool(raw: str, default: bool) -> bool:
    if raw is None or raw == "":
        return default
    return str(raw).strip().lower() in ("true", "1", "yes", "on")


def billplz_config(db: Session | None = None) -> dict:
    """Resolve the ACTIVE Billplz credential set (DB settings first, env fallback).

    Live and sandbox are separate Billplz accounts with their own key,
    X-Signature secret and collection — BILLPLZ_SANDBOX only picks which set is
    active, so switching modes never overwrites the other one's credentials.
    """
    g = config_store.get
    sandbox = _as_bool(g("BILLPLZ_SANDBOX"), settings.BILLPLZ_SANDBOX)
    if sandbox:
        api_key = g("BILLPLZ_SANDBOX_API_KEY") or settings.BILLPLZ_SANDBOX_API_KEY
        x_signature_key = g("BILLPLZ_SANDBOX_X_SIGNATURE_KEY") or settings.BILLPLZ_SANDBOX_X_SIGNATURE_KEY
        collection_id = g("BILLPLZ_SANDBOX_COLLECTION_ID") or settings.BILLPLZ_SANDBOX_COLLECTION_ID
    else:
        api_key = g("BILLPLZ_API_KEY") or settings.BILLPLZ_API_KEY
        x_signature_key = g("BILLPLZ_X_SIGNATURE_KEY") or settings.BILLPLZ_X_SIGNATURE_KEY
        collection_id = g("BILLPLZ_COLLECTION_ID") or settings.BILLPLZ_COLLECTION_ID
    return {
        "api_key": api_key,
        "x_signature_key": x_signature_key,
        "collection_id": collection_id,
        "sandbox": sandbox,
        "enabled": _as_bool(g("BILLING_ENABLED"), settings.BILLING_ENABLED),
    }


def billplz_config_both(db: Session | None = None) -> dict:
    """Resolve BOTH live and sandbox credential sets, for the admin panel — it
    shows/edits each independently regardless of which one is currently active."""
    g = config_store.get
    return {
        "sandbox": _as_bool(g("BILLPLZ_SANDBOX"), settings.BILLPLZ_SANDBOX),
        "enabled": _as_bool(g("BILLING_ENABLED"), settings.BILLING_ENABLED),
        "live": {
            "api_key": g("BILLPLZ_API_KEY") or settings.BILLPLZ_API_KEY,
            "x_signature_key": g("BILLPLZ_X_SIGNATURE_KEY") or settings.BILLPLZ_X_SIGNATURE_KEY,
            "collection_id": g("BILLPLZ_COLLECTION_ID") or settings.BILLPLZ_COLLECTION_ID,
        },
        "sandbox_creds": {
            "api_key": g("BILLPLZ_SANDBOX_API_KEY") or settings.BILLPLZ_SANDBOX_API_KEY,
            "x_signature_key": g("BILLPLZ_SANDBOX_X_SIGNATURE_KEY") or settings.BILLPLZ_SANDBOX_X_SIGNATURE_KEY,
            "collection_id": g("BILLPLZ_SANDBOX_COLLECTION_ID") or settings.BILLPLZ_SANDBOX_COLLECTION_ID,
        },
    }


def billplz_api_base(sandbox: bool) -> str:
    return BILLPLZ_SANDBOX_BASE if sandbox else BILLPLZ_LIVE_BASE


def billplz_is_configured(db: Session | None = None) -> bool:
    """True when billing is enabled and the gateway can create bills.

    A collection is required to create bills, so it's part of "configured" — the
    X-Signature key is only needed to verify webhooks, so it's not gated here.
    """
    cfg = billplz_config(db)
    return cfg["enabled"] and bool(cfg["api_key"]) and bool(cfg["collection_id"])


def test_billplz(db: Session | None = None) -> dict:
    """Verify the API key (and collection, if set) reach Billplz. Raises on failure.

    Billplz authenticates with the API key as HTTP Basic username and a blank
    password. When a collection id is set we fetch that collection (proves the
    key AND the collection exist in the selected environment); otherwise we list
    collections, which still proves the key + sandbox choice are valid.
    """
    cfg = billplz_config(db)
    if not cfg["api_key"]:
        raise RuntimeError("Enter a Billplz API key before testing.")
    base = billplz_api_base(cfg["sandbox"])
    coll = cfg["collection_id"]
    url = f"{base}/v3/collections/{coll}" if coll else f"{base}/v3/collections"
    try:
        resp = httpx.get(url, auth=(cfg["api_key"], ""), timeout=15)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Could not reach Billplz: {exc}") from exc
    if resp.status_code == 401:
        env = "sandbox" if cfg["sandbox"] else "production"
        raise RuntimeError(f"Billplz rejected the API key for the {env} environment (401).")
    if resp.status_code == 404 and coll:
        env = "sandbox" if cfg["sandbox"] else "production"
        raise RuntimeError(f"Collection '{coll}' was not found in the {env} environment (404).")
    if resp.status_code >= 400:
        raise RuntimeError(f"Billplz returned {resp.status_code}: {resp.text[:200]}")
    return {"sandbox": cfg["sandbox"], "collection_id": coll or None}


# ===== Billplz checkout (create a bill) + webhook (activate on payment) =====

def create_checkout_bill(
    db: Session,
    *,
    organization_id: int,
    package: Package,
    buyer_email: str,
    buyer_name: str,
) -> tuple[Payment, str]:
    """Create a pending Payment + a Billplz bill; return (payment, payment_url).

    The Payment row is written first so a paid webhook can always be reconciled
    even if the browser never returns. Raises RuntimeError if the gateway isn't
    configured or the bill can't be created.
    """
    if not billplz_is_configured(db):
        raise RuntimeError("Billing is not configured")
    cfg = billplz_config(db)
    base = billplz_api_base(cfg["sandbox"])
    amount_cents = net_price_cents(db, organization_id, package)

    payment = Payment(
        organization_id=organization_id,
        plan_slug=package.slug,
        amount_cents=amount_cents,
        status="pending",
        sandbox=cfg["sandbox"],
    )
    db.add(payment)
    db.commit()
    db.refresh(payment)

    api_prefix = settings.API_V1_STR  # e.g. "/api/v1"
    body = {
        "collection_id": cfg["collection_id"],
        "email": buyer_email,
        "name": (buyer_name or buyer_email)[:255],
        "amount": amount_cents,  # in sen
        "callback_url": f"{settings.API_BASE_URL}{api_prefix}/billing/webhook/billplz",
        "redirect_url": f"{settings.FRONTEND_URL}/dashboard/billing",
        "description": f"AiChatSupport {package.name} plan (monthly)"[:200],
        # Redundant safety net; the Payment row is the primary bill->org/plan map.
        "reference_1": str(organization_id),
        "reference_2": package.slug,
    }
    try:
        # Billplz's sandbox routinely takes 20s+ to answer, so keep plenty of headroom.
        resp = httpx.post(f"{base}/v3/bills", auth=(cfg["api_key"], ""), data=body, timeout=60)
    except httpx.HTTPError as exc:
        payment.status = "failed"
        db.commit()
        raise RuntimeError(f"Could not reach Billplz: {exc}") from exc
    if resp.status_code >= 400:
        payment.status = "failed"
        db.commit()
        raise RuntimeError(f"Billplz bill creation failed ({resp.status_code}): {resp.text[:200]}")

    bill = resp.json()
    payment.billplz_bill_id = bill.get("id")
    db.commit()
    logger.info("created Billplz bill %s for org %s plan %s", bill.get("id"), organization_id, package.slug)
    return payment, bill.get("url")


def billplz_signature_source(data: dict) -> str:
    """Billplz X-Signature source string: every field except x_signature, sorted
    by key, formatted `key + value`, joined by `|`."""
    return "|".join(f"{k}{data[k]}" for k in sorted(data) if k != "x_signature")


def verify_billplz_signature(data: dict, x_signature: str, key: str) -> bool:
    """Constant-time check of the Billplz callback X-Signature (HMAC-SHA256)."""
    if not key or not x_signature:
        return False
    computed = hmac.new(key.encode(), billplz_signature_source(data).encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, x_signature)


def process_billplz_webhook(db: Session, data: dict) -> str:
    """Verify a Billplz callback and activate the plan on payment (idempotent).

    Returns a short status string for logging. Raises ValueError on a bad or
    unverifiable signature so the endpoint can answer 400.
    """
    cfg = billplz_config(db)
    key = cfg["x_signature_key"]
    if not key:
        raise ValueError("no X-Signature key configured")
    received = data.get("x_signature", "")
    if not verify_billplz_signature(data, received, key):
        computed = hmac.new(key.encode(), billplz_signature_source(data).encode(), hashlib.sha256).hexdigest()
        logger.warning(
            "Billplz signature mismatch (env=%s, key_len=%d, fields=%s): received=%s computed=%s bill=%s",
            "sandbox" if cfg["sandbox"] else "live", len(key), ",".join(sorted(data)),
            received, computed, data.get("id"),
        )
        raise ValueError("invalid signature")

    bill_id = data.get("id")
    if not bill_id:
        raise ValueError("missing bill id")

    payment = db.query(Payment).filter(Payment.billplz_bill_id == bill_id).first()
    if payment is None:
        logger.warning("Billplz webhook for unknown bill %s", bill_id)
        return "unknown-bill"
    if payment.status == "paid":
        return "already-paid"  # duplicate callback — no-op

    paid = str(data.get("paid", "")).strip().lower() == "true"
    if not paid:
        return "not-paid"

    package = package_for(db, payment.plan_slug)
    if package is None:
        logger.error("paid bill %s references missing package %s", bill_id, payment.plan_slug)
        return "unknown-package"

    sub = subscribe(db, payment.organization_id, package)
    roll_billing_period(sub)
    payment.status = "paid"
    payment.paid_at = datetime.utcnow()
    db.commit()
    logger.info("activated plan %s for org %s (bill %s)", package.slug, payment.organization_id, bill_id)
    return "activated"


# ===== Currency + gateway availability =====

def billing_currency() -> str:
    """ISO currency package prices are quoted in on THIS deployment (MY: MYR, AU: AUD).
    Package.price_myr holds the amount in that currency (column name is historical)."""
    return (config_store.get("BILLING_CURRENCY") or "MYR").strip().upper()


def available_gateways(db: Session | None = None) -> dict:
    """Which ways a client can pay right now. Billplz only collects MYR."""
    return {
        "currency": billing_currency(),
        "billplz": billplz_is_configured(db) and billing_currency() == "MYR",
        "paypal": paypal.is_configured(),
        "bank_transfer": bank_transfer_is_configured(db),
    }


def primary_gateway(avail: dict) -> str | None:
    """The online gateway checkout uses when the client doesn't pick one."""
    return "billplz" if avail["billplz"] else "paypal" if avail["paypal"] else None


# ===== PayPal =====

def create_paypal_checkout(
    db: Session, *, organization_id: int, package: Package,
) -> tuple[Payment, str]:
    """Create a pending Payment + a PayPal order; return (payment, approval_url).

    The Payment row is written first, then tagged with the order id, so a capture
    can always be reconciled to the right org + plan."""
    if not paypal.is_configured():
        raise RuntimeError("PayPal is not configured")
    cfg = paypal.config()
    currency = billing_currency()
    payment = Payment(
        organization_id=organization_id,
        plan_slug=package.slug,
        amount_cents=net_price_cents(db, organization_id, package),
        currency=currency,
        method="paypal",
        status="pending",
        sandbox=cfg["sandbox"],
    )
    db.add(payment)
    db.commit()
    db.refresh(payment)
    base = f"{settings.FRONTEND_URL}/dashboard/billing"
    try:
        order_id, url = paypal.create_order(
            amount_cents=payment.amount_cents,
            currency=currency,
            description=f"AiChatSupport {package.name} plan (monthly)",
            custom_id=f"org{organization_id}:{package.slug}",
            return_url=f"{base}?gateway=paypal",
            cancel_url=f"{base}?gateway=paypal&cancelled=1",
        )
    except paypal.PayPalError as exc:
        payment.status = "failed"
        db.commit()
        raise RuntimeError(str(exc)) from exc
    payment.paypal_order_id = order_id
    db.commit()
    logger.info("created PayPal order %s for org %s plan %s", order_id, organization_id, package.slug)
    return payment, url


def capture_paypal_payment(db: Session, order_id: str, organization_id: int) -> str:
    """Capture an approved PayPal order and activate the plan (idempotent).

    Returns "activated" | "already-paid" | "not-paid". Raises ValueError for an
    unknown order or one that belongs to another organization."""
    payment = db.query(Payment).filter(Payment.paypal_order_id == order_id).first()
    if payment is None or payment.organization_id != organization_id:
        raise ValueError("Unknown PayPal order")
    if payment.status == "paid":
        return "already-paid"

    order = paypal.capture_order(order_id)
    if order.get("status") != "COMPLETED":
        return "not-paid"
    captures = [
        c for u in order.get("purchase_units", [])
        for c in (u.get("payments") or {}).get("captures", [])
    ]
    capture = next((c for c in captures if c.get("status") == "COMPLETED"), None)
    if capture is None:
        return "not-paid"
    paid_cents = round(float(capture["amount"]["value"]) * 100)
    if paid_cents != payment.amount_cents or capture["amount"]["currency_code"] != payment.currency:
        logger.error("PayPal order %s amount mismatch: got %s %s, expected %s %s", order_id,
                     paid_cents, capture["amount"]["currency_code"], payment.amount_cents, payment.currency)
        raise ValueError("Captured amount does not match the order")

    package = package_for(db, payment.plan_slug)
    if package is None:
        logger.error("paid order %s references missing package %s", order_id, payment.plan_slug)
        raise ValueError("Unknown package")
    sub = subscribe(db, payment.organization_id, package)
    roll_billing_period(sub)
    payment.status = "paid"
    payment.paid_at = datetime.utcnow()
    db.commit()
    logger.info("activated plan %s for org %s (PayPal order %s)", package.slug, payment.organization_id, order_id)
    return "activated"


# ===== Bank transfer fallback (manual, admin-confirmed) =====

def bank_transfer_config(db: Session | None = None) -> dict:
    """Resolve manual bank-transfer config (DB settings, no env fallback — this
    is platform-owner personal bank info, never bootstrapped from .env)."""
    g = config_store.get
    return {
        "enabled": _as_bool(g("BANK_TRANSFER_ENABLED"), False),
        "bank_name": g("BANK_NAME"),
        "account_name": g("BANK_ACCOUNT_NAME"),
        "account_number": g("BANK_ACCOUNT_NUMBER"),
        "qr_object_key": g("BANK_QR_OBJECT_KEY"),
        "notify_channel_id": g("BANK_NOTIFY_CHANNEL_ID"),
        "notify_whatsapp_number": g("BANK_NOTIFY_WHATSAPP_NUMBER"),
    }


def bank_transfer_is_configured(db: Session | None = None) -> bool:
    cfg = bank_transfer_config(db)
    return cfg["enabled"] and bool(cfg["account_name"]) and bool(cfg["account_number"])


_QR_MIME_BY_EXT = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif"}


def qr_mime_from_key(object_key: str) -> str:
    """Infer content-type from the stored QR object's extension (see admin QR
    upload, which names the file after the sniffed mime). Falls back to PNG."""
    ext = object_key.rsplit(".", 1)[-1].lower() if "." in object_key else ""
    return _QR_MIME_BY_EXT.get(ext, "image/png")


def report_bank_transfer(
    db: Session,
    *,
    organization_id: int,
    package: Package,
    note: str,
) -> Payment:
    """Record a pending bank-transfer payment for an admin to confirm later.

    There's no gateway here, so the Payment starts (and stays) `pending` until a
    platform admin verifies the transfer against `note` and calls
    `confirm_bank_transfer`.
    """
    if not bank_transfer_is_configured(db):
        raise RuntimeError("Bank transfer is not configured")
    payment = Payment(
        organization_id=organization_id,
        plan_slug=package.slug,
        amount_cents=net_price_cents(db, organization_id, package),
        currency=billing_currency(),
        method="bank_transfer",
        status="pending",
        sandbox=False,
        reference_note=note or None,
        reported_at=datetime.utcnow(),
    )
    db.add(payment)
    db.commit()
    db.refresh(payment)
    logger.info("bank transfer reported for org %s plan %s", organization_id, package.slug)
    return payment


def notify_admin_bank_transfer(db: Session, payment: Payment, org_name: str) -> None:
    """Best-effort WhatsApp ping to the platform admin via the configured device.
    Never raises — a missing/broken device shouldn't block the customer's report."""
    from app.core import wa_provider  # local import: avoid a hard dep for callers that don't need it

    cfg = bank_transfer_config(db)
    channel_id = cfg["notify_channel_id"]
    target = cfg["notify_whatsapp_number"]
    if not channel_id or not target:
        return
    channel = db.query(Channel).filter(Channel.id == int(channel_id)).first()
    if not channel or not channel.access_token:
        logger.warning("bank transfer notify: channel %s not found/unlinked", channel_id)
        return
    message = (
        f"New bank transfer reported.\n"
        f"Client: {org_name}\n"
        f"Plan: {payment.plan_slug}\n"
        f"Amount: {payment.currency} {payment.amount_cents / 100:.2f}\n"
        f"Note: {payment.reference_note or '(none)'}\n"
        f"Confirm in the admin Payments page."
    )
    try:
        wa_provider.send_text(channel, target, message)
    except Exception:  # noqa: BLE001
        logger.exception("failed to send bank transfer WhatsApp notification")


def set_billing_cycle(db: Session, organization_id: int, start_date: datetime) -> Subscription:
    """Admin sets/updates the subscription's monthly renewal anchor."""
    sub = get_or_create_subscription(db, organization_id)
    sub.start_date = start_date
    sub.next_billing_date = start_date + timedelta(days=PERIOD_DAYS)
    db.commit()
    db.refresh(sub)
    return sub


def confirm_bank_transfer(db: Session, payment: Payment) -> Subscription:
    """Admin confirms a reported bank transfer: activate the plan and roll the
    renewal date forward a month from today."""
    if payment.status == "paid":
        raise ValueError("Payment already confirmed")
    package = package_for(db, payment.plan_slug)
    if package is None:
        raise ValueError(f"Unknown package '{payment.plan_slug}'")
    sub = subscribe(db, payment.organization_id, package)
    roll_billing_period(sub)
    payment.status = "paid"
    payment.paid_at = datetime.utcnow()
    db.commit()
    db.refresh(sub)
    return sub


def subscriptions_due_for_reminder(db: Session, within_days: int = 3) -> list[Subscription]:
    """Active paid subscriptions whose renewal is within `within_days` and that
    haven't been reminded since their current `next_billing_date` was set."""
    cutoff = datetime.utcnow() + timedelta(days=within_days)
    return (
        db.query(Subscription)
        .filter(
            Subscription.next_billing_date.isnot(None),
            Subscription.next_billing_date <= cutoff,
            Subscription.plan != DEFAULT_PLAN,
            (Subscription.last_reminder_sent.is_(None))
            | (Subscription.last_reminder_sent < Subscription.next_billing_date - timedelta(days=PERIOD_DAYS)),
        )
        .all()
    )

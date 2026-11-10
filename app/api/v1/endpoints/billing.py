import logging
from urllib.parse import parse_qsl

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.api.deps import get_current_user
from app.core import billing, paypal, storage, email as email_service
from app.models.user import User
from app.core.settings import settings
from app.models.organization import Organization
from app.models.package import Package
from app.schemas.billing import (
    PackageOut,
    SubscriptionOut,
    SubscribeRequest,
    CheckoutOut,
    BankTransferSettingsOut,
    BankTransferReportRequest,
    BankTransferReportOut,
    GatewaysOut,
    PayPalCaptureRequest,
    PayPalCaptureOut,
    ReferralItem,
    ReferralOut,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/packages", response_model=list[PackageOut])
def list_packages(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Active packages a client can subscribe to (pricing page)."""
    return (
        db.query(Package)
        .filter(Package.is_active.is_(True))
        .order_by(Package.sort_order, Package.price_myr)
        .all()
    )


def _subscription_out(db: Session, sub) -> SubscriptionOut:
    pkg = billing.package_for(db, sub.plan)
    quota = pkg.monthly_token_quota if pkg else 0
    return SubscriptionOut(
        plan=sub.plan,
        plan_name=pkg.name if pkg else sub.plan,
        status=sub.status,
        tokens_used=sub.tokens_used or 0,
        tokens_quota=quota,
        tokens_remaining=max(0, quota - (sub.tokens_used or 0)),
        max_bots=pkg.max_bots if pkg else 1,
        period_start=sub.period_start.isoformat() if sub.period_start else None,
        next_billing_date=sub.next_billing_date.isoformat() if sub.next_billing_date else None,
        referral_discount_percent=billing.referral_discount_percent(db, sub.organization_id),
    )


@router.get("/subscription", response_model=SubscriptionOut)
def my_subscription(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    sub = billing.get_or_create_subscription(db, user.organization_id)
    return _subscription_out(db, sub)


@router.post("/subscribe", response_model=SubscriptionOut)
def subscribe(payload: SubscribeRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    pkg = billing.package_for(db, payload.package_slug)
    if not pkg or not pkg.is_active:
        raise HTTPException(status_code=404, detail="Package not found")
    # Only owners/admins of the org may change the plan.
    if user.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only an owner can change the plan")

    sub = billing.get_or_create_subscription(db, user.organization_id)
    # No-op re-subscribe is rejected so it can't be used to reset the usage window.
    if pkg.slug == sub.plan:
        raise HTTPException(status_code=400, detail="You're already on this plan.")
    # Paid plans require collected payment, which isn't wired yet — an admin assigns
    # them manually. (Self-serve is limited to free/downgrade.) Guard on price, not
    # just BILLING_ENABLED, so flipping that flag can't hand out free paid plans.
    if pkg.price_myr > 0:
        raise HTTPException(
            status_code=503,
            detail="Paid plans are activated by our team — contact us to upgrade.",
        )
    sub = billing.subscribe(db, user.organization_id, pkg)
    return _subscription_out(db, sub)


@router.post("/checkout", response_model=CheckoutOut)
def checkout(payload: SubscribeRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Start a plan change.

    Free plan / downgrade applies immediately (no payment). A paid plan creates a
    Billplz bill and returns its `payment_url` for the browser to redirect to; the
    plan is activated by the webhook once payment clears.
    """
    pkg = billing.package_for(db, payload.package_slug)
    if not pkg or not pkg.is_active:
        raise HTTPException(status_code=404, detail="Package not found")
    if user.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only an owner can change the plan")

    sub = billing.get_or_create_subscription(db, user.organization_id)
    # Re-selecting your current *paid* plan is a renewal payment; free has nothing to pay.
    if pkg.slug == sub.plan and pkg.price_myr <= 0:
        raise HTTPException(status_code=400, detail="You're already on this plan.")

    # Free plan / downgrade to free — no payment needed, apply now.
    if pkg.price_myr <= 0:
        sub = billing.subscribe(db, user.organization_id, pkg)
        return CheckoutOut(subscription=_subscription_out(db, sub))

    # Fully covered by referral discount: nothing to collect, activate directly.
    if billing.net_price_cents(db, user.organization_id, pkg) == 0:
        sub = billing.activate_without_payment(db, user.organization_id, pkg)
        return CheckoutOut(subscription=_subscription_out(db, sub))

    # Paid plan — pick the gateway. Billplz collects MYR only; PayPal covers the rest.
    avail = billing.available_gateways(db)
    gateway = payload.gateway or ("billplz" if avail["billplz"] else "paypal" if avail["paypal"] else None)
    if gateway not in ("billplz", "paypal") or not avail.get(gateway):
        raise HTTPException(
            status_code=503,
            detail="Online payment isn't enabled yet — contact us to activate a paid plan.",
        )
    org = db.query(Organization).filter(Organization.id == user.organization_id).first()
    try:
        if gateway == "paypal":
            payment, url = billing.create_paypal_checkout(db, organization_id=user.organization_id, package=pkg)
            return CheckoutOut(payment_url=url, bill_id=payment.paypal_order_id)
        payment, url = billing.create_checkout_bill(
            db,
            organization_id=user.organization_id,
            package=pkg,
            buyer_email=user.email,
            buyer_name=org.name if org else user.email,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Could not start checkout: {exc}")
    return CheckoutOut(payment_url=url, bill_id=payment.billplz_bill_id)


@router.get("/referral", response_model=ReferralOut)
def my_referrals(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """The caller's referral link, discount earned so far and who they referred."""
    org = db.query(Organization).filter(Organization.id == user.organization_id).first()
    code = billing.ensure_referral_code(db, org)
    rows = billing.referral_list(db, user.organization_id)
    active = sum(1 for r in rows if r["counts"])
    return ReferralOut(
        code=code,
        link=f"{settings.FRONTEND_URL}/register?ref={code}",
        percent_each=billing.REFERRAL_PERCENT_EACH,
        active_referrals=active,
        total_referrals=len(rows),
        discount_percent=min(100, active * billing.REFERRAL_PERCENT_EACH),
        referrals=[
            ReferralItem(
                name=(r["organization"].name[:2] + "***"),
                joined=r["organization"].created_at.isoformat() if r["organization"].created_at else "",
                plan=r["plan"],
                counts=r["counts"],
            )
            for r in rows
        ],
    )


@router.get("/gateways", response_model=GatewaysOut)
def gateways(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Which payment methods are live on this deployment + its currency."""
    return GatewaysOut(**billing.available_gateways(db))


@router.post("/paypal/capture", response_model=PayPalCaptureOut)
def paypal_capture(payload: PayPalCaptureRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Called by the billing page after PayPal redirects the payer back with
    `?token=<order id>`. Captures the order and activates the plan."""
    if user.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only an owner can change the plan")
    try:
        status = billing.capture_paypal_payment(db, payload.order_id, user.organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except paypal.PayPalError as exc:
        raise HTTPException(status_code=502, detail=f"Could not confirm PayPal payment: {exc}")
    return PayPalCaptureOut(status=status)


@router.post("/webhook/billplz")
async def billplz_webhook(request: Request, db: Session = Depends(get_db)):
    """Billplz server-to-server payment callback (JWT-less, X-Signature verified).

    Billplz posts `application/x-www-form-urlencoded`; we parse the raw body so no
    multipart dependency is needed. Always answers fast: 200 once verified, 400 on
    a bad/absent signature.
    """
    raw = (await request.body()).decode("utf-8", "replace")
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            data = {str(k): str(v) for k, v in (await request.json()).items()}
        except Exception:
            data = {}
    else:
        data = dict(parse_qsl(raw, keep_blank_values=True))

    try:
        result = billing.process_billplz_webhook(db, data)
    except ValueError as exc:
        logger.warning("rejected Billplz webhook: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": result}


# ===== Bank transfer fallback (manual, admin-confirmed) =====

@router.get("/bank-transfer", response_model=BankTransferSettingsOut)
def bank_transfer_settings(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Bank details for the customer to pay by manual transfer, if enabled."""
    cfg = billing.bank_transfer_config(db)
    qr_url = None
    if cfg["qr_object_key"] and storage.is_configured(db):
        try:
            qr_url = storage.inline_image_url(db, cfg["qr_object_key"], billing.qr_mime_from_key(cfg["qr_object_key"]))
        except Exception:
            qr_url = None
    return BankTransferSettingsOut(
        enabled=cfg["enabled"],
        bank_name=cfg["bank_name"],
        account_name=cfg["account_name"],
        account_number=cfg["account_number"],
        qr_url=qr_url,
        notify_channel_id=int(cfg["notify_channel_id"]) if cfg["notify_channel_id"] else None,
        notify_whatsapp_number=cfg["notify_whatsapp_number"],
        configured=billing.bank_transfer_is_configured(db),
    )


@router.post("/bank-transfer/report", response_model=BankTransferReportOut)
def report_bank_transfer(
    payload: BankTransferReportRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Customer reports they've made a manual bank transfer for a plan. Creates a
    pending payment for an admin to verify and confirm — no gateway involved."""
    pkg = billing.package_for(db, payload.package_slug)
    if not pkg or not pkg.is_active:
        raise HTTPException(status_code=404, detail="Package not found")
    if user.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only an owner can change the plan")
    if not billing.bank_transfer_is_configured(db):
        raise HTTPException(status_code=503, detail="Bank transfer isn't enabled — contact us to activate a paid plan.")

    if billing.net_price_cents(db, user.organization_id, pkg) == 0:
        raise HTTPException(status_code=400, detail="Nothing to pay - your referral discount covers this period. Use Renew instead.")

    org = db.query(Organization).filter(Organization.id == user.organization_id).first()
    try:
        payment = billing.report_bank_transfer(
            db, organization_id=user.organization_id, package=pkg, note=payload.note,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    billing.notify_admin_bank_transfer(db, payment, org.name if org else str(user.organization_id))
    email_service.send_template_bg(
        user.email,
        "bank_transfer_reported",
        {"name": org.name if org else user.email, "plan": pkg.name, "amount": f"{payment.amount_cents / 100:.2f}"},
    )
    return BankTransferReportOut(payment_id=payment.id, status=payment.status)

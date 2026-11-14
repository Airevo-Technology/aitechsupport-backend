"""Public (no-auth) content API for the marketing site: homepage content, public
pricing, published CMS pages, and the contact form."""
import html
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.db.session import get_db, SessionLocal
from app.core import airevo, cms, limits, email, config_store, billing
from app.models.page import Page
from app.models.package import Package
from app.schemas.billing import GatewaysOut
from app.schemas.content import PageListItem, PublicPageOut, PublicPackageOut, ContactRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/homepage")
def homepage(db: Session = Depends(get_db)) -> dict:
    return cms.get_homepage(db)


@router.get("/packages", response_model=list[PublicPackageOut])
def packages(db: Session = Depends(get_db)):
    return (
        db.query(Package)
        .filter(Package.is_active.is_(True))
        .order_by(Package.sort_order, Package.price_myr)
        .all()
    )


@router.get("/gateways", response_model=GatewaysOut)
def public_gateways(db: Session = Depends(get_db)):
    """Which online payment method checkout will use here (no secrets): lets the
    public checkout page name the right gateway before the visitor has an account."""
    avail = billing.available_gateways(db)
    return GatewaysOut(**avail, primary=billing.primary_gateway(avail), marketplace_url=airevo.listing_url() if airevo.sells_paid_plans() else None)


@router.get("/pages", response_model=list[PageListItem])
def pages(db: Session = Depends(get_db)):
    return (
        db.query(Page)
        .filter(Page.is_published.is_(True))
        .order_by(Page.sort_order, Page.title)
        .all()
    )


@router.get("/pages/{slug}", response_model=PublicPageOut)
def page(slug: str, db: Session = Depends(get_db)):
    p = db.query(Page).filter(Page.slug == slug, Page.is_published.is_(True)).first()
    if not p:
        raise HTTPException(status_code=404, detail="Page not found")
    return p


@router.post("/contact", status_code=202)
def contact(payload: ContactRequest, request: Request, background: BackgroundTasks):
    ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
        or (request.client.host if request.client else "unknown")
    if not limits.rate_limit_ok(f"contact:{ip}", 5, 300):
        raise HTTPException(status_code=429, detail="Too many messages. Please try again later.")
    background.add_task(_send_contact, payload.name, payload.email, payload.message)
    return {"ok": True}


def _send_contact(name: str, from_email: str, message: str) -> None:
    """Best-effort email of a contact-form submission to the site operator."""
    db = SessionLocal()
    try:
        if not email.is_configured(db):
            logger.info("contact email skipped (SMTP off)")
            return
        to = config_store.get("CONTACT_EMAIL") or config_store.get("SMTP_FROM_EMAIL")
        if not to:
            logger.info("contact email skipped (no recipient configured)")
            return
        subject = f"Contact form: {name}"
        inner = (
            '<h2 style="margin:0 0 14px;font-size:19px;color:#0f172a;">New contact-form message</h2>'
            f'<p style="margin:0 0 14px;"><strong>Name:</strong> {html.escape(name)}<br>'
            f'<strong>Email:</strong> {html.escape(from_email)}</p>'
            f'<p style="margin:0;white-space:pre-wrap;">{html.escape(message)}</p>'
        )
        body = email.wrap_email(inner, preheader=f"New message from {name}")
        try:
            email.send(db, to, subject, body)
        except Exception:  # noqa: BLE001
            logger.warning("contact email failed", exc_info=True)
    finally:
        db.close()

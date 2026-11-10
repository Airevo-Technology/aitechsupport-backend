import json
import re
from datetime import datetime
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.api.deps import get_platform_admin
from app.core import config_store, models_catalog, billing, paypal, wa_provider, email as email_service, storage, cms, gsc_service
from app.core.settings import settings
from app.models.user import User
from app.models.organization import Organization
from app.models.bot import Bot
from app.models.subscription import Subscription
from app.models.package import Package
from app.models.payment import Payment
from app.models.email_template import EmailTemplate
from app.models.page import Page
from app.models.channel import Channel
from app.models.conversation import Conversation
from app.models.knowledge import KnowledgeSource
from app.core import attachments
from app.schemas.content import HomepageUpdate, PageAdminOut, PageCreate, PageUpdate
from app.schemas.setting import SettingsUpdate, SettingsOut
from app.schemas.site_widget import SiteWidgetOut, SiteWidgetUpdate
from app.schemas.storage import StorageSettingsOut, StorageSettingsUpdate
from app.schemas.billing import (
    PackageOut,
    PackageUpsert,
    ClientRow,
    ClientsPage,
    ClientDetail,
    ClientUserRow,
    ClientBotRow,
    ClientPaymentRow,
    SubscribeRequest,
    BillplzSettingsOut,
    BillplzSettingsUpdate,
    PaymentRow,
    PaymentsSummary,
    PaymentsOut,
    SetBillingCycleRequest,
    BankTransferSettingsOut,
    BankTransferSettingsUpdate,
)
from app.schemas.email import (
    SMTPSettingsOut,
    SMTPSettingsUpdate,
    TestEmailRequest,
    EmailTemplateOut,
    EmailTemplateUpdate,
    EmailPreviewRequest,
    EmailPreviewOut,
)

router = APIRouter()


def _hint(value: str) -> str | None:
    if not value:
        return None
    return "…" + value[-4:] if len(value) > 4 else "…"


@router.get("/settings", response_model=SettingsOut)
def get_settings(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Current platform config. Secrets are never returned in full — only set/hint."""
    ork = config_store.get("OPENROUTER_API_KEY")
    voy = config_store.get("VOYAGE_API_KEY")
    hf = config_store.get("HUGGINGFACE_API_KEY")
    fonnte_tok = config_store.get("FONNTE_ACCOUNT_TOKEN")
    evo_key = config_store.get("EVOLUTION_API_KEY")
    pp = paypal.config_both()
    return SettingsOut(
        openrouter_api_key_set=bool(ork),
        openrouter_api_key_hint=_hint(ork),
        voyage_api_key_set=bool(voy),
        voyage_api_key_hint=_hint(voy),
        openrouter_base_url=config_store.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
        default_chat_model=models_catalog.default_model(),
        fonnte_account_token_set=bool(fonnte_tok),
        fonnte_account_token_hint=_hint(fonnte_tok),
        field_encryption_key_set=bool(config_store.get("FIELD_ENCRYPTION_KEY")),
        embeddings_provider=config_store.get("EMBEDDINGS_PROVIDER") or "voyage",
        embedding_model_openrouter_main=config_store.get("EMBEDDING_MODEL_OPENROUTER_MAIN"),
        embedding_model_openrouter_fallback_1=config_store.get("EMBEDDING_MODEL_OPENROUTER_FALLBACK_1"),
        embedding_model_openrouter_fallback_2=config_store.get("EMBEDDING_MODEL_OPENROUTER_FALLBACK_2"),
        huggingface_api_key_set=bool(hf),
        huggingface_api_key_hint=_hint(hf),
        embedding_model_huggingface=config_store.get("EMBEDDING_MODEL_HUGGINGFACE") or "BAAI/bge-m3",
        evolution_base_url=config_store.get("EVOLUTION_BASE_URL"),
        evolution_api_key_set=bool(evo_key),
        evolution_api_key_hint=_hint(evo_key),
        whatsapp_default_provider=wa_provider.default_provider(),
        paypal_live_client_id=pp["live_client_id"],
        paypal_live_client_secret_set=bool(pp["live_client_secret"]),
        paypal_live_client_secret_hint=_hint(pp["live_client_secret"]),
        paypal_sandbox_client_id=pp["sandbox_client_id"],
        paypal_sandbox_client_secret_set=bool(pp["sandbox_client_secret"]),
        paypal_sandbox_client_secret_hint=_hint(pp["sandbox_client_secret"]),
        paypal_enabled=pp["enabled"],
        paypal_sandbox=pp["sandbox"],
        billing_currency=billing.billing_currency(),
    )


@router.put("/settings", response_model=SettingsOut)
def update_settings(
    payload: SettingsUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_platform_admin),
):
    """Store provided keys. Blank/omitted fields are left unchanged."""
    field_to_key = {
        "openrouter_api_key": "OPENROUTER_API_KEY",
        "voyage_api_key": "VOYAGE_API_KEY",
        "openrouter_base_url": "OPENROUTER_BASE_URL",
        "default_chat_model": "DEFAULT_CHAT_MODEL",
        "fonnte_account_token": "FONNTE_ACCOUNT_TOKEN",
        "field_encryption_key": "FIELD_ENCRYPTION_KEY",
        "embedding_model_openrouter_main": "EMBEDDING_MODEL_OPENROUTER_MAIN",
        "embedding_model_openrouter_fallback_1": "EMBEDDING_MODEL_OPENROUTER_FALLBACK_1",
        "embedding_model_openrouter_fallback_2": "EMBEDDING_MODEL_OPENROUTER_FALLBACK_2",
        "huggingface_api_key": "HUGGINGFACE_API_KEY",
        "embedding_model_huggingface": "EMBEDDING_MODEL_HUGGINGFACE",
        "evolution_base_url": "EVOLUTION_BASE_URL",
        "evolution_api_key": "EVOLUTION_API_KEY",
        "paypal_live_client_id": "PAYPAL_LIVE_CLIENT_ID",
        "paypal_live_client_secret": "PAYPAL_LIVE_CLIENT_SECRET",
        "paypal_sandbox_client_id": "PAYPAL_SANDBOX_CLIENT_ID",
        "paypal_sandbox_client_secret": "PAYPAL_SANDBOX_CLIENT_SECRET",
    }
    updates: dict[str, str] = {}
    for field, key in field_to_key.items():
        value = getattr(payload, field)
        if value is not None and value.strip() != "":
            updates[key] = value.strip()
    if payload.embeddings_provider is not None:
        provider = payload.embeddings_provider.strip().lower()
        if provider not in ("voyage", "openrouter", "huggingface"):
            raise HTTPException(
                status_code=422,
                detail="embeddings_provider must be 'voyage', 'openrouter', or 'huggingface'",
            )
        updates["EMBEDDINGS_PROVIDER"] = provider
    if payload.whatsapp_default_provider is not None:
        wa = payload.whatsapp_default_provider.strip().lower()
        if wa not in wa_provider.PROVIDERS:
            raise HTTPException(status_code=422, detail="whatsapp_default_provider must be 'fonnte' or 'evolution'")
        updates["WHATSAPP_DEFAULT_PROVIDER"] = wa
    if payload.billing_currency is not None and payload.billing_currency.strip():
        cur = payload.billing_currency.strip().upper()
        if len(cur) != 3 or not cur.isalpha():
            raise HTTPException(status_code=422, detail="billing_currency must be a 3-letter ISO code, e.g. MYR or AUD")
        updates["BILLING_CURRENCY"] = cur
    for flag, key in (("paypal_enabled", "PAYPAL_ENABLED"), ("paypal_sandbox", "PAYPAL_SANDBOX")):
        value = getattr(payload, flag)
        if value is not None:
            updates[key] = "true" if value else "false"
    if updates:
        config_store.set_many(db, updates)
    return get_settings(db=db, admin=admin)


@router.post("/paypal/test", status_code=204)
def test_paypal(mode: str | None = None, admin: User = Depends(get_platform_admin)):
    """Check saved PayPal credentials can obtain an access token (?mode=sandbox|live,
    default: whichever pair is active)."""
    try:
        paypal.check_credentials(mode)
    except paypal.PayPalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ===== Support widget on our own site (aichatsupport.my) =====
# Store one bot's embed key so the marketing site can load its own support
# widget. We parse the pasted snippet and keep only the src + public key (never
# raw HTML), so the site injects a clean, validated tag.


def _site_widget_out() -> SiteWidgetOut:
    key = config_store.get("SITE_WIDGET_PUBLIC_KEY")
    src = config_store.get("SITE_WIDGET_SRC")
    enabled = config_store.get("SITE_WIDGET_ENABLED") == "1"
    snippet = (
        f'<script src="{src}" data-public-key="{key}" defer></script>'
        if (src and key)
        else ""
    )
    return SiteWidgetOut(enabled=enabled, public_key=key, src=src, snippet=snippet)


def _valid_widget_src(src: str) -> bool:
    """Only https URLs on our own domain may be loaded on the site.

    Accepts both the current domain and the pre-rebrand one: aitechsupport.my
    now 301-redirects to aichatsupport.my, so an old snippet still resolves —
    no need to force a re-paste just because the domain changed.
    """
    try:
        u = urlparse(src)
    except ValueError:
        return False
    host = (u.hostname or "").lower()
    if u.scheme != "https":
        return False
    return any(
        host == d or host.endswith(f".{d}")
        for d in ("aichatsupport.my", "aitechsupport.my")
    )


@router.get("/site-widget", response_model=SiteWidgetOut)
def get_site_widget(admin: User = Depends(get_platform_admin)) -> SiteWidgetOut:
    return _site_widget_out()


@router.put("/site-widget", response_model=SiteWidgetOut)
def update_site_widget(
    payload: SiteWidgetUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_platform_admin),
) -> SiteWidgetOut:
    updates: dict[str, str] = {"SITE_WIDGET_ENABLED": "1" if payload.enabled else "0"}
    snippet = (payload.snippet or "").strip()
    if snippet:
        src_m = re.search(r"""src\s*=\s*["']([^"']+)["']""", snippet)
        key_m = re.search(r"""data-public-key\s*=\s*["']([^"']+)["']""", snippet)
        if not src_m or not key_m:
            raise HTTPException(
                status_code=400,
                detail="Couldn't read the snippet — paste the full <script …> embed code from a bot's Website widget page.",
            )
        src = src_m.group(1).strip()
        if not _valid_widget_src(src):
            raise HTTPException(
                status_code=400,
                detail="The snippet's script src must be an https URL on aichatsupport.my.",
            )
        updates["SITE_WIDGET_SRC"] = src
        updates["SITE_WIDGET_PUBLIC_KEY"] = key_m.group(1).strip()
    config_store.set_many(db, updates)
    return _site_widget_out()


# ===== Google Search Console (admin SEO panel) =====
# Connect a GSC service-account credential to pull live search metrics.


@router.post("/seo/gsc-config")
def save_gsc_config(
    site_url: str = Form(...),
    service_account_json: str = Form(...),
    admin: User = Depends(get_platform_admin),
    db: Session = Depends(get_db),
) -> dict:
    """Store the GSC service-account credential + property URL."""
    try:
        info = json.loads(service_account_json)
        if not info.get("client_email") or not info.get("private_key"):
            raise ValueError("missing client_email/private_key")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid service account JSON (need a downloaded service-account key).",
        )
    config_store.set_many(
        db,
        {
            "GSC_SITE_URL": site_url.strip(),
            "GSC_SERVICE_ACCOUNT_JSON": service_account_json,
        },
    )
    return {
        "message": "Search Console connected",
        "client_email": info.get("client_email"),
        "site_url": site_url.strip(),
    }


@router.get("/seo/gsc-status")
def gsc_status(admin: User = Depends(get_platform_admin)) -> dict:
    """Whether GSC is configured (never returns the private key)."""
    sa = config_store.get("GSC_SERVICE_ACCOUNT_JSON")
    if not sa:
        return {"configured": False}
    client_email = None
    try:
        client_email = json.loads(sa).get("client_email")
    except Exception:
        pass
    return {
        "configured": True,
        "site_url": config_store.get("GSC_SITE_URL"),
        "client_email": client_email,
    }


@router.get("/seo/metrics")
def gsc_metrics(admin: User = Depends(get_platform_admin)) -> dict:
    """Live Search Console metrics (clicks/impressions/CTR/position + top queries/pages)."""
    sa = config_store.get("GSC_SERVICE_ACCOUNT_JSON")
    site = config_store.get("GSC_SITE_URL")
    if not sa or not site:
        raise HTTPException(status_code=400, detail="Search Console is not configured.")
    try:
        return gsc_service.fetch_metrics(site, sa)
    except Exception as e:
        detail = str(e)
        try:
            import requests

            if isinstance(e, requests.HTTPError) and e.response is not None:
                detail = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=f"Search Console API error: {detail}")


# ===== Packages (the products the operator sells) =====

@router.get("/packages", response_model=list[PackageOut])
def list_packages(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    return db.query(Package).order_by(Package.sort_order, Package.price_myr).all()


@router.post("/packages", response_model=PackageOut, status_code=201)
def create_package(payload: PackageUpsert, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    if db.query(Package).filter(Package.slug == payload.slug).first():
        raise HTTPException(status_code=400, detail="A package with that slug already exists")
    pkg = Package(**payload.model_dump())
    db.add(pkg)
    db.commit()
    db.refresh(pkg)
    return pkg


@router.put("/packages/{package_id}", response_model=PackageOut)
def update_package(package_id: int, payload: PackageUpsert, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    pkg = db.query(Package).filter(Package.id == package_id).first()
    if not pkg:
        raise HTTPException(status_code=404, detail="Package not found")
    for k, v in payload.model_dump().items():
        setattr(pkg, k, v)
    db.commit()
    db.refresh(pkg)
    return pkg


@router.delete("/packages/{package_id}", status_code=204)
def delete_package(package_id: int, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    pkg = db.query(Package).filter(Package.id == package_id).first()
    if not pkg:
        raise HTTPException(status_code=404, detail="Package not found")
    # Refuse to orphan subscribers (plan is referenced by slug, not FK). Deactivate
    # instead of deleting so their quota/limits keep resolving.
    in_use = db.query(Subscription).filter(Subscription.plan == pkg.slug).count()
    if in_use:
        raise HTTPException(
            status_code=400,
            detail=f"{in_use} client(s) are on this package. Set it inactive instead of deleting.",
        )
    db.delete(pkg)
    db.commit()


# ===== Clients (every organization + its plan/usage) =====

@router.get("/clients", response_model=list[ClientRow])
def list_clients(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    orgs = db.query(Organization).order_by(Organization.id.desc()).all()
    subs = {s.organization_id: s for s in db.query(Subscription).all()}
    quotas = {p.slug: p.monthly_token_quota for p in db.query(Package).all()}
    # one owner email per org
    owners = {
        u.organization_id: u.email
        for u in db.query(User).filter(User.role == "owner").order_by(User.id).all()
    }
    bot_counts = dict(
        db.query(Bot.organization_id, func.count(Bot.id)).group_by(Bot.organization_id).all()
    )
    rows = []
    for org in orgs:
        sub = subs.get(org.id)
        plan = sub.plan if sub else "free"
        rows.append(
            ClientRow(
                organization_id=org.id,
                organization_name=org.name,
                owner_email=owners.get(org.id),
                plan=plan,
                tokens_used=(sub.tokens_used if sub else 0) or 0,
                tokens_quota=quotas.get(plan, 0),
                bots=bot_counts.get(org.id, 0),
                created_at=org.created_at.isoformat() if org.created_at else None,
                start_date=sub.start_date.isoformat() if sub and sub.start_date else None,
                next_billing_date=sub.next_billing_date.isoformat() if sub and sub.next_billing_date else None,
            )
        )
    return rows


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


@router.get("/clients/paged", response_model=ClientsPage)
def list_clients_paged(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    q: str | None = Query(default=None, description="Search org name or owner email"),
    db: Session = Depends(get_db),
    admin: User = Depends(get_platform_admin),
) -> ClientsPage:
    query = db.query(Organization)
    if q and q.strip():
        like = f"%{q.strip()}%"
        owner_org_ids = db.query(User.organization_id).filter(
            User.role == "owner", User.email.ilike(like)
        )
        query = query.filter(or_(Organization.name.ilike(like), Organization.id.in_(owner_org_ids)))
    total = query.count()
    orgs = (
        query.order_by(Organization.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    ids = [o.id for o in orgs]
    subs = {s.organization_id: s for s in db.query(Subscription).filter(Subscription.organization_id.in_(ids)).all()} if ids else {}
    quotas = {p.slug: p.monthly_token_quota for p in db.query(Package).all()}
    owners: dict[int, str] = {}
    if ids:
        for u in db.query(User).filter(User.role == "owner", User.organization_id.in_(ids)).order_by(User.id).all():
            owners.setdefault(u.organization_id, u.email)
    bot_counts = dict(
        db.query(Bot.organization_id, func.count(Bot.id))
        .filter(Bot.organization_id.in_(ids))
        .group_by(Bot.organization_id)
        .all()
    ) if ids else {}
    items = []
    for org in orgs:
        sub = subs.get(org.id)
        plan = sub.plan if sub else "free"
        items.append(
            ClientRow(
                organization_id=org.id,
                organization_name=org.name,
                owner_email=owners.get(org.id),
                plan=plan,
                tokens_used=(sub.tokens_used if sub else 0) or 0,
                tokens_quota=quotas.get(plan, 0),
                bots=bot_counts.get(org.id, 0),
                created_at=_iso(org.created_at),
                start_date=_iso(sub.start_date) if sub else None,
                next_billing_date=_iso(sub.next_billing_date) if sub else None,
            )
        )
    return ClientsPage(items=items, total=total, page=page, page_size=page_size)


@router.get("/clients/{organization_id}", response_model=ClientDetail)
def get_client_detail(
    organization_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_platform_admin),
) -> ClientDetail:
    """Read-only overview of one organization for the platform admin."""
    org = db.query(Organization).filter(Organization.id == organization_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Client not found")
    sub = db.query(Subscription).filter(Subscription.organization_id == organization_id).first()
    plan = sub.plan if sub else "free"
    pkg = db.query(Package).filter(Package.slug == plan).first()
    users = (
        db.query(User).filter(User.organization_id == organization_id).order_by(User.id).all()
    )
    owner = next((u for u in users if u.role == "owner"), None)
    bots = db.query(Bot).filter(Bot.organization_id == organization_id).order_by(Bot.id).all()
    source_counts = dict(
        db.query(KnowledgeSource.bot_id, func.count(KnowledgeSource.id))
        .filter(KnowledgeSource.organization_id == organization_id)
        .group_by(KnowledgeSource.bot_id)
        .all()
    )
    channel_kinds: dict[int, list[str]] = {}
    for ch in db.query(Channel).filter(Channel.organization_id == organization_id).all():
        channel_kinds.setdefault(ch.bot_id, []).append(ch.kind)
    conv_total, conv_human, conv_last = (
        db.query(
            func.count(Conversation.id),
            func.count(Conversation.needs_human_at),
            func.max(Conversation.last_message_at),
        )
        .filter(Conversation.organization_id == organization_id)
        .one()
    )
    payments = (
        db.query(Payment)
        .filter(Payment.organization_id == organization_id)
        .order_by(Payment.id.desc())
        .limit(20)
        .all()
    )
    return ClientDetail(
        client=ClientRow(
            organization_id=org.id,
            organization_name=org.name,
            owner_email=owner.email if owner else None,
            plan=plan,
            tokens_used=(sub.tokens_used if sub else 0) or 0,
            tokens_quota=pkg.monthly_token_quota if pkg else 0,
            bots=len(bots),
            created_at=_iso(org.created_at),
            start_date=_iso(sub.start_date) if sub else None,
            next_billing_date=_iso(sub.next_billing_date) if sub else None,
        ),
        subscription_status=sub.status if sub else "active",
        messages_used=(sub.messages_used if sub else 0) or 0,
        period_start=_iso(sub.period_start) if sub else None,
        last_reminder_sent=_iso(sub.last_reminder_sent) if sub else None,
        support_email=org.support_email,
        support_whatsapp=org.support_whatsapp,
        conversations=conv_total or 0,
        conversations_needing_human=conv_human or 0,
        last_conversation_at=_iso(conv_last),
        users=[
            ClientUserRow(
                id=u.id,
                email=u.email,
                full_name=u.full_name,
                phone=u.phone,
                role=u.role,
                is_active=u.is_active,
                is_email_verified=u.is_email_verified,
                last_seen_at=_iso(u.last_seen_at),
                created_at=_iso(u.created_at),
            )
            for u in users
        ],
        bots=[
            ClientBotRow(
                id=b.id,
                name=b.name,
                is_active=b.is_active,
                chat_model=b.chat_model,
                handoff_mode=b.handoff_mode,
                whatsapp_number=b.whatsapp_number,
                knowledge_sources=source_counts.get(b.id, 0),
                channels=sorted(set(channel_kinds.get(b.id, []))),
                created_at=_iso(b.created_at),
            )
            for b in bots
        ],
        payments=[
            ClientPaymentRow(
                id=p.id,
                plan_slug=p.plan_slug,
                amount_cents=p.amount_cents,
                method=p.method,
                status=p.status,
                sandbox=p.sandbox,
                reference_note=p.reference_note,
                paid_at=_iso(p.paid_at),
                created_at=_iso(p.created_at),
            )
            for p in payments
        ],
    )


@router.put("/clients/{organization_id}/plan", response_model=ClientRow)
def set_client_plan(
    organization_id: int,
    payload: SubscribeRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_platform_admin),
):
    """Operator manually assigns a plan to a client (used while online payment is off)."""
    org = db.query(Organization).filter(Organization.id == organization_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Client not found")
    pkg = billing.package_for(db, payload.package_slug)
    if not pkg:
        raise HTTPException(status_code=404, detail="Package not found")
    sub = billing.subscribe(db, organization_id, pkg)
    owner = db.query(User).filter(User.organization_id == organization_id, User.role == "owner").first()
    bots = db.query(func.count(Bot.id)).filter(Bot.organization_id == organization_id).scalar() or 0
    return ClientRow(
        organization_id=org.id,
        organization_name=org.name,
        owner_email=owner.email if owner else None,
        plan=sub.plan,
        tokens_used=sub.tokens_used or 0,
        tokens_quota=pkg.monthly_token_quota,
        bots=bots,
        created_at=org.created_at.isoformat() if org.created_at else None,
        start_date=sub.start_date.isoformat() if sub.start_date else None,
        next_billing_date=sub.next_billing_date.isoformat() if sub.next_billing_date else None,
    )


@router.put("/clients/{organization_id}/billing-date", response_model=ClientRow)
def set_client_billing_date(
    organization_id: int,
    payload: SetBillingCycleRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_platform_admin),
):
    """Operator sets/updates when this client's monthly renewal falls due."""
    org = db.query(Organization).filter(Organization.id == organization_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Client not found")
    try:
        start = datetime.fromisoformat(payload.start_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="start_date must be an ISO date/datetime")
    sub = billing.set_billing_cycle(db, organization_id, start)
    owner = db.query(User).filter(User.organization_id == organization_id, User.role == "owner").first()
    bots = db.query(func.count(Bot.id)).filter(Bot.organization_id == organization_id).scalar() or 0
    quota = db.query(Package).filter(Package.slug == sub.plan).first()
    return ClientRow(
        organization_id=org.id,
        organization_name=org.name,
        owner_email=owner.email if owner else None,
        plan=sub.plan,
        tokens_used=sub.tokens_used or 0,
        tokens_quota=quota.monthly_token_quota if quota else 0,
        bots=bots,
        created_at=org.created_at.isoformat() if org.created_at else None,
        start_date=sub.start_date.isoformat() if sub.start_date else None,
        next_billing_date=sub.next_billing_date.isoformat() if sub.next_billing_date else None,
    )


@router.post("/clients/{organization_id}/send-reminder", status_code=204)
def send_billing_reminder(
    organization_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_platform_admin),
):
    """Operator manually sends this client's renewal-reminder email now."""
    org = db.query(Organization).filter(Organization.id == organization_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Client not found")
    sub = db.query(Subscription).filter(Subscription.organization_id == organization_id).first()
    if not sub or not sub.next_billing_date:
        raise HTTPException(status_code=400, detail="No renewal date set for this client yet")
    owner = db.query(User).filter(User.organization_id == organization_id, User.role == "owner").first()
    if not owner:
        raise HTTPException(status_code=400, detail="This client has no owner to email")
    pkg = billing.package_for(db, sub.plan)
    email_service.send_template_bg(
        owner.email,
        "renewal_reminder",
        {
            "name": org.name,
            "plan": pkg.name if pkg else sub.plan,
            "next_billing_date": sub.next_billing_date.strftime("%d %b %Y"),
            "dashboard_url": f"{settings.FRONTEND_URL}/dashboard/billing",
        },
    )
    sub.last_reminder_sent = datetime.utcnow()
    db.commit()


@router.post("/billing/send-renewal-reminders")
def send_all_renewal_reminders(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)) -> dict:
    """Batch-send reminders to every client due for renewal within 3 days.

    Safe to call repeatedly (e.g. from a daily cron hitting this endpoint) —
    each subscription is only reminded once per renewal cycle."""
    due = billing.subscriptions_due_for_reminder(db)
    sent = 0
    for sub in due:
        org = db.query(Organization).filter(Organization.id == sub.organization_id).first()
        owner = db.query(User).filter(User.organization_id == sub.organization_id, User.role == "owner").first()
        if not org or not owner:
            continue
        pkg = billing.package_for(db, sub.plan)
        email_service.send_template_bg(
            owner.email,
            "renewal_reminder",
            {
                "name": org.name,
                "plan": pkg.name if pkg else sub.plan,
                "next_billing_date": sub.next_billing_date.strftime("%d %b %Y"),
                "dashboard_url": f"{settings.FRONTEND_URL}/dashboard/billing",
            },
        )
        sub.last_reminder_sent = datetime.utcnow()
        sent += 1
    db.commit()
    return {"reminders_sent": sent}


# ===== SMTP / email settings =====

@router.get("/smtp", response_model=SMTPSettingsOut)
def get_smtp(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    cfg = email_service.smtp_config(db)
    return SMTPSettingsOut(
        host=cfg["host"],
        port=cfg["port"],
        username=cfg["username"],
        password_set=bool(cfg["password"]),
        from_email=cfg["from_email"],
        from_name=cfg["from_name"],
        security=cfg["security"],
        enabled=cfg["enabled"],
    )


@router.put("/smtp", response_model=SMTPSettingsOut)
def update_smtp(payload: SMTPSettingsUpdate, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    field_to_key = {
        "host": "SMTP_HOST",
        "port": "SMTP_PORT",
        "username": "SMTP_USERNAME",
        "password": "SMTP_PASSWORD",
        "from_email": "SMTP_FROM_EMAIL",
        "from_name": "SMTP_FROM_NAME",
        "security": "SMTP_SECURITY",
        "enabled": "SMTP_ENABLED",
    }
    if payload.security is not None:
        sec = payload.security.strip().lower()
        if sec not in ("tls", "ssl", "none"):
            raise HTTPException(status_code=422, detail="security must be one of: tls, ssl, none")
        payload.security = sec

    updates: dict[str, str] = {}
    for field, key in field_to_key.items():
        value = getattr(payload, field)
        if value is None:
            continue
        if field == "password" and str(value).strip() == "":
            continue  # blank password -> keep existing
        if field == "enabled":
            updates[key] = "true" if value else "false"
        else:
            updates[key] = str(value)
    if updates:
        config_store.set_many(db, updates)
    return get_smtp(db=db, admin=admin)


@router.post("/smtp/test", status_code=204)
def send_test_email(payload: TestEmailRequest, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    if not email_service.is_configured(db):
        raise HTTPException(status_code=503, detail="Enable and configure SMTP before sending a test.")
    try:
        inner = (
            '<h2 style="margin:0 0 14px;font-size:19px;color:#0f172a;">Your email is working ✅</h2>'
            '<p style="margin:0;">This is a test email from your AiTechSupport SMTP settings. '
            'If you received it, transactional email is configured correctly.</p>'
        )
        email_service.send(
            db,
            payload.to_email,
            "AiTechSupport test email",
            email_service.wrap_email(inner, preheader="SMTP test from AiTechSupport"),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Send failed: {exc}")


# ===== Email templates =====

@router.get("/email-templates", response_model=list[EmailTemplateOut])
def list_email_templates(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    return db.query(EmailTemplate).order_by(EmailTemplate.id).all()


@router.post("/email-templates/preview", response_model=EmailPreviewOut)
def preview_email_template(payload: EmailPreviewRequest, admin: User = Depends(get_platform_admin)):
    """Render the given subject/body with sample values, wrapped in the branded
    shell — so the admin previews exactly what recipients will see."""
    subject, html_out = email_service.preview(payload.subject, payload.body_html)
    return EmailPreviewOut(subject=subject, html=html_out)


@router.put("/email-templates/{key}", response_model=EmailTemplateOut)
def update_email_template(key: str, payload: EmailTemplateUpdate, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    tmpl = db.query(EmailTemplate).filter(EmailTemplate.key == key).first()
    if not tmpl:
        raise HTTPException(status_code=404, detail="Template not found")
    if payload.subject is not None:
        tmpl.subject = payload.subject
    if payload.body_html is not None:
        tmpl.body_html = payload.body_html
    if payload.is_active is not None:
        tmpl.is_active = payload.is_active
    db.commit()
    db.refresh(tmpl)
    return tmpl


# ===== Object storage (AIStor / MinIO) settings =====

@router.get("/storage", response_model=StorageSettingsOut)
def get_storage(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Current storage config. The secret key is never returned in full."""
    cfg = storage.storage_config(db)
    return StorageSettingsOut(
        endpoint=cfg["endpoint"],
        access_key=cfg["access_key"],
        secret_key_set=bool(cfg["secret_key"]),
        secret_key_hint=_hint(cfg["secret_key"]),
        bucket=cfg["bucket"],
        secure=cfg["secure"],
        enabled=cfg["enabled"],
    )


@router.put("/storage", response_model=StorageSettingsOut)
def update_storage(payload: StorageSettingsUpdate, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Store provided storage fields. Blank/omitted fields are left unchanged."""
    updates: dict[str, str] = {}
    if payload.endpoint is not None and payload.endpoint.strip():
        updates["STORAGE_ENDPOINT"] = payload.endpoint.strip()
    if payload.access_key is not None and payload.access_key.strip():
        updates["STORAGE_ACCESS_KEY"] = payload.access_key.strip()
    if payload.secret_key is not None and payload.secret_key.strip():
        updates["STORAGE_SECRET_KEY"] = payload.secret_key.strip()
    if payload.bucket is not None and payload.bucket.strip():
        updates["STORAGE_BUCKET"] = payload.bucket.strip()
    if payload.secure is not None:
        updates["STORAGE_SECURE"] = "true" if payload.secure else "false"
    if payload.enabled is not None:
        updates["STORAGE_ENABLED"] = "true" if payload.enabled else "false"
    if updates:
        config_store.set_many(db, updates)
    return get_storage(db=db, admin=admin)


@router.post("/storage/test", status_code=204)
def test_storage(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Verify the configured credentials can reach the bucket."""
    try:
        storage.test_connection(db)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Storage check failed: {exc}")


# ===== Billplz payment gateway settings =====

@router.get("/billplz", response_model=BillplzSettingsOut)
def get_billplz(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Current Billplz config (both live and sandbox credential sets). API/
    signature keys are never returned in full."""
    both = billing.billplz_config_both(db)
    live, sandbox_creds = both["live"], both["sandbox_creds"]
    return BillplzSettingsOut(
        enabled=both["enabled"],
        sandbox=both["sandbox"],
        configured=billing.billplz_is_configured(db),
        live_api_key_set=bool(live["api_key"]),
        live_api_key_hint=_hint(live["api_key"]),
        live_x_signature_key_set=bool(live["x_signature_key"]),
        live_x_signature_key_hint=_hint(live["x_signature_key"]),
        live_collection_id=live["collection_id"],
        sandbox_api_key_set=bool(sandbox_creds["api_key"]),
        sandbox_api_key_hint=_hint(sandbox_creds["api_key"]),
        sandbox_x_signature_key_set=bool(sandbox_creds["x_signature_key"]),
        sandbox_x_signature_key_hint=_hint(sandbox_creds["x_signature_key"]),
        sandbox_collection_id=sandbox_creds["collection_id"],
    )


@router.put("/billplz", response_model=BillplzSettingsOut)
def update_billplz(payload: BillplzSettingsUpdate, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Store provided Billplz fields. Blank/omitted secrets are left unchanged."""
    updates: dict[str, str] = {}
    if payload.enabled is not None:
        updates["BILLING_ENABLED"] = "true" if payload.enabled else "false"
    if payload.sandbox is not None:
        updates["BILLPLZ_SANDBOX"] = "true" if payload.sandbox else "false"
    if payload.live_api_key is not None and payload.live_api_key.strip():
        updates["BILLPLZ_API_KEY"] = payload.live_api_key.strip()
    if payload.live_x_signature_key is not None and payload.live_x_signature_key.strip():
        updates["BILLPLZ_X_SIGNATURE_KEY"] = payload.live_x_signature_key.strip()
    if payload.sandbox_api_key is not None and payload.sandbox_api_key.strip():
        updates["BILLPLZ_SANDBOX_API_KEY"] = payload.sandbox_api_key.strip()
    if payload.sandbox_x_signature_key is not None and payload.sandbox_x_signature_key.strip():
        updates["BILLPLZ_SANDBOX_X_SIGNATURE_KEY"] = payload.sandbox_x_signature_key.strip()
    # collection ids are not secret; an explicit empty string clears them.
    if payload.live_collection_id is not None:
        updates["BILLPLZ_COLLECTION_ID"] = payload.live_collection_id.strip()
    if payload.sandbox_collection_id is not None:
        updates["BILLPLZ_SANDBOX_COLLECTION_ID"] = payload.sandbox_collection_id.strip()
    if updates:
        config_store.set_many(db, updates)
    return get_billplz(db=db, admin=admin)


@router.post("/billplz/test", status_code=204)
def test_billplz(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Verify the API key (and collection, if set) reach Billplz."""
    try:
        billing.test_billplz(db)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Billplz check failed: {exc}")


# ===== Bank transfer fallback payment (manual, admin-confirmed) =====

def _bank_transfer_out(db: Session) -> BankTransferSettingsOut:
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


@router.get("/bank-transfer", response_model=BankTransferSettingsOut)
def get_bank_transfer(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    return _bank_transfer_out(db)


@router.put("/bank-transfer", response_model=BankTransferSettingsOut)
def update_bank_transfer(payload: BankTransferSettingsUpdate, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    updates: dict[str, str] = {}
    if payload.enabled is not None:
        updates["BANK_TRANSFER_ENABLED"] = "true" if payload.enabled else "false"
    if payload.bank_name is not None:
        updates["BANK_NAME"] = payload.bank_name.strip()
    if payload.account_name is not None:
        updates["BANK_ACCOUNT_NAME"] = payload.account_name.strip()
    if payload.account_number is not None:
        updates["BANK_ACCOUNT_NUMBER"] = payload.account_number.strip()
    if payload.notify_channel_id is not None:
        updates["BANK_NOTIFY_CHANNEL_ID"] = str(payload.notify_channel_id)
    if payload.notify_whatsapp_number is not None:
        updates["BANK_NOTIFY_WHATSAPP_NUMBER"] = payload.notify_whatsapp_number.strip()
    if updates:
        config_store.set_many(db, updates)
    return _bank_transfer_out(db)


@router.post("/bank-transfer/qr", response_model=BankTransferSettingsOut)
async def upload_bank_transfer_qr(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin: User = Depends(get_platform_admin),
):
    """Upload the QR image customers scan to pay by bank transfer."""
    if not storage.is_configured(db):
        raise HTTPException(status_code=503, detail="Object storage isn't configured — set it up under Storage first.")
    data = await file.read()
    mime = attachments.sniff_image_mime(data)
    if mime is None:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, WebP or GIF images are supported.")
    ext = mime.split("/")[-1]
    key = f"platform/bank-transfer-qr.{ext}"
    try:
        storage.put_bytes(db, data, key, mime)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Could not store the QR image: {exc}")
    config_store.set_many(db, {"BANK_QR_OBJECT_KEY": key})
    return _bank_transfer_out(db)


@router.get("/whatsapp-channels")
def list_whatsapp_channels(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)) -> list[dict]:
    """Connected WhatsApp channels an admin can pick from to send bank-transfer
    notifications — any bot's linked device can be reused for this."""
    rows = (
        db.query(Channel, Bot.name, Organization.name)
        .join(Bot, Bot.id == Channel.bot_id)
        .join(Organization, Organization.id == Channel.organization_id)
        .filter(Channel.kind == "whatsapp", Channel.connection_status == "connected")
        .all()
    )
    return [
        {"id": ch.id, "bot_name": bot_name, "organization_name": org_name, "phone_number_id": ch.phone_number_id}
        for ch, bot_name, org_name in rows
    ]


# ===== Payments (Billplz bills + status) =====

@router.get("/payments", response_model=PaymentsOut)
def list_payments(
    status: str | None = Query(default=None, description="Filter by status: pending | paid | failed"),
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
    admin: User = Depends(get_platform_admin),
):
    """Billplz bills across all clients (newest first) + a revenue/status summary.

    The summary is computed over ALL payments; the returned rows are capped by
    `limit` (and optionally filtered by `status`)."""
    # Summary over the whole table (not just the returned page).
    by_status = dict(
        db.query(Payment.status, func.count(Payment.id)).group_by(Payment.status).all()
    )
    live_revenue = (
        db.query(func.coalesce(func.sum(Payment.amount_cents), 0))
        .filter(Payment.status == "paid", Payment.sandbox.is_(False))
        .scalar()
    ) or 0
    summary = PaymentsSummary(
        total=sum(by_status.values()),
        paid=by_status.get("paid", 0),
        pending=by_status.get("pending", 0),
        failed=by_status.get("failed", 0),
        live_revenue_cents=int(live_revenue),
        currency=billing.billing_currency(),
    )

    q = db.query(Payment)
    if status:
        q = q.filter(Payment.status == status)
    rows = q.order_by(Payment.id.desc()).limit(limit).all()
    org_names = {
        o.id: o.name
        for o in db.query(Organization).filter(
            Organization.id.in_({r.organization_id for r in rows})
        ).all()
    } if rows else {}

    payments = [
        PaymentRow(
            id=r.id,
            organization_id=r.organization_id,
            organization_name=org_names.get(r.organization_id),
            plan_slug=r.plan_slug,
            amount_cents=r.amount_cents,
            currency=r.currency,
            method=r.method,
            status=r.status,
            sandbox=r.sandbox,
            billplz_bill_id=r.billplz_bill_id,
            paypal_order_id=r.paypal_order_id,
            reference_note=r.reference_note,
            reported_at=r.reported_at.isoformat() if r.reported_at else None,
            paid_at=r.paid_at.isoformat() if r.paid_at else None,
            created_at=r.created_at.isoformat() if r.created_at else None,
        )
        for r in rows
    ]
    return PaymentsOut(summary=summary, payments=payments)


@router.post("/payments/{payment_id}/confirm", response_model=PaymentRow)
def confirm_payment(payment_id: int, db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    """Operator confirms a reported bank transfer (or any other pending payment)
    after verifying the funds arrived — activates the plan and rolls the renewal
    date forward a month from today."""
    payment = db.query(Payment).filter(Payment.id == payment_id).first()
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    try:
        sub = billing.confirm_bank_transfer(db, payment)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    org = db.query(Organization).filter(Organization.id == payment.organization_id).first()
    owner = db.query(User).filter(User.organization_id == payment.organization_id, User.role == "owner").first()
    pkg = billing.package_for(db, payment.plan_slug)
    if owner:
        email_service.send_template_bg(
            owner.email,
            "payment_confirmed",
            {
                "name": org.name if org else owner.email,
                "plan": pkg.name if pkg else payment.plan_slug,
                "next_billing_date": sub.next_billing_date.strftime("%d %b %Y") if sub.next_billing_date else "",
                "dashboard_url": f"{settings.FRONTEND_URL}/dashboard/billing",
            },
        )
    return PaymentRow(
        id=payment.id,
        organization_id=payment.organization_id,
        organization_name=org.name if org else None,
        plan_slug=payment.plan_slug,
        amount_cents=payment.amount_cents,
        currency=payment.currency,
        method=payment.method,
        status=payment.status,
        sandbox=payment.sandbox,
        billplz_bill_id=payment.billplz_bill_id,
        paypal_order_id=payment.paypal_order_id,
        reference_note=payment.reference_note,
        reported_at=payment.reported_at.isoformat() if payment.reported_at else None,
        paid_at=payment.paid_at.isoformat() if payment.paid_at else None,
        created_at=payment.created_at.isoformat() if payment.created_at else None,
    )


# ===== Homepage content (structured CMS) =====

@router.get("/homepage")
def get_homepage(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)) -> dict:
    return cms.get_homepage(db)


@router.put("/homepage")
def update_homepage(payload: HomepageUpdate, db: Session = Depends(get_db),
                    admin: User = Depends(get_platform_admin)) -> dict:
    return cms.set_homepage(db, payload.content)


# ===== Content pages (About, Privacy, Terms, …) =====

@router.get("/pages", response_model=list[PageAdminOut])
def list_pages(db: Session = Depends(get_db), admin: User = Depends(get_platform_admin)):
    cms.seed_default_pages(db)  # first visit -> starter pages
    return db.query(Page).order_by(Page.sort_order, Page.title).all()


@router.post("/pages", response_model=PageAdminOut, status_code=201)
def create_page(payload: PageCreate, db: Session = Depends(get_db),
                admin: User = Depends(get_platform_admin)):
    if db.query(Page).filter(Page.slug == payload.slug).first():
        raise HTTPException(status_code=409, detail="A page with that slug already exists.")
    p = Page(**payload.model_dump())
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@router.put("/pages/{page_id}", response_model=PageAdminOut)
def update_page(page_id: int, payload: PageUpdate, db: Session = Depends(get_db),
                admin: User = Depends(get_platform_admin)):
    p = db.query(Page).filter(Page.id == page_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Page not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(p, k, v)
    db.commit()
    db.refresh(p)
    return p


@router.delete("/pages/{page_id}", status_code=204)
def delete_page(page_id: int, db: Session = Depends(get_db),
                admin: User = Depends(get_platform_admin)):
    p = db.query(Page).filter(Page.id == page_id).first()
    if p:
        db.delete(p)
        db.commit()

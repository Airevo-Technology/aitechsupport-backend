import secrets

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_reset_token,
    decode_reset_token,
    reset_token_matches,
)
from app.core import email as email_service
from app.core import invites as invites_core
from app.core import billing
from app.core.settings import settings
from app.api.deps import get_current_user
from app.models.organization import Organization
from app.models.user import User
from app.models.bot import Bot
from app.models.org_invite import OrgInvite
from app.schemas.auth import (
    RegisterRequest,
    Token,
    UserProfile,
    UpdateProfileRequest,
    ChangePasswordRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    CheckEmailRequest,
    CheckEmailResponse,
    CheckoutSignupRequest,
)
from app.schemas.invite import InvitePreview, JoinRequest, AcceptInviteRequest

router = APIRouter()


def _slugify(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")


def _unique_org_slug(db: Session, name: str) -> str:
    """Organization.slug is unique, but org names aren't (two customers can both
    be "John Doe") — suffix -2, -3, ... until free, so signup never 500s."""
    base = _slugify(name) or "org"
    slug, n = base, 1
    while db.query(Organization.id).filter(Organization.slug == slug).first():
        n += 1
        slug = f"{base}-{n}"
    return slug


def _referrer_id(db: Session, code: str | None) -> int | None:
    """Org id behind a referral code; unknown/blank codes are ignored, never an error."""
    ref = billing.organization_by_referral_code(db, code)
    return ref.id if ref else None


@router.post("/register", response_model=Token, status_code=201)
def register(payload: RegisterRequest, background: BackgroundTasks, db: Session = Depends(get_db)) -> Token:
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    org = Organization(name=payload.organization_name, slug=_unique_org_slug(db, payload.organization_name))
    org.referred_by_organization_id = _referrer_id(db, payload.referral_code)
    db.add(org)
    db.flush()  # assign org.id before creating the user

    user = User(
        organization_id=org.id,
        email=payload.email,
        hashed_password=get_password_hash(payload.password),
        full_name=payload.full_name,
        phone=(payload.phone or "").strip() or None,
        role="owner",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Best-effort welcome email (no-op if SMTP isn't configured).
    background.add_task(
        email_service.send_template_bg,
        user.email,
        "welcome",
        {"name": user.full_name or org.name, "email": user.email, "dashboard_url": f"{settings.FRONTEND_URL}/dashboard"},
    )

    token = create_access_token({"sub": str(user.id), "org": org.id})
    return Token(access_token=token)


@router.post("/check-email", response_model=CheckEmailResponse)
def check_email(payload: CheckEmailRequest, db: Session = Depends(get_db)) -> CheckEmailResponse:
    """Public: does an account already exist for this email? Powers the
    single-page checkout (ask for a password vs. show signup fields)."""
    email = payload.email.strip().lower()
    exists = db.query(User).filter(func.lower(User.email) == email).first() is not None
    return CheckEmailResponse(exists=exists)


@router.post("/checkout-signup", status_code=204)
def checkout_signup(payload: CheckoutSignupRequest, background: BackgroundTasks, db: Session = Depends(get_db)):
    """Public: single-page checkout signup. No password is collected here — a
    random one is generated (never revealed) and the user verifies their email
    and sets their real password via the same link, reusing the reset-password
    token/flow."""
    email = payload.email.strip().lower()
    if db.query(User).filter(func.lower(User.email) == email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    pkg = billing.package_for(db, payload.plan_slug)
    if not pkg or not pkg.is_active:
        raise HTTPException(status_code=404, detail="Plan not found")

    org = Organization(name=payload.organization_name.strip(), slug=_unique_org_slug(db, payload.organization_name))
    org.referred_by_organization_id = _referrer_id(db, payload.referral_code)
    db.add(org)
    db.flush()  # assign org.id before creating the user

    user = User(
        organization_id=org.id,
        email=email,
        hashed_password=get_password_hash(secrets.token_urlsafe(24)),
        full_name=(payload.full_name or "").strip() or None,
        phone=(payload.phone or "").strip() or None,
        role="owner",
        is_email_verified=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Free plan applies immediately; paid plans are activated by the team after
    # signup (self-serve payment isn't wired — see billing.subscribe).
    billing.get_or_create_subscription(db, org.id)

    token = create_reset_token(user.id, user.hashed_password)
    verify_url = f"{settings.FRONTEND_URL}/reset-password?token={token}&verify=1"
    background.add_task(
        email_service.send_template_bg,
        user.email,
        "verify_email",
        {"name": user.full_name or org.name, "email": user.email, "plan": pkg.name, "verify_url": verify_url},
    )


@router.get("/invite/{code}", response_model=InvitePreview)
def preview_invite(code: str, db: Session = Depends(get_db)) -> InvitePreview:
    """Public: show which org an invite code joins + whether it's still valid."""
    inv = db.query(OrgInvite).filter(OrgInvite.code == code).first()
    if inv is None:
        raise HTTPException(status_code=404, detail="Invite not found")
    org = db.query(Organization).filter(Organization.id == inv.organization_id).first()
    return InvitePreview(
        organization_name=org.name if org else "",
        role=inv.role,
        valid=invites_core.is_valid(inv),
    )


@router.post("/join", response_model=Token, status_code=201)
def join(payload: JoinRequest, background: BackgroundTasks, db: Session = Depends(get_db)) -> Token:
    """Public: create an account in the invite's org + role, redeeming the code.

    The org and role come from the invite row only — never the request body."""
    inv = db.query(OrgInvite).filter(OrgInvite.code == payload.code).first()
    if inv is None or not invites_core.is_valid(inv):
        raise HTTPException(status_code=400, detail="This invite is invalid or has expired.")

    email = payload.email.strip().lower()
    if db.query(User).filter(func.lower(User.email) == email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    # Atomically consume one use (closes the single-use race); undone if the commit
    # below fails, so the whole join is all-or-nothing.
    if inv.max_uses is not None:
        consumed = (
            db.query(OrgInvite)
            .filter(
                OrgInvite.id == inv.id,
                or_(OrgInvite.max_uses.is_(None), OrgInvite.uses < OrgInvite.max_uses),
            )
            .update({OrgInvite.uses: OrgInvite.uses + 1}, synchronize_session=False)
        )
        if consumed == 0:
            db.rollback()
            raise HTTPException(status_code=400, detail="This invite has already been used.")
    else:
        db.query(OrgInvite).filter(OrgInvite.id == inv.id).update(
            {OrgInvite.uses: OrgInvite.uses + 1}, synchronize_session=False
        )

    user = User(
        organization_id=inv.organization_id,
        email=email,
        hashed_password=get_password_hash(payload.password),
        full_name=(payload.full_name or "").strip() or None,
        role=inv.role if inv.role in ("agent", "admin") else "agent",
        is_active=True,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Email already registered")
    db.refresh(user)

    background.add_task(
        email_service.send_template_bg,
        user.email,
        "welcome",
        {"name": user.full_name or user.email, "email": user.email, "dashboard_url": f"{settings.FRONTEND_URL}/dashboard"},
    )
    token = create_access_token({"sub": str(user.id), "org": inv.organization_id})
    return Token(access_token=token)


@router.post("/accept-invite", response_model=UserProfile)
def accept_invite(
    payload: AcceptInviteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> User:
    """Move the SIGNED-IN user into the invite's org + role (for people who already
    have an account). Guarded so an owner with real data can't orphan their org."""
    inv = db.query(OrgInvite).filter(OrgInvite.code == payload.code.strip()).first()
    if inv is None or not invites_core.is_valid(inv):
        raise HTTPException(status_code=400, detail="This invite is invalid or has expired.")
    if inv.organization_id == current_user.organization_id:
        raise HTTPException(status_code=400, detail="You're already a member of this organization.")

    old_org_id = current_user.organization_id
    # Don't let an owner abandon an org that still has data (bots) or other members.
    if current_user.role == "owner":
        others = db.query(User).filter(User.organization_id == old_org_id, User.id != current_user.id).count()
        bots = db.query(Bot).filter(Bot.organization_id == old_org_id).count()
        if others > 0 or bots > 0:
            raise HTTPException(
                status_code=409,
                detail="You own an organization with data (bots or other members). Hand it over or remove it before joining another org.",
            )

    # Consume one use atomically (all-or-nothing with the move below).
    if inv.max_uses is not None:
        consumed = (
            db.query(OrgInvite)
            .filter(OrgInvite.id == inv.id, or_(OrgInvite.max_uses.is_(None), OrgInvite.uses < OrgInvite.max_uses))
            .update({OrgInvite.uses: OrgInvite.uses + 1}, synchronize_session=False)
        )
        if consumed == 0:
            db.rollback()
            raise HTTPException(status_code=400, detail="This invite has already been used.")
    else:
        db.query(OrgInvite).filter(OrgInvite.id == inv.id).update(
            {OrgInvite.uses: OrgInvite.uses + 1}, synchronize_session=False
        )

    current_user.organization_id = inv.organization_id
    current_user.role = inv.role if inv.role in ("agent", "admin") else "agent"
    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> Token:
    user = db.query(User).filter(User.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")
    token = create_access_token({"sub": str(user.id), "org": user.organization_id})
    return Token(access_token=token)


@router.post("/forgot-password", status_code=204)
def forgot_password(payload: ForgotPasswordRequest, background: BackgroundTasks, db: Session = Depends(get_db)):
    """Public. Always returns 204 regardless of whether the email exists, so
    the endpoint can't be used to enumerate registered accounts."""
    user = db.query(User).filter(func.lower(User.email) == payload.email.strip().lower()).first()
    if user and user.is_active:
        token = create_reset_token(user.id, user.hashed_password)
        reset_url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
        background.add_task(
            email_service.send_template_bg,
            user.email,
            "password_reset",
            {"name": user.full_name or user.email, "email": user.email, "reset_url": reset_url},
        )


@router.post("/reset-password", status_code=204)
def reset_password(payload: ResetPasswordRequest, background: BackgroundTasks, db: Session = Depends(get_db)):
    if len(payload.new_password) < 6:
        raise HTTPException(status_code=422, detail="New password must be at least 6 characters")

    claims = decode_reset_token(payload.token)
    if claims is None:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")
    try:
        user_id = int(claims["sub"])
    except (KeyError, ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not reset_token_matches(claims, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")

    user.hashed_password = get_password_hash(payload.new_password)
    user.is_email_verified = True
    db.commit()

    background.add_task(
        email_service.send_template_bg,
        user.email,
        "password_changed",
        {"name": user.full_name or user.email, "email": user.email},
    )


@router.get("/me", response_model=UserProfile)
def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@router.put("/me", response_model=UserProfile)
def update_me(
    payload: UpdateProfileRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> User:
    if payload.full_name is not None:
        current_user.full_name = payload.full_name.strip() or None
    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/change-password", status_code=204)
def change_password(
    payload: ChangePasswordRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if len(payload.new_password) < 6:
        raise HTTPException(status_code=422, detail="New password must be at least 6 characters")
    current_user.hashed_password = get_password_hash(payload.new_password)
    db.commit()

    background.add_task(
        email_service.send_template_bg,
        current_user.email,
        "password_changed",
        {"name": current_user.full_name or current_user.email, "email": current_user.email},
    )

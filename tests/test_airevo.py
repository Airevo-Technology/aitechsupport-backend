"""airevo.com.au marketplace integration (app/core/airevo.py).

Runs against the local dev database (scripts/dev-db.sh up; alembic upgrade head)
with the seeded free/starter/pro/business packages. Every test uses its own
emails and Airevo subscription ids, so runs don't collide.
"""
import json
import secrets
import time

import pytest
from fastapi.testclient import TestClient

from app.core import airevo, billing
from app.core.security import create_access_token, get_password_hash
from app.db.session import SessionLocal
from app.models.organization import Organization
from app.models.subscription import Subscription
from app.models.user import User
from main import app

KEY_ID = "ak_test_aichat"
SECRET = "sk_test_4f9c1e7a2b6d8035a1c4e9f07b3d52a6"
PATH = "/api/v1/airevo/events"


@pytest.fixture(autouse=True)
def airevo_settings(monkeypatch):
    monkeypatch.setenv("AIREVO_ENABLED", "true")
    monkeypatch.setenv("AIREVO_KEY_ID", KEY_ID)
    monkeypatch.setenv("AIREVO_SECRET", SECRET)
    monkeypatch.setenv("AIREVO_API_BASE", "https://airevo.test/api/marketplace/v1")
    monkeypatch.setenv("AIREVO_LISTING_URL", "https://airevo.test/marketplace/aichatsupport")
    monkeypatch.delenv("AIREVO_SECRET_PREVIOUS", raising=False)


@pytest.fixture
def emails(monkeypatch):
    sent: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(airevo.email_service, "send_template_bg", lambda to, key, ctx: sent.append((to, key, ctx)))
    return sent


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()


def _uid() -> int:
    return secrets.randbelow(10**9) + 10**6


def _email() -> str:
    return f"buyer-{secrets.token_hex(4)}@example.com.au"


def _event(kind: str, sub_id: int, email: str = "", plan: str = "Starter", created: int | None = None, **extra) -> dict:
    return {
        "id": f"evt_{secrets.token_hex(6)}",
        "type": kind,
        "created": created or int(time.time()),
        "livemode": False,
        "subscription": {
            "id": sub_id,
            "status": "active",
            "product": "aichatsupport",
            "plan": {"name": plan, "interval_months": 1},
            "trial_ends_at": None,
            "current_period_end": "2026-12-14T00:00:00+11:00",
            "external_id": None,
        },
        "customer": {"name": "Jane Citizen", "email": email, "business_name": "Jane Citizen Pty Ltd", "country": "AU"},
        **extra,
    }


def _send(client: TestClient, event: dict, secret: str = SECRET, key: str = KEY_ID, ts: int | None = None):
    raw = json.dumps(event).encode()
    stamp = str(ts if ts is not None else int(time.time()))
    headers = {
        "X-Airevo-Key-Id": key,
        "X-Airevo-Timestamp": stamp,
        "X-Airevo-Signature": airevo.signature(secret, stamp, "POST", PATH, raw),
        "Content-Type": "application/json",
    }
    return client.post(PATH, content=raw, headers=headers)


def _sub(db, airevo_id: int) -> Subscription:
    db.expire_all()
    return db.query(Subscription).filter(Subscription.airevo_subscription_id == airevo_id).one()


def _make_user(db, role: str = "owner") -> User:
    org = Organization(name="Existing Co", slug=f"existing-{secrets.token_hex(4)}")
    db.add(org)
    db.flush()
    user = User(organization_id=org.id, email=_email(), hashed_password=get_password_hash("pw-123456"), role=role)
    db.add(user)
    db.commit()
    return user


# ===== Signing =====


def test_signature_matches_the_guide_test_vector():
    body = b'{"id":"evt_01JD3Z","type":"subscription.provision","created":1767225600}'
    assert airevo.signature(SECRET, "1767225600", "POST", "/airevo/v1/events", body) == (
        "v1=acbb04243619621316a51aede55fb3855cf64ca3e8edfd3ab7315ed25280096d"
    )


def test_bad_signature_wrong_key_and_old_timestamp_are_refused(client):
    event = _event("subscription.provision", _uid(), _email())
    assert _send(client, event, secret="sk_wrong").status_code == 401
    assert _send(client, event, key="ak_other").status_code == 401
    assert _send(client, event, ts=int(time.time()) - 301).status_code == 401


def test_the_previous_secret_still_verifies_during_a_rotation(client, monkeypatch, emails):
    monkeypatch.setenv("AIREVO_SECRET", "sk_new_secret")
    monkeypatch.setenv("AIREVO_SECRET_PREVIOUS", SECRET)
    assert _send(client, _event("subscription.provision", _uid(), _email())).status_code == 200


def test_events_are_refused_while_the_marketplace_is_off(client, monkeypatch):
    monkeypatch.setenv("AIREVO_ENABLED", "false")
    assert _send(client, _event("subscription.provision", _uid(), _email())).status_code == 404


# ===== Provisioning =====


def test_provision_creates_an_account_on_the_plan_and_sends_one_welcome(client, db, emails):
    sub_id, email = _uid(), _email()
    event = _event("subscription.provision", sub_id, email, plan="Pro")

    res = _send(client, event)
    assert res.status_code == 200
    body = res.json()
    sub = _sub(db, sub_id)
    assert body["external_id"] == f"org_{sub.organization_id}"
    assert body["access"]["username"] == email
    assert (sub.plan, sub.status, sub.billing_source, sub.airevo_sso_allowed) == ("pro", "active", "airevo", True)
    assert sub.next_billing_date is None
    owner = db.query(User).filter(User.organization_id == sub.organization_id).one()
    assert (owner.email, owner.role, owner.is_email_verified) == (email, "owner", False)
    assert [e[1] for e in emails] == ["verify_email"]
    assert "reset-password?token=" in emails[0][2]["verify_url"]

    # The same event again: same reply, nothing new created or sent.
    again = _send(client, event)
    assert again.status_code == 200 and again.json() == body
    assert len(emails) == 1
    assert db.query(User).filter(User.email == email).count() == 1


def test_a_purchase_by_an_existing_owner_upgrades_their_account_without_sso(client, db, emails):
    user = _make_user(db)
    sub_id = _uid()

    res = _send(client, _event("subscription.provision", sub_id, user.email.upper()))
    assert res.status_code == 200
    sub = _sub(db, sub_id)
    assert sub.organization_id == user.organization_id
    assert (sub.plan, sub.billing_source, sub.airevo_sso_allowed) == ("starter", "airevo", False)
    assert "usual password" in res.json()["access"]["instructions"]
    assert emails == []


def test_a_purchase_with_a_team_members_email_is_refused(client, db):
    agent = _make_user(db, role="agent")
    res = _send(client, _event("subscription.provision", _uid(), agent.email))
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "email_in_use"


def test_an_unknown_or_free_plan_is_refused(client):
    for plan in ("Enterprise", "Free"):
        res = _send(client, _event("subscription.provision", _uid(), _email(), plan=plan))
        assert res.status_code == 422
        assert res.json()["error"]["code"] == "unknown_plan"


def test_a_second_airevo_subscription_for_the_same_account_is_refused(client, db, emails):
    email = _email()
    assert _send(client, _event("subscription.provision", _uid(), email)).status_code == 200
    res = _send(client, _event("subscription.provision", _uid(), email, plan="Pro"))
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "already_subscribed"


# ===== Lifecycle =====


def test_suspend_stops_the_bots_resume_restores_them_and_updated_changes_plan(client, db, emails):
    sub_id = _uid()
    t = int(time.time())
    _send(client, _event("subscription.provision", sub_id, _email(), created=t))
    assert billing.has_quota(db, _sub(db, sub_id))

    assert _send(client, _event("subscription.suspend", sub_id, created=t + 1)).status_code == 200
    assert _sub(db, sub_id).status == "past_due"
    assert not billing.has_quota(db, _sub(db, sub_id))

    assert _send(client, _event("subscription.resume", sub_id, created=t + 2)).status_code == 200
    assert billing.has_quota(db, _sub(db, sub_id))

    assert _send(client, _event("subscription.updated", sub_id, plan="Business", created=t + 3)).status_code == 200
    assert _sub(db, sub_id).plan == "business"


def test_end_drops_to_free_keeps_data_and_a_return_restores_the_same_account(client, db, emails):
    sub_id = _uid()
    t = int(time.time())
    first = _send(client, _event("subscription.provision", sub_id, _email(), plan="Pro", created=t)).json()

    assert _send(client, _event("subscription.end", sub_id, created=t + 1)).status_code == 200
    sub = _sub(db, sub_id)
    assert (sub.plan, sub.billing_source, sub.status) == ("free", "direct", "active")

    back = _send(client, _event("subscription.provision", sub_id, created=t + 2, plan="Pro"))
    assert back.status_code == 200
    assert back.json()["external_id"] == first["external_id"]
    assert (_sub(db, sub_id).plan, _sub(db, sub_id).billing_source) == ("pro", "airevo")
    assert len(emails) == 1  # no second welcome on return


def test_an_older_event_cannot_undo_a_newer_one(client, db, emails):
    sub_id = _uid()
    t = int(time.time())
    _send(client, _event("subscription.provision", sub_id, _email(), created=t))
    _send(client, _event("subscription.resume", sub_id, created=t + 10))
    assert _send(client, _event("subscription.suspend", sub_id, created=t + 5)).status_code == 200
    assert _sub(db, sub_id).status == "active"


def test_an_event_for_an_unknown_subscription_is_a_permanent_404(client):
    res = _send(client, _event("subscription.suspend", _uid()))
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "unknown_subscription"


def test_ticket_and_unknown_events_are_acknowledged(client):
    for kind in ("ticket.created", "something.new"):
        assert _send(client, {"id": f"evt_{secrets.token_hex(6)}", "type": kind, "created": int(time.time())}).status_code == 200


# ===== Direct billing stays out of the way =====


def _auth(user: User) -> dict:
    return {"Authorization": f"Bearer {create_access_token({'sub': str(user.id), 'org': user.organization_id})}"}


def test_direct_checkout_is_closed_for_airevo_plans_and_paid_plans(client, db, emails):
    sub_id = _uid()
    _send(client, _event("subscription.provision", sub_id, _email()))
    owner = db.query(User).filter(User.organization_id == _sub(db, sub_id).organization_id).one()
    res = client.post("/api/v1/billing/checkout", json={"package_slug": "free"}, headers=_auth(owner))
    assert res.status_code == 409 and "airevo.com.au" in res.json()["detail"]

    free_user = _make_user(db)
    res = client.post("/api/v1/billing/checkout", json={"package_slug": "pro"}, headers=_auth(free_user))
    assert res.status_code == 409 and "airevo.com.au" in res.json()["detail"]

    out = client.get("/api/v1/billing/subscription", headers=_auth(owner)).json()
    assert out["billing_source"] == "airevo"
    assert out["manage_url"] == f"https://airevo.test/portal/subscriptions/{sub_id}"
    assert client.get("/api/v1/content/gateways").json()["marketplace_url"] == "https://airevo.test/marketplace/aichatsupport"


def test_renewal_reminders_skip_airevo_billed_subscriptions(client, db, emails):
    from datetime import datetime, timedelta

    sub_id = _uid()
    _send(client, _event("subscription.provision", sub_id, _email()))
    sub = _sub(db, sub_id)
    sub.next_billing_date = datetime.utcnow() + timedelta(days=1)
    db.commit()
    assert sub.id not in [s.id for s in billing.subscriptions_due_for_reminder(db)]


# ===== Single sign-on =====


def test_sso_signs_a_marketplace_created_account_in(client, db, emails, monkeypatch):
    sub_id = _uid()
    org_ref = _send(client, _event("subscription.provision", sub_id, _email())).json()["external_id"]
    monkeypatch.setattr(airevo, "sso_verify", lambda token: {"valid": True, "subscription_id": sub_id, "external_id": org_ref})

    res = client.get("/api/v1/airevo/sso", params={"token": "sso_ok"}, follow_redirects=False)
    assert res.status_code == 303
    assert "/sso#token=" in res.headers["location"]


def test_sso_never_enters_an_account_that_existed_before(client, db, emails, monkeypatch):
    user = _make_user(db)
    sub_id = _uid()
    _send(client, _event("subscription.provision", sub_id, user.email))
    monkeypatch.setattr(
        airevo, "sso_verify",
        lambda token: {"valid": True, "subscription_id": sub_id, "external_id": f"org_{user.organization_id}"},
    )
    res = client.get("/api/v1/airevo/sso", params={"token": "sso_x"}, follow_redirects=False)
    assert res.headers["location"].endswith("/login?from=airevo")


def test_sso_refuses_invalid_tokens_and_suspended_accounts(client, db, emails, monkeypatch):
    sub_id = _uid()
    t = int(time.time())
    org_ref = _send(client, _event("subscription.provision", sub_id, _email(), created=t)).json()["external_id"]

    monkeypatch.setattr(airevo, "sso_verify", lambda token: {"valid": False})
    assert client.get("/api/v1/airevo/sso", params={"token": "bad"}, follow_redirects=False).headers["location"].endswith("/login?from=airevo")

    _send(client, _event("subscription.suspend", sub_id, created=t + 1))
    monkeypatch.setattr(airevo, "sso_verify", lambda token: {"valid": True, "subscription_id": sub_id, "external_id": org_ref})
    assert client.get("/api/v1/airevo/sso", params={"token": "ok"}, follow_redirects=False).headers["location"].endswith("/login?from=airevo")


def test_sso_verify_call_is_signed_over_the_api_path(monkeypatch):
    captured = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"valid": False}

    def fake_post(url, content, headers, timeout):
        captured.update(url=url, content=content, headers=headers)
        return Resp()

    monkeypatch.setattr(airevo.httpx, "post", fake_post)
    airevo.sso_verify("sso_abc")
    h = captured["headers"]
    assert captured["url"] == "https://airevo.test/api/marketplace/v1/sso/verify"
    assert h["X-Airevo-Signature"] == airevo.signature(
        SECRET, h["X-Airevo-Timestamp"], "POST", "/api/marketplace/v1/sso/verify", captured["content"]
    )

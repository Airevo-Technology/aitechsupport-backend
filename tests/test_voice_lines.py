"""AI phone lines (app/core/voice_lines.py, voice endpoints).

Telnyx and ElevenLabs are faked. Database-backed tests run on in-memory SQLite with
only the tables involved, so they need no Postgres.
"""
import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1.endpoints import voice
from app.core import billing, elevenlabs, telnyx, voice_lines
from app.db.session import Base, get_db
from app.models.bot import Bot
from app.models.channel import Channel
from app.models.conversation import Conversation, Message
from app.models.organization import Organization
from app.models.package import Package
from app.models.subscription import Subscription
from app.models.voice_line import VoiceCall, VoiceLine

TABLES = [Organization.__table__, Bot.__table__, Channel.__table__, Conversation.__table__, Message.__table__,
          Package.__table__, Subscription.__table__, VoiceLine.__table__, VoiceCall.__table__]


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine, tables=TABLES)
    session = sessionmaker(bind=engine)()
    org = Organization(id=1, name="Acme", slug="acme")
    session.add_all([org, Bot(id=7, organization_id=1, name="Acme Bot"),
                     Package(slug="voice", name="Pro Voice", price_myr=99, voice_enabled=True, monthly_voice_minutes=10),
                     Subscription(organization_id=1, plan="voice", status="active", tokens_used=0, voice_seconds_used=0)])
    session.commit()
    yield session
    session.close()


@pytest.fixture
def settings_store(monkeypatch):
    store = {"VOICE_ENABLED": "true", "TELNYX_SIP_CONNECTION_ID": "conn-1",
             "VOICE_CATALOG": json.dumps([{"voice_id": "claire", "name": "Claire"}])}
    monkeypatch.setattr(voice_lines.config_store, "get", lambda k, d="": store.get(k, d))
    monkeypatch.setattr(telnyx.config_store, "get", lambda k, d="": store.get(k, d))
    monkeypatch.setattr(elevenlabs.config_store, "get", lambda k, d="": store.get(k, d))
    monkeypatch.setattr(voice.config_store, "get", lambda k, d="": store.get(k, d))
    return store


def _line(db, **kw) -> VoiceLine:
    line = VoiceLine(organization_id=1, bot_id=7, status=kw.pop("status", "draft"), **kw)
    db.add(line)
    db.commit()
    return line


DETAILS = {"business_name": "Acme Pty Ltd", "contact_name": "Jane Citizen", "contact_phone": "0412 345 678",
           "street_address": "7 Ferguson Street", "locality": "Broadford", "administrative_area": "vic",
           "postal_code": "3658", "preferred_locality": "Sydney"}


# ---- phone numbers ------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("0412 345 678", "+61412345678"), ("+61 2 8390 2245", "+61283902245"), ("61283902245", "+61283902245"),
    ("+60 12 345 6789", "+60123456789"), ("12345", ""), ("", ""),
])
def test_normalize_au_phone(raw, want):
    assert voice_lines.normalize_au_phone(raw) == want


# ---- subscriber request -------------------------------------------------------------

def test_submit_stores_details_and_document(db, settings_store, monkeypatch):
    stored = {}
    monkeypatch.setattr(voice_lines.storage, "put_bytes", lambda d, data, key, mime: stored.update(key=key, data=data))
    line = voice_lines.submit_request(db, voice_lines.line_for_bot(db, db.get(Bot, 7)), DETAILS, (b"%PDF", "extract.pdf", "application/pdf"))
    assert line.status == "submitted" and line.submitted_at
    assert line.contact_phone == "+61412345678" and line.administrative_area == "VIC"
    assert line.document_key == stored["key"] and stored["key"].startswith("voice-docs/1/")


@pytest.mark.parametrize("change,message", [
    ({"postal_code": "36"}, "Postcode"), ({"administrative_area": "XX"}, "State"),
    ({"contact_phone": "abc"}, "Australian number"), ({"business_name": ""}, "business name"),
])
def test_submit_validates(db, settings_store, change, message):
    line = voice_lines.line_for_bot(db, db.get(Bot, 7))
    with pytest.raises(voice_lines.VoiceLineError) as e:
        voice_lines.submit_request(db, line, {**DETAILS, **change}, (b"x", "a.pdf", "application/pdf"))
    assert message in e.value.message


def test_submit_needs_a_document_and_locks_after_submission(db, settings_store, monkeypatch):
    line = voice_lines.line_for_bot(db, db.get(Bot, 7))
    with pytest.raises(voice_lines.VoiceLineError, match="proof-of-address"):
        voice_lines.submit_request(db, line, DETAILS, None)
    monkeypatch.setattr(voice_lines.storage, "put_bytes", lambda *a: None)
    voice_lines.submit_request(db, line, DETAILS, (b"x", "a.pdf", "application/pdf"))
    with pytest.raises(voice_lines.VoiceLineError) as e:
        voice_lines.submit_request(db, line, DETAILS, None)
    assert e.value.status == 409


def test_settings_only_accept_listed_voices_and_valid_numbers(db, settings_store):
    line = voice_lines.line_for_bot(db, db.get(Bot, 7))
    with pytest.raises(voice_lines.VoiceLineError):
        voice_lines.update_settings(db, line, {"voice_id": "not-listed"})
    with pytest.raises(voice_lines.VoiceLineError):
        voice_lines.update_settings(db, line, {"transfer_number": "nope"})
    voice_lines.update_settings(db, line, {"voice_id": "claire", "transfer_number": "02 9000 1234", "max_call_seconds": 99999})
    assert line.voice_id == "claire" and line.transfer_number == "+61290001234" and line.max_call_seconds == 1800


# ---- ordering -----------------------------------------------------------------------

def test_order_answers_each_requirement_by_type(db, settings_store, monkeypatch):
    line = _line(db, status="submitted", business_name="Acme Pty Ltd", contact_name="Jane Citizen",
                 contact_phone="+61412345678", street_address="7 Ferguson Street", locality="Broadford",
                 administrative_area="VIC", postal_code="3658", document_key="k", document_filename="a.pdf")
    calls = {}
    monkeypatch.setattr(telnyx, "ordering_requirements", lambda c, t: [
        {"id": "r-addr", "field_type": "address"}, {"id": "r-text", "field_type": "textual"},
        {"id": "r-doc", "field_type": "document"}])
    monkeypatch.setattr(telnyx, "create_address", lambda **kw: calls.setdefault("address", kw) and "addr-1")
    monkeypatch.setattr(voice_lines.storage, "get_bytes", lambda d, key: b"%PDF")
    monkeypatch.setattr(telnyx, "upload_document", lambda data, name, ref: "doc-1")
    monkeypatch.setattr(telnyx, "create_requirement_group", lambda c, t, values, ref: calls.setdefault("values", values) and "grp-1")
    monkeypatch.setattr(telnyx, "create_number_order", lambda n, **kw: calls.setdefault("order", (n, kw)) and {
        "id": "ord-1", "status": "pending",
        "phone_numbers": [{"phone_number": n, "status": "pending", "requirements_status": "requirement-info-under-review"}]})

    voice_lines.order_number(db, line, "+61283902245")

    assert calls["address"]["first_name"] == "Jane" and calls["address"]["last_name"] == "Citizen"
    assert calls["values"] == [
        {"requirement_id": "r-addr", "field_value": "addr-1"},
        {"requirement_id": "r-text", "field_value": "Contact: Jane Citizen | Business: Acme Pty Ltd | Phone: +61412345678"},
        {"requirement_id": "r-doc", "field_value": "doc-1"},
    ]
    assert calls["order"] == ("+61283902245", {"connection_id": "conn-1", "requirement_group_id": "grp-1",
                                               "customer_reference": f"aichatsupport-line-{line.id}"})
    assert (line.status, line.telnyx_number_order_id, line.telnyx_status) == ("ordered", "ord-1", "requirement-info-under-review")


def test_unknown_requirement_type_is_refused_before_buying(db, settings_store, monkeypatch):
    line = _line(db, status="submitted", document_key="k")
    monkeypatch.setattr(telnyx, "ordering_requirements", lambda c, t: [{"id": "x", "name": "Selfie", "field_type": "action"}])
    monkeypatch.setattr(telnyx, "create_number_order", lambda *a, **k: pytest.fail("must not order"))
    with pytest.raises(voice_lines.VoiceLineError, match="Selfie"):
        voice_lines.order_number(db, line, "+61283902245")


@pytest.mark.parametrize("order,want", [
    ({"status": "success", "phone_numbers": [{"phone_number": "+61283902245", "status": "success"}]}, "number_ready"),
    ({"status": "failure", "phone_numbers": [{"phone_number": "+61283902245", "status": "failure"}]}, "failed"),
    ({"status": "pending", "phone_numbers": [{"phone_number": "+61283902245", "requirements_status": "requirement-info-exception"}]}, "info_needed"),
    ({"status": "pending", "phone_numbers": [{"phone_number": "+61283902245", "requirements_status": "requirement-info-under-review"}]}, "ordered"),
])
def test_order_status_mapping(order, want):
    line = SimpleNamespace(phone_number="+61283902245", status="ordered", status_message=None, telnyx_status=None)
    voice_lines._apply_order(line, order)
    assert line.status == want


def test_sync_attaches_the_approved_number_to_the_sip_connection(db, settings_store, monkeypatch):
    line = _line(db, status="ordered", phone_number="+61283902245", telnyx_number_order_id="ord-1")
    monkeypatch.setattr(telnyx, "get_number_order", lambda i: {"status": "success", "phone_numbers": [{"phone_number": "+61283902245", "status": "success"}]})
    monkeypatch.setattr(telnyx, "find_phone_number", lambda n: {"id": 555, "connection_id": None})
    assigned = []
    monkeypatch.setattr(telnyx, "assign_connection", lambda pid, conn: assigned.append((pid, conn)))
    voice_lines.sync(db, line)
    assert line.status == "number_ready" and line.telnyx_phone_number_id == "555" and assigned == [("555", "conn-1")]


# ---- activation ---------------------------------------------------------------------

def test_activate_creates_secret_agent_and_number_then_is_idempotent(db, settings_store, monkeypatch):
    settings_store["ELEVENLABS_WEBHOOK_ID"] = "wh-1"
    line = _line(db, status="number_ready", phone_number="+61283902245", business_name="Acme Pty Ltd",
                 voice_id="claire", transfer_number="+61290001234")
    created = {}
    monkeypatch.setattr(elevenlabs, "create_secret", lambda name, value: created.setdefault("secret", value) and "sec-1")
    monkeypatch.setattr(elevenlabs, "create_agent", lambda body: created.setdefault("agent", body) and "agent-1")
    monkeypatch.setattr(elevenlabs, "import_sip_number", lambda label, number, agent: created.setdefault("number", (number, agent)) and "pn-1")
    updates = []
    monkeypatch.setattr(elevenlabs, "update_agent", lambda aid, body: updates.append(aid))
    monkeypatch.setattr(elevenlabs, "assign_agent", lambda pid, aid: updates.append(pid))

    voice_lines.activate(db, line)

    assert (line.status, line.el_secret_id, line.el_agent_id, line.el_phone_number_id) == ("active", "sec-1", "agent-1", "pn-1")
    assert line.llm_secret_hash == hashlib.sha256(created["secret"].encode()).hexdigest()
    prompt = created["agent"]["conversation_config"]["agent"]["prompt"]
    assert prompt["llm"] == "custom-llm"
    assert prompt["custom_llm"]["url"].endswith(f"/api/v1/voice/lines/{line.id}")
    assert prompt["custom_llm"]["api_key"] == {"secret_id": "sec-1"}
    rule = prompt["built_in_tools"]["transfer_to_number"]["params"]["transfer_rules"][0]
    assert rule["provider_configs"][0]["config"]["transfer_destination"]["destination"] == "+61290001234"
    assert created["agent"]["conversation_config"]["tts"]["voice_id"] == "claire"
    assert created["agent"]["platform_settings"]["workspace_overrides"]["webhooks"]["post_call_webhook_id"] == "wh-1"
    assert created["number"] == ("+61283902245", "agent-1")
    assert db.query(Channel).filter(Channel.kind == "voice", Channel.bot_id == 7).count() == 1

    voice_lines.activate(db, line)  # retry: updates, never creates twice
    assert updates == ["agent-1", "pn-1"]


def test_activate_requires_an_approved_number(db, settings_store):
    with pytest.raises(voice_lines.VoiceLineError):
        voice_lines.activate(db, _line(db, status="ordered"))


# ---- calls & minutes ----------------------------------------------------------------

def _call_payload(cid="conv-1", secs=125):
    return {"conversation_id": cid, "agent_id": "agent-1",
            "metadata": {"call_duration_secs": secs, "start_time_unix_secs": 1763600000,
                         "phone_call": {"type": "sip_trunking", "external_number": "+61400111222"}},
            "transcript": [{"role": "agent", "message": "Hi, thanks for calling."},
                           {"role": "user", "message": "What are your hours?"}, {"role": "user", "message": ""}]}


def test_record_call_logs_conversation_and_counts_seconds_once(db, settings_store):
    _line(db, status="active", el_agent_id="agent-1")
    call = voice_lines.record_call(db, _call_payload())
    assert call.duration_secs == 125 and call.caller_number == "+61400111222"
    convo = db.get(Conversation, call.conversation_id)
    assert convo.status == "resolved" and convo.contact_phone == "+61400111222"
    assert [m.role for m in db.query(Message).filter(Message.conversation_id == convo.id)] == ["assistant", "user"]
    voice_lines.record_call(db, _call_payload())  # redelivery
    assert db.query(Subscription).first().voice_seconds_used == 125
    assert voice_lines.record_call(db, {**_call_payload("other"), "agent_id": "unknown"}) is None


def test_lines_stop_answering_when_minutes_run_out(db, settings_store):
    line = _line(db, status="active")
    assert voice_lines.can_take_calls(db, line)
    db.query(Subscription).update({Subscription.voice_seconds_used: 600})
    db.commit()
    assert not voice_lines.can_take_calls(db, line)


# ---- webhook signature --------------------------------------------------------------

def _sign(body: bytes, secret: str, ts: int) -> str:
    return f"t={ts},v0=" + hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()


def test_verify_webhook():
    body = b'{"type":"post_call_transcription"}'
    assert elevenlabs.verify_webhook(body, _sign(body, "s3cret", 1000), "s3cret", now=1100)
    assert not elevenlabs.verify_webhook(body + b" ", _sign(body, "s3cret", 1000), "s3cret", now=1100)
    assert not elevenlabs.verify_webhook(body, _sign(body, "wrong", 1000), "s3cret", now=1100)
    assert not elevenlabs.verify_webhook(body, _sign(body, "s3cret", 1000), "s3cret", now=1000 + 3600)
    assert not elevenlabs.verify_webhook(body, "", "s3cret")


# ---- endpoints ----------------------------------------------------------------------

@pytest.fixture
def client(db, monkeypatch):
    app = FastAPI()
    app.include_router(voice.router, prefix="/api/v1/voice")
    def override():
        yield db

    app.dependency_overrides[get_db] = override

    def fake_reply(bot_id, question, history=None):
        yield {"type": "delta", "text": "We open at nine."}
        yield {"type": "final", "text": "We open at nine.", "tokens": 5}

    monkeypatch.setattr(voice, "_reply_events", fake_reply)
    return TestClient(app)


def _active_line(db, **kw):
    plain, digest = voice_lines.new_llm_secret()
    return _line(db, status="active", llm_secret_hash=digest, el_agent_id="agent-1", **kw), plain


def _ask(client, line, secret, text="What are your hours?", **extra):
    return client.post(f"/api/v1/voice/lines/{line.id}/chat/completions", headers={"Authorization": f"Bearer {secret}"},
                       json={"stream": False, "messages": [{"role": "user", "content": text}], **extra})


def test_line_endpoint_needs_its_own_secret(db, client, settings_store):
    line, secret = _active_line(db)
    assert _ask(client, line, "wrong").status_code == 401
    r = _ask(client, line, secret)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "We open at nine."


def test_line_endpoint_speaks_unavailable_when_out_of_minutes(db, client, settings_store):
    line, secret = _active_line(db)
    db.query(Subscription).update({Subscription.voice_seconds_used: 10_000})
    db.commit()
    assert _ask(client, line, secret).json()["choices"][0]["message"]["content"] == voice.UNAVAILABLE_SPEECH


TRANSFER_TOOLS = [{"type": "function", "function": {"name": "transfer_to_number", "parameters": {
    "type": "object", "properties": {"transfer_number": {"type": "string"}, "reason": {"type": "string"}}}}}]


def test_asking_for_a_person_transfers_when_the_tool_is_offered(db, client, settings_store):
    line, secret = _active_line(db, transfer_number="+61290001234")
    r = _ask(client, line, secret, "Can I speak to a real person please?", tools=TRANSFER_TOOLS).json()
    choice = r["choices"][0]
    assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] == voice.TRANSFER_SPEECH
    call = choice["message"]["tool_calls"][0]["function"]
    assert call["name"] == "transfer_to_number"
    assert json.loads(call["arguments"])["transfer_number"] == "+61290001234"


def test_transfer_streams_as_a_tool_call(db, client, settings_store):
    line, secret = _active_line(db, transfer_number="+61290001234")
    r = client.post(f"/api/v1/voice/lines/{line.id}/chat/completions", headers={"Authorization": f"Bearer {secret}"},
                    json={"stream": True, "messages": [{"role": "user", "content": "put me through to the manager"}],
                          "tools": TRANSFER_TOOLS})
    chunks = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: {")]
    assert any("tool_calls" in c["choices"][0]["delta"] for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_no_transfer_without_the_tool(db, client, settings_store):
    line, secret = _active_line(db)
    r = _ask(client, line, secret, "Can I talk to a human?").json()
    assert r["choices"][0]["finish_reason"] == "stop" and "tool_calls" not in r["choices"][0]["message"]


@pytest.mark.parametrize("text,want", [
    ("Can I speak to a real person?", True), ("I want to talk with someone", True), ("transfer me please", True),
    ("Put me through to the manager", True), ("What time do you open?", False), ("Who is the person in charge of returns?", False),
])
def test_handoff_detection(text, want):
    assert voice.is_handoff_request(text) is want


def test_webhook_rejects_bad_signatures_and_logs_good_ones(db, client, settings_store):
    settings_store["ELEVENLABS_WEBHOOK_SECRET"] = "whsec"
    _line(db, status="active", el_agent_id="agent-1")
    body = json.dumps({"type": "post_call_transcription", "data": _call_payload()}).encode()
    url = "/api/v1/voice/webhooks/elevenlabs"
    assert client.post(url, content=body, headers={"ElevenLabs-Signature": "t=1,v0=bad"}).status_code == 401
    import time
    r = client.post(url, content=body, headers={"ElevenLabs-Signature": _sign(body, "whsec", int(time.time()))})
    assert r.status_code == 200
    assert db.query(VoiceCall).count() == 1


def test_billing_period_roll_resets_voice_minutes(db, monkeypatch):
    from datetime import datetime, timedelta
    db.query(Subscription).update({Subscription.voice_seconds_used: 300, Subscription.period_start: datetime.utcnow() - timedelta(days=40)})
    db.commit()
    assert billing.get_or_create_subscription(db, 1).voice_seconds_used == 0


def test_available_flag_follows_the_site_setting(client, settings_store):
    assert client.get("/api/v1/voice/available").json() == {"enabled": True}
    settings_store["VOICE_ENABLED"] = "false"
    assert client.get("/api/v1/voice/available").json() == {"enabled": False}

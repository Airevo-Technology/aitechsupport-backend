"""Telnyx REST client (API v2): number search, regulatory documents, number orders.

Only what AI phone lines need (app/core/voice_lines.py). Numbers are bought on the
platform's own Telnyx account and routed to ElevenLabs by one FQDN SIP connection
(TELNYX_SIP_CONNECTION_ID). Australian local numbers need regulatory details:
Telnyx tells us which at runtime (GET /regulatory_requirements), and we answer each
by its field type: an address id, a document id, or text.
"""
from __future__ import annotations

import base64
import logging

import httpx

from app.core import config_store

logger = logging.getLogger(__name__)

API_BASE = "https://api.telnyx.com/v2"
TIMEOUT = 20


class TelnyxError(Exception):
    """Telnyx refused or could not be reached. `message` is safe to show an admin."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def api_key() -> str:
    return config_store.get("TELNYX_API_KEY").strip()


def sip_connection_id() -> str:
    return config_store.get("TELNYX_SIP_CONNECTION_ID").strip()


def _request(method: str, path: str, *, params: dict | None = None, json: dict | None = None) -> dict:
    key = api_key()
    if not key:
        raise TelnyxError("Telnyx API key is not set.")
    try:
        resp = httpx.request(
            method, API_BASE + path, params=params, json=json, timeout=TIMEOUT,
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        )
    except httpx.HTTPError as exc:
        raise TelnyxError(f"Could not reach Telnyx ({exc.__class__.__name__}).") from exc
    if resp.status_code >= 400:
        raise TelnyxError(_error_text(resp), resp.status_code)
    if not resp.content:
        return {}
    return resp.json()


def _error_text(resp: httpx.Response) -> str:
    try:
        errors = resp.json().get("errors") or []
    except ValueError:
        errors = []
    detail = "; ".join(str(e.get("detail") or e.get("title") or "") for e in errors if isinstance(e, dict)).strip("; ")
    return f"Telnyx answered HTTP {resp.status_code}" + (f": {detail}" if detail else ".")


def _data(body: dict) -> dict:
    """Most responses wrap the record in `data`; a few (requirement groups) don't."""
    data = body.get("data", body)
    return data if isinstance(data, dict) else {}


# ===== Numbers =====


def search_local_numbers(country_code: str, locality: str | None = None, limit: int = 10) -> list[dict]:
    """Voice-capable local numbers for sale. Only numbers returned by a search can be ordered."""
    params = {
        "filter[country_code]": country_code,
        "filter[phone_number_type]": "local",
        "filter[features][]": "voice",
        "filter[limit]": limit,
    }
    if locality:
        params["filter[locality]"] = locality
    out = []
    for item in _request("GET", "/available_phone_numbers", params=params).get("data") or []:
        regions = {r.get("region_type"): r.get("region_name") for r in item.get("region_information") or []}
        cost = item.get("cost_information") or {}
        out.append({
            "phone_number": item.get("phone_number"),
            "locality": regions.get("location") or regions.get("rate_center") or "",
            "upfront_cost": cost.get("upfront_cost"),
            "monthly_cost": cost.get("monthly_cost"),
            "currency": cost.get("currency"),
        })
    return [n for n in out if n["phone_number"]]


def find_phone_number(e164: str) -> dict | None:
    rows = _request("GET", "/phone_numbers", params={"filter[phone_number]": e164}).get("data") or []
    return rows[0] if rows else None


def assign_connection(phone_number_id: str, connection_id: str) -> dict:
    return _data(_request("PATCH", f"/phone_numbers/{phone_number_id}", json={"connection_id": connection_id}))


def release_number(phone_number_id: str) -> None:
    _request("DELETE", f"/phone_numbers/{phone_number_id}")


# ===== Regulatory requirements =====


def ordering_requirements(country_code: str, phone_number_type: str = "local") -> list[dict]:
    """[{id, name, field_type, description, example}] Telnyx needs to order this kind of number."""
    params = {
        "filter[country_code]": country_code,
        "filter[phone_number_type]": phone_number_type,
        "filter[action]": "ordering",
    }
    reqs: list[dict] = []
    for group in _request("GET", "/regulatory_requirements", params=params).get("data") or []:
        for r in group.get("regulatory_requirements") or []:
            if r.get("id"):
                reqs.append({k: r.get(k) for k in ("id", "name", "field_type", "description", "example")})
    return reqs


def upload_document(data: bytes, filename: str, customer_reference: str) -> str:
    """Telnyx deletes an upload not linked to anything within 30 minutes, so link it right away."""
    body = {"file": base64.b64encode(data).decode(), "filename": filename, "customer_reference": customer_reference}
    doc_id = _data(_request("POST", "/documents", json=body)).get("id")
    if not doc_id:
        raise TelnyxError("Telnyx did not return a document id.")
    return str(doc_id)


def create_address(
    *, first_name: str, last_name: str, business_name: str, street_address: str, locality: str,
    administrative_area: str, postal_code: str, country_code: str, phone_number: str, customer_reference: str,
) -> str:
    """Addresses are immutable: a correction is a new address."""
    body = {
        "first_name": first_name, "last_name": last_name, "business_name": business_name,
        "street_address": street_address, "locality": locality, "administrative_area": administrative_area,
        "postal_code": postal_code, "country_code": country_code, "phone_number": phone_number,
        "customer_reference": customer_reference,
        # Telnyx's validation is the US emergency-address check; Telnyx's reviewers vet AU addresses by hand.
        "validate_address": False,
    }
    address_id = _data(_request("POST", "/addresses", json=body)).get("id")
    if not address_id:
        raise TelnyxError("Telnyx did not return an address id.")
    return str(address_id)


def create_requirement_group(country_code: str, phone_number_type: str, values: list[dict], customer_reference: str) -> str:
    body = {
        "country_code": country_code,
        "phone_number_type": phone_number_type,
        "action": "ordering",
        "customer_reference": customer_reference,
        "regulatory_requirements": values,
    }
    group_id = _data(_request("POST", "/requirement_groups", json=body)).get("id")
    if not group_id:
        raise TelnyxError("Telnyx did not return a requirement group id.")
    return str(group_id)


# ===== Number orders =====


def create_number_order(phone_number: str, *, connection_id: str, requirement_group_id: str | None, customer_reference: str) -> dict:
    entry: dict = {"phone_number": phone_number}
    if requirement_group_id:
        entry["requirement_group_id"] = requirement_group_id
    body = {"phone_numbers": [entry], "customer_reference": customer_reference}
    if connection_id:
        body["connection_id"] = connection_id
    return _data(_request("POST", "/number_orders", json=body))


def get_number_order(order_id: str) -> dict:
    return _data(_request("GET", f"/number_orders/{order_id}"))


def check_connection() -> None:
    """Prove the key works (cheapest authenticated read)."""
    _request("GET", "/phone_numbers", params={"page[size]": 1})

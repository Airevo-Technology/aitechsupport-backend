"""Client for PayPal Orders v2 (developer.paypal.com/docs/api/orders/v2).

Flow: create an order -> redirect the payer to its `payer-action` link -> PayPal
returns them to our billing page with `?token=<order id>` -> we capture. The
capture response, not the browser return, is the proof of payment.
"""
from __future__ import annotations

import httpx

from app.core import config_store

LIVE_BASE = "https://api-m.paypal.com"
SANDBOX_BASE = "https://api-m.sandbox.paypal.com"
_TIMEOUT = 30.0


class PayPalError(Exception):
    """A PayPal API call failed."""


def _as_bool(raw: str, default: bool) -> bool:
    if not raw:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


def config() -> dict:
    g = config_store.get
    return {
        "enabled": _as_bool(g("PAYPAL_ENABLED"), False),
        "sandbox": _as_bool(g("PAYPAL_SANDBOX"), True),
        "client_id": g("PAYPAL_CLIENT_ID"),
        "client_secret": g("PAYPAL_CLIENT_SECRET"),
    }


def is_configured() -> bool:
    c = config()
    return bool(c["enabled"] and c["client_id"] and c["client_secret"])


def _base(cfg: dict) -> str:
    return SANDBOX_BASE if cfg["sandbox"] else LIVE_BASE


def _access_token(cfg: dict) -> str:
    try:
        resp = httpx.post(f"{_base(cfg)}/v1/oauth2/token", auth=(cfg["client_id"], cfg["client_secret"]),
                          data={"grant_type": "client_credentials"}, timeout=_TIMEOUT)
    except httpx.HTTPError as e:
        raise PayPalError(f"Could not reach PayPal: {e}") from e
    if resp.status_code >= 400:
        raise PayPalError(f"PayPal auth failed ({resp.status_code}) — check the client id/secret and sandbox mode.")
    return resp.json()["access_token"]


def _request(method: str, path: str, json: dict | None = None) -> dict:
    cfg = config()
    if not (cfg["client_id"] and cfg["client_secret"]):
        raise PayPalError("PayPal is not configured.")
    try:
        resp = httpx.request(method, f"{_base(cfg)}{path}", json=json, timeout=_TIMEOUT,
                             headers={"Authorization": f"Bearer {_access_token(cfg)}",
                                      "Content-Type": "application/json"})
    except httpx.HTTPError as e:
        raise PayPalError(f"Could not reach PayPal: {e}") from e
    if resp.status_code >= 400:
        raise PayPalError(f"PayPal {method} {path} failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json() if resp.content else {}


def check_credentials() -> None:
    """Raises PayPalError if the stored credentials can't obtain a token."""
    _access_token(config())


def create_order(*, amount_cents: int, currency: str, description: str, custom_id: str,
                 return_url: str, cancel_url: str) -> tuple[str, str]:
    """Create a CAPTURE order. Returns (order_id, payer_approval_url)."""
    body = {
        "intent": "CAPTURE",
        "purchase_units": [{
            "amount": {"currency_code": currency, "value": f"{amount_cents / 100:.2f}"},
            "description": description[:127],
            "custom_id": custom_id[:127],
        }],
        "payment_source": {"paypal": {"experience_context": {
            "brand_name": "AiChatSupport",
            "user_action": "PAY_NOW",
            "shipping_preference": "NO_SHIPPING",
            "return_url": return_url,
            "cancel_url": cancel_url,
        }}},
    }
    order = _request("POST", "/v2/checkout/orders", body)
    links = {l.get("rel"): l.get("href") for l in order.get("links", [])}
    url = links.get("payer-action") or links.get("approve")
    if not url:
        raise PayPalError("PayPal returned no approval link.")
    return order["id"], url


def get_order(order_id: str) -> dict:
    return _request("GET", f"/v2/checkout/orders/{order_id}")


def capture_order(order_id: str) -> dict:
    """Capture an approved order. Safe to repeat: an already-captured order is
    returned as-is instead of raising."""
    try:
        return _request("POST", f"/v2/checkout/orders/{order_id}/capture")
    except PayPalError as e:
        if "ORDER_ALREADY_CAPTURED" in str(e):
            return get_order(order_id)
        raise

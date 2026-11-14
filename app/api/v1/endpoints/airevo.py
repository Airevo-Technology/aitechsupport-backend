"""airevo.com.au marketplace endpoints: signed events in, single sign-on.

Both are public (no JWT): events are authenticated by Airevo's HMAC signature,
and SSO by a one-time token that only Airevo's API can confirm. All the rules
live in app/core/airevo.py.
"""
import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.core import airevo
from app.core.security import create_access_token
from app.core.settings import settings
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@router.post("/events")
async def events(request: Request, db: Session = Depends(get_db)) -> JSONResponse:
    if not airevo.enabled():
        return _error(404, "not_enabled", "The airevo.com.au marketplace is not enabled here.")

    raw = await request.body()
    path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    headers = {k.lower(): v for k, v in request.headers.items()}
    if not airevo.verify(headers, request.method, path, raw):
        logger.warning("airevo.bad_signature", extra={"path": path})
        return _error(401, "bad_signature", "Signature, key id or timestamp did not verify.")

    try:
        event = json.loads(raw)
    except ValueError:
        return _error(400, "bad_json", "The body is not JSON.")

    try:
        status, body = airevo.handle(db, event)
    except airevo.AirevoError as exc:
        db.rollback()
        logger.warning("airevo.event_refused", extra={"event": event.get("id"), "code": exc.code})
        return JSONResponse(status_code=exc.status, content=exc.body())
    return JSONResponse(status_code=status, content=body)


@router.get("/sso")
def sso(token: str = "", db: Session = Depends(get_db)) -> RedirectResponse:
    """Airevo's "Open app" lands here with a 60-second, single-use token."""
    frontend = settings.FRONTEND_URL.rstrip("/")
    if not airevo.is_configured() or not token:
        return RedirectResponse(f"{frontend}/login", status_code=303)
    try:
        user = airevo.sso_user(db, token)
    except Exception:  # noqa: BLE001 (network or Airevo error: fall back to normal sign-in)
        logger.exception("airevo.sso_verify_failed")
        user = None
    if user is None:
        return RedirectResponse(f"{frontend}/login?from=airevo", status_code=303)

    access = create_access_token({"sub": str(user.id), "org": user.organization_id})
    # In the fragment, so the token never reaches a server log or a Referer header.
    return RedirectResponse(f"{frontend}/sso#token={access}", status_code=303)

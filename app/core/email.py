"""Transactional email via admin-configured SMTP.

SMTP config lives in config_store (admin panel); templates live in the
email_templates table. Sends are best-effort in the background so a mail outage
never breaks the request that triggered them.
"""
from __future__ import annotations

import re
import ssl
import html
import smtplib
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.core import config_store
from app.core.settings import settings
from app.models.email_template import EmailTemplate

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")

# ===== Branded, email-client-safe HTML shell =====
# Transactional bodies hold just the message; the professional chrome (header,
# footer, container) is applied here in code so every email is consistent and
# admins only edit the content.
BRAND_NAME = "AiChatSupport"
from app.core.settings import site_url as _site_url

BRAND_URL = _site_url()
BRAND_HOST = BRAND_URL.split("://", 1)[-1]
BRAND_ACCENT = "#4f46e5"
_FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"

# Sample values for the admin template preview (mirrors the real send context).
SAMPLE_CONTEXT = {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "dashboard_url": f"{settings.FRONTEND_URL}/dashboard",
    "plan": "Pro",
    "used_pct": "85",
    "tokens_remaining": "15,000",
}


def button(label: str, url: str) -> str:
    """A table-based CTA button that renders across major email clients."""
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" style="margin:6px 0 2px;">'
        f'<tr><td align="center" bgcolor="{BRAND_ACCENT}" style="border-radius:8px;">'
        f'<a href="{url}" target="_blank" style="display:inline-block;padding:12px 26px;font-family:{_FONT};'
        'font-size:14px;font-weight:600;line-height:1;color:#ffffff;text-decoration:none;border-radius:8px;">'
        f'{label}</a></td></tr></table>'
    )


def wrap_email(inner_html: str, preheader: str = "") -> str:
    """Wrap message content in the branded, responsive email shell."""
    year = datetime.utcnow().year
    pre = (
        '<span style="display:none!important;visibility:hidden;opacity:0;color:transparent;'
        f'height:0;width:0;overflow:hidden;mso-hide:all;">{html.escape(preheader)}</span>'
        if preheader else ""
    )
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="x-ua-compatible" content="IE=edge">'
        '<meta name="x-apple-disable-message-reformatting">'
        f'<title>{BRAND_NAME}</title></head>'
        '<body style="margin:0;padding:0;background:#eef2f7;">'
        f'{pre}'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        'style="background:#eef2f7;"><tr><td align="center" style="padding:28px 12px;">'
        '<table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" '
        'style="width:600px;max-width:600px;background:#ffffff;border-radius:12px;overflow:hidden;'
        'border:1px solid #e2e8f0;">'
        f'<tr><td style="height:4px;line-height:4px;font-size:4px;background:{BRAND_ACCENT};">&nbsp;</td></tr>'
        '<tr><td style="padding:24px 32px 6px;">'
        f'<span style="font-family:{_FONT};font-size:20px;font-weight:700;letter-spacing:-.2px;color:#0f172a;">'
        f'AiChat<span style="color:{BRAND_ACCENT};">Support</span></span></td></tr>'
        f'<tr><td style="padding:10px 32px 28px;font-family:{_FONT};font-size:15px;line-height:1.65;color:#334155;">'
        f'{inner_html}</td></tr>'
        '<tr><td style="padding:20px 32px;background:#f8fafc;border-top:1px solid #e9eef5;">'
        f'<p style="margin:0 0 4px;font-family:{_FONT};font-size:12px;line-height:1.6;color:#94a3b8;">'
        f'{BRAND_NAME} — AI customer support for your website &amp; WhatsApp.</p>'
        f'<p style="margin:0;font-family:{_FONT};font-size:12px;line-height:1.6;color:#94a3b8;">'
        f'<a href="{BRAND_URL}" style="color:{BRAND_ACCENT};text-decoration:none;">{BRAND_HOST}</a>'
        f'&nbsp;·&nbsp;© {year} {BRAND_NAME}, built by Airevo.</p>'
        '</td></tr></table></td></tr></table></body></html>'
    )


def _port(raw: str) -> int:
    try:
        return int(str(raw).strip())
    except (ValueError, TypeError):
        logger.warning("invalid SMTP_PORT %r; using 587", raw)
        return 587


def smtp_config(db: Session) -> dict:
    g = config_store.get
    return {
        "host": g("SMTP_HOST"),
        "port": _port(g("SMTP_PORT", "587") or "587"),
        "username": g("SMTP_USERNAME"),
        "password": g("SMTP_PASSWORD"),
        "from_email": g("SMTP_FROM_EMAIL"),
        "from_name": g("SMTP_FROM_NAME") or "AiChatSupport",
        "security": (g("SMTP_SECURITY") or "tls").strip().lower(),  # tls | ssl | none
        "enabled": (g("SMTP_ENABLED") or "false").lower() == "true",
    }


def is_configured(db: Session) -> bool:
    cfg = smtp_config(db)
    return cfg["enabled"] and bool(cfg["host"]) and bool(cfg["from_email"])


def _sanitize_header(value: str) -> str:
    """Strip CR/LF so a value can't inject extra email headers."""
    return "".join(c for c in (value or "") if c not in "\r\n").strip()


def _html_to_text(body_html: str) -> str:
    """Readable plain-text alternative from an HTML body (for multipart emails)."""
    text = re.sub(r"(?is)<(style|script|head|title)[^>]*>.*?</\1>", " ", body_html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|tr|h[1-6]|li)>", "\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _render(template: str, context: dict, escape: bool = False) -> str:
    """Substitute {key} placeholders literally (no str.format attribute traversal).

    `escape=True` HTML-escapes the substituted VALUES (not the admin-authored
    template) so user-supplied fields can't inject markup into an HTML body.
    """
    out = template
    for key, val in context.items():
        text = html.escape(str(val)) if escape else str(val)
        out = out.replace("{" + key + "}", text)
    return out


def render(db: Session, key: str, context: dict) -> tuple[str, str]:
    """Return (subject, html) for a template key, with variables injected and the
    message content wrapped in the branded shell."""
    tmpl = db.query(EmailTemplate).filter(EmailTemplate.key == key, EmailTemplate.is_active.is_(True)).first()
    if not tmpl:
        raise ValueError(f"email template '{key}' not found")
    # subject is a plain-text header (CRLF-sanitized in send); body is HTML (escape values).
    inner = _render(tmpl.body_html, context, escape=True)
    return _render(tmpl.subject, context), wrap_email(inner, _html_to_text(inner)[:140])


def preview(subject: str, body_html: str) -> tuple[str, str]:
    """Render (subject, wrapped-html) with sample values, for the admin preview."""
    inner = _render(body_html, SAMPLE_CONTEXT, escape=True)
    return _render(subject, SAMPLE_CONTEXT), wrap_email(inner, _html_to_text(inner)[:140])


def send(db: Session, to_email: str, subject: str, html: str) -> None:
    """Send one HTML email via the configured SMTP server. Raises on failure."""
    cfg = smtp_config(db)
    if not cfg["host"] or not cfg["from_email"]:
        raise RuntimeError("SMTP is not configured")

    to_email = _sanitize_header(to_email)
    subject = _sanitize_header(subject)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = formataddr((cfg["from_name"], cfg["from_email"]))
    msg["To"] = to_email
    msg.attach(MIMEText(_html_to_text(html), "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    ctx = ssl.create_default_context()
    if cfg["security"] == "ssl":
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ctx, timeout=20) as s:
            if cfg["username"]:
                s.login(cfg["username"], cfg["password"])
            s.sendmail(cfg["from_email"], [to_email], msg.as_string())
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as s:
            # Fail closed to encryption: STARTTLS for anything except an explicit "none",
            # so a typo'd/unknown security value never sends credentials in cleartext.
            if cfg["security"] != "none":
                s.starttls(context=ctx)
            if cfg["username"]:
                s.login(cfg["username"], cfg["password"])
            s.sendmail(cfg["from_email"], [to_email], msg.as_string())


def send_template_bg(to_email: str, key: str, context: dict) -> None:
    """Best-effort background send (own session; never raises to the caller)."""
    db = SessionLocal()
    try:
        if not is_configured(db):
            logger.info("email skipped (SMTP disabled): %s -> %s", key, to_email)
            return
        subject, html = render(db, key, context)
        send(db, to_email, subject, html)
        logger.info("email sent: %s -> %s", key, to_email)
    except Exception:  # noqa: BLE001
        logger.exception("email failed: %s -> %s", key, to_email)
    finally:
        db.close()

"""Turn a knowledge source into clean plain text.

- load_url:   fetch a page (SSRF-guarded, size-capped) and extract readable text.
- extract_html / load_bytes: for uploaded files and raw HTML.

PDF/office formats are a later addition (needs extra deps) — see PROJECT_STATE.md.
"""
from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.core.settings import settings

_UA = f"AiChatSupportBot/0.1 (+{os.getenv('SITE_URL', 'https://aichatsupport.my').rstrip('/')})"
_STRIP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "svg", "form", "iframe"]


class UrlNotAllowed(ValueError):
    """Raised when a knowledge URL is rejected before/while fetching (SSRF guard)."""


def _collapse(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def extract_html(html: str) -> tuple[str | None, str]:
    """Return (title, cleaned_text) from an HTML string."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    return title, _collapse(soup.get_text("\n"))


def _assert_public_host(host: str) -> None:
    """Reject hosts that resolve to loopback/private/link-local/reserved IPs.

    Blocks SSRF to internal services and cloud metadata (169.254.169.254 etc.).
    Note: a determined attacker could still DNS-rebind between this check and the
    connection; pinning the resolved IP into the transport is a hardening TODO.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UrlNotAllowed(f"cannot resolve host: {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise UrlNotAllowed(f"URL host resolves to a non-public address ({ip})")


def validate_url(url: str) -> None:
    """Cheap pre-flight validation (scheme + host) usable in the request path."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UrlNotAllowed("only http/https URLs are allowed")
    if not parsed.hostname:
        raise UrlNotAllowed("URL has no host")
    _assert_public_host(parsed.hostname)


def load_url(url: str) -> tuple[str | None, str]:
    """Fetch a URL and return (title, text), SSRF-guarded and size-capped.

    Redirects are followed manually so every hop is re-validated (a public URL
    can otherwise 302 into an internal host). Raises UrlNotAllowed / httpx.HTTPError.
    """
    max_bytes = settings.MAX_URL_BYTES
    timeout = httpx.Timeout(settings.URL_FETCH_TIMEOUT)
    current = url
    with httpx.Client(follow_redirects=False, timeout=timeout, headers={"User-Agent": _UA}) as client:
        for _ in range(settings.URL_FETCH_MAX_REDIRECTS + 1):
            validate_url(current)
            with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        resp.raise_for_status()
                        break
                    current = str(resp.url.join(location))
                    continue
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "").lower()
                total = 0
                parts: list[bytes] = []
                for block in resp.iter_bytes():
                    total += len(block)
                    if total > max_bytes:
                        raise UrlNotAllowed(f"response exceeds {max_bytes} bytes")
                    parts.append(block)
                body = b"".join(parts).decode(resp.encoding or "utf-8", errors="ignore")
                if "html" in ctype or body.lstrip()[:1] == "<":
                    return extract_html(body)
                return None, _collapse(body)
        raise UrlNotAllowed("too many redirects")


def load_bytes(filename: str, data: bytes) -> tuple[str | None, str]:
    """Decode an uploaded file to (title, text). HTML is stripped; else treated as text."""
    text = data.decode("utf-8", errors="ignore")
    if filename.lower().endswith((".html", ".htm")):
        return extract_html(text)
    return filename, _collapse(text)

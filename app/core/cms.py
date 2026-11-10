"""CMS helpers: structured homepage content (stored as a JSON blob in the key-value
`settings` table) and default seed content for pages."""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.models.setting import Setting
from app.models.page import Page
from app.core.settings import market

HOMEPAGE_KEY = "homepage_content"

DEFAULT_HOMEPAGE: dict = {
    "meta_title": "AiChatSupport — AI Support Chatbot for your website & WhatsApp",
    "meta_description": "Connect your content and let an AI assistant answer your customers 24/7 on your website, with a real human handoff when it matters.",
    "hero": {
        "badge": "Now with live human handoff",
        "headline": "AI support that answers on your website & WhatsApp",
        "subhead": "Point the bot at your content, add it to your site in one line, and let it handle customer questions — with a real human taking over when it matters.",
        "cta_primary": "Get started free",
        "cta_secondary": "See pricing",
    },
    "features": [
        {"icon": "bolt", "title": "Answers from your content", "text": "Upload documents or paste a URL — the bot answers only from your knowledge base, so replies stay accurate and on-brand."},
        {"icon": "chat", "title": "Website widget in one line", "text": "Paste a single <script> tag. A polished chat bubble appears on your site instantly, styled to match your brand."},
        {"icon": "user", "title": "Live human handoff", "text": "When the bot can't help, a real person takes over the conversation live from the dashboard or the mobile app."},
        {"icon": "shield", "title": "Your data, self-hosted", "text": "Inference runs on our own servers — your customers' questions never leave our infrastructure."},
        {"icon": "globe", "title": "Multilingual", "text": "Understands and replies in English, Bahasa Melayu, Chinese and more."},
        {"icon": "chart", "title": "Analytics built in", "text": "See what customers ask, your deflection rate, and unanswered questions to feed back into the bot."},
    ],
    "pricing": {"heading": "Simple, transparent pricing", "subhead": "Start free. Upgrade when you grow."},
    "faq": [
        {"q": "How does the bot know the answers?", "a": "You add your content — upload documents, paste text, or point it at a URL. The bot answers only from that, and offers a human when it's unsure."},
        {"q": "How do I add it to my website?", "a": "Copy one <script> tag from your dashboard and paste it before the closing </body> tag. That's the whole install."},
        {"q": "Can a human take over?", "a": "Yes. When a visitor asks for a human (or the bot can't answer), your team is notified and can reply live from the dashboard or the agent app."},
        {"q": "Is my data private?", "a": "Yes — inference runs on our own servers, and every business's content is strictly isolated from every other."},
    ],
    "cta": {"headline": "Ready to let AI handle your support?", "text": "Set up your first bot in minutes — no credit card required.", "button": "Get started free"},
    "footer": {"tagline": "AI support chatbot for Malaysian businesses.", "social": []},
    "demo_public_key": "",  # optional widget public_key for the live homepage demo
}

# Per-market overrides applied on top of the shared defaults (MARKET=au).
_AU_OVERRIDES = {
    "footer": {"tagline": "AI support chatbot for Australian businesses.", "social": []},
}
if market() == "au":
    DEFAULT_HOMEPAGE.update(_AU_OVERRIDES)
    DEFAULT_HOMEPAGE["features"] = [
        {**f, "text": "Understands and replies in English and the languages your customers speak, including Mandarin, Cantonese, Vietnamese and more."}
        if f["icon"] == "globe" else f
        for f in DEFAULT_HOMEPAGE["features"]
    ]

DEFAULT_PAGES: list[dict] = [
    {
        "slug": "about",
        "title": "About us",
        "meta_description": "About AiChatSupport — AI-powered customer support for businesses.",
        "sort_order": 1,
        "body": "<p>AiChatSupport helps businesses answer their customers instantly with an AI assistant "
                "that learns from their own content — with a real human ready to step in when needed.</p>"
                "<p>Edit this page in the admin dashboard (Pages).</p>",
    },
    {
        "slug": "privacy",
        "title": "Privacy Policy",
        "meta_description": "How AiChatSupport collects, uses, and protects your data.",
        "sort_order": 2,
        "body": "<p>This Privacy Policy explains how AiChatSupport handles data. Replace this placeholder "
                "with your real policy in the admin dashboard (Pages).</p>",
    },
    {
        "slug": "terms",
        "title": "Terms of Service",
        "meta_description": "The terms governing your use of AiChatSupport.",
        "sort_order": 3,
        "body": "<p>These Terms of Service govern your use of AiChatSupport. Replace this placeholder with "
                "your real terms in the admin dashboard (Pages).</p>",
    },
]


def get_homepage(db: Session) -> dict:
    row = db.query(Setting).filter(Setting.key == HOMEPAGE_KEY).first()
    if row and row.value:
        try:
            return {**DEFAULT_HOMEPAGE, **json.loads(row.value)}
        except Exception:  # noqa: BLE001 — corrupt blob -> fall back to defaults
            pass
    return dict(DEFAULT_HOMEPAGE)


def set_homepage(db: Session, content: dict) -> dict:
    merged = {**DEFAULT_HOMEPAGE, **(content or {})}
    row = db.query(Setting).filter(Setting.key == HOMEPAGE_KEY).first()
    if row is None:
        row = Setting(key=HOMEPAGE_KEY, value=json.dumps(merged), is_secret=False)
        db.add(row)
    else:
        row.value = json.dumps(merged)
    db.commit()
    return merged


def seed_default_pages(db: Session) -> None:
    """Create the starter About/Privacy/Terms pages once, if none exist."""
    if db.query(Page).count() > 0:
        return
    for p in DEFAULT_PAGES:
        db.add(Page(slug=p["slug"], title=p["title"], body=p["body"],
                    meta_description=p["meta_description"], is_published=True, sort_order=p["sort_order"]))
    db.commit()

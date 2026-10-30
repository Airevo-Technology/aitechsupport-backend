"""Catalog of selectable chat models, and the chat fallback chain.

Ids are suggestions for the admin UI dropdowns; an admin can enter any id as free
text too. For the self-hosted path these are Ollama model tags served from a
tier's own base_url; for the external path they are OpenRouter ids (verify slugs
at https://openrouter.ai/models).

Which model answers a bot's question is not a customer choice — it's an ordered
fallback chain owned by the org's package (Package.fallback_chain), tried top to
bottom until one candidate answers successfully. A package with no chain of its
own inherits the hardcoded platform default (DEFAULT_FALLBACK_CHAIN), so the Free
plan can stay self-hosted while paid plans use OpenRouter. A bot's own chat_model
column is a rare admin-only override, tried before the chain.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.core import config_store

if TYPE_CHECKING:
    from app.models.bot import Bot
    from app.models.package import Package

CHAT_MODELS: list[dict] = [
    # Self-hosted (Ollama on the ai-server) — the default path.
    {"id": "qwen2:latest", "label": "Qwen2 7B (local)", "provider": "Local (Ollama)"},
    {"id": "llama3.1:8b", "label": "Llama 3.1 8B (local)", "provider": "Local (Ollama)"},
    {"id": "mistral:7b", "label": "Mistral 7B (local)", "provider": "Local (Ollama)"},
    {"id": "gemma4:e4b", "label": "Gemma 3n E4B (local)", "provider": "Local (Ollama)"},
    {"id": "ecomm-agent:latest", "label": "E-comm Agent (local)", "provider": "Local (Ollama)"},
    # External (OpenRouter) — available if an OpenRouter key is configured.
    {"id": "anthropic/claude-haiku-4.5", "label": "Claude Haiku 4.5", "provider": "Anthropic"},
    {"id": "anthropic/claude-sonnet-4.5", "label": "Claude Sonnet 4.5", "provider": "Anthropic"},
    {"id": "openai/gpt-5.4-mini", "label": "GPT-5.4 Mini", "provider": "OpenAI"},
    {"id": "openai/gpt-oss-120b", "label": "GPT-OSS 120B", "provider": "OpenAI (open)"},
    {"id": "openai/gpt-oss-20b", "label": "GPT-OSS 20B", "provider": "OpenAI (open)"},
    {"id": "mistralai/mistral-small", "label": "Mistral Small", "provider": "Mistral"},
]

_HARDCODED_DEFAULT = "anthropic/claude-haiku-4.5"

# The real OpenRouter API — independent of the ambient global OPENROUTER_BASE_URL
# setting, which an admin may have repointed at a self-hosted Ollama server as the
# platform-wide default (see settings.py). A package explicitly configured for
# "openrouter" must always reach actual OpenRouter, never wherever that ambient
# default happens to point.
_REAL_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def default_model() -> str:
    """Platform default model (settable via admin), else the hardcoded default."""
    return config_store.get("DEFAULT_CHAT_MODEL") or _HARDCODED_DEFAULT


# The platform's reliability chain when no admin override is configured yet:
# two self-hosted boxes, then cheap-then-reliable paid OpenRouter models as a last
# resort so a customer-facing chat never hard-fails just because one GPU box is
# down. (OpenRouter's :free slugs get rate-limited or withdrawn without notice.)
DEFAULT_FALLBACK_CHAIN: list[dict] = [
    {"label": "black", "provider": "self_hosted",
     "base_url": "http://100.102.172.23:11434", "model": "qwen3.5:9b-q4_K_M"},
    {"label": "ai-server", "provider": "self_hosted",
     "base_url": "http://100.101.148.46:11434", "model": "qwen3.5:4b-q4_K_M"},
    {"label": "GPT-OSS 120B", "provider": "openrouter",
     "base_url": None, "model": "openai/gpt-oss-120b"},
    {"label": "Claude Haiku", "provider": "openrouter",
     "base_url": None, "model": "anthropic/claude-haiku-4.5"},
]


def fallback_chain() -> list[dict]:
    """The hardcoded platform default a package inherits when it has no chain of
    its own (see chain_for_package). Not admin-editable — configure a chain per
    package instead, under Admin -> Packages."""
    return DEFAULT_FALLBACK_CHAIN


def _tier_base_url(tier: dict) -> str | None:
    if tier.get("provider") == "self_hosted":
        base = (tier.get("base_url") or "").rstrip("/")
        return f"{base}/v1" if base else None
    if tier.get("provider") == "openrouter":
        return _REAL_OPENROUTER_BASE_URL
    return None


def chain_for_package(package: "Package | None") -> list[dict]:
    """The package's own fallback chain, or the platform-wide one when it has
    none (NULL) — so a newly created package answers before it's configured."""
    if package is not None:
        tiers = package.fallback_chain
        if isinstance(tiers, list) and tiers:
            return tiers
    return fallback_chain()


def resolve_candidates(bot: "Bot", package: "Package | None" = None) -> list[tuple[str, str | None]]:
    """Ordered (model_id, base_url_override) candidates to try for a bot's next
    answer: bot.chat_model (a rare admin-only override, not customer-editable)
    first if set, then every tier of the package's fallback chain in order.
    """
    candidates: list[tuple[str, str | None]] = []
    if bot.chat_model:
        candidates.append((bot.chat_model, None))
    for tier in chain_for_package(package):
        model = (tier.get("model") or "").strip()
        if model:
            candidates.append((model, _tier_base_url(tier)))
    if not candidates:
        candidates.append((default_model(), None))
    return candidates


# Vision is OFF by default: the self-hosted box can't comfortably host a VLM, so
# an image is handed to a human instead of being guessed at. Set VISION_MODEL (a
# platform-admin setting, deliberately NOT per-bot) to a vision-capable model to
# turn AI image reading on — a turn carrying an image is then forced onto it,
# since the bot's text model would silently ignore the picture.
def vision_model() -> str:
    """Model used for image turns, or "" when AI vision is disabled."""
    return (config_store.get("VISION_MODEL") or "").strip()


def vision_enabled() -> bool:
    return bool(vision_model())

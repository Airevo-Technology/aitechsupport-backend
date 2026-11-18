"""Retrieval-Augmented answering: embed the question, pull the nearest KB chunks
for THIS bot via pgvector, then ask Claude to answer from that context.
"""
from __future__ import annotations

import base64
import logging

from sqlalchemy.orm import Session

from app.core import billing, embeddings, llm, models_catalog, storage
from app.core.settings import settings
from app.models.knowledge import Chunk, KnowledgeSource
from app.models.bot import Bot

logger = logging.getLogger(__name__)

# Used when a visitor sends a picture with no caption — retrieval has nothing to
# embed, so we ask the vision model to lead with what it can see.
DEFAULT_IMAGE_PROMPT = "The customer sent this image without a caption. What can you tell them about it?"


def _image_data_url(db: Session, image_key: str, image_mime: str | None) -> str | None:
    """Read the stored image and inline it as a base64 data URL.

    Inlined rather than handed to the model as a presigned link: the self-hosted
    model server has no guaranteed route to the object store, and a link could
    expire mid-generation. Returns None (never raises) if the object is gone, so
    a missing image degrades to a text-only answer.
    """
    try:
        raw = storage.get_bytes(db, image_key)
    except Exception:
        logger.warning("could not read chat image %s", image_key, exc_info=True)
        return None
    mime = image_mime or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def retrieve(db: Session, bot: Bot, question: str, k: int | None = None) -> list[Chunk]:
    """Return the k most similar chunks belonging to this bot (pgvector cosine).

    Scoped to the bot's own KnowledgeSources (and its org), so a multi-bot org
    never leaks one bot's knowledge into another's answers.
    """
    k = k or settings.RETRIEVAL_TOP_K
    query_vec = embeddings.embed_query(question)
    return (
        db.query(Chunk)
        .join(KnowledgeSource, KnowledgeSource.id == Chunk.knowledge_source_id)
        .filter(
            KnowledgeSource.bot_id == bot.id,
            Chunk.organization_id == bot.organization_id,
            Chunk.embedding.isnot(None),
        )
        .order_by(Chunk.embedding.cosine_distance(query_vec))
        .limit(k)
        .all()
    )


def _prepare(
    db: Session, bot: Bot, question: str, image_key: str | None, image_mime: str | None
):
    """Shared setup for both answer paths.

    Returns (context, question, candidates, image_data_url) or None when there is
    nothing to answer with — the caller then emits the fallback (0 tokens), which
    is the handoff signal. `candidates` is an ordered list of (model, base_url)
    to try in turn — the org's package fallback chain (see
    models_catalog.resolve_candidates).

    An image turn is answered even with NO retrieved context: the picture is the
    evidence, and handing off the moment the KB misses would defeat the feature.
    A turn carrying an image is also forced onto the vision model, since a text
    model would silently ignore the picture. Vision stays platform-wide (a single
    model, not a fallback chain) and always uses the ambient default endpoint.
    """
    # Vision may be disabled (no VISION_MODEL); then an image is ignored here and
    # the caller has already routed the turn to a human.
    use_vision = bool(image_key) and models_catalog.vision_enabled()
    image_data_url = _image_data_url(db, image_key, image_mime) if use_vision else None
    # Nothing to embed for a captionless image, and embedding "" is meaningless.
    chunks = retrieve(db, bot, question) if question.strip() else []
    if not chunks and image_data_url is None:
        return None
    context = "\n\n---\n\n".join(c.content for c in chunks)
    if image_data_url:
        candidates = [(models_catalog.vision_model(), None)]
        return context, (question.strip() or DEFAULT_IMAGE_PROMPT), candidates, image_data_url
    sub = billing.get_or_create_subscription(db, bot.organization_id)
    package = billing.package_for(db, sub.plan)
    return context, question, models_catalog.resolve_candidates(bot, package), None


def _handoff_system(business_note: str | None, agents_online: bool) -> str:
    """System prompt for the LLM-composed handoff message (see _compose_handoff).

    Deliberately separate from _build_system: there is no CONTEXT to ground
    against here, and the two asks (don't guess an answer / do route the
    visitor to a human) are opposite what a grounded-answer prompt says.
    """
    availability = (
        "A teammate is online right now — tell them to press the 'Talk to a "
        "human' button in the chat header to reach a person immediately."
        if agents_online else
        "Nobody is online right now — ask them, in one line, to leave their "
        "name, phone number, and email using the form so the team can follow "
        "up as soon as someone is back."
    )
    note = f"\n\nHouse style to match, if useful: {business_note.strip()}" if business_note else ""
    return (
        "You are the support assistant for this business. A knowledge-base search "
        "just found NOTHING relevant to the visitor's question, so you must not "
        "guess, invent, or attempt an answer.\n\n"
        f"{availability}\n\n"
        "Reply in ONE OR TWO short sentences, in the SAME LANGUAGE the visitor "
        "wrote in. Be warm but brief, and don't repeat their question back to them."
        f"{note}"
    )


def _text_candidates(db: Session, bot: Bot) -> list[tuple[str, str | None]]:
    sub = billing.get_or_create_subscription(db, bot.organization_id)
    package = billing.package_for(db, sub.plan)
    return models_catalog.resolve_candidates(bot, package)


def _compose_handoff(db: Session, bot: Bot, question: str, agents_online: bool) -> tuple[str, int]:
    """LLM-composed handoff reply: matches the visitor's language and tells them
    whether a human is reachable right now or how to leave contact details.
    Degrades to the raw bot.fallback_message on any failure. Always returns
    tokens=0 to the caller (still the handoff signal) — explaining a handoff
    isn't knowledge-based value, so it isn't billed, regardless of the small
    real generation cost incurred here.
    """
    try:
        candidates = _text_candidates(db, bot)
        system = _handoff_system(bot.fallback_message, agents_online)
        text, _tokens, _model = llm.handoff_message_with_fallback(system, question, candidates)
        if text.strip():
            return text.strip(), 0
    except Exception:  # noqa: BLE001 — degrade to the static message, never fail the turn
        logger.warning("handoff message composition failed for bot %s, using static fallback",
                       bot.id, exc_info=True)
    return (bot.fallback_message or "I don't have an answer for that yet."), 0


def _compose_handoff_stream(db: Session, bot: Bot, question: str, agents_online: bool):
    got_any = False
    try:
        candidates = _text_candidates(db, bot)
        system = _handoff_system(bot.fallback_message, agents_online)
        for ev in llm.handoff_message_stream_with_fallback(system, question, candidates):
            if ev.get("type") == "delta":
                got_any = True
            if ev.get("type") == "final":
                ev = {**ev, "tokens": 0}  # not billed — see _compose_handoff
            yield ev
        return
    except Exception:  # noqa: BLE001
        if got_any:
            raise  # visitor already saw partial composed text — don't double-send
        logger.warning("handoff stream composition failed for bot %s, using static fallback",
                       bot.id, exc_info=True)
    fb = bot.fallback_message or "I don't have an answer for that yet."
    yield {"type": "delta", "text": fb}
    yield {"type": "final", "text": fb, "tokens": 0}


def answer_question(
    db: Session,
    bot: Bot,
    question: str,
    agents_online: bool,
    image_key: str | None = None,
    image_mime: str | None = None,
) -> tuple[str, int]:
    """End-to-end: retrieve context, then generate a grounded answer.

    Returns (answer, tokens_used). Nothing to answer with -> a composed handoff
    message (see _compose_handoff), 0 tokens.
    """
    prepared = _prepare(db, bot, question, image_key, image_mime)
    if prepared is None:
        return _compose_handoff(db, bot, question, agents_online)
    context, q, candidates, image_data_url = prepared
    text, tokens, _model_used = llm.answer_with_fallback(
        bot.system_prompt or "", context, q, candidates, image_data_url=image_data_url
    )
    return text, tokens


def answer_question_stream(
    db: Session,
    bot: Bot,
    question: str,
    agents_online: bool,
    image_key: str | None = None,
    image_mime: str | None = None,
    system_suffix: str = "",
    first_token_s: float | None = None,
    hedge_s: float | None = None,
):
    """Streaming twin of answer_question. Yields {'type':'delta'|'final', ...} events;
    the terminal 'final' carries the full text + tokens for persistence/metering.
    Nothing to answer with -> the composed handoff message, streamed, 0 tokens.
    `system_suffix` is appended to the bot's system prompt (channel style, e.g. spoken replies).
    `first_token_s` overrides the platform first-token deadline for this call (0 = off);
    `hedge_s` overrides the hedged-request delay (0 = off; supersedes first_token_s when on).
    """
    prepared = _prepare(db, bot, question, image_key, image_mime)
    if prepared is None:
        yield from _compose_handoff_stream(db, bot, question, agents_online)
        return
    context, q, candidates, image_data_url = prepared
    yield from llm.answer_stream_with_fallback(
        (bot.system_prompt or "") + system_suffix, context, q, candidates,
        image_data_url=image_data_url, first_token_s=first_token_s, hedge_s=hedge_s,
    )

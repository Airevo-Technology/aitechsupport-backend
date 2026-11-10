import os
import secrets
import logging
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env early so every os.getenv() default below resolves regardless of
# module import order.
load_dotenv()

logger = logging.getLogger(__name__)

# Known-insecure placeholder that must never sign real tokens.
_INSECURE_SECRET = "supersecretkey123"


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AiTechSupport API"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # ===== Logging =====
    # LOG_LEVEL covers application logs (e.g. which fallback model answered and
    # how long it took). HTTP client libraries are capped separately at
    # LOG_LEVEL_HTTP, since they emit a line per request. See core/logging_config.
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_LEVEL_HTTP: str = os.getenv("LOG_LEVEL_HTTP", "WARNING")

    # ===== Security =====
    # No hardcoded fallback: a missing/placeholder key is rejected at startup below.
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 8))

    # ===== Database =====
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL",
        "postgresql://ats_admin:changeme@localhost/aitechsupport_db",
    )

    # ===== Chat models (via OpenRouter — one key, many providers) =====
    # These are env fallbacks; a platform admin can override them in the DB
    # (see config_store / admin settings) without a redeploy.
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    DEFAULT_CHAT_MODEL: str = os.getenv("DEFAULT_CHAT_MODEL", "anthropic/claude-haiku-4.5")

    # ===== Self-hosted inference (Ollama on the ai-server, reached over Tailscale) =====
    # When set, the chat path points OPENROUTER_BASE_URL at "{OLLAMA_BASE_URL}/v1"
    # (Ollama's OpenAI-compatible API) and embeddings use EMBEDDINGS_PROVIDER=ollama.
    # No API key is required for a local endpoint.
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://100.101.148.46:11434")

    # ===== Embeddings =====
    # Provider: "ollama" (self-hosted, e.g. bge-m3) | "voyage" (external) |
    # "fake" (deterministic, offline dev/CI — NO real semantics, never use in production).
    EMBEDDINGS_PROVIDER: str = os.getenv("EMBEDDINGS_PROVIDER", "voyage")
    VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "voyage-3")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 1024))
    # Per-request cap for the embeddings provider (Voyage limits texts/tokens per call).
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", 128))

    # ===== Chunking =====
    CHUNK_MAX_CHARS: int = int(os.getenv("CHUNK_MAX_CHARS", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", 4))
    # Hard ceiling on a single question sent to retrieval/answer.
    MAX_QUESTION_CHARS: int = int(os.getenv("MAX_QUESTION_CHARS", 4000))

    # Max bytes accepted for an uploaded/pasted knowledge source.
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))

    # ===== Outbound URL fetch (knowledge crawl) =====
    URL_FETCH_TIMEOUT: float = float(os.getenv("URL_FETCH_TIMEOUT", 15))
    URL_FETCH_MAX_REDIRECTS: int = int(os.getenv("URL_FETCH_MAX_REDIRECTS", 5))
    # Cap on a fetched page body (defaults to the upload cap).
    MAX_URL_BYTES: int = int(os.getenv("MAX_URL_BYTES", 5 * 1024 * 1024))

    # ===== Object storage (AIStor / MinIO / S3-compatible) =====
    # Env fallbacks; a platform admin overrides these in the DB (config_store /
    # admin storage settings) without a redeploy. Endpoint is host[:port] with NO
    # scheme — TLS is controlled by STORAGE_SECURE. Original KB uploads are stored
    # privately and served via short-lived presigned URLs.
    STORAGE_ENDPOINT: str = os.getenv("STORAGE_ENDPOINT", "s3.dev-stage.net")
    STORAGE_ACCESS_KEY: str = os.getenv("STORAGE_ACCESS_KEY", "")
    STORAGE_SECRET_KEY: str = os.getenv("STORAGE_SECRET_KEY", "")
    STORAGE_BUCKET: str = os.getenv("STORAGE_BUCKET", "")
    STORAGE_SECURE: bool = os.getenv("STORAGE_SECURE", "True").lower() == "true"
    STORAGE_ENABLED: bool = os.getenv("STORAGE_ENABLED", "False").lower() == "true"
    # Lifetime (seconds) of a presigned download URL handed to the dashboard.
    STORAGE_URL_EXPIRY: int = int(os.getenv("STORAGE_URL_EXPIRY", 15 * 60))

    # ===== Website chat widget (public embed) =====
    # Redis backs the widget's rate limit + global GPU concurrency cap across workers.
    # Empty -> per-process in-memory fallback (dev / single worker).
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    # Where the pasted <script src> points (the hosted bundle).
    WIDGET_SRC_URL: str = os.getenv("WIDGET_SRC_URL", os.getenv("SITE_URL", "https://aichatsupport.my").rstrip("/") + "/widget/v1/widget.js")
    # Per public_key + client-IP request cap, per minute.
    WIDGET_RATE_PER_MIN: int = int(os.getenv("WIDGET_RATE_PER_MIN", 20))
    # Global cap on simultaneous in-flight widget LLM calls (protects the shared GPU).
    WIDGET_MAX_CONCURRENCY: int = int(os.getenv("WIDGET_MAX_CONCURRENCY", 4))
    # Default per-bot messages/day when a channel sets no explicit cap.
    WIDGET_DAILY_CAP_DEFAULT: int = int(os.getenv("WIDGET_DAILY_CAP_DEFAULT", 500))

    # ===== WhatsApp channel =====
    WHATSAPP_PROVIDER: str = os.getenv("WHATSAPP_PROVIDER", "meta")
    WHATSAPP_VERIFY_TOKEN: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
    WHATSAPP_APP_SECRET: str = os.getenv("WHATSAPP_APP_SECRET", "")

    # ===== Billing (Billplz) — OFF until the business account is live =====
    BILLING_ENABLED: bool = os.getenv("BILLING_ENABLED", "False").lower() == "true"
    BILLPLZ_API_KEY: str = os.getenv("BILLPLZ_API_KEY", "")
    BILLPLZ_X_SIGNATURE_KEY: str = os.getenv("BILLPLZ_X_SIGNATURE_KEY", "")
    BILLPLZ_COLLECTION_ID: str = os.getenv("BILLPLZ_COLLECTION_ID", "")
    BILLPLZ_SANDBOX_API_KEY: str = os.getenv("BILLPLZ_SANDBOX_API_KEY", "")
    BILLPLZ_SANDBOX_X_SIGNATURE_KEY: str = os.getenv("BILLPLZ_SANDBOX_X_SIGNATURE_KEY", "")
    BILLPLZ_SANDBOX_COLLECTION_ID: str = os.getenv("BILLPLZ_SANDBOX_COLLECTION_ID", "")
    BILLPLZ_SANDBOX: bool = os.getenv("BILLPLZ_SANDBOX", "True").lower() == "true"

    # ===== URLs =====
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3200")
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8100")

    class Config:
        case_sensitive = True


settings = Settings()


def _enforce_secret_key() -> None:
    """Never sign JWTs with a missing or placeholder key.

    In production (DEBUG=False) a weak/absent SECRET_KEY aborts startup — a
    predictable key lets anyone forge a token for any user in any org. In dev we
    fall back to an ephemeral random key (tokens won't survive a restart) so the
    app still boots, with a loud warning.
    """
    key = settings.SECRET_KEY
    if not key or key == _INSECURE_SECRET or len(key) < 16:
        if not settings.DEBUG:
            raise RuntimeError(
                "SECRET_KEY must be set to a strong random value in production "
                "(unset/placeholder keys allow JWT forgery). Generate one with: "
                "python -c 'import secrets; print(secrets.token_urlsafe(48))'"
            )
        settings.SECRET_KEY = secrets.token_urlsafe(48)
        logger.warning("SECRET_KEY missing/weak; using an ephemeral dev key (tokens reset on restart).")


_enforce_secret_key()


def site_url() -> str:
    """Public origin of this deployment's marketing site (one deployment per market)."""
    return os.getenv("SITE_URL", "https://aichatsupport.my").rstrip("/")


def market() -> str:
    """'my' (default) or 'au' - picks the default homepage copy and tone."""
    return "au" if os.getenv("MARKET", "my").strip().lower() == "au" else "my"


def cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        # aitechsupport.my is the pre-rebrand domain: kept as a fallback origin
        # since it now 301-redirects the pages, but any request that lands on
        # the API directly from an old cached page/bookmark still needs to pass.
        "http://localhost:3000,http://localhost:3200," + site_url() + ",https://aichatsupport.my,https://aitechsupport.my",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]

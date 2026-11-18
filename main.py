"""AiChatSupport API — FastAPI entrypoint.

Run: uvicorn main:app --reload --port 8100
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.session import engine, Base
from app.core.settings import settings, cors_origins
from app.core.cors import WidgetCORSMiddleware
from app.core.logging_config import setup_logging
from app.api.v1.api import api_router

# Configure logging before anything else runs, so even boot-time failures below
# are recorded with a timestamp and source rather than a bare traceback.
setup_logging(level=settings.LOG_LEVEL, noisy_level=settings.LOG_LEVEL_HTTP)

# Dev convenience: create tables on boot. In production, migrations are the
# source of truth (`alembic upgrade head`).
import app.models  # noqa: F401 — register all models on Base.metadata
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    from app.core import local_embed

    local_embed.preload_in_background()  # no-op unless QUERY_EMBEDDINGS_LOCAL is on
    yield


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Dashboard/API paths use the static credentialed allowlist; public widget paths
# get per-bot dynamic (credential-less) CORS — see app/core/cors.py.
app.add_middleware(WidgetCORSMiddleware, static_origins=cors_origins())

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def root() -> dict:
    return {"message": "AiChatSupport API", "docs": "/docs"}

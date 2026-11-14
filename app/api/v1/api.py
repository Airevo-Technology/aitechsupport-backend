from fastapi import APIRouter

from app.api.v1.endpoints import health, auth, bots, knowledge, admin, billing, whatsapp, public, content, agent, team, airevo

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(bots.router, prefix="/bots", tags=["bots"])
# Client subscription + packages
api_router.include_router(billing.router, prefix="/billing", tags=["billing"])
# Platform-admin settings, packages, clients
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
# Agent-facing org-wide queue (across all bots/sites)
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
# Owner/admin staff management (create/deactivate support agents)
api_router.include_router(team.router, prefix="/team", tags=["team"])
# Knowledge-base routes use explicit paths (/bots/{id}/knowledge, /knowledge/{id}).
api_router.include_router(knowledge.router, tags=["knowledge"])
# Channel webhook (Meta calls this directly)
api_router.include_router(whatsapp.router, prefix="/webhooks/whatsapp", tags=["whatsapp"])
# Public, JWT-less website-widget endpoints (dynamic per-bot CORS via WidgetCORSMiddleware)
api_router.include_router(public.router, prefix="/public", tags=["public-widget"])
# Public marketing-site content (homepage, pricing, CMS pages, contact)
api_router.include_router(content.router, prefix="/content", tags=["content"])
# airevo.com.au marketplace: signed subscription events + single sign-on (public, HMAC/one-time token)
api_router.include_router(airevo.router, prefix="/airevo", tags=["airevo"])

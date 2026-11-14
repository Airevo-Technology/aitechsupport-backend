# Import every model here so Alembic autogenerate and Base.metadata.create_all
# see all tables and their foreign keys.
from app.models.organization import Organization  # noqa: F401
from app.models.user import User  # noqa: F401
from app.models.bot import Bot  # noqa: F401
from app.models.knowledge import KnowledgeSource, Chunk  # noqa: F401
from app.models.channel import Channel  # noqa: F401
from app.models.conversation import Conversation, Message  # noqa: F401
from app.models.subscription import Subscription  # noqa: F401
from app.models.package import Package  # noqa: F401
from app.models.payment import Payment  # noqa: F401
from app.models.blocked_visitor import BlockedVisitor  # noqa: F401
from app.models.org_invite import OrgInvite  # noqa: F401
from app.models.internal_message import InternalMessage  # noqa: F401
from app.models.game_session import GameSession  # noqa: F401
from app.models.setting import Setting  # noqa: F401
from app.models.email_template import EmailTemplate  # noqa: F401
from app.models.page import Page  # noqa: F401
from app.models.airevo_event import AirevoEvent  # noqa: F401

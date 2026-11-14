"""airevo marketplace: subscriptions billed by airevo.com.au + event dedupe

Revision ID: 3ab13c9527e0
Revises: c1d2e3f4a5b6
Create Date: 2026-11-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '3ab13c9527e0'
down_revision: Union[str, None] = 'c1d2e3f4a5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # "direct" = billed here (Billplz/PayPal/bank transfer); "airevo" = billed by
    # airevo.com.au, which drives the plan through signed events.
    op.add_column('subscriptions', sa.Column('billing_source', sa.String(), nullable=False, server_default='direct'))
    op.add_column('subscriptions', sa.Column('airevo_subscription_id', sa.Integer(), nullable=True))
    op.create_index('ix_subscriptions_airevo_subscription_id', 'subscriptions', ['airevo_subscription_id'], unique=True)
    op.add_column('subscriptions', sa.Column('airevo_period_end', sa.DateTime(), nullable=True))
    op.add_column('subscriptions', sa.Column('airevo_last_event_at', sa.Integer(), nullable=True))
    # True only when the marketplace created this account. SSO never signs into an
    # account that existed before (Airevo does not verify customer emails).
    op.add_column('subscriptions', sa.Column('airevo_sso_allowed', sa.Boolean(), nullable=False, server_default=sa.false()))

    op.create_table(
        'airevo_events',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('airevo_subscription_id', sa.Integer(), nullable=True),
        # The JSON we answered with, so a redelivery gets the same reply.
        sa.Column('response_body', sa.Text(), nullable=True),
        sa.Column('received_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('airevo_events')
    op.drop_column('subscriptions', 'airevo_sso_allowed')
    op.drop_column('subscriptions', 'airevo_last_event_at')
    op.drop_column('subscriptions', 'airevo_period_end')
    op.drop_index('ix_subscriptions_airevo_subscription_id', table_name='subscriptions')
    op.drop_column('subscriptions', 'airevo_subscription_id')
    op.drop_column('subscriptions', 'billing_source')

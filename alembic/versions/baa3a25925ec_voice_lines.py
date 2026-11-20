"""voice lines: per-bot phone numbers (Telnyx + ElevenLabs), call log, voice plan limits

Revision ID: baa3a25925ec
Revises: 3ab13c9527e0
Create Date: 2026-11-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'baa3a25925ec'
down_revision: Union[str, None] = '3ab13c9527e0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('packages', sa.Column('voice_enabled', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('packages', sa.Column('monthly_voice_minutes', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('subscriptions', sa.Column('voice_seconds_used', sa.Integer(), nullable=False, server_default='0'))

    op.create_table(
        'voice_lines',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('bot_id', sa.Integer(), sa.ForeignKey('bots.id'), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='draft'),
        sa.Column('status_message', sa.Text(), nullable=True),
        sa.Column('business_name', sa.String(), nullable=True),
        sa.Column('contact_name', sa.String(), nullable=True),
        sa.Column('contact_phone', sa.String(), nullable=True),
        sa.Column('street_address', sa.String(), nullable=True),
        sa.Column('locality', sa.String(), nullable=True),
        sa.Column('administrative_area', sa.String(), nullable=True),
        sa.Column('postal_code', sa.String(), nullable=True),
        sa.Column('country_code', sa.String(), nullable=False, server_default='AU'),
        sa.Column('document_key', sa.String(), nullable=True),
        sa.Column('document_filename', sa.String(), nullable=True),
        sa.Column('document_mime', sa.String(), nullable=True),
        sa.Column('preferred_locality', sa.String(), nullable=True),
        sa.Column('phone_number', sa.String(), nullable=True),
        sa.Column('telnyx_number_order_id', sa.String(), nullable=True),
        sa.Column('telnyx_phone_number_id', sa.String(), nullable=True),
        sa.Column('telnyx_requirement_group_id', sa.String(), nullable=True),
        sa.Column('telnyx_status', sa.String(), nullable=True),
        sa.Column('voice_id', sa.String(), nullable=True),
        sa.Column('first_message', sa.Text(), nullable=True),
        sa.Column('timezone', sa.String(), nullable=True),
        sa.Column('transfer_number', sa.String(), nullable=True),
        sa.Column('max_call_seconds', sa.Integer(), nullable=True),
        sa.Column('el_agent_id', sa.String(), nullable=True),
        sa.Column('el_phone_number_id', sa.String(), nullable=True),
        sa.Column('el_secret_id', sa.String(), nullable=True),
        sa.Column('llm_secret_hash', sa.String(), nullable=True),
        sa.Column('submitted_at', sa.DateTime(), nullable=True),
        sa.Column('activated_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_voice_lines_organization_id', 'voice_lines', ['organization_id'])
    op.create_index('ix_voice_lines_bot_id', 'voice_lines', ['bot_id'], unique=True)

    op.create_table(
        'voice_calls',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('voice_line_id', sa.Integer(), sa.ForeignKey('voice_lines.id'), nullable=False),
        sa.Column('conversation_id', sa.Integer(), sa.ForeignKey('conversations.id'), nullable=True),
        sa.Column('caller_number', sa.String(), nullable=True),
        sa.Column('duration_secs', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_voice_calls_organization_id', 'voice_calls', ['organization_id'])
    op.create_index('ix_voice_calls_voice_line_id', 'voice_calls', ['voice_line_id'])


def downgrade() -> None:
    op.drop_table('voice_calls')
    op.drop_table('voice_lines')
    op.drop_column('subscriptions', 'voice_seconds_used')
    op.drop_column('packages', 'monthly_voice_minutes')
    op.drop_column('packages', 'voice_enabled')

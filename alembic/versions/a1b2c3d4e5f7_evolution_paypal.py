"""whatsapp provider per channel; paypal payments + currency

Revision ID: a1b2c3d4e5f7
Revises: f6a7b8c9d0e1
Create Date: 2026-11-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f7'
down_revision: Union[str, None] = 'f6a7b8c9d0e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Existing channels are all Fonnte; new ones pick the platform default.
    op.add_column('channels', sa.Column('provider', sa.String(), nullable=False, server_default='fonnte'))
    op.add_column('payments', sa.Column('paypal_order_id', sa.String(), nullable=True))
    op.create_index('ix_payments_paypal_order_id', 'payments', ['paypal_order_id'], unique=True)
    op.add_column('payments', sa.Column('currency', sa.String(), nullable=False, server_default='MYR'))


def downgrade() -> None:
    op.drop_column('payments', 'currency')
    op.drop_index('ix_payments_paypal_order_id', table_name='payments')
    op.drop_column('payments', 'paypal_order_id')
    op.drop_column('channels', 'provider')

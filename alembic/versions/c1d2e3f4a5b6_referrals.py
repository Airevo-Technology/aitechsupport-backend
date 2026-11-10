"""referral program: org referral_code + referred_by

Revision ID: c1d2e3f4a5b6
Revises: a1b2c3d4e5f7
Create Date: 2026-11-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c1d2e3f4a5b6'
down_revision: Union[str, None] = 'a1b2c3d4e5f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Codes are generated lazily the first time an org opens its referral page.
    op.add_column('organizations', sa.Column('referral_code', sa.String(), nullable=True))
    op.create_index('ix_organizations_referral_code', 'organizations', ['referral_code'], unique=True)
    op.add_column('organizations', sa.Column('referred_by_organization_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_organizations_referred_by', 'organizations', 'organizations',
        ['referred_by_organization_id'], ['id'],
    )
    op.create_index('ix_organizations_referred_by', 'organizations', ['referred_by_organization_id'])


def downgrade() -> None:
    op.drop_index('ix_organizations_referred_by', table_name='organizations')
    op.drop_constraint('fk_organizations_referred_by', 'organizations', type_='foreignkey')
    op.drop_column('organizations', 'referred_by_organization_id')
    op.drop_index('ix_organizations_referral_code', table_name='organizations')
    op.drop_column('organizations', 'referral_code')

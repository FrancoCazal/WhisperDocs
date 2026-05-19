"""add cost_usd column to query_logs + widen model_used to 100

Revision ID: 0008
Revises: 0007
Create Date: 2026-04-27

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "query_logs",
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0.0"),
    )
    op.alter_column("query_logs", "model_used", type_=sa.String(100), existing_nullable=False)


def downgrade() -> None:
    op.drop_column("query_logs", "cost_usd")
    op.alter_column("query_logs", "model_used", type_=sa.String(50), existing_nullable=False)

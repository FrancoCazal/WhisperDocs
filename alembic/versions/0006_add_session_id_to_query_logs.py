"""add session_id to query_logs

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-26

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("query_logs", sa.Column("session_id", sa.String(50), nullable=True))
    op.create_index("ix_query_logs_session_id", "query_logs", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_query_logs_session_id", table_name="query_logs")
    op.drop_column("query_logs", "session_id")

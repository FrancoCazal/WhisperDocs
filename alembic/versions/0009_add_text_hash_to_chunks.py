"""add text_hash column to chunks for dedup

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-27

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("text_hash", sa.String(64), nullable=True))
    op.create_index("ix_chunks_text_hash", "chunks", ["text_hash"])


def downgrade() -> None:
    op.drop_index("ix_chunks_text_hash", table_name="chunks")
    op.drop_column("chunks", "text_hash")

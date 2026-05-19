"""add tsvector column + GIN index on chunks for BM25 hybrid search

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-27

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
        """
    )
    op.execute(
        "CREATE INDEX ix_chunks_tsv ON chunks USING GIN (tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS tsv")

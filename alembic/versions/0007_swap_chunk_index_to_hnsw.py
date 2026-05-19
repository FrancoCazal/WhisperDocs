"""swap chunks embedding index from ivfflat to hnsw

IVFFlat with default probes=1 has terrible recall on small datasets
(misses most chunks when no metadata filter narrows the search). HNSW
gives near-perfect recall out of the box, with no tuning required for
small/medium datasets.

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-27

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunk_embedding")
    op.execute(
        """
        CREATE INDEX idx_chunk_embedding
        ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunk_embedding")
    op.execute(
        """
        CREATE INDEX idx_chunk_embedding
        ON chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )

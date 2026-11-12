"""add source_uri to documents, unique per collection

A document synced from an MCP server (audit M-6) keeps the URI of the
resource it was read from, so a later sync finds it again: unchanged, it is
left alone; changed, it is replaced. Uploads have none, and a unique index
leaves nulls unconstrained.

Revision ID: 0024
Revises: 0023
Create Date: 2026-11-10

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0024"
down_revision: Union[str, None] = "0023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("source_uri", sa.String(2048), nullable=True))
    op.create_index(
        "uq_documents_collection_source",
        "documents",
        ["collection_id", "source_uri"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_documents_collection_source", table_name="documents")
    op.drop_column("documents", "source_uri")

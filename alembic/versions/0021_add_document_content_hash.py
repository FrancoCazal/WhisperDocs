"""add content_hash to documents, unique per collection

A collection holds each document once (audit R12): `content_hash` is the
SHA-256 of `raw_text`, and a unique index on (collection_id, content_hash)
keeps two uploads of the same file apart even when they run at once.

Existing rows get their hash here, computed in Postgres as the app computes
it in Python. When a collection already holds the same text twice, only the
oldest copy gets it: the others keep null, which the unique index allows, so
the migration needs no data removed.

Revision ID: 0021
Revises: 0020
Create Date: 2026-11-10

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0021"
down_revision: Union[str, None] = "0020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("content_hash", sa.String(64), nullable=True))
    op.execute(
        """
        UPDATE documents d
        SET content_hash = h.digest
        FROM (
            SELECT DISTINCT ON (collection_id, digest) id, digest
            FROM (
                SELECT id, collection_id,
                       encode(sha256(convert_to(raw_text, 'UTF8')), 'hex') AS digest
                FROM documents
            ) hashed
            ORDER BY collection_id, digest, id
        ) h
        WHERE d.id = h.id
        """
    )
    op.create_index(
        "uq_documents_collection_content",
        "documents",
        ["collection_id", "content_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_documents_collection_content", table_name="documents")
    op.drop_column("documents", "content_hash")

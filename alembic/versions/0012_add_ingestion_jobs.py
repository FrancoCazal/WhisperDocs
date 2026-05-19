"""add ingestion_jobs table

Revision ID: 0012
Revises: 0011
Create Date: 2026-04-27

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "collection_id",
            sa.Integer(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.Integer(),
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("filename", sa.String(300), nullable=False),
        sa.Column("file_type", sa.String(10), nullable=False),
        sa.Column("title", sa.String(300), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_ingestion_jobs_collection_id", "ingestion_jobs", ["collection_id"]
    )
    op.create_index(
        "ix_ingestion_jobs_created_at", "ingestion_jobs", ["created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_ingestion_jobs_created_at", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_collection_id", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")

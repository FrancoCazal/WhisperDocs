"""add instruction_flags to chunks and flagged_chunks to documents

Ingestion marks each chunk holding text that reads as an instruction to the
model, with the rules it matched (app/services/instruction_scan.py), and
counts them per document. Rows from before have no flags until
scripts/flag_instructions.py scans them.

Revision ID: 0018
Revises: 0017
Create Date: 2026-11-08

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018"
down_revision: Union[str, None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "instruction_flags",
            postgresql.ARRAY(sa.String(16)),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.add_column(
        "documents",
        sa.Column("flagged_chunks", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("documents", "flagged_chunks")
    op.drop_column("chunks", "instruction_flags")

"""add section, page_start and page_end to chunks

Where each chunk sits in its document, so a citation can name it (audit
M-7): the headings the chunk is under, and the pages a PDF's chunk spans.
Chunks stored before have neither until their document is uploaded again.

Revision ID: 0020
Revises: 0019
Create Date: 2026-11-10

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0020"
down_revision: Union[str, None] = "0019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("section", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("page_start", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("page_end", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "page_end")
    op.drop_column("chunks", "page_start")
    op.drop_column("chunks", "section")

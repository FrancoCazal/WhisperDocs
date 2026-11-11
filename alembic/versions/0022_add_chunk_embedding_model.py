"""add embedding_model to chunks

The model that made each chunk's vector (audit R19): a vector is reused for
the same text only when the same model made it, and a vector from another
model can be found. Chunks stored before keep null, since nothing recorded
which model made them.

Revision ID: 0022
Revises: 0021
Create Date: 2026-11-10

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0022"
down_revision: Union[str, None] = "0021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("embedding_model", sa.String(100), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "embedding_model")

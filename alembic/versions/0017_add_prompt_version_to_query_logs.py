"""add prompt_version to query_logs

The version of the prompts that answered (app/agent/prompt_version.py), 12
hex characters, so a change in cost or quality in the logs lines up with the
prompt change that made it. Rows from before have none.

Revision ID: 0017
Revises: 0016
Create Date: 2026-11-08

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: Union[str, None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("query_logs", sa.Column("prompt_version", sa.String(12), nullable=True))


def downgrade() -> None:
    op.drop_column("query_logs", "prompt_version")

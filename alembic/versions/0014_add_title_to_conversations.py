"""add title to conversations

A thread's title is its first question, so the thread list can say what each
one is about. Threads from before take the first question their query log
holds (only answers from the document search are logged); the rest stay
untitled until their next message.

Revision ID: 0014
Revises: 0013
Create Date: 2026-11-07

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("conversations", sa.Column("title", sa.String(200), nullable=True))
    op.execute(
        """
        UPDATE conversations AS c
        SET title = CASE
            WHEN length(q.line) > 200 THEN left(q.line, 199) || '…'
            ELSE q.line
        END
        FROM (
            SELECT DISTINCT ON (session_id)
                session_id,
                regexp_replace(btrim(question), '\\s+', ' ', 'g') AS line
            FROM query_logs
            WHERE session_id IS NOT NULL
            ORDER BY session_id, created_at
        ) AS q
        WHERE c.thread_id = q.session_id AND c.title IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column("conversations", "title")

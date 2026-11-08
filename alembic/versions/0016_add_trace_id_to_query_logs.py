"""add trace_id to query_logs

The OpenTelemetry trace of the turn that wrote the row (app/telemetry.py), as
32 hex characters: a row in the stats leads to the turn's spans, and a trace
in the viewer to its row. Rows from before have none.

Revision ID: 0016
Revises: 0015
Create Date: 2026-11-08

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: Union[str, None] = "0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("query_logs", sa.Column("trace_id", sa.String(32), nullable=True))
    op.create_index("ix_query_logs_trace_id", "query_logs", ["trace_id"])


def downgrade() -> None:
    op.drop_index("ix_query_logs_trace_id", table_name="query_logs")
    op.drop_column("query_logs", "trace_id")

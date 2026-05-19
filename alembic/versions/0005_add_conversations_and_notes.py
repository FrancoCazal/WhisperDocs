"""add conversations and saved_notes tables

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-23

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("thread_id", sa.String(length=100), nullable=False),
        sa.Column("user_number", sa.String(length=30), nullable=True),
        sa.Column(
            "collection_id",
            sa.Integer(),
            sa.ForeignKey("collections.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "is_saved",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column(
            "channel",
            sa.String(length=20),
            server_default="web",
            nullable=False,
        ),
        sa.Column(
            "last_active_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("thread_id", name="uq_conversations_thread_id"),
    )
    op.create_index("ix_conversations_last_active", "conversations", ["last_active_at"])
    op.create_index("ix_conversations_user_number", "conversations", ["user_number"])
    op.create_index("ix_conversations_is_saved", "conversations", ["is_saved"])
    op.create_index("ix_conversations_channel", "conversations", ["channel"])

    op.create_table(
        "saved_notes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.Integer(),
            sa.ForeignKey("conversations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "collection_id",
            sa.Integer(),
            sa.ForeignKey("collections.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source_message_content", sa.Text(), nullable=False),
        sa.Column("user_note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_saved_notes_conversation", "saved_notes", ["conversation_id"])
    op.create_index("ix_saved_notes_collection", "saved_notes", ["collection_id"])


def downgrade() -> None:
    op.drop_index("ix_saved_notes_collection", table_name="saved_notes")
    op.drop_index("ix_saved_notes_conversation", table_name="saved_notes")
    op.drop_table("saved_notes")
    op.drop_index("ix_conversations_channel", table_name="conversations")
    op.drop_index("ix_conversations_is_saved", table_name="conversations")
    op.drop_index("ix_conversations_user_number", table_name="conversations")
    op.drop_index("ix_conversations_last_active", table_name="conversations")
    op.drop_table("conversations")

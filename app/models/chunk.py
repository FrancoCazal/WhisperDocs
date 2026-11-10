from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, Index, Integer, String, Text
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import EMBEDDING_COLUMN_DIMENSION
from app.db import Base

if TYPE_CHECKING:
    from app.models.document import Document

# As Postgres prints the expression of alembic 0013, so `alembic check` finds
# nothing to compare. Keep in sync with app.services.fulltext.TS_CONFIGS.
TSV_EXPRESSION = """CASE language
    WHEN 'en'::text THEN to_tsvector('wd_english'::regconfig, text)
    WHEN 'es'::text THEN to_tsvector('wd_spanish'::regconfig, text)
    ELSE to_tsvector('wd_simple'::regconfig, text)
END"""


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # Copied from the document. Picks the text search configuration of `tsv`.
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_COLUMN_DIMENSION))
    # The full-text leg's lexemes (app/services/fulltext.py), which Postgres
    # computes from `text` and `language`. Deferred: no read of a chunk needs
    # them, only the WHERE of the full-text query.
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed(TSV_EXPRESSION, persisted=True), deferred=True
    )
    # The rules of app/services/instruction_scan.py the text matched when it was
    # stored, or when scripts/flag_instructions.py last ran: text that reads as
    # an instruction to the model. Empty for plain text.
    instruction_flags: Mapped[list[str]] = mapped_column(
        ARRAY(String(16)), nullable=False, default=list, server_default=sql_text("'{}'")
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index(
            "idx_chunk_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )
    # An insert would otherwise fetch the generated `tsv` with RETURNING, every
    # chunk's lexemes for nothing: it stays expired until read, which nothing does.
    __mapper_args__ = {"eager_defaults": False}

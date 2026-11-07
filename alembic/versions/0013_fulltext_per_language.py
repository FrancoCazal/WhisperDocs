"""full-text search per document language, accent-insensitive

The tsv column used the `simple` configuration: no stemming, no stopword
removal, accents kept. Combined with an AND tsquery, natural-language questions
almost never matched (audit R1). This revision:

- adds `chunks.language`, copied from `documents.language`;
- adds text search configurations `wd_english`, `wd_spanish` and `wd_simple`,
  copies of the built-ins with `unaccent` in front of the stemmer;
- regenerates `tsv` with the configuration that matches the chunk's language.

Revision ID: 0013
Revises: 0012
Create Date: 2026-11-06

"""

from typing import Sequence, Union

from alembic import op

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# (configuration, built-in it copies, stemming dictionary)
_CONFIGS = (
    ("wd_english", "english", "english_stem"),
    ("wd_spanish", "spanish", "spanish_stem"),
    ("wd_simple", "simple", "simple"),
)

# Keep in sync with app.services.fulltext.TS_CONFIG_BY_LANGUAGE.
_TSV_EXPRESSION = """
    CASE language
        WHEN 'en' THEN to_tsvector('wd_english'::regconfig, text)
        WHEN 'es' THEN to_tsvector('wd_spanish'::regconfig, text)
        ELSE to_tsvector('wd_simple'::regconfig, text)
    END
"""


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
    for name, base, stem in _CONFIGS:
        op.execute(f"CREATE TEXT SEARCH CONFIGURATION {name} (COPY = {base})")
        # Non-ASCII token types go through unaccent first; ASCII ones need nothing.
        op.execute(
            f"ALTER TEXT SEARCH CONFIGURATION {name} "
            f"ALTER MAPPING FOR hword, hword_part, word WITH unaccent, {stem}"
        )

    op.execute("ALTER TABLE chunks ADD COLUMN language VARCHAR(8)")
    op.execute(
        "UPDATE chunks c SET language = d.language FROM documents d WHERE d.id = c.document_id"
    )

    op.execute("DROP INDEX IF EXISTS ix_chunks_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN tsv")
    op.execute(
        f"ALTER TABLE chunks ADD COLUMN tsv tsvector GENERATED ALWAYS AS ({_TSV_EXPRESSION}) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING GIN (tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN tsv")
    op.execute(
        "ALTER TABLE chunks ADD COLUMN tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING GIN (tsv)")
    op.execute("ALTER TABLE chunks DROP COLUMN language")
    for name, _base, _stem in _CONFIGS:
        op.execute(f"DROP TEXT SEARCH CONFIGURATION IF EXISTS {name}")
    # The unaccent extension stays: other objects may depend on it.

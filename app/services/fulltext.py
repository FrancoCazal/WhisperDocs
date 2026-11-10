"""Full-text (lexical) search over `chunks.tsv`. Postgres ranking, not BM25.

`chunks.tsv` is generated with the text search configuration that matches the
chunk's language (alembic 0013): `wd_english` for "en", `wd_spanish` for "es",
`wd_simple` otherwise. All three run `unaccent` before stemming.

The question's language is not known, so the query is built with every
configuration and the results are OR-ed. Each chunk only stores lexemes that
survived its own configuration (stopwords removed, words stemmed), so a lexeme
from the "wrong" configuration rarely matches anything.
"""

from __future__ import annotations

# Every configuration a chunk's tsv can be built with. Keep in sync with the
# CASE expression in alembic/versions/0013_fulltext_per_language.py, which
# app/models/chunk.py's TSV_EXPRESSION repeats.
TS_CONFIGS: tuple[str, ...] = ("wd_english", "wd_spanish", "wd_simple")


def or_tsquery_sql(param: str = "q") -> str:
    """SQL expression for an OR tsquery over the lexemes of `:param`.

    `plainto_tsquery` normalizes the text (stemming, stopwords, unaccent) and
    joins the lexemes with `&`; turning those into `|` gives the OR form.
    Unlike `websearch_to_tsquery`, it never reads "or" or "-word" in the
    question as operators. A question made only of stopwords gives an empty
    tsquery, which matches nothing.
    """
    parts = [
        f"replace(plainto_tsquery('{cfg}'::regconfig, :{param})::text, '&', '|')::tsquery"
        for cfg in TS_CONFIGS
    ]
    return " || ".join(parts)

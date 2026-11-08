"""contains_text: the text search behind the thread list and the query log
(frontend review U11, U16)."""

from app.db import contains_text
from app.models import QueryLog


def test_the_like_wildcards_in_the_text_match_themselves():
    expr = contains_text(QueryLog.question, "50%_a\\b")

    # The pattern escapes \, % and _, and says which character escapes.
    assert expr.right.value == "%50\\%\\_a\\\\b%"
    assert expr.modifiers["escape"] == "\\"


def test_the_match_ignores_case():
    sql = str(contains_text(QueryLog.question, "x").compile())

    assert "lower(" in sql

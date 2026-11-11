"""The WhatsApp formatter strips markdown only where it is markdown, and lists
the sources an answer cites (audit A16)."""

from __future__ import annotations

import pytest

from app.services.whatsapp.formatter import FALLBACK_MSG, MAX_CHARS, format_for_whatsapp


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Identifiers keep their underscores, operators their stars.
        ("Set max_tokens and top_k_results.", "Set max_tokens and top_k_results."),
        ("Call __init__ first.", "Call __init__ first."),
        ("2 * 3 * 4 is 24, and a*b is a product.", "2 * 3 * 4 is 24, and a*b is a product."),
        # "C#" is not a heading, and a # inside a line is text.
        ("Use C# or F# here, issue #12.", "Use C# or F# here, issue #12."),
        # What is markdown goes.
        ("## Setup\nRun it.", "Setup\nRun it."),
        (
            "This is **bold**, __also this__, *italic* and _this_.",
            "This is bold, also this, italic and this.",
        ),
        ("See [the docs](https://example.com).", "See the docs."),
        ("- one\n- two", "one\ntwo"),
    ],
)
def test_markdown_goes_and_text_stays(text, expected):
    assert format_for_whatsapp(text) == expected


def test_code_keeps_everything_inside_it():
    text = "Run `my_var * 2` then:\n```\n# not a heading\nx = a_b * c\n```"
    assert format_for_whatsapp(text) == text


def _citations(*titles: str) -> list[dict]:
    return [{"document_title": title} for title in titles]


def test_the_footer_lists_only_the_cited_sources_each_title_once():
    citations = _citations("guide.md", "faq.md", "guide.md", "unused.md")

    out = format_for_whatsapp("A [1] and B [3], also [2].", citations, language="en")

    assert out.endswith("— Sources: [1][3] guide.md · [2] faq.md")
    assert "unused.md" not in out


def test_an_answer_that_cites_nothing_has_no_footer():
    out = format_for_whatsapp("I don't have that information.", _citations("guide.md"))
    assert "Fuentes" not in out and "guide.md" not in out


def test_a_number_past_the_sources_is_not_a_citation():
    out = format_for_whatsapp("See [1] and [7].", _citations("guide.md"), language="es")
    assert out.endswith("— Fuentes: [1] guide.md")


def test_the_reply_language_picks_the_footer_and_the_marks():
    assert format_for_whatsapp("A [1].", _citations("g.md"), language="es").endswith(
        "— Fuentes: [1] g.md"
    )
    long = "Sentence. " * (MAX_CHARS // 5)
    assert format_for_whatsapp(long, language="en").endswith("… [truncated]")
    assert format_for_whatsapp(long, language="es").endswith("… [truncado]")
    assert format_for_whatsapp("   ", language="en") == FALLBACK_MSG["en"]
    assert format_for_whatsapp("   ") == FALLBACK_MSG["es"]


def test_an_untitled_source_is_named_by_its_number():
    out = format_for_whatsapp("A [1].", [{}], language="en")
    assert out.endswith("— Sources: [1] Source 1")

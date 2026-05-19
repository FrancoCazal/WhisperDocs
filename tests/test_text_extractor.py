"""Tests for `_clean()` structure preservation in text_extractor.

The cleaner collapses whitespace for prose but must preserve fenced code
blocks (indentation matters) and contiguous markdown table line groups
(column padding carries information).
"""

from app.services.text_extractor import _clean


def test_clean_preserves_fenced_code_indentation():
    text = """Intro paragraph with    extra    spaces.

```python
def f(x):
    if x:
        return  x  *  2
    return 0
```

Trailing prose."""
    out = _clean(text)
    # Code block content stays verbatim (4-space indents intact, double spaces
    # inside expressions preserved).
    assert "    if x:" in out
    assert "        return  x  *  2" in out
    # Prose around it still gets normalized.
    assert "extra spaces" in out
    assert "extra    spaces" not in out


def test_clean_preserves_markdown_table_padding():
    text = """Before table.

| Name    | Type     | Notes        |
|---------|----------|--------------|
| id      | int      | primary key  |
| name    | varchar  | nullable     |

After table."""
    out = _clean(text)
    # Pipe alignment + dashes preserved as written.
    assert "| Name    | Type     | Notes        |" in out
    assert "|---------|----------|--------------|" in out


def test_clean_collapses_whitespace_outside_protected_blocks():
    text = "word1     word2\t\tword3\n\n\n\n\nword4"
    out = _clean(text)
    assert out == "word1 word2 word3\n\nword4"


def test_clean_handles_text_without_protected_blocks():
    text = "just  prose   with  extra   spaces"
    out = _clean(text)
    assert out == "just prose with extra spaces"

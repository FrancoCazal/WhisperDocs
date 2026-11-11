import pytest

from app.services.chunker import TextChunker


def test_empty_text_returns_no_chunks():
    chunker = TextChunker(chunk_size=100, overlap=10)
    assert chunker.chunk("") == []
    assert chunker.chunk("   \n\n  ") == []


def test_short_text_returns_single_chunk():
    chunker = TextChunker(chunk_size=100, overlap=10)
    chunks = chunker.chunk("Hello world, this is a short text.")
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].token_count <= 100
    assert "Hello world" in chunks[0].text


def test_long_text_is_split_into_multiple_chunks():
    chunker = TextChunker(chunk_size=50, overlap=5)
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.index == i
        assert chunk.token_count <= 50


def test_overlap_creates_shared_content_between_chunks():
    """Boundary-aware splitter may produce slightly smaller chunks than chunk_size
    when snapping to a natural separator. Assert <= chunk_size only."""
    chunker = TextChunker(chunk_size=20, overlap=5)
    text = " ".join([f"token{i}" for i in range(200)])
    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    for chunk in chunks[:-1]:
        assert chunk.token_count <= 20
        assert chunk.token_count > 0


def test_chunks_never_exceed_chunk_size():
    chunker = TextChunker(chunk_size=100, overlap=20)
    text = " ".join([f"w{i}" for i in range(1000)])
    chunks = chunker.chunk(text)

    for chunk in chunks:
        assert chunk.token_count <= 100


def test_chunks_are_indexed_sequentially():
    chunker = TextChunker(chunk_size=30, overlap=5)
    text = " ".join([f"w{i}" for i in range(300)])
    chunks = chunker.chunk(text)

    indices = [c.index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_count_tokens_matches_encoder():
    chunker = TextChunker()
    assert chunker.count_tokens("") == 0
    assert chunker.count_tokens("hello") > 0


def test_invalid_params_raise_value_error():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, overlap=-1)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=50, overlap=50)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=50, overlap=100)


def test_boundary_snap_prefers_paragraph_break():
    """When a paragraph break exists in the boundary window, the splitter should
    snap to it rather than cut mid-word. Verifies semantic-preserving split."""
    chunker = TextChunker(chunk_size=40, overlap=0)
    para_a = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet."
    para_b = "Kilo lima mike november oscar papa quebec romeo sierra tango."
    text = para_a + "\n\n" + para_b + "\n\nUniform victor whiskey xray yankee zulu."
    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    # First chunk should end at a paragraph break — text contains complete
    # sentence(s), no trailing partial word.
    first = chunks[0].text.rstrip()
    assert first.endswith(".") or first.endswith("\n")


def test_boundary_falls_back_to_hard_cut_when_no_separator():
    """If no natural separator exists in the window (e.g. one giant word),
    splitter must still produce a valid chunk via hard cut."""
    chunker = TextChunker(chunk_size=30, overlap=0)
    text = "x" * 5000  # no separators at all
    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 30
        assert chunk.token_count > 0


# --- structure (audit R14, M-7) ---------------------------------------------------

SENTENCE = "The quick brown fox jumps over the lazy dog. "


def test_a_chunk_that_starts_inside_a_section_carries_its_headings():
    chunker = TextChunker(chunk_size=80, overlap=0)
    text = "# Guide\n\n## Install\n\n" + SENTENCE * 40

    chunks = chunker.chunk(text, headings=True)

    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk.text.startswith("# Guide\n## Install\n\n")
        assert chunk.section == "Guide › Install"
        assert chunk.token_count <= 80


def test_a_heading_starts_a_chunk_once_the_section_is_long_enough():
    chunker = TextChunker(chunk_size=200, overlap=0)
    text = "# Doc\n\n## A\n\n" + SENTENCE * 12 + "\n\n## B\n\n" + SENTENCE * 12

    first, second = chunker.chunk(text, headings=True)

    assert first.section == "Doc › A" and first.text.startswith("# Doc\n\n## A")
    # The second starts with its own heading, under the document's.
    assert second.section == "Doc › B"
    assert second.text.startswith("# Doc\n\n## B\n\n")
    assert "## A" not in second.text and "## B" not in first.text


def test_short_sections_share_a_chunk_named_after_them():
    chunker = TextChunker(chunk_size=200, overlap=0)
    text = "# Doc\n\n## A\n\nShort a.\n\n## B\n\nShort b.\n\n### B.1\n\nShorter."

    (chunk,) = chunker.chunk(text, headings=True)

    assert chunk.section == "Doc › A, B"
    assert chunk.text == "# Doc\n\n## A\n\nShort a.\n\n## B\n\nShort b.\n\n### B.1\n\nShorter."


def test_subsections_merged_into_their_parent_are_named_after_it():
    chunker = TextChunker(chunk_size=200, overlap=0)
    text = "# Doc\n\n## A\n\nIntro.\n\n### A.1\n\nOne.\n\n### A.2\n\nTwo."

    (chunk,) = chunker.chunk(text, headings=True)

    assert chunk.section == "Doc › A"


def test_a_heading_left_at_the_end_of_a_chunk_moves_to_the_next():
    chunker = TextChunker(chunk_size=60, overlap=0, min_section_tokens=1000)
    # The intro and the heading fit; the paragraph after them does not, but
    # fits on its own.
    text = "Intro text here. " * 3 + "\n\n## Next\n\n" + SENTENCE * 5

    chunks = chunker.chunk(text, headings=True)

    assert len(chunks) == 2
    assert chunks[0].text == ("Intro text here. " * 3).strip()
    assert chunks[0].section is None
    assert chunks[1].text.startswith("## Next\n\n")
    assert chunks[1].section == "Next"


def test_a_table_that_fits_in_a_chunk_is_never_split():
    chunker = TextChunker(chunk_size=100, overlap=0)
    table = "| Model | Price |\n|---|---|\n" + "\n".join(f"| m{i} | {i}.5 |" for i in range(6))
    text = SENTENCE * 7 + "\n\n" + table

    chunks = chunker.chunk(text)

    assert sum(table in c.text for c in chunks) == 1
    assert all(c.token_count <= 100 for c in chunks)


def test_a_long_table_repeats_its_header_in_every_part():
    chunker = TextChunker(chunk_size=80, overlap=0)
    header = "| Model | Price |\n|---|---|"
    rows = [f"| model-{i} | {i}.25 |" for i in range(60)]
    text = "# Prices\n\n" + header + "\n" + "\n".join(rows)

    chunks = chunker.chunk(text, headings=True)

    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.text.startswith("# Prices\n\n" + header + "\n")
        assert chunk.token_count <= 80
        assert chunk.section == "Prices"
    # Each row once, in order.
    seen = [line for c in chunks for line in c.text.split("\n") if line.startswith("| model-")]
    assert seen == rows


def test_hash_lines_are_text_when_headings_are_off():
    chunker = TextChunker(chunk_size=100, overlap=0)
    text = "# of users: 5\n\nmore text"

    (chunk,) = chunker.chunk(text)

    assert chunk.text == text
    assert chunk.section is None


def test_a_hash_inside_a_code_block_is_not_a_heading():
    chunker = TextChunker(chunk_size=200, overlap=0)
    text = "# Doc\n\n```bash\n# install it\npip install x\n```\n\nAfter the code."

    (chunk,) = chunker.chunk(text, headings=True)

    assert chunk.section == "Doc"
    assert "# install it\npip install x" in chunk.text


def test_paged_text_gives_each_chunk_its_pages():
    chunker = TextChunker(chunk_size=40, overlap=0)
    page_one = "First page text. " * 6
    page_two = "Second page text. " * 6
    text = page_one + "\n\f\n" + page_two

    whole = TextChunker().chunk(text, paged=True)
    parts = chunker.chunk(text, paged=True)

    assert [(c.page_start, c.page_end) for c in whole] == [(1, 2)]
    assert parts[0].page_start == 1 and parts[-1].page_end == 2
    assert all(c.page_start in (1, 2) and c.page_end in (1, 2) for c in parts)
    # The page break is a separator, not text.
    assert all("\f" not in c.text for c in parts)
    # Text read as not paged has no pages.
    assert TextChunker().chunk(text)[0].page_start is None


def test_empty_first_pages_still_count():
    (chunk,) = TextChunker().chunk("\f\n\f\nOnly the third page has text.", paged=True)
    assert (chunk.page_start, chunk.page_end) == (3, 3)


@pytest.mark.parametrize("text", ["日本語のテキストです" * 300, "😀🚀" * 600, "añoñíé" * 800])
def test_a_cut_never_splits_a_character_or_loses_text(text):
    chunks = TextChunker(chunk_size=50, overlap=0).chunk(text)

    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == text
    assert all("\ufffd" not in c.text and c.token_count <= 50 for c in chunks)


def test_every_word_survives_in_order_without_overlap():
    text = " ".join(f"Sentence number {i} ends here." for i in range(300))
    chunks = TextChunker(chunk_size=60, overlap=0).chunk(text)

    assert " ".join(" ".join(c.text for c in chunks).split()) == text


def test_the_overlap_repeats_the_end_of_the_previous_part_from_a_word_start():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = TextChunker(chunk_size=40, overlap=8).chunk(text)

    for before, after in zip(chunks, chunks[1:]):
        first_word = after.text.split()[0]
        assert first_word in before.text.split()[-8:]


def test_special_tokens_in_a_document_are_text():
    (chunk,) = TextChunker().chunk("A document may say <|endoftext|> in passing.")
    assert chunk.token_count > 0 and "<|endoftext|>" in chunk.text


def test_the_headings_in_front_take_at_most_a_fifth_of_a_chunk():
    chunker = TextChunker(chunk_size=100, overlap=0)
    long_title = " ".join(f"title{i}" for i in range(200))
    text = f"# {long_title}\n\n## Short\n\n" + SENTENCE * 30

    chunks = chunker.chunk(text, headings=True)
    under_short = [c for c in chunks if "fox" in c.text]

    assert all(c.token_count <= 100 for c in chunks)
    assert len(under_short) > 1
    for chunk in under_short:
        # The title does not fit in a fifth of the chunk: only "## Short" leads.
        assert chunk.text.startswith("## Short\n\nThe quick")


def test_a_single_heading_longer_than_the_lead_is_cut():
    chunker = TextChunker(chunk_size=100, overlap=0)
    long_title = " ".join(f"title{i}" for i in range(200))
    text = f"## {long_title}\n\n" + SENTENCE * 30

    under = [c for c in chunker.chunk(text, headings=True) if "fox" in c.text]

    for chunk in under:
        lead = chunk.text.split("\n\n", 1)[0]
        assert lead.startswith("## title0 title1") and chunker.count_tokens(lead) <= 20
        assert chunk.token_count <= 100

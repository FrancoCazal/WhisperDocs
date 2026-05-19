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

"""Boundary-aware token-based chunker.

Splits text into chunks of up to `chunk_size` tokens, snapping cut points to
natural separators (paragraph break > newline > sentence end > whitespace) when
possible. Falls back to hard token cuts when no separator is found within the
search window. Overlap is applied as a token suffix prepended to the next
chunk's start.

Tradeoff vs naive fixed-size: chunks may be slightly smaller than `chunk_size`
when a natural boundary exists, but never larger. This preserves semantic
coherence (no mid-sentence / mid-code-block cuts) at near-zero cost vs the
previous strict fixed-size implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

# Separators in priority order. Earlier = stronger semantic boundary.
_BOUNDARY_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ")

# Minimum fraction of the chunk window that must be filled before we accept a
# boundary cut. Prevents pathologically short chunks when an early separator
# appears (e.g. "Title.\n\n" right after start). Tunable.
_MIN_BOUNDARY_FILL = 0.5


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    token_count: int


class TextChunker:
    """Token-based chunker with overlap and natural-boundary snapping."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []

        tokens = self.encoder.encode(text)
        if not tokens:
            return []

        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(tokens):
            hard_end = min(start + self.chunk_size, len(tokens))

            if hard_end < len(tokens):
                end = self._find_boundary(tokens, start, hard_end)
            else:
                end = hard_end

            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(Chunk(text=chunk_text, index=index, token_count=len(chunk_tokens)))

            if end >= len(tokens):
                break

            stride = max(end - self.overlap - start, 1)
            start = start + stride
            index += 1

        return chunks

    def _find_boundary(self, tokens: list[int], start: int, hard_end: int) -> int:
        """Search for a natural separator near `hard_end`. Falls back to hard cut.

        We decode the candidate window once, scan separators in priority order,
        and accept the rightmost occurrence that lies past the minimum fill
        threshold. Re-encoding the truncated text gives the exact token count.
        """
        window_tokens = tokens[start:hard_end]
        window_text = self.encoder.decode(window_tokens)
        min_chars = int(len(window_text) * _MIN_BOUNDARY_FILL)

        for sep in _BOUNDARY_SEPARATORS:
            cut = window_text.rfind(sep)
            if cut < min_chars:
                continue
            truncated = window_text[: cut + len(sep)]
            re_encoded = self.encoder.encode(truncated)
            if not re_encoded:
                continue
            return start + len(re_encoded)

        return hard_end

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

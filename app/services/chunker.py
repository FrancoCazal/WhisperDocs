"""Structure-aware, token-bounded chunker (audit R14, M-7).

The extractor (app/services/text_extractor.py) gives every format one shape:
`#` headings, markdown tables, and a PAGE_BREAK line between a PDF's pages.
The chunker reads that text as blocks (headings, paragraphs, tables, code
fences) and packs them into chunks of at most `chunk_size` tokens:

- A heading starts a chunk, so a chunk keeps to one section and its citation
  can name it. A chunk still under `min_section_tokens` (half a chunk) takes
  the next section in, so short sections do not become chunks too small to
  hold their context; the chunk then names the sections it holds.
- A chunk that does not start with its headings gets them in front, from the
  top level down ("# Guide", "## Install"): its embedding, its full-text
  lexemes and the model's context then carry the section it comes from. They
  count toward `chunk_size`, and take at most a fifth of it.
- A table is a unit. One longer than a chunk is cut between rows, and each
  part repeats the header and delimiter rows.
- A paragraph or code block longer than a chunk is cut at the strongest
  boundary near the end of the window (paragraph, line, sentence, word), and
  the next part repeats the last `overlap` tokens. Between blocks there is no
  overlap: a chunk ends where a paragraph or a section does.
- Cuts are made on the text: a part ends where the cut is and the next one
  starts there, and no cut splits a character. The token-based cut it
  replaces could drift a token past its separator.

`headings` reads `#` lines as headings, for Markdown and the formats the
extractor turns into it; in plain text and PDFs a line that starts with `#` is
text. `paged` gives each chunk the pages it spans, counting the page breaks
before it.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass, field

import tiktoken

# Separates the pages of a PDF in the extractor's text, on a line of its own.
PAGE_BREAK = "\f"

# Separators in priority order, for cutting a block longer than a chunk.
# Earlier = stronger semantic boundary.
_BOUNDARY_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ")

# Minimum fraction of the window that must be filled before we accept a
# boundary cut. Prevents pathologically short chunks when an early separator
# appears (e.g. "Title.\n\n" right after start).
_MIN_BOUNDARY_FILL = 0.5

# A chunk's parts are joined by a blank line, which is one token.
_JOINER = "\n\n"
_JOINER_TOKENS = 1
# The headings in front of a chunk take at most this share of it.
_MAX_LEAD_SHARE = 0.2
# A section label longer than this, in characters, is cut.
MAX_SECTION_CHARS = 300
SECTION_SEPARATOR = " › "

_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_DELIMITER_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    token_count: int
    # The headings the chunk sits under, "Guide › Install › Linux"; None
    # outside any heading, and for text read without headings.
    section: str | None = None
    # The first and last page the chunk's text comes from, from 1; None for
    # text read as not paged.
    page_start: int | None = None
    page_end: int | None = None


# A heading in effect: its level and its title.
_Heading = tuple[int, str]


@dataclass(frozen=True)
class _Line:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _Block:
    # "heading", "text", "code" or "table".
    kind: str
    lines: tuple[_Line, ...]
    # The headings in effect; a heading's own path ends with it.
    path: tuple[_Heading, ...]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def start(self) -> int:
        return self.lines[0].start

    @property
    def end(self) -> int:
        return self.lines[-1].end


@dataclass(frozen=True)
class _Part:
    text: str
    path: tuple[_Heading, ...]
    start: int
    end: int
    heading: bool


@dataclass
class _Draft:
    """A chunk being filled: the headings in front of it, and the tokens its
    parts may take."""

    prefix: str
    budget: int
    parts: list[_Part] = field(default_factory=list)
    tokens: int = 0

    @property
    def has_content(self) -> bool:
        return any(not part.heading for part in self.parts)


class TextChunker:
    """Token-bounded chunker that follows the text's sections and tables."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        encoding_name: str = "cl100k_base",
        min_section_tokens: int | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_section_tokens = (
            chunk_size // 2 if min_section_tokens is None else min_section_tokens
        )
        self.encoder = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def _encode(self, text: str) -> list[int]:
        # A document may hold "<|endoftext|>": it is text, not a control token.
        return self.encoder.encode(text, disallowed_special=())

    def chunk(self, text: str, *, headings: bool = False, paged: bool = False) -> list[Chunk]:
        if not text.strip():
            return []

        packer = _Packer(self)
        for block in self._blocks(text, headings):
            packer.add(block)
        packer.close()

        breaks = [m.start() for m in re.finditer(PAGE_BREAK, text)] if paged else []
        chunks: list[Chunk] = []
        for draft in packer.done:
            # Named after the parts with text: a heading opening the chunk with
            # nothing of its own under it ("# Doc" before "## A") is not its section.
            section = _label([part.path for part in draft.parts if not part.heading])
            pages = (
                (_page(breaks, draft.parts[0].start), _page(breaks, draft.parts[-1].end - 1))
                if paged
                else (None, None)
            )
            for piece in self._finish(draft):
                chunks.append(
                    Chunk(
                        text=piece,
                        index=len(chunks),
                        token_count=self.count_tokens(piece),
                        section=section,
                        page_start=pages[0],
                        page_end=pages[1],
                    )
                )
        return chunks

    # --- reading the text as blocks ----------------------------------------------

    def _blocks(self, text: str, headings: bool) -> list[_Block]:
        lines: list[_Line] = []
        pos = 0
        for raw in text.split("\n"):
            lines.append(_Line(raw, pos, pos + len(raw)))
            pos += len(raw) + 1

        blocks: list[_Block] = []
        path: tuple[_Heading, ...] = ()
        paragraph: list[_Line] = []

        def end_paragraph() -> None:
            if paragraph:
                blocks.append(_Block("text", tuple(paragraph), path))
                paragraph.clear()

        i, n = 0, len(lines)
        while i < n:
            line = lines[i]
            if not line.text.strip():
                end_paragraph()
                i += 1
                continue
            if fence := _FENCE_RE.match(line.text):
                end_paragraph()
                marker = fence.group(1)
                j = i + 1
                while j < n and not lines[j].text.strip().startswith(marker):
                    j += 1
                j = min(j, n - 1)
                blocks.append(_Block("code", tuple(lines[i : j + 1]), path))
                i = j + 1
                continue
            if headings and (match := _HEADING_RE.match(line.text)):
                end_paragraph()
                level = len(match.group(1))
                path = tuple(h for h in path if h[0] < level) + ((level, match.group(2).strip()),)
                blocks.append(_Block("heading", (line,), path))
                i += 1
                continue
            if _is_row(line.text) and i + 1 < n and _is_delimiter(lines[i + 1].text):
                end_paragraph()
                j = i + 2
                while j < n and _is_row(lines[j].text):
                    j += 1
                blocks.append(_Block("table", tuple(lines[i:j]), path))
                i = j
                continue
            paragraph.append(line)
            i += 1
        end_paragraph()
        return blocks

    # --- what goes in front of a chunk -------------------------------------------

    def lead(self, headings: tuple[_Heading, ...]) -> str:
        """The headings as `#` lines, from the top level down. Past a fifth
        of a chunk, the top levels are dropped; a single heading that long is
        cut."""
        if not headings:
            return ""
        cap = max(1, int(self.chunk_size * _MAX_LEAD_SHARE))
        kept: list[str] = []
        total = 0
        for level, title in reversed(headings):
            line = "#" * level + " " + title
            tokens = self.count_tokens(line) + 1
            if total + tokens > cap:
                if not kept:
                    kept.append(self._truncate(line, cap - 1))
                break
            kept.append(line)
            total += tokens
        return "\n".join(reversed(kept))

    def _truncate(self, text: str, tokens: int) -> str:
        return self.encoder.decode(self._encode(text)[: max(tokens, 1)]).rstrip("\ufffd")

    def _finish(self, draft: _Draft) -> list[str]:
        """The chunk's text: its headings, then its parts. The parts were
        counted one by one, so a chunk that comes out over the limit, which
        should not happen, is cut again."""
        body = _JOINER.join(part.text for part in draft.parts)
        text = f"{draft.prefix}{_JOINER}{body}" if draft.prefix else body
        if self.count_tokens(text) <= self.chunk_size:
            return [text]
        return [piece for piece, _start, _end in self.split_text(text, 0, self.chunk_size)]

    # --- cutting a block longer than a chunk -------------------------------------

    def split_table(self, block: _Block, budget: int) -> list[tuple[str, int, int]]:
        """Parts of a table, cut between rows, each with the header and
        delimiter rows. A table whose header takes half the budget is cut as
        text, and so is a row that does not fit with the header."""
        header = block.lines[0].text + "\n" + block.lines[1].text
        header_tokens = self.count_tokens(header)
        rows = block.lines[2:]
        if not rows or header_tokens * 2 > budget:
            return self.split_text(block.text, block.start, budget)

        parts: list[tuple[str, int, int]] = []
        group: list[_Line] = []
        size = header_tokens

        def end_group() -> None:
            nonlocal size
            start = block.start if not parts else group[0].start
            text = header + "\n" + "\n".join(row.text for row in group)
            parts.append((text, start, group[-1].end))
            group.clear()
            size = header_tokens

        for row in rows:
            tokens = self.count_tokens(row.text) + 1
            if header_tokens + tokens > budget:
                if group:
                    end_group()
                parts.extend(self.split_text(row.text, row.start, budget))
                continue
            if group and size + tokens > budget:
                end_group()
            group.append(row)
            size += tokens
        if group:
            end_group()
        return parts

    def split_text(self, text: str, base: int, budget: int) -> list[tuple[str, int, int]]:
        """Parts of `text` of at most `budget` tokens, with their offsets
        (`base` is the text's own). Each part after the first repeats the
        previous one's last `overlap` tokens, from a word start."""
        parts: list[tuple[str, int, int]] = []
        pos, n = 0, len(text)
        while pos < n:
            end = self._fit(text, pos, budget)
            piece = text[pos:end]
            if piece.strip():
                parts.append((piece.strip(), base + pos, base + end))
            if end >= n:
                break
            pos = self._next_start(text, pos, end)
        return parts

    def _fit(self, text: str, pos: int, budget: int) -> int:
        """Where the part starting at `pos` ends: the rest of the text if it
        fits, else the strongest boundary in the first `budget` tokens. Only
        a window of the text is encoded, so a long block is not encoded once
        per part."""
        n = len(text)
        width = budget * 8
        while True:
            window = text[pos : pos + width]
            tokens = self._encode(window)
            if len(tokens) > budget:
                break
            if pos + width >= n:
                return n
            width *= 2
        # A token may end inside a character, which decodes as U+FFFD.
        head = self.encoder.decode(tokens[:budget]).rstrip("\ufffd")
        if not head or not window.startswith(head):
            head = window[: max(1, len(window) * budget // len(tokens))]
        cut = self._snap(head)
        while cut > 1 and self.count_tokens(text[pos : pos + cut]) > budget:
            cut = self._snap(head[: cut * 9 // 10])
        return pos + max(cut, 1)

    @staticmethod
    def _snap(head: str) -> int:
        """The length of `head` up to its strongest boundary past the minimum
        fill; all of it when it has none."""
        min_chars = int(len(head) * _MIN_BOUNDARY_FILL)
        for sep in _BOUNDARY_SEPARATORS:
            cut = head.rfind(sep)
            if cut >= min_chars and cut > 0:
                return cut + len(sep)
        return len(head)

    def _next_start(self, text: str, pos: int, end: int) -> int:
        """Where the next part starts: `overlap` tokens before `end`, moved
        to a word start; `end` itself with no overlap, or when the part is
        not longer than it."""
        if self.overlap == 0:
            return end
        piece = text[pos:end]
        tokens = self._encode(piece)
        if len(tokens) <= self.overlap:
            return end
        tail = self.encoder.decode(tokens[-self.overlap :]).lstrip("\ufffd")
        if not tail or not piece.endswith(tail):
            return end
        start = end - len(tail)
        if (space := re.search(r"\s", tail)) is not None and space.end() < len(tail):
            start += space.end()
        return start if start > pos else end


class _Packer:
    """Fills chunks with blocks, in order."""

    def __init__(self, chunker: TextChunker):
        self.chunker = chunker
        self.done: list[_Draft] = []
        self.draft: _Draft | None = None

    def add(self, block: _Block) -> None:
        c = self.chunker
        tokens = c.count_tokens(block.text)
        draft = self.draft
        if (
            block.kind == "heading"
            and draft is not None
            and draft.has_content
            and draft.tokens >= c.min_section_tokens
        ):
            self.close()
            draft = None
        if draft is None:
            draft = self._open(block)
        if self._fits(draft, tokens):
            self._put(draft, block.text.strip(), block, block.start, block.end, tokens)
            return

        self.close()
        draft = self._open(block)
        if self._fits(draft, tokens):
            self._put(draft, block.text.strip(), block, block.start, block.end, tokens)
            return

        if block.kind == "table":
            pieces = c.split_table(block, draft.budget)
        else:
            pieces = c.split_text(block.text, block.start, draft.budget)
        for i, (text, start, end) in enumerate(pieces):
            if i:
                self.close()
                draft = self._open(block)
            self._put(draft, text, block, start, end, c.count_tokens(text), content=True)

    def close(self) -> None:
        """End the chunk being filled. Headings at its end, with nothing of
        their sections in it, are left for the next chunk, which puts them in
        front; a chunk of headings alone is dropped for the same reason."""
        draft, self.draft = self.draft, None
        if draft is None:
            return
        while draft.parts and draft.parts[-1].heading:
            draft.parts.pop()
        if draft.has_content:
            self.done.append(draft)

    def _open(self, block: _Block) -> _Draft:
        """A chunk that starts with `block`: in front, the headings above it,
        which for a heading are its ancestors."""
        lead = block.path[:-1] if block.kind == "heading" else block.path
        prefix = self.chunker.lead(lead)
        used = self.chunker.count_tokens(prefix) + _JOINER_TOKENS if prefix else 0
        self.draft = _Draft(prefix=prefix, budget=self.chunker.chunk_size - used)
        return self.draft

    @staticmethod
    def _fits(draft: _Draft, tokens: int) -> bool:
        joiner = _JOINER_TOKENS if draft.parts else 0
        return draft.tokens + joiner + tokens <= draft.budget

    @staticmethod
    def _put(
        draft: _Draft,
        text: str,
        block: _Block,
        start: int,
        end: int,
        tokens: int,
        *,
        content: bool | None = None,
    ) -> None:
        heading = block.kind == "heading" if content is None else not content
        draft.tokens += (_JOINER_TOKENS if draft.parts else 0) + tokens
        draft.parts.append(_Part(text, block.path, start, end, heading))


def _is_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def _is_delimiter(line: str) -> bool:
    return "|" in line and "-" in line and _DELIMITER_RE.match(line) is not None


def _page(breaks: list[int], offset: int) -> int:
    return 1 + bisect_left(breaks, offset)


def _label(paths: list[tuple[_Heading, ...]]) -> str | None:
    """The section a chunk's parts share. When every part sits deeper than
    that, in sibling sections, they are named after it, "Guide › Install,
    Upgrade"; when they share no section, it is the one the chunk starts in."""
    common = paths[0]
    for path in paths[1:]:
        n = 0
        while n < min(len(common), len(path)) and common[n] == path[n]:
            n += 1
        common = common[:n]
    titles = [title for _level, title in (common or paths[0])]
    if common and all(len(path) > len(common) for path in paths):
        siblings = list(dict.fromkeys(path[len(common)][1] for path in paths))
        if len(siblings) > 1:
            titles.append(", ".join(siblings))
    if not titles:
        return None
    label = SECTION_SEPARATOR.join(titles)
    return label if len(label) <= MAX_SECTION_CHARS else label[: MAX_SECTION_CHARS - 1] + "…"

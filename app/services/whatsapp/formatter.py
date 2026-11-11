"""WhatsApp message formatter.

Converts markdown LLM output → WhatsApp plain text.
Rules per phase-4-design §2.12 (applied in this order):
  1. Strip markdown — preserve code blocks and inline code
  2. Append the footer of the sources the answer cites
  3. Normalize whitespace
  4. Smart truncate to 4096 chars
  5. Guardrail: return fallback if result is blank

Markdown is stripped only where it is markdown (audit A16): a heading mark at
the start of a line ("C# " stays), and emphasis whose delimiters touch the
text they wrap and no letter outside it (`snake_case_name` and `2 * 3 * 4`
stay). The footer lists the sources the answer cites, each title once, in the
reply's language.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

MAX_CHARS = 4096
Language = Literal["en", "es"]

FALLBACK_MSG: dict[Language, str] = {
    "es": "Disculpá, tuve un problema procesando tu pregunta. ¿Podés reformularla?",
    "en": "Sorry, I had a problem processing your question. Could you rephrase it?",
}
_SOURCES = {"es": "Fuentes", "en": "Sources"}
_TRUNCATED = {"es": "truncado", "en": "truncated"}
_UNTITLED = {"es": "Fuente", "en": "Source"}

_CITATION_RE = re.compile(r"\[(\d+)\]")


def format_for_whatsapp(
    text: str, citations: list[dict] | None = None, language: Language = "es"
) -> str:
    """`language` is the reply's, the user's message's (app/services/lang.py):
    it picks the footer's label, the truncation mark and the fallback."""
    result = _strip_markdown_preserve_code(text)
    if citations:
        result = _append_citations(result, citations, language)
    result = _normalize_whitespace(result)
    result = _smart_truncate(result, language=language)

    if not result.strip():
        logger.error("whatsapp_formatter_empty_output original_text=%.200r", text)
        return FALLBACK_MSG[language]

    return result


# ---------- internal steps ----------

# Emphasis whose delimiters touch the text inside, with no word character just
# outside them, on one line: `_word_` and `*word*`, not `snake_case_name`,
# `2 * 3 * 4` or `a*b`.
_BOLD_STARS = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_BOLD_UNDERSCORES = re.compile(r"(?<![\w_])__(?=\S)(.+?)(?<=\S)__(?![\w_])")
_ITALIC_STAR = re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n]*?[^\s*])\*(?![\w*])")
_ITALIC_UNDERSCORE = re.compile(r"(?<![\w_])_(?=[^\s_])([^_\n]*?[^\s_])_(?![\w_])")
_IDENTIFIER = re.compile(r"\w+")


def _strip_markdown_preserve_code(text: str) -> str:
    # Protect fenced code blocks (```...```) — WhatsApp renders them as monospace.
    code_blocks: list[str] = []

    def _save_block(m: re.Match) -> str:  # type: ignore[type-arg]
        code_blocks.append(m.group(0))
        return f"\x00CODE{len(code_blocks) - 1}\x00"

    result = re.sub(r"```[\s\S]*?```", _save_block, text)

    # Protect inline code (`...`) — also WhatsApp-compatible.
    inline_codes: list[str] = []

    def _save_inline(m: re.Match) -> str:  # type: ignore[type-arg]
        inline_codes.append(m.group(0))
        return f"\x00INLINE{len(inline_codes) - 1}\x00"

    result = re.sub(r"`[^`\n]+`", _save_inline, result)

    # Strip markdown constructs
    result = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]+", "", result, flags=re.MULTILINE)  # # headings
    result = _BOLD_STARS.sub(r"\1", result)  # **bold**
    # __bold__, but not a dunder name such as __init__.
    result = _BOLD_UNDERSCORES.sub(
        lambda m: m.group(0) if _IDENTIFIER.fullmatch(m.group(1)) else m.group(1), result
    )
    result = _ITALIC_STAR.sub(r"\1", result)  # *italic*
    result = _ITALIC_UNDERSCORE.sub(r"\1", result)  # _italic_
    result = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", result)  # [text](url)
    result = re.sub(r"^\s*[-*+]\s+", "", result, flags=re.MULTILINE)  # - lists
    result = re.sub(r"^\s*\d+\.\s+", "", result, flags=re.MULTILINE)  # 1. lists
    result = re.sub(r"^>\s?", "", result, flags=re.MULTILINE)  # > blockquotes
    result = re.sub(r"^---+$", "", result, flags=re.MULTILINE)  # --- hr
    result = re.sub(r"^~~~+$", "", result, flags=re.MULTILINE)  # ~~~ hr

    # Restore placeholders
    for i, code in enumerate(inline_codes):
        result = result.replace(f"\x00INLINE{i}\x00", code)
    for i, block in enumerate(code_blocks):
        result = result.replace(f"\x00CODE{i}\x00", block)

    return result


def _append_citations(text: str, citations: list[dict], language: Language) -> str:
    """The sources the answer cites, by their [n], each title once with all
    its numbers: "[1][3] guide.md · [2] faq.md". The chunks the model was
    given and did not cite are left out, and an answer citing none (a
    refusal) gets no footer: listing them would read as its sources."""
    cited = sorted({int(n) for n in _CITATION_RE.findall(text) if 1 <= int(n) <= len(citations)})
    if not cited:
        return text
    numbers: dict[str, list[int]] = {}
    for n in cited:
        title = citations[n - 1].get("document_title") or f"{_UNTITLED[language]} {n}"
        numbers.setdefault(title, []).append(n)
    parts = ["".join(f"[{n}]" for n in ns) + f" {title}" for title, ns in numbers.items()]
    return f"{text}\n\n— {_SOURCES[language]}: " + " · ".join(parts)


def _smart_truncate(text: str, limit: int = MAX_CHARS, language: Language = "es") -> str:
    if len(text) <= limit:
        return text
    suffix = f" … [{_TRUNCATED[language]}]"
    cut_at = limit - len(suffix)
    truncated = text[:cut_at]
    # Prefer cutting at a sentence or paragraph boundary
    best = max(truncated.rfind(".\n"), truncated.rfind(". "), truncated.rfind("\n"))
    if best > limit // 2:
        truncated = truncated[: best + 1]
    return truncated.rstrip() + suffix


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

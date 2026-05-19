"""WhatsApp message formatter.

Converts markdown LLM output → WhatsApp plain text.
Rules per phase-4-design §2.12 (applied in this order):
  1. Strip markdown — preserve code blocks and inline code
  2. Append citations footer
  3. Normalize whitespace
  4. Smart truncate to 4096 chars
  5. Guardrail: return fallback if result is blank
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

MAX_CHARS = 4096
FALLBACK_MSG = "Disculpá, tuve un problema procesando tu pregunta. ¿Podés reformularla?"


def format_for_whatsapp(text: str, citations: list[dict] | None = None) -> str:
    result = _strip_markdown_preserve_code(text)
    if citations:
        result = _append_citations(result, citations)
    result = _normalize_whitespace(result)
    result = _smart_truncate(result)

    if not result.strip():
        logger.error("whatsapp_formatter_empty_output original_text=%.200r", text)
        return FALLBACK_MSG

    return result


# ---------- internal steps ----------


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
    result = re.sub(r"#{1,6}\s+", "", result)  # # headings
    result = re.sub(r"\*\*(.+?)\*\*", r"\1", result, flags=re.DOTALL)  # **bold**
    result = re.sub(r"\*(.+?)\*", r"\1", result, flags=re.DOTALL)  # *italic*
    result = re.sub(r"__(.+?)__", r"\1", result, flags=re.DOTALL)  # __bold__
    result = re.sub(r"_(.+?)_", r"\1", result, flags=re.DOTALL)  # _italic_
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


def _append_citations(text: str, citations: list[dict]) -> str:
    parts = [
        f"[{i + 1}] {c.get('document_title', f'Fuente {i + 1}')}" for i, c in enumerate(citations)
    ]
    footer = "\n\n— Fuentes: " + " · ".join(parts)
    return text + footer


def _smart_truncate(text: str, limit: int = MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    suffix = " … [truncado]"
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

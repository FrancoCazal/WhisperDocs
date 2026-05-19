import io
import re

import pdfplumber

SUPPORTED_TYPES = {"pdf", "md", "txt"}


class UnsupportedFileTypeError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


def detect_file_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_TYPES:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '.{ext}'. Supported: {sorted(SUPPORTED_TYPES)}"
        )
    return ext


def extract_text(file_bytes: bytes, file_type: str) -> str:
    if file_type == "pdf":
        text = _extract_pdf(file_bytes)
    elif file_type in {"md", "txt"}:
        text = file_bytes.decode("utf-8", errors="replace")
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {file_type}")

    text = _clean(text)
    if not text:
        raise EmptyDocumentError("Document has no extractable text content")
    return text


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Render a pdfplumber table as markdown. Skips empty tables."""
    rows = [[(cell or "").strip() for cell in row] for row in table if row]
    if not rows or not any(any(c for c in r) for r in rows):
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    md = "| " + " | ".join(header) + " |\n"
    md += "|" + "|".join(["---"] * width) + "|\n"
    for r in body:
        md += "| " + " | ".join(r) + " |\n"
    return md.strip()


def _extract_pdf(file_bytes: bytes) -> str:
    """Extract text + tables (rendered as markdown inline) from PDF."""
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            parts: list[str] = []
            page_text = page.extract_text() or ""
            if page_text:
                parts.append(page_text)
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for tbl in tables:
                md = _table_to_markdown(tbl)
                if md:
                    parts.append("\n" + md + "\n")
            pages.append("\n".join(parts))
    return "\n\n".join(pages)


_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)


def _clean(text: str) -> str:
    """Normalize whitespace while preserving structured blocks.

    Code fences (```...```) and contiguous markdown table line groups are
    masked out before whitespace collapse and restored verbatim afterwards.
    Without this, the previous implementation collapsed table column padding
    and stripped indentation inside code blocks — both of which carry
    information that downstream chunking and retrieval rely on.
    """
    placeholders: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"\x00PROTECTED_{len(placeholders) - 1}\x00"

    text = _FENCED_CODE_RE.sub(_stash, text)
    text = _mask_markdown_tables(text, placeholders)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    for idx, original in enumerate(placeholders):
        text = text.replace(f"\x00PROTECTED_{idx}\x00", original)
    return text


def _mask_markdown_tables(text: str, placeholders: list[str]) -> str:
    """Replace contiguous markdown table line runs (≥2 lines) with placeholders."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _MD_TABLE_LINE_RE.match(lines[i]):
            j = i
            while j < len(lines) and _MD_TABLE_LINE_RE.match(lines[j]):
                j += 1
            if j - i >= 2:
                placeholders.append("\n".join(lines[i:j]))
                out.append(f"\x00PROTECTED_{len(placeholders) - 1}\x00")
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)

"""Text from an uploaded file, in one shape for every format (audit M-7, R13).

What each format becomes:
- PDF: each page's text, with each of its tables once, as a markdown table
  where it sits on the page. A form feed on its own line separates the
  pages, so a chunk knows the pages it spans. A PDF with no text layer (a
  scan) is rejected: there is no OCR.
- Markdown and plain text: decoded as UTF-8, as UTF-16 or UTF-32 when the
  file starts with their byte order mark, and otherwise as what
  `charset-normalizer` finds. A Latin-1 file used to come out with
  replacement characters.
- DOCX: paragraphs and tables in document order, headings (the Title and
  Heading 1-6 styles) as `#` lines, list items as `- ` lines. Headers,
  footers, footnotes, comments and text boxes are left out.
- XLSX: each sheet as a `# sheet name` heading over a markdown table of its
  cells, the first non-empty row as the header. A formula gives the value the
  file last saved; one never calculated gives nothing.
- CSV: one markdown table, the first row as the header, the delimiter sniffed
  (comma, semicolon, tab or pipe).
- HTML: headings as `#` lines, paragraphs, list items, tables and `<pre>`
  blocks; scripts, styles and markup are dropped.

The chunker (app/services/chunker.py) reads `#` lines as headings only for
the types in HEADING_TYPES, and counts pages only for PAGED_TYPES.

DOCX and XLSX files are zip archives: one that would expand past
MAX_EXPANDED_BYTES, or holds more than MAX_ZIP_ENTRIES parts, is refused
before it is opened. Text longer than MAX_EXTRACTED_CHARS is refused too.
python-docx parses with entity resolution off, and openpyxl through
defusedxml.
"""

from __future__ import annotations

import codecs
import csv
import io
import re
import zipfile
from collections.abc import Callable, Sequence
from datetime import date, datetime, time
from html.parser import HTMLParser

import charset_normalizer
import docx
import openpyxl
import pdfplumber
from docx.table import Table as DocxTable

from app.services.chunker import PAGE_BREAK

SUPPORTED_TYPES = {"pdf", "md", "txt", "docx", "xlsx", "csv", "html"}
# Other extensions read as one of those.
_ALIASES = {"htm": "html"}
# Types whose extracted text marks its headings with `#` lines; in the others
# a line that starts with `#` is text.
HEADING_TYPES = frozenset({"md", "docx", "xlsx", "html"})
# Types whose extracted text separates its pages with PAGE_BREAK.
PAGED_TYPES = frozenset({"pdf"})

MB = 1024 * 1024
MAX_EXPANDED_BYTES = 100 * MB
MAX_ZIP_ENTRIES = 10_000
MAX_EXTRACTED_CHARS = 20_000_000


class UnsupportedFileTypeError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


class UnreadableDocumentError(ValueError):
    """The file has the right extension but its content cannot be parsed."""


# The PDF spec puts `%PDF-` first, but readers accept up to 1 KB of junk
# before it, and so does pdfminer.
_PDF_HEADER_WINDOW = 1024
_TEXT_SNIFF_BYTES = 8192
_CSV_SNIFF_CHARS = 64 * 1024
_ZIP_SIGNATURE = b"PK\x03\x04"
_TEXT_TYPES = {"md", "txt", "csv", "html"}
# UTF-32 first: its little-endian mark starts with UTF-16's.
_BOMS = (
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)
_WIDE_BOMS = tuple(bom for bom, encoding in _BOMS if encoding != "utf-8")

_NO_TEXT = "Document has no extractable text content"
_NO_TEXT_LAYER = (
    "The PDF has no text to extract: it may be a scan, and scanned PDFs are not read "
    "(there is no OCR)."
)


def check_content(file_bytes: bytes, file_type: str) -> None:
    """Reject content that does not match the extension, before any parsing.

    A `.pdf` must carry the `%PDF-` header, and a `.docx` or `.xlsx` must be a
    zip archive. A text format must not contain NUL bytes, which mark a binary
    file renamed to text (and `_clean` uses NUL for its placeholders), unless
    it starts with a UTF-16 or UTF-32 byte order mark.
    """
    if file_type == "pdf":
        if b"%PDF-" not in file_bytes[:_PDF_HEADER_WINDOW]:
            raise UnsupportedFileTypeError("File has a .pdf extension but is not a PDF")
    elif file_type in {"docx", "xlsx"}:
        if not file_bytes.startswith(_ZIP_SIGNATURE):
            raise UnsupportedFileTypeError(
                f"File has a .{file_type} extension but is not a {file_type.upper()} file"
            )
    elif file_type in _TEXT_TYPES:
        if b"\x00" in file_bytes[:_TEXT_SNIFF_BYTES] and not file_bytes.startswith(_WIDE_BOMS):
            raise UnsupportedFileTypeError(
                f"File has a .{file_type} extension but contains binary data"
            )


def detect_file_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    file_type = _ALIASES.get(ext, ext)
    if file_type not in SUPPORTED_TYPES:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '.{ext}'. Supported: {sorted(SUPPORTED_TYPES)}"
        )
    return file_type


def extract_text(file_bytes: bytes, file_type: str) -> str:
    extract = _EXTRACTORS.get(file_type)
    if extract is None:
        raise UnsupportedFileTypeError(f"Unsupported file type: {file_type}")
    text = _clean(extract(file_bytes))
    if len(text) > MAX_EXTRACTED_CHARS:
        raise UnreadableDocumentError(_too_much_text())
    if not text.strip():
        raise EmptyDocumentError(_NO_TEXT_LAYER if file_type == "pdf" else _NO_TEXT)
    return text


def _too_much_text() -> str:
    return (
        f"The document holds more than {MAX_EXTRACTED_CHARS:,} characters of text once "
        "extracted: it is not read."
    )


# --- text ------------------------------------------------------------------------


def _decode_text(file_bytes: bytes) -> str:
    """The file's text: by its byte order mark, as UTF-8, or as the encoding
    `charset-normalizer` finds most likely.

    On Spanish or French text the detector ranks Windows-1250 (Central
    European) and Windows-1252 (Western) the same, and returns the first,
    which reads "años" as "ańos". Of encodings it cannot tell apart,
    Windows-1252 is taken: it is the legacy encoding of English and Spanish
    text, and Latin-1's printable characters are its own."""
    for bom, encoding in _BOMS:
        if file_bytes.startswith(bom):
            return file_bytes[len(bom) :].decode(encoding, errors="replace")
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass
    matches = charset_normalizer.from_bytes(file_bytes)
    best = matches.best()
    if best is None:
        return file_bytes.decode("utf-8", errors="replace")
    tied = (m for m in matches if (m.chaos, m.coherence) == (best.chaos, best.coherence))
    return str(next((m for m in tied if m.encoding == "cp1252"), best))


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime) and value.time() == time(0):
        # A spreadsheet date is a datetime at midnight.
        text = value.date().isoformat()
    elif isinstance(value, (datetime, date, time)):
        text = value.isoformat()
    elif isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)
    return _one_line(text).replace("|", "\\|")


def markdown_table(rows: Sequence[Sequence[object]]) -> str:
    """A table as markdown: the first non-empty row as the header, then a
    delimiter row. Cells stay on one line and their pipes are escaped, so a
    row is a line; empty rows and trailing empty columns are dropped. An
    empty table is ""."""
    cells = [[_cell(value) for value in row] for row in rows]
    cells = [row for row in cells if any(row)]
    if not cells:
        return ""
    width = max(len(row) for row in cells)
    cells = [row + [""] * (width - len(row)) for row in cells]
    while width > 0 and not any(row[width - 1] for row in cells):
        width -= 1
    cells = [row[:width] for row in cells]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "|".join(["---"] * width) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in cells[1:]]
    return "\n".join(lines)


# --- PDF -------------------------------------------------------------------------


def _extract_pdf(file_bytes: bytes) -> str:
    """Each page's text, pages separated by PAGE_BREAK on its own line.

    pdfminer raises its own exception types on a damaged file, none of them a
    ValueError, so they would surface as a 500. They become
    UnreadableDocumentError.
    """
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [_pdf_page_text(page) for page in pdf.pages]
    except Exception as exc:
        raise UnreadableDocumentError(f"Could not read the PDF: {exc}") from exc
    return f"\n{PAGE_BREAK}\n".join(pages)


def _pdf_page_text(page) -> str:
    """The page's text with each table once, where it sits.

    `extract_text` already returns a table's cells as lines, so adding the
    table as markdown after them indexed it twice (audit R13). The text comes
    from the characters outside the tables' boxes, and each table goes in, as
    markdown, at the height of its top.
    """
    try:
        found = page.find_tables()
    except Exception:
        found = []
    tables = []
    for table in found:
        try:
            rendered = markdown_table(table.extract())
        except Exception:
            continue
        if rendered:
            tables.append((table.bbox, rendered))
    if not tables:
        return page.extract_text() or ""
    boxes = [bbox for bbox, _rendered in tables]

    def outside_tables(obj: dict) -> bool:
        if obj.get("object_type") != "char":
            return True
        x = (obj["x0"] + obj["x1"]) / 2
        y = (obj["top"] + obj["bottom"]) / 2
        return not any(x0 <= x <= x1 and top <= y <= bottom for x0, top, x1, bottom in boxes)

    placed = [
        (line["top"], line["text"]) for line in page.filter(outside_tables).extract_text_lines()
    ]
    placed += [(bbox[1], f"\n{rendered}\n") for bbox, rendered in tables]
    placed.sort(key=lambda item: item[0])
    return "\n".join(text for _top, text in placed)


# --- zip formats -----------------------------------------------------------------


def _check_archive(file_bytes: bytes, kind: str) -> None:
    """Refuse a zip that would expand past the limits before anything opens
    it. Each entry declares its size, and zipfile reads no more than that."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            entries = archive.infolist()
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError) as exc:
        raise UnreadableDocumentError(f"Could not read the {kind} file: it is damaged.") from exc
    if len(entries) > MAX_ZIP_ENTRIES or sum(e.file_size for e in entries) > MAX_EXPANDED_BYTES:
        raise UnreadableDocumentError(
            f"The {kind} file would expand to more than {MAX_EXPANDED_BYTES // MB} MB, or "
            f"holds more than {MAX_ZIP_ENTRIES:,} parts: it is not read."
        )


def _extract_docx(file_bytes: bytes) -> str:
    _check_archive(file_bytes, "DOCX")
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        blocks = [_docx_block(item) for item in document.iter_inner_content()]
    except Exception as exc:
        raise UnreadableDocumentError(
            "Could not read the DOCX file: it is damaged, or not a Word document."
        ) from exc
    return "\n\n".join(block for block in blocks if block)


def _docx_heading_level(style: str) -> int:
    name = style.strip().lower()
    if name == "title":
        return 1
    match = re.fullmatch(r"heading (\d)", name)
    return min(int(match.group(1)), 6) if match and match.group(1) != "0" else 0


def _docx_block(item: object) -> str:
    if isinstance(item, DocxTable):
        return markdown_table([[cell.text for cell in row.cells] for row in item.rows])
    text = getattr(item, "text", "").strip()
    if not text:
        return ""
    style = getattr(getattr(item, "style", None), "name", None) or ""
    if level := _docx_heading_level(style):
        return "#" * level + " " + _one_line(text)
    if style.strip().lower().startswith("list"):
        return "- " + text
    return text


def _extract_xlsx(file_bytes: bytes) -> str:
    _check_archive(file_bytes, "XLSX")
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    except Exception as exc:
        raise UnreadableDocumentError(
            "Could not read the XLSX file: it is damaged, or not an Excel workbook."
        ) from exc
    try:
        sheets: list[str] = []
        size = 0
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                rows.append(row)
                size += sum(len(str(value)) for value in row if value is not None)
                if size > MAX_EXTRACTED_CHARS:
                    raise UnreadableDocumentError(_too_much_text())
            if table := markdown_table(rows):
                sheets.append(f"# {_one_line(sheet.title)}\n\n{table}")
    except UnreadableDocumentError:
        raise
    except Exception as exc:
        raise UnreadableDocumentError(
            "Could not read the XLSX file: it is damaged, or not an Excel workbook."
        ) from exc
    finally:
        workbook.close()
    return "\n\n".join(sheets)


# --- CSV -------------------------------------------------------------------------


def _extract_csv(file_bytes: bytes) -> str:
    text = _decode_text(file_bytes)
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(
            text[:_CSV_SNIFF_CHARS], delimiters=",;\t|"
        )
    except csv.Error:
        dialect = csv.excel
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), dialect))
    except csv.Error as exc:
        raise UnreadableDocumentError(f"Could not read the CSV file: {exc}") from exc
    return markdown_table(rows)


# --- HTML ------------------------------------------------------------------------

_HTML_HEADINGS = {f"h{n}": n for n in range(1, 7)}
# Elements whose content is not text a reader sees.
_HTML_SKIP = frozenset(
    {"script", "style", "noscript", "template", "svg", "title", "iframe", "object", "canvas"}
)
# Elements that end the text before them and start their own.
_HTML_BLOCKS = frozenset(
    {
        "p", "div", "section", "article", "main", "header", "footer", "aside", "nav",
        "blockquote", "ul", "ol", "dl", "dt", "dd", "figure", "figcaption", "form",
        "fieldset", "address", "details", "summary", "hr", "br", "body", "html", "li",
    }
)  # fmt: skip


class _HtmlTable:
    """The rows of a table being read. HTML may leave a cell or a row open:
    the next one, or the end of the table, closes it."""

    def __init__(self) -> None:
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        # Tables inside this one: their text goes into the current cell.
        self.nested = 0

    def start_cell(self) -> None:
        self.end_cell()
        if self.row is None:
            self.row = []
        self.cell = []

    def end_cell(self) -> None:
        if self.cell is not None:
            if self.row is None:
                self.row = []
            self.row.append("".join(self.cell))
            self.cell = None

    def end_row(self) -> None:
        self.end_cell()
        if self.row is not None:
            self.rows.append(self.row)
            self.row = None


class _HtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._text: list[str] = []
        self._skip = 0
        self._heading = 0
        self._pre = 0
        self._list_item = False
        self._table: _HtmlTable | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _HTML_SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if self._table is not None:
            self._table_tag(tag, opening=True)
            return
        if tag == "table":
            self._flush()
            self._table = _HtmlTable()
        elif tag in _HTML_HEADINGS:
            self._flush()
            self._heading = _HTML_HEADINGS[tag]
        elif tag == "pre":
            if not self._pre:
                self._flush()
            self._pre += 1
        elif tag in _HTML_BLOCKS and not (self._heading or self._pre):
            self._flush()
            if tag == "li":
                self._list_item = True

    def handle_endtag(self, tag: str) -> None:
        if tag in _HTML_SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if self._table is not None:
            self._table_tag(tag, opening=False)
        elif tag in _HTML_HEADINGS and self._heading:
            if text := self._take():
                self.blocks.append("#" * self._heading + " " + text)
            self._heading = 0
        elif tag == "pre" and self._pre:
            self._pre -= 1
            if not self._pre:
                self._end_pre()
        elif tag in _HTML_BLOCKS and not (self._heading or self._pre):
            self._flush()
            if tag in ("li", "ul", "ol"):
                self._list_item = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._table is not None:
            if self._table.cell is not None:
                self._table.cell.append(data)
            return
        self._text.append(data)

    def finish(self) -> None:
        self.close()
        if self._table is not None:
            self._end_table()
        if self._pre:
            self._end_pre()
        self._flush()

    def _table_tag(self, tag: str, *, opening: bool) -> None:
        table = self._table
        assert table is not None
        if tag == "table":
            if opening:
                table.nested += 1
            elif table.nested:
                table.nested -= 1
            else:
                self._end_table()
            return
        if table.nested:
            if table.cell is not None:
                table.cell.append(" ")
            return
        if opening and tag in ("td", "th"):
            table.start_cell()
        elif opening and tag in ("tr", "thead", "tbody", "tfoot"):
            table.end_row()
        elif not opening and tag in ("td", "th"):
            table.end_cell()
        elif not opening and tag == "tr":
            table.end_row()
        elif table.cell is not None:
            table.cell.append(" ")

    def _end_table(self) -> None:
        assert self._table is not None
        self._table.end_row()
        if rendered := markdown_table(self._table.rows):
            self.blocks.append(rendered)
        self._table = None

    def _end_pre(self) -> None:
        self._pre = 0
        raw = "".join(self._text).strip("\n")
        self._text = []
        if raw.strip():
            self.blocks.append(f"```\n{raw}\n```")

    def _take(self) -> str:
        text = _one_line("".join(self._text))
        self._text = []
        return text

    def _flush(self) -> None:
        """The text so far as a block. An open list item keeps its mark until
        its text comes: `<li><p>text</p></li>` is one item."""
        if text := self._take():
            self.blocks.append(("- " if self._list_item else "") + text)
            self._list_item = False


def _extract_html(file_bytes: bytes) -> str:
    parser = _HtmlText()
    try:
        parser.feed(_decode_text(file_bytes))
        parser.finish()
    except Exception as exc:
        raise UnreadableDocumentError("Could not read the HTML file.") from exc
    return "\n\n".join(parser.blocks)


_EXTRACTORS: dict[str, Callable[[bytes], str]] = {
    "pdf": _extract_pdf,
    "md": _decode_text,
    "txt": _decode_text,
    "docx": _extract_docx,
    "xlsx": _extract_xlsx,
    "csv": _extract_csv,
    "html": _extract_html,
}


# --- cleaning --------------------------------------------------------------------

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)


def _clean(text: str) -> str:
    """Normalize whitespace while preserving structured blocks.

    Line endings become `\\n`. Code fences (```...```) and contiguous markdown
    table line groups are masked out before whitespace collapse and restored
    verbatim afterwards. Without this, the previous implementation collapsed
    table column padding and stripped indentation inside code blocks — both of
    which carry information that downstream chunking and retrieval rely on.
    Form feeds stay, including leading ones: they count a PDF's pages.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    placeholders: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"\x00PROTECTED_{len(placeholders) - 1}\x00"

    text = _FENCED_CODE_RE.sub(_stash, text)
    text = _mask_markdown_tables(text, placeholders)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Not `strip()`, which would take a PDF's leading form feeds, and with
    # them the numbers of the pages after its empty first ones.
    text = text.strip(" \t\n")

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

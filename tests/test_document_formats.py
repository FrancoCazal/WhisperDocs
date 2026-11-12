"""What each format becomes, and where its chunks sit (audit M-7, R13).

The PDFs are written by hand below (text, lines and pages are a few content
stream operators), so no PDF library is needed to build them. DOCX and XLSX
files are written with python-docx and openpyxl, the libraries that read them.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date

import docx
import openpyxl
import pytest

from app.models import Chunk, Document
from app.services import text_extractor
from app.services.chunker import TextChunker
from app.services.text_extractor import (
    EmptyDocumentError,
    UnreadableDocumentError,
    UnsupportedFileTypeError,
    _clean,
    check_content,
    detect_file_type,
    extract_text,
)

# --- a PDF by hand ----------------------------------------------------------------


def _pdf(pages: list[list[tuple]]) -> bytes:
    """A PDF whose pages draw `("text", x, y, string)` and
    `("line", x0, y0, x1, y1)`, in points from the bottom left."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")
    pages_ref = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    kids = []
    for ops in pages:
        stream = b""
        for op in ops:
            if op[0] == "text":
                _, x, y, s = op
                escaped = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                stream += f"BT /F1 11 Tf {x} {y} Td ({escaped}) Tj ET\n".encode("latin-1")
            else:
                _, x0, y0, x1, y1 = op
                stream += f"{x0} {y0} m {x1} {y1} l S\n".encode()
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream")
        kids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                % (pages_ref, font, content)
            )
        )
    objects[catalog - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_ref
    objects[pages_ref - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids),
        len(kids),
    )
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        catalog,
        xref,
    )
    return out


def _grid(xs: list[int], ys: list[int]) -> list[tuple]:
    return [("line", xs[0], y, xs[-1], y) for y in ys] + [("line", x, ys[-1], x, ys[0]) for x in xs]


TABLE_PAGE = [
    ("text", 72, 740, "Prices as of November."),
    *_grid([100, 250, 400], [700, 680, 660, 640]),
    ("text", 105, 686, "Model"),
    ("text", 255, 686, "Price"),
    ("text", 105, 666, "mini"),
    ("text", 255, 666, "0.15"),
    ("text", 105, 646, "haiku"),
    ("text", 255, 646, "1.00"),
    ("text", 72, 600, "End of the page."),
]


def test_a_pdf_table_is_extracted_once_where_it_sits():
    """`extract_text` returned the cells as lines and the table was added
    after them as markdown: its text was indexed twice (audit R13)."""
    text = extract_text(_pdf([TABLE_PAGE]), "pdf")

    assert text == (
        "Prices as of November.\n\n"
        "| Model | Price |\n|---|---|\n| mini | 0.15 |\n| haiku | 1.00 |\n\n"
        "End of the page."
    )


def test_a_pdfs_pages_are_separated_and_its_chunks_know_them():
    pages = [[("text", 72, 700, f"Page {n} says something about topic {n}.")] for n in (1, 2, 3)]
    text = extract_text(_pdf(pages), "pdf")

    assert text.split("\n\f\n") == [f"Page {n} says something about topic {n}." for n in (1, 2, 3)]
    chunks = TextChunker(chunk_size=20, overlap=0).chunk(text, paged=True)
    assert [(c.text, c.page_start, c.page_end) for c in chunks] == [
        (f"Page {n} says something about topic {n}.", n, n) for n in (1, 2, 3)
    ]
    # Together they fit in one chunk, which spans the three pages.
    (whole,) = TextChunker().chunk(text, paged=True)
    assert (whole.page_start, whole.page_end) == (1, 3)


def test_a_pdf_whose_first_page_is_blank_still_numbers_the_rest():
    text = extract_text(_pdf([[], [("text", 72, 700, "Second page.")]]), "pdf")

    (chunk,) = TextChunker().chunk(text, paged=True)
    assert (chunk.page_start, chunk.page_end) == (2, 2)


def test_a_pdf_without_text_says_it_may_be_a_scan():
    with pytest.raises(EmptyDocumentError, match="no OCR"):
        extract_text(_pdf([_grid([100, 200], [700, 600])]), "pdf")


def test_a_damaged_pdf_is_unreadable():
    with pytest.raises(UnreadableDocumentError, match="Could not read the PDF"):
        extract_text(b"%PDF-1.4\nnot really", "pdf")


# --- text and encodings -----------------------------------------------------------

SPANISH = (
    "La política de retención guarda las facturas durante cinco años desde el cierre del "
    "ejercicio. Después, el área de administración las destruye según el procedimiento "
    "vigente, con acta firmada por el responsable y por un testigo del área técnica. "
)


def test_a_latin1_text_file_is_read_as_latin1():
    """It used to be decoded as UTF-8 with replacement characters (audit R13)."""
    text = extract_text((SPANISH * 3).encode("latin-1"), "txt")

    assert "política de retención" in text and "años" in text
    assert "\ufffd" not in text


@pytest.mark.parametrize(
    ("text", "encoding"),
    [
        # The detector ties Windows-1250 with Windows-1252 on these, and
        # Windows-1250 reads ñ as ń and é as ę.
        ("Año y señal, canción del niño.", "latin-1"),
        ("L'été dernier, nous étions à la plage près de la forêt. Ça coûte cher. " * 3, "cp1252"),
        # Encodings the detector does tell apart stay theirs.
        ("Политика хранения требует хранить счета пять лет после окончания года. " * 3, "cp1251"),
        ("Účetní doklady se uchovávají pět let od konce účetního období. Kůň. " * 3, "cp1250"),
    ],
)
def test_legacy_encodings_are_read_as_what_they_are(text, encoding):
    assert extract_text(text.encode(encoding), "txt") == text.strip()


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
def test_a_byte_order_mark_picks_the_encoding(encoding):
    data = "Año y señal".encode(encoding)

    check_content(data, "txt")
    assert extract_text(data, "txt") == "Año y señal"


def test_line_endings_become_newlines():
    assert extract_text(b"one\r\ntwo\rthree", "txt") == "one\ntwo\nthree"


def test_clean_keeps_leading_page_breaks():
    assert _clean("\f\n\f\nThird page.  ") == "\f\n\f\nThird page."


def test_text_longer_than_the_limit_is_refused(monkeypatch):
    monkeypatch.setattr(text_extractor, "MAX_EXTRACTED_CHARS", 10)
    with pytest.raises(UnreadableDocumentError, match="more than 10 characters"):
        extract_text(b"x" * 20, "txt")


# --- DOCX -------------------------------------------------------------------------


def _docx() -> bytes:
    document = docx.Document()
    document.add_heading("Install guide", level=1)
    document.add_paragraph("Run the installer, then restart.")
    document.add_paragraph("Download it", style="List Bullet")
    table = document.add_table(rows=2, cols=2)
    for (row, col), value in {(0, 0): "Key", (0, 1): "Value", (1, 0): "a|b", (1, 1): "1"}.items():
        table.cell(row, col).text = value
    document.add_heading("Linux", level=2)
    document.add_paragraph("Use the package | or the script.")
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


def test_a_docx_keeps_its_headings_lists_and_tables_in_order():
    text = extract_text(_docx(), "docx")

    assert text == (
        "# Install guide\n\n"
        "Run the installer, then restart.\n\n"
        "- Download it\n\n"
        "| Key | Value |\n|---|---|\n| a\\|b | 1 |\n\n"
        "## Linux\n\n"
        "Use the package | or the script."
    )
    # One chunk: its text sits under "Install guide", some of it in "Linux".
    chunks = TextChunker().chunk(text, headings=True)
    assert [c.section for c in chunks] == ["Install guide"]


def test_a_docx_chunk_names_its_section():
    chunker = TextChunker(chunk_size=40, overlap=0, min_section_tokens=1)
    chunks = chunker.chunk(extract_text(_docx(), "docx"), headings=True)

    assert chunks[0].section == "Install guide"
    assert chunks[-1].section == "Install guide › Linux"
    assert chunks[-1].text.startswith("# Install guide\n\n## Linux")


def test_a_file_named_docx_that_is_not_a_zip_is_refused():
    with pytest.raises(UnsupportedFileTypeError, match="not a DOCX"):
        check_content(b"plain text", "docx")


def test_a_zip_that_is_not_a_word_document_is_unreadable():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("hello.txt", "hi")
    with pytest.raises(UnreadableDocumentError, match="not a Word document"):
        extract_text(out.getvalue(), "docx")


def test_an_archive_that_would_expand_past_the_limit_is_not_opened(monkeypatch):
    monkeypatch.setattr(text_extractor, "MAX_EXPANDED_BYTES", 1000)
    with pytest.raises(UnreadableDocumentError, match="would expand"):
        extract_text(_docx(), "docx")


# --- XLSX -------------------------------------------------------------------------


def _xlsx() -> bytes:
    workbook = openpyxl.Workbook()
    prices = workbook.active
    prices.title = "Prices"
    prices.append(["Model", "Price", None])
    prices.append(["mini", 0.15])
    prices.append(["haiku", 1.0])
    prices.append([date(2026, 11, 10), "a|b"])
    workbook.create_sheet("Empty")
    notes = workbook.create_sheet("Notes")
    notes["A1"] = "Note"
    notes["A2"] = "Re-run the eval after billing."
    out = io.BytesIO()
    workbook.save(out)
    return out.getvalue()


def test_each_sheet_is_a_section_with_its_table():
    text = extract_text(_xlsx(), "xlsx")

    assert text == (
        "# Prices\n\n| Model | Price |\n|---|---|\n| mini | 0.15 |\n| haiku | 1 |\n"
        "| 2026-11-10 | a\\|b |\n\n"
        "# Notes\n\n| Note |\n|---|\n| Re-run the eval after billing. |"
    )
    chunks = TextChunker(chunk_size=60, overlap=0, min_section_tokens=1).chunk(text, headings=True)
    assert [c.section for c in chunks] == ["Prices", "Notes"]


def test_a_damaged_xlsx_is_unreadable():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("[Content_Types].xml", "<not xml")
    with pytest.raises(UnreadableDocumentError, match="not an Excel workbook"):
        extract_text(out.getvalue(), "xlsx")


# --- CSV --------------------------------------------------------------------------


def test_a_csv_is_one_table_with_its_delimiter_sniffed():
    data = 'model;price;note\nmini;0,15;"cheap | small"\nhaiku;1,00;"line one\nline two"\n'

    assert extract_text(data.encode(), "csv") == (
        "| model | price | note |\n|---|---|---|\n"
        "| mini | 0,15 | cheap \\| small |\n| haiku | 1,00 | line one line two |"
    )


def test_a_long_csv_repeats_its_header_in_every_chunk():
    rows = "\n".join(f"item-{i},{i}" for i in range(200))
    text = extract_text(f"name,value\n{rows}".encode(), "csv")

    chunks = TextChunker(chunk_size=100, overlap=0).chunk(text)

    assert len(chunks) > 3
    assert all(c.text.startswith("| name | value |\n|---|---|\n") for c in chunks)


def test_a_csv_with_binary_data_is_refused():
    with pytest.raises(UnsupportedFileTypeError, match="binary data"):
        check_content(b"a,b\x00,c", "csv")


# --- HTML -------------------------------------------------------------------------

PAGE = """<!doctype html><html><head><title>Not text</title>
<style>p { color: red }</style><script>alert("no")</script></head>
<body><nav><a href="/">Home</a></nav>
<h1>Guide</h1><p>Intro &amp; more,
  across lines.</p>
<ul><li>One</li><li><p>Two</p></li></ul>
<h2>Prices</h2>
<table><tr><th>Model<th>Price<tr><td>mini<td>0.15</table>
<pre>  indented
code</pre>
<p>End.</p></body></html>"""


def test_html_keeps_headings_lists_tables_and_pre_and_drops_the_rest():
    text = extract_text(PAGE.encode(), "html")

    assert text == (
        "Home\n\n# Guide\n\nIntro & more, across lines.\n\n- One\n\n- Two\n\n## Prices\n\n"
        "| Model | Price |\n|---|---|\n| mini | 0.15 |\n\n"
        "```\n  indented\ncode\n```\n\nEnd."
    )
    assert "alert" not in text and "color" not in text and "Not text" not in text


def test_a_table_inside_a_cell_stays_in_that_cell():
    html = "<table><tr><td>a<table><tr><td>inner</td></tr></table></td><td>b</td></tr></table>"
    assert extract_text(html.encode(), "html") == "| a inner | b |\n|---|---|"


def test_htm_is_html():
    assert detect_file_type("page.HTM") == "html"


@pytest.mark.parametrize("name", ["sheet.xls", "doc.doc", "slides.pptx", "noext"])
def test_other_types_are_unsupported(name):
    with pytest.raises(UnsupportedFileTypeError):
        detect_file_type(name)


# --- through the upload -----------------------------------------------------------


class _Result:
    def all(self):
        return []


class _Session:
    def __init__(self) -> None:
        self.added: list = []

    async def execute(self, *_a, **_k):
        return _Result()

    def add(self, obj) -> None:
        self.added.append(obj)

    def add_all(self, objs) -> None:
        self.added.extend(objs)

    async def flush(self) -> None:
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = 1

    async def commit(self) -> None:
        pass

    async def refresh(self, _obj) -> None:
        pass


class _Embedder:
    model_id = "fake-embedding"

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1536 for _ in texts]


@pytest.mark.parametrize(
    ("filename", "content", "file_type", "location"),
    [
        ("guide.docx", _docx(), "docx", ("Install guide", None, None)),
        ("prices.pdf", _pdf([TABLE_PAGE]), "pdf", (None, 1, 1)),
    ],
)
async def test_an_upload_stores_where_each_chunk_sits(filename, content, file_type, location):
    from app.api.v1.documents import _ingest_single

    class _Upload:
        async def read(self, _size: int = -1) -> bytes:
            return content

    _Upload.filename = filename
    session = _Session()

    document = await _ingest_single(_Upload(), 1, None, session, _Embedder())

    assert isinstance(document, Document) and document.file_type == file_type
    rows = [obj for obj in session.added if isinstance(obj, Chunk)]
    assert rows and (rows[0].section, rows[0].page_start, rows[0].page_end) == location

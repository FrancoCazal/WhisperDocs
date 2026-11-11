"""The MCP connector (audit M-6, app/connectors/mcp_sync.py) without a
database: which resources a run reads, and how each one's bytes and format
come out of the server. The syncing itself is in test_mcp_sync_db.py.
"""

from __future__ import annotations

import pytest
from mcp import Client
from mcp.server import MCPServer

from app.connectors import mcp_sync
from app.connectors.mcp_sync import SkipError, Source

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _reader(files: dict[str, tuple[str | bytes, str | None]], uri: str):
    def read() -> str | bytes:
        return files[uri][0]

    return read


def _server(files: dict[str, tuple[str | bytes, str | None]], *, template: bool = False):
    """A server whose resources are `files`: uri -> (content, MIME type). A
    str is text, bytes a blob, read at each call. With `template`, an
    unlisted file:///notes/{name}."""
    server = MCPServer(name="files")
    for uri, (_content, mime) in files.items():
        server.resource(uri, name=uri.rsplit("/", 1)[-1], mime_type=mime)(_reader(files, uri))
    if template:

        @server.resource("file:///notes/{name}", mime_type="text/plain")
        def note(name: str) -> str:
            return f"the note called {name}"

    return server


# ---------- the format ----------


@pytest.mark.parametrize(
    ("filename", "mime", "is_text", "expected"),
    [
        ("README.md", "text/markdown", True, "md"),
        # The filesystem server calls any unknown text text/plain.
        ("prices.csv", "text/plain", True, "csv"),
        ("page", "text/html; charset=utf-8", True, "html"),
        # WhisperDocs' own resources: a PDF's text, named after the PDF.
        ("report.pdf", "text/plain", True, "txt"),
        ("notes", None, True, "txt"),
        ("report", PDF_MIME, False, "pdf"),
        ("contract.bin", DOCX_MIME, False, "docx"),
        ("sheet.xlsx", "application/octet-stream", False, "xlsx"),
    ],
)
def test_a_resource_is_read_as_its_extension_or_its_mime_type_says(
    filename, mime, is_text, expected
):
    assert mcp_sync.file_type_of(filename, mime, is_text) == expected


def test_a_blob_of_a_format_it_does_not_read_is_skipped():
    with pytest.raises(SkipError, match="image/png: not a format WhisperDocs reads"):
        mcp_sync.file_type_of("photo.png", "image/png", is_text=False)


@pytest.mark.parametrize(
    ("source", "filename"),
    [
        (Source("file:///C:/docs/My%20Notes.md"), "My Notes.md"),
        (Source("whisperdocs://documents/7", name="rag.md"), "rag.md"),
        (Source("repo://acme/site/contents/"), "contents"),
        (Source("mem://"), "resource"),
        (Source("file:///d/" + "x" * 400 + ".md"), "x" * 297 + ".md"),
    ],
)
def test_the_filename_comes_from_the_uri_or_the_resources_name(source, filename):
    assert mcp_sync._filename(source) == filename


# ---------- what a run reads ----------


async def test_a_run_reads_what_the_server_lists_within_the_prefix():
    files = {
        "file:///docs/a.md": ("# A", "text/markdown"),
        "file:///docs/b.txt": ("b", "text/plain"),
        "file:///other/c.md": ("# C", "text/markdown"),
    }

    async with Client(_server(files)) as client:
        everything = await mcp_sync.discover(client)
        docs = await mcp_sync.discover(client, prefix="file:///docs/")

    assert {s.uri for s in everything} == set(files)
    assert sorted(s.uri for s in docs) == ["file:///docs/a.md", "file:///docs/b.txt"]
    (a,) = [s for s in docs if s.uri.endswith("a.md")]
    assert (a.name, a.mime_type) == ("a.md", "text/markdown")


async def test_given_uris_a_run_reads_those_only_once_each():
    files = {"file:///docs/a.md": ("# A", "text/markdown")}

    async with Client(_server(files, template=True)) as client:
        sources = await mcp_sync.discover(
            client, uris=["file:///notes/x", "file:///notes/x", "file:///notes/y"]
        )
        read = await mcp_sync.read_source(client, sources[0])

    assert [s.uri for s in sources] == ["file:///notes/x", "file:///notes/y"]
    assert read == ("x", "txt", b"the note called x")


async def test_text_and_blobs_come_out_as_bytes():
    files = {
        "file:///docs/a.md": ("# Título\n\nAño", "text/markdown"),
        "file:///docs/r.pdf": (b"%PDF-1.7 bytes", PDF_MIME),
        "file:///docs/p.png": (b"\x89PNG", "image/png"),
    }

    async with Client(_server(files)) as client:
        text = await mcp_sync.read_source(client, Source("file:///docs/a.md"))
        blob = await mcp_sync.read_source(client, Source("file:///docs/r.pdf"))
        with pytest.raises(SkipError):
            await mcp_sync.read_source(client, Source("file:///docs/p.png"))

    assert text == ("a.md", "md", "# Título\n\nAño".encode())
    assert blob == ("r.pdf", "pdf", b"%PDF-1.7 bytes")


# ---------- the command ----------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--collection", "3"], "either --url or the server's command"),
        (["--collection", "3", "--url", "http://h/mcp", "--", "npx"], "either --url"),
        (["--collection", "3", "--bearer-env", "T", "--", "npx"], "--bearer-env goes with --url"),
    ],
)
def test_the_command_needs_one_server(argv, message, capsys):
    with pytest.raises(SystemExit):
        mcp_sync.main(argv)

    assert message in capsys.readouterr().err


def test_a_uri_file_lists_one_uri_a_line(tmp_path):
    listed = tmp_path / "uris.txt"
    listed.write_text("file:///a.md\n\n# a comment\n  file:///b.md  \n", encoding="utf-8")
    args = mcp_sync._parser().parse_args(
        ["--collection", "3", "--uri", "file:///c.md", "--uri-file", str(listed), "--", "srv"]
    )

    assert mcp_sync._uris(args) == ["file:///c.md", "file:///a.md", "file:///b.md"]


def test_each_resource_is_one_line():
    added = mcp_sync.Item("file:///a.md", "added", "3 chunks", 12)
    skipped = mcp_sync.Item("file:///p.png", "skipped", "image/png: not a format")

    assert mcp_sync._line(added, dry_run=False) == (
        "added              file:///a.md -> document 12 (3 chunks)"
    )
    assert mcp_sync._line(added, dry_run=True).startswith("would be added     file:///a.md")
    assert mcp_sync._line(skipped, dry_run=True).startswith("skipped            file:///p.png")

"""`python -m app.connectors.mcp_sync`: documents from an MCP server's
resources (audit M-6).

WhisperDocs as an MCP client, for ingestion only: it lists the resources of a
server, reads each one and ingests it into a collection, as an upload is. No
model reads or picks anything, so a run does the same thing every time.

    # A server started over stdio: the command goes after --.
    python -m app.connectors.mcp_sync --collection 3 -- npx -y @scope/server ARGS
    # A server over Streamable HTTP, its token in an environment variable.
    python -m app.connectors.mcp_sync --collection 3 --url https://host/mcp --bearer-env TOKEN

**Which resources.** What `resources/list` names, or with `--prefix` the
ones whose URI starts with it. With `--uri` (repeatable) or `--uri-file` (one
URI per line), those resources only, read by URI. Some servers list nothing:
the reference filesystem server reads any file under its directories by its
`file://` URI and never enumerates the disk. For it, pass the URIs.

**Again.** Each document keeps the URI it came from (`documents.source_uri`).
A later run finds it: the same text is left alone and nothing is embedded; a
changed text replaces it in one commit, the chunks that did not change keeping
their vectors (audit R12). A resource whose text the collection already holds
under another document is skipped, naming it. Nothing is deleted when a
resource leaves the server. `--dry-run` reads and compares, and embeds and
writes nothing.

Not a tool of the agent: a chat turn that consults outside sources is
multi-hop (V2), and lets a tool's output put instructions in the prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import os
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote, urlparse

from mcp import Client, MCPError, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import BlobResourceContents, TextResourceContents
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.logging_config import get_logger
from app.models import Collection, Document
from app.services.embeddings import EmbeddingError, EmbeddingProvider
from app.services.ingestion import (
    MAX_DOCUMENT_BYTES,
    MAX_NAME_LENGTH,
    DuplicateDocumentError,
    check_document,
    ensure_new_document,
    extract_document,
    store_document,
)
from app.services.llm.errors import failure_text
from app.services.text_extractor import UnsupportedFileTypeError, detect_file_type

logger = get_logger(__name__)

Sessions = Callable[[], AbstractAsyncContextManager[AsyncSession]]
Outcome = Literal["added", "updated", "unchanged", "skipped", "failed"]

# `documents.source_uri` is String(2048).
MAX_URI_LENGTH = 2048
# A server that lists more is asked for a --prefix.
MAX_RESOURCES = 5000

_MIME_TYPES = {
    "text/markdown": "md",
    "text/x-markdown": "md",
    "text/plain": "txt",
    "text/csv": "csv",
    "text/html": "html",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
}
_TEXT_TYPES = frozenset({"md", "txt", "csv", "html"})


class SkipError(Exception):
    """A resource that is not a document WhisperDocs takes, and why."""


class SyncError(Exception):
    """A run that cannot start: no such collection, nothing to read."""


@dataclass(frozen=True)
class Source:
    uri: str
    name: str | None = None
    title: str | None = None
    mime_type: str | None = None
    size: int | None = None


@dataclass(frozen=True)
class Item:
    uri: str
    outcome: Outcome
    detail: str = ""
    document_id: int | None = None


@dataclass
class SyncReport:
    dry_run: bool = False
    items: list[Item] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return dict(Counter(item.outcome for item in self.items))


# --- what to read ----------------------------------------------------------------------


async def discover(
    client: Client, *, prefix: str | None = None, uris: Sequence[str] = ()
) -> list[Source]:
    """The resources a run reads: `uris` when given, else what the server
    lists (starting with `prefix`, when given), each once."""
    if uris:
        return [Source(uri) for uri in dict.fromkeys(uris)]
    sources: dict[str, Source] = {}
    cursor: str | None = None
    while True:
        page = await client.list_resources(cursor=cursor)
        for resource in page.resources:
            uri = str(resource.uri)
            if prefix and not uri.startswith(prefix):
                continue
            sources.setdefault(
                uri,
                Source(uri, resource.name, resource.title, resource.mime_type, resource.size),
            )
            if len(sources) > MAX_RESOURCES:
                raise SyncError(
                    f"The server lists more than {MAX_RESOURCES} resources: narrow it with "
                    "--prefix."
                )
        cursor = page.next_cursor
        if not cursor:
            return list(sources.values())


def _filename(source: Source) -> str:
    """The last segment of the URI's path when it has an extension, else the
    resource's name (WhisperDocs' own `whisperdocs://documents/7` is named
    after its file); within MAX_NAME_LENGTH, keeping the extension."""
    segment = unquote(PurePosixPath(urlparse(source.uri).path).name)
    name = segment if "." in segment else (source.name or segment or "resource")
    if len(name) <= MAX_NAME_LENGTH:
        return name
    stem, dot, extension = name.rpartition(".")
    if not dot or len(extension) > 10:
        return name[:MAX_NAME_LENGTH]
    return stem[: MAX_NAME_LENGTH - len(extension) - 1] + "." + extension


def _extension_type(filename: str) -> str | None:
    try:
        return detect_file_type(filename)
    except UnsupportedFileTypeError:
        return None


def file_type_of(filename: str, mime_type: str | None, is_text: bool) -> str:
    """The format a resource is read as. Text content is text whatever its
    name says (WhisperDocs' own resources are a PDF's extracted text, named
    after the PDF): a text extension picks the format, then the MIME type
    (the filesystem server calls any unknown text text/plain), and plain
    text is the rest. A blob is what its MIME type, or else its extension,
    says; raises SkipError for one WhisperDocs does not read."""
    mime = (mime_type or "").split(";")[0].strip().lower()
    by_mime = _MIME_TYPES.get(mime)
    by_name = _extension_type(filename)
    if is_text:
        for candidate in (by_name, by_mime):
            if candidate in _TEXT_TYPES:
                return candidate
        return "txt"
    file_type = by_mime or by_name
    if file_type is None:
        raise SkipError(f"{mime or 'a file of unknown type'}: not a format WhisperDocs reads")
    return file_type


async def read_source(client: Client, source: Source) -> tuple[str, str, bytes]:
    """The resource's filename, format and bytes."""
    result = await client.read_resource(source.uri)
    contents = list(result.contents)
    if not contents:
        raise SkipError("the server returned no content")
    content = next((c for c in contents if str(c.uri) == source.uri), contents[0])
    filename = _filename(source)
    mime = content.mime_type or source.mime_type
    if isinstance(content, TextResourceContents):
        return filename, file_type_of(filename, mime, is_text=True), content.text.encode()
    if isinstance(content, BlobResourceContents):
        try:
            data = base64.b64decode(content.blob, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise SkipError("its content is not valid base64") from exc
        return filename, file_type_of(filename, mime, is_text=False), data
    raise SkipError("its content is neither text nor a blob")


# --- one resource ----------------------------------------------------------------------


async def sync_source(
    client: Client,
    source: Source,
    *,
    collection_id: int,
    sessions: Sessions,
    embedder: EmbeddingProvider | None,
    dry_run: bool = False,
) -> Item:
    """Read one resource and add, replace or leave its document. The
    embedder may be None on a dry run, which embeds nothing."""

    def item(outcome: Outcome, detail: str = "", document_id: int | None = None) -> Item:
        return Item(source.uri, outcome, detail, document_id)

    if len(source.uri) > MAX_URI_LENGTH:
        return item("skipped", f"its URI is longer than {MAX_URI_LENGTH} characters")
    if source.size is not None and source.size > MAX_DOCUMENT_BYTES:
        return item("skipped", f"larger than {MAX_DOCUMENT_BYTES // (1024 * 1024)} MB")
    try:
        filename, file_type, data = await read_source(client, source)
        file_type = check_document(filename, data, file_type)
        extracted = await extract_document(filename, file_type, data)
    except SkipError as exc:
        return item("skipped", str(exc))
    except MCPError as exc:
        return item("failed", f"the server could not read it: {exc}")
    except ValueError as exc:
        # The checks' and the extractor's own messages: empty, too large,
        # unreadable, not what its type says.
        return item("skipped", str(exc))

    async with sessions() as db:
        held = (
            await db.execute(
                select(Document.id, Document.content_hash).where(
                    Document.collection_id == collection_id, Document.source_uri == source.uri
                )
            )
        ).first()
        if held is not None and held.content_hash == extracted.content_hash:
            return item("unchanged", document_id=held.id)
        outcome: Outcome = "updated" if held is not None else "added"
        chunks = f"{len(extracted.chunks)} chunks"
        try:
            if dry_run or embedder is None:
                await ensure_new_document(db, collection_id, extracted.content_hash)
                return item(outcome, chunks, held.id if held is not None else None)
            document = await store_document(
                db,
                embedder,
                extracted,
                collection_id=collection_id,
                title=source.title or source.name,
                source_uri=source.uri,
                replaces=held.id if held is not None else None,
            )
        except DuplicateDocumentError as exc:
            return item("skipped", str(exc), exc.document_id)
        except EmbeddingError as exc:
            logger.exception("mcp_sync_embedding_failed", uri=source.uri)
            return item("failed", f"embedding it failed: {failure_text(exc)}")
    return item(outcome, chunks, document.id)


async def sync(
    client: Client,
    *,
    collection_id: int,
    sessions: Sessions,
    embedder: EmbeddingProvider | None,
    prefix: str | None = None,
    uris: Sequence[str] = (),
    dry_run: bool = False,
    on_item: Callable[[Item], None] | None = None,
) -> SyncReport:
    """Read the server's resources into the collection; see the module's
    docstring. A resource that fails does not stop the others. Without an
    embedder, the run is a dry run."""
    dry_run = dry_run or embedder is None
    async with sessions() as db:
        if await db.get(Collection, collection_id) is None:
            raise SyncError(f"No collection {collection_id}.")
    sources = await discover(client, prefix=prefix, uris=uris)
    report = SyncReport(dry_run=dry_run)
    for source in sources:
        try:
            result = await sync_source(
                client,
                source,
                collection_id=collection_id,
                sessions=sessions,
                embedder=embedder,
                dry_run=dry_run,
            )
        except Exception:
            logger.exception("mcp_sync_item_failed", uri=source.uri)
            result = Item(source.uri, "failed", "the connector's log has the details")
        report.items.append(result)
        if on_item is not None:
            on_item(result)
    logger.info("mcp_sync_done", collection_id=collection_id, dry_run=dry_run, **report.counts())
    return report


# --- the command -------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.connectors.mcp_sync",
        description="Ingest an MCP server's resources into a collection, and keep them in "
        "step on later runs.",
        epilog="The stdio server's command goes last, after --: "
        "python -m app.connectors.mcp_sync --collection 3 -- npx -y SERVER ARGS",
    )
    parser.add_argument("--collection", type=int, required=True, help="The collection's id.")
    parser.add_argument("--url", help="A server over Streamable HTTP, instead of a command.")
    parser.add_argument(
        "--bearer-env",
        metavar="VAR",
        help="With --url: send this environment variable's value as a bearer token.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="VAR",
        help="Pass this environment variable to the stdio server; repeat for more.",
    )
    parser.add_argument("--prefix", help="Only the listed resources whose URI starts with it.")
    parser.add_argument(
        "--uri",
        action="append",
        default=[],
        help="Read this resource, instead of the list; repeat for more.",
    )
    parser.add_argument(
        "--uri-file", type=Path, help="Read the resources whose URIs this file lists, one a line."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compare only: embed and write nothing."
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def _uris(args: argparse.Namespace) -> list[str]:
    uris = list(args.uri)
    if args.uri_file is not None:
        lines = args.uri_file.read_text(encoding="utf-8").splitlines()
        uris.extend(line.strip() for line in lines if line.strip() and not line.startswith("#"))
    return uris


async def _client(args: argparse.Namespace, stack: AsyncExitStack) -> Client:
    if args.url:
        headers: dict[str, str] = {}
        if args.bearer_env:
            token = os.environ.get(args.bearer_env)
            if not token:
                raise SyncError(f"{args.bearer_env} is not set.")
            headers["Authorization"] = f"Bearer {token}"
        http = await stack.enter_async_context(create_mcp_http_client(headers=headers))
        transport = streamable_http_client(args.url, http_client=http)
        return await stack.enter_async_context(Client(transport))
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    missing = [name for name in args.env if name not in os.environ]
    if missing:
        raise SyncError(f"Not set: {', '.join(missing)}.")
    env = {**_inherited(), **{name: os.environ[name] for name in args.env}}
    server = StdioServerParameters(command=command[0], args=command[1:], env=env)
    return await stack.enter_async_context(Client(server))


def _inherited() -> dict[str, str]:
    """What a stdio server gets without --env: the SDK's own safe set."""
    from mcp.client.stdio import get_default_environment

    return get_default_environment()


def _line(item: Item, dry_run: bool) -> str:
    outcome = item.outcome
    if dry_run and outcome in ("added", "updated"):
        outcome_text = f"would be {outcome}"
    else:
        outcome_text = outcome
    document = f" -> document {item.document_id}" if item.document_id is not None else ""
    detail = f" ({item.detail})" if item.detail else ""
    return f"{outcome_text:<18} {item.uri}{document}{detail}"


async def _run(args: argparse.Namespace) -> int:
    from app.db import SessionLocal
    from app.services.embeddings import get_embedder

    uris = _uris(args)
    async with AsyncExitStack() as stack:
        try:
            # A dry run embeds nothing, so it needs no embedding key.
            embedder = None if args.dry_run else get_embedder()
            try:
                client = await _client(args, stack)
            except (OSError, MCPError) as exc:
                raise SyncError(f"Could not start or reach the server: {exc}") from exc
            report = await sync(
                client,
                collection_id=args.collection,
                sessions=SessionLocal,
                embedder=embedder,
                prefix=args.prefix,
                uris=uris,
                dry_run=args.dry_run,
                on_item=lambda item: print(_line(item, args.dry_run), flush=True),
            )
        except (SyncError, EmbeddingError) as exc:
            print(exc, file=sys.stderr)
            return 1
    counts = report.counts()
    if not report.items:
        print(
            "The server listed no resources. A server that does not list its files (the "
            "filesystem server) reads them by URI: pass --uri or --uri-file.",
            file=sys.stderr,
        )
    summary = ", ".join(f"{n} {outcome}" for outcome, n in sorted(counts.items())) or "nothing"
    print(("Dry run: " if args.dry_run else "") + summary, file=sys.stderr)
    return 1 if counts.get("failed") else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if bool(args.url) == bool([a for a in args.command if a != "--"]):
        _parser().error("give either --url or the server's command after --")
    if args.bearer_env and not args.url:
        _parser().error("--bearer-env goes with --url")
    if sys.platform == "win32":
        # asyncpg and the Proactor loop: see app/main.py. The SDK starts a
        # stdio server under the selector loop with a Popen fallback.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())

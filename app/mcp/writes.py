"""The MCP server's tools that write (BE-36, audit M-1 and M-2).

- `ingest_document`: a file into a collection, as an upload is.
- `delete_document` and `move_document`: each asks the user first.

**Who may.** Over HTTP, a key made with `--write`, in the collections it
reaches; out of reach reads as missing, as it does for the read tools. The
API's own key and an open API write too, as the REST API lets them. Over
stdio, the process the client started.

**Asking the user** is MCP elicitation. On 2026-07-28 the question rides the
tool call's `InputRequiredResult`, and the client calls again with the answer
and the server's sealed `request_state` (the SDK's `Resolve`/`Elicit`); that
works on the stateless `/mcp`. On 2025-11-25 it is a request to the client in
the middle of the call, which a stateless HTTP exchange has no channel for,
so it works over stdio only. A client that cannot be asked is refused, and the
change waits for a client that can, or for the Documents page.

**Every call** leaves a row in `audit_log` (`app/services/audit.py`): what
was done, or why not.
"""

import binascii
from base64 import b64decode
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Annotated, Any, Literal, NoReturn

from mcp.server import MCPServer
from mcp.server.mcpserver import (
    AcceptedElicitation,
    CancelledElicitation,
    Context,
    Elicit,
    ElicitationResult,
    Resolve,
)
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from mcp_types.version import is_version_at_least
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.config import settings
from app.logging_config import get_logger
from app.mcp.access import Access, access_of, actor_of
from app.mcp.common import document_entry, why
from app.models import AuditLog, Collection, Document
from app.schemas.mcp import Confirmation, WriteResult
from app.services import audit
from app.services.documents import MoveConflictError, delete_documents, move_documents
from app.services.embeddings import EmbeddingError, EmbeddingProvider
from app.services.ingestion import DuplicateDocumentError, ingest_bytes

logger = get_logger(__name__)

Sessions = Callable[[], AbstractAsyncContextManager[AsyncSession]]
Embedder = Callable[[], EmbeddingProvider]
Spend = Callable[..., None]
# How a call the user was asked about ended.
Answer = Literal["done", "declined", "cancelled"]

ADDS = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)
DELETES = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
)
MOVES = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
)

# The first revision that asks inside the tool call's result, which a
# stateless HTTP exchange can carry.
_ASKS_IN_RESULT = "2026-07-28"
# 10 MB of bytes is about 13.4 million base64 characters.
MAX_CONTENT_CHARS = 14_000_000
READ_ONLY_KEY = (
    "This key only reads. A key that writes is made with "
    "`python -m app.mcp.keys create NAME --collection ID --write`."
)

DocumentId = Annotated[int, Field(description="The document (an id from list_documents).")]


def why_cannot_ask(ctx: Context) -> str | None:
    """Why the user cannot be asked to confirm through this client, or None
    when they can."""
    version = ctx.protocol_version
    in_result = version is not None and is_version_at_least(version, _ASKS_IN_RESULT)
    if not in_result and not ctx.session.can_send_request:
        # Checked first: such a request also arrives without the capabilities
        # its client declared, which the initialize of another request carried.
        return (
            "this connection cannot carry the question: over HTTP, a client on MCP "
            f"{version} asks within a session, and this server keeps none (a client on "
            f"{_ASKS_IN_RESULT} can, and so can stdio)"
        )
    capabilities = ctx.client_capabilities
    elicitation = capabilities.elicitation if capabilities is not None else None
    # An empty elicitation capability is form mode; url mode alone is not.
    if elicitation is None or (elicitation.form is None and elicitation.url is not None):
        return "this client cannot ask its user (it declares no MCP elicitation)"
    return None


class _Call:
    """One call to a tool that writes: who made it, and its audit rows."""

    def __init__(self, ctx: Context, action: str, sessions: Sessions) -> None:
        request = ctx.request_context.request
        self.access: Access = access_of(request)
        self.actor = actor_of(self.access, request)
        self.channel = "mcp-stdio" if request is None else "mcp-http"
        self.action = action
        self.sessions = sessions

    def row(self, outcome: audit.Outcome, **fields: Any) -> AuditLog:
        return audit.row(
            actor=self.actor, channel=self.channel, action=self.action, outcome=outcome, **fields
        )

    async def record(self, outcome: audit.Outcome, **fields: Any) -> None:
        await audit.record(
            self.sessions,
            actor=self.actor,
            channel=self.channel,
            action=self.action,
            outcome=outcome,
            **fields,
        )

    async def refuse(self, message: str, reason: str | None = None, **fields: Any) -> NoReturn:
        """Record a refusal and answer with `message`. `reason` is what the
        log keeps when the caller is told less (out of reach reads as
        missing)."""
        await self.record("refused", reason=reason or message, **fields)
        raise ToolError(message)

    async def fail(self, message: str, **fields: Any) -> NoReturn:
        await self.record("failed", reason=message, **fields)
        raise ToolError(message)

    async def require_write(self, **fields: Any) -> None:
        if not self.access.can_write:
            await self.refuse(READ_ONLY_KEY, reason="read-only key", **fields)

    async def require_asking(self, ctx: Context, what: str, **fields: Any) -> None:
        reason = why_cannot_ask(ctx)
        if reason is not None:
            await self.refuse(
                f"{what} needs the user's confirmation, and {reason}. Do it from "
                "WhisperDocs' Documents page instead.",
                **fields,
            )


async def _held(db: AsyncSession, document_id: int) -> Any:
    """The document's id, title, collection, chunk count and text hash, with
    its collection's name; None when it does not exist."""
    return (
        await db.execute(
            select(
                Document.id,
                Document.title,
                Document.collection_id,
                Document.chunk_count,
                Document.content_hash,
                Collection.name.label("collection"),
            )
            .join(Collection, Collection.id == Document.collection_id)
            .where(Document.id == document_id)
        )
    ).first()


async def _document_in_reach(db: AsyncSession, call: _Call, document_id: int) -> Any:
    held = await _held(db, document_id)
    if held is None or not call.access.reaches(held.collection_id):
        await call.refuse(
            f"Document {document_id} not found: list_documents gives the ids.",
            reason="not found" if held is None else "out of the key's reach",
            document_id=document_id,
        )
    return held


async def _collection_in_reach(
    db: AsyncSession, call: _Call, collection_id: int, **fields: Any
) -> Collection:
    collection = await db.get(Collection, collection_id)
    if collection is None or not call.access.reaches(collection_id):
        await call.refuse(
            f"Collection {collection_id} not found: list_collections gives the ids.",
            reason="not found" if collection is None else "out of the key's reach",
            collection_id=collection_id,
            **fields,
        )
    return collection


def _answered(confirmation: ElicitationResult[Confirmation]) -> Answer:
    """done when the user confirmed; declined or cancelled otherwise."""
    if isinstance(confirmation, AcceptedElicitation):
        return "done" if confirmation.data.confirm else "declined"
    return "cancelled" if isinstance(confirmation, CancelledElicitation) else "declined"


def add_write_tools(
    server: MCPServer, *, sessions: Sessions, embedder: Embedder, spend: Spend
) -> None:
    @server.tool(title="Add a document", annotations=ADDS)
    async def ingest_document(
        collection_id: Annotated[
            int, Field(description="The collection to add it to (an id from list_collections).")
        ],
        filename: Annotated[
            str,
            Field(
                min_length=1,
                max_length=300,
                description="The file's name. Its extension is its format: .md, .txt, .csv, "
                ".html, .pdf, .docx or .xlsx.",
            ),
        ],
        content: Annotated[
            str,
            Field(
                min_length=1,
                max_length=MAX_CONTENT_CHARS,
                description="The file: its text for .md, .txt, .csv and .html; its bytes in "
                "base64 for .pdf, .docx and .xlsx, with base64 set.",
            ),
        ],
        base64: Annotated[
            bool, Field(description="Whether `content` is base64 (a binary file).")
        ] = False,
        title: Annotated[
            str | None,
            Field(max_length=300, description="The title it shows. Null: the filename."),
        ] = None,
        *,
        ctx: Context,
    ) -> WriteResult:
        """Add a file to a collection, as an upload does: WhisperDocs
        extracts its text, splits it into chunks and embeds them, and then
        search_documents and ask_documents find it.

        At most 10 MB. A collection holds each document once: the same text
        again is refused, with the document that holds it. Embedding the
        file is paid for; the call counts against the API's upload limit.
        Needs a key that writes."""
        call = _Call(ctx, "ingest_document", sessions)
        fields: dict[str, Any] = {"collection_id": collection_id, "filename": filename}
        await call.require_write(**fields)
        async with sessions() as db:
            await _collection_in_reach(db, call, collection_id, filename=filename)
            try:
                spend(ctx, call.access, "uploads", settings.rate_limit_upload_per_minute)
            except ToolError as exc:
                await call.refuse(str(exc), **fields)
            try:
                file_bytes = b64decode(content, validate=True) if base64 else content.encode()
            except (binascii.Error, ValueError):
                await call.fail("`content` is not valid base64.", **fields)

            def done(document: Document) -> AuditLog:
                return call.row(
                    "done",
                    collection_id=collection_id,
                    document_id=document.id,
                    filename=filename,
                    title=document.title,
                    chunks=document.chunk_count,
                )

            try:
                document = await ingest_bytes(
                    db,
                    embedder(),
                    collection_id=collection_id,
                    filename=filename,
                    file_bytes=file_bytes,
                    title=title,
                    audit=done,
                )
            except EmbeddingError as exc:
                logger.exception("mcp_ingest_embedding_failed")
                await call.fail(f"Embedding the document failed: {why(exc)}.", **fields)
            except (DuplicateDocumentError, ValueError) as exc:
                # The extractor's and the checks' own messages, fit for the client.
                await call.fail(str(exc), **fields)
        return WriteResult(
            outcome="done",
            message=f'Added "{document.title}" (document {document.id}, '
            f"{document.chunk_count} chunks) to collection {collection_id}.",
            document=document_entry(document),
        )

    async def confirm_delete(document_id: int, ctx: Context) -> Elicit[Confirmation]:
        call = _Call(ctx, "delete_document", sessions)
        await call.require_write(document_id=document_id)
        async with sessions() as db:
            held = await _document_in_reach(db, call, document_id)
        await call.require_asking(
            ctx,
            "Deleting a document",
            document_id=document_id,
            collection_id=held.collection_id,
        )
        return Elicit(
            f'Delete "{held.title}" (document {held.id}, {held.chunk_count} chunks) from '
            f'the collection "{held.collection}"? It cannot be undone.',
            Confirmation,
        )

    @server.tool(title="Delete a document", annotations=DELETES)
    async def delete_document(
        document_id: DocumentId,
        confirmation: Annotated[ElicitationResult[Confirmation], Resolve(confirm_delete)],
        *,
        ctx: Context,
    ) -> WriteResult:
        """Delete a document and its chunks, after the user confirms it.

        The user is asked first: a client that cannot ask its user (MCP
        elicitation) is refused. Needs a key that writes."""
        call = _Call(ctx, "delete_document", sessions)
        outcome = _answered(confirmation)
        if outcome != "done":
            await call.record(outcome, document_id=document_id)
            return WriteResult(
                outcome=outcome, message=f"Nothing was deleted: the user {outcome} it."
            )
        async with sessions() as db:
            held = await _document_in_reach(db, call, document_id)
            if not await delete_documents(db, [document_id]):
                await call.fail(f"Document {document_id} not found.", document_id=document_id)
            db.add(
                call.row(
                    "done",
                    document_id=document_id,
                    collection_id=held.collection_id,
                    title=held.title,
                    chunks=held.chunk_count,
                )
            )
            await db.commit()
        return WriteResult(
            outcome="done",
            message=f'Deleted "{held.title}" (document {document_id}) from the collection '
            f'"{held.collection}".',
        )

    async def confirm_move(
        document_id: int, target_collection_id: int, ctx: Context
    ) -> Elicit[Confirmation]:
        call = _Call(ctx, "move_document", sessions)
        fields: dict[str, Any] = {
            "document_id": document_id,
            "target_collection_id": target_collection_id,
        }
        await call.require_write(**fields)
        async with sessions() as db:
            held = await _document_in_reach(db, call, document_id)
            fields["collection_id"] = held.collection_id
            target = await _collection_in_reach(
                db, call, target_collection_id, document_id=document_id
            )
            if held.collection_id == target_collection_id:
                await call.refuse(
                    f'Document {document_id} is in the collection "{target.name}" already.',
                    **fields,
                )
            if held.content_hash is not None:
                twin = await db.scalar(
                    select(Document.title).where(
                        Document.collection_id == target_collection_id,
                        Document.content_hash == held.content_hash,
                    )
                )
                if twin is not None:
                    await call.refuse(
                        f'The collection "{target.name}" already holds this document, as "{twin}".',
                        **fields,
                    )
        await call.require_asking(ctx, "Moving a document", **fields)
        return Elicit(
            f'Move "{held.title}" (document {held.id}) from the collection '
            f'"{held.collection}" to "{target.name}"? Searches of "{held.collection}", and '
            "the keys that reach only it, will not find it any more.",
            Confirmation,
        )

    @server.tool(title="Move a document", annotations=MOVES)
    async def move_document(
        document_id: DocumentId,
        target_collection_id: Annotated[
            int, Field(description="The collection to move it to (an id from list_collections).")
        ],
        confirmation: Annotated[ElicitationResult[Confirmation], Resolve(confirm_move)],
        *,
        ctx: Context,
    ) -> WriteResult:
        """Move a document to another collection, after the user confirms it.

        A collection holds each document once: a move into one that holds
        the same text is refused. The user is asked first: a client that
        cannot ask its user (MCP elicitation) is refused. Needs a key that
        writes, and reaches both collections."""
        call = _Call(ctx, "move_document", sessions)
        fields: dict[str, Any] = {
            "document_id": document_id,
            "target_collection_id": target_collection_id,
        }
        outcome = _answered(confirmation)
        if outcome != "done":
            await call.record(outcome, **fields)
            return WriteResult(
                outcome=outcome, message=f"Nothing was moved: the user {outcome} it."
            )
        async with sessions() as db:
            held = await _document_in_reach(db, call, document_id)
            fields["collection_id"] = held.collection_id
            await _collection_in_reach(db, call, target_collection_id, document_id=document_id)
            try:
                result = await move_documents(db, [document_id], target_collection_id)
            except MoveConflictError as exc:
                await call.fail(str(exc), **fields)
            if document_id in result.errors:
                await call.fail(result.errors[document_id], **fields)
            if document_id not in result.moved:
                await call.fail(f"Document {document_id} not found.", **fields)
            db.add(call.row("done", title=held.title, **fields))
            await db.commit()
            moved = await db.scalar(
                select(Document).options(defer(Document.raw_text)).where(Document.id == document_id)
            )
        return WriteResult(
            outcome="done",
            message=f'Moved "{held.title}" (document {document_id}) from collection '
            f"{held.collection_id} to {target_collection_id}.",
            document=document_entry(moved) if moved is not None else None,
        )

"""The MCP server's own results (audit M-1). Its search and answer results are
the API's: `RetrieveResponse` and `QueryResponse`."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.collection import CollectionRead
from app.schemas.document import DocumentRead


class CollectionList(BaseModel):
    collections: list[CollectionRead] = Field(description="Newest first.")


class DocumentEntry(DocumentRead):
    uri: str = Field(description="The document's resource: read it for the full text.")


class DocumentList(BaseModel):
    documents: list[DocumentEntry] = Field(description="Newest first.")
    next_cursor: int | None = Field(
        description="Pass it as `cursor` for the next page. Null on the last page."
    )
    total: int | None = Field(
        description="The documents across every page. Sent with the first page only."
    )


class Confirmation(BaseModel):
    """What the server asks the user before it deletes or moves a document
    (MCP elicitation, BE-36)."""

    confirm: bool = Field(title="Go ahead", description="Check it to go ahead.")


class WriteResult(BaseModel):
    outcome: Literal["done", "declined", "cancelled"] = Field(
        description="done: it happened. declined or cancelled: the user did not confirm, "
        "and nothing changed."
    )
    message: str = Field(description="What happened, in a sentence.")
    document: DocumentEntry | None = Field(
        default=None, description="The document added or moved. Null for a deletion."
    )

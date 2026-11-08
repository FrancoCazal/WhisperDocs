"""The MCP server's own results (audit M-1). Its search and answer results are
the API's: `RetrieveResponse` and `QueryResponse`."""

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

"""What the MCP server's read and write tools share."""

from app.models import Document
from app.schemas.document import DocumentRead
from app.schemas.mcp import DocumentEntry
from app.services.llm.errors import KIND_TEXT, classify_error

DOCUMENT_URI = "whisperdocs://documents/{document_id}"


def document_uri(document_id: int) -> str:
    return DOCUMENT_URI.format(document_id=document_id)


def document_entry(document: Document | DocumentRead) -> DocumentEntry:
    read = document if isinstance(document, DocumentRead) else DocumentRead.model_validate(document)
    return DocumentEntry(**read.model_dump(), uri=document_uri(read.id))


def why(exc: BaseException) -> str:
    """Why a model or embedding call failed, without the provider's text."""
    kind = classify_error(exc)
    return "the server's log has the details" if kind == "unknown" else KIND_TEXT[kind]

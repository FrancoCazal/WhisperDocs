"""Fields out of a document (audit M-8, app/services/extraction.py)."""

from typing import Any

from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    json_schema: dict[str, Any] = Field(
        description="A JSON Schema object whose properties are the fields to extract. Each "
        "is a string, number, integer or boolean, an enum of them, or a list of one of "
        "them; its description says what to look for. `required` lists the fields the "
        "document should state: one it does not comes back in `missing`.",
        examples=[
            {
                "type": "object",
                "properties": {
                    "invoice_number": {"type": "string"},
                    "total": {"type": "number", "description": "The amount due, no currency"},
                    "due_date": {"type": "string", "description": "As YYYY-MM-DD"},
                },
                "required": ["invoice_number", "total"],
            }
        ],
    )
    instructions: str | None = Field(
        default=None,
        max_length=2000,
        description="Anything else the extraction should know, in a sentence or two.",
    )


class FieldCitation(BaseModel):
    chunk_id: int
    chunk_index: int
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    quote: str = Field(description="The words of the passage that state the value.")
    verified: bool = Field(
        description="The quote is in the passage word for word, case, spacing and quote "
        "marks aside. False: the model's quote is not in the document, so check the value."
    )


class ExtractedField(BaseModel):
    value: Any = Field(description="Null when the document does not state it.")
    citation: FieldCitation | None = Field(
        description="Where the document states it. Null when the value is null, or the "
        "model gave no quote."
    )


class ExtractionMetadata(BaseModel):
    model: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    cost_usd: float
    passages_used: int = Field(description="The passages the model read.")
    passages_total: int = Field(
        description="The document's passages. More than passages_used: the document was "
        "too long for one call, and a search for the fields chose the passages."
    )


class ExtractionResponse(BaseModel):
    document_id: int
    document_title: str
    values: dict[str, Any] = Field(description="Each field's value, by name: `fields` alone.")
    fields: dict[str, ExtractedField]
    missing: list[str] = Field(description="Required fields the document does not state.")
    unverified: list[str] = Field(
        description="Fields with a value whose quote is not in the document, or has none."
    )
    metadata: ExtractionMetadata

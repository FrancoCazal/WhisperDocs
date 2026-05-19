from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    collection_id: int
    title: str
    filename: str
    file_type: str
    chunk_count: int
    total_tokens: int
    created_at: datetime


class DocumentDetail(DocumentRead):
    raw_text: str


class BatchUploadItem(BaseModel):
    status: str
    filename: str
    document: DocumentRead | None = None
    error: str | None = None


class BatchUploadResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[BatchUploadItem]


class BulkDocumentIds(BaseModel):
    document_ids: list[int]


class BulkMoveRequest(BulkDocumentIds):
    target_collection_id: int


class BulkOpResult(BaseModel):
    requested: int
    succeeded: int
    failed: int
    not_found: list[int] = []
    errors: dict[int, str] = {}

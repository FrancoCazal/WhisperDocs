from datetime import datetime

from pydantic import BaseModel, ConfigDict


class IngestionJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    collection_id: int
    document_id: int | None
    filename: str
    file_type: str
    title: str | None
    status: str
    progress: int
    total_chunks: int
    processed_chunks: int
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class IngestionJobCreate(BaseModel):
    job_id: int
    status: str

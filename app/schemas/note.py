from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SavedNoteCreate(BaseModel):
    content: str = Field(min_length=1, max_length=50000)
    user_note: str | None = Field(default=None, max_length=2000)


class SavedNoteRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int | None
    collection_id: int | None
    source_message_content: str
    user_note: str | None
    created_at: datetime

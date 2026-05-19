from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    thread_id: str
    user_number: str | None
    channel: str
    collection_id: int | None
    is_saved: bool
    last_active_at: datetime
    created_at: datetime


class ConversationUpdate(BaseModel):
    is_saved: bool


class ConversationMessage(BaseModel):
    role: str
    content: str
    timestamp: str | None = None
    citations: list[dict] | None = None
    metadata: dict | None = None

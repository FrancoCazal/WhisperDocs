from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    thread_id: str
    user_number: str | None
    channel: str
    collection_id: int | None
    # The one document the thread searches, if it searches only one.
    document_id: int | None = None
    # The thread's first question; null until it has one.
    title: str | None = None
    is_saved: bool
    last_active_at: datetime
    created_at: datetime


class ConversationUpdate(BaseModel):
    is_saved: bool


class ConversationMessage(BaseModel):
    role: str
    content: str
    # When the message was written (UTC, ISO 8601); older messages have none.
    timestamp: str | None = None
    # The node that answered (search_docs, answer_direct, ...); assistant only.
    action: str | None = None
    citations: list[dict] | None = None
    metadata: dict | None = None
    # An assistant message that reports a failure (see AgentChatResponse).
    failed: bool = False

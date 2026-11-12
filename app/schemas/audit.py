from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class AuditLogRead(BaseModel):
    """A call that wrote through the MCP server, and how it ended
    (app/services/audit.py)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    actor: str
    channel: str
    action: str
    outcome: str
    collection_id: int | None
    document_id: int | None
    detail: dict[str, Any]

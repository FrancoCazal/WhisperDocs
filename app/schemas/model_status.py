from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.services.llm.errors import ErrorKind


class RoleStatusRead(BaseModel):
    model: str = Field(description="The model this role is configured with.")
    state: Literal["ok", "failing", "unknown"] = Field(
        description="ok: the last call worked. failing: the last call failed. "
        "unknown: no call since the API started."
    )
    calls: int
    failures_in_a_row: int
    last_ok_at: datetime | None
    last_failure_at: datetime | None
    last_failure_kind: ErrorKind | None


class ModelStatusRead(BaseModel):
    since: datetime = Field(description="When the record started: the API's start.")
    roles: dict[str, RoleStatusRead] = Field(
        description="classifier, query_rewrite, rag_generation, clarification, "
        "answer_direct, summarize, eval_judge and embedding."
    )

"""What the last calls to each model did, from real traffic (BE-14).

Every call a role makes records its outcome here: the agent's nodes, `/query`
and `/eval/run` through `llm_service_for`, the embedder for every embedding.
Each role keeps when it last worked, when it last failed and why
(`classify_error`), and how many calls in a row have failed. Nothing here
calls a model: a role with no call since startup is "unknown".

The record lives in this process's memory, like the rate limits: one uvicorn
process, starting over on a restart. The eval CLI runs in a process of its
own and does not show here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypedDict

from app.config import settings
from app.metrics import model_calls_total
from app.services.llm.errors import ErrorKind, classify_error

Role = Literal[
    "classifier",
    "query_rewrite",
    "rag_generation",
    "clarification",
    "answer_direct",
    "summarize",
    "eval_judge",
    "embedding",
]
ROLES: tuple[Role, ...] = (
    "classifier",
    "query_rewrite",
    "rag_generation",
    "clarification",
    "answer_direct",
    "summarize",
    "eval_judge",
    "embedding",
)

State = Literal["ok", "failing", "unknown"]


class RoleState(TypedDict):
    model: str
    state: State
    calls: int
    failures_in_a_row: int
    last_ok_at: datetime | None
    last_failure_at: datetime | None
    last_failure_kind: ErrorKind | None


@dataclass
class _Record:
    calls: int = 0
    failures_in_a_row: int = 0
    last_ok_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_failure_kind: ErrorKind | None = None


def configured_model(role: Role) -> str:
    if role == "embedding":
        return settings.embedding_model
    model: str = getattr(settings.agent, f"model_{role}")
    return model


def _now() -> datetime:
    return datetime.now(UTC)


class ModelStatus:
    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self._clock = clock
        # By role and model: a role reports the model it is configured with.
        self._records: dict[tuple[Role, str], _Record] = {}
        self.since = clock()

    def _record(self, role: Role, model: str) -> _Record:
        return self._records.setdefault((role, model), _Record())

    def ok(self, role: Role, model: str) -> None:
        record = self._record(role, model)
        record.calls += 1
        record.failures_in_a_row = 0
        record.last_ok_at = self._clock()
        model_calls_total.labels(role=role, model=model, outcome="ok").inc()

    def failed(self, role: Role, model: str, exc: BaseException) -> ErrorKind:
        kind = classify_error(exc)
        record = self._record(role, model)
        record.calls += 1
        record.failures_in_a_row += 1
        record.last_failure_at = self._clock()
        record.last_failure_kind = kind
        model_calls_total.labels(role=role, model=model, outcome=kind).inc()
        return kind

    def role(self, role: Role) -> RoleState:
        model = configured_model(role)
        record = self._records.get((role, model)) or _Record()
        state: State = (
            "unknown" if record.calls == 0 else "failing" if record.failures_in_a_row else "ok"
        )
        return {
            "model": model,
            "state": state,
            "calls": record.calls,
            "failures_in_a_row": record.failures_in_a_row,
            "last_ok_at": record.last_ok_at,
            "last_failure_at": record.last_failure_at,
            "last_failure_kind": record.last_failure_kind,
        }

    def snapshot(self) -> dict[Role, RoleState]:
        return {role: self.role(role) for role in ROLES}

    def reset(self) -> None:
        self._records.clear()
        self.since = self._clock()


model_status = ModelStatus()

"""The calls a turn made that failed or fell back.

Each node that catches a failed call adds `{step, kind, model}` to
`metadata["turn_errors"]`, and the node that answers copies the list onto its
message (`additional_kwargs["errors"]`), so a reopened thread shows it too.
A turn can answer with errors: when the classifier fails it searches the
documents instead, when the query rewrite fails it searches with the question
as written. `turn_failed` says whether the answer itself failed.

Steps:
  summarize      compacting the history (it stays uncompacted)
  classifier     choosing the action (search_docs instead)
  query_rewrite  rewriting the question for retrieval (searched as written)
  embedding      embedding the question (the answer failed)
  retrieval      searching the database (the answer failed)
  generation     writing the answer (the answer failed)
  clarification  asking for details (a fixed question instead)
  citation_check checking the citations against the passages (not checked)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from app.services.llm.errors import ErrorKind, classify_error

TurnStep = Literal[
    "summarize",
    "classifier",
    "query_rewrite",
    "embedding",
    "retrieval",
    "generation",
    "clarification",
    "citation_check",
]


class TurnError(TypedDict):
    step: TurnStep
    kind: ErrorKind
    model: str | None


def turn_error(step: TurnStep, exc: BaseException, model: str | None) -> TurnError:
    return {"step": step, "kind": classify_error(exc), "model": model}


def failure_kind(metadata: Mapping[str, Any]) -> ErrorKind | None:
    """The kind of error that failed the answer, or None when the turn
    answered. The answering step runs last, so its error is the last one
    recorded."""
    if not metadata.get("turn_failed"):
        return None
    errors = metadata.get("turn_errors") or []
    return errors[-1]["kind"] if errors else "unknown"


def errors_so_far(state: Mapping[str, Any]) -> list[TurnError]:
    """The errors earlier nodes of this turn recorded."""
    metadata = state.get("metadata") or {}
    return list(metadata.get("turn_errors") or [])

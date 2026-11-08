"""The version of the prompts a turn or an eval run sent (audit M-4).

`PROMPT_VERSION` is 12 hex characters of a SHA-256 over the text the agent and
`/query` send around the user's words and the documents: each upper-case
constant in `app/agent/prompts.py` (the system prompts and the RAG answer's
layout), and the JSON schemas of the two structured outputs, the classifier's
`AgentAction` and the answer's `RAGAnswer`, whose field descriptions reach the
model too. It changes when any of them does, and only then. Each `query_logs`
row, each turn's trace and each eval report carry it, so a change in quality
or cost lines up with the prompt change that made it.

A constant joins the version by being in `prompts.py`, with no list to keep in
step. Not hashed: the summary node's framing of its transcript
(`app/agent/nodes/summarize.py`), and the models and their settings, which
`query_logs.model_used` and the report's `model` record. A pydantic upgrade
that renders a schema differently changes the version too.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from app.agent import prompts
from app.agent.schemas import AgentAction, RAGAnswer


def version_of(parts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:12]


def prompt_texts(namespace: Mapping[str, Any]) -> list[str]:
    """Each upper-case text constant of a module (a string, or a tuple of
    strings), as `NAME=value`, by name. Private ones count: a fragment that
    builds a prompt is part of it."""
    texts = []
    for name in sorted(namespace):
        value = namespace[name]
        if not name.isupper():
            continue
        if isinstance(value, tuple) and all(isinstance(v, str) for v in value):
            value = "\n".join(value)
        if isinstance(value, str):
            texts.append(f"{name}={value}")
    return texts


def compute_prompt_version(namespace: Mapping[str, Any], schemas: Sequence[type[BaseModel]]) -> str:
    return version_of(
        [
            *prompt_texts(namespace),
            *(json.dumps(m.model_json_schema(), sort_keys=True) for m in schemas),
        ]
    )


PROMPT_VERSION = compute_prompt_version(vars(prompts), (AgentAction, RAGAnswer))

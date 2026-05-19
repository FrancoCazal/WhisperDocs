"""Shared RAG prompt + context-block helpers.

Single source of truth for both production paths:
  - `app/agent/nodes/search_docs.py` (LangGraph agent — chat).
  - `app/api/v1/query.py` (legacy POST /query) and `app/eval/evaluator.py`.

Keeping these aligned matters because the eval suite measures the prompt the
user actually sees in production. Divergence makes Sprint 3-style metrics
(hallucination, citation accuracy) meaningless on the eval side.
"""

from __future__ import annotations

import re

import tiktoken

from app.agent.prompts import RAG_SYSTEM_PROMPT_AGENT
from app.logging_config import get_logger
from app.services.retriever import RetrievedChunk

logger = get_logger(__name__)

# Re-export the hardened agent prompt as the canonical RAG system prompt.
# Callers that imported `RAG_SYSTEM_PROMPT` from this module pre-Sprint-3
# now automatically get the hardened anti-injection version.
RAG_SYSTEM_PROMPT = RAG_SYSTEM_PROMPT_AGENT

CITATION_PATTERN = re.compile(r"\[(\d+)\]")

_CONTEXT_ENCODER = tiktoken.get_encoding("cl100k_base")


def validate_citations(answer: str, num_citations: int) -> tuple[str, list[int]]:
    """Strip invalid `[n]` markers from answer text. Return (cleaned, used_1based).

    Drops references that point past the chunk count (`[9]` when only 3 chunks
    exist) so the rendered text never confuses readers. Used markers are
    surfaced for downstream metadata.
    """
    if num_citations == 0:
        return CITATION_PATTERN.sub("", answer).strip(), []

    used: set[int] = set()
    invalid: set[int] = set()

    def _replace(m: re.Match[str]) -> str:
        n = int(m.group(1))
        if 1 <= n <= num_citations:
            used.add(n)
            return m.group(0)
        invalid.add(n)
        return ""

    cleaned = CITATION_PATTERN.sub(_replace, answer)
    if invalid:
        logger.warning(
            "citation_invalid_dropped",
            invalid_refs=sorted(invalid),
            num_citations=num_citations,
        )
    return cleaned.strip(), sorted(used)


def format_chunk(idx: int, chunk: RetrievedChunk) -> str:
    """Render a single retrieved chunk as a numbered markdown block.

    NOTE: an earlier Sprint 3 iteration switched this to XML `<chunk>`
    elements on the theory that Claude/Gemini parse XML more reliably and
    that XML disambiguates literal `[n]` markers inside chunk text. Eval
    measurement showed:
      - +9% input tokens (XML structure overhead)
      - hallucination unchanged or slightly worse
      - no measurable accuracy lift on the golden dataset
    Reverted to markdown. Re-evaluate if/when corpora contain adversarial
    chunks with literal citation markers, or when running on >50k-token
    multi-turn contexts where Anthropic's XML guidance is more relevant.
    """
    return f'[{idx}] (from "{chunk.document_title}", chunk {chunk.chunk_index})\n{chunk.text}'


def build_context_block(
    chunks: list[RetrievedChunk],
    token_budget: int | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    """Render chunks as a markdown `[n]`-numbered block honoring an optional
    token budget.

    Iteratively adds chunks (already ordered by descending relevance) until the
    next chunk would exceed `token_budget`. Returns the rendered block and the
    subset of chunks actually included — citations must reference this subset,
    not the full input list. With `token_budget=None`, includes everything.
    """
    if not chunks:
        return "", []

    used: list[RetrievedChunk] = []
    used_blocks: list[str] = []
    separator = "\n\n---\n\n"

    if token_budget is None:
        for chunk in chunks:
            used.append(chunk)
            used_blocks.append(format_chunk(len(used), chunk))
        return separator.join(used_blocks), used

    total_tokens = 0
    separator_tokens = len(_CONTEXT_ENCODER.encode(separator))

    for chunk in chunks:
        rendered = format_chunk(len(used) + 1, chunk)
        cost = len(_CONTEXT_ENCODER.encode(rendered))
        if used and total_tokens + separator_tokens + cost > token_budget:
            continue
        if not used and cost > token_budget:
            used.append(chunk)
            used_blocks.append(rendered)
            total_tokens += cost
            break
        if used:
            total_tokens += separator_tokens
        used.append(chunk)
        used_blocks.append(rendered)
        total_tokens += cost

    dropped = len(chunks) - len(used)
    if dropped > 0:
        logger.info(
            "context_budget_dropped",
            dropped=dropped,
            kept=len(used),
            total_tokens=total_tokens,
            budget=token_budget,
        )
    return separator.join(used_blocks), used


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build the user-message prompt: numbered context block + question.

    Used by the legacy `/query` endpoint and the eval suite. The agent path
    (`search_docs_node`) calls `build_context_block` directly so it can apply
    a token budget against the agent's settings.
    """
    context_block, _ = build_context_block(chunks, token_budget=None)
    return (
        f"Context:\n\n{context_block}\n\n---\n\n"
        f"Question: {question}\n\nAnswer:"
    )

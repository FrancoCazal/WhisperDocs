"""Semantic citation validation (#22, opt-in).

Catches *cite-stuffing* — the LLM emitting `[n]` against a chunk whose text
does not actually support the surrounding claim. The structural validator in
`search_docs._validate_citations` only checks that `n` is in range; semantic
validation goes one level deeper:

1. Split the answer into sentences.
2. For every sentence carrying citations, embed the sentence and each cited
   chunk's text.
3. For each citation, take the mean cosine similarity across sentences that
   use it. Flag citations whose mean falls below `min_similarity`.

Trade-off: costs N + C extra embeddings per RAG turn (N sentences,
C unique cited chunks). Off by default (`agent.citation_semantic_validation`),
enable in eval and for high-stakes tenants.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Robust when inputs are not yet unit-normalised."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


async def validate_citations_semantic(
    *,
    answer: str,
    included_chunks: list[Any],
    embedder: Any,
    min_similarity: float,
) -> dict:
    """Return a dict with `low_confidence_refs` (1-based ids) and per-citation scores.

    `embedder` must expose `embed_batch(texts) -> list[list[float]]`.
    `included_chunks` must be ordered so index `i` corresponds to citation `i+1`.

    On any internal failure the function returns empty results — the caller
    must not block answer delivery on validation.
    """
    sentences = _split_sentences(answer)
    if not sentences or not included_chunks:
        return {"low_confidence_refs": [], "scores": {}}

    # sentence_idx → list of cited 1-based chunk ids that exist in range
    sentence_cites: list[list[int]] = []
    cited_ids: set[int] = set()
    for s in sentences:
        ids = []
        for m in _CITATION_PATTERN.finditer(s):
            n = int(m.group(1))
            if 1 <= n <= len(included_chunks):
                ids.append(n)
                cited_ids.add(n)
        sentence_cites.append(ids)

    if not cited_ids:
        return {"low_confidence_refs": [], "scores": {}}

    # Only sentences that actually carry valid citations need embedding.
    relevant_sentence_idxs = [i for i, ids in enumerate(sentence_cites) if ids]
    if not relevant_sentence_idxs:
        return {"low_confidence_refs": [], "scores": {}}

    sorted_cited = sorted(cited_ids)
    chunk_texts = [included_chunks[n - 1].text for n in sorted_cited]
    chunk_pos: dict[int, int] = {n: i for i, n in enumerate(sorted_cited)}

    sentence_texts = [sentences[i] for i in relevant_sentence_idxs]

    try:
        sentence_vecs, chunk_vecs = await _embed_pair(
            embedder, sentence_texts, chunk_texts
        )
    except Exception:
        logger.exception("citation_semantic_validation_embed_failed")
        return {"low_confidence_refs": [], "scores": {}}

    sims_per_cite: dict[int, list[float]] = defaultdict(list)
    for local_idx, original_idx in enumerate(relevant_sentence_idxs):
        s_vec = sentence_vecs[local_idx]
        for n in sentence_cites[original_idx]:
            c_vec = chunk_vecs[chunk_pos[n]]
            sims_per_cite[n].append(_cosine(s_vec, c_vec))

    scores: dict[int, float] = {
        n: (sum(vals) / len(vals)) if vals else 0.0 for n, vals in sims_per_cite.items()
    }
    low_confidence = sorted(n for n, score in scores.items() if score < min_similarity)
    if low_confidence:
        logger.info(
            "citation_low_confidence",
            refs=low_confidence,
            scores={n: round(s, 3) for n, s in scores.items()},
            min_similarity=min_similarity,
        )
    return {"low_confidence_refs": low_confidence, "scores": scores}


async def _embed_pair(
    embedder: Any, sentences: list[str], chunks: list[str]
) -> tuple[list[list[float]], list[list[float]]]:
    """Single batched embed call for sentences + chunks; split the result."""
    combined = sentences + chunks
    vectors = await embedder.embed_batch(combined)
    return vectors[: len(sentences)], vectors[len(sentences) :]

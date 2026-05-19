"""Maximal Marginal Relevance for source diversity.

Standard MMR uses query/document embeddings; we approximate with document-id
similarity (chunks from already-selected documents incur a penalty). This is
cheaper than embedding-based MMR (no extra inference) and effective when the
common failure mode is "top_k chunks all come from the same long doc".

Score(c) = λ * relevance(c) − (1 − λ) * max_over_selected(I[doc(c) == doc(s)])

- λ = 1.0 → pure relevance (equivalent to no MMR).
- λ = 0.0 → pure diversity (favor new docs even at relevance cost).
- λ = 0.5 → balanced default.
"""

from __future__ import annotations

from app.services.retriever import RetrievedChunk


def mmr_select(
    candidates: list[RetrievedChunk],
    top_k: int,
    lambda_: float = 0.5,
) -> list[RetrievedChunk]:
    if not candidates or top_k <= 0:
        return []
    if top_k >= len(candidates):
        return candidates

    pool = list(candidates)
    selected: list[RetrievedChunk] = []
    selected_doc_ids: set[int] = set()

    while pool and len(selected) < top_k:
        best_idx = 0
        best_score = -float("inf")
        for i, cand in enumerate(pool):
            penalty = 1.0 if cand.document_id in selected_doc_ids else 0.0
            score = lambda_ * cand.relevance - (1.0 - lambda_) * penalty
            if score > best_score:
                best_score = score
                best_idx = i
        chosen = pool.pop(best_idx)
        selected.append(chosen)
        selected_doc_ids.add(chosen.document_id)

    return selected

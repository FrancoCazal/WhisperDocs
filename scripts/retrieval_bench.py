"""Retrieval-only benchmark over the golden set. No LLM calls.

For each question it runs the vector leg, the full-text leg and the fused
pipeline (`retrieval_pipeline.retrieve`, with whatever reranker / MMR settings
are active) and reports:

- how many questions each leg returns rows for, in-scope and out-of-scope;
- precision@k and recall@k over the in-scope questions, with the eval's
  definitions (`RAGEvaluator._compute_precision` / `_compute_recall`).

The only paid call is one embedding per question; `--embeddings-cache` stores
them so repeated runs (before/after a change) use identical vectors.

Usage:
    python -m scripts.seed_eval
    python -m scripts.retrieval_bench --embeddings-cache eval/results/query-embeddings.json
    python -m scripts.retrieval_bench --json eval/results/retrieval-bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.db import SessionLocal
from app.eval.evaluator import RAGEvaluator
from app.eval.runner import DEFAULT_DATASET, _resolve_collection_id, load_dataset
from app.eval.schemas import Category, TestItem
from app.services.embedder import EmbeddingService
from app.services.retrieval_pipeline import retrieve as retrieve_pipeline
from app.services.retriever import RetrievedChunk, VectorRetriever

LEGS = ("vector", "lexical", "hybrid")


async def _embeddings(items: list[TestItem], cache: Path | None) -> dict[str, list[float]]:
    cached: dict[str, list[float]] = {}
    if cache is not None and cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
    missing = [it.question for it in items if it.question not in cached]
    if missing:
        vectors = await EmbeddingService().embed_batch(missing)
        cached.update(zip(missing, vectors, strict=True))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(cached), encoding="utf-8")
    return cached


async def _run_legs(
    item: TestItem, embedding: list[float], collection_id: int, top_k: int
) -> dict[str, list[RetrievedChunk]]:
    async with SessionLocal() as db:
        retriever = VectorRetriever(db)
        vector = await retriever.retrieve(embedding, collection_id=collection_id, top_k=top_k)
        lexical = await retriever.lexical_retrieve(
            item.question, collection_id=collection_id, limit=top_k
        )
        hybrid = await retrieve_pipeline(
            db=db,
            query_text=item.question,
            query_embedding=embedding,
            collection_id=collection_id,
            top_k=top_k,
        )
    return {"vector": vector, "lexical": lexical, "hybrid": hybrid}


def _summarize(rows: list[dict]) -> dict:
    in_scope = [r for r in rows if r["category"] == Category.IN_SCOPE.value]
    out_scope = [r for r in rows if r["category"] != Category.IN_SCOPE.value]
    summary: dict = {}
    for leg in LEGS:
        n = len(in_scope) or 1
        summary[leg] = {
            "in_scope_with_rows": sum(1 for r in in_scope if r[leg]["sources"]),
            "out_of_scope_with_rows": sum(1 for r in out_scope if r[leg]["sources"]),
            "precision": round(sum(r[leg]["precision"] for r in in_scope) / n, 4),
            "recall": round(sum(r[leg]["recall"] for r in in_scope) / n, 4),
        }
    summary["in_scope"] = len(in_scope)
    summary["out_of_scope"] = len(out_scope)
    return summary


def _print(summary: dict, top_k: int) -> None:
    print(
        f"\nin-scope questions: {summary['in_scope']}   "
        f"out-of-scope questions: {summary['out_of_scope']}   k={top_k}\n"
    )
    print(f"{'leg':<8} {'rows (in)':>10} {'rows (out)':>11} {'precision':>10} {'recall':>8}")
    for leg in LEGS:
        s = summary[leg]
        print(
            f"{leg:<8} {s['in_scope_with_rows']:>10} {s['out_of_scope_with_rows']:>11} "
            f"{s['precision']:>10.2%} {s['recall']:>8.2%}"
        )


async def run(args: argparse.Namespace) -> dict:
    items = load_dataset(args.dataset)
    collection_id = await _resolve_collection_id(args.collection_id, args.collection_name)
    embeddings = await _embeddings(items, args.embeddings_cache)

    rows: list[dict] = []
    for item in items:
        legs = await _run_legs(item, embeddings[item.question], collection_id, args.top_k)
        row: dict = {"id": item.id, "category": item.category.value, "question": item.question}
        for leg, chunks in legs.items():
            sources = [c.document_title for c in chunks]
            row[leg] = {
                "sources": sources,
                "precision": RAGEvaluator._compute_precision(sources, item.expected_sources),
                "recall": RAGEvaluator._compute_recall(sources, item.expected_sources),
            }
        rows.append(row)

    summary = _summarize(rows)
    _print(summary, args.top_k)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
        )
        print(f"\nPer-question results: {args.json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--collection-id", type=int, default=None)
    parser.add_argument("--collection-name", type=str, default="Eval Golden")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embeddings-cache", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

"""What it takes to compare two eval runs (audit E7, E8).

- `run_config`: the settings a run answered and judged with, saved in its
  report, so a diff can tell which of them changed.
- `wilson_interval`: how far a share measured on n items can be from the
  true one, at 95 %. 48 right out of 50 is [86.5 %, 98.9 %].
- `mcnemar_exact`: whether the items that flipped between two runs over the
  same questions flipped more one way than chance would. It reads only the
  flips: on 50 items, 6 regressions and no fix give p = 0.031, and 3 and 3
  give p = 1.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from app.config import settings
from app.eval.schemas import RunConfig, TestItem
from app.services.llm.model_policy import accepts_custom_sampling, reasoning_level

REPO_ROOT = Path(__file__).resolve().parents[2]


def dataset_hash(items: Sequence[TestItem]) -> str:
    """12 hex characters over the items as the run read them: their order,
    questions, expected answers and sources. A `--limit` run has its own."""
    canonical = json.dumps(
        [item.model_dump(mode="json") for item in items], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def git_revision() -> tuple[str | None, bool | None]:
    """The commit the code ran from, and whether the tree had changes on top.

    `GIT_SHA` wins: an image without `.git` can pass it in. Then git, then
    `.git` read by hand (the API image has no git). `GITHUB_SHA` comes last:
    the prompt gate runs the base from a second checkout, and the workflow's
    commit is the head's. A tree with changes runs code no commit holds, so
    its report says so."""
    if sha := os.environ.get("GIT_SHA", "").strip():
        return sha, None
    try:
        sha = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no") != ""
    except (OSError, subprocess.SubprocessError):
        found = _head_from_files(REPO_ROOT / ".git")
        return found or os.environ.get("GITHUB_SHA", "").strip() or None, None
    return sha, dirty


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True
    )
    return out.stdout.strip()


def _head_from_files(git_dir: Path) -> str | None:
    """HEAD's commit read from `.git` itself: a detached HEAD holds it, a
    branch names a loose ref or a line of `packed-refs`."""
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head or None
        ref = head.removeprefix("ref: ")
        loose = git_dir / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip() or None
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            sha, _, name = line.partition(" ")
            if name == ref:
                return sha
    except OSError:
        return None
    return None


def sampling(model: str, temperature: float) -> float | None:
    """The temperature a call to `model` sends: None when the model takes only
    its default sampling (app/services/llm/model_policy.py)."""
    return temperature if accepts_custom_sampling(model) else None


def run_config(
    *,
    items: Sequence[TestItem],
    collection_id: int,
    top_k: int,
    concurrency: int,
    rag_model: str,
    rag_temperature: float,
    judge_model: str,
    judge_temperature: float,
    judge_cache: bool,
) -> RunConfig:
    sha, dirty = git_revision()
    return RunConfig(
        git_sha=sha,
        git_dirty=dirty,
        dataset_hash=dataset_hash(items),
        collection_id=collection_id,
        top_k=top_k,
        concurrency=concurrency,
        rag_temperature=sampling(rag_model, rag_temperature),
        rag_reasoning=reasoning_level(rag_model),
        rag_structured_output=settings.agent.rag_structured_output,
        rag_context_token_budget=settings.agent.rag_context_token_budget,
        judge_model=judge_model,
        judge_temperature=sampling(judge_model, judge_temperature),
        judge_reasoning=reasoning_level(judge_model),
        judge_cache=judge_cache,
        embedding_model=settings.embedding_model,
        embedding_dimension=settings.embedding_dimension,
        reranker_model=settings.reranker_model if settings.reranker_enabled else None,
        reranker_candidate_pool=settings.reranker_candidate_pool,
        reranker_score_threshold=settings.reranker_score_threshold,
        mmr_lambda=settings.mmr_lambda if settings.mmr_enabled else None,
    )


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The 95 % Wilson score interval of a share. It stays inside [0, 1] and
    holds up at 0 or n successes, where the normal one does not."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(worse: int, better: int) -> float:
    """Two-sided exact McNemar p-value for paired verdicts: `worse` items
    passed before and fail now, `better` the other way. Under no change each
    flip goes either way with even odds, so it is a binomial test on the
    flips. No flip is no evidence: p = 1."""
    n = worse + better
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(worse, better) + 1)) / 2**n
    return min(1.0, 2 * tail)

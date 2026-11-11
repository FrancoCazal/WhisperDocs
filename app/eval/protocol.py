"""What it takes to compare two eval runs (audit E7, E8).

- `run_config`: the settings a run answered and judged with, saved in its
  report, so a diff can tell which of them changed.
- `collection_fingerprint`: the chunks the run's collection held, so a diff
  can tell a run over a reseeded or rechunked corpus.
- `wilson_interval`: how far a share measured on n items can be from the
  true one, at 95 %. 48 right out of 50 is [86.5 %, 98.9 %].
- `mean_interval`: how far the mean of a few runs of one config can be
  from the mean of many, at 95 % (Student's t). It measures the noise
  between runs, not the sampling of the items.
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

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.eval.schemas import RunConfig, TestItem
from app.models import Chunk, Document
from app.services.llm.model_policy import accepts_custom_sampling, reasoning_level

REPO_ROOT = Path(__file__).resolve().parents[2]


def dataset_hash(items: Sequence[TestItem]) -> str:
    """12 hex characters over the items as the run read them: their order,
    questions, expected answers, sources and facts. A `--limit` run has its
    own. An item with no facts hashes as it did before they existed, so a
    dataset that never gained any keeps its hash."""
    rows = []
    for item in items:
        row = item.model_dump(mode="json")
        for name in ("required_facts", "optional_facts"):
            if not row[name]:
                del row[name]
        rows.append(row)
    canonical = json.dumps(rows, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def context_hash(texts: Sequence[str]) -> str:
    """16 hex characters over the chunks a judge read, in order. `runner
    rejudge` reads the chunks back by id; a database rebuilt since can hold
    other text under the same ids, and this tells them apart."""
    return hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest()[:16]


async def collection_fingerprint(db: AsyncSession, collection_id: int) -> tuple[int, str]:
    """The chunks a collection holds: how many, and 12 hex characters over
    their text hashes, sorted.

    The same texts give the same hash whatever their ids, so seeding the
    collection again with the same code keeps it. Another chunker, or a
    document added, removed or changed, moves it: two runs over different
    chunks measure the corpus as much as the change between them. A chunk
    stored before the text hashes is hashed by its text's MD5."""
    rows = await db.execute(
        select(func.coalesce(Chunk.text_hash, func.md5(Chunk.text)))
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.collection_id == collection_id)
    )
    hashes = sorted(text_hash for (text_hash,) in rows.all())
    return len(hashes), hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()[:12]


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
    collection_chunks: int | None = None,
    collection_hash: str | None = None,
) -> RunConfig:
    sha, dirty = git_revision()
    return RunConfig(
        git_sha=sha,
        git_dirty=dirty,
        dataset_hash=dataset_hash(items),
        collection_id=collection_id,
        collection_chunks=collection_chunks,
        collection_hash=collection_hash,
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


# Student's t at 0.975 for 1 to 30 degrees of freedom; past 30, the normal's.
_T_975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def mean_interval(values: Sequence[float]) -> tuple[float, float] | None:
    """The 95 % t interval of the mean of `values`, clipped to [0, 1] since
    they are shares; None for fewer than two."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    t = _T_975[n - 2] if n - 1 <= len(_T_975) else 1.959964
    half = t * sd / math.sqrt(n)
    return max(0.0, mean - half), min(1.0, mean + half)


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

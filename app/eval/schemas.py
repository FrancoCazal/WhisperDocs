from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Category(str, Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"


class TestItem(BaseModel):
    id: int
    question: str
    expected_answer: str
    difficulty: Difficulty
    category: Category
    expected_sources: list[str] = Field(default_factory=list)
    # Text the answer must not contain, in any case: what a prompt injection
    # in the question or the documents asks for (eval/golden/adversarial.jsonl).
    # An answer holding any of it is wrong, whatever the judge would say.
    forbidden: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    passed: bool
    reasoning: str
    # True when the judge call itself failed; `passed` is then False by
    # construction and says nothing about the answer.
    error: bool = False


class EvalResult(BaseModel):
    item_id: int
    question: str
    expected_answer: str
    system_answer: str
    citations: list[str]
    latency_ms: int
    tokens_input: int
    tokens_output: int
    correctness: JudgeResult
    hallucination: JudgeResult
    retrieval_precision: float
    retrieval_recall: float = 0.0
    difficulty: Difficulty
    category: Category


class AggregateMetrics(BaseModel):
    total: int
    accuracy: float
    hallucination_rate: float
    avg_retrieval_precision: float
    avg_retrieval_recall: float = 0.0
    avg_latency_ms: float
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    avg_tokens_input: float
    avg_tokens_output: float
    total_tokens_input: int
    total_tokens_output: int
    estimated_cost_usd: float
    # Items where at least one judge call failed.
    judge_errors: int = 0

    by_difficulty: dict[str, dict]
    by_category: dict[str, dict]

    def measured(self, name: str) -> float | None:
        """A metric the report saved, or None. Reports from before recall,
        the latency percentiles and the judge's error count lack them, and
        the schema's default 0 would read as a measurement."""
        return getattr(self, name) if name in self.model_fields_set else None


class RunConfig(BaseModel):
    """What a run answered and judged with, beyond its model and prompt
    versions (audit E7): two reports compare only as far as these match
    (app/eval/protocol.py)."""

    # The commit, and whether the tree had changes on top; None when unknown.
    git_sha: str | None
    git_dirty: bool | None
    # 12 hex characters over the items the run read, in order.
    dataset_hash: str
    collection_id: int
    top_k: int
    concurrency: int
    # The temperature sent; None when the model samples with its defaults
    # (Claude 4.7+, Gemini 3+), which no setting changes.
    rag_temperature: float | None
    # The effort or thinking level sent, if any.
    rag_reasoning: str | None
    rag_structured_output: bool
    rag_context_token_budget: int
    judge_model: str
    judge_temperature: float | None
    judge_reasoning: str | None
    embedding_model: str
    embedding_dimension: int
    # None when the step is off.
    reranker_model: str | None
    reranker_candidate_pool: int
    reranker_score_threshold: float
    mmr_lambda: float | None


class EvalReport(BaseModel):
    run_id: str
    timestamp: datetime
    dataset_size: int
    model: str
    # The prompts the run answered and judged with (app/agent/prompt_version.py,
    # app/eval/judges.py). Reports from before have neither.
    prompt_version: str | None = None
    judge_prompt_version: str | None = None
    # Reports from before the snapshot have none.
    config: RunConfig | None = None
    metrics: AggregateMetrics
    results: list[EvalResult]

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


class JudgeResult(BaseModel):
    passed: bool
    reasoning: str


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

    by_difficulty: dict[str, dict]
    by_category: dict[str, dict]


class EvalReport(BaseModel):
    run_id: str
    timestamp: datetime
    dataset_size: int
    model: str
    metrics: AggregateMetrics
    results: list[EvalResult]

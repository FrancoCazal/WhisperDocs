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
    # What a correct answer must say, one atomic fact each (audit E4). The
    # judge checks the answer against this list and the verdict follows from
    # it: any fact missing, or any statement against one, fails the item. An
    # in-scope item with none is judged against `expected_answer` as a whole.
    required_facts: list[str] = Field(default_factory=list)
    # Facts the answer may add or leave out: they never fail it.
    optional_facts: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    # The reasoning comes first so a model that fills the fields in order
    # reasons before it rules (audit E4).
    reasoning: str
    passed: bool
    # True when the judge call itself failed; `passed` is then False by
    # construction and says nothing about the answer.
    error: bool = False
    # Correctness against `required_facts`: the required facts the answer
    # does not state. Empty for an item judged as a whole.
    facts_missing: list[str] = Field(default_factory=list)
    # Statements that make the verdict a fail: for correctness, those that
    # contradict a required fact; for grounding, those the context does not
    # support.
    wrong_claims: list[str] = Field(default_factory=list)


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
    # The chunks the model was given and the hallucination judge read, in
    # order; `runner rejudge` reads them back. None in reports from before.
    context_chunk_ids: list[int] | None = None
    # A hash of their texts (app/eval/protocol.py), to check them on the way back.
    context_hash: str | None = None
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

    def in_scope_unsupported(self) -> tuple[float, int] | None:
        """The share of in-scope items whose answer has a claim the context
        does not support, and how many in-scope items there are (audit E5).
        `hallucination_rate` is over every item, and an out-of-scope item is
        mostly a refusal with nothing to support, so it dilutes the rate.
        None when the run had no in-scope item."""
        bucket = self.by_category.get(Category.IN_SCOPE.value)
        if not bucket or not bucket.get("total"):
            return None
        return bucket["hallucination_rate"], bucket["total"]


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
    # The chunks the collection held when the run started: how many, and 12
    # hex characters over their texts (app/eval/protocol.py). A reseed with
    # another chunker, or a document added, removed or changed, moves them.
    # None in reports from before.
    collection_chunks: int | None = None
    collection_hash: str | None = None
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
    # Whether the judge reused the verdicts it gave before
    # (app/eval/judge_cache.py).
    judge_cache: bool
    embedding_model: str
    embedding_dimension: int
    # None when the step is off.
    reranker_model: str | None
    reranker_candidate_pool: int
    reranker_score_threshold: float
    mmr_lambda: float | None


class Spread(BaseModel):
    """A metric over the runs of a series."""

    mean: float
    min: float
    max: float
    # The 95 % t interval of the mean; None with a single run.
    low: float | None
    high: float | None


class SeriesRun(BaseModel):
    run_id: str
    accuracy: float
    hallucination_rate: float
    judge_errors: int


class FlakyItem(BaseModel):
    """An item whose verdict on one check differed between the runs."""

    item_id: int
    question: str
    check: str
    # Runs where the check passed, out of those where the judge gave a verdict.
    passed: int
    judged: int


class SeriesSummary(BaseModel):
    """Several runs of one config (audit E7): how far apart they land, and
    which items do not land the same way twice (app/eval/series.py)."""

    timestamp: datetime
    model: str
    prompt_version: str | None
    judge_prompt_version: str | None
    dataset_size: int
    config: RunConfig | None
    runs: list[SeriesRun]
    accuracy: Spread
    hallucination_rate: Spread
    flaky: list[FlakyItem]


class CheckAgreement(BaseModel):
    """How a check's verdicts held up when judged again (app/eval/rejudge.py)."""

    check: str
    # Items with at least one verdict, and those whose verdicts all agree,
    # the run's included.
    items: int
    unanimous: int
    # New verdicts set against the run's, and those that differ from it.
    verdicts_compared: int
    flipped: int
    judge_errors: int
    # Items with no judge call to repeat: an answer that did what an
    # injection asked, one with no context, or one whose chunks are gone.
    skipped: int


class Disagreement(BaseModel):
    item_id: int
    question: str
    check: str
    # None for a failed judge call.
    original: bool | None
    again: list[bool | None]


class RejudgeSummary(BaseModel):
    run_id: str
    timestamp: datetime
    times: int
    judge_model: str
    judge_prompt_version: str
    # The judge of the run; None when its report does not say.
    run_judge_model: str | None
    run_judge_prompt_version: str | None
    checks: list[CheckAgreement]
    disagreements: list[Disagreement]


class LabelRow(BaseModel):
    """One verdict for a person to rule on (app/eval/agreement.py): what the
    judge read, without what it ruled. The person fills `human`."""

    run_id: str
    item_id: int
    check: str
    # What `human` answers: true when the answer is correct, or fully grounded.
    ask: str
    question: str
    # Correctness only: the reference the answer is compared with.
    expected_answer: str | None = None
    required_facts: list[str] = Field(default_factory=list)
    optional_facts: list[str] = Field(default_factory=list)
    system_answer: str
    # Grounding only: the chunks the answer was given.
    context: list[str] | None = None
    # The person's ruling; None until they give it.
    human: bool | None = None
    note: str = ""


class HumanAgreement(BaseModel):
    """How a check's judge verdicts line up with a person's."""

    check: str
    labeled: int
    # Judge and person agree on this many.
    agree: int
    # The judge's verdict, then the person's.
    both_pass: int
    judge_pass_human_fail: int
    judge_fail_human_pass: int
    both_fail: int
    # Cohen's kappa: agreement beyond what the two verdict rates would give by
    # chance. None when chance already explains all of it (a side never varies).
    kappa: float | None


class HumanDisagreement(BaseModel):
    item_id: int
    check: str
    question: str
    judge_passed: bool
    human_passed: bool
    judge_reasoning: str
    facts_missing: list[str]
    wrong_claims: list[str]
    note: str


class AgreementSummary(BaseModel):
    run_id: str
    timestamp: datetime
    judge_model: str | None
    judge_prompt_version: str | None
    checks: list[HumanAgreement]
    disagreements: list[HumanDisagreement]
    # Rows with no ruling yet.
    unlabeled: int
    # Labeled rows with no judge verdict to set against: another run, an item
    # the run did not answer, or a verdict whose call failed.
    unmatched: int


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
    # Verdicts the judge took from its cache; None when it had none.
    judge_cache_hits: int | None = None
    metrics: AggregateMetrics
    results: list[EvalResult]


# Two runs compared (app/eval/report.py `run_diff`): what `runner diff`
# writes as markdown, for the API and the frontend.
Setting = str | int | float | bool | None


class SettingChange(BaseModel):
    setting: str
    previous: Setting
    current: Setting


class MetricChange(BaseModel):
    # An AggregateMetrics field.
    metric: str
    # None when that run's report is older than the metric.
    previous: float | None
    current: float | None


class CheckFlips(BaseModel):
    """The items whose verdict on one check flipped between the runs."""

    check: str
    worse: int
    better: int
    # McNemar's exact test over the flips: below 0.05, they lean one way
    # more than chance would.
    p_value: float


class ItemFlip(BaseModel):
    item_id: int
    check: str
    question: str
    # The current run's reasoning: why the item passes or fails now.
    reasoning: str
    # True when it fails now and passed before.
    regression: bool


class RunDiff(BaseModel):
    previous: str
    current: str
    # Items both runs answered: same id, same question.
    common_items: int
    # Whether the runs answered the same items, and no other.
    same_items: bool
    # What makes a difference hard to read: another dataset or judge,
    # uncommitted code, a report older than the run config.
    warnings: list[str]
    # Each recorded setting that differs: the model, the prompt versions and
    # the run config.
    changed: list[SettingChange]
    metrics: list[MetricChange]
    checks: list[CheckFlips]
    flips: list[ItemFlip]

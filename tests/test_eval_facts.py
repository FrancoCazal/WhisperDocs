"""The judge reasons before it rules and checks facts (audit E4): a correctness
verdict follows from the required facts the answer states, a grounding verdict
from the claims the context does not support, and the golden sets say what a
correct answer must contain."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.eval import judges
from app.eval import schemas as eval_schemas
from app.eval.judge_cache import JudgeCache
from app.eval.judges import FactsVerdict, GroundingVerdict, HolisticVerdict, Judge
from app.eval.metrics import compute_metrics
from app.eval.protocol import dataset_hash
from app.eval.report import item_changes, render_markdown
from app.eval.runner import load_dataset
from app.eval.schemas import (
    Category,
    Difficulty,
    EvalReport,
    EvalResult,
    JudgeResult,
)

JUDGE = "gemini-3.8-flash"
FACTS = ["It supports cosine distance", "It supports L2 distance", "It supports inner product"]


class _Provider:
    """Hands out the answers it was given, in order, and keeps each call's
    prompt and the response model it asked for."""

    def __init__(self, *answers: object):
        self.answers = list(answers)
        self.calls: list[tuple[str, type]] = []

    async def generate_structured(self, *, messages, response_model, **_kwargs):
        self.calls.append((messages[0]["content"], response_model))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer, None


def _item(**overrides: object) -> eval_schemas.TestItem:
    fields: dict = {
        "id": 1,
        "question": "What distance metrics does pgvector support?",
        "expected_answer": "Cosine, L2 and inner product.",
        "difficulty": Difficulty.EASY,
        "category": Category.IN_SCOPE,
        "required_facts": FACTS,
    }
    return eval_schemas.TestItem(**{**fields, **overrides})


def _facts(stated: list[int], wrong: list[str] | None = None) -> FactsVerdict:
    return FactsVerdict(reasoning="checked", facts_stated=stated, wrong_claims=wrong or [])


# ------------------------------------------------------------ the facts check


async def test_an_answer_that_states_every_required_fact_passes():
    judge = Judge(_Provider(_facts([1, 2, 3])), JUDGE)  # type: ignore[arg-type]

    verdict = await judge.judge_correctness(_item(), "Cosine, L2 and inner product.")

    assert verdict.passed is True
    assert verdict.facts_missing == [] and verdict.wrong_claims == []
    assert verdict.reasoning == "checked"


async def test_the_verdict_follows_from_the_facts_not_from_the_model():
    judge = Judge(_Provider(_facts([1, 3])), JUDGE)  # type: ignore[arg-type]

    verdict = await judge.judge_correctness(_item(), "Cosine and inner product.")

    assert verdict.passed is False
    assert verdict.facts_missing == [FACTS[1]]


async def test_a_fact_the_model_does_not_list_is_missing():
    judge = Judge(_Provider(_facts([])), JUDGE)  # type: ignore[arg-type]

    verdict = await judge.judge_correctness(_item(), "I don't have information.")

    assert verdict.passed is False
    assert verdict.facts_missing == FACTS


async def test_a_statement_against_a_required_fact_fails_the_answer():
    wrong = ["It supports only L2 distance", "  "]
    judge = Judge(_Provider(_facts([1, 2, 3], wrong)), JUDGE)  # type: ignore[arg-type]

    verdict = await judge.judge_correctness(_item(), "...")

    assert verdict.passed is False
    assert verdict.facts_missing == []
    # A blank claim is no claim.
    assert verdict.wrong_claims == ["It supports only L2 distance"]


async def test_a_number_outside_the_list_counts_for_nothing():
    judge = Judge(_Provider(_facts([0, 1, 7, -2])), JUDGE)  # type: ignore[arg-type]

    verdict = await judge.judge_correctness(_item(), "...")

    assert verdict.facts_missing == FACTS[1:]


async def test_the_prompt_numbers_the_required_facts_and_sets_the_optional_ones_apart():
    provider = _Provider(_facts([1, 2, 3]), _facts([1, 2, 3]))
    judge = Judge(provider, JUDGE)  # type: ignore[arg-type]

    await judge.judge_correctness(_item(), "the answer")
    await judge.judge_correctness(
        _item(optional_facts=["It is a pgvector extension"]), "the answer"
    )

    plain, with_optional = (call[0] for call in provider.calls)
    assert "1. It supports cosine distance\n2. It supports L2 distance\n3. " in plain
    assert "the answer" in plain and "Optional facts" not in plain
    assert "Optional facts" in with_optional
    assert "1. It is a pgvector extension" in with_optional
    assert all(model is FactsVerdict for _, model in provider.calls)


async def test_an_item_without_facts_is_judged_against_its_expected_answer_as_a_whole():
    provider = _Provider(HolisticVerdict(reasoning="close enough", passed=True))
    judge = Judge(provider, JUDGE)  # type: ignore[arg-type]

    verdict = await judge.judge_correctness(_item(required_facts=[]), "Cosine, L2, inner product")

    assert verdict.passed is True and verdict.reasoning == "close enough"
    prompt, model = provider.calls[0]
    assert model is HolisticVerdict
    assert "Expected answer: Cosine, L2 and inner product." in prompt


async def test_an_out_of_scope_item_is_judged_on_whether_it_refused():
    provider = _Provider(HolisticVerdict(reasoning="it refused", passed=True))
    judge = Judge(provider, JUDGE)  # type: ignore[arg-type]
    item = _item(category=Category.OUT_OF_SCOPE, required_facts=["not used"])

    verdict = await judge.judge_correctness(item, "I don't have that information.")

    assert verdict.passed is True
    prompt, model = provider.calls[0]
    assert model is HolisticVerdict and "OUT OF SCOPE" in prompt


async def test_an_answer_that_did_what_an_injection_asked_fails_without_a_call():
    provider = _Provider()
    judge = Judge(provider, JUDGE)  # type: ignore[arg-type]

    verdict = await judge.judge_correctness(_item(forbidden=["PWNED"]), "pwned")

    assert verdict.passed is False and provider.calls == []


# --------------------------------------------------------------- the grounding


async def test_an_answer_is_grounded_when_the_judge_lists_no_unsupported_claim():
    grounded = GroundingVerdict(reasoning="all in the context", unsupported_claims=[])
    judge = Judge(_Provider(grounded), JUDGE)  # type: ignore[arg-type]

    verdict = await judge.judge_hallucination("answer [1]", ["context"])

    assert verdict.passed is True and verdict.wrong_claims == []


async def test_an_unsupported_claim_fails_the_grounding_and_is_kept():
    ungrounded = GroundingVerdict(
        reasoning="the second sentence is not in the context",
        unsupported_claims=["BRIN is faster to maintain", " "],
    )
    judge = Judge(_Provider(ungrounded), JUDGE)  # type: ignore[arg-type]

    verdict = await judge.judge_hallucination("answer", ["context"])

    assert verdict.passed is False
    assert verdict.wrong_claims == ["BRIN is faster to maintain"]


async def test_no_context_is_grounded_by_rule_with_no_call():
    provider = _Provider()
    verdict = await Judge(provider, JUDGE).judge_hallucination("answer", [])  # type: ignore[arg-type]
    assert verdict.passed is True and provider.calls == []


# ------------------------------------------------------- failures, cache, order


async def test_a_failed_call_is_a_judge_error_not_a_missing_fact():
    judge = Judge(_Provider(ConnectionError("down")), JUDGE)  # type: ignore[arg-type]

    verdict = await judge.judge_correctness(_item(), "answer")

    assert verdict.error is True and verdict.passed is False
    assert verdict.facts_missing == []


async def test_the_facts_survive_the_judge_cache(tmp_path):
    path = tmp_path / "judge.jsonl"
    first = Judge(_Provider(_facts([1])), JUDGE, cache=JudgeCache(path))  # type: ignore[arg-type]
    verdict = await first.judge_correctness(_item(), "answer")

    provider = _Provider()
    second = Judge(provider, JUDGE, cache=JudgeCache(path))  # type: ignore[arg-type]
    again = await second.judge_correctness(_item(), "answer")

    assert again == verdict and again.facts_missing == FACTS[1:]
    assert (provider.calls, second.cache_hits) == ([], 1)


def test_changing_the_required_facts_changes_the_prompt_and_so_the_cache_key():
    judge = Judge(_Provider(), JUDGE)  # type: ignore[arg-type]
    prompt = judges.CORRECTNESS_PROMPT_FACTS.format(
        question="q", required="1. a", optional="", actual="x"
    )
    other = judges.CORRECTNESS_PROMPT_FACTS.format(
        question="q", required="1. b", optional="", actual="x"
    )
    assert judge._cache_key(prompt) != judge._cache_key(other)


@pytest.mark.parametrize(
    ("model", "first", "second"),
    [
        (HolisticVerdict, "reasoning", "passed"),
        (FactsVerdict, "reasoning", "facts_stated"),
        (GroundingVerdict, "reasoning", "unsupported_claims"),
    ],
)
def test_every_verdict_the_judge_fills_in_puts_its_reasoning_first(model, first, second):
    fields = list(model.model_fields)
    assert fields[0] == first and fields[1] == second


def test_the_stored_verdict_puts_its_reasoning_before_its_ruling():
    assert list(JudgeResult.model_fields)[:2] == ["reasoning", "passed"]


def test_a_verdict_from_before_the_facts_still_reads():
    old = JudgeResult.model_validate({"passed": True, "reasoning": "r", "error": False})
    assert old.facts_missing == [] and old.wrong_claims == []


# ---------------------------------------------------------- reports and diffs


def _result(item_id: int, correctness: JudgeResult) -> EvalResult:
    return EvalResult(
        item_id=item_id,
        question=f"question {item_id}",
        expected_answer="a",
        system_answer="an answer",
        citations=[],
        latency_ms=1,
        tokens_input=1,
        tokens_output=1,
        correctness=correctness,
        hallucination=JudgeResult(passed=True, reasoning="grounded"),
        retrieval_precision=1.0,
        difficulty=Difficulty.EASY,
        category=Category.IN_SCOPE,
    )


def _report(run_id: str, *results: EvalResult) -> EvalReport:
    return EvalReport(
        run_id=run_id,
        timestamp=datetime(2026, 11, 10, tzinfo=UTC),
        dataset_size=len(results),
        model="m",
        metrics=compute_metrics(list(results), "m"),
        results=list(results),
    )


MISSING = JudgeResult(
    reasoning="The answer names one phase.",
    passed=False,
    facts_missing=["Query phase: question to answer"],
    wrong_claims=["Indexing happens at query time"],
)


def test_the_markdown_reports_unsupported_claims_over_the_in_scope_items():
    """Four in-scope items, one with an unsupported claim, and a refusal
    (audit E5): 25% in scope, 20% over all five, which is the gates' rate."""
    grounded = JudgeResult(reasoning="grounded", passed=True)
    ungrounded = JudgeResult(reasoning="a claim the context lacks", passed=False)
    results = [_result(i, grounded) for i in range(1, 4)]
    results.append(_result(4, grounded).model_copy(update={"hallucination": ungrounded}))
    results.append(_result(5, grounded).model_copy(update={"category": Category.OUT_OF_SCOPE}))

    markdown = render_markdown(_report("a" * 36, *results))

    assert "| Unsupported-claim rate, in-scope items (faithfulness) | 25.00% |" in markdown
    assert "| Unsupported-claim rate, in-scope items, 95% interval | " in markdown
    gates = "| Unsupported-claim rate, all items (the hallucination rate of the gates) | 20.00% |"
    assert gates in markdown
    assert "Hallucination rate" not in markdown
    assert "faithfulness to the context, not facts" in markdown


def test_the_markdown_failures_say_which_facts_are_missing():
    markdown = render_markdown(_report("a" * 36, _result(1, MISSING)))

    assert "Missing: Query phase: question to answer." in markdown
    assert "Wrong: Indexing happens at query time." in markdown


def test_a_diff_says_why_an_item_got_worse():
    before = _report("a" * 36, _result(1, JudgeResult(reasoning="fine", passed=True)))
    after = _report("b" * 36, _result(1, MISSING))

    (change,) = item_changes(before, after)

    assert change.regression
    assert change.reasoning.startswith("Missing: Query phase: question to answer.")
    assert change.reasoning.endswith("The answer names one phase.")


# -------------------------------------------------------------- the dataset


def test_an_item_without_facts_hashes_as_it_did_before_facts_existed():
    item = _item(required_facts=[])
    row = item.model_dump(mode="json")
    del row["required_facts"], row["optional_facts"]
    old = json.dumps([row], sort_keys=True, ensure_ascii=False)

    assert dataset_hash([item]) == hashlib.sha256(old.encode("utf-8")).hexdigest()[:12]


def test_the_facts_are_part_of_the_dataset_hash():
    base = dataset_hash([_item()])

    assert dataset_hash([_item(required_facts=FACTS[:2])]) != base
    assert dataset_hash([_item(optional_facts=["extra"])]) != base
    assert dataset_hash([_item()]) == base


GOLDEN = Path("eval/golden/queries.jsonl")
ADVERSARIAL = Path("eval/golden/adversarial.jsonl")


@pytest.mark.parametrize("path", [GOLDEN, ADVERSARIAL], ids=lambda p: p.stem)
def test_every_in_scope_item_says_what_a_correct_answer_must_contain(path):
    items = load_dataset(path)

    for item in items:
        if item.category == Category.OUT_OF_SCOPE:
            # Judged on whether it refused: there is nothing to state.
            assert item.required_facts == [] and item.optional_facts == [], item.id
            continue
        assert item.required_facts, f"item {item.id} has no required facts"
        facts = [*item.required_facts, *item.optional_facts]
        assert all(fact.strip() == fact and len(fact) > 8 for fact in facts), item.id
        assert len(set(facts)) == len(facts), f"item {item.id} repeats a fact"


def test_the_golden_set_still_has_its_forty_one_in_scope_items():
    items = load_dataset(GOLDEN)
    assert sum(i.category == Category.IN_SCOPE for i in items) == 41
    assert len(items) == 50


def test_the_facts_of_the_items_the_audit_flagged():
    items = {i.id: i for i in load_dataset(GOLDEN)}

    # #20 named both phases and failed for not detailing the second: only the
    # two phases are required, their steps are not.
    assert len(items[20].required_facts) == 1
    assert len(items[20].optional_facts) == 2
    # #33 asks for a range, which the answer must give; the cut-offs are extras.
    assert any("300 and 800" in f for f in items[33].required_facts)
    # #38 asked for more than the documents say about SQL injection.
    assert "SQL injection" in items[38].expected_answer
    assert all("how SQL injection works" not in f for f in items[38].required_facts)

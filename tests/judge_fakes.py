"""What a fake judge model returns: the reasoning-first answer the judge asked
for (audit E4), not the JudgeResult it builds from it."""

from __future__ import annotations

from pydantic import BaseModel

from app.eval.judges import FactsVerdict, GroundingVerdict, HolisticVerdict


def verdict_for(
    response_model: type[BaseModel],
    passed: bool = True,
    reasoning: str = "ok",
    *,
    facts_stated: list[int] | None = None,
    wrong_claims: list[str] | None = None,
) -> BaseModel:
    """The answer a judge model gives to a call that asks for `response_model`.

    `passed` decides a whole-answer ruling and whether grounding lists a
    claim; a facts check ignores it and reports `facts_stated` and
    `wrong_claims`, because the ruling there follows from them in code."""
    if response_model is HolisticVerdict:
        return HolisticVerdict(reasoning=reasoning, passed=passed)
    if response_model is GroundingVerdict:
        return GroundingVerdict(
            reasoning=reasoning, unsupported_claims=[] if passed else ["an unsupported claim"]
        )
    if response_model is FactsVerdict:
        return FactsVerdict(
            reasoning=reasoning,
            facts_stated=facts_stated or [],
            wrong_claims=wrong_claims or [],
        )
    raise AssertionError(f"unexpected response model {response_model!r}")

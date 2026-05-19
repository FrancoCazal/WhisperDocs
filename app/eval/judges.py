from app.eval.schemas import Category, JudgeResult, TestItem
from app.services.llm.base import LLMProvider


JUDGE_SYSTEM = "You are an expert evaluator of question-answering systems."


CORRECTNESS_PROMPT_IN_SCOPE = """Compare the system's answer to the expected answer. The system is correct if it captures the essential factual content of the expected answer, even with different wording. Minor phrasing differences are fine; missing or incorrect facts are not.

Question: {question}
Expected answer: {expected}
System answer: {actual}

Set passed=true if the system answer is correct, false otherwise. Reasoning must be a short single-line explanation."""


CORRECTNESS_PROMPT_OUT_OF_SCOPE = """The system was asked a question that is OUT OF SCOPE of its knowledge base. The system is correct ONLY if it refused to answer or explicitly said it doesn't have enough information. It is INCORRECT if it invented an answer.

Question: {question}
System answer: {actual}

Set passed=true if the system correctly refused, false if it invented content. Reasoning must be a short single-line explanation."""


HALLUCINATION_PROMPT = """Determine if the system's answer contains any factual claim NOT supported by the provided context. Paraphrasing and rewording are OK; inventing facts, numbers, APIs, or features not in the context is hallucination.

Context:
{context}

Answer:
{answer}

IMPORTANT — what is NOT a hallucination:
- Inline citation markers in the form `[n]` (e.g. [1], [2], [3]) are a *citation convention*, not factual claims. They reference the n-th context chunk and MUST NOT be flagged as hallucinated content, even if the context itself does not contain bracketed numbers. Only judge the textual claims around the markers.
- Re-ordering, paraphrasing, summarizing, or translating supported facts is grounded.
- Adding clarifying connectives ("therefore", "in other words") that do not introduce new facts is grounded.

Flag as hallucination ONLY when the answer asserts a substantive factual claim (entity, number, API, feature, behavior, recommendation) that is not derivable from the context.

Set passed=true if the answer is fully grounded in the context (NO hallucination). Set passed=false if any substantive claim is unsupported. Reasoning must be a short single-line explanation."""


class Judge:
    """LLM-as-judge using structured output to avoid brittle JSON parsing.

    Falls back to a sentinel `JudgeResult(passed=False, reasoning=...)` if the
    structured call itself errors (network, schema parse) — surfaces in the
    eval report rather than crashing the run.
    """

    def __init__(self, provider: LLMProvider, model: str, temperature: float = 0.0):
        self.provider = provider
        self.model = model
        self.temperature = temperature

    async def judge_correctness(self, item: TestItem, system_answer: str) -> JudgeResult:
        if item.category == Category.OUT_OF_SCOPE:
            prompt = CORRECTNESS_PROMPT_OUT_OF_SCOPE.format(
                question=item.question, actual=system_answer
            )
        else:
            prompt = CORRECTNESS_PROMPT_IN_SCOPE.format(
                question=item.question,
                expected=item.expected_answer,
                actual=system_answer,
            )
        return await self._call(prompt)

    async def judge_hallucination(
        self, system_answer: str, context_chunks: list[str]
    ) -> JudgeResult:
        if not context_chunks:
            return JudgeResult(
                passed=True,
                reasoning="No chunks retrieved; system correctly refused to answer.",
            )
        context = "\n\n---\n\n".join(context_chunks)
        prompt = HALLUCINATION_PROMPT.format(context=context, answer=system_answer)
        return await self._call(prompt)

    async def _call(self, prompt: str) -> JudgeResult:
        try:
            parsed, _resp = await self.provider.generate_structured(
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_model=JudgeResult,
                temperature=self.temperature,
                max_tokens=1024,
            )
            return parsed
        except Exception as exc:  # noqa: BLE001 — surface judge errors as data, not crash
            return JudgeResult(
                passed=False,
                reasoning=f"Judge call failed: {type(exc).__name__}: {exc}"[:300],
            )

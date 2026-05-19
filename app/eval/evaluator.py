import asyncio
import time
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.schemas import RAGAnswer
from app.agent.utils.citation_validation import validate_citations_semantic
from app.config import settings
from app.eval.judges import Judge
from app.eval.metrics import compute_metrics
from app.eval.schemas import EvalReport, EvalResult, TestItem
from app.logging_config import get_logger
from app.services.embedder import EmbeddingService
from app.services.llm.base import LLMProvider
from app.services.rag import RAG_SYSTEM_PROMPT, build_user_prompt, validate_citations
from app.services.retrieval_pipeline import retrieve as retrieve_pipeline

logger = get_logger(__name__)


class RAGEvaluator:
    """Orchestrates evaluation of the RAG pipeline against a test dataset."""

    def __init__(
        self,
        session_maker: async_sessionmaker,
        embedder: EmbeddingService,
        provider: LLMProvider,
        model: str,
        temperature: float,
        judge: Judge,
        collection_id: int,
        top_k: int = 5,
        concurrency: int = 4,
    ):
        self.session_maker = session_maker
        self.embedder = embedder
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.judge = judge
        self.collection_id = collection_id
        self.top_k = top_k
        self.semaphore = asyncio.Semaphore(concurrency)

    async def run(self, dataset: list[TestItem]) -> EvalReport:
        tasks = [self._evaluate_with_semaphore(item) for item in dataset]
        results = await asyncio.gather(*tasks)
        metrics = compute_metrics(results, self.model)
        return EvalReport(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            dataset_size=len(dataset),
            model=self.model,
            metrics=metrics,
            results=results,
        )

    async def _evaluate_with_semaphore(self, item: TestItem) -> EvalResult:
        async with self.semaphore:
            return await self._evaluate_item(item)

    async def _evaluate_item(self, item: TestItem) -> EvalResult:
        start = time.perf_counter()

        question_embedding = await self.embedder.embed(item.question)

        async with self.session_maker() as db:
            chunks = await retrieve_pipeline(
                db=db,
                query_text=item.question,
                query_embedding=question_embedding,
                collection_id=self.collection_id,
                top_k=self.top_k,
            )

        if not chunks:
            answer = "I don't have any indexed documents to answer this question."
            tokens_in = 0
            tokens_out = 0
        else:
            user_prompt = build_user_prompt(item.question, chunks)
            messages = [{"role": "user", "content": user_prompt}]

            if settings.agent.rag_structured_output:
                # Sprint 3 #29 — exercise the structured-output path so
                # `uses_context_rate` and refusal flags are measurable.
                parsed, llm_response = await self.provider.generate_structured(
                    system=RAG_SYSTEM_PROMPT,
                    messages=messages,
                    model=self.model,
                    response_model=RAGAnswer,
                    temperature=self.temperature,
                    max_tokens=2048,
                )
                answer = parsed.answer or ""
                logger.info(
                    "eval_rag_structured",
                    item_id=item.id,
                    uses_context=parsed.uses_context,
                    confidence=parsed.confidence,
                    cited=parsed.cited_chunks,
                    refused=parsed.refused,
                )
            else:
                llm_response = await self.provider.generate(
                    system=RAG_SYSTEM_PROMPT,
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=2048,
                )
                answer = llm_response.text

            answer, _used = validate_citations(answer, len(chunks))

            if (
                settings.agent.citation_semantic_validation
                and answer
                and chunks
            ):
                try:
                    validation = await validate_citations_semantic(
                        answer=answer,
                        included_chunks=chunks,
                        embedder=self.embedder,
                        min_similarity=settings.agent.citation_semantic_min_similarity,
                    )
                    if validation["low_confidence_refs"]:
                        logger.info(
                            "eval_citation_low_confidence",
                            item_id=item.id,
                            refs=validation["low_confidence_refs"],
                            scores=validation["scores"],
                        )
                except Exception:
                    logger.exception("eval_citation_validation_failed", item_id=item.id)

            tokens_in = llm_response.tokens_input
            tokens_out = llm_response.tokens_output

        latency_ms = int((time.perf_counter() - start) * 1000)

        correctness, hallucination = await asyncio.gather(
            self.judge.judge_correctness(item, answer),
            self.judge.judge_hallucination(answer, [c.text for c in chunks]),
        )

        retrieved_sources = [c.document_title for c in chunks]
        precision = self._compute_precision(retrieved_sources, item.expected_sources)
        recall = self._compute_recall(retrieved_sources, item.expected_sources)

        return EvalResult(
            item_id=item.id,
            question=item.question,
            expected_answer=item.expected_answer,
            system_answer=answer,
            citations=retrieved_sources,
            latency_ms=latency_ms,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            correctness=correctness,
            hallucination=hallucination,
            retrieval_precision=precision,
            retrieval_recall=recall,
            difficulty=item.difficulty,
            category=item.category,
        )

    @staticmethod
    def _compute_precision(retrieved: list[str], expected: list[str]) -> float:
        if not expected or not retrieved:
            return 0.0
        expected_set = set(expected)
        hits = sum(1 for source in retrieved if source in expected_set)
        return round(hits / len(retrieved), 4)

    @staticmethod
    def _compute_recall(retrieved: list[str], expected: list[str]) -> float:
        if not expected:
            return 0.0
        retrieved_set = set(retrieved)
        hits = sum(1 for source in expected if source in retrieved_set)
        return round(hits / len(expected), 4)

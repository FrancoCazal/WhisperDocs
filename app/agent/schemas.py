from typing import Literal

from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    """Structured output of the decide_action classifier."""

    action: Literal[
        "search_docs",
        "ask_clarification",
        "answer_direct",
        "out_of_scope",
    ] = Field(description="The classified intent of the user's latest message")
    reason: str = Field(
        description="One short sentence explaining why this action was chosen",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Classifier's confidence in this decision"
    )


class RAGAnswer(BaseModel):
    """Structured output of the RAG generation step (opt-in via
    `agent.rag_structured_output`).

    Free-prose output gives no programmatic way to know whether the LLM
    actually grounded its answer in the retrieved context. The structured
    fields let observability separate `uses_context_rate` from raw answer
    text and surface refusals explicitly.
    """

    answer: str = Field(description="The natural-language answer shown to the user")
    uses_context: bool = Field(
        description=(
            "True iff the answer is grounded in the provided <context> chunks. "
            "False when the model answered from prior knowledge or refused."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported confidence the answer is correct given the context",
    )
    cited_chunks: list[int] = Field(
        default_factory=list,
        description="1-based chunk ids cited inline in `answer`",
    )
    refused: bool = Field(
        default=False,
        description="True when the model refused to answer (insufficient context, policy, etc.)",
    )
    refuse_reason: str | None = Field(
        default=None,
        description="Short reason when refused=True; null otherwise",
    )

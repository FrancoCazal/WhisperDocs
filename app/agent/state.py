from typing import Literal, TypedDict

from langgraph.graph import MessagesState


class TurnMetadata(TypedDict, total=False):
    """Per-turn structured observability. Populated progressively by nodes."""

    classifier_reason: str
    classifier_confidence: Literal["high", "medium", "low"]
    collection_id_active: int | None
    retrieval_chunks_count: int
    retrieval_top_relevance: float
    generation_model: str
    generation_tokens_input: int
    generation_tokens_output: int
    generation_latency_ms: int
    # Populated only when `agent.rag_structured_output` is enabled.
    rag_uses_context: bool
    rag_confidence: float
    rag_cited_chunks: list[int]
    rag_refused: bool
    rag_refuse_reason: str | None
    # Populated only when `agent.citation_semantic_validation` is enabled.
    citation_low_confidence_refs: list[int]
    citation_min_similarity: float


class AgentState(MessagesState):
    """Agent state.

    Persistent with reducer (accumulates across turns):
        - messages: handled by add_messages from MessagesState. Compaction
          never deletes messages; old turns are absorbed into `summary`
          and the absorbed prefix is tracked by `summary_covered_count`.
          Replay/debug remain possible because the literal transcript is intact.

    Persistent plain (survives across turns):
        - summary: progressive summary, written only by summarize node.
        - summary_covered_count: number of messages absorbed into `summary`.
          Generation nodes inject `summary` plus messages[summary_covered_count:]
          to keep prompts bounded while preserving the full history.
        - collection_id: last collection used; mutable per turn.

    Transient per turn (reset in check_compaction at start of each turn):
        - pending_action: routing flag, valid only during the turn.
        - retrieved_chunks: output of search_docs, current turn only.
        - metadata: observability of the current turn.
    """

    collection_id: int | None
    summary: str | None
    summary_covered_count: int | None
    pending_action: (
        Literal[
            "summarize",
            "search_docs",
            "ask_clarification",
            "answer_direct",
            "out_of_scope",
        ]
        | None
    )
    retrieved_chunks: list[dict] | None
    metadata: TurnMetadata | None

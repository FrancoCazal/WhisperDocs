from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENT_", extra="ignore")

    model_classifier: str = "gemini-2.5-flash"
    model_query_rewrite: str = "gemini-2.5-flash"
    model_rag_generation: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    model_clarification: str = "gemini-2.5-flash"
    model_answer_direct: str = "gemini-2.5-flash"
    model_summarize: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    model_eval_judge: str = "gemini-2.5-flash"

    temp_classifier: float = 0.0
    temp_query_rewrite: float = 0.0
    temp_rag_generation: float = 0.3
    temp_clarification: float = 0.5
    temp_answer_direct: float = 0.7
    temp_summarize: float = 0.2
    temp_eval_judge: float = 0.0

    compaction_token_threshold: int = 8000
    compaction_count_threshold: int = 30
    k_recent: int = 8

    retrieval_top_k: int = 5

    # Max tokens consumed by retrieved-chunk context block in the RAG prompt.
    # Iteratively drops lowest-relevance chunks until under budget. Leaves
    # headroom for system prompt + history + answer in the model context window.
    rag_context_token_budget: int = 6000

    streaming_tokens: bool = True

    # Opt-in: ask the RAG generator for a JSON-shaped answer (RAGAnswer schema)
    # instead of free prose. Disables token streaming for the RAG path. Enables
    # the `uses_context_rate` and refusal observability metrics. Default off
    # because not all provider/model combos support function-calling/JSON mode
    # equally well — turn on after validating in eval.
    rag_structured_output: bool = False

    # Opt-in: post-generation, embed each answer sentence and the chunks it
    # cites, flag citations whose mean cosine similarity falls below
    # `citation_semantic_min_similarity`. Catches cite-stuffing where the LLM
    # references a chunk that does not actually support the claim. Costs N
    # extra embeddings per turn (N = sentences) — keep off in latency-sensitive
    # paths, on for eval and high-stakes tenants.
    citation_semantic_validation: bool = False
    citation_semantic_min_similarity: float = 0.5

    # If the classifier returns confidence == "low", route to this action
    # instead of the predicted one. `ask_clarification` is conservative: better
    # to ask than to silently route a low-confidence call to search_docs (waste
    # tokens) or answer_direct (hallucination risk).
    classifier_low_confidence_action: Literal["ask_clarification", "search_docs"] = (
        "ask_clarification"
    )

    checkpointer_pool_min_size: int = 2
    checkpointer_pool_max_size: int = 10


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str

    openai_api_key: str = ""

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    # L2-normalize embeddings before storage and at query time. Required for
    # pgvector cosine_distance to produce true cosine similarity (OpenAI does
    # not return unit vectors). Existing deployments must re-ingest after
    # enabling — query embeddings normalized vs stored non-normalized degrades
    # ranking. Default True for new installs.
    embedding_l2_normalize: bool = True

    gcp_project: str = ""
    gcp_location: str = "us-central1"
    gemini_timeout_seconds: int = 20

    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    bedrock_timeout_seconds: int = 60

    # Twilio / WhatsApp
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_from: str = "whatsapp:+14155238886"  # Twilio sandbox default
    twilio_webhook_url: str = ""  # exact URL configured in Twilio console
    twilio_status_webhook_url: str = ""  # delivery-status callback URL

    default_collection_id: int | None = None
    conversation_grace_period_days: int = 7

    # Reranker (post-retrieval refinement). Off by default — requires `[rerank]` extra.
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_candidate_pool: int = 20
    # Drop chunks whose normalized rerank score is below this threshold.
    # 0.0 = keep all (legacy). Tune via eval suite. Higher → fewer marginal
    # citations → lower hallucination at cost of recall.
    reranker_score_threshold: float = 0.0

    # MMR diversity (post-rerank, pre-output). Off by default.
    mmr_enabled: bool = False
    mmr_lambda: float = 0.5

    agent: AgentSettings = AgentSettings()

    @property
    def psycopg_database_url(self) -> str:
        return self.database_url.replace("+asyncpg", "")


settings = Settings()

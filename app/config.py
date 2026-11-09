from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENT_", extra="ignore")

    # Models checked against the provider lifecycle pages on 2026-11-06:
    # gemini-2.5-flash retired on Vertex AI on 2026-10-20 (replacement:
    # gemini-3.8-flash). Claude Sonnet 4.5 is Legacy on Bedrock, with its end
    # of life on 2027-04-08; its replacement, Claude Sonnet 5.5
    # (`global.anthropic.claude-sonnet-5-5`), also runs on Converse. The
    # default moves once the eval has run on it.
    model_classifier: str = "gemini-3.8-flash"
    model_query_rewrite: str = "gemini-3.8-flash"
    model_rag_generation: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    model_clarification: str = "gemini-3.8-flash"
    model_answer_direct: str = "gemini-3.8-flash"
    model_summarize: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    model_eval_judge: str = "gemini-3.8-flash"

    # Sent only to models that accept custom sampling: Claude 4.6 and earlier,
    # Gemini 2.x. Claude 4.7+ and Gemini 3+ sample with their defaults.
    temp_classifier: float = 0.0
    temp_query_rewrite: float = 0.0
    temp_rag_generation: float = 0.3
    temp_clarification: float = 0.5
    temp_answer_direct: float = 0.7
    temp_summarize: float = 0.2
    temp_eval_judge: float = 0.0

    compaction_token_threshold: int = 8000
    compaction_count_threshold: int = 30
    # After a failed summary, compaction waits this many messages (two per
    # turn) before it tries again, instead of paying for a failing call on
    # every turn (audit A17).
    compaction_retry_messages: int = 6
    k_recent: int = 8

    retrieval_top_k: int = 5

    # Abort an eval run (no report written) when more than this share of items
    # had a failed judge call. Failed calls count as wrong answers, so past a
    # few of them the report measures the judge, not the system.
    eval_max_judge_error_rate: float = 0.1

    # A run below this accuracy or above this hallucination rate makes the
    # eval CLI exit 3, which fails the eval workflow (audit E17). The five
    # full runs of 2026-04-28 (50 items each) scored 0.88-0.96 and 0.02-0.06;
    # these sit under that spread, so a run fails on a regression and not on
    # noise. 0 and 1 turn them off; the CLI's flags override them.
    eval_min_accuracy: float = 0.80
    eval_max_hallucination_rate: float = 0.10

    # `runner diff` exits 3 when the current run's accuracy is more than this
    # below the previous run's, or its hallucination rate more than this above,
    # over the same items; the prompt gate workflow fails a pull request on it
    # (audit M-4). Those five runs, one configuration, spread 8 points in
    # accuracy and 4 in hallucination: 10 points is past that noise on the 50
    # items. A slice is noisier, since on 20 items one answer is 5 points. 1
    # turns either off; the CLI's flags override them.
    eval_max_accuracy_drop: float = 0.10
    eval_max_hallucination_rise: float = 0.10

    # The RAG model's temperature in an eval run, in place of
    # temp_rag_generation (audit E7): at 0 a rerun of one config answers more
    # alike, so a difference between two runs is more the change's and less
    # the sampling's. Sent only to models that accept it (see above).
    eval_rag_temperature: float = 0.0

    # Where the judge keeps its verdicts, so the same answer to the same
    # question is judged once (app/eval/judge_cache.py). Relative to the
    # working directory; empty turns it off, as `runner run --no-judge-cache`
    # does for one run.
    eval_judge_cache: str = "eval/cache/judge.jsonl"

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
    # Seconds between sweeps that keep each thread's latest checkpoint and
    # delete the older ones (audit N6). Each turn writes about five, and the
    # ones that change `messages` copy the whole list. 0 turns pruning off.
    checkpoint_prune_interval_s: float = 600.0


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
    # gemini-3.8-flash is served from the `global` endpoint and the `us` / `eu`
    # multi-regions, not from single regions such as us-central1.
    gcp_location: str = "global"
    gemini_timeout_seconds: int = 20
    # Thinking level for Gemini 3+ calls. Every Gemini node here is a cheap
    # hot-path call (classifier, rewrite, clarification, direct answer) or the
    # eval judge; `low` keeps latency close to Gemini 2.5 Flash. Empty = model default.
    gemini_thinking_level: Literal["low", "medium", "high"] | None = "low"

    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    bedrock_timeout_seconds: int = 60
    # Effort for Claude 4.7+ on Bedrock: low | medium | high | xhigh | max.
    # Anthropic suggests `low` for chat, extraction and search, and the Bedrock
    # nodes are the answer and the summary. Claude Sonnet 5.5 defaults to
    # `high`, which thinks before almost every reply. Empty = model default.
    bedrock_claude_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "low"

    # Twilio / WhatsApp
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_from: str = "whatsapp:+14155238886"  # Twilio sandbox default
    twilio_webhook_url: str = ""  # exact URL configured in Twilio console
    twilio_status_webhook_url: str = ""  # delivery-status callback URL

    default_collection_id: int | None = None
    conversation_grace_period_days: int = 7

    # Browser origins allowed to call the API. Without API_KEY the API has no
    # auth, so `*` would let any open web page read and delete local data.
    # Env: CORS_ALLOWED_ORIGINS='["http://localhost:8080"]'
    cors_allowed_origins: list[str] = ["http://localhost:8080", "http://localhost:5173"]

    # The key every route needs, sent as `X-API-Key` (audit R6). Unset or
    # empty, the API is open, as a local demo is: set it before the API can be
    # reached from anywhere else. /health without `deep` and the WhatsApp
    # webhook, which checks Twilio's signature, never need it.
    api_key: SecretStr | None = None
    # Requests a minute per client (its API key, or its IP without one) on the
    # routes that spend (audit R9, app/api/rate_limit.py); 0 turns a limit
    # off. Questions: /query, /retrieve, /agent/chat*, /health?deep=true,
    # and each WhatsApp number's messages.
    rate_limit_per_minute: int = 30
    # Uploads, one request per file: the frontend sends a folder three at a
    # time.
    rate_limit_upload_per_minute: int = 60
    # /eval/run, which calls a model twice for each item.
    rate_limit_eval_per_minute: int = 2

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

    # OpenTelemetry traces (app/telemetry.py). Without an endpoint, spans get
    # trace ids for the logs and `query_logs` and never leave the process.
    # The endpoint is the OTLP/HTTP base URL; `/v1/traces` is appended, as the
    # standard variable does. Local Langfuse (`docker compose --profile
    # tracing`): http://langfuse-web:3000/api/public/otel. Headers use the
    # standard format, `name=value` pairs separated by commas, URL-encoded.
    otel_exporter_otlp_endpoint: str = ""
    otel_exporter_otlp_headers: SecretStr | None = None

    agent: AgentSettings = AgentSettings()

    @property
    def psycopg_database_url(self) -> str:
        return self.database_url.replace("+asyncpg", "")


settings = Settings()

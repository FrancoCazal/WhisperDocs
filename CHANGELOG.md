# Changelog

Notable changes, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are git tags.
Changes land through pull requests with CI green.

## [Unreleased]

Fixes from the November 2026 audit, and a review of the frontend.

### Added

- Chat: each `[n]` citation shows its passage on hover and opens the
  sources, with a link to the chunk in its document. The thread is in the
  URL. A streaming answer shows its progress and can be stopped; a failed
  one is an error card with Retry. Each turn shows its cost, tokens and
  time, and the header the thread's total.
- Threads have titles (`conversations.title`, migration 0014), a search
  and a saved filter, and notes can be listed by thread.
- Evaluation: a run's answers and verdicts item by item
  (`GET /eval/results/{run_id}`), a diff between two runs, and a cost
  confirmation before a run starts.
- Documents: the document page can delete or move the document, search its
  chunks and ask the chat about it: a thread that searches only that
  document (`conversations.document_id`, migration 0015; `document_id` on
  `/agent/chat`). Uploads are checked before they are
  sent and show their progress.
- Each answer says which of its calls failed or fell back, and why, for
  example "classifier · billing or quota · gemini-3.8-flash". This is `errors`
  (`{step, kind, model}`) on `/agent/chat`, the stream's final event and a
  thread's messages, and the chat shows it under the answer. When a
  provider refuses every classifier call, the turn still answers from the
  documents, so before this list nothing showed the outage.
- Each model role (classifier, query rewrite, answers, clarification, direct
  answers, compacting, eval judge, embedding) keeps the outcome of its last
  calls: when it last worked, when it last failed and why, and how many
  calls in a row failed. The record comes from real calls; nothing is called
  to check. `GET /admin/model-status` returns it, `/health?deep=true` adds it
  and turns `degraded` while a role fails, and
  `whisperdocs_model_calls_total{role,model,outcome}` counts the calls.
  While a role fails, every page shows a notice naming the roles, the
  reason and what the app does meanwhile. SETTINGS shows each role's
  status.
- `POST /retrieve`: the chunks for a query, without an answer. Each
  chunk comes with its text, its document, its relevance and its cosine,
  scoped to a collection or a single document. This is what another agent
  needs from a knowledge base, and the base of the MCP server. It runs the
  search the chat uses, and counts against the questions rate limit.
- Every operation has a fixed `operationId` (`retrieve_chunks`,
  `answer_question`, ...), where FastAPI generated
  `query_api_v1_query_post`. A test lists them all.
- An MCP server (`python -m app.mcp`, over stdio) for Claude Code, Claude
  Desktop, Cursor and other MCP clients: `search_documents` (the passages,
  nothing generated), `ask_documents` (an answer with its citations),
  `list_collections`, `list_documents`, and a resource per document with
  its full text. No tool changes the knowledge base. It runs on the
  official `mcp` SDK 2.3, speaks the 2026-07-28 spec and the older
  handshake, and each tool calls the service its REST route calls. The
  README has the configuration for each client; ADR 0005 has the design.
- The MCP server over Streamable HTTP, at the API's `/mcp` (audit M-1,
  M-2): stateless, with keys of its own, each scoped to some collections
  (`api_keys`, migration 0019; `python -m app.mcp.keys create | list |
  revoke`). A collection or document out of a key's reach gets the error
  a missing one gets, and the lists and resources show only what it
  reaches. Over HTTP, `search_documents` and `ask_documents` count
  against the questions limit, per key. ADR 0007 has the design.
- MCP tools that write (BE-36, audit M-1 and M-2): `ingest_document` adds
  a file as an upload does (text, or base64 for PDF, Word and Excel),
  and `delete_document` and `move_document` act only after the user
  confirms, through MCP elicitation. On MCP 2026-07-28 the question rides
  the tool's result, which works on the stateless `/mcp`; a client that
  cannot ask its user, or a 2025-11-25 client over HTTP, is refused. A key
  writes only when made with `--write` (`api_keys.can_write`, migration
  0023), in the collections it reaches. Every call to them leaves a row in
  `audit_log`: who, the tool, the document and collection, and whether it
  was done, declined, cancelled, refused or failed, with why;
  `GET /admin/audit-log` pages it. ADR 0008 has the design.
- Structured extraction (audit M-8): `POST /documents/{id}/extract` and the
  MCP tool `extract_fields` take a JSON Schema of flat fields and return
  each value, validated by a Pydantic model built from the schema, with the
  chunk and the words that state it. A field the document does not state is
  null, a required one is listed in `missing`, and a value whose quote is not
  in the document word for word is listed in `unverified`. A document over
  12k tokens is searched for the fields. One call to `AGENT_MODEL_EXTRACTION`
  (Sonnet 4.6 by default), a new model role in the model status.
- An MCP connector (audit M-6): `python -m app.connectors.mcp_sync` reads an
  MCP server's resources (over stdio or Streamable HTTP) into a collection.
  Each document keeps its resource's URI (`documents.source_uri`, migration
  0024): a later run leaves an unchanged one alone, replaces a changed one in
  one commit re-embedding only its new chunks, and skips a text the
  collection already holds. `--dry-run` embeds and writes nothing. No model
  is involved, and it is not a tool of the agent.
- History: filters by origin and by question text
  (`GET /query/history?source=&q=`), and pagination.
- Frontend tests: vitest for the API client's stream parser, and a
  Playwright test of the chat against a mocked backend.
- OpenTelemetry traces. Each turn is a trace: FastAPI's span of the request,
  the turn, a span per graph node, and under them each model call,
  embedding request and retrieval, with the GenAI semantic conventions'
  attributes (model, provider, tokens, effort, the error's kind) and the
  cost from `pricing.py`. The turn adds up its calls, the classifier's and
  the summary's included. No span holds prompt, answer or document text.
  Log lines carry `trace_id`, and `query_logs.trace_id` (migration 0016)
  links a row to its trace. Spans are exported over OTLP/HTTP only when
  `OTEL_EXPORTER_OTLP_ENDPOINT` is set; `docker compose --profile tracing`
  runs Langfuse locally to read them. ADR 0006 has the design.
- HTTP metrics: `whisperdocs_http_requests_total` and
  `whisperdocs_http_request_duration_seconds`, by method, route template
  and status. A Grafana dashboard and Prometheus alert rules in `ops/`
  (the API unreachable, 5xx, a failing model role, slow turns);
  `docker compose --profile metrics` runs both.
- An adversarial eval set (audit M-5): 8 questions over documents that
  carry prompt injections, and questions that are injections, run apart
  from the golden set over their own collection (`eval/golden/adversarial.jsonl`,
  `fixtures/adversarial-docs/`, the `adversarial` mode of the `Eval suite`
  workflow). An answer that does what the injection asked fails without a
  judge call.
- Prompt versions: a hash of the prompts in `app/agent/prompts.py` and of
  the structured outputs' schemas goes on each `query_logs` row
  (`prompt_version`, migration 0017), each turn's trace and each eval
  report, with the judge's own version. `runner diff` exits 3 when accuracy
  drops, or the hallucination rate rises, more than 10 points
  (`AGENT_EVAL_MAX_ACCURACY_DROP`, `AGENT_EVAL_MAX_HALLUCINATION_RISE`). The
  `Prompt gate` workflow runs it on a pull request that changes a prompt,
  base against head, once the repository variable `PROMPT_GATE` is `on`.
- The frontend shows what the backend records about a document, a run and
  a turn:
  - Documents: a document with flagged chunks is marked in the list, and
    its page names each chunk's rules and can list only those.
  - Evaluation: each run names its prompt and judge versions, and a
    comparison warns when the judge's changed.
  - History: each logged turn shows its `trace_id` and prompt version, now
    on `GET /query/history`, with a link to the trace when the frontend is
    built with `VITE_TRACE_URL`. A question with placeholders says why.
- Comparing eval runs:
  - Each report records its run config (commit, dataset hash, retrieval
    settings, what the RAG model and the judge were sent) and shows the 95%
    Wilson interval of its accuracy and hallucination rate.
  - An eval run answers at `AGENT_EVAL_RAG_TEMPERATURE` (0), not the chat's
    temperature.
  - The judge reuses its verdicts for the same answer to the same question
    (`AGENT_EVAL_JUDGE_CACHE`, `--no-judge-cache`).
  - `runner run --runs N` writes a series summary (mean, range, t interval,
    the items that flip) and gates on the mean.
  - `runner diff` lists what changed between the configs, warns when the
    judge, the dataset or uncommitted code differ, lists the items that got
    worse and better with McNemar's exact p, and does not compare runs over
    different datasets.
  - The Evaluation page shows the same comparison, from `GET /eval/diff`:
    what changed in the config, the warnings, the items that got worse and
    better per check with McNemar's p. A run's view shows its config, the
    Wilson interval of its accuracy and hallucination rate, and how many
    verdicts came from the judge's cache.
  - `runner rejudge REPORT --times N` judges a saved run's answers again,
    with no cache, and reports how often the verdicts agree. A report keeps
    the ids of the chunks the judge read and a hash of their text.
  - The judge's answer to a correctness check is a fact check (audit E4). An
    in-scope item lists the `required_facts` a correct answer must state and
    the `optional_facts` it may add (`TestItem`; every in-scope item of the
    golden and adversarial sets has them, written against the documents).
    The judge reasons first, then says which required facts the answer states
    and which of its statements contradict one, and the verdict follows from
    that in code: any fact missing, or any contradiction, fails the item. A
    verdict names what is missing (`facts_missing`) and what counts against
    the answer (`wrong_claims`); the Evaluation page and the Markdown report
    show both. An item with no facts is judged against its expected answer as
    a whole, as before, and an out-of-scope item on whether it refused.
    Grounding lists the claims the context does not support and passes when
    there are none. Every judge schema puts its reasoning first.
  - `runner label REPORT` writes a sample of a run's verdicts for a person to
    rule on, without what the judge ruled: half of each check and, within a
    check, half the judge passed and half it failed. `runner agreement REPORT
    LABELS` sets the person's rulings against the judge's: agreement with its
    Wilson interval, Cohen's kappa, what a person says of each kind of
    verdict, and the verdicts they split on. Neither calls a model, and
    `label` never overwrites a labels file.
- The eval results of 2026-11-12, in `eval/reports/2026-11-12_series/`: five
  runs of one config at commit `a1eaf5f`, on the English golden set, with
  Claude Sonnet 4.6 answering and Nova 2 Lite judging. Accuracy averaged
  97.6% (96% to 98%), with no unsupported claim on the 41 in-scope items
  and one over all 50 (#44, a refusal the judge faulted), at about $0.010
  per answer. The README's "Last full run" describes the failures (#19
  misses a document in retrieval; #37 passed four of five) and keeps the
  April run, in Spanish, as the run before.
- Uploads take DOCX, XLSX, CSV and HTML (`.htm` too), besides PDF, Markdown
  and plain text (audit M-7). Each comes out as headings and markdown
  tables: DOCX paragraphs, headings, list items and tables in order; each
  XLSX sheet as a heading over its cells; a CSV as one table with its
  delimiter sniffed; HTML without scripts, styles or markup. A DOCX or XLSX
  that would expand past 100 MB is refused unopened, and so is any document
  past 20 million characters of text.
- Each citation says where its passage sits: the headings it is under
  (`section`, "Guide › Install") and, for a PDF, its pages (`page_start`,
  `page_end`; migration 0020). They come back from `/agent/chat` and its
  stream, `/query`, `/retrieve`, the MCP tools and the document's chunks,
  and the chat, Query and the document page show them. Chunks stored before
  have neither until their document is uploaded again.

### Security

- Untrusted text reaches the models as data (audit A10). The conversation
  summary, written from documents and users' words, no longer goes in with
  the system role: it heads the first user message, in a
  `<conversation_summary>` element. The RAG prompt holds the chunks in one
  `<documents>` element, and the summarizer reads its transcript in
  `<transcript>`. A tag of the element's name inside the text is escaped,
  so a document cannot end its element and forge a question or an
  instruction.
- Ingestion flags instruction-like text (audit M-5). Each chunk keeps the
  rules its text matched (`chunks.instruction_flags`, migration 0018): an
  override, a role marker, a new persona, a request for the system prompt,
  text addressed to the assistant, or a tag of a prompt element. The
  document counts its flagged chunks (`flagged_chunks`), the documents and
  chunks endpoints return both, and a flagged document is logged.
  `scripts/flag_instructions.py` scans stored chunks again.
- No personal data in `query_logs` (audit M-5). Each question and answer is
  stored with emails, phone numbers, card numbers, CBUs, CUITs/CUILs and
  DNIs replaced by a placeholder. Cards, CBUs and CUITs are told apart by
  their check digits.

- An API key (`API_KEY`, sent as `X-API-Key`). With it set, every route
  needs it except `GET /health` and the WhatsApp webhook, which checks
  Twilio's signature. `/health?deep=true` makes an embedding call and needs
  it too. Unset, the API stays open, as the local demo is. The frontend
  takes the key in SETTINGS. `/mcp` takes it too, or a key of its own
  that reaches only its collections; only a key's SHA-256 is stored, and
  `/mcp` answers only the hosts in `MCP_ALLOWED_HOSTS` (localhost).
- Rate limits per client and minute: questions (30), uploads (60) and eval
  runs (2), set by `RATE_LIMIT_*`. Past a limit the API answers 429 with
  `Retry-After`. An eval run takes at most 200 items.
- Error responses no longer carry exception text. Provider and database
  messages stay in the logs. `/health` reports a failed check as
  `error: <kind>`, and a failed ingestion job says what failed without
  internals.
- The checkpointer moves to `langgraph-checkpoint` 4.x and
  `langgraph-checkpoint-postgres` 3.x. A `<3.0` cap held versions with known
  advisories, including CVE-2025-64439 (deserialization of stored
  checkpoints). Threads written by the old versions read back unchanged.
- Uploads are size-checked before the multipart body is parsed (413). A
  `.pdf` must carry the PDF header, text files must not contain NUL bytes, and
  list endpoints bound `limit`.
- Database passwords are redacted from logs, and CORS uses an allowlist
  (`CORS_ALLOWED_ORIGINS`).
- WhatsApp: no log line carries a phone number. Delivery statuses and sent
  messages no longer log the recipient, a Twilio failure logs its status
  and code but not its message, and database errors leave out their
  statement's values (`hide_parameters`).
- WhatsApp: a repeated message (same `MessageSid`) is answered once, and
  each number gets the questions' limit per minute.
- The eval workflow reads the service account key from the environment
  instead of pasting the secret into its script, declares read-only
  permissions, and no longer publishes the reports, answers included, to
  `gh-pages`; they stay in the run's artifacts.

### Changed

- The synchronous upload, the MCP server's `ingest_document` and the MCP
  connector store a document through one path (`check_document`,
  `extract_document`, `store_document` in `app/services/ingestion.py`), and
  the delete and move routes share `app/services/documents.py` with the MCP
  tools. mypy covers `app/connectors` too.
- CI measures the coverage of `app/` and fails under 88% (audit E14). It
  was 90.6% with the database tests on 2026-11-10; the audit measured 36%.
  `pytest-cov` is in the `dev` extra.
- The two LLM providers share their message conversion and the recovery of
  JSON from a structured output that did not parse (`to_lc_messages`,
  `salvage_structured` in `app/services/llm/base.py`, audit A14), which were
  copied in each.
- Chunking follows the document's structure (audit R14). A heading starts a
  chunk once the current one has 250 tokens (shorter sections share one),
  each chunk starts with the headings above it, a table that fits is never
  cut and one that does not repeats its header in each part, and cuts are
  made on the text, so none splits a character or drifts a token past its
  separator. There is no overlap between paragraphs, only inside one cut in
  two. The eval corpus goes from 71 chunks of median 467 tokens to 95 of
  median 305. A collection seeded before keeps its old chunks, since
  `seed_eval` skips the files it holds: delete it and seed it again before
  a run. A report records
  how many chunks its collection held and a hash of their texts, and a diff
  warns when two runs searched different chunks.
- "Hallucination rate" is called what it measures (audit E5): the unsupported-claim
  rate, the share of answers with a claim the retrieved context does not support,
  true or not. It is faithfulness to the context, not a fact check, and the grounding
  judge's prompt now says so. A report gives it over the in-scope items, since the
  out-of-scope ones are mostly refusals with nothing to support, and over all items.
  The Evaluation page headlines the in-scope figure; `runner` prints both. The API
  field (`hallucination_rate`), `--max-hallucination`,
  `AGENT_EVAL_MAX_HALLUCINATION_RATE` and the diff and gate thresholds keep their
  names and stay over all items, so stored reports and scripts still work.
- The eval judge changed (audit E4), so its verdicts are not comparable with
  those of earlier reports: its prompts and schemas have a new version
  (`judge_prompt_version`), the golden set has new facts, and #38's expected
  answer now says only what the documents hold about SQL injection. COMPARE
  and `runner diff` warn that the judge changed, and the golden set's hash
  changed with its facts, so a diff (and the prompt gate) will not compare a
  run of it with one of the earlier set when both reports record their
  dataset hash. A verdict's reasoning is no longer a single line, and the
  judge answers up to 2048 tokens instead of 1024.
- The models declare what the migrations built: `chunks.tsv`, the generated
  column of the full-text leg, and its GIN index; the names of the
  conversation, query log and saved note indexes; the vector's size.
  `alembic --autogenerate` used to propose dropping `tsv`. CI now runs
  `alembic check` after the migrations, which leaves LangGraph's checkpoint
  tables alone. `tsv` is never read by the ORM, nor fetched on insert.
- `EMBEDDING_DIMENSION` must match `chunks.embedding` (1536): another value
  stops the API at startup instead of failing the first upload.
- The backend's `Dockerfile` builds in stages. The default image, `runtime`, has no
  compilers and no dev tools, and is 925 MB instead of 2.17 GB. Compose builds `dev`,
  which adds pytest, ruff and mypy. Both images:
  - run as a non-root user;
  - have a `HEALTHCHECK`;
  - include the seed corpora, which `.dockerignore` used to leave out.
- `GET /health` answers 503 when degraded, with the same body, so a load
  balancer or a monitor reads it without parsing; the frontend still shows
  DEGRADED. `?deep=true` keeps the embedder's result 5 minutes (30 seconds
  after a failure) instead of paying an embedding on every call.
- The conversation summary keeps where its facts came from: each user
  message records the collection or document its turn searched, and each
  fact in the summary names its source. Each pass carries over the
  previous summary's facts, within 500 tokens instead of 300.
- The eval CLI exits 3 when a run is below `AGENT_EVAL_MIN_ACCURACY` (0.80)
  or above `AGENT_EVAL_MAX_HALLUCINATION_RATE` (0.10), so the eval workflow
  fails on a regression. The workflow drops its branches for pull requests
  and schedules, which never ran, sets `AWS_REGION` (the app ignored
  `AWS_DEFAULT_REGION`) and defaults `GCP_LOCATION` to `global`.
- The WhatsApp webhook answers 204 at once and runs the turn in the
  background: Twilio waits at most 15 s, and a turn's p95 was 9.1 s. A
  number's messages are answered one at a time, in order, so a
  `/coleccion` sent before a question applies to it. One Twilio client is
  reused instead of one per message.
- Every error body is `{detail, code, request_id}`, with a stable `code`
  (`not_found`, `rate_limited`, `upstream_error`, ...). A 5xx message quotes
  the request id, which is also the `X-Request-ID` header and is on every
  log line of the request. A failed model or embedding call on `/query` is a
  502 with its `kind`, where a generation failure used to be a bare 500. An
  error the routes did not handle is a 500 the browser can read, and it no
  longer surfaces as a CORS error.
- Models: `gemini-3.8-flash` replaced `gemini-2.5-flash`, retired on
  2026-10-20, on the hot path and the eval judge, until every role moved to
  Bedrock (below). Sampling parameters are
  sent per model, and startup fails on a model ID that cannot be routed.
- Claude 4.7 and later run on Bedrock through Converse, which serves
  Claude Sonnet 5.5 (`global.anthropic.claude-sonnet-5-5`), the replacement
  for Sonnet 4.5 (Legacy on Bedrock, end of life on 2027-04-08). They get
  the effort in `BEDROCK_CLAUDE_EFFORT` (default `low`), room in
  `max_tokens` for their thinking, and structured output as a JSON schema,
  since Sonnet 5.5 rejects a forced tool call. The defaults do not use it:
  Bedrock does not offer Sonnet 5.5 to every account.
- Models: every role runs on Bedrock by default, so one AWS account runs
  every model (the embeddings are OpenAI's). Claude Haiku 4.5 takes the
  hot path (classifier, query rewrite, clarification, direct answers,
  summary), Claude Sonnet 4.6 the answers and the extraction, and Amazon
  Nova 2 Lite judges the eval, of another family than the Claude it judges.
  Sonnet 4.5, the answering model before, is Legacy, and Bedrock refuses a
  Legacy model to an account that has not called it in 30 days. With the
  hot path on Gemini, a clone with only AWS keys sent every classifier call
  to a provider it could not reach. `AGENT_MODEL_*` still moves a node to
  Gemini. ADR 0004 has an amendment.
- `pricing.py` prices Claude Sonnet 4.6 (global profile) and Amazon Nova 2
  Lite and Nova Pro, so every role can run on Bedrock with known costs. A
  Legacy model is refused to an account that has not called it in 30 days,
  so one that has not used Sonnet 4.5 lately cannot get it; Sonnet 4.6 is
  Active. Nova 2 Lite is a judge of another family than the Claude that
  answers: on eight judge cases with a known verdict it ruled all eight
  right, as Nova Pro did, and Haiku 4.5 seven. Nova Premier is listed, and
  refused as past its end of life.
- Full-text retrieval ORs the question's terms with per-document language
  and accent-insensitive matching. Before, it matched 5 of 41 golden
  questions; now it matches all 41.
- `relevance` is the score results are ordered by, and the vector
  similarity moved to its own `cosine` field.
- The request's `top_k` is honored by `/agent/chat`.
- A turn with no retrieved chunks gets a fixed answer, with no model call.
  A retrieval failure gets its own answer.
- Dependencies are locked in `uv.lock`; CI and the Docker image install with
  `uv sync --frozen`.
- The frontend is a static single-page app, served by nginx in its container
  (port 8080), with long-lived caching for the hashed assets. The pages'
  paths are in English (`/query`, `/documents`, `/collections`, `/history`,
  `/evaluation`); the old Spanish paths redirect.
- The frontend ships its own font, and loads the code highlighter with the
  first code block. Answers are at reading size.
- The compose database's port on the host is `DB_PORT`, 5442 by default:
  another project's Postgres took 5435, the earlier default, and kept this
  one from starting.
- The README and the architecture document describe the agent as
  intent-routed and single pass: one path per turn, one retrieval, no loop.
- Answers written without the model (no documents, a failure, out of scope)
  take the language of the user's message, and the generation prompts ask for
  answers in that language. Summary headings are fixed, in English.
- WhatsApp's own texts (a new thread's notice, the apology for a failed
  turn, the commands' replies) take the language of the user's message too;
  they were always in Spanish. A command answers in the language of the name
  it was sent as: `/save`, `/collection NAME`, `/collection all` and `/help`,
  or `/guardar`, `/coleccion`, `/coleccion todas` and `/ayuda`.
- The classifier's prompt is in English, as the other prompts are, with a
  few Spanish examples so it still routes either language. It moves
  `PROMPT_VERSION`.
- The query rewrite no longer calls the model on a conversation's first
  message, where there is nothing to resolve.

### Removed

- `eval/dataset.json`, the 10-question Spanish starter set (audit E22).
  Nothing read it; the runner uses `eval/golden/queries.jsonl`.

### Fixed

- Uploading a document took text out of its collection when another document
  was deleted (audit R12). A chunk whose text another document held was not
  stored; deleting that document left neither. Every document now stores all
  its chunks, and a text already embedded by the same model reuses its
  vector (`chunks.embedding_model`, migration 0022). The same document twice
  in a collection is a 409 naming the one it holds, checked before anything
  is embedded and kept under concurrent uploads by a unique index
  (`documents.content_hash`, migration 0021). A move into a collection
  holding the same text leaves that document in place, with the reason in
  `errors`. An edited document no longer comes in with only the chunks that
  changed.
- `text-embedding-3-large` broke every insert: it returns 3072 floats unless
  told otherwise (audit R19). The embedder asks for `EMBEDDING_DIMENSION`
  and refuses a vector of another size before it reaches the database.
- Two test isolation gaps on a developer machine: the dev `.env`'s
  `DEFAULT_COLLECTION_ID` reached the WhatsApp tests, and psycopg could not
  run on Windows' default event loop, which failed the checkpoint pruning
  test. CI was not affected.
- WhatsApp replies broke identifiers and listed sources the answer did not
  use (audit A16). The formatter stripped `_..._` and `#` wherever they
  appeared, so `max_tokens` lost its underscore, `__init__` became bold and
  "C# " lost its hash; it now strips emphasis only where its delimiters
  touch the text they wrap, and a `#` only at the start of a line. The
  sources footer listed every chunk the model was given, in Spanish; it now
  lists the ones the answer cites, each title once, in the language of the
  user's message.
- The reranker's score said nothing across queries (audit R16): it was
  min-max normalized per query, so the best chunk scored 1.0 even when none
  answered. It is now the sigmoid of the cross-encoder's logit, and
  `RERANKER_SCORE_THRESHOLD` never drops the best chunk. A threshold tuned
  before means something else now. The model loads at startup when the
  reranker is on, not on the first question; the `rerank` extra needs
  sentence-transformers 4.0 or later.
- With `AGENT_CITATION_SEMANTIC_VALIDATION` on, a failed embedding call
  left the citations unchecked without saying so: the check returned an
  empty result, so the turn's `errors` never held `citation_check` and the
  chat never showed "CITATIONS NOT CHECKED". The failure now reaches the
  node, which records it and still delivers the answer (audit A19).
- A PDF's tables were indexed twice: `extract_text` already returned their
  cells as lines, and each table was added again as markdown (audit R13).
  The text now comes from outside the tables, and each table goes in once,
  where it sits. A PDF with no text layer says it may be a scan, and that
  there is no OCR.
- A text file in Latin-1 or Windows-1252 came out with replacement
  characters, and one in UTF-16 was refused as binary (audit R13). Text is
  read by its byte order mark, as UTF-8, or as the encoding
  `charset-normalizer` finds, Windows-1252 when it ties: the detector put
  Windows-1250 first on Spanish text, which reads "años" as "ańos". Line
  endings are `\n`.
- A document containing `<|endoftext|>` or another tokenizer control string
  failed its upload; it is text now.
- An embedding request has a time limit, `EMBEDDING_TIMEOUT_SECONDS` (30 s),
  as the model calls have. The OpenAI client had none: an endpoint that
  took the connection and never answered held a chat turn, an upload or an
  MCP call for 600 s a try, three tries.
- Edge cases of the agent's stream. `out_of_scope` sent no `node_start` or
  `node_end`: the stream now reports every node of the graph. A `final` that
  replaced the tokens already shown (a generation that failed partway,
  citations taken out) came without warning; a `reset` event now comes first,
  and before an `error` that follows tokens. A client that closes the stream
  leaves a log line and a span marked `cancelled`, and the thread counts as
  active. Two turns on one thread, from `/agent/chat`, the stream or
  WhatsApp, run one after the other.
- Retrieval no longer reads each candidate's whole document. It selected the
  `Chunk` and `Document` rows, so each of its 20 candidates brought its
  embedding and its document's full text, and a second query read the
  documents again. It now selects the columns a result shows. The document list
  and a document's chunk list no longer read the text either.
- Deleting a collection no longer loads its documents and their chunks,
  embeddings included, to delete them one by one: the foreign keys' `ON DELETE
  CASCADE` does it. Deleting a document, and the bulk delete and move, are one
  statement each. The per-document `try/except` in the bulk operations never
  caught anything, since an error only came at the commit; it is gone, and
  `errors` in their result is always empty.
- A failed summary is no longer retried on every turn: compaction waits
  `AGENT_COMPACTION_RETRY_MESSAGES` (6) messages first. A summary cut at
  `max_tokens`, or empty, is not kept; before, a cut one replaced the
  previous summary.
- The checkpoint tables no longer grow with the square of a thread's turns.
  Each turn saved about five checkpoints, and each change to `messages`
  stored the whole list again: 120 turns kept 70 MB of copies of 0.6 MB of
  messages. A background sweep keeps each thread's latest checkpoint, every
  `AGENT_CHECKPOINT_PRUNE_INTERVAL_S` (600 s).
- An eval item that retrieved no chunks no longer ends the whole run with
  `UnboundLocalError`.
- Compaction never leaves the conversation starting with the assistant,
  which Bedrock rejects. The summarizer no longer returns empty summaries.
- Filtered vector search no longer comes back empty for small collections.
- Document parsing runs off the event loop. An ingestion job cut short by a
  restart ends as `failed` instead of staying in progress.
- Bedrock streams retry throttling before the first token.
- The eval averages retrieval over every in-scope query, and aborts when too
  many judge calls fail.
- `GET /documents/jobs` answered 422 on every call.
- A failed answer ("I had a problem…") stays in the thread but no longer
  reaches later prompts, the summary or `query_logs`.
- A turn answered without a model call, or whose answer failed, reported the
  classifier's tokens as the answer's. The classifier's usage has its own
  metadata fields.
- A failed answer no longer lists the retrieved chunks as its citations.
- The documents page's bulk delete could delete documents outside the
  filter. A streamed answer could land in another thread. A document's page
  could not be opened.
- The UI's costs came from an old price table; they are the backend's
  `cost_usd`. Lists stopped at 50 rows. Model output could load remote
  images.
- After a Postgres restart the API kept handing out dead connections.
- A question whose stream was cut off was left without an answer, and broke
  every later prompt of its thread.
- `GET /eval/results` read the reports on the event loop, and applied
  `limit` before it skipped the files that are not reports.
- The eval CLI's reports showed metrics an old run did not save as 0.
- On a phone, iOS zoomed into a search box, a select or TOP_K when it took
  focus: every field is 16px there now.
- When the classifier's model failed, every question got "Can you give me
  more details?": the clarification it fell back to calls a model too, and
  failed the same way. A classifier failure now searches the documents.

### CI

- Lint, format and file hygiene (pre-commit), gitleaks, pip-audit and mypy.
- The full test suite runs against a pgvector Postgres service.
- Dependabot for Python, npm, GitHub Actions and Docker.
- A frontend job: types, lint and format, build, and the unit and browser
  tests.
- No test can reach a model or embedding API, even with credentials in
  `.env`: a test that resolves a host other than this machine or the test
  database fails.
- The tests that check that ingestion parses off the event loop read no
  clock. They compared the loop's longest gap with 150 ms, and a
  collection of the GC's oldest generation in the full suite (up to 1.2 s)
  failed them about one run in four. Now each CPU-bound step waits for the
  loop to run a callback it scheduled, which only a step off the loop
  sees; chunking is checked too. A test still running after 5 minutes
  dumps every thread's stack (`faulthandler_timeout`).
- Tests for the semantic citation check (which citations it flags, what it
  embeds, the turn when it fails) and for token streaming (the events the
  stream forwards, streaming off, a stream cut midway, a direct answer
  through the compiled graph), audit A19.

## [1.0] - 2026-05-19

First public snapshot, published as a single commit. After the tag, `main`
received CI, documentation and the license only.

[Unreleased]: https://github.com/FrancoCazal/WhisperDocs/compare/v1.0...HEAD
[1.0]: https://github.com/FrancoCazal/WhisperDocs/releases/tag/v1.0

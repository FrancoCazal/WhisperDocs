// API client for WhisperDocs FastAPI backend.
// Configure with VITE_API_BASE_URL env var; defaults to http://localhost:8000.

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

const ROOT = `${API_BASE_URL}/api/v1`;

// The API key, sent as X-API-Key on every call: the API asks for it when it
// runs with API_KEY set (audit R6). It is kept in this browser's storage; where
// storage is blocked (a private window, say) it lasts until the page reloads.
const API_KEY_STORAGE = "whisperdocs.api-key";
let keyInMemory = "";

export function getApiKey(): string {
  try {
    return localStorage.getItem(API_KEY_STORAGE) ?? "";
  } catch {
    return keyInMemory;
  }
}

/** Keep `key` for the next calls; an empty one removes it. False when the
 *  browser would not store it, so it is gone on reload. */
export function setApiKey(key: string): boolean {
  keyInMemory = key.trim();
  try {
    if (keyInMemory) localStorage.setItem(API_KEY_STORAGE, keyInMemory);
    else localStorage.removeItem(API_KEY_STORAGE);
    return true;
  } catch {
    return false;
  }
}

function keyHeader(): Record<string, string> {
  const key = getApiKey();
  return key ? { "X-API-Key": key } : {};
}

// Text limits the backend validates (app/schemas): past them it answers 422.
export const LIMITS = {
  chatMessage: 5000,
  queryQuestion: 2000,
  collectionName: 200,
  collectionDescription: 2000,
} as const;

export class ApiError extends Error {
  status: number;
  detail: unknown;
  // The API's error `code` (not_found, rate_limited, ...) and the request id
  // that finds the call in its logs; null when the response had none.
  code: string | null;
  requestId: string | null;
  constructor(
    status: number,
    detail: unknown,
    message: string,
    { code = null, requestId = null }: { code?: string | null; requestId?: string | null } = {},
  ) {
    super(message);
    this.status = status;
    this.detail = detail;
    this.code = code;
    this.requestId = requestId;
  }
}

// Messages for the codes the user acts on here, in place of the API's.
const CODE_MESSAGES: Record<string, string> = {
  api_key_required: "The API needs a key: set it in SETTINGS.",
  api_key_invalid: "The API refused the key: check it in SETTINGS.",
};

// FastAPI's `detail`: a string, or for a 422 a list of {loc, msg}.
function detailMessage(detail: unknown): string | null {
  if (typeof detail === "string") return detail || null;
  if (Array.isArray(detail)) {
    const parts = detail.map((d) => {
      if (d && typeof d === "object" && "msg" in d) {
        const { loc, msg } = d as { loc?: unknown; msg: unknown };
        // ["body", "message"] -> "message"; the "body" prefix says nothing.
        const where = Array.isArray(loc) ? loc.filter((p) => p !== "body").join(".") : "";
        return where ? `${where}: ${String(msg)}` : String(msg);
      }
      return JSON.stringify(d);
    });
    return parts.join("; ") || null;
  }
  return detail == null ? null : JSON.stringify(detail);
}

/** The error for a failed response, with its `detail` readable as the message. */
async function toApiError(res: Response): Promise<ApiError> {
  const text = await res.text().catch(() => "");
  let body: unknown = text || null;
  try {
    body = JSON.parse(text);
  } catch {
    /* not JSON: a proxy's error page, say; keep it as the detail */
  }
  const fields = isRecord(body) ? body : {};
  const code = str(fields.code);
  const message =
    (code && CODE_MESSAGES[code]) ?? detailMessage(fields.detail ?? null) ?? `HTTP ${res.status}`;
  return new ApiError(res.status, body, message, {
    code,
    requestId: str(fields.request_id) ?? res.headers.get("X-Request-ID"),
  });
}

// `accept`: error statuses whose body is the answer, such as /health's 503
// when degraded; any other one throws.
async function request<T>(
  path: string,
  init?: RequestInit,
  accept: readonly number[] = [],
): Promise<T> {
  const res = await fetch(`${ROOT}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...keyHeader(),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok && !accept.includes(res.status)) throw await toApiError(res);
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ============ Types ============
export type Health = {
  status: string;
  database?: string;
  checks?: Record<string, string>;
  // With ?deep=true only: each role's last call ("ok", "failing: <kind>", "unknown").
  models?: Record<string, string>;
};

// GET /admin/model-status: what each role's last calls did, since the API
// started. "unknown": no call yet.
export type ModelRoleState = {
  model: string;
  state: "ok" | "failing" | "unknown";
  calls: number;
  failures_in_a_row: number;
  last_ok_at: string | null;
  last_failure_at: string | null;
  last_failure_kind: string | null;
};

export type ModelStatus = { since: string; roles: Record<string, ModelRoleState> };

export type Collection = {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
  // Over every document in it, summed by the server.
  document_count: number;
  chunk_count: number;
  total_tokens: number;
};

export type DocumentRead = {
  id: number;
  collection_id: number;
  title: string;
  filename: string;
  file_type: "pdf" | "md" | "txt" | string;
  chunk_count: number;
  total_tokens: number;
  // Chunks whose text reads as an instruction to the model
  // (app/services/instruction_scan.py). Missing from a backend older than it.
  flagged_chunks?: number;
  created_at: string;
};

export type DocumentDetail = DocumentRead & { raw_text: string };

export type Chunk = {
  id: number;
  document_id: number;
  chunk_index: number;
  text: string;
  token_count: number;
  // The rules the text matched (lib/instructionFlags.ts); empty when none.
  instruction_flags?: string[];
};

export type Citation = {
  chunk_id: number;
  document_id: number;
  document_title: string;
  chunk_index: number;
  // The fused rank score the citations are ordered by: relative to the other
  // results of the same query, so not a quality measure.
  relevance: number;
  // Cosine similarity to the question. null when only full-text search found
  // the chunk; missing in citations stored before the field existed.
  cosine?: number | null;
  text_preview?: string | null;
};

export type QueryMetadata = {
  model: string;
  tokens_input: number;
  tokens_output: number;
  latency_ms: number;
  chunks_retrieved: number;
  cost_usd: number;
};

export type QueryResponse = {
  answer: string;
  citations: Citation[];
  metadata: QueryMetadata;
};

export type QueryLog = {
  id: number;
  question: string;
  answer: string;
  collection_id: number | null;
  chunks_retrieved: number;
  model_used: string;
  tokens_input: number;
  tokens_output: number;
  latency_ms: number;
  cost_usd: number;
  session_id: string | null;
  agent_action: string | null;
  // The turn's OpenTelemetry trace (lib/traceUrl.ts) and the prompts that
  // answered it. Null on rows from before each column; missing from a
  // backend older than the fields.
  trace_id?: string | null;
  prompt_version?: string | null;
  created_at: string;
};

// One question of an eval dataset (TestItem in app/eval/schemas.py).
export type EvalItem = {
  id: number;
  question: string;
  expected_answer: string;
  difficulty: "easy" | "medium" | "hard";
  category: "in_scope" | "out_of_scope";
  // Titles of the documents a good retrieval returns.
  expected_sources?: string[];
};

export type EvalMetrics = {
  total: number;
  accuracy: number;
  hallucination_rate: number;
  avg_retrieval_precision: number;
  // Reports from before these three were measured do not have them.
  avg_retrieval_recall?: number;
  avg_latency_ms: number;
  latency_p50_ms?: number;
  latency_p95_ms?: number;
  avg_tokens_input: number;
  avg_tokens_output: number;
  total_tokens_input: number;
  total_tokens_output: number;
  // Generation only: the judge's calls and the embeddings are not counted.
  estimated_cost_usd: number;
  // Items where a judge call failed: their verdicts say nothing.
  judge_errors?: number;
  by_difficulty?: Record<string, EvalBucket>;
  by_category?: Record<string, EvalBucket>;
};

export type EvalBucket = {
  total: number;
  accuracy: number;
  hallucination_rate: number;
  avg_retrieval_precision?: number;
  avg_retrieval_recall?: number;
};

// The prompts a run answered and judged with (app/agent/prompt_version.py,
// app/eval/judges.py). Reports from before the versions have neither.
export type PromptVersions = {
  prompt_version?: string | null;
  judge_prompt_version?: string | null;
};

export type EvalReport = PromptVersions & {
  run_id: string;
  timestamp: string;
  dataset_size: number;
  model: string;
  metrics: EvalMetrics;
  results: EvalResult[];
};

// A judge's verdict. For hallucination, `passed` means grounded. `error`: the
// judge call failed, so `passed` is false by construction.
export type JudgeVerdict = { passed: boolean; reasoning: string; error?: boolean };

export type EvalResult = {
  item_id: number;
  question: string;
  expected_answer: string;
  system_answer: string;
  // Titles of the retrieved documents, in rank order.
  citations: string[];
  latency_ms: number;
  tokens_input: number;
  tokens_output: number;
  correctness: JudgeVerdict;
  hallucination: JudgeVerdict;
  retrieval_precision: number;
  retrieval_recall?: number;
  difficulty: EvalItem["difficulty"];
  category: EvalItem["category"];
};

export type EvalRunSummary = PromptVersions & {
  run_id: string;
  timestamp: string;
  dataset_size: number;
  model: string;
  accuracy: number;
  hallucination_rate: number;
  estimated_cost_usd: number;
};

// ============ Agent / Conversations / Notes ============
export type AgentCitation = Citation;

// The answer's model call. `cost_usd` is priced by the backend, which keeps
// the only pricing table.
export type AgentTurnMetadata = {
  model: string | null;
  tokens_input: number;
  tokens_output: number;
  latency_ms: number;
  chunks_retrieved: number;
  cost_usd: number;
};

// A call the turn made that failed or fell back. A turn can answer with
// some: when the classifier fails, it searches the documents instead.
export type TurnError = {
  // summarize, classifier, query_rewrite, embedding, retrieval, generation,
  // clarification, citation_check
  step: string;
  // billing, auth, rate_limit, timeout, unavailable, invalid_request,
  // bad_response, unknown
  kind: string;
  model: string | null;
};

export type AgentChatResponse = {
  thread_id: string;
  action: string | null;
  answer: string;
  citations: AgentCitation[];
  metadata: AgentTurnMetadata;
  // The answer reports a failure ("I had a problem…") and cites nothing.
  failed: boolean;
  errors: TurnError[];
};

export type AgentStreamEvent =
  | { type: "thread"; thread_id: string }
  | { type: "node_start"; node: string }
  | { type: "node_end"; node: string; summary: { action?: string; chunks?: number } }
  | { type: "token"; node: string | null; delta: string }
  // Drop the tokens shown so far: the final answer or the error that follows
  // replaces them (a generation that failed partway, citations taken out).
  | { type: "reset"; node: string | null }
  | { type: "final"; payload: AgentChatResponse }
  | { type: "error"; message: string };

export type IngestionStatus =
  "pending" | "extracting" | "chunking" | "embedding" | "persisting" | "complete" | "failed";

export type Page<T> = {
  items: T[];
  next_cursor: number | null;
  // Matching items across every page; sent with the first page only.
  total?: number | null;
};

export type BulkOpResult = {
  requested: number;
  succeeded: number;
  failed: number;
  not_found: number[];
  errors: Record<string, string>;
};

export type IngestionJob = {
  id: number;
  collection_id: number;
  document_id: number | null;
  filename: string;
  file_type: string;
  title: string | null;
  status: IngestionStatus;
  progress: number;
  total_chunks: number;
  processed_chunks: number;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
};

export type IngestionJobEvent =
  | {
      type: "progress";
      job_id: number;
      status: IngestionStatus;
      progress: number;
      processed_chunks: number;
      total_chunks: number;
    }
  | { type: "complete"; job_id: number; document_id: number | null }
  | { type: "error"; job_id?: number; message: string };

export type Conversation = {
  id: number;
  thread_id: string;
  user_number: string | null;
  channel: string;
  collection_id: number | null;
  // The one document the thread searches, in place of its collection.
  document_id?: number | null;
  // The thread's first question; null until it has one.
  title: string | null;
  is_saved: boolean;
  last_active_at: string;
  created_at: string;
};

export type ConversationMessage = {
  role: "user" | "assistant" | "system";
  content: string;
  // When it was written (UTC ISO); messages saved before the stamp have none.
  timestamp: string | null;
  // The node that answered (search_docs, answer_direct, ...); assistant only.
  action: string | null;
  citations: Citation[] | null;
  metadata: AgentTurnMetadata | null;
  failed: boolean;
  // None from a backend older than the field.
  errors?: TurnError[];
};

// Which answers a query log holds: the agent's chat or the Query page.
export type LogSource = "chat" | "query";

export type SavedNote = {
  id: number;
  conversation_id: number | null;
  // The note's thread, for a link to it; null for a note without one.
  thread_id: string | null;
  collection_id: number | null;
  source_message_content: string;
  user_note: string | null;
  created_at: string;
};

// ============ Server-sent events ============
// One message of a text/event-stream: its event name and its data, which the
// backend always sends as a JSON object.
type SseMessage = { event: string; data: Record<string, unknown> };

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/** The messages of an event stream, as they arrive. Lines may end in CRLF, LF
 *  or CR, and a message's data lines are joined with "\n", as the SSE spec
 *  says. A message whose data is not a JSON object is skipped, and one cut
 *  off by the end of the stream is dropped. */
async function* readSse(res: Response): AsyncGenerator<SseMessage> {
  if (!res.body) throw new ApiError(res.status, null, "Stream failed: empty response");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let event = "";
  let data: string[] = [];
  try {
    while (true) {
      const { value, done } = await reader.read();
      let text = buf + (done ? decoder.decode() : decoder.decode(value, { stream: true }));
      // A CR at the end may be the first half of a CRLF: keep it for the next
      // chunk. At the end of the stream there is none, and it ends its line.
      const held = !done && text.endsWith("\r") ? "\r" : "";
      if (held) text = text.slice(0, -1);
      const lines = text.split(/\r\n|\r|\n/);
      buf = (lines.pop() ?? "") + held;
      for (const line of lines) {
        if (line === "") {
          const parsed = data.length > 0 ? parseJson(data.join("\n")) : null;
          if (isRecord(parsed)) yield { event: event || "message", data: parsed };
          event = "";
          data = [];
        } else if (!line.startsWith(":")) {
          // "field: value"; a line starting with ":" is a comment (a keep-alive).
          const colon = line.indexOf(":");
          const field = colon === -1 ? line : line.slice(0, colon);
          const raw = colon === -1 ? "" : line.slice(colon + 1);
          const val = raw.startsWith(" ") ? raw.slice(1) : raw;
          if (field === "event") event = val;
          else if (field === "data") data.push(val);
        }
      }
      // What is left in buf is a line the stream cut off: dropped.
      if (done) return;
    }
  } finally {
    // Also when the reader stops early: the request is not left open.
    await reader.cancel().catch(() => undefined);
  }
}

const INGESTION_STATUSES: readonly string[] = [
  "pending",
  "extracting",
  "chunking",
  "embedding",
  "persisting",
  "complete",
  "failed",
] satisfies IngestionStatus[];

/** An ingestion stream's message as an event; null for one that is not. */
function ingestionEvent({ event, data }: SseMessage): IngestionJobEvent | null {
  const jobId = num(data.job_id);
  switch (event) {
    case "progress": {
      const status = str(data.status);
      if (jobId === null || status === null || !INGESTION_STATUSES.includes(status)) return null;
      return {
        type: "progress",
        job_id: jobId,
        status: status as IngestionStatus,
        progress: num(data.progress) ?? 0,
        processed_chunks: num(data.processed_chunks) ?? 0,
        total_chunks: num(data.total_chunks) ?? 0,
      };
    }
    case "complete":
      return jobId === null
        ? null
        : { type: "complete", job_id: jobId, document_id: num(data.document_id) };
    case "error":
      return {
        type: "error",
        job_id: jobId ?? undefined,
        message: str(data.message) ?? "ingestion error",
      };
    default:
      return null;
  }
}

function turnErrors(v: unknown): TurnError[] {
  if (!Array.isArray(v)) return [];
  return v.filter(isRecord).flatMap((e) => {
    const step = str(e.step);
    const kind = str(e.kind);
    return step && kind ? [{ step, kind, model: str(e.model) }] : [];
  });
}

function turnMetadata(v: unknown): AgentTurnMetadata {
  const m = isRecord(v) ? v : {};
  return {
    model: str(m.model),
    tokens_input: num(m.tokens_input) ?? 0,
    tokens_output: num(m.tokens_output) ?? 0,
    latency_ms: num(m.latency_ms) ?? 0,
    chunks_retrieved: num(m.chunks_retrieved) ?? 0,
    cost_usd: num(m.cost_usd) ?? 0,
  };
}

/** An agent stream's message as an event; null for one that is not. */
function agentEvent({ event, data }: SseMessage): AgentStreamEvent | null {
  switch (event) {
    case "thread": {
      const threadId = str(data.thread_id);
      return threadId ? { type: "thread", thread_id: threadId } : null;
    }
    case "node_start": {
      const node = str(data.node);
      return node ? { type: "node_start", node } : null;
    }
    case "node_end": {
      const node = str(data.node);
      if (!node) return null;
      const s = isRecord(data.summary) ? data.summary : {};
      const summary: { action?: string; chunks?: number } = {};
      const action = str(s.action);
      const chunks = num(s.chunks);
      if (action) summary.action = action;
      if (chunks !== null) summary.chunks = chunks;
      return { type: "node_end", node, summary };
    }
    case "token":
      return { type: "token", node: str(data.node), delta: str(data.delta) ?? "" };
    case "reset":
      return { type: "reset", node: str(data.node) };
    case "final": {
      const threadId = str(data.thread_id);
      if (!threadId) return null;
      return {
        type: "final",
        payload: {
          thread_id: threadId,
          action: str(data.action),
          answer: str(data.answer) ?? "",
          citations: Array.isArray(data.citations)
            ? (data.citations.filter(isRecord) as AgentCitation[])
            : [],
          metadata: turnMetadata(data.metadata),
          failed: data.failed === true,
          errors: turnErrors(data.errors),
        },
      };
    }
    case "error":
      return { type: "error", message: str(data.message) ?? "stream error" };
    default:
      return null;
  }
}

// ============ Endpoints ============
export const api = {
  // A degraded API answers 503 with the same body: it is up, and says what fails.
  health: () => request<Health>("/health", undefined, [503]),
  adminSettings: () =>
    request<{
      models: Record<string, { model: string; temperature: number }>;
      retrieval: { top_k: number };
      compaction: { token_threshold: number; count_threshold: number; k_recent: number };
      embedding: { model: string; dimension: number };
    }>("/admin/settings"),
  modelStatus: () => request<ModelStatus>("/admin/model-status"),

  listCollections: () => request<Collection[]>("/collections"),
  createCollection: (body: { name: string; description?: string }) =>
    request<Collection>("/collections", { method: "POST", body: JSON.stringify(body) }),
  deleteCollection: (id: number) => request<void>(`/collections/${id}`, { method: "DELETE" }),

  listDocumentsPage: (params: { collection_id?: number; limit?: number; cursor?: number }) => {
    const q = new URLSearchParams();
    if (params.collection_id !== undefined) q.set("collection_id", String(params.collection_id));
    if (params.limit !== undefined) q.set("limit", String(params.limit));
    if (params.cursor !== undefined) q.set("cursor", String(params.cursor));
    return request<Page<DocumentRead>>(`/documents?${q.toString()}`);
  },
  getDocument: (id: number) => request<DocumentDetail>(`/documents/${id}`),
  getChunks: (id: number) => request<Chunk[]>(`/documents/${id}/chunks`),
  deleteDocument: (id: number) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  bulkDeleteDocuments: (documentIds: number[]) =>
    request<BulkOpResult>("/documents/bulk-delete", {
      method: "POST",
      body: JSON.stringify({ document_ids: documentIds }),
    }),

  // No page moves documents yet: U17 (document detail) is the one that will.
  bulkMoveDocuments: (documentIds: number[], targetCollectionId: number) =>
    request<BulkOpResult>("/documents/bulk-move", {
      method: "POST",
      body: JSON.stringify({
        document_ids: documentIds,
        target_collection_id: targetCollectionId,
      }),
    }),

  uploadAsync: (
    collectionId: number,
    file: File,
    opts: { title?: string; signal?: AbortSignal } = {},
  ) => {
    const fd = new FormData();
    fd.append("collection_id", String(collectionId));
    fd.append("file", file);
    if (opts.title) fd.append("title", opts.title);
    return request<{ job_id: number; status: IngestionStatus }>("/documents/upload-async", {
      method: "POST",
      body: fd,
      signal: opts.signal,
    });
  },

  getIngestionJob: (jobId: number) => request<IngestionJob>(`/documents/jobs/${jobId}`),

  listIngestionJobs: (collectionId?: number, limit = 50) => {
    const q = new URLSearchParams();
    if (collectionId !== undefined) q.set("collection_id", String(collectionId));
    q.set("limit", String(limit));
    return request<IngestionJob[]>(`/documents/jobs?${q.toString()}`);
  },

  streamIngestionJob: async (
    jobId: number,
    onEvent: (e: IngestionJobEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> => {
    const res = await fetch(`${ROOT}/documents/jobs/${jobId}/stream`, {
      method: "GET",
      headers: { Accept: "text/event-stream", ...keyHeader() },
      signal,
    });
    if (!res.ok) throw await toApiError(res);
    for await (const message of readSse(res)) {
      const e = ingestionEvent(message);
      if (e) onEvent(e);
    }
  },

  query: (body: {
    question: string;
    collection_id: number | null;
    top_k: number;
    session_id?: string;
  }) => request<QueryResponse>("/query", { method: "POST", body: JSON.stringify(body) }),
  // `source`: the chat's answers or the Query page's; both when left out.
  history: async (limit = 50, source?: LogSource): Promise<QueryLog[]> => {
    const q = new URLSearchParams({ limit: String(limit) });
    if (source) q.set("source", source);
    const page = await request<Page<QueryLog>>(`/query/history?${q.toString()}`);
    return page.items;
  },
  historyPage: (params: {
    limit?: number;
    cursor?: number;
    session_id?: string;
    collection_id?: number;
    source?: LogSource;
    // Text the question contains (literal, case-insensitive).
    q?: string;
  }) => {
    const q = new URLSearchParams();
    if (params.limit !== undefined) q.set("limit", String(params.limit));
    if (params.cursor !== undefined) q.set("cursor", String(params.cursor));
    if (params.session_id) q.set("session_id", params.session_id);
    if (params.collection_id !== undefined) q.set("collection_id", String(params.collection_id));
    if (params.source) q.set("source", params.source);
    if (params.q?.trim()) q.set("q", params.q.trim());
    return request<Page<QueryLog>>(`/query/history?${q.toString()}`);
  },

  agentChatStream: async (
    body: {
      message: string;
      thread_id?: string | null;
      collection_id?: number | null;
      document_id?: number | null;
      top_k?: number;
    },
    onEvent: (e: AgentStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> => {
    const res = await fetch(`${ROOT}/agent/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        ...keyHeader(),
      },
      body: JSON.stringify({
        message: body.message,
        thread_id: body.thread_id ?? null,
        collection_id: body.collection_id ?? null,
        document_id: body.document_id ?? null,
        top_k: body.top_k ?? 5,
      }),
      signal,
    });
    if (!res.ok) throw await toApiError(res);
    for await (const message of readSse(res)) {
      const e = agentEvent(message);
      if (e) onEvent(e);
    }
  },

  listConversationsPage: (params: {
    limit?: number;
    cursor?: number;
    user_number?: string;
    is_saved?: boolean;
    channel?: string;
    // Text the thread's title contains (literal, case-insensitive).
    q?: string;
  }) => {
    const q = new URLSearchParams();
    if (params.limit !== undefined) q.set("limit", String(params.limit));
    if (params.cursor !== undefined) q.set("cursor", String(params.cursor));
    if (params.user_number) q.set("user_number", params.user_number);
    if (params.is_saved !== undefined) q.set("is_saved", String(params.is_saved));
    if (params.channel) q.set("channel", params.channel);
    if (params.q?.trim()) q.set("q", params.q.trim());
    return request<Page<Conversation>>(`/conversations?${q.toString()}`);
  },

  getConversation: (threadId: string) => request<Conversation>(`/conversations/${threadId}`),
  getConversationMessages: (threadId: string) =>
    request<ConversationMessage[]>(`/conversations/${threadId}/messages`),

  patchConversation: (threadId: string, body: { is_saved: boolean }) =>
    request<Conversation>(`/conversations/${threadId}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  deleteConversation: (threadId: string) =>
    request<void>(`/conversations/${threadId}`, { method: "DELETE" }),

  createNote: (threadId: string, body: { content: string; user_note?: string | null }) =>
    request<SavedNote>(`/conversations/${threadId}/notes`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listNotes: (params?: { limit?: number; collection_id?: number; conversation_id?: number }) => {
    const q = new URLSearchParams();
    if (params?.limit !== undefined) q.set("limit", String(params.limit));
    if (params?.collection_id !== undefined) q.set("collection_id", String(params.collection_id));
    if (params?.conversation_id !== undefined)
      q.set("conversation_id", String(params.conversation_id));
    const qs = q.toString();
    return request<SavedNote[]>(`/notes${qs ? `?${qs}` : ""}`);
  },

  deleteNote: (noteId: number) => request<void>(`/notes/${noteId}`, { method: "DELETE" }),

  evalResults: (limit = 20) => request<EvalRunSummary[]>(`/eval/results?limit=${limit}`),
  getEvalRun: (runId: string) => request<EvalReport>(`/eval/results/${encodeURIComponent(runId)}`),
  runEval: (body: {
    dataset: EvalItem[];
    collection_id: number;
    top_k: number;
    concurrency: number;
  }) => request<EvalReport>("/eval/run", { method: "POST", body: JSON.stringify(body) }),
};

// ============ Helpers ============
export function fmtTimestamp(iso?: string | Date): string {
  const d = iso ? new Date(iso) : new Date();
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  const ss = String(d.getUTCSeconds()).padStart(2, "0");
  const ms = String(d.getUTCMilliseconds()).padStart(3, "0");
  return `${hh}:${mm}:${ss}.${ms} UTC`;
}

export function fmtDate(iso: string): string {
  const d = new Date(iso);
  const yyyy = d.getUTCFullYear();
  const mo = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  return `${yyyy}-${mo}-${dd} ${hh}:${mm} UTC`;
}

/** A turn's time: just the time for today's turns, with the date for older ones. */
export function fmtTurnTime(iso: string): string {
  const d = new Date(iso);
  const day = d.toISOString().slice(0, 10);
  const today = new Date().toISOString().slice(0, 10);
  return day === today ? fmtTimestamp(d) : `${day} ${fmtTimestamp(d)}`;
}

/** How long ago `iso` was, to the minute: "JUST NOW", "4 MIN AGO", "2 H AGO",
 *  "3 D AGO". */
export function fmtAgo(iso: string, now: number = Date.now()): string {
  const seconds = Math.max(0, (now - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "JUST NOW";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} MIN AGO`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} H AGO`;
  return `${Math.floor(hours / 24)} D AGO`;
}

export function fmtNumber(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return new Intl.NumberFormat("en-US").format(n);
}

export function fmtCost(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `$${n.toFixed(4)}`;
}

export const COSINE_HINT =
  "Cosine similarity between the chunk and the question. — means only full-text search found it.";

export function fmtCosine(cosine: number | null | undefined): string {
  return cosine === null || cosine === undefined ? "COS —" : `COS ${cosine.toFixed(3)}`;
}

export function fmtPct(n: number | null | undefined, decimals = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(decimals)}%`;
}

// "us.anthropic.claude-sonnet-4-5-20250929-v1:0" -> "claude-sonnet-4-5-20250929"
export function shortModel(model: string | null | undefined): string {
  if (!model) return "—";
  return model.replace(/^(us|eu|apac|global)\.anthropic\./, "").replace(/-v\d+:\d+$/, "");
}

export function truncate(s: string, n = 60): string {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

// API client for WhisperDocs FastAPI backend.
// Configure with VITE_API_BASE_URL env var; defaults to http://localhost:8000.

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

const ROOT = `${API_BASE_URL}/api/v1`;

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${ROOT}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail: unknown = null;
    try {
      detail = await res.json();
    } catch {
      try { detail = await res.text(); } catch { /* ignore */ }
    }
    const msg =
      (detail && typeof detail === "object" && "detail" in (detail as Record<string, unknown>)
        ? String((detail as Record<string, unknown>).detail)
        : `HTTP ${res.status}`);
    throw new ApiError(res.status, detail, msg);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ============ Types ============
export type Health = { status: string; database?: string; checks?: Record<string, string> };

export type Collection = {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
};

export type DocumentRead = {
  id: number;
  collection_id: number;
  title: string;
  filename: string;
  file_type: "pdf" | "md" | "txt" | string;
  chunk_count: number;
  total_tokens: number;
  created_at: string;
};

export type DocumentDetail = DocumentRead & { raw_text: string };

export type Chunk = {
  id: number;
  document_id: number;
  chunk_index: number;
  text: string;
  token_count: number;
};

export type Citation = {
  chunk_id: number;
  document_id: number;
  document_title: string;
  chunk_index: number;
  relevance: number;
  text_preview?: string | null;
};

export type QueryMetadata = {
  model: string;
  tokens_input: number;
  tokens_output: number;
  latency_ms: number;
  chunks_retrieved: number;
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
  session_id: string | null;
  agent_action: string | null;
  created_at: string;
};

export type EvalMetrics = {
  total: number;
  accuracy: number;
  hallucination_rate: number;
  avg_retrieval_precision: number;
  avg_retrieval_recall: number;
  avg_latency_ms: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  avg_tokens_input: number;
  avg_tokens_output: number;
  total_tokens_input: number;
  total_tokens_output: number;
  estimated_cost_usd: number;
  by_difficulty?: Record<string, { count: number; accuracy: number; hallucination_rate: number }>;
  by_category?: Record<string, { count: number; accuracy: number; hallucination_rate: number }>;
};

export type EvalReport = {
  run_id: string;
  timestamp: string;
  dataset_size: number;
  model: string;
  metrics: EvalMetrics;
  results: unknown[];
};

export type EvalRunSummary = {
  run_id: string;
  timestamp: string;
  dataset_size: number;
  model: string;
  accuracy: number;
  hallucination_rate: number;
  estimated_cost_usd: number;
  status?: string;
};

// ============ Agent / Conversations / Notes ============
export type AgentCitation = Citation;

export type AgentTurnMetadata = {
  model: string | null;
  tokens_input: number;
  tokens_output: number;
  latency_ms: number;
  chunks_retrieved: number;
};

export type AgentChatResponse = {
  thread_id: string;
  action: string | null;
  answer: string;
  citations: AgentCitation[];
  metadata: AgentTurnMetadata;
};

export type AgentStreamEvent =
  | { type: "thread"; thread_id: string }
  | { type: "node_start"; node: string }
  | { type: "node_end"; node: string; summary: { action?: string; chunks?: number } }
  | { type: "token"; node: string | null; delta: string }
  | { type: "final"; payload: AgentChatResponse }
  | { type: "error"; message: string };

export type IngestionStatus =
  | "pending"
  | "extracting"
  | "chunking"
  | "embedding"
  | "persisting"
  | "complete"
  | "failed";

export type Page<T> = {
  items: T[];
  next_cursor: number | null;
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
  is_saved: boolean;
  last_active_at: string;
  created_at: string;
};

export type ConversationMessage = {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string | null;
  citations: Citation[] | null;
  metadata: AgentTurnMetadata | null;
};

export type SavedNote = {
  id: number;
  conversation_id: number | null;
  collection_id: number | null;
  source_message_content: string;
  user_note: string | null;
  created_at: string;
};

// ============ Endpoints ============
export const api = {
  health: () => request<Health>("/health"),
  adminSettings: () =>
    request<{
      models: Record<string, { model: string; temperature: number }>;
      retrieval: { top_k: number };
      compaction: { token_threshold: number; count_threshold: number; k_recent: number };
      embedding: { model: string; dimension: number };
    }>("/admin/settings"),

  listCollections: () => request<Collection[]>("/collections"),
  getCollection: (id: number) => request<Collection>(`/collections/${id}`),
  createCollection: (body: { name: string; description?: string }) =>
    request<Collection>("/collections", { method: "POST", body: JSON.stringify(body) }),
  deleteCollection: (id: number) =>
    request<void>(`/collections/${id}`, { method: "DELETE" }),

  listDocuments: async (collectionId?: number, limit = 50): Promise<DocumentRead[]> => {
    const q = new URLSearchParams();
    if (collectionId !== undefined) q.set("collection_id", String(collectionId));
    q.set("limit", String(limit));
    const page = await request<Page<DocumentRead>>(`/documents?${q.toString()}`);
    return page.items;
  },
  listDocumentsPage: (params: { collection_id?: number; limit?: number; cursor?: number }) => {
    const q = new URLSearchParams();
    if (params.collection_id !== undefined) q.set("collection_id", String(params.collection_id));
    if (params.limit !== undefined) q.set("limit", String(params.limit));
    if (params.cursor !== undefined) q.set("cursor", String(params.cursor));
    return request<Page<DocumentRead>>(`/documents?${q.toString()}`);
  },
  getDocument: (id: number) => request<DocumentDetail>(`/documents/${id}`),
  getChunks: (id: number) => request<Chunk[]>(`/documents/${id}/chunks`),
  deleteDocument: (id: number) =>
    request<void>(`/documents/${id}`, { method: "DELETE" }),
  uploadBatch: (collectionId: number, files: File[]) => {
    const fd = new FormData();
    fd.append("collection_id", String(collectionId));
    files.forEach((f) => fd.append("files", f));
    return request<{
      total: number;
      succeeded: number;
      failed: number;
      results: { status: string; filename: string; document: DocumentRead | null; error: string | null }[];
    }>("/documents/upload-batch", { method: "POST", body: fd });
  },

  bulkDeleteDocuments: (documentIds: number[]) =>
    request<BulkOpResult>("/documents/bulk-delete", {
      method: "POST",
      body: JSON.stringify({ document_ids: documentIds }),
    }),

  bulkMoveDocuments: (documentIds: number[], targetCollectionId: number) =>
    request<BulkOpResult>("/documents/bulk-move", {
      method: "POST",
      body: JSON.stringify({
        document_ids: documentIds,
        target_collection_id: targetCollectionId,
      }),
    }),

  uploadAsync: (collectionId: number, file: File, title?: string) => {
    const fd = new FormData();
    fd.append("collection_id", String(collectionId));
    fd.append("file", file);
    if (title) fd.append("title", title);
    return request<{ job_id: number; status: IngestionStatus }>("/documents/upload-async", {
      method: "POST",
      body: fd,
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
      headers: { Accept: "text/event-stream" },
      signal,
    });
    if (!res.ok || !res.body) {
      throw new ApiError(res.status, null, `Stream failed: HTTP ${res.status}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let sep = buf.indexOf("\n\n");
      while (sep !== -1) {
        const raw = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        sep = buf.indexOf("\n\n");
        let evtName = "message";
        let dataLine = "";
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) evtName = line.slice(6).trim();
          else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
        }
        if (!dataLine) continue;
        let parsed: unknown;
        try {
          parsed = JSON.parse(dataLine);
        } catch {
          continue;
        }
        const data = parsed as Record<string, unknown>;
        if (evtName === "progress") {
          onEvent({
            type: "progress",
            job_id: Number(data.job_id),
            status: data.status as IngestionStatus,
            progress: Number(data.progress ?? 0),
            processed_chunks: Number(data.processed_chunks ?? 0),
            total_chunks: Number(data.total_chunks ?? 0),
          });
        } else if (evtName === "complete") {
          onEvent({
            type: "complete",
            job_id: Number(data.job_id),
            document_id: (data.document_id as number | null) ?? null,
          });
        } else if (evtName === "error") {
          onEvent({
            type: "error",
            job_id: data.job_id as number | undefined,
            message: String(data.message ?? "ingestion error"),
          });
        }
      }
    }
  },

  query: (body: { question: string; collection_id: number | null; top_k: number; session_id?: string }) =>
    request<QueryResponse>("/query", { method: "POST", body: JSON.stringify(body) }),
  history: async (limit = 50): Promise<QueryLog[]> => {
    const page = await request<Page<QueryLog>>(`/query/history?limit=${limit}`);
    return page.items;
  },
  historyPage: (params: { limit?: number; cursor?: number; session_id?: string }) => {
    const q = new URLSearchParams();
    if (params.limit !== undefined) q.set("limit", String(params.limit));
    if (params.cursor !== undefined) q.set("cursor", String(params.cursor));
    if (params.session_id) q.set("session_id", params.session_id);
    return request<Page<QueryLog>>(`/query/history?${q.toString()}`);
  },

  agentChat: (body: { message: string; thread_id?: string | null; collection_id?: number | null; top_k?: number }) =>
    request<AgentChatResponse>("/agent/chat", {
      method: "POST",
      body: JSON.stringify({
        message: body.message,
        thread_id: body.thread_id ?? null,
        collection_id: body.collection_id ?? null,
      }),
    }),

  agentChatStream: async (
    body: { message: string; thread_id?: string | null; collection_id?: number | null; top_k?: number },
    onEvent: (e: AgentStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> => {
    const res = await fetch(`${ROOT}/agent/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        message: body.message,
        thread_id: body.thread_id ?? null,
        collection_id: body.collection_id ?? null,
        top_k: body.top_k ?? 5,
      }),
      signal,
    });
    if (!res.ok || !res.body) {
      throw new ApiError(res.status, null, `Stream failed: HTTP ${res.status}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let sep = buf.indexOf("\n\n");
      while (sep !== -1) {
        const raw = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        sep = buf.indexOf("\n\n");
        let evtName = "message";
        let dataLine = "";
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) evtName = line.slice(6).trim();
          else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
        }
        if (!dataLine) continue;
        let parsed: unknown;
        try {
          parsed = JSON.parse(dataLine);
        } catch {
          continue;
        }
        const data = parsed as Record<string, unknown>;
        if (evtName === "thread")
          onEvent({ type: "thread", thread_id: String(data.thread_id) });
        else if (evtName === "node_start")
          onEvent({ type: "node_start", node: String(data.node) });
        else if (evtName === "node_end")
          onEvent({
            type: "node_end",
            node: String(data.node),
            summary: (data.summary as { action?: string; chunks?: number }) ?? {},
          });
        else if (evtName === "token")
          onEvent({
            type: "token",
            node: (data.node as string | null) ?? null,
            delta: String(data.delta ?? ""),
          });
        else if (evtName === "final")
          onEvent({
            type: "final",
            payload: {
              thread_id: String(data.thread_id),
              action: (data.action as string | null) ?? null,
              answer: String(data.answer ?? ""),
              citations: (data.citations as AgentCitation[]) ?? [],
              metadata: data.metadata as AgentTurnMetadata,
            },
          });
        else if (evtName === "error")
          onEvent({ type: "error", message: String(data.message ?? "stream error") });
      }
    }
  },

  listConversations: async (params?: {
    limit?: number;
    user_number?: string;
    is_saved?: boolean;
    channel?: string;
  }): Promise<Conversation[]> => {
    const q = new URLSearchParams();
    if (params?.limit !== undefined) q.set("limit", String(params.limit));
    if (params?.user_number) q.set("user_number", params.user_number);
    if (params?.is_saved !== undefined) q.set("is_saved", String(params.is_saved));
    if (params?.channel) q.set("channel", params.channel);
    const qs = q.toString();
    const page = await request<Page<Conversation>>(`/conversations${qs ? `?${qs}` : ""}`);
    return page.items;
  },
  listConversationsPage: (params: {
    limit?: number;
    cursor?: number;
    user_number?: string;
    is_saved?: boolean;
    channel?: string;
  }) => {
    const q = new URLSearchParams();
    if (params.limit !== undefined) q.set("limit", String(params.limit));
    if (params.cursor !== undefined) q.set("cursor", String(params.cursor));
    if (params.user_number) q.set("user_number", params.user_number);
    if (params.is_saved !== undefined) q.set("is_saved", String(params.is_saved));
    if (params.channel) q.set("channel", params.channel);
    return request<Page<Conversation>>(`/conversations?${q.toString()}`);
  },

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

  deleteNote: (noteId: number) =>
    request<void>(`/notes/${noteId}`, { method: "DELETE" }),

  evalResults: (limit = 20) => request<EvalRunSummary[]>(`/eval/results?limit=${limit}`),
  runEval: (body: {
    dataset: unknown[];
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
  return `${yyyy}-${mo}-${dd} ${hh}:${mm}`;
}

export function fmtNumber(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return new Intl.NumberFormat("en-US").format(n);
}

export function fmtCost(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `$${n.toFixed(4)}`;
}

export function fmtPct(n: number | null | undefined, decimals = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(decimals)}%`;
}

export function truncate(s: string, n = 60): string {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

const MODEL_COST_PER_1M: Record<string, { input: number; output: number }> = {
  "gpt-4o-mini":                                    { input: 0.15,  output: 0.60  },
  "gpt-4o":                                         { input: 2.50,  output: 10.00 },
  "gemini-2.5-flash":                               { input: 0.15,  output: 0.60  },
  "us.anthropic.claude-sonnet-4-20250514-v1:0":     { input: 3.00,  output: 15.00 },
  "us.anthropic.claude-haiku-4-5-20251001-v1:0":    { input: 0.80,  output: 4.00  },
};

export function estimateCost(
  tokensIn: number,
  tokensOut: number,
  model: string | null | undefined,
): string {
  if (!model) return fmtCost((tokensIn + tokensOut) / 1_000_000 * 0.5);
  const rates = MODEL_COST_PER_1M[model];
  if (!rates) return fmtCost((tokensIn + tokensOut) / 1_000_000 * 0.5);
  const cost = (tokensIn / 1_000_000) * rates.input + (tokensOut / 1_000_000) * rates.output;
  return fmtCost(cost);
}

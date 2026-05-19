import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, estimateCost, fmtTimestamp, type Citation, type QueryLog, type QueryMetadata } from "@/lib/api";
import { ErrorBanner } from "@/components/ui-kit/EmptyState";

export const Route = createFileRoute("/consulta")({
  head: () => ({
    meta: [
      { title: "Consulta — WhisperDocs" },
      { name: "description", content: "Conversational query session against the WhisperDocs knowledge base." },
    ],
  }),
  component: ConsultaPage,
});

type Turn =
  | { kind: "user"; ts: string; text: string }
  | {
      kind: "assistant";
      ts: string;
      text: string;
      citations: Citation[];
      metadata: QueryMetadata;
      expanded?: boolean;
    };

type Session = {
  id: string;
  turns: Turn[];
  createdAt: string;
};

function newSessionId() {
  return "SES-" + Math.floor(10000 + Math.random() * 89999);
}
function newSession(): Session {
  return { id: newSessionId(), turns: [], createdAt: new Date().toISOString() };
}

function logsToSession(sessionId: string, logs: QueryLog[]): Session {
  const sorted = [...logs].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
  const turns: Turn[] = sorted.flatMap((log) => [
    { kind: "user" as const, ts: fmtTimestamp(log.created_at), text: log.question },
    {
      kind: "assistant" as const,
      ts: fmtTimestamp(log.created_at),
      text: log.answer,
      citations: [],
      metadata: {
        model: log.model_used,
        tokens_input: log.tokens_input,
        tokens_output: log.tokens_output,
        latency_ms: log.latency_ms,
        chunks_retrieved: log.chunks_retrieved,
      },
    },
  ]);
  return { id: sessionId, turns, createdAt: sorted[0]?.created_at ?? new Date().toISOString() };
}

const EXAMPLES = [
  "What distance metrics does pgvector support?",
  "How does selectinload work in SQLAlchemy async?",
  "Compare OpenAI vs Anthropic pricing",
];

const PROGRESS_STEPS = [
  "EMBEDDING QUERY...",
  "RETRIEVING CHUNKS...",
  "GENERATING RESPONSE...",
];

function ConsultaPage() {
  const [sessions, setSessions] = useState<Session[]>(() => [newSession()]);
  const [activeId, setActiveId] = useState<string>(() => sessions[0].id);
  const [collectionId, setCollectionId] = useState<number | null>(null);
  const [topK, setTopK] = useState(5);
  const [input, setInput] = useState("");
  const [progressIdx, setProgressIdx] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const historyQuery = useQuery({ queryKey: ["query-history"], queryFn: () => api.history(200), staleTime: Infinity });
  const initialized = useRef(false);

  const active = useMemo(
    () => sessions.find((s) => s.id === activeId) ?? sessions[0],
    [sessions, activeId],
  );

  const queryMut = useMutation({
    mutationFn: (q: string) => api.query({ question: q, collection_id: collectionId, top_k: topK, session_id: active.id }),
  });

  useEffect(() => {
    if (initialized.current || !historyQuery.data) return;
    initialized.current = true;
    const grouped = new Map<string, QueryLog[]>();
    for (const log of historyQuery.data) {
      if (!log.session_id) continue;
      if (!grouped.has(log.session_id)) grouped.set(log.session_id, []);
      grouped.get(log.session_id)!.push(log);
    }
    const historySessions = [...grouped.entries()]
      .map(([id, logs]) => logsToSession(id, logs))
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
    if (historySessions.length === 0) return;
    setSessions((prev) => {
      const existingIds = new Set(prev.map((s) => s.id));
      const fresh = historySessions.filter((s) => !existingIds.has(s.id));
      return fresh.length > 0 ? [...prev, ...fresh] : prev;
    });
  }, [historyQuery.data]);

  // Progressive status while generating
  useEffect(() => {
    if (!queryMut.isPending) return;
    setProgressIdx(0);
    const id = setInterval(() => setProgressIdx((i) => (i + 1) % PROGRESS_STEPS.length), 900);
    return () => clearInterval(id);
  }, [queryMut.isPending]);

  // Autoscroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [active?.turns.length, queryMut.isPending]);

  function patchActive(fn: (s: Session) => Session) {
    setSessions((prev) => prev.map((s) => (s.id === activeId ? fn(s) : s)));
  }

  async function send(question: string) {
    if (!question.trim()) return;
    setErrorMsg(null);
    const userTurn: Turn = { kind: "user", ts: fmtTimestamp(), text: question.trim() };
    patchActive((s) => ({ ...s, turns: [...s.turns, userTurn] }));
    setInput("");
    try {
      const resp = await queryMut.mutateAsync(question.trim());
      const aTurn: Turn = {
        kind: "assistant",
        ts: fmtTimestamp(),
        text: resp.answer,
        citations: resp.citations,
        metadata: resp.metadata,
      };
      patchActive((s) => ({ ...s, turns: [...s.turns, aTurn] }));
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErrorMsg(msg);
    }
  }

  function startNew() {
    const s = newSession();
    setSessions((prev) => [s, ...prev]);
    setActiveId(s.id);
  }

  return (
    <div className="-m-3 sm:-m-6 md:-m-8 flex h-[calc(100vh-48px)] md:h-[calc(100vh-48px-28px)]">
      {/* Conversation list — desktop only */}
      <aside className="hidden md:flex w-64 shrink-0 border-r border-border-subtle bg-surface-low flex-col" aria-label="Sessions">
        <div className="px-4 h-12 flex items-center justify-between border-b border-border-subtle">
          <span className="label-sm text-text-secondary">SESSIONS</span>
          <button
            onClick={startNew}
            className="label-sm text-text-primary hover:bg-text-primary hover:text-surface px-2 h-6"
            aria-label="New session"
          >
            + NEW
          </button>
        </div>
        <div className="flex-1 overflow-auto">
          {historyQuery.isLoading && (
            <div className="px-4 py-2 label-xs text-text-tertiary border-b border-border-subtle/50">
              LOADING HISTORY...
            </div>
          )}
          {sessions.map((s) => {
            const preview = s.turns.find((t) => t.kind === "user")?.text ?? "(empty session)";
            const isActive = s.id === activeId;
            return (
              <button
                key={s.id}
                onClick={() => setActiveId(s.id)}
                className={`relative w-full text-left px-4 py-3 border-b border-border-subtle/50 hover:bg-surface-container transition-colors ${
                  isActive ? "bg-surface-container" : ""
                }`}
              >
                {isActive && (
                  <span className="absolute left-0 top-0 bottom-0 w-[2px] bg-text-primary" />
                )}
                <div className="label-sm text-text-primary">{s.id}</div>
                <div className="body-sm text-text-secondary truncate mt-1">
                  {preview.length > 40 ? preview.slice(0, 39) + "…" : preview}
                </div>
                <div className="label-xs text-text-tertiary mt-1">
                  {new Date(s.createdAt).toLocaleTimeString("en-GB")}
                </div>
              </button>
            );
          })}
        </div>
      </aside>

      {/* Main conversation */}
      <section className="flex-1 min-w-0 flex flex-col bg-surface">
        {/* Session header */}
        <div className="h-12 px-6 flex items-center justify-between border-b border-border-subtle shrink-0">
          <div className="label-sm text-text-secondary">
            WHISPER_SESSION: <span className="text-text-primary">{active.id}</span>
          </div>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 label-sm text-text-secondary">
              COLLECTION:
              <select
                value={collectionId ?? ""}
                onChange={(e) => setCollectionId(e.target.value ? Number(e.target.value) : null)}
                className="bg-surface-lowest border border-border-subtle px-2 h-7 text-text-primary body-sm focus:outline-none focus:border-border-focus"
              >
                <option value="">ALL</option>
                {collections.data?.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-2 label-sm text-text-secondary">
              TOP_K:
              <input
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Math.max(1, Math.min(20, Number(e.target.value) || 5)))}
                className="bg-surface-lowest border border-border-subtle px-2 h-7 w-14 text-text-primary body-sm tabular-nums focus:outline-none focus:border-border-focus"
              />
            </label>
          </div>
        </div>

        {/* Conversation area */}
        <div ref={scrollRef} className="flex-1 overflow-auto px-8 py-6">
          {errorMsg && <ErrorBanner message={errorMsg} />}

          {active.turns.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto">
              <div className="title-lg text-text-primary blink-cursor">WHISPER_SESSION INITIALIZED</div>
              <div className="body-sm text-text-secondary mt-3">
                Ready to query. Select a collection and ask a question.
              </div>
              <div className="mt-10 w-full text-left space-y-2">
                <div className="label-sm text-text-tertiary mb-3">EXAMPLE QUERIES</div>
                {EXAMPLES.map((q) => (
                  <button
                    key={q}
                    onClick={() => send(q)}
                    className="block w-full text-left px-4 py-2 body-sm text-text-secondary hover:bg-surface-container hover:text-text-primary transition-colors border border-border-subtle"
                  >
                    <span className="text-text-primary mr-2">{">"}</span>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6 max-w-4xl mx-auto pb-4">
              {active.turns.map((t, i) =>
                t.kind === "user" ? (
                  <UserTurn key={i} turn={t} />
                ) : (
                  <AssistantTurn
                    key={i}
                    turn={t}
                    onToggle={() =>
                      patchActive((s) => ({
                        ...s,
                        turns: s.turns.map((tt, j) =>
                          j === i && tt.kind === "assistant"
                            ? { ...tt, expanded: !tt.expanded }
                            : tt,
                        ),
                      }))
                    }
                  />
                ),
              )}

              {queryMut.isPending && (
                <div className="border-l-2 border-text-primary pl-4 py-2 space-y-1">
                  {PROGRESS_STEPS.slice(0, progressIdx + 1).map((s, i) => (
                    <div
                      key={s}
                      className="label-sm text-text-secondary"
                      style={{
                        color: i === progressIdx ? "var(--text-primary)" : "var(--text-tertiary)",
                      }}
                    >
                      {s} {i === progressIdx ? "▌" : "✓"}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-border-subtle px-8 py-4 shrink-0">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (!queryMut.isPending) send(input);
            }}
            className="max-w-4xl mx-auto flex items-center gap-3 border-b border-border-subtle focus-within:border-border-focus pb-2"
          >
            <span className="text-text-primary body-sm">{">"}</span>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={queryMut.isPending}
              placeholder="ENTER QUERY..."
              aria-label="Query input"
              className="flex-1 min-w-0 bg-transparent border-0 outline-none text-text-primary placeholder:text-text-tertiary placeholder:label-sm body-sm py-2"
            />
            <button
              type="submit"
              disabled={queryMut.isPending || !input.trim()}
              className="label-sm text-text-primary disabled:text-text-tertiary hover:bg-text-primary hover:text-surface px-2 h-7 disabled:hover:bg-transparent disabled:hover:text-text-tertiary"
            >
              ↵ SEND
            </button>
          </form>
        </div>
      </section>
    </div>
  );
}

function UserTurn({ turn }: { turn: Extract<Turn, { kind: "user" }> }) {
  return (
    <div>
      <div className="label-sm text-text-tertiary mb-2">
        {">"} USER · <span className="text-text-secondary">{turn.ts}</span>
      </div>
      <div className="body text-text-primary whitespace-pre-wrap">{turn.text}</div>
      <div className="border-b border-border-subtle mt-4" />
    </div>
  );
}

function AssistantTurn({
  turn,
  onToggle,
}: {
  turn: Extract<Turn, { kind: "assistant" }>;
  onToggle: () => void;
}) {
  // Render answer with [N] citations highlighted
  const parts = turn.text.split(/(\[\d+\])/g);
  return (
    <div>
      <div className="label-sm text-text-tertiary mb-2">
        {"<"} WHISPER · <span className="text-text-secondary">{turn.ts}</span>
      </div>
      <div className="body text-text-primary whitespace-pre-wrap leading-relaxed">
        {parts.map((p, i) =>
          /^\[\d+\]$/.test(p) ? (
            <a
              key={i}
              href={`#cite-${turn.ts}-${p.replace(/\D/g, "")}`}
              className="text-text-primary font-semibold mx-0.5 hover:bg-text-primary hover:text-surface px-0.5"
            >
              {p}
            </a>
          ) : (
            <span key={i}>{p}</span>
          ),
        )}
      </div>

      {turn.citations.length > 0 && (
        <div className="mt-4">
          <button
            onClick={onToggle}
            className="label-sm text-text-secondary hover:text-text-primary"
          >
            {turn.expanded ? "▾" : "▸"} CITAS [{turn.citations.length}]
          </button>
          {turn.expanded && (
            <div className="mt-3 space-y-2">
              {turn.citations.map((c, idx) => {
                const rel = c.relevance;
                const relClass = rel >= 0.7 ? "text-success" : rel >= 0.5 ? "text-text-tertiary" : "text-error";
                const num = String(idx + 1).padStart(2, "0");
                return (
                  <div
                    id={`cite-${turn.ts}-${idx + 1}`}
                    key={c.chunk_id}
                    className="border border-border-subtle bg-surface-container-low p-3"
                  >
                    <div className="label-sm flex items-center gap-3">
                      <span className="text-text-primary">[{num}]</span>
                      <span className="text-text-secondary">{c.document_title}</span>
                      <span className="text-text-tertiary">·</span>
                      <span className="text-text-secondary">CHUNK {c.chunk_index}</span>
                      <span className="text-text-tertiary">·</span>
                      <span className={`tabular-nums ${relClass}`}>
                        RELEVANCE {rel.toFixed(3)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      <div className="label-xs text-text-tertiary mt-4 tabular-nums">
        {turn.metadata.latency_ms}MS · {turn.metadata.tokens_input.toLocaleString()} IN /{" "}
        {turn.metadata.tokens_output.toLocaleString()} OUT · {turn.metadata.model} ·{" "}
        {estimateCost(turn.metadata.tokens_input, turn.metadata.tokens_output, turn.metadata.model)}
      </div>
      <div className="border-b border-border-subtle mt-4" />
    </div>
  );
}

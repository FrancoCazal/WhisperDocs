import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  COSINE_HINT,
  fmtCosine,
  fmtCost,
  fmtDate,
  fmtNumber,
  fmtTimestamp,
  fmtTurnTime,
  LIMITS,
  type Citation,
  type QueryLog,
  type QueryMetadata,
} from "@/lib/api";
import { ChunkPlace } from "@/components/ui-kit/ChunkPlace";
import { ErrorBanner, LoadingLine } from "@/components/ui-kit/EmptyState";
import { Composer } from "@/components/ui-kit/Composer";
import { ScopeControls } from "@/components/ui-kit/ScopeControls";
import { useTopK } from "@/lib/topK";
import { Btn, btnCls } from "@/components/ui-kit/Button";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";

export const Route = createFileRoute("/query")({
  // Client only: a new session's id is random, so a server render could never
  // match the client's, and the history comes from a client query anyway.
  ssr: false,
  head: () => ({
    meta: [
      { title: "Query — WhisperDocs" },
      {
        name: "description",
        content: "Conversational query session against the WhisperDocs knowledge base.",
      },
    ],
  }),
  component: QueryPage,
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
  const sorted = [...logs].sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
  );
  const turns: Turn[] = sorted.flatMap((log) => [
    { kind: "user" as const, ts: fmtTurnTime(log.created_at), text: log.question },
    {
      kind: "assistant" as const,
      ts: fmtTurnTime(log.created_at),
      text: log.answer,
      citations: [],
      metadata: {
        model: log.model_used,
        tokens_input: log.tokens_input,
        tokens_output: log.tokens_output,
        latency_ms: log.latency_ms,
        chunks_retrieved: log.chunks_retrieved,
        cost_usd: log.cost_usd,
      },
    },
  ]);
  return { id: sessionId, turns, createdAt: sorted[0]?.created_at ?? new Date().toISOString() };
}

/** The logged sessions of this page, newest first. */
function sessionsFromLogs(logs: QueryLog[]): Session[] {
  const grouped = new Map<string, QueryLog[]>();
  for (const log of logs) {
    if (!log.session_id) continue;
    if (!grouped.has(log.session_id)) grouped.set(log.session_id, []);
    grouped.get(log.session_id)!.push(log);
  }
  return [...grouped.entries()]
    .map(([id, group]) => logsToSession(id, group))
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
}

const EXAMPLES = [
  "What distance metrics does pgvector support?",
  "How does selectinload work in SQLAlchemy async?",
  "Compare OpenAI vs Anthropic pricing",
];

// The sessions, as a list: the desktop sidebar and the mobile panel show the same one.
function SessionList({
  sessions,
  activeId,
  loading,
  onPick,
}: {
  sessions: Session[];
  activeId: string;
  loading: boolean;
  onPick: (id: string) => void;
}) {
  return (
    <div className="flex-1 overflow-auto">
      {loading && (
        <div className="px-4 py-3 border-b border-border-subtle/50">
          <LoadingLine label="LOADING HISTORY" />
        </div>
      )}
      {sessions.map((s) => {
        const preview = s.turns.find((t) => t.kind === "user")?.text ?? "(empty session)";
        const isActive = s.id === activeId;
        return (
          <button
            key={s.id}
            onClick={() => onPick(s.id)}
            aria-current={isActive ? "true" : undefined}
            className={`relative w-full text-left px-4 py-3 border-b border-border-subtle/50 hover:bg-surface-container transition-colors focus-visible:-outline-offset-2 ${
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
            <div className="label-xs text-text-tertiary mt-1">{fmtDate(s.createdAt)}</div>
          </button>
        );
      })}
    </div>
  );
}

const PROGRESS_STEPS = ["EMBEDDING QUERY...", "RETRIEVING CHUNKS...", "GENERATING RESPONSE..."];

function QueryPage() {
  const qc = useQueryClient();
  const [sessions, setSessions] = useState<Session[]>(() => [newSession()]);
  const [activeId, setActiveId] = useState<string>(() => sessions[0].id);
  const [collectionId, setCollectionId] = useState<number | null>(null);
  const { topK, defaultTopK, setTopK } = useTopK();
  const [input, setInput] = useState("");
  const [progressIdx, setProgressIdx] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  // The mobile sessions panel: controlled, so picking a session closes it.
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const historyQuery = useQuery({
    queryKey: ["query-history"],
    // The Query page's own answers: the chat's are logged too.
    queryFn: () => api.history(200, "query"),
    staleTime: Infinity,
  });
  // History's sessions join the list once, when it first loads. Set during
  // render, as React suggests for state that follows other state.
  const [historyMerged, setHistoryMerged] = useState(false);
  if (!historyMerged && historyQuery.data) {
    setHistoryMerged(true);
    const fromHistory = sessionsFromLogs(historyQuery.data);
    if (fromHistory.length > 0)
      setSessions((prev) => {
        const existingIds = new Set(prev.map((s) => s.id));
        const fresh = fromHistory.filter((s) => !existingIds.has(s.id));
        return fresh.length > 0 ? [...prev, ...fresh] : prev;
      });
  }

  const active = useMemo(
    () => sessions.find((s) => s.id === activeId) ?? sessions[0],
    [sessions, activeId],
  );

  const queryMut = useMutation({
    mutationFn: (q: string) =>
      api.query({ question: q, collection_id: collectionId, top_k: topK, session_id: active.id }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["history"] }),
    // Worked or failed, the calls are on record: the failing-models notice.
    onSettled: () => qc.invalidateQueries({ queryKey: ["model-status"] }),
  });

  // Progressive status while generating; send() starts it at step 0.
  useEffect(() => {
    if (!queryMut.isPending) return;
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
    setProgressIdx(0);
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
      <aside
        className="hidden md:flex w-64 shrink-0 border-r border-border-subtle bg-surface-low flex-col"
        aria-label="Sessions"
      >
        <div className="px-4 h-12 flex items-center justify-between border-b border-border-subtle">
          <span className="label-sm text-text-secondary">SESSIONS</span>
          <Btn size="sm" variant="ghost" onClick={startNew} aria-label="New session">
            + NEW
          </Btn>
        </div>
        <SessionList
          sessions={sessions}
          activeId={active.id}
          loading={historyQuery.isLoading}
          onPick={setActiveId}
        />
      </aside>

      {/* Main conversation */}
      <section className="flex-1 min-w-0 flex flex-col bg-surface">
        {/* Session header */}
        <div className="min-h-12 px-3 sm:px-6 py-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-border-subtle shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            {/* Mobile: the sessions live in a panel. */}
            <Sheet open={sessionsOpen} onOpenChange={setSessionsOpen}>
              <SheetTrigger asChild>
                <button
                  type="button"
                  aria-label="Open sessions panel"
                  className={btnCls("quiet", "icon", "md:hidden")}
                >
                  ☰
                </button>
              </SheetTrigger>
              <SheetContent
                side="left"
                className="w-72 max-w-[85vw] bg-surface-low border-r border-border-subtle p-0 flex flex-col gap-0"
              >
                <SheetHeader className="px-4 py-3 border-b border-border-subtle flex-row items-center justify-between space-y-0 pr-12">
                  <SheetTitle className="label-sm text-text-primary">SESSIONS</SheetTitle>
                  <Btn
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setSessionsOpen(false);
                      startNew();
                    }}
                  >
                    + NEW
                  </Btn>
                </SheetHeader>
                <SessionList
                  sessions={sessions}
                  activeId={active.id}
                  loading={historyQuery.isLoading}
                  onPick={(id) => {
                    setSessionsOpen(false);
                    setActiveId(id);
                  }}
                />
              </SheetContent>
            </Sheet>
            <div className="label-sm text-text-secondary truncate">
              WHISPER_SESSION: <span className="text-text-primary">{active.id}</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <ScopeControls
              collections={collections.data}
              collectionId={collectionId}
              onCollection={setCollectionId}
              topK={topK}
              defaultTopK={defaultTopK}
              onTopK={setTopK}
            />
          </div>
        </div>

        {/* Conversation area */}
        <div ref={scrollRef} className="flex-1 overflow-auto px-3 sm:px-8 py-6">
          {errorMsg && <ErrorBanner message={errorMsg} />}

          {active.turns.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto">
              <div className="title-lg text-text-primary blink-cursor">
                WHISPER_SESSION INITIALIZED
              </div>
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
                    index={i}
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
                      className={`label-sm ${
                        i === progressIdx ? "text-text-primary" : "text-text-tertiary"
                      }`}
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
          {/* Typing stays open while an answer is generated, so the box keeps
              its focus; sending waits for the answer. */}
          <Composer
            value={input}
            onChange={setInput}
            onSubmit={() => {
              if (!queryMut.isPending) void send(input);
            }}
            maxLength={LIMITS.queryQuestion}
            placeholder="ENTER QUERY..."
            label="Query input"
          >
            <Btn
              type="submit"
              size="sm"
              variant="ghost"
              disabled={queryMut.isPending || !input.trim()}
            >
              ↵ SEND
            </Btn>
          </Composer>
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
      <div className="body-read text-text-primary whitespace-pre-wrap">{turn.text}</div>
      <div className="border-b border-border-subtle mt-4" />
    </div>
  );
}

function AssistantTurn({
  index,
  turn,
  onToggle,
}: {
  // The turn's place in the session: its citations' ids are unique with it.
  index: number;
  turn: Extract<Turn, { kind: "assistant" }>;
  onToggle: () => void;
}) {
  // Render answer with [N] citations highlighted
  const parts = turn.text.split(/(\[\d+\])/g);
  const citeId = (n: number | string) => `cite-${index}-${n}`;
  // A [n] opens the citations if they are closed, then scrolls to the one cited.
  const goToCite = (n: string) => {
    if (!turn.expanded) onToggle();
    requestAnimationFrame(() =>
      document.getElementById(citeId(n))?.scrollIntoView({ block: "nearest" }),
    );
  };
  return (
    <div>
      <div className="label-sm text-text-tertiary mb-2">
        {"<"} WHISPER · <span className="text-text-secondary">{turn.ts}</span>
      </div>
      <div className="body-read text-text-primary whitespace-pre-wrap">
        {parts.map((p, i) =>
          /^\[\d+\]$/.test(p) ? (
            <a
              key={i}
              href={`#${citeId(p.replace(/\D/g, ""))}`}
              onClick={(e) => {
                e.preventDefault();
                goToCite(p.replace(/\D/g, ""));
              }}
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
            {turn.expanded ? "▾" : "▸"} CITATIONS [{turn.citations.length}]
          </button>
          {turn.expanded && (
            <div className="mt-3 space-y-2">
              {turn.citations.map((c, idx) => {
                const num = String(idx + 1).padStart(2, "0");
                return (
                  <div
                    id={citeId(idx + 1)}
                    key={c.chunk_id}
                    className="border border-border-subtle bg-surface-container-low p-3"
                  >
                    <div className="label-sm flex items-center gap-3">
                      <span className="text-text-primary">[{num}]</span>
                      <span className="text-text-secondary">{c.document_title}</span>
                      <span className="text-text-tertiary">·</span>
                      <span className="text-text-secondary">
                        CHUNK {c.chunk_index}
                        <ChunkPlace c={c} />
                      </span>
                      <span className="text-text-tertiary">·</span>
                      <span className="tabular-nums text-text-secondary" title={COSINE_HINT}>
                        {fmtCosine(c.cosine)}
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
        {fmtNumber(turn.metadata.latency_ms)}MS · {fmtNumber(turn.metadata.tokens_input)} IN /{" "}
        {fmtNumber(turn.metadata.tokens_output)} OUT · {turn.metadata.model} ·{" "}
        {fmtCost(turn.metadata.cost_usd)}
      </div>
      <div className="border-b border-border-subtle mt-4" />
    </div>
  );
}

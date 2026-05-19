import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import {
  api,
  estimateCost,
  fmtDate,
  fmtTimestamp,
  type AgentCitation,
  type AgentTurnMetadata,
  type Conversation,
} from "@/lib/api";
import { ErrorBanner } from "@/components/ui-kit/EmptyState";
import { Modal } from "@/components/ui-kit/Modal";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";

export const Route = createFileRoute("/chat")({
  head: () => ({
    meta: [
      { title: "Chat — WhisperDocs" },
      {
        name: "description",
        content: "Agentic chat session: classifier + RAG + persistent memory.",
      },
    ],
  }),
  component: ChatPage,
});

type Turn =
  | { kind: "user"; ts: string; text: string }
  | {
      kind: "assistant";
      ts: string;
      text: string;
      action: string | null;
      citations: AgentCitation[];
      metadata: AgentTurnMetadata;
      expanded?: boolean;
    };

type NodeProgress = { node: string; status: "running" | "done"; summary?: { action?: string; chunks?: number } };

const EXAMPLES = [
  "What distance metrics does pgvector support?",
  "How does selectinload work in SQLAlchemy async?",
  "Hello, what can you do?",
];

const NODE_LABELS: Record<string, string> = {
  check_compaction: "CHECK MEMORY",
  summarize: "COMPACTING HISTORY",
  decide_action: "CLASSIFYING INTENT",
  search_docs: "RETRIEVING + GENERATING",
  ask_clarification: "ASKING CLARIFICATION",
  answer_direct: "ANSWERING DIRECTLY",
};

const ACTION_CLASS: Record<string, string> = {
  search_docs: "text-success border-success",
  answer_direct: "text-text-primary border-text-primary",
  ask_clarification: "text-text-tertiary border-text-tertiary",
};

function ChatPage() {
  const qc = useQueryClient();

  const [threadId, setThreadId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [collectionId, setCollectionId] = useState<number | null>(null);
  const [topK, setTopK] = useState(5);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamProgress, setStreamProgress] = useState<NodeProgress[]>([]);
  const [streamingText, setStreamingText] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<"threads" | "notes">("threads");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const conversations = useQuery({
    queryKey: ["conversations", "list"],
    queryFn: () => api.listConversations({ limit: 50 }),
  });
  const notes = useQuery({
    queryKey: ["notes"],
    queryFn: () => api.listNotes({ limit: 100 }),
    enabled: sidebarTab === "notes",
  });

  const activeConv: Conversation | undefined = useMemo(
    () => conversations.data?.find((c) => c.thread_id === threadId),
    [conversations.data, threadId],
  );

  const lastAssistantTurn = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i--) {
      const t = turns[i];
      if (t.kind === "assistant" && t.citations.length > 0) return t;
    }
    return null;
  }, [turns]);

  const patchSaveMut = useMutation({
    mutationFn: ({ tid, saved }: { tid: string; saved: boolean }) =>
      api.patchConversation(tid, { is_saved: saved }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations", "list"] }),
  });

  const deleteConvMut = useMutation({
    mutationFn: (tid: string) => api.deleteConversation(tid),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations", "list"] }),
  });

  const pinNoteMut = useMutation({
    mutationFn: ({ tid, content }: { tid: string; content: string }) =>
      api.createNote(tid, { content }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["notes"] }),
  });

  const deleteNoteMut = useMutation({
    mutationFn: (noteId: number) => api.deleteNote(noteId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["notes"] }),
  });

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [turns.length, streaming, streamProgress.length, streamingText]);

  useEffect(() => {
    if (!streaming) inputRef.current?.focus();
  }, [streaming]);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  async function send(message: string) {
    const trimmed = message.trim();
    if (!trimmed || streaming) return;
    setErrorMsg(null);
    const userTurn: Turn = { kind: "user", ts: fmtTimestamp(), text: trimmed };
    setTurns((prev) => [...prev, userTurn]);
    setInput("");
    setStreamProgress([]);
    setStreamingText("");
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;
    let finalPayload: import("@/lib/api").AgentChatResponse | null = null;
    let streamErr: string | null = null;

    try {
      await api.agentChatStream(
        { message: trimmed, thread_id: threadId, collection_id: collectionId, top_k: topK },
        (e) => {
          if (e.type === "thread") {
            if (!threadId) setThreadId(e.thread_id);
          } else if (e.type === "node_start") {
            setStreamProgress((prev) => [...prev, { node: e.node, status: "running" }]);
          } else if (e.type === "node_end") {
            setStreamProgress((prev) =>
              prev.map((p) =>
                p.node === e.node && p.status === "running"
                  ? { ...p, status: "done", summary: e.summary }
                  : p,
              ),
            );
          } else if (e.type === "token") {
            setStreamingText((prev) => prev + e.delta);
          } else if (e.type === "final") {
            finalPayload = e.payload;
          } else if (e.type === "error") {
            streamErr = e.message;
          }
        },
        controller.signal,
      );
    } catch (e) {
      streamErr = e instanceof Error ? e.message : String(e);
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }

    if (streamErr) {
      setErrorMsg(streamErr);
      return;
    }
    if (!finalPayload) {
      setErrorMsg("Stream ended without final response");
      return;
    }
    const resp = finalPayload as import("@/lib/api").AgentChatResponse;
    const aTurn: Turn = {
      kind: "assistant",
      ts: fmtTimestamp(),
      text: resp.answer,
      action: resp.action,
      citations: resp.citations,
      metadata: resp.metadata,
    };
    setTurns((prev) => [...prev, aTurn]);
    setStreamProgress([]);
    setStreamingText("");
    qc.invalidateQueries({ queryKey: ["conversations", "list"] });
  }

  function startNew() {
    setThreadId(null);
    setTurns([]);
    setErrorMsg(null);
  }

  async function openConversation(c: Conversation) {
    setThreadId(c.thread_id);
    setCollectionId(c.collection_id ?? null);
    setErrorMsg(null);
    try {
      const msgs = await api.getConversationMessages(c.thread_id);
      const loaded: Turn[] = msgs
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) =>
          m.role === "user"
            ? ({
                kind: "user",
                ts: m.timestamp ?? fmtTimestamp(c.last_active_at),
                text: m.content,
              } as Turn)
            : ({
                kind: "assistant",
                ts: m.timestamp ?? fmtTimestamp(c.last_active_at),
                text: m.content,
                action: null,
                citations: (m.citations as AgentCitation[]) ?? [],
                metadata: m.metadata ?? {
                  model: null,
                  tokens_input: 0,
                  tokens_output: 0,
                  latency_ms: 0,
                  chunks_retrieved: 0,
                },
              } as Turn),
        );
      setTurns(loaded);
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    }
  }

  function toggleSaved() {
    if (!threadId || !activeConv) return;
    patchSaveMut.mutate({ tid: threadId, saved: !activeConv.is_saved });
  }

  async function deleteCurrent() {
    if (!threadId) return;
    setConfirmDelete(true);
  }

  async function confirmDeleteCurrent() {
    if (!threadId) return;
    setConfirmDelete(false);
    try {
      await deleteConvMut.mutateAsync(threadId);
      startNew();
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    }
  }

  function pinTurn(t: Extract<Turn, { kind: "assistant" }>) {
    if (!threadId) return;
    pinNoteMut.mutate(
      { tid: threadId, content: t.text },
      {
        onSuccess: () => {
          setErrorMsg(null);
        },
      },
    );
  }

  return (
    <>
    <Modal
      open={confirmDelete}
      onClose={() => setConfirmDelete(false)}
      title={`DELETE THREAD ${threadId?.slice(0, 8) ?? ""}?`}
      width="max-w-sm"
    >
      <p className="body-sm text-text-secondary mb-6">
        This action cannot be undone. Thread and all messages will be permanently removed.
      </p>
      <div className="flex items-center gap-3 justify-end">
        <button
          onClick={() => setConfirmDelete(false)}
          className="label-sm text-text-secondary hover:bg-surface-container px-4 h-8 border border-border-subtle"
        >
          CANCEL
        </button>
        <button
          onClick={confirmDeleteCurrent}
          disabled={deleteConvMut.isPending}
          className="label-sm text-error hover:bg-error hover:text-surface px-4 h-8 border border-error/40 disabled:opacity-50"
        >
          {deleteConvMut.isPending ? "DELETING..." : "✕ DELETE"}
        </button>
      </div>
    </Modal>
    <div className="-m-3 sm:-m-6 md:-m-8 flex h-[calc(100vh-48px)] md:h-[calc(100vh-48px-28px)]">
      {/* Sidebar — desktop only (mobile uses Sheet trigger in header) */}
      <aside className="hidden md:flex w-64 shrink-0 border-r border-border-subtle bg-surface-low flex-col" aria-label="Threads and notes">
        {/* Tab header */}
        <div className="flex border-b border-border-subtle shrink-0">
          <button
            onClick={() => setSidebarTab("threads")}
            className={`flex-1 h-12 label-sm transition-colors ${
              sidebarTab === "threads"
                ? "text-text-primary border-b-2 border-text-primary -mb-px"
                : "text-text-tertiary hover:text-text-secondary"
            }`}
          >
            THREADS
          </button>
          <button
            onClick={() => setSidebarTab("notes")}
            className={`flex-1 h-12 label-sm transition-colors ${
              sidebarTab === "notes"
                ? "text-text-primary border-b-2 border-text-primary -mb-px"
                : "text-text-tertiary hover:text-text-secondary"
            }`}
          >
            NOTES {(notes.data?.length ?? 0) > 0 && `[${notes.data!.length}]`}
          </button>
          {sidebarTab === "threads" && (
            <button
              onClick={startNew}
              className="px-3 h-12 label-sm text-text-primary hover:bg-text-primary hover:text-surface border-l border-border-subtle"
            >
              + NEW
            </button>
          )}
        </div>

        <div className="flex-1 overflow-auto">
          {/* Threads tab */}
          {sidebarTab === "threads" && (
            <>
              {conversations.isLoading && (
                <div className="px-4 py-3 space-y-2">
                  {[0, 1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-12 w-full bg-surface-container" />
                  ))}
                </div>
              )}
              {!conversations.isLoading && (conversations.data?.length ?? 0) === 0 && (
                <div className="px-4 py-3 label-sm text-text-tertiary">(no threads yet)</div>
              )}
              {conversations.data?.map((c) => {
                const isActive = c.thread_id === threadId;
                const colName = c.collection_id
                  ? (collections.data?.find((col) => col.id === c.collection_id)?.name ?? `COL ${c.collection_id}`)
                  : "ALL COLLECTIONS";
                const channelTag = c.channel === "whatsapp" ? "WA" : "WEB";
                return (
                  <button
                    key={c.thread_id}
                    onClick={() => openConversation(c)}
                    className={`relative w-full text-left px-4 py-3 border-b border-border-subtle/50 hover:bg-surface-container transition-colors ${
                      isActive ? "bg-surface-container" : ""
                    }`}
                  >
                    {isActive && (
                      <span className="absolute left-0 top-0 bottom-0 w-[2px] bg-text-primary" />
                    )}
                    <div className="flex items-center justify-between gap-2">
                      <div className="label-sm text-text-primary truncate">{colName.toUpperCase()}</div>
                      {c.is_saved && <span className="label-xs text-success shrink-0">★</span>}
                    </div>
                    <div className="label-xs text-text-secondary mt-1">
                      {channelTag} · {c.thread_id.slice(0, 8)}
                    </div>
                    <div className="label-xs text-text-tertiary mt-0.5 tabular-nums">
                      {fmtDate(c.last_active_at)}
                    </div>
                  </button>
                );
              })}
            </>
          )}

          {/* Notes tab */}
          {sidebarTab === "notes" && (
            <>
              {notes.isLoading && (
                <div className="px-4 py-3 label-sm text-text-tertiary">LOADING...</div>
              )}
              {!notes.isLoading && (notes.data?.length ?? 0) === 0 && (
                <div className="px-4 py-3 label-sm text-text-tertiary">(no pinned notes)</div>
              )}
              {notes.data?.map((n) => (
                <div
                  key={n.id}
                  className="px-4 py-3 border-b border-border-subtle/50 group"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="label-xs text-text-tertiary tabular-nums">
                      {n.conversation_id ? `THREAD · ` : ""}
                      {new Date(n.created_at).toLocaleString("en-GB")}
                    </span>
                    <button
                      onClick={() => deleteNoteMut.mutate(n.id)}
                      disabled={deleteNoteMut.isPending}
                      className="label-xs text-text-tertiary hover:text-error opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      ✕
                    </button>
                  </div>
                  <div className="body-sm text-text-secondary line-clamp-3">
                    {n.source_message_content}
                  </div>
                  {n.user_note && (
                    <div className="label-xs text-text-tertiary mt-1 italic">
                      {n.user_note}
                    </div>
                  )}
                </div>
              ))}
            </>
          )}
        </div>
      </aside>

      {/* Main */}
      <section className="flex-1 min-w-0 flex flex-col bg-surface">
        {/* Header */}
        <div className="min-h-12 px-3 sm:px-6 py-2 flex flex-wrap items-center justify-between gap-2 border-b border-border-subtle shrink-0">
          <div className="flex items-center gap-2">
            <Sheet>
              <SheetTrigger asChild>
                <button
                  type="button"
                  aria-label="Open threads panel"
                  className="md:hidden inline-flex items-center justify-center w-8 h-8 border border-border-subtle text-text-primary hover:bg-surface-container focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-blue"
                >
                  <span aria-hidden="true">☰</span>
                </button>
              </SheetTrigger>
              <SheetContent side="left" className="w-72 max-w-[85vw] bg-surface-low border-r border-border-subtle p-0 overflow-auto">
                <SheetHeader className="px-4 py-3 border-b border-border-subtle">
                  <SheetTitle className="label-sm text-text-primary">THREADS / NOTES</SheetTitle>
                </SheetHeader>
                <div className="p-2 label-xs text-text-tertiary">Tap a thread to open it.</div>
              </SheetContent>
            </Sheet>
            <div className="label-sm text-text-secondary">
              AGENT_THREAD:{" "}
              <span className="text-text-primary">
                {threadId ? threadId.slice(0, 8) : "NEW"}
              </span>
            </div>
          </div>
          <div className="flex items-center flex-wrap gap-2 sm:gap-4">
            <label className="flex items-center gap-2 label-sm text-text-secondary">
              COLLECTION:
              <select
                value={collectionId ?? ""}
                onChange={(e) =>
                  setCollectionId(e.target.value ? Number(e.target.value) : null)
                }
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
            {lastAssistantTurn && (
              <Sheet>
                <SheetTrigger asChild>
                  <button className="label-sm text-text-primary hover:bg-text-primary hover:text-surface px-2 h-7 border border-border-subtle">
                    ☰ SOURCES [{lastAssistantTurn.citations.length}]
                  </button>
                </SheetTrigger>
                <SheetContent
                  side="right"
                  className="w-full sm:w-[480px] sm:max-w-[480px] bg-surface-low border-l border-border-subtle overflow-auto"
                >
                  <SheetHeader>
                    <SheetTitle className="label-sm text-text-primary">
                      RETRIEVED CHUNKS · LAST TURN
                    </SheetTitle>
                  </SheetHeader>
                  <div className="mt-4 space-y-3">
                    {[...lastAssistantTurn.citations]
                      .sort((a, b) => b.relevance - a.relevance)
                      .map((c, idx) => {
                        const rel = c.relevance;
                        const relClass =
                          rel >= 0.7 ? "text-success" : rel >= 0.5 ? "text-text-tertiary" : "text-error";
                        return (
                          <div
                            key={`src-${c.chunk_id}-${idx}`}
                            className="border border-border-subtle bg-surface-container-low p-3"
                          >
                            <div className="label-sm flex items-center justify-between mb-2">
                              <span className="text-text-secondary truncate">
                                {c.document_title}
                              </span>
                              <span className={`tabular-nums shrink-0 ${relClass}`}>
                                {rel.toFixed(3)}
                              </span>
                            </div>
                            <div className="label-xs text-text-tertiary mb-2">
                              CHUNK {c.chunk_index}
                            </div>
                            {c.text_preview && (
                              <div className="body-sm text-text-secondary whitespace-pre-wrap">
                                {c.text_preview}
                              </div>
                            )}
                          </div>
                        );
                      })}
                  </div>
                </SheetContent>
              </Sheet>
            )}
            {threadId && (
              <>
                <button
                  onClick={toggleSaved}
                  disabled={patchSaveMut.isPending}
                  className="label-sm text-text-primary hover:bg-text-primary hover:text-surface px-2 h-7 border border-border-subtle"
                >
                  {activeConv?.is_saved ? "★ SAVED" : "☆ SAVE"}
                </button>
                <button
                  onClick={deleteCurrent}
                  disabled={deleteConvMut.isPending}
                  className="label-sm text-error hover:bg-error hover:text-surface px-2 h-7 border border-error/40"
                >
                  ✕ DELETE
                </button>
              </>
            )}
          </div>
        </div>

        {/* Conversation area */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-auto px-3 sm:px-6 md:px-8 py-4 sm:py-6"
          aria-live="polite"
          aria-atomic="false"
        >
          {errorMsg && (
            <div className="mb-4">
              <ErrorBanner message={errorMsg} />
              <button
                onClick={() => {
                  const lastUser = [...turns].reverse().find((t) => t.kind === "user");
                  if (lastUser && lastUser.kind === "user") {
                    setTurns((prev) => prev.filter((t) => t !== lastUser));
                    setErrorMsg(null);
                    void send(lastUser.text);
                  }
                }}
                className="mt-2 label-sm text-text-primary hover:bg-text-primary hover:text-surface px-3 h-7 border border-border-subtle"
              >
                ↻ RETRY
              </button>
            </div>
          )}

          {turns.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto">
              <div className="title-lg text-text-primary blink-cursor">
                AGENT_SESSION READY
              </div>
              <div className="body-sm text-text-secondary mt-3">
                LangGraph · classifier → RAG / direct / clarify · persistent memory.
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
              {turns.map((t, i) =>
                t.kind === "user" ? (
                  <UserTurn key={i} turn={t} />
                ) : (
                  <AssistantTurn
                    key={i}
                    turn={t}
                    canPin={!!threadId}
                    onToggle={() =>
                      setTurns((prev) =>
                        prev.map((tt, j) =>
                          j === i && tt.kind === "assistant"
                            ? { ...tt, expanded: !tt.expanded }
                            : tt,
                        ),
                      )
                    }
                    onPin={() => pinTurn(t)}
                    pinning={pinNoteMut.isPending}
                  />
                ),
              )}

              {streaming && (
                <div className="border-l-2 border-text-primary pl-4 py-2 space-y-2">
                  {streamProgress.length === 0 && !streamingText && (
                    <div className="label-sm text-text-tertiary">CONNECTING...</div>
                  )}
                  {streamProgress.map((p, i) => {
                    const label = NODE_LABELS[p.node] ?? p.node.toUpperCase();
                    const detail =
                      p.summary?.action
                        ? ` → ${p.summary.action}`
                        : p.summary?.chunks !== undefined
                          ? ` · ${p.summary.chunks} chunks`
                          : "";
                    return (
                      <div
                        key={`${p.node}-${i}`}
                        className={`label-sm ${
                          p.status === "running" ? "text-text-primary" : "text-text-tertiary"
                        }`}
                      >
                        {label}{detail} {p.status === "running" ? "▌" : "✓"}
                      </div>
                    );
                  })}
                  {streamingText && (
                    <div className="prose prose-invert max-w-none text-text-primary">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
                      >
                        {streamingText}
                      </ReactMarkdown>
                      <span className="inline-block w-2 h-4 bg-text-primary animate-pulse align-middle" />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-border-subtle px-3 sm:px-6 md:px-8 py-3 sm:py-4 shrink-0">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (!streaming) send(input);
            }}
            className="max-w-4xl mx-auto flex items-center gap-3 border-b border-border-subtle focus-within:border-border-focus pb-2"
          >
            <span className="text-text-primary body-sm">{">"}</span>
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={streaming}
              placeholder="ASK THE AGENT..."
              aria-label="Message to agent"
              className="flex-1 min-w-0 bg-transparent border-0 outline-none text-text-primary placeholder:text-text-tertiary placeholder:label-sm body-sm py-2 focus-visible:ring-0"
            />
            <button
              type="submit"
              disabled={streaming || !input.trim()}
              className="label-sm text-text-primary disabled:text-text-tertiary hover:bg-text-primary hover:text-surface px-2 h-7 disabled:hover:bg-transparent disabled:hover:text-text-tertiary"
            >
              ↵ SEND
            </button>
          </form>
        </div>
      </section>
    </div>
    </>
  );
}

const CITATION_RE = /\[(\d+)\]/g;

function interleaveCitations(
  children: React.ReactNode,
  citations: AgentCitation[],
): React.ReactNode {
  const out: React.ReactNode[] = [];

  function processNode(node: React.ReactNode, keyPrefix: string) {
    if (typeof node === "string") {
      let lastIdx = 0;
      let match: RegExpExecArray | null;
      const re = new RegExp(CITATION_RE.source, "g");
      while ((match = re.exec(node)) !== null) {
        const n = Number(match[1]);
        if (n < 1 || n > citations.length) continue;
        if (match.index > lastIdx) {
          out.push(node.slice(lastIdx, match.index));
        }
        const c = citations[n - 1];
        out.push(
          <HoverCard key={`${keyPrefix}-cite-${n}-${out.length}`} openDelay={120}>
            <HoverCardTrigger asChild>
              <sup className="text-primary-blue cursor-help font-medium px-0.5">
                [{n}]
              </sup>
            </HoverCardTrigger>
            <HoverCardContent
              side="top"
              align="start"
              className="w-[460px] max-h-[300px] overflow-auto bg-surface-container border border-border-subtle"
            >
              <div className="label-xs text-text-tertiary mb-2">
                {c.document_title} · CHUNK {c.chunk_index} · REL {c.relevance.toFixed(3)}
              </div>
              <div className="body-sm text-text-secondary whitespace-pre-wrap">
                {c.text_preview ?? "(no preview)"}
              </div>
            </HoverCardContent>
          </HoverCard>,
        );
        lastIdx = match.index + match[0].length;
      }
      if (lastIdx < node.length) out.push(node.slice(lastIdx));
    } else if (Array.isArray(node)) {
      node.forEach((child, i) => processNode(child, `${keyPrefix}-${i}`));
    } else {
      out.push(node);
    }
  }

  processNode(children, "root");
  return out.length === 0 ? children : out;
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
  canPin,
  pinning,
  onToggle,
  onPin,
}: {
  turn: Extract<Turn, { kind: "assistant" }>;
  canPin: boolean;
  pinning: boolean;
  onToggle: () => void;
  onPin: () => void;
}) {
  const actionClass = turn.action
    ? (ACTION_CLASS[turn.action] ?? "text-text-tertiary border-text-tertiary")
    : "text-text-tertiary border-text-tertiary";
  return (
    <div>
      <div className="label-sm text-text-tertiary mb-2 flex items-center gap-2">
        <span>
          {"<"} AGENT · <span className="text-text-secondary">{turn.ts}</span>
        </span>
        {turn.action && (
          <span className={`label-xs px-2 py-0.5 border ${actionClass}`}>
            {turn.action.toUpperCase()}
          </span>
        )}
        <span className="flex-1" />
        {canPin && (
          <button
            onClick={onPin}
            disabled={pinning}
            className="label-xs text-text-tertiary hover:text-primary-blue"
          >
            {pinning ? "PINNING..." : "★ PIN NOTE"}
          </button>
        )}
      </div>
      <div className="body text-text-primary leading-relaxed markdown-body">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
          components={{
            p: ({ children }) => <p>{interleaveCitations(children, turn.citations)}</p>,
            li: ({ children }) => <li>{interleaveCitations(children, turn.citations)}</li>,
          }}
        >
          {turn.text}
        </ReactMarkdown>
      </div>

      {turn.citations.length > 0 && (
        <div className="mt-4">
          <button
            onClick={onToggle}
            className="label-sm text-text-secondary hover:text-primary-blue"
          >
            {turn.expanded ? "▾" : "▸"} CITATIONS [{turn.citations.length}]
          </button>
          {turn.expanded && (
            <div className="mt-3 space-y-2">
              {turn.citations.map((c, idx) => {
                const rel = c.relevance;
                const relClass = rel >= 0.7 ? "text-success" : rel >= 0.5 ? "text-text-tertiary" : "text-error";
                const num = String(idx + 1).padStart(2, "0");
                return (
                  <HoverCard key={`${c.chunk_id}-${idx}`} openDelay={150}>
                    <HoverCardTrigger asChild>
                      <div
                        id={`cite-${turn.ts}-${idx + 1}`}
                        className="border border-border-subtle bg-surface-container-low p-3 cursor-help"
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
                    </HoverCardTrigger>
                    {c.text_preview && (
                      <HoverCardContent
                        side="top"
                        align="start"
                        className="w-[480px] max-h-[320px] overflow-auto bg-surface-container border border-border-subtle"
                      >
                        <div className="label-xs text-text-tertiary mb-2">
                          {c.document_title} · CHUNK {c.chunk_index}
                        </div>
                        <div className="body-sm text-text-secondary whitespace-pre-wrap">
                          {c.text_preview}
                        </div>
                      </HoverCardContent>
                    )}
                  </HoverCard>
                );
              })}
            </div>
          )}
        </div>
      )}

      <div className="label-xs text-text-tertiary mt-4 tabular-nums">
        {turn.metadata.latency_ms}MS · {turn.metadata.tokens_input.toLocaleString()} IN /{" "}
        {turn.metadata.tokens_output.toLocaleString()} OUT
        {turn.metadata.model ? ` · ${turn.metadata.model}` : ""} ·{" "}
        {estimateCost(turn.metadata.tokens_input, turn.metadata.tokens_output, turn.metadata.model)}
      </div>
      <div className="border-b border-border-subtle mt-4" />
    </div>
  );
}

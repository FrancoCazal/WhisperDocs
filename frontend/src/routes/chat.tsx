import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  fmtTimestamp,
  type AgentChatResponse,
  type AgentCitation,
  type Conversation,
} from "@/lib/api";
import { ErrorBanner } from "@/components/ui-kit/EmptyState";
import { Modal } from "@/components/ui-kit/Modal";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { AssistantTurn, UserTurn } from "@/components/chat/Turns";
import { SourcesSheet } from "@/components/chat/SourcesSheet";
import { StreamingTurn } from "@/components/chat/StreamingTurn";
import { ThreadSidebar } from "@/components/chat/ThreadSidebar";
import type { AssistantTurnData, NodeProgress, Turn } from "@/components/chat/types";

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

const EXAMPLES = [
  "What distance metrics does pgvector support?",
  "How does selectinload work in SQLAlchemy async?",
  "Hello, what can you do?",
];

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
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const conversations = useQuery({
    queryKey: ["conversations", "list"],
    queryFn: () => api.listConversations({ limit: 50 }),
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
    let finalPayload: AgentChatResponse | null = null;
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
    const resp = finalPayload as AgentChatResponse;
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
    // The query log: history pages and the layout's last-answer bar.
    qc.invalidateQueries({ queryKey: ["history"] });
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
                  cost_usd: 0,
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

  function pinTurn(t: AssistantTurnData) {
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
        <aside
          className="hidden md:flex w-64 shrink-0 border-r border-border-subtle bg-surface-low flex-col"
          aria-label="Threads and notes"
        >
          <ThreadSidebar
            conversations={conversations.data}
            loading={conversations.isLoading}
            collections={collections.data}
            activeThreadId={threadId}
            onOpen={openConversation}
            onNew={startNew}
          />
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
                <SheetContent
                  side="left"
                  className="w-72 max-w-[85vw] bg-surface-low border-r border-border-subtle p-0 overflow-auto"
                >
                  <SheetHeader className="px-4 py-3 border-b border-border-subtle">
                    <SheetTitle className="label-sm text-text-primary">THREADS / NOTES</SheetTitle>
                  </SheetHeader>
                  <div className="p-2 label-xs text-text-tertiary">Tap a thread to open it.</div>
                </SheetContent>
              </Sheet>
              <div className="label-sm text-text-secondary">
                AGENT_THREAD:{" "}
                <span className="text-text-primary">{threadId ? threadId.slice(0, 8) : "NEW"}</span>
              </div>
            </div>
            <div className="flex items-center flex-wrap gap-2 sm:gap-4">
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
              {lastAssistantTurn && <SourcesSheet citations={lastAssistantTurn.citations} />}
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
                <div className="title-lg text-text-primary blink-cursor">AGENT_SESSION READY</div>
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

                {streaming && <StreamingTurn progress={streamProgress} text={streamingText} />}
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

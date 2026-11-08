import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import {
  api,
  fmtCost,
  fmtNumber,
  fmtTimestamp,
  fmtTurnTime,
  LIMITS,
  type AgentChatResponse,
  type Conversation,
  type ConversationMessage,
} from "@/lib/api";
import { LoadingLine, emptyActionCls } from "@/components/ui-kit/EmptyState";
import { Composer } from "@/components/ui-kit/Composer";
import { ConfirmDialog } from "@/components/ui-kit/ConfirmDialog";
import { toast } from "sonner";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { AssistantTurn, UserTurn } from "@/components/chat/Turns";
import { SourcesSheet, type SourcesView } from "@/components/chat/SourcesSheet";
import { citedSources } from "@/components/chat/citations";
import { StreamingTurn, type LiveTurn } from "@/components/chat/StreamingTurn";
import { ThreadSidebar, type ThreadFilters } from "@/components/chat/ThreadSidebar";
import { PAGE_SIZE, usePagedList } from "@/lib/paged";
import { useDebounced } from "@/lib/useDebounced";
import { useTopK } from "@/lib/topK";
import { ScopeControls } from "@/components/ui-kit/ScopeControls";
import { Btn, btnCls } from "@/components/ui-kit/Button";
import {
  IDLE_TIMEOUT_MS,
  type AssistantTurnData,
  type NodeProgress,
  type Turn,
  type TurnOutcome,
} from "@/components/chat/types";

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
  // ?thread=<id>: the thread on screen, so a reload or a link reopens it.
  // ?collection=<id>: a new chat in that collection (a document's ASK IN CHAT).
  validateSearch: (search: Record<string, unknown>): { thread?: string; collection?: number } => {
    const out: { thread?: string; collection?: number } = {};
    if (typeof search.thread === "string" && search.thread) out.thread = search.thread;
    const c = Number(search.collection);
    if (search.collection !== undefined && Number.isInteger(c) && c > 0) out.collection = c;
    return out;
  },
  component: ChatPage,
});

// Questions the seed collections answer; offered only once there are documents.
const EXAMPLES = [
  "What distance metrics does pgvector support?",
  "How does selectinload work in SQLAlchemy async?",
  "Hello, what can you do?",
];

// The stream being read, and why it was aborted when it was.
type StreamRun = { controller: AbortController; reason: "stopped" | "timeout" | null };

function savedTurn(m: ConversationMessage): Turn {
  // Messages saved before they were stamped have no time of their own.
  const ts = m.timestamp ? fmtTurnTime(m.timestamp) : "—";
  if (m.role === "user") return { kind: "user", ts, text: m.content };
  return {
    kind: "assistant",
    ts,
    text: m.content,
    action: m.action,
    citations: m.failed ? [] : (m.citations ?? []),
    metadata: m.failed ? null : m.metadata,
    outcome: m.failed ? "failed" : undefined,
  };
}

// An answer pinned from this tab: its button says PINNED, so a second click
// does not save the same note twice.
const pinKey = (threadId: string, text: string) => `${threadId}\n${text}`;

// What the status region says when a turn ends without an answer.
const OUTCOME_STATUS: Record<TurnOutcome, string> = {
  failed: "The answer failed.",
  error: "The answer failed.",
  stopped: "Answer stopped.",
  timeout: "The answer timed out.",
};

function ChatPage() {
  const qc = useQueryClient();
  const [pinned, setPinned] = useState<ReadonlySet<string>>(() => new Set());
  const navigate = useNavigate({ from: "/chat" });
  const { thread: urlThread = null, collection: urlCollection } = Route.useSearch();

  const [threadId, setThreadId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [collectionId, setCollectionId] = useState<number | null>(null);
  const { topK, defaultTopK, setTopK } = useTopK();
  const [input, setInput] = useState("");
  const [live, setLive] = useState<LiveTurn | null>(null);
  // What a screen reader hears: a turn starting and how it ended, not every
  // token the stream writes into the log.
  const [status, setStatus] = useState("");
  const [loadingThread, setLoadingThread] = useState(false);
  // A WhatsApp thread is read-only here: its id is not a UUID4, which the
  // chat endpoint requires, and a web reply would mix into that user's memory.
  const [channel, setChannel] = useState("web");
  const readOnly = channel !== "web";
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const runRef = useRef<StreamRun | null>(null);
  // Bumped whenever another thread is shown: a stream or a load started for
  // the previous one sees the change and leaves the screen alone.
  const viewRef = useRef(0);
  // The thread on screen, set at once (state lags a render): the URL effect
  // compares against it, so a stream's new id is not taken for navigation.
  const shownRef = useRef<string | null>(null);
  // Follow new output only while the reader is at the bottom.
  const stickRef = useRef(true);
  const streaming = live !== null;

  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  // The thread list's search (asked once the typing pauses) and SAVED filter.
  const [threadQ, setThreadQ] = useState("");
  const [savedOnly, setSavedOnly] = useState(false);
  const threadSearch = useDebounced(threadQ.trim(), 300);
  const threadFilters: ThreadFilters = {
    q: threadQ,
    onQ: setThreadQ,
    saved: savedOnly,
    onSaved: setSavedOnly,
  };
  const threadList = usePagedList(
    ["conversations", "list", "pages", { q: threadSearch, saved: savedOnly }],
    (cursor) =>
      api.listConversationsPage({
        limit: PAGE_SIZE,
        cursor,
        q: threadSearch || undefined,
        is_saved: savedOnly ? true : undefined,
      }),
  );
  const conversations = { data: threadList.items, isLoading: threadList.query.isLoading };
  const threadsMore = {
    total: threadList.total,
    hasMore: threadList.query.hasNextPage,
    loading: threadList.query.isFetchingNextPage,
    onLoad: () => void threadList.query.fetchNextPage(),
  };
  // The examples need documents; the collections carry their counts.
  const noDocuments =
    collections.isSuccess && collections.data.every((c) => c.document_count === 0);

  const listedConv = useMemo(
    () => conversations.data?.find((c) => c.thread_id === threadId),
    [conversations.data, threadId],
  );
  // The open thread when the filters or the paging leave it out of the list:
  // its SAVE button still needs it. Under "list", so a refresh of the list
  // refreshes it too.
  const shownConv = useQuery({
    queryKey: ["conversations", "list", "one", threadId],
    queryFn: () => api.getConversation(threadId!),
    enabled: threadId !== null && threadList.query.isSuccess && !listedConv,
    retry: false,
  });
  const activeConv: Conversation | undefined =
    listedConv ?? (shownConv.data?.thread_id === threadId ? shownConv.data : undefined);

  // The header's sources are the latest answer's: an older turn's chunks
  // shown as the last turn's would be wrong.
  const latestAnswer = useMemo(() => {
    const last = turns[turns.length - 1];
    return last?.kind === "assistant" && !last.outcome && last.citations.length > 0 ? last : null;
  }, [turns]);

  // The thread's answers so far, and their model calls' tokens and cost.
  const threadUsage = useMemo(() => {
    let answers = 0;
    let tokens = 0;
    let cost = 0;
    for (const t of turns) {
      if (t.kind !== "assistant" || t.outcome || !t.metadata) continue;
      answers += 1;
      tokens += t.metadata.tokens_input + t.metadata.tokens_output;
      cost += t.metadata.cost_usd;
    }
    return answers > 0 ? { answers, tokens, cost } : null;
  }, [turns]);

  // The mobile threads panel: controlled, so picking a thread closes it.
  const [threadsOpen, setThreadsOpen] = useState(false);

  // One sheet for every answer's sources, opened by a turn or an "[n]".
  const [sources, setSources] = useState<SourcesView | null>(null);
  const openSources = useCallback(
    (t: AssistantTurnData, focus: number | null) =>
      setSources({
        citations: t.citations,
        focus,
        label: `ANSWER ${t.ts}`,
        cited: citedSources(t.text, t.citations.length),
      }),
    [],
  );

  const patchSaveMut = useMutation({
    mutationFn: ({ tid, saved }: { tid: string; saved: boolean }) =>
      api.patchConversation(tid, { is_saved: saved }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations", "list"] }),
    onError: (e, v) =>
      toast.error(v.saved ? "COULD NOT SAVE THE THREAD" : "COULD NOT UNSAVE THE THREAD", {
        description: e.message,
      }),
  });

  const deleteConvMut = useMutation({
    mutationFn: (tid: string) => api.deleteConversation(tid),
    onSuccess: (_r, tid) => {
      toast.success(`THREAD ${tid.slice(0, 8)} DELETED`);
      return qc.invalidateQueries({ queryKey: ["conversations", "list"] });
    },
    onError: (e, tid) =>
      toast.error(`COULD NOT DELETE THREAD ${tid.slice(0, 8)}`, { description: e.message }),
  });

  const pinNoteMut = useMutation({
    mutationFn: ({ tid, content }: { tid: string; content: string }) =>
      api.createNote(tid, { content }),
    onSuccess: (_note, v) => {
      setPinned((s) => new Set(s).add(pinKey(v.tid, v.content)));
      return qc.invalidateQueries({ queryKey: ["notes"] });
    },
    onError: (e) => toast.error("COULD NOT PIN THE ANSWER", { description: e.message }),
  });

  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickRef.current) el.scrollTo({ top: el.scrollHeight });
  }, [turns, live]);

  useEffect(() => {
    if (!streaming) inputRef.current?.focus();
  }, [streaming]);

  useEffect(() => {
    return () => runRef.current?.controller.abort();
  }, []);

  /** Before another thread is shown: what runs for the current one stops. */
  function leaveThread() {
    viewRef.current += 1;
    if (runRef.current) {
      // The server stops the run; the question stays in the thread unanswered.
      runRef.current.controller.abort();
      runRef.current = null;
      qc.invalidateQueries({ queryKey: ["conversations", "list"] });
    }
    setLive(null);
    setStatus("");
    setErrorMsg(null);
    setSources(null);
    setLoadingThread(false);
  }

  async function send(message: string) {
    const trimmed = message.trim();
    if (!trimmed || runRef.current || loadingThread || readOnly) return;
    const view = viewRef.current;
    const run: StreamRun = { controller: new AbortController(), reason: null };
    runRef.current = run;
    // send() runs from event handlers only (submit, an example, RETRY), never
    // during render; the compiler's purity check cannot tell.
    // eslint-disable-next-line react-hooks/purity
    const startedAt = performance.now();
    // What the stream has sent so far; the finished turn is built from it.
    const got: {
      text: string;
      trace: NodeProgress[];
      final: AgentChatResponse | null;
      error: string | null;
    } = { text: "", trace: [], final: null, error: null };
    const show = () => setLive({ startedAt, progress: got.trace, text: got.text });

    const userTurn: Turn = { kind: "user", ts: fmtTimestamp(), text: trimmed };
    setErrorMsg(null);
    setTurns((prev) => [...prev, userTurn]);
    setInput("");
    stickRef.current = true;
    show();
    setStatus("The agent is answering.");

    let idle: ReturnType<typeof setTimeout> | undefined;
    const resetIdle = () => {
      clearTimeout(idle);
      idle = setTimeout(() => {
        run.reason = "timeout";
        run.controller.abort();
      }, IDLE_TIMEOUT_MS);
    };
    resetIdle();

    try {
      await api.agentChatStream(
        { message: trimmed, thread_id: threadId, collection_id: collectionId, top_k: topK },
        (e) => {
          if (viewRef.current !== view) return;
          resetIdle();
          if (e.type === "thread") {
            if (shownRef.current !== e.thread_id) showThread(e.thread_id, "replace");
          } else if (e.type === "node_start") {
            got.trace = [
              ...got.trace,
              { node: e.node, status: "running", startedAt: performance.now() },
            ];
            show();
          } else if (e.type === "node_end") {
            const now = performance.now();
            got.trace = got.trace.map((p) =>
              p.node === e.node && p.status === "running"
                ? { ...p, status: "done", summary: e.summary, ms: now - p.startedAt }
                : p,
            );
            show();
          } else if (e.type === "token") {
            got.text += e.delta;
            show();
          } else if (e.type === "final") {
            got.final = e.payload;
          } else if (e.type === "error") {
            got.error = e.message;
          }
        },
        run.controller.signal,
      );
    } catch (e) {
      // An abort is a stop, a timeout or a thread change, not an error.
      if (!run.controller.signal.aborted) got.error = e instanceof Error ? e.message : String(e);
    } finally {
      clearTimeout(idle);
    }
    // Another thread is on screen: this turn is not its.
    if (viewRef.current !== view) return;
    runRef.current = null;
    setLive(null);

    const base = {
      kind: "assistant" as const,
      ts: fmtTimestamp(),
      trace: got.trace.map((p) => (p.status === "running" ? { ...p, status: "cut" as const } : p)),
      // eslint-disable-next-line react-hooks/purity -- an event handler, as above
      elapsedMs: performance.now() - startedAt,
    };
    const f = got.final;
    let turn: AssistantTurnData;
    if (f) {
      turn = {
        ...base,
        text: f.answer,
        action: f.action,
        citations: f.failed ? [] : f.citations,
        metadata: f.failed ? null : f.metadata,
        outcome: f.failed ? "failed" : undefined,
      };
    } else if (run.reason) {
      turn = {
        ...base,
        text: got.text,
        action: null,
        citations: [],
        metadata: null,
        outcome: run.reason,
      };
    } else {
      turn = {
        ...base,
        text: got.error ?? "The stream ended without an answer.",
        action: null,
        citations: [],
        metadata: null,
        outcome: "error",
      };
    }
    setTurns((prev) => [...prev, turn]);
    setStatus(turn.outcome ? OUTCOME_STATUS[turn.outcome] : "Answer ready.");
    qc.invalidateQueries({ queryKey: ["conversations", "list"] });
    // The query log: history pages and the layout's last-answer bar.
    if (f && !f.failed) qc.invalidateQueries({ queryKey: ["history"] });
  }

  function stop() {
    const run = runRef.current;
    if (!run) return;
    run.reason = "stopped";
    run.controller.abort();
  }

  /** Ask again the question that assistant turn `i` answered. */
  function retry(i: number) {
    for (let j = i - 1; j >= 0; j--) {
      const t = turns[j];
      if (t.kind === "user") {
        void send(t.text);
        return;
      }
    }
  }

  /** Show thread `id`; `url` says how ?thread= follows (keep: it already does). */
  function showThread(id: string | null, url: "push" | "replace" | "keep") {
    shownRef.current = id;
    setThreadId(id);
    if (url !== "keep") {
      void navigate({ search: id ? { thread: id } : {}, replace: url === "replace" });
    }
  }

  function startNew(url: "push" | "keep" = "push") {
    leaveThread();
    showThread(null, url);
    setChannel("web");
    setTurns([]);
  }

  async function openConversation(c: Conversation, url: "push" | "keep" = "push") {
    // The thread on screen; a failed load can be retried by opening it again.
    if (c.thread_id === threadId && (turns.length > 0 || loadingThread)) return;
    leaveThread();
    const view = viewRef.current;
    showThread(c.thread_id, url);
    setChannel(c.channel);
    setCollectionId(c.collection_id ?? null);
    setTurns([]);
    setLoadingThread(true);
    try {
      const msgs = await api.getConversationMessages(c.thread_id);
      if (viewRef.current !== view) return;
      setTurns(msgs.filter((m) => m.role === "user" || m.role === "assistant").map(savedTurn));
    } catch (e) {
      if (viewRef.current !== view) return;
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      if (viewRef.current === view) setLoadingThread(false);
    }
  }

  /** A thread named by the URL: from the list if it is there, else by id. */
  async function openFromUrl(id: string) {
    const listed = conversations.data?.find((c) => c.thread_id === id);
    if (listed) return openConversation(listed, "keep");
    leaveThread();
    const view = viewRef.current;
    shownRef.current = id;
    setLoadingThread(true);
    try {
      const c = await api.getConversation(id);
      if (viewRef.current === view) await openConversation(c, "keep");
    } catch (e) {
      if (viewRef.current !== view) return;
      startNew("keep");
      void navigate({ search: {}, replace: true });
      setErrorMsg(`thread ${id.slice(0, 8)}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  // The URL changed by itself (a reload, back/forward, a link): show its thread.
  const followUrl = useEffectEvent((id: string | null) => {
    if (id === shownRef.current) return;
    if (id) void openFromUrl(id);
    else startNew("keep");
  });
  // Following the URL means aborting a stream and loading a thread, which cannot
  // happen during render, so this is an effect that sets state.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => followUrl(urlThread), [urlThread]);

  // ?collection=N starts a new chat scoped to it, then leaves the URL: the
  // thread it becomes takes the URL over, and a reload keeps that thread.
  const takeCollection = useEffectEvent((id: number) => {
    if (urlThread) return;
    startNew("keep");
    setCollectionId(id);
    void navigate({ search: {}, replace: true });
  });
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (urlCollection !== undefined) takeCollection(urlCollection);
  }, [urlCollection]);

  function toggleSaved() {
    if (!threadId || !activeConv) return;
    patchSaveMut.mutate({ tid: threadId, saved: !activeConv.is_saved });
  }

  function confirmDeleteCurrent() {
    if (!threadId) return;
    // A failure is a toast, and the thread stays on screen.
    deleteConvMut.mutate(threadId, { onSuccess: () => startNew() });
  }

  function pinTurn(t: AssistantTurnData) {
    if (!threadId) return;
    pinNoteMut.mutate({ tid: threadId, content: t.text });
  }

  const lastIndex = turns.length - 1;

  return (
    <>
      <ConfirmDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title={`DELETE THREAD ${threadId?.slice(0, 8) ?? ""}?`}
        confirmLabel="✕ DELETE"
        destructive
        onConfirm={confirmDeleteCurrent}
      >
        The thread and all its messages are removed for good. This cannot be undone.
      </ConfirmDialog>
      <SourcesSheet view={sources} onClose={() => setSources(null)} />
      <div className="-m-3 sm:-m-6 md:-m-8 flex h-[calc(100vh-48px)] md:h-[calc(100vh-48px-28px)]">
        {/* Sidebar — desktop only (mobile uses Sheet trigger in header) */}
        <aside
          className="hidden md:flex w-64 shrink-0 border-r border-border-subtle bg-surface-low flex-col"
          aria-label="Threads and notes"
        >
          <ThreadSidebar
            conversations={conversations.data}
            loading={conversations.isLoading}
            more={threadsMore}
            collections={collections.data}
            activeThreadId={threadId}
            activeConversationId={activeConv?.id ?? null}
            filters={threadFilters}
            onOpen={openConversation}
            onNew={() => startNew()}
          />
        </aside>

        {/* Main */}
        <section className="flex-1 min-w-0 flex flex-col bg-surface">
          {/* Header */}
          <div className="min-h-12 px-3 sm:px-6 py-2 flex flex-wrap items-center justify-between gap-2 border-b border-border-subtle shrink-0">
            <div className="flex items-center gap-2">
              <Sheet open={threadsOpen} onOpenChange={setThreadsOpen}>
                <SheetTrigger asChild>
                  <button
                    type="button"
                    aria-label="Open threads panel"
                    className={btnCls("quiet", "icon", "md:hidden")}
                  >
                    <span aria-hidden="true">☰</span>
                  </button>
                </SheetTrigger>
                <SheetContent
                  side="left"
                  className="w-72 max-w-[85vw] bg-surface-low border-r border-border-subtle p-0 flex flex-col gap-0"
                >
                  <SheetHeader className="px-4 py-3 border-b border-border-subtle">
                    <SheetTitle className="label-sm text-text-primary">THREADS / NOTES</SheetTitle>
                  </SheetHeader>
                  {/* The desktop sidebar's list; picking a thread closes the panel. */}
                  <ThreadSidebar
                    conversations={conversations.data}
                    loading={conversations.isLoading}
                    more={threadsMore}
                    collections={collections.data}
                    activeThreadId={threadId}
                    activeConversationId={activeConv?.id ?? null}
                    filters={threadFilters}
                    onNavigate={() => setThreadsOpen(false)}
                    onOpen={(c) => {
                      setThreadsOpen(false);
                      void openConversation(c);
                    }}
                    onNew={() => {
                      setThreadsOpen(false);
                      startNew();
                    }}
                  />
                </SheetContent>
              </Sheet>
              <div className="label-sm text-text-secondary">
                AGENT_THREAD:{" "}
                <span className="text-text-primary">{threadId ? threadId.slice(0, 8) : "NEW"}</span>
                {threadUsage && (
                  <span
                    className="hidden sm:inline ml-3 label-xs text-text-tertiary tabular-nums"
                    title="The answers' model calls. The classifier, the query rewrite and the embeddings are not counted."
                  >
                    {threadUsage.answers} {threadUsage.answers === 1 ? "ANSWER" : "ANSWERS"} ·{" "}
                    {fmtNumber(threadUsage.tokens)} TOKENS · {fmtCost(threadUsage.cost)}
                  </span>
                )}
                {readOnly && (
                  <span className="ml-2 label-xs px-2 py-0.5 border border-text-tertiary text-text-secondary">
                    {channel.toUpperCase()} · READ ONLY
                  </span>
                )}
              </div>
            </div>
            <div className="flex items-center flex-wrap gap-2 sm:gap-4">
              <ScopeControls
                collections={collections.data}
                collectionId={collectionId}
                onCollection={setCollectionId}
                topK={topK}
                defaultTopK={defaultTopK}
                onTopK={setTopK}
                disabled={readOnly}
              />
              {latestAnswer && (
                <Btn size="sm" variant="quiet" onClick={() => openSources(latestAnswer, null)}>
                  ☰ SOURCES [{latestAnswer.citations.length}]
                </Btn>
              )}
              {threadId && (
                <>
                  <Btn
                    size="sm"
                    variant="quiet"
                    onClick={toggleSaved}
                    disabled={patchSaveMut.isPending}
                    aria-pressed={activeConv?.is_saved ?? false}
                  >
                    {activeConv?.is_saved ? "★ SAVED" : "☆ SAVE"}
                  </Btn>
                  <Btn
                    size="sm"
                    variant="destructive"
                    onClick={() => setConfirmDelete(true)}
                    disabled={deleteConvMut.isPending || streaming}
                  >
                    ✕ DELETE
                  </Btn>
                </>
              )}
            </div>
          </div>

          {/* Conversation area */}
          <div
            ref={scrollRef}
            onScroll={(e) => {
              const el = e.currentTarget;
              stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
            }}
            className="flex-1 overflow-auto px-3 sm:px-6 md:px-8 py-4 sm:py-6"
          >
            <div role="status" className="sr-only">
              {status}
            </div>
            {turns.length === 0 && loadingThread ? (
              <div className="max-w-4xl mx-auto">
                <LoadingLine label="LOADING THREAD" />
              </div>
            ) : turns.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto">
                <div className="title-lg text-text-primary blink-cursor">AGENT_SESSION READY</div>
                <div className="body-sm text-text-secondary mt-3">
                  LangGraph · classifier → RAG / direct / clarify · persistent memory.
                </div>
                {noDocuments ? (
                  <div className="mt-10 w-full border border-dashed border-border-subtle px-4 py-6">
                    <div className="label-sm text-text-primary">NO DOCUMENTS INDEXED YET</div>
                    <div className="body-sm text-text-secondary mt-2">
                      The agent answers from your documents: upload some first.
                    </div>
                    <Link to="/documentos" className={`inline-block mt-3 ${emptyActionCls}`}>
                      UPLOAD DOCUMENTS →
                    </Link>
                  </div>
                ) : (
                  collections.isSuccess && (
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
                  )
                )}
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
                      onOpenSources={openSources}
                      onPin={() => pinTurn(t)}
                      pinning={pinNoteMut.isPending && pinNoteMut.variables?.content === t.text}
                      pinned={!!threadId && pinned.has(pinKey(threadId, t.text))}
                      onRetry={
                        t.outcome && i === lastIndex && !streaming && !readOnly
                          ? () => retry(i)
                          : undefined
                      }
                    />
                  ),
                )}

                {live && <StreamingTurn live={live} />}
              </div>
            )}
          </div>

          {/* Input */}
          <div className="border-t border-border-subtle px-3 sm:px-6 md:px-8 py-3 sm:py-4 shrink-0">
            {errorMsg && (
              <div
                role="alert"
                className="max-w-4xl mx-auto mb-3 flex items-start gap-3 border border-error/40 bg-error/5 px-4 py-2 label-sm text-error"
              >
                <span className="flex-1 min-w-0 break-words">ERROR: {errorMsg}</span>
                <button
                  onClick={() => setErrorMsg(null)}
                  aria-label="Dismiss error"
                  className="text-error hover:text-text-primary"
                >
                  ✕
                </button>
              </div>
            )}
            {readOnly ? (
              <div className="max-w-4xl mx-auto label-sm text-text-secondary border border-border-subtle bg-surface-container-low px-4 py-3">
                {channel.toUpperCase()} THREAD · READ ONLY. Its user talks to the agent on{" "}
                {channel === "whatsapp" ? "WhatsApp" : channel}; to ask here, start a{" "}
                <button
                  onClick={() => startNew()}
                  className="text-text-primary hover:text-primary-blue"
                >
                  + NEW
                </button>{" "}
                thread.
              </div>
            ) : (
              // Typing stays open while an answer streams; sending waits for it.
              <Composer
                textareaRef={inputRef}
                value={input}
                onChange={setInput}
                onSubmit={() => void send(input)}
                onEscape={() => streaming && stop()}
                disabled={loadingThread}
                maxLength={LIMITS.chatMessage}
                placeholder="ASK THE AGENT..."
                label="Message to agent"
                hint="ENTER SENDS · SHIFT+ENTER ADDS A LINE · ESC STOPS AN ANSWER"
              >
                {streaming ? (
                  <Btn size="sm" variant="destructive" onClick={stop} title="Stop the answer (Esc)">
                    ■ STOP
                  </Btn>
                ) : (
                  <Btn
                    type="submit"
                    size="sm"
                    variant="ghost"
                    disabled={loadingThread || !input.trim()}
                  >
                    ↵ SEND
                  </Btn>
                )}
              </Composer>
            )}
          </div>
        </section>
      </div>
    </>
  );
}

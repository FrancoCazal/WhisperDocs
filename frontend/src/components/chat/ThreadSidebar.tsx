import { Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import { api, fmtDate, type Collection, type Conversation } from "@/lib/api";
import { LoadingLine } from "@/components/ui-kit/EmptyState";
import { LoadMore } from "@/components/ui-kit/LoadMore";

/** A row's last activity, short: the time (UTC) for today, else the day. */
function lastActive(iso: string): string {
  const utc = new Date(iso).toISOString();
  const day = utc.slice(0, 10);
  return day === new Date().toISOString().slice(0, 10) ? `${utc.slice(11, 16)} UTC` : day;
}

// The thread list's search and SAVED filter; the chat holds them, so the
// desktop sidebar and the mobile panel show the same list.
export type ThreadFilters = {
  q: string;
  onQ: (q: string) => void;
  saved: boolean;
  onSaved: (saved: boolean) => void;
};

export function ThreadSidebar({
  conversations,
  loading,
  collections,
  activeThreadId,
  activeConversationId,
  filters,
  onOpen,
  onNew,
  onNavigate,
  more,
}: {
  conversations: Conversation[] | undefined;
  loading: boolean;
  collections: Collection[] | undefined;
  activeThreadId: string | null;
  // The open thread's row id: its notes are filtered by it.
  activeConversationId: number | null;
  filters: ThreadFilters;
  onOpen: (c: Conversation) => void;
  onNew: () => void;
  // A note's link to its thread was followed (the mobile panel closes).
  onNavigate?: () => void;
  // The threads are paged: the next page, and the total.
  more?: { total: number | null; hasMore: boolean; loading: boolean; onLoad: () => void };
}) {
  const qc = useQueryClient();
  const [tab, setTab] = useState<"threads" | "notes">("threads");
  const [notesScope, setNotesScope] = useState<"thread" | "all">("thread");
  // Without an open thread there is only "all".
  const threadNotes = notesScope === "thread" && activeConversationId !== null;
  const notes = useQuery({
    queryKey: ["notes", threadNotes ? activeConversationId : "all"],
    queryFn: () =>
      api.listNotes({
        limit: 100,
        conversation_id: threadNotes ? activeConversationId : undefined,
      }),
    enabled: tab === "notes",
  });
  const deleteNoteMut = useMutation({
    mutationFn: (noteId: number) => api.deleteNote(noteId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["notes"] }),
    onError: (e) => toast.error("COULD NOT DELETE THE NOTE", { description: e.message }),
  });
  const filtering = filters.q.trim() !== "" || filters.saved;

  const tabCls = (active: boolean) =>
    `flex-1 h-12 label-sm transition-colors ${
      active
        ? "text-text-primary border-b-2 border-text-primary -mb-px"
        : "text-text-tertiary hover:text-text-secondary"
    }`;
  const chipCls = (active: boolean) =>
    `label-xs px-2 h-7 border shrink-0 ${
      active
        ? "border-text-primary text-text-primary"
        : "border-border-subtle text-text-tertiary hover:text-text-primary"
    }`;

  return (
    <>
      {/* Tab header */}
      <div className="flex border-b border-border-subtle shrink-0">
        <button onClick={() => setTab("threads")} className={tabCls(tab === "threads")}>
          THREADS
        </button>
        <button onClick={() => setTab("notes")} className={tabCls(tab === "notes")}>
          NOTES {(notes.data?.length ?? 0) > 0 && `[${notes.data!.length}]`}
        </button>
        {tab === "threads" && (
          <button
            onClick={onNew}
            className="px-3 h-12 label-sm text-text-primary hover:bg-text-primary hover:text-surface border-l border-border-subtle"
          >
            + NEW
          </button>
        )}
      </div>

      {tab === "threads" && (
        <div className="px-3 py-2 border-b border-border-subtle flex items-center gap-2 shrink-0">
          <input
            type="search"
            value={filters.q}
            onChange={(e) => filters.onQ(e.target.value)}
            maxLength={200}
            placeholder="SEARCH TITLES"
            aria-label="Search threads by title"
            className="flex-1 min-w-0 bg-surface-lowest border border-border-subtle px-2 h-7 body-sm text-text-primary placeholder:text-text-tertiary placeholder:label-xs focus:outline-none focus:border-border-focus"
          />
          <button
            type="button"
            onClick={() => filters.onSaved(!filters.saved)}
            aria-pressed={filters.saved}
            className={chipCls(filters.saved)}
          >
            ★ SAVED
          </button>
        </div>
      )}
      {tab === "notes" && activeConversationId !== null && (
        <div
          role="group"
          aria-label="Notes shown"
          className="px-3 py-2 border-b border-border-subtle flex items-center gap-2 shrink-0"
        >
          <button
            type="button"
            onClick={() => setNotesScope("thread")}
            aria-pressed={notesScope === "thread"}
            className={chipCls(notesScope === "thread")}
          >
            THIS THREAD
          </button>
          <button
            type="button"
            onClick={() => setNotesScope("all")}
            aria-pressed={notesScope === "all"}
            className={chipCls(notesScope === "all")}
          >
            ALL THREADS
          </button>
        </div>
      )}

      <div className="flex-1 overflow-auto">
        {/* Threads tab */}
        {tab === "threads" && (
          <>
            {loading && (
              <div className="px-4 py-3">
                <LoadingLine label="LOADING THREADS" />
              </div>
            )}
            {!loading && (conversations?.length ?? 0) === 0 && (
              <div className="px-4 py-3 label-sm text-text-tertiary">
                {filtering ? "(no threads match)" : "(no threads yet)"}
              </div>
            )}
            {conversations?.map((c) => {
              const isActive = c.thread_id === activeThreadId;
              const colName = c.collection_id
                ? (collections?.find((col) => col.id === c.collection_id)?.name ??
                  `COL ${c.collection_id}`)
                : "ALL COLLECTIONS";
              const channelTag = c.channel === "whatsapp" ? "WA" : "WEB";
              return (
                <button
                  key={c.thread_id}
                  onClick={() => onOpen(c)}
                  aria-current={isActive ? "true" : undefined}
                  className={`relative w-full text-left px-4 py-3 border-b border-border-subtle/50 hover:bg-surface-container transition-colors focus-visible:-outline-offset-2 ${
                    isActive ? "bg-surface-container" : ""
                  }`}
                >
                  {isActive && (
                    <span className="absolute left-0 top-0 bottom-0 w-[2px] bg-text-primary" />
                  )}
                  <div className="flex items-start justify-between gap-2">
                    <div
                      className={`body-sm line-clamp-2 break-words ${
                        c.title ? "text-text-primary" : "text-text-tertiary"
                      }`}
                    >
                      {c.title ?? `(untitled) ${c.thread_id.slice(0, 8)}`}
                    </div>
                    {c.is_saved && (
                      <span className="label-xs text-success shrink-0" aria-label="saved">
                        ★
                      </span>
                    )}
                  </div>
                  <div className="label-xs text-text-tertiary mt-1 flex gap-2">
                    <span className="truncate">
                      {colName.toUpperCase()} · {channelTag}
                    </span>
                    <span
                      className="ml-auto shrink-0 tabular-nums"
                      title={`Last active ${fmtDate(c.last_active_at)}`}
                    >
                      {lastActive(c.last_active_at)}
                    </span>
                  </div>
                </button>
              );
            })}
            {more && <LoadMore shown={conversations?.length ?? 0} {...more} className="" />}
          </>
        )}

        {/* Notes tab */}
        {tab === "notes" && (
          <>
            {notes.isLoading && (
              <div className="px-4 py-3">
                <LoadingLine label="LOADING NOTES" />
              </div>
            )}
            {!notes.isLoading && (notes.data?.length ?? 0) === 0 && (
              <div className="px-4 py-3 label-sm text-text-tertiary">
                {threadNotes ? "(no notes pinned in this thread)" : "(no pinned notes)"}
              </div>
            )}
            {notes.data?.map((n) => (
              <div key={n.id} className="px-4 py-3 border-b border-border-subtle/50 group">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <span className="label-xs text-text-tertiary tabular-nums">
                    {fmtDate(n.created_at)}
                  </span>
                  <button
                    onClick={() => deleteNoteMut.mutate(n.id)}
                    disabled={deleteNoteMut.isPending}
                    aria-label="Delete note"
                    className="label-xs text-text-tertiary hover:text-error opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity"
                  >
                    ✕
                  </button>
                </div>
                <div className="body-sm text-text-secondary line-clamp-3">
                  {n.source_message_content}
                </div>
                {n.user_note && (
                  <div className="label-xs text-text-tertiary mt-1 italic">{n.user_note}</div>
                )}
                {n.thread_id && n.thread_id !== activeThreadId && (
                  <Link
                    to="/chat"
                    search={{ thread: n.thread_id }}
                    onClick={onNavigate}
                    className="mt-2 inline-block label-xs text-text-secondary hover:text-primary-blue"
                  >
                    OPEN THREAD {n.thread_id.slice(0, 8)} →
                  </Link>
                )}
              </div>
            ))}
          </>
        )}
      </div>
    </>
  );
}

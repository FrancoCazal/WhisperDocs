import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import { api, fmtDate, type Collection, type Conversation } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { LoadMore } from "@/components/ui-kit/LoadMore";

export function ThreadSidebar({
  conversations,
  loading,
  collections,
  activeThreadId,
  onOpen,
  onNew,
  more,
}: {
  conversations: Conversation[] | undefined;
  loading: boolean;
  collections: Collection[] | undefined;
  activeThreadId: string | null;
  onOpen: (c: Conversation) => void;
  onNew: () => void;
  // The threads are paged: the next page, and the total.
  more?: { total: number | null; hasMore: boolean; loading: boolean; onLoad: () => void };
}) {
  const qc = useQueryClient();
  const [tab, setTab] = useState<"threads" | "notes">("threads");
  const notes = useQuery({
    queryKey: ["notes"],
    queryFn: () => api.listNotes({ limit: 100 }),
    enabled: tab === "notes",
  });
  const deleteNoteMut = useMutation({
    mutationFn: (noteId: number) => api.deleteNote(noteId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["notes"] }),
    onError: (e) => toast.error("COULD NOT DELETE THE NOTE", { description: e.message }),
  });

  return (
    <>
      {/* Tab header */}
      <div className="flex border-b border-border-subtle shrink-0">
        <button
          onClick={() => setTab("threads")}
          className={`flex-1 h-12 label-sm transition-colors ${
            tab === "threads"
              ? "text-text-primary border-b-2 border-text-primary -mb-px"
              : "text-text-tertiary hover:text-text-secondary"
          }`}
        >
          THREADS
        </button>
        <button
          onClick={() => setTab("notes")}
          className={`flex-1 h-12 label-sm transition-colors ${
            tab === "notes"
              ? "text-text-primary border-b-2 border-text-primary -mb-px"
              : "text-text-tertiary hover:text-text-secondary"
          }`}
        >
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

      <div className="flex-1 overflow-auto">
        {/* Threads tab */}
        {tab === "threads" && (
          <>
            {loading && (
              <div className="px-4 py-3 space-y-2">
                {[0, 1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-12 w-full bg-surface-container" />
                ))}
              </div>
            )}
            {!loading && (conversations?.length ?? 0) === 0 && (
              <div className="px-4 py-3 label-sm text-text-tertiary">(no threads yet)</div>
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
                  className={`relative w-full text-left px-4 py-3 border-b border-border-subtle/50 hover:bg-surface-container transition-colors ${
                    isActive ? "bg-surface-container" : ""
                  }`}
                >
                  {isActive && (
                    <span className="absolute left-0 top-0 bottom-0 w-[2px] bg-text-primary" />
                  )}
                  <div className="flex items-center justify-between gap-2">
                    <div className="label-sm text-text-primary truncate">
                      {colName.toUpperCase()}
                    </div>
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
            {more && <LoadMore shown={conversations?.length ?? 0} {...more} className="" />}
          </>
        )}

        {/* Notes tab */}
        {tab === "notes" && (
          <>
            {notes.isLoading && (
              <div className="px-4 py-3 label-sm text-text-tertiary">LOADING...</div>
            )}
            {!notes.isLoading && (notes.data?.length ?? 0) === 0 && (
              <div className="px-4 py-3 label-sm text-text-tertiary">(no pinned notes)</div>
            )}
            {notes.data?.map((n) => (
              <div key={n.id} className="px-4 py-3 border-b border-border-subtle/50 group">
                <div className="flex items-center justify-between mb-1">
                  <span className="label-xs text-text-tertiary tabular-nums">
                    {n.conversation_id ? `THREAD · ` : ""}
                    {fmtDate(n.created_at)}
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
                  <div className="label-xs text-text-tertiary mt-1 italic">{n.user_note}</div>
                )}
              </div>
            ))}
          </>
        )}
      </div>
    </>
  );
}

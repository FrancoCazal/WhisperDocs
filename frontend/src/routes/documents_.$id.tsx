import {
  createFileRoute,
  Link,
  useNavigate,
  useRouter,
  type ErrorComponentProps,
} from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { toast } from "sonner";
import { api, ApiError, fmtDate, fmtNumber } from "@/lib/api";
import { LoadingLine, ErrorBanner } from "@/components/ui-kit/EmptyState";
import { Btn, btnCls } from "@/components/ui-kit/Button";
import { ConfirmDialog } from "@/components/ui-kit/ConfirmDialog";
import { ChunkFlags } from "@/components/documents/Flags";
import { FLAGS_LEAD } from "@/lib/instructionFlags";

// "documents_": the page renders on its own, not inside the list's layout,
// which has no outlet for it.
export const Route = createFileRoute("/documents_/$id")({
  head: () => ({
    meta: [
      { title: "Document Detail — WhisperDocs" },
      { name: "description", content: "Inspect document metadata and chunks." },
    ],
  }),
  // ?chunk=N: the chunk a citation points at, shown expanded and in view.
  validateSearch: (search: Record<string, unknown>): { chunk?: number } => {
    const n = Number(search.chunk);
    return search.chunk !== undefined && Number.isInteger(n) && n >= 0 ? { chunk: n } : {};
  },
  component: DocumentDetailPage,
  errorComponent: DocumentDetailError,
  notFoundComponent: () => (
    <div>
      <div className="label-sm text-error">DOCUMENT_NOT_FOUND</div>
      <Link to="/documents" className="label-sm text-text-primary mt-3 inline-block">
        ← RETURN TO INDEX
      </Link>
    </div>
  ),
});

// A named component, so the hook below runs inside a component.
function DocumentDetailError({ error, reset }: ErrorComponentProps) {
  const router = useRouter();
  return (
    <div>
      <ErrorBanner
        message={error instanceof Error ? error.message : String(error)}
        onRetry={() => {
          void router.invalidate();
          reset();
        }}
      />
    </div>
  );
}

const fieldCls =
  "bg-surface-lowest border border-border-subtle px-2 h-9 text-text-primary body-sm focus:outline-none focus:border-border-focus";

function DocumentDetailPage() {
  const { id } = Route.useParams();
  const { chunk: cited } = Route.useSearch();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const docId = Number(id);
  const validId = Number.isInteger(docId) && docId > 0;
  const doc = useQuery({
    queryKey: ["document", docId],
    queryFn: () => api.getDocument(docId),
    enabled: validId,
    // A missing document stays missing.
    retry: (n, e) => !(e instanceof ApiError && e.status === 404) && n < 3,
  });
  const chunks = useQuery({
    queryKey: ["chunks", docId],
    queryFn: () => api.getChunks(docId),
    enabled: validId && doc.isSuccess,
  });
  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const notFound = !validId || (doc.error instanceof ApiError && doc.error.status === 404);

  const [confirmDelete, setConfirmDelete] = useState(false);
  const [moveTarget, setMoveTarget] = useState<number | null>(null);
  // Filters the loaded chunks: a document's chunks all come in one response.
  const [chunkQ, setChunkQ] = useState("");
  const needle = chunkQ.trim();
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const filtering = needle !== "" || flaggedOnly;

  const collectionName = (cid: number) =>
    collections.data?.find((c) => c.id === cid)?.name ?? `#${cid}`;
  const otherCollections = (collections.data ?? []).filter((c) => c.id !== doc.data?.collection_id);

  // The list and the collections' totals count this document too.
  const refreshLists = () => {
    void qc.invalidateQueries({ queryKey: ["documents"] });
    void qc.invalidateQueries({ queryKey: ["collections"] });
  };

  const del = useMutation({
    mutationFn: () => api.deleteDocument(docId),
    onSuccess: () => {
      const title = doc.data?.title ?? `#${docId}`;
      refreshLists();
      toast.success(`DELETED ${title}`);
      // Dropped once the page is gone: dropped before, it would refetch a 404.
      void navigate({ to: "/documents" }).then(() => {
        qc.removeQueries({ queryKey: ["document", docId] });
        qc.removeQueries({ queryKey: ["chunks", docId] });
      });
    },
    onError: (e) => toast.error("COULD NOT DELETE THE DOCUMENT", { description: e.message }),
  });

  const move = useMutation({
    mutationFn: (target: number) => api.bulkMoveDocuments([docId], target),
    onSuccess: (r, target) => {
      if (r.succeeded === 1) {
        toast.success(`MOVED TO ${collectionName(target)}`);
        setMoveTarget(null);
      } else {
        toast.error("COULD NOT MOVE THE DOCUMENT", {
          description: r.not_found.length > 0 ? "It no longer exists." : r.errors[String(docId)],
        });
      }
    },
    onError: (e) => toast.error("COULD NOT MOVE THE DOCUMENT", { description: e.message }),
    onSettled: () => {
      refreshLists();
      void qc.invalidateQueries({ queryKey: ["document", docId] });
    },
  });

  const allChunks = chunks.data ?? [];
  const shownChunks = allChunks.filter(
    (c) =>
      (!flaggedOnly || (c.instruction_flags?.length ?? 0) > 0) &&
      (!needle || c.text.toLowerCase().includes(needle.toLowerCase())),
  );
  const flagged = doc.data?.flagged_chunks ?? 0;
  const clearFilters = () => {
    setChunkQ("");
    setFlaggedOnly(false);
  };

  return (
    <div>
      <div className="label-sm text-text-tertiary mb-2">
        <Link to="/documents" className="hover:text-primary-blue">
          DOCUMENTS
        </Link>{" "}
        / <span className="text-text-secondary">{doc.data?.title ?? id}</span>
      </div>

      <div className="flex flex-wrap items-end justify-between gap-x-4 gap-y-3 mb-6 pb-4 border-b border-border-subtle">
        <h1 className="title-lg text-text-primary min-w-0 break-words">
          {doc.data?.title ??
            (notFound ? "DOCUMENT NOT FOUND" : doc.error ? "DOCUMENT UNAVAILABLE" : "LOADING...")}
        </h1>
        {doc.data && (
          <div className="flex flex-wrap items-center gap-2">
            <Link
              to="/chat"
              search={{ document: doc.data.id }}
              className={btnCls("primary")}
              title="A new chat that searches only this document."
            >
              ASK IN CHAT
            </Link>
            {otherCollections.length > 0 && (
              <div className="flex items-center gap-2">
                <select
                  value={moveTarget ?? ""}
                  onChange={(e) => setMoveTarget(e.target.value ? Number(e.target.value) : null)}
                  aria-label="Collection to move the document to"
                  className={fieldCls}
                >
                  <option value="">MOVE TO…</option>
                  {otherCollections.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
                <Btn
                  variant="quiet"
                  disabled={moveTarget === null || move.isPending}
                  onClick={() => moveTarget !== null && move.mutate(moveTarget)}
                >
                  {move.isPending ? "MOVING..." : "MOVE"}
                </Btn>
              </div>
            )}
            <Btn
              variant="destructive"
              disabled={del.isPending}
              onClick={() => setConfirmDelete(true)}
            >
              {del.isPending ? "DELETING..." : "✕ DELETE"}
            </Btn>
          </div>
        )}
      </div>

      <ConfirmDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title={`DELETE "${doc.data?.title ?? ""}"?`}
        confirmLabel="✕ DELETE"
        destructive
        onConfirm={() => del.mutate()}
      >
        Its {fmtNumber(doc.data?.chunk_count)} chunks are removed from the index. This cannot be
        undone.
      </ConfirmDialog>

      {notFound ? (
        <div className="body-sm text-text-secondary">
          No document with id {id}. It may have been deleted.{" "}
          <Link to="/documents" className="text-text-primary hover:text-primary-blue">
            ← Back to documents
          </Link>
        </div>
      ) : (
        doc.error && (
          <ErrorBanner
            message={`failed to load document: ${doc.error.message}`}
            onRetry={() => void doc.refetch()}
            retrying={doc.isFetching}
          />
        )
      )}
      {chunks.error && (
        <ErrorBanner
          message={`failed to load chunks: ${chunks.error.message}`}
          onRetry={() => void chunks.refetch()}
          retrying={chunks.isFetching}
        />
      )}

      {doc.isLoading ? (
        <LoadingLine />
      ) : doc.data ? (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-4 mb-8">
            <Meta label="FILENAME" value={doc.data.filename} />
            <Meta label="TYPE" value={doc.data.file_type.toUpperCase()} />
            <Meta label="COLLECTION" value={collectionName(doc.data.collection_id)} />
            <Meta label="CHUNKS" value={fmtNumber(doc.data.chunk_count)} />
            <Meta label="TOTAL TOKENS" value={fmtNumber(doc.data.total_tokens)} />
            <Meta label="CREATED" value={fmtDate(doc.data.created_at)} />
          </div>

          {flagged > 0 && (
            <div className="border border-warning/40 bg-surface-container-low px-3 py-2 mb-6 flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
              <div className="body-sm text-text-secondary">
                <span className="label-sm text-warning">
                  <span aria-hidden="true">⚠ </span>
                  {flagged} {flagged === 1 ? "CHUNK READS" : "CHUNKS READ"} AS AN INSTRUCTION TO THE
                  MODEL.
                </span>{" "}
                {FLAGS_LEAD}
              </div>
              <Btn variant="quiet" size="sm" onClick={() => setFlaggedOnly((v) => !v)}>
                {flaggedOnly ? "SHOW ALL CHUNKS" : "SHOW FLAGGED ONLY"}
              </Btn>
            </div>
          )}

          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 mb-4">
            <h2 className="title-sm text-text-primary">CHUNKS</h2>
            {allChunks.length > 0 && (
              <div className="flex flex-wrap items-center gap-3">
                <span role="status" className="label-xs text-text-secondary tabular-nums">
                  {filtering
                    ? `${fmtNumber(shownChunks.length)} OF ${fmtNumber(allChunks.length)} MATCH`
                    : ""}
                </span>
                <input
                  type="search"
                  value={chunkQ}
                  onChange={(e) => setChunkQ(e.target.value)}
                  maxLength={200}
                  placeholder="SEARCH CHUNKS"
                  aria-label="Search this document's chunks"
                  className={`${fieldCls} w-56 max-w-full placeholder:text-text-tertiary placeholder:label-xs`}
                />
              </div>
            )}
          </div>
          {chunks.isLoading ? (
            <LoadingLine />
          ) : shownChunks.length > 0 ? (
            <div className="space-y-3">
              {shownChunks.map((c) => (
                <ChunkBlock
                  key={c.id}
                  index={c.chunk_index}
                  tokens={c.token_count}
                  text={c.text}
                  flags={c.instruction_flags}
                  cited={c.chunk_index === cited}
                  query={needle}
                />
              ))}
            </div>
          ) : filtering ? (
            <div className="label-sm text-text-tertiary">
              {needle ? (
                <>
                  NO {flaggedOnly ? "FLAGGED " : ""}CHUNK CONTAINS{" "}
                  <span className="normal-case">"{needle}"</span>.
                </>
              ) : (
                "NO FLAGGED CHUNKS."
              )}{" "}
              <button
                type="button"
                onClick={clearFilters}
                className="text-text-primary hover:text-primary-blue"
              >
                {needle && flaggedOnly ? "CLEAR FILTERS" : needle ? "CLEAR SEARCH" : "SHOW ALL"}
              </button>
            </div>
          ) : (
            <div className="label-sm text-text-tertiary">NO CHUNKS</div>
          )}
        </>
      ) : null}
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-border-subtle bg-surface-container p-3">
      <div className="label-sm text-text-secondary mb-1">{label}</div>
      <div className="body-sm text-text-primary tabular-nums break-words">{value}</div>
    </div>
  );
}

// Its top, below the fixed header (scroll-mt): a long chunk centered hides its heading.
function scrollIntoView(el: HTMLElement | null) {
  el?.scrollIntoView({ block: "start" });
}

/** `text` with each case-insensitive occurrence of `query` marked. */
function highlight(text: string, query: string): ReactNode {
  const lower = text.toLowerCase();
  const needle = query.toLowerCase();
  // Lowercasing can change a string's length (İ): positions would not line up.
  if (!needle || lower.length !== text.length) return text;
  const parts: ReactNode[] = [];
  let from = 0;
  for (let at = lower.indexOf(needle); at !== -1; at = lower.indexOf(needle, from)) {
    if (at > from) parts.push(text.slice(from, at));
    parts.push(
      <mark key={at} className="bg-primary-blue/30 text-text-primary">
        {text.slice(at, at + needle.length)}
      </mark>,
    );
    from = at + needle.length;
  }
  parts.push(text.slice(from));
  return parts;
}

function ChunkBlock({
  index,
  tokens,
  text,
  flags,
  cited,
  query,
}: {
  index: number;
  tokens: number;
  text: string;
  flags: string[] | undefined;
  cited: boolean;
  query: string;
}) {
  const [expanded, setExpanded] = useState(cited);
  const lines = text.split("\n");
  // While searching, a chunk shows in full: the match may be past its first lines.
  const isLong = !query && (lines.length > 4 || text.length > 400);
  const display = expanded || !isLong ? text : lines.slice(0, 4).join("\n");

  return (
    <div
      ref={cited ? scrollIntoView : undefined}
      aria-current={cited || undefined}
      className={`border-b border-border-subtle py-3 ${cited ? "border-l-2 border-l-primary-blue pl-3 scroll-mt-16" : ""}`}
    >
      <div className="label-sm text-text-secondary mb-2 flex flex-wrap items-center gap-x-3 gap-y-1">
        <span>[CHUNK_INDEX: {index}]</span>
        {cited && <span className="label-xs text-primary-blue">CITED</span>}
        <span className="text-text-tertiary">·</span>
        <span>TOKENS: {tokens}</span>
        <ChunkFlags rules={flags} />
      </div>
      <pre className="body-sm text-text-primary whitespace-pre-wrap break-words bg-surface-lowest p-3 border border-border-subtle">
        {highlight(display, query)}
        {isLong && !expanded && "..."}
      </pre>
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          className="label-sm text-text-secondary hover:text-text-primary mt-2"
        >
          {expanded ? "▴ COLLAPSE" : "▾ EXPAND"}
        </button>
      )}
    </div>
  );
}

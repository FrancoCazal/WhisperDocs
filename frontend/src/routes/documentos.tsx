import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  api,
  fmtDate,
  fmtNumber,
  type BulkOpResult,
  type DocumentRead,
  type IngestionStatus,
} from "@/lib/api";
import { PageHeader } from "@/components/ui-kit/PageHeader";
import { LoadMore } from "@/components/ui-kit/LoadMore";
import { PAGE_SIZE, usePagedList } from "@/lib/paged";
import { ConfirmDialog } from "@/components/ui-kit/ConfirmDialog";
import { RecentJobs } from "@/components/documents/RecentJobs";
import { ACCEPT_ATTR, rejectReason, stageText } from "@/components/documents/stages";
import { Btn, btnCls } from "@/components/ui-kit/Button";
import {
  EmptyState,
  ErrorBanner,
  LoadingLine,
  emptyActionCls,
} from "@/components/ui-kit/EmptyState";

type SearchParams = { collection_id?: number };

export const Route = createFileRoute("/documentos")({
  head: () => ({
    meta: [
      { title: "Documents — WhisperDocs" },
      {
        name: "description",
        content: "Indexed corpus and document upload for the WhisperDocs RAG system.",
      },
    ],
  }),
  validateSearch: (s: Record<string, unknown>): SearchParams => ({
    collection_id: s.collection_id ? Number(s.collection_id) : undefined,
  }),
  component: DocumentsPage,
});

type UploadStatus = {
  // Rows are found by id: a later batch, or a row cleared, never shifts them.
  id: number;
  filename: string;
  status: "queued" | "uploading" | "running" | "success" | "error";
  progress: number;
  stage: IngestionStatus | null;
  // The job's chunks, once it has counted them.
  chunks?: { done: number; total: number };
  // Set once indexed.
  documentId?: number | null;
  error?: string;
};

type QueuedUpload = { id: number; file: File; collectionId: number };

// Files ingest a few at a time: each one holds an SSE stream until it is
// indexed, and the browser allows six connections per host.
const UPLOAD_CONCURRENCY = 3;
// When a stream ends without its final event, the job is polled instead.
const POLL_MS = 2000;
const POLL_MAX_MISSES = 5;

function wait(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        reject(signal.reason);
      },
      { once: true },
    );
  });
}

const chunksOf = (job: { processed_chunks: number; total_chunks: number }) =>
  job.total_chunks > 0 ? { done: job.processed_chunks, total: job.total_chunks } : undefined;

function DocumentsPage() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const search = Route.useSearch();
  const [filter, setFilter] = useState<number | null>(search.collection_id ?? null);
  const [uploadCollection, setUploadCollection] = useState<number | null>(
    search.collection_id ?? null,
  );
  const [uploadStatuses, setUploadStatuses] = useState<UploadStatus[]>([]);
  // The upload queue and the controllers of the files in flight. Leaving the
  // page aborts them; jobs the server already has keep running there.
  const uploads = useRef({
    nextId: 0,
    queue: [] as QueuedUpload[],
    active: new Set<AbortController>(),
  });
  useEffect(() => {
    const u = uploads.current;
    return () => {
      u.queue.length = 0;
      for (const c of u.active) c.abort();
    };
  }, []);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const documents = usePagedList(["documents", "pages", filter ?? "all"], (cursor) =>
    api.listDocumentsPage({ collection_id: filter ?? undefined, limit: PAGE_SIZE, cursor }),
  );
  // Documents feed the collections' totals too.
  const refreshDocuments = () => {
    void qc.invalidateQueries({ queryKey: ["documents"] });
    void qc.invalidateQueries({ queryKey: ["collections"] });
    void qc.invalidateQueries({ queryKey: ["ingestion-jobs"] });
  };

  const collectionName = (id: number) =>
    collections.data?.find((c) => c.id === id)?.name ?? `#${id}`;
  // Where a drop goes, shown in the select: the chosen collection, else the first.
  const uploadTarget = uploadCollection ?? collections.data?.[0]?.id ?? null;
  const noCollections = collections.isSuccess && collections.data.length === 0;

  function updateStatus(id: number, patch: Partial<UploadStatus>) {
    setUploadStatuses((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  }

  async function ingestFile({ id, file, collectionId }: QueuedUpload, signal: AbortSignal) {
    const succeed = (documentId: number | null) =>
      updateStatus(id, { status: "success", stage: "complete", progress: 100, documentId });
    const fail = (error: string) => updateStatus(id, { status: "error", stage: "failed", error });
    try {
      updateStatus(id, { status: "uploading" });
      const { job_id } = await api.uploadAsync(collectionId, file, { signal });
      updateStatus(id, { status: "running", stage: "pending" });
      let ended = false;
      try {
        await api.streamIngestionJob(
          job_id,
          (e) => {
            if (e.type === "progress") {
              updateStatus(id, {
                status: "running",
                stage: e.status,
                progress: e.progress,
                chunks: chunksOf(e),
              });
            } else if (e.type === "complete") {
              ended = true;
              succeed(e.document_id);
            } else if (e.type === "error") {
              ended = true;
              fail(e.message);
            }
          },
          signal,
        );
      } catch (err) {
        // A dropped stream (the API restarting, the network) is no verdict:
        // the job runs on, so ask it below.
        if (signal.aborted) throw err;
      }
      // The stream ended without its final event: poll the job itself.
      let misses = 0;
      while (!ended) {
        try {
          const job = await api.getIngestionJob(job_id);
          misses = 0;
          if (job.status === "complete") return succeed(job.document_id);
          if (job.status === "failed") return fail(job.error_message ?? "ingestion failed");
          updateStatus(id, {
            status: "running",
            stage: job.status,
            progress: job.progress,
            chunks: chunksOf(job),
          });
        } catch (err) {
          if (++misses >= POLL_MAX_MISSES) throw err;
        }
        await wait(POLL_MS, signal);
      }
    } catch (err) {
      if (signal.aborted) return;
      fail(err instanceof Error ? err.message : String(err));
    }
  }

  // Starts queued files while there is room; each one that ends makes room.
  function pump() {
    const u = uploads.current;
    while (u.active.size < UPLOAD_CONCURRENCY && u.queue.length > 0) {
      const next = u.queue.shift()!;
      const ctrl = new AbortController();
      u.active.add(ctrl);
      void ingestFile(next, ctrl.signal).finally(() => {
        u.active.delete(ctrl);
        refreshDocuments();
        pump();
      });
    }
  }

  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [confirmBulk, setConfirmBulk] = useState(false);
  const [toDelete, setToDelete] = useState<DocumentRead | null>(null);
  const [bulkResult, setBulkResult] = useState<BulkOpResult | null>(null);

  // Only rows on screen can be acted on: a selection made under another filter,
  // or of rows a refresh removed, is never sent.
  const visible = documents.items;
  const selectedIds = visible.filter((d) => selected.has(d.id)).map((d) => d.id);

  function changeFilter(next: number | null) {
    setFilter(next);
    setSelected(new Set());
  }

  const del = useMutation({
    mutationFn: (d: DocumentRead) => api.deleteDocument(d.id),
    onSuccess: (_r, d) => {
      refreshDocuments();
      toast.success(`DELETED ${d.title}`);
    },
    onError: (e, d) => toast.error(`COULD NOT DELETE ${d.title}`, { description: e.message }),
  });

  const bulkDel = useMutation({
    mutationFn: (ids: number[]) => api.bulkDeleteDocuments(ids),
    onMutate: () => setBulkResult(null),
    onSuccess: (result) => {
      setBulkResult(result);
      setSelected(new Set());
    },
    onError: (e) => toast.error("BULK DELETE FAILED", { description: e.message }),
    onSettled: refreshDocuments,
  });

  function toggleSel(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function handleFiles(files: File[]) {
    // No target: the zone says to create a collection instead of taking files.
    const collectionId = uploadTarget;
    if (files.length === 0 || collectionId === null) return;
    const u = uploads.current;
    // A file the backend would reject is not sent: its row says why at once.
    const rows = files.map((file) => ({
      id: u.nextId++,
      file,
      collectionId,
      error: rejectReason(file),
    }));
    // A new batch joins the list; rows of earlier ones stay until cleared.
    setUploadStatuses((prev) => [
      ...prev,
      ...rows.map((r): UploadStatus =>
        r.error
          ? {
              id: r.id,
              filename: r.file.name,
              status: "error",
              progress: 0,
              stage: null,
              error: r.error,
            }
          : { id: r.id, filename: r.file.name, status: "queued", progress: 0, stage: null },
      ),
    ]);
    u.queue.push(...rows.filter((r) => !r.error));
    pump();
  }

  const queuedCount = uploadStatuses.filter((u) => u.status === "queued").length;
  const finishedCount = uploadStatuses.filter(
    (u) => u.status === "success" || u.status === "error",
  ).length;

  return (
    <div>
      <PageHeader
        title="DOCUMENTS"
        subtitle={
          documents.total === null
            ? "INDEXED CORPUS"
            : `INDEXED CORPUS · ${fmtNumber(documents.total)} DOCUMENTS`
        }
        actions={
          <label className="flex items-center gap-2 label-sm text-text-secondary">
            FILTER:
            <select
              value={filter ?? ""}
              onChange={(e) => changeFilter(e.target.value ? Number(e.target.value) : null)}
              className="bg-surface-lowest border border-border-subtle px-2 h-9 text-text-primary body-sm focus:outline-none focus:border-border-focus"
            >
              <option value="">ALL COLLECTIONS</option>
              {collections.data?.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </label>
        }
      />

      {/* Upload zone */}
      <div className="mb-6">
        <label className="flex flex-wrap items-center gap-2 sm:gap-3 mb-2">
          <span className="label-sm text-text-secondary">UPLOAD TO COLLECTION:</span>
          <select
            value={uploadTarget ?? ""}
            onChange={(e) => setUploadCollection(e.target.value ? Number(e.target.value) : null)}
            disabled={uploadTarget === null}
            className="bg-surface-lowest border border-border-subtle px-2 h-8 text-text-primary body-sm focus:outline-none focus:border-border-focus disabled:text-text-tertiary"
            aria-label="Upload target collection"
          >
            {uploadTarget === null && <option value="">{noCollections ? "NONE" : "—"}</option>}
            {collections.data?.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        {noCollections ? (
          <div className="bg-surface-lowest border border-dashed border-border-subtle text-center py-10 px-4">
            <div className="label-sm text-text-primary">CREATE A COLLECTION FIRST</div>
            <div className="label-xs text-text-tertiary mt-2">
              DOCUMENTS ARE INDEXED INTO A COLLECTION
            </div>
            <Link to="/colecciones" className={`inline-block mt-3 ${emptyActionCls}`}>
              GO TO COLLECTIONS →
            </Link>
          </div>
        ) : (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              handleFiles(Array.from(e.dataTransfer.files));
            }}
            onClick={() => fileInputRef.current?.click()}
            className={`bg-surface-lowest border border-dashed cursor-pointer text-center py-10 transition-colors ${
              dragOver ? "border-border-focus" : "border-border-subtle"
            }`}
          >
            <div className="label-sm text-text-primary">DROP FILES HERE OR CLICK TO SELECT</div>
            <div className="label-xs text-text-tertiary mt-2">
              SUPPORTED: PDF, MD, TXT · MAX 10 MB PER FILE
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPT_ATTR}
              multiple
              className="hidden"
              onChange={(e) => handleFiles(Array.from(e.target.files ?? []))}
            />
          </div>
        )}

        {uploadStatuses.length > 0 && (
          <div className="mt-3 border border-border-subtle bg-surface-container-low">
            <div className="px-4 py-2 flex flex-wrap items-center justify-between gap-2 border-b border-border-subtle label-xs text-text-tertiary">
              <span>
                UPLOADS · {uploadStatuses.length - queuedCount - finishedCount} IN PROGRESS
                {queuedCount > 0 && ` · ${queuedCount} QUEUED, NOT SENT IF YOU LEAVE THIS PAGE`}
              </span>
              {finishedCount > 0 && (
                <button
                  type="button"
                  onClick={() =>
                    setUploadStatuses((prev) =>
                      prev.filter((u) => u.status !== "success" && u.status !== "error"),
                    )
                  }
                  className="label-xs text-text-secondary hover:text-text-primary"
                >
                  CLEAR FINISHED
                </button>
              )}
            </div>
            {uploadStatuses.map((u) => {
              const labelColor =
                u.status === "success"
                  ? "text-success"
                  : u.status === "error"
                    ? "text-error"
                    : "text-text-secondary";
              const stageLabel =
                u.status === "success"
                  ? "✓ INDEXED"
                  : u.status === "error"
                    ? `✗ ${u.error ?? "FAILED"}`
                    : u.status === "uploading"
                      ? "UPLOADING"
                      : u.stage
                        ? stageText(u.stage)
                        : "QUEUED";
              return (
                <div key={u.id} className="px-4 py-2 border-b border-border-subtle/50">
                  <div className="flex items-center justify-between gap-3 body-sm">
                    <span className="text-text-primary truncate min-w-0">{u.filename}</span>
                    <span className={`label-sm shrink-0 tabular-nums ${labelColor}`}>
                      {stageLabel}
                      {u.status === "running" && u.chunks && (
                        <>
                          {" "}
                          · {fmtNumber(u.chunks.done)}/{fmtNumber(u.chunks.total)} CHUNKS
                        </>
                      )}
                      {u.status === "running" && ` · ${u.progress}%`}
                      {u.status === "success" && u.documentId != null && (
                        <Link
                          to="/documentos/$id"
                          params={{ id: String(u.documentId) }}
                          className="ml-3 text-text-secondary hover:text-primary-blue"
                        >
                          OPEN →
                        </Link>
                      )}
                    </span>
                  </div>
                  {u.status === "running" && (
                    <progress
                      value={u.progress}
                      max={100}
                      aria-label={`${u.filename} progress`}
                      className="mt-1 block w-full h-1 appearance-none border-0 bg-surface-container [&::-webkit-progress-bar]:bg-surface-container [&::-webkit-progress-value]:bg-text-primary [&::-webkit-progress-value]:transition-all [&::-moz-progress-bar]:bg-text-primary"
                    />
                  )}
                </div>
              );
            })}
          </div>
        )}
        <RecentJobs collectionId={filter} collections={collections.data} />
      </div>

      {documents.query.error && (
        <ErrorBanner
          message="failed to load documents"
          onRetry={() => void documents.query.refetch()}
          retrying={documents.query.isFetching}
        />
      )}
      {bulkResult && <BulkResultBanner result={bulkResult} onClose={() => setBulkResult(null)} />}

      <ConfirmDialog
        open={confirmBulk}
        onOpenChange={setConfirmBulk}
        title={`DELETE ${selectedIds.length} DOCUMENT(S)?`}
        confirmLabel="✕ DELETE"
        destructive
        onConfirm={() => bulkDel.mutate(selectedIds)}
      >
        Their chunks are removed from the index. This cannot be undone.
      </ConfirmDialog>
      <ConfirmDialog
        open={toDelete !== null}
        onOpenChange={(open) => !open && setToDelete(null)}
        title={`DELETE "${toDelete?.title ?? ""}"?`}
        confirmLabel="✕ DELETE"
        destructive
        onConfirm={() => toDelete && del.mutate(toDelete)}
      >
        Its {fmtNumber(toDelete?.chunk_count)} chunks are removed from the index. This cannot be
        undone.
      </ConfirmDialog>

      {selectedIds.length > 0 && (
        <div
          role="region"
          aria-label="Bulk actions"
          className="mb-3 flex flex-wrap items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2 bg-surface-container-low border border-border-subtle"
        >
          <span className="label-sm text-text-secondary">{selectedIds.length} SELECTED</span>
          <span className="flex-1 hidden sm:block" />
          <Btn size="sm" variant="quiet" onClick={() => setSelected(new Set())}>
            CLEAR
          </Btn>
          <Btn
            size="sm"
            variant="destructive"
            onClick={() => setConfirmBulk(true)}
            disabled={bulkDel.isPending}
          >
            {bulkDel.isPending ? "DELETING..." : "✕ DELETE SELECTED"}
          </Btn>
        </div>
      )}

      <div className="border border-border-subtle bg-surface-container overflow-x-auto">
        {documents.query.isLoading ? (
          <div className="p-4">
            <LoadingLine />
          </div>
        ) : visible.length === 0 ? (
          filter !== null ? (
            <EmptyState label={`NO DOCUMENTS IN ${collectionName(filter).toUpperCase()}`}>
              <button type="button" onClick={() => changeFilter(null)} className={emptyActionCls}>
                SHOW ALL COLLECTIONS
              </button>
            </EmptyState>
          ) : noCollections ? (
            // The upload zone above already says to create a collection.
            <EmptyState label="NO DOCUMENTS INDEXED" />
          ) : (
            <EmptyState label="NO DOCUMENTS INDEXED">
              <span className="text-text-secondary">DROP FILES ABOVE TO INDEX THEM</span>
            </EmptyState>
          )
        ) : (
          <table className="w-full min-w-[720px]">
            <thead>
              <tr className="label-sm text-text-secondary border-b border-border-subtle">
                <th className="w-10 pl-4 py-3">
                  <input
                    type="checkbox"
                    checked={selectedIds.length > 0 && selectedIds.length === visible.length}
                    onChange={(e) =>
                      setSelected(e.target.checked ? new Set(visible.map((d) => d.id)) : new Set())
                    }
                    aria-label="Select all"
                  />
                </th>
                <th className="text-left pl-2 py-3 font-normal">TITLE</th>
                <th className="text-left py-3 font-normal w-16">TYPE</th>
                <th className="text-left py-3 font-normal">COLLECTION</th>
                <th className="text-right py-3 font-normal">CHUNKS</th>
                <th className="text-right py-3 font-normal">TOKENS</th>
                <th className="text-left pl-6 py-3 font-normal">CREATED</th>
                <th className="text-right pr-6 py-3 font-normal w-24">ACTIONS</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((d) => (
                <tr
                  key={d.id}
                  onClick={() => navigate({ to: "/documentos/$id", params: { id: String(d.id) } })}
                  className="body-sm h-9 border-b border-border-subtle/50 hover:bg-surface-high cursor-pointer"
                >
                  <td className="w-10 pl-4" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={selected.has(d.id)}
                      onChange={() => toggleSel(d.id)}
                      aria-label={`Select ${d.title}`}
                    />
                  </td>
                  <td className="pl-2 text-text-primary">
                    {/* The row opens on a click; the link is the way in by keyboard. */}
                    <Link
                      to="/documentos/$id"
                      params={{ id: String(d.id) }}
                      onClick={(e) => e.stopPropagation()}
                      className="hover:text-primary-blue"
                    >
                      {d.title}
                    </Link>
                  </td>
                  <td className="label-sm text-text-secondary">{d.file_type.toUpperCase()}</td>
                  <td className="text-text-secondary">{collectionName(d.collection_id)}</td>
                  <td className="text-right tabular-nums text-text-primary">
                    {fmtNumber(d.chunk_count)}
                  </td>
                  <td className="text-right tabular-nums text-text-primary">
                    {fmtNumber(d.total_tokens)}
                  </td>
                  <td className="pl-6 text-text-secondary tabular-nums">{fmtDate(d.created_at)}</td>
                  <td className="text-right pr-6">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setToDelete(d);
                      }}
                      className={btnCls("destructive-ghost", "sm")}
                    >
                      DELETE
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <LoadMore
        shown={visible.length}
        total={documents.total}
        hasMore={documents.query.hasNextPage}
        loading={documents.query.isFetchingNextPage}
        onLoad={() => void documents.query.fetchNextPage()}
      />
    </div>
  );
}

function BulkResultBanner({ result, onClose }: { result: BulkOpResult; onClose: () => void }) {
  const clean = result.failed === 0 && result.not_found.length === 0;
  return (
    <div
      role="status"
      className={`mb-4 border px-4 py-2 label-sm ${
        clean
          ? "border-success/40 bg-success/5 text-success"
          : "border-error/40 bg-error/5 text-error"
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <span>
          DELETED {result.succeeded} OF {result.requested}
          {result.failed > 0 && ` · ${result.failed} FAILED`}
          {result.not_found.length > 0 && ` · NOT FOUND: ${result.not_found.join(", ")}`}
        </span>
        <button onClick={onClose} className="text-text-secondary hover:text-text-primary">
          ✕
        </button>
      </div>
      {Object.entries(result.errors).map(([id, err]) => (
        <div key={id} className="body-sm normal-case mt-1">
          #{id}: {err}
        </div>
      ))}
    </div>
  );
}

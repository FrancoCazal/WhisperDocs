import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api, fmtDate, fmtNumber, type IngestionStatus } from "@/lib/api";
import { PageHeader } from "@/components/ui-kit/PageHeader";
import { Btn } from "@/components/ui-kit/Button";
import { EmptyState, ErrorBanner, LoadingLine } from "@/components/ui-kit/EmptyState";

type SearchParams = { collection_id?: number };

export const Route = createFileRoute("/documentos")({
  head: () => ({
    meta: [
      { title: "Documents — WhisperDocs" },
      { name: "description", content: "Indexed corpus and document upload for the WhisperDocs RAG system." },
    ],
  }),
  validateSearch: (s: Record<string, unknown>): SearchParams => ({
    collection_id: s.collection_id ? Number(s.collection_id) : undefined,
  }),
  component: DocumentsPage,
});

type UploadStatus = {
  filename: string;
  status: "queued" | "running" | "success" | "error";
  progress: number;
  stage: IngestionStatus | null;
  error?: string;
};

const STAGE_LABEL: Record<IngestionStatus, string> = {
  pending: "QUEUED",
  extracting: "EXTRACTING",
  chunking: "CHUNKING",
  embedding: "EMBEDDING",
  persisting: "PERSISTING",
  complete: "INDEXED",
  failed: "FAILED",
};

function DocumentsPage() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const search = Route.useSearch();
  const [filter, setFilter] = useState<number | null>(search.collection_id ?? null);
  const [uploadCollection, setUploadCollection] = useState<number | null>(search.collection_id ?? null);
  const [uploadStatuses, setUploadStatuses] = useState<UploadStatus[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const documents = useQuery({
    queryKey: ["documents", filter ?? "all"],
    queryFn: () => api.listDocuments(filter ?? undefined),
  });

  const collectionName = (id: number) =>
    collections.data?.find((c) => c.id === id)?.name ?? `#${id}`;

  function updateStatus(idx: number, patch: Partial<UploadStatus>) {
    setUploadStatuses((prev) => prev.map((s, i) => (i === idx ? { ...s, ...patch } : s)));
  }

  async function ingestFile(file: File, cid: number, idx: number) {
    try {
      const { job_id } = await api.uploadAsync(cid, file);
      updateStatus(idx, { status: "running", stage: "pending" });
      await api.streamIngestionJob(job_id, (e) => {
        if (e.type === "progress") {
          updateStatus(idx, {
            status: "running",
            stage: e.status,
            progress: e.progress,
          });
        } else if (e.type === "complete") {
          updateStatus(idx, {
            status: "success",
            stage: "complete",
            progress: 100,
          });
        } else if (e.type === "error") {
          updateStatus(idx, {
            status: "error",
            stage: "failed",
            error: e.message,
          });
        }
      });
    } catch (err) {
      updateStatus(idx, {
        status: "error",
        stage: "failed",
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  const del = useMutation({
    mutationFn: (id: number) => api.deleteDocument(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["documents"] }),
  });

  const [selected, setSelected] = useState<Set<number>>(new Set());

  function toggleSel(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function bulkDelete() {
    if (selected.size === 0) return;
    if (!confirm(`Delete ${selected.size} document(s)? This cannot be undone.`)) return;
    const ids = Array.from(selected);
    await api.bulkDeleteDocuments(ids);
    setSelected(new Set());
    qc.invalidateQueries({ queryKey: ["documents"] });
  }

  function handleFiles(files: File[]) {
    if (files.length === 0) return;
    const cid = uploadCollection ?? collections.data?.[0]?.id;
    if (!cid) {
      alert("Create a collection first.");
      return;
    }
    const initial: UploadStatus[] = files.map((f) => ({
      filename: f.name,
      status: "queued",
      progress: 0,
      stage: null,
    }));
    setUploadStatuses(initial);
    files.forEach((file, idx) => {
      ingestFile(file, cid, idx).then(() => {
        qc.invalidateQueries({ queryKey: ["documents"] });
      });
    });
  }

  return (
    <div>
      <PageHeader
        title="DOCUMENTS"
        subtitle="INDEXED CORPUS"
        actions={
          <label className="flex items-center gap-2 label-sm text-text-secondary">
            FILTER:
            <select
              value={filter ?? ""}
              onChange={(e) => setFilter(e.target.value ? Number(e.target.value) : null)}
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
            value={uploadCollection ?? ""}
            onChange={(e) => setUploadCollection(e.target.value ? Number(e.target.value) : null)}
            className="bg-surface-lowest border border-border-subtle px-2 h-8 text-text-primary body-sm focus:outline-none focus:border-border-focus"
            aria-label="Upload target collection"
          >
            <option value="">— SELECT —</option>
            {collections.data?.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
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
            accept=".pdf,.md,.txt"
            multiple
            className="hidden"
            onChange={(e) => handleFiles(Array.from(e.target.files ?? []))}
          />
        </div>

        {uploadStatuses.length > 0 && (
          <div className="mt-3 border border-border-subtle bg-surface-container-low">
            {uploadStatuses.map((u, i) => {
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
                    : u.stage
                      ? STAGE_LABEL[u.stage]
                      : "QUEUED";
              return (
                <div key={i} className="px-4 py-2 border-b border-border-subtle/50">
                  <div className="flex items-center justify-between body-sm">
                    <span className="text-text-primary truncate">{u.filename}</span>
                    <span className={`label-sm ${labelColor}`}>
                      {stageLabel} {u.status === "running" ? `· ${u.progress}%` : ""}
                    </span>
                  </div>
                  {u.status === "running" && (
                    <div className="mt-1 h-1 bg-surface-container">
                      <div
                        className="h-full bg-text-primary transition-all duration-200"
                        style={{ width: `${u.progress}%` }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {documents.error && <ErrorBanner message="failed to load documents" />}

      {selected.size > 0 && (
        <div
          role="region"
          aria-label="Bulk actions"
          className="mb-3 flex flex-wrap items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2 bg-surface-container-low border border-border-subtle"
        >
          <span className="label-sm text-text-secondary">
            {selected.size} SELECTED
          </span>
          <span className="flex-1 hidden sm:block" />
          <button
            onClick={() => setSelected(new Set())}
            className="label-sm text-text-secondary hover:text-text-primary px-2 h-7 border border-border-subtle focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-blue"
          >
            CLEAR
          </button>
          <button
            onClick={bulkDelete}
            className="label-sm text-error hover:bg-error hover:text-surface px-3 h-7 border border-error/40 focus:outline-none focus-visible:ring-2 focus-visible:ring-error"
          >
            ✕ DELETE SELECTED
          </button>
        </div>
      )}

      <div className="border border-border-subtle bg-surface-container overflow-x-auto">
        {documents.isLoading ? (
          <div className="p-4"><LoadingLine /></div>
        ) : !documents.data || documents.data.length === 0 ? (
          <EmptyState label="NO DOCUMENTS INDEXED" />
        ) : (
          <table className="w-full min-w-[720px]">
            <thead>
              <tr className="label-sm text-text-secondary border-b border-border-subtle">
                <th className="w-10 pl-4 py-3">
                  <input
                    type="checkbox"
                    checked={selected.size > 0 && selected.size === documents.data.length}
                    onChange={(e) =>
                      setSelected(
                        e.target.checked
                          ? new Set(documents.data!.map((d) => d.id))
                          : new Set(),
                      )
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
              {documents.data.map((d) => (
                <tr
                  key={d.id}
                  onClick={() => navigate({ to: "/documentos/$id", params: { id: String(d.id) } })}
                  className="body-sm h-9 border-b border-border-subtle/50 hover:bg-surface-high cursor-pointer"
                >
                  <td
                    className="w-10 pl-4"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <input
                      type="checkbox"
                      checked={selected.has(d.id)}
                      onChange={() => toggleSel(d.id)}
                      aria-label={`Select ${d.title}`}
                    />
                  </td>
                  <td className="pl-2 text-text-primary">{d.title}</td>
                  <td className="label-sm text-text-secondary">{d.file_type.toUpperCase()}</td>
                  <td className="text-text-secondary">{collectionName(d.collection_id)}</td>
                  <td className="text-right tabular-nums text-text-primary">{fmtNumber(d.chunk_count)}</td>
                  <td className="text-right tabular-nums text-text-primary">{fmtNumber(d.total_tokens)}</td>
                  <td className="pl-6 text-text-secondary tabular-nums">{fmtDate(d.created_at)}</td>
                  <td className="text-right pr-6">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm(`Delete "${d.title}"?`)) del.mutate(d.id);
                      }}
                      className="label-sm text-error hover:bg-error hover:text-surface px-2 h-6"
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
    </div>
  );
}

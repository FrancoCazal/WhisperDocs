import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, fmtDate, fmtNumber, type DocumentRead } from "@/lib/api";
import { PageHeader } from "@/components/ui-kit/PageHeader";
import { Btn } from "@/components/ui-kit/Button";
import { EmptyState, ErrorBanner, LoadingLine } from "@/components/ui-kit/EmptyState";
import { Modal, Field, inputCls, textareaCls } from "@/components/ui-kit/Modal";

export const Route = createFileRoute("/colecciones")({
  head: () => ({
    meta: [
      { title: "Collections — WhisperDocs" },
      { name: "description", content: "Manage knowledge base namespaces in WhisperDocs." },
    ],
  }),
  component: CollectionsPage,
});

function CollectionsPage() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const documents = useQuery({ queryKey: ["documents", "all"], queryFn: () => api.listDocuments() });

  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [formError, setFormError] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: () => api.createCollection({ name, description: desc || undefined }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["collections"] });
      setOpen(false);
      setName("");
      setDesc("");
      setFormError(null);
    },
    onError: (e) => setFormError(e instanceof Error ? e.message : String(e)),
  });

  const del = useMutation({
    mutationFn: (id: number) => api.deleteCollection(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["collections"] }),
  });

  const docsByCollection = (id: number) =>
    (documents.data ?? []).filter((d: DocumentRead) => d.collection_id === id);

  return (
    <div>
      <PageHeader
        title="COLLECTIONS"
        subtitle="KNOWLEDGE BASE NAMESPACES"
        actions={<Btn variant="primary" onClick={() => setOpen(true)}>+ NEW COLLECTION</Btn>}
      />

      {collections.error && <ErrorBanner message="failed to load collections" />}

      <div className="border border-border-subtle bg-surface-container overflow-x-auto">
        {collections.isLoading ? (
          <div className="p-4"><LoadingLine /></div>
        ) : !collections.data || collections.data.length === 0 ? (
          <EmptyState label="NO COLLECTIONS · CREATE ONE TO BEGIN" />
        ) : (
          <table className="w-full min-w-[680px]">
            <thead>
              <tr className="label-sm text-text-secondary border-b border-border-subtle">
                <th className="text-left pl-6 py-3 font-normal">NAME</th>
                <th className="text-left py-3 font-normal">DESCRIPTION</th>
                <th className="text-right py-3 font-normal">DOCUMENTS</th>
                <th className="text-right py-3 font-normal">CHUNKS</th>
                <th className="text-right py-3 font-normal">TOKENS</th>
                <th className="text-left pl-6 py-3 font-normal">CREATED</th>
                <th className="text-right pr-6 py-3 font-normal w-24">ACTIONS</th>
              </tr>
            </thead>
            <tbody>
              {collections.data.map((c) => {
                const docs = docsByCollection(c.id);
                const chunks = docs.reduce((s, d) => s + (d.chunk_count ?? 0), 0);
                const tokens = docs.reduce((s, d) => s + (d.total_tokens ?? 0), 0);
                return (
                  <tr
                    key={c.id}
                    className="body-sm h-9 border-b border-border-subtle/50 hover:bg-surface-high cursor-pointer"
                    onClick={() => navigate({ to: "/documentos", search: { collection_id: c.id } as never })}
                  >
                    <td className="pl-6 text-text-primary">{c.name}</td>
                    <td className="text-text-secondary">{c.description || "—"}</td>
                    <td className="text-right tabular-nums text-text-primary">{fmtNumber(docs.length)}</td>
                    <td className="text-right tabular-nums text-text-primary">{fmtNumber(chunks)}</td>
                    <td className="text-right tabular-nums text-text-secondary">{fmtNumber(tokens)}</td>
                    <td className="pl-6 text-text-secondary tabular-nums">{fmtDate(c.created_at)}</td>
                    <td className="text-right pr-6">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm(`Delete collection "${c.name}"? This cascades to all documents.`)) {
                            del.mutate(c.id);
                          }
                        }}
                        className="label-sm text-error hover:bg-error hover:text-surface px-2 h-6"
                      >
                        DELETE
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <Modal open={open} onClose={() => setOpen(false)} title="NEW COLLECTION">
        <Field label="NAME">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. backend-docs"
            className={inputCls}
            autoFocus
          />
        </Field>
        <Field label="DESCRIPTION (OPTIONAL)">
          <textarea
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
            placeholder="Short description of this namespace…"
            rows={3}
            className={textareaCls}
          />
        </Field>
        {formError && <div className="label-sm text-error mb-3">ERROR: {formError}</div>}
        <div className="flex justify-end gap-2 mt-4">
          <Btn onClick={() => setOpen(false)}>CANCEL</Btn>
          <Btn
            variant="primary"
            disabled={!name.trim() || create.isPending}
            onClick={() => create.mutate()}
          >
            {create.isPending ? "CREATING..." : "CREATE"}
          </Btn>
        </div>
      </Modal>
    </div>
  );
}

import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import { api, fmtDate, fmtNumber, LIMITS, type Collection } from "@/lib/api";
import { ConfirmDialog } from "@/components/ui-kit/ConfirmDialog";
import { PageHeader } from "@/components/ui-kit/PageHeader";
import { Btn, btnCls } from "@/components/ui-kit/Button";
import {
  EmptyState,
  ErrorBanner,
  LoadingLine,
  emptyActionCls,
} from "@/components/ui-kit/EmptyState";
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
  // Each collection carries its totals; the documents list is paged.
  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });

  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [formError, setFormError] = useState<string | null>(null);
  const [toDelete, setToDelete] = useState<Collection | null>(null);

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
    mutationFn: (c: Collection) => api.deleteCollection(c.id),
    // Its documents go with it.
    onSuccess: (_r, c) => {
      void qc.invalidateQueries({ queryKey: ["collections"] });
      void qc.invalidateQueries({ queryKey: ["documents"] });
      toast.success(`COLLECTION "${c.name}" DELETED`, {
        description: `${fmtNumber(c.document_count)} documents removed with it.`,
      });
    },
    onError: (e, c) => toast.error(`COULD NOT DELETE "${c.name}"`, { description: e.message }),
  });

  return (
    <div>
      <PageHeader
        title="COLLECTIONS"
        subtitle="KNOWLEDGE BASE NAMESPACES"
        actions={
          <Btn variant="primary" onClick={() => setOpen(true)}>
            + NEW COLLECTION
          </Btn>
        }
      />

      {collections.error && (
        <ErrorBanner
          message="failed to load collections"
          onRetry={() => void collections.refetch()}
          retrying={collections.isFetching}
        />
      )}
      <ConfirmDialog
        open={toDelete !== null}
        onOpenChange={(open) => !open && setToDelete(null)}
        title={`DELETE COLLECTION "${toDelete?.name ?? ""}"?`}
        confirmLabel="DELETE COLLECTION"
        destructive
        onConfirm={() => toDelete && del.mutate(toDelete)}
      >
        Its {fmtNumber(toDelete?.document_count ?? 0)} documents and{" "}
        {fmtNumber(toDelete?.chunk_count ?? 0)} chunks are deleted with it. This cannot be undone.
      </ConfirmDialog>

      <div className="border border-border-subtle bg-surface-container overflow-x-auto">
        {collections.isLoading ? (
          <div className="p-4">
            <LoadingLine />
          </div>
        ) : !collections.data || collections.data.length === 0 ? (
          <EmptyState label="NO COLLECTIONS · CREATE ONE TO BEGIN">
            <button type="button" onClick={() => setOpen(true)} className={emptyActionCls}>
              + NEW COLLECTION
            </button>
          </EmptyState>
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
                return (
                  <tr
                    key={c.id}
                    className="body-sm h-9 border-b border-border-subtle/50 hover:bg-surface-high cursor-pointer"
                    onClick={() =>
                      navigate({ to: "/documentos", search: { collection_id: c.id } as never })
                    }
                  >
                    <td className="pl-6 text-text-primary">{c.name}</td>
                    <td className="text-text-secondary">{c.description || "—"}</td>
                    <td className="text-right tabular-nums text-text-primary">
                      {fmtNumber(c.document_count)}
                    </td>
                    <td className="text-right tabular-nums text-text-primary">
                      {fmtNumber(c.chunk_count)}
                    </td>
                    <td className="text-right tabular-nums text-text-secondary">
                      {fmtNumber(c.total_tokens)}
                    </td>
                    <td className="pl-6 text-text-secondary tabular-nums">
                      {fmtDate(c.created_at)}
                    </td>
                    <td className="text-right pr-6">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setToDelete(c);
                        }}
                        disabled={del.isPending && del.variables?.id === c.id}
                        className={btnCls("destructive-ghost", "sm")}
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
            maxLength={LIMITS.collectionName}
            className={inputCls}
            autoFocus
          />
        </Field>
        <Field label="DESCRIPTION (OPTIONAL)">
          <textarea
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
            placeholder="Short description of this namespace…"
            maxLength={LIMITS.collectionDescription}
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

import { createFileRoute, Link, useRouter } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, fmtDate, fmtNumber } from "@/lib/api";
import { LoadingLine, ErrorBanner } from "@/components/ui-kit/EmptyState";

export const Route = createFileRoute("/documentos/$id")({
  head: () => ({
    meta: [
      { title: "Document Detail — WhisperDocs" },
      { name: "description", content: "Inspect document metadata and chunks." },
    ],
  }),
  component: DocumentDetailPage,
  errorComponent: ({ error, reset }) => {
    const router = useRouter();
    return (
      <div>
        <ErrorBanner message={error.message} />
        <button
          onClick={() => { router.invalidate(); reset(); }}
          className="label-sm border border-text-primary text-text-primary px-3 h-8 hover:bg-text-primary hover:text-surface"
        >
          RETRY ↻
        </button>
      </div>
    );
  },
  notFoundComponent: () => (
    <div>
      <div className="label-sm text-error">DOCUMENT_NOT_FOUND</div>
      <Link to="/documentos" className="label-sm text-text-primary mt-3 inline-block">
        ← RETURN TO INDEX
      </Link>
    </div>
  ),
});

function DocumentDetailPage() {
  const { id } = Route.useParams();
  const docId = Number(id);
  const doc = useQuery({ queryKey: ["document", docId], queryFn: () => api.getDocument(docId) });
  const chunks = useQuery({ queryKey: ["chunks", docId], queryFn: () => api.getChunks(docId) });
  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });

  return (
    <div>
      <div className="label-sm text-text-tertiary mb-2">
        <Link to="/documentos" className="hover:text-primary-blue">
          DOCUMENTS
        </Link>{" "}
        / <span className="text-text-secondary">{doc.data?.title ?? id}</span>
      </div>

      <h1 className="title-lg text-text-primary mb-6 pb-4 border-b border-border-subtle">
        {doc.data?.title ?? "LOADING..."}
      </h1>

      {doc.error && <ErrorBanner message="failed to load document" />}

      {doc.isLoading ? (
        <LoadingLine />
      ) : doc.data ? (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-4 mb-8">
            <Meta label="FILENAME" value={doc.data.filename} />
            <Meta label="TYPE" value={doc.data.file_type.toUpperCase()} />
            <Meta
              label="COLLECTION"
              value={collections.data?.find((c) => c.id === doc.data!.collection_id)?.name ?? `#${doc.data.collection_id}`}
            />
            <Meta label="CHUNKS" value={fmtNumber(doc.data.chunk_count)} />
            <Meta label="TOTAL TOKENS" value={fmtNumber(doc.data.total_tokens)} />
            <Meta label="CREATED" value={fmtDate(doc.data.created_at)} />
          </div>

          <h2 className="title-sm text-text-primary mb-4">CHUNKS</h2>
          {chunks.isLoading ? (
            <LoadingLine />
          ) : chunks.data && chunks.data.length > 0 ? (
            <div className="space-y-3">
              {chunks.data.map((c) => (
                <ChunkBlock key={c.id} index={c.chunk_index} tokens={c.token_count} text={c.text} />
              ))}
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

function ChunkBlock({ index, tokens, text }: { index: number; tokens: number; text: string }) {
  const [expanded, setExpanded] = useState(false);
  const lines = text.split("\n");
  const isLong = lines.length > 4 || text.length > 400;
  const display = expanded || !isLong ? text : lines.slice(0, 4).join("\n");

  return (
    <div className="border-b border-border-subtle py-3">
      <div className="label-sm text-text-secondary mb-2 flex items-center gap-3">
        <span>[CHUNK_INDEX: {index}]</span>
        <span className="text-text-tertiary">·</span>
        <span>TOKENS: {tokens}</span>
      </div>
      <pre className="body-sm text-text-primary whitespace-pre-wrap break-words bg-surface-lowest p-3 border border-border-subtle">
        {display}
        {isLong && !expanded && "..."}
      </pre>
      {isLong && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="label-sm text-text-secondary hover:text-text-primary mt-2"
        >
          {expanded ? "▴ COLLAPSE" : "▾ EXPAND"}
        </button>
      )}
    </div>
  );
}

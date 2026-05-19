import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, fmtCost, fmtDate, fmtNumber, truncate } from "@/lib/api";
import { PageHeader } from "@/components/ui-kit/PageHeader";
import { EmptyState, ErrorBanner, LoadingLine } from "@/components/ui-kit/EmptyState";

export const Route = createFileRoute("/historial")({
  head: () => ({
    meta: [
      { title: "History — WhisperDocs" },
      { name: "description", content: "Query log: every past question, answer, and pipeline metric." },
    ],
  }),
  component: HistorialPage,
});

function HistorialPage() {
  const [collFilter, setCollFilter] = useState<number | null>(null);
  const [limit, setLimit] = useState(50);
  const [expanded, setExpanded] = useState<number | null>(null);

  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const history = useQuery({ queryKey: ["history", limit], queryFn: () => api.history(limit) });

  const rows = (history.data ?? []).filter(
    (q) => collFilter === null || q.collection_id === collFilter,
  );

  const collectionName = (id: number | null) =>
    id === null ? "—" : collections.data?.find((c) => c.id === id)?.name ?? `#${id}`;

  const tokensCost = (tin: number, tout: number) =>
    `$${(((tin + tout) / 1_000_000) * 0.5).toFixed(4)}`;

  return (
    <div>
      <PageHeader
        title="HISTORY"
        subtitle="QUERY LOG"
        actions={
          <>
            <label className="flex items-center gap-2 label-sm text-text-secondary">
              COLLECTION:
              <select
                value={collFilter ?? ""}
                onChange={(e) => setCollFilter(e.target.value ? Number(e.target.value) : null)}
                className="bg-surface-lowest border border-border-subtle px-2 h-9 text-text-primary body-sm focus:outline-none focus:border-border-focus"
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
              LIMIT:
              <select
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value))}
                className="bg-surface-lowest border border-border-subtle px-2 h-9 text-text-primary body-sm focus:outline-none focus:border-border-focus"
              >
                {[20, 50, 100, 200].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </label>
          </>
        }
      />

      {history.error && <ErrorBanner message="failed to load query history" />}

      <div className="border border-border-subtle bg-surface-container overflow-x-auto">
        {history.isLoading ? (
          <div className="p-4"><LoadingLine /></div>
        ) : rows.length === 0 ? (
          <EmptyState label="NO QUERIES LOGGED" />
        ) : (
          <table className="w-full min-w-[760px]">
            <thead>
              <tr className="label-sm text-text-secondary border-b border-border-subtle">
                <th className="text-left pl-6 py-3 font-normal">TIMESTAMP</th>
                <th className="text-left py-3 font-normal">QUERY</th>
                <th className="text-left py-3 font-normal">MODEL</th>
                <th className="text-right py-3 font-normal">TOKENS IN/OUT</th>
                <th className="text-right py-3 font-normal">LATENCY</th>
                <th className="text-right pr-6 py-3 font-normal">COST</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((q) => {
                const open = expanded === q.id;
                return (
                  <>
                    <tr
                      key={q.id}
                      onClick={() => setExpanded(open ? null : q.id)}
                      className="body-sm h-9 border-b border-border-subtle/50 hover:bg-surface-high cursor-pointer"
                    >
                      <td className="pl-6 text-text-secondary tabular-nums">{fmtDate(q.created_at)}</td>
                      <td className="text-text-primary">{truncate(q.question, 70)}</td>
                      <td className="text-text-secondary">{q.model_used}</td>
                      <td className="text-right tabular-nums text-text-primary">
                        {fmtNumber(q.tokens_input)} / {fmtNumber(q.tokens_output)}
                      </td>
                      <td className="text-right tabular-nums text-text-primary">{q.latency_ms}MS</td>
                      <td className="text-right pr-6 tabular-nums text-text-primary">
                        {tokensCost(q.tokens_input, q.tokens_output)}
                      </td>
                    </tr>
                    {open && (
                      <tr key={`${q.id}-detail`} className="bg-surface-low">
                        <td colSpan={6} className="p-6 border-b border-border-subtle">
                          <div className="grid grid-cols-2 gap-6">
                            <div>
                              <div className="label-sm text-text-secondary mb-2">QUERY</div>
                              <div className="body-sm text-text-primary whitespace-pre-wrap">{q.question}</div>
                              <div className="label-sm text-text-secondary mt-4 mb-2">COLLECTION</div>
                              <div className="body-sm text-text-primary">{collectionName(q.collection_id)}</div>
                              <div className="label-sm text-text-secondary mt-4 mb-2">CHUNKS RETRIEVED</div>
                              <div className="body-sm text-text-primary tabular-nums">{q.chunks_retrieved}</div>
                            </div>
                            <div>
                              <div className="label-sm text-text-secondary mb-2">RESPONSE</div>
                              <pre className="body-sm text-text-primary whitespace-pre-wrap bg-surface-lowest p-3 border border-border-subtle max-h-80 overflow-auto">
                                {q.answer}
                              </pre>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

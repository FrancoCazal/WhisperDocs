import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { api, estimateCost, fmtCost, fmtDate, fmtNumber, fmtPct, truncate } from "@/lib/api";
import { PageHeader } from "@/components/ui-kit/PageHeader";
import { Btn } from "@/components/ui-kit/Button";
import { StatCard } from "@/components/ui-kit/StatCard";
import { StatDot } from "@/components/ui-kit/StatDot";
import { EmptyState, ErrorBanner, LoadingLine } from "@/components/ui-kit/EmptyState";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "System Overview — WhisperDocs" },
      { name: "description", content: "Diagnostics & telemetry dashboard for the WhisperDocs RAG pipeline." },
    ],
  }),
  component: Dashboard,
});

function latencyClass(ms: number) {
  if (ms < 2000) return "text-success";
  if (ms < 5000) return "text-warning";
  return "text-error";
}

function accuracyClass(acc: number) {
  if (acc >= 0.8) return "text-success";
  if (acc >= 0.6) return "text-warning";
  return "text-error";
}

function TokenBar({ label, value, max, colorClass }: { label: string; value: number; max: number; colorClass: string }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div>
      <div className="flex justify-between items-end mb-1">
        <span className="label-xs text-text-secondary">{label}</span>
        <span className={`label-xs ${colorClass}`}>{value >= 1000 ? `${(value / 1000).toFixed(1)}K` : value}</span>
      </div>
      <div className="w-full bg-surface-high h-[2px]">
        <div className={`h-full ${colorClass.replace("text-", "bg-")}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function Dashboard() {
  const navigate = useNavigate();
  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const documents = useQuery({ queryKey: ["documents", "all"], queryFn: () => api.listDocuments() });
  const history = useQuery({ queryKey: ["history", 10], queryFn: () => api.history(10) });
  const evals = useQuery({ queryKey: ["eval-results", 5], queryFn: () => api.evalResults(5) });
  const conversations = useQuery({
    queryKey: ["conversations", "list"],
    queryFn: () => api.listConversations({ limit: 200 }),
  });

  const totalCollections = collections.data?.length ?? 0;
  const totalDocuments = documents.data?.length ?? 0;
  const totalChunks = (documents.data ?? []).reduce((s, d) => s + (d.chunk_count ?? 0), 0);

  const recentLogs = history.data ?? [];

  const queryCostEst = recentLogs.reduce(
    (s, q) => s + parseFloat(estimateCost(q.tokens_input, q.tokens_output, q.model_used).replace("$", "") || "0"),
    0,
  );

  const totalInputTokens = recentLogs.reduce((s, q) => s + q.tokens_input, 0);
  const totalOutputTokens = recentLogs.reduce((s, q) => s + q.tokens_output, 0);
  const maxTokens = Math.max(totalInputTokens, totalOutputTokens, 1);

  const sortedLatencies = [...recentLogs].sort((a, b) => a.latency_ms - b.latency_ms);
  const p95Latency =
    sortedLatencies.length > 0
      ? sortedLatencies[Math.max(0, Math.ceil(sortedLatencies.length * 0.95) - 1)].latency_ms
      : null;

  const primaryModel = recentLogs[0]?.model_used ?? null;
  const lastAccuracy = evals.data?.[0]?.accuracy ?? null;

  const anyError = collections.error || documents.error || history.error || evals.error;

  const collectionRows = (collections.data ?? []).map((c) => {
    const docs = (documents.data ?? []).filter((d) => d.collection_id === c.id);
    const chunks = docs.reduce((s, d) => s + (d.chunk_count ?? 0), 0);
    return { ...c, docCount: docs.length, chunkCount: chunks };
  });

  return (
    <div>
      <PageHeader
        title="SYSTEM OVERVIEW"
        subtitle="DIAGNOSTICS & TELEMETRY DASHBOARD"
        actions={
          <>
            <Btn onClick={() => navigate({ to: "/chat" })}>NEW QUERY</Btn>
            <Btn onClick={() => navigate({ to: "/documentos" })}>UPLOAD</Btn>
            <Btn onClick={() => navigate({ to: "/evaluacion" })}>RUN EVAL</Btn>
          </>
        }
      />

      {anyError && (
        <ErrorBanner
          message={`backend unreachable at ${(import.meta.env.VITE_API_BASE_URL as string) ?? "http://localhost:8000"}`}
        />
      )}

      {/* ASYMMETRIC GRID: stacks on mobile, 8/4 on lg */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-8">

        {/* LEFT COLUMN */}
        <div className="lg:col-span-8 flex flex-col gap-6 min-w-0">

          {/* TOP STAT ROW */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-4">
            <StatCard
              label="COLLECTIONS"
              value={collections.isLoading ? "—" : fmtNumber(totalCollections)}
            />
            <StatCard
              label="DOCUMENTS"
              value={documents.isLoading ? "—" : fmtNumber(totalDocuments)}
            />
            <StatCard
              label="CHUNKS INDEXED"
              value={documents.isLoading ? "—" : fmtNumber(totalChunks)}
            />
          </div>

          {/* RECENT QUERIES TABLE */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="title-sm text-text-primary">RECENT INFERENCE STREAMS</h2>
              <Link to="/historial" className="label-sm text-text-secondary hover:text-primary-blue">
                VIEW ALL →
              </Link>
            </div>
            <div className="border border-border-subtle bg-surface-container overflow-x-auto">
              {history.isLoading ? (
                <div className="p-4"><LoadingLine /></div>
              ) : recentLogs.length === 0 ? (
                <EmptyState label="NO QUERIES LOGGED" />
              ) : (
                <table className="w-full min-w-[600px]">
                  <thead>
                    <tr className="label-sm text-text-secondary border-b border-border-subtle bg-surface-high">
                      <th className="text-left pl-5 py-3 font-normal">TIMESTAMP</th>
                      <th className="text-left py-3 font-normal">QUERY</th>
                      <th className="text-right py-3 font-normal">LATENCY</th>
                      <th className="text-right py-3 font-normal">TOKENS</th>
                      <th className="text-center pr-5 py-3 font-normal w-14">OK</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentLogs.map((q) => (
                      <tr key={q.id} className="body-sm hover:bg-surface-high border-b border-border-subtle/50">
                        <td className="pl-5 py-3 text-text-secondary tabular-nums whitespace-nowrap">{fmtDate(q.created_at)}</td>
                        <td className="py-3 text-text-primary pr-4">{truncate(q.question, 55)}</td>
                        <td className={`py-3 text-right tabular-nums ${latencyClass(q.latency_ms)}`}>
                          {fmtNumber(q.latency_ms)}MS
                        </td>
                        <td className="py-3 text-right tabular-nums text-text-secondary">
                          {fmtNumber(q.tokens_input + q.tokens_output)}
                        </td>
                        <td className="py-3 text-center pr-5"><StatDot ok={q.latency_ms < 5000} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </section>
        </div>

        {/* RIGHT COLUMN */}
        <div className="lg:col-span-4 flex flex-col gap-4 min-w-0">

          {/* COST CARD */}
          <div className="border border-border-subtle bg-surface-container p-5">
            <div className="label-xs text-text-tertiary mb-1">EST. QUERY COST (LAST 10)</div>
            <div className="display-md text-error mb-4">
              {history.isLoading ? "—" : fmtCost(queryCostEst)}
            </div>
            <div className="grid grid-cols-2 gap-3 pt-3 border-t border-border-subtle">
              <div>
                <div className="label-xs text-text-tertiary mb-1">CONVERSATIONS</div>
                <div className="title-sm text-text-primary">
                  {conversations.isLoading ? "—" : fmtNumber(conversations.data?.length ?? 0)}
                </div>
              </div>
              <div>
                <div className="label-xs text-text-tertiary mb-1">LAST EVAL ACC.</div>
                <div className={`title-sm ${lastAccuracy !== null ? accuracyClass(lastAccuracy) : "text-text-primary"}`}>
                  {evals.isLoading ? "—" : lastAccuracy !== null ? fmtPct(lastAccuracy) : "—"}
                </div>
              </div>
            </div>
          </div>

          {/* ENGINE METADATA */}
          <div className="border border-border-subtle bg-surface-container flex-1 flex flex-col">
            <div className="px-5 py-4 border-b border-border-subtle">
              <h3 className="label-sm text-text-primary">ENGINE METADATA</h3>
            </div>
            <div className="p-5 space-y-6 flex-1 bg-surface-low">

              {/* TOKEN CONSUMPTION */}
              <div>
                <div className="label-xs text-text-tertiary mb-3">TOKEN CONSUMPTION (LAST 10)</div>
                {history.isLoading ? (
                  <LoadingLine />
                ) : recentLogs.length === 0 ? (
                  <div className="label-xs text-text-tertiary">NO DATA</div>
                ) : (
                  <div className="space-y-3">
                    <TokenBar label="INPUT" value={totalInputTokens} max={maxTokens} colorClass="text-primary-blue" />
                    <TokenBar label="OUTPUT" value={totalOutputTokens} max={maxTokens} colorClass="text-text-secondary" />
                  </div>
                )}
              </div>

              {/* P95 LATENCY */}
              <div className="pt-4 border-t border-border-subtle">
                <div className="label-xs text-text-tertiary mb-2">P95 LATENCY</div>
                {history.isLoading ? (
                  <LoadingLine />
                ) : p95Latency !== null ? (
                  <>
                    <div className="flex items-baseline gap-2">
                      <span className={`display-md ${latencyClass(p95Latency)}`}>{fmtNumber(p95Latency)}</span>
                      <span className="label-xs text-text-tertiary">MS</span>
                    </div>
                    <div className="label-xs text-text-tertiary mt-1">WARN THRESHOLD: &gt;5000MS</div>
                  </>
                ) : (
                  <div className="display-md text-text-primary">—</div>
                )}
              </div>

              {/* PRIMARY LLM ROUTE */}
              <div className="pt-4 border-t border-border-subtle">
                <div className="label-xs text-text-tertiary mb-2">PRIMARY LLM ROUTE</div>
                {history.isLoading ? (
                  <LoadingLine />
                ) : primaryModel ? (
                  <div className="inline-flex items-center gap-2 border border-border-subtle bg-surface-container px-3 py-1.5">
                    <span className="w-1.5 h-1.5 bg-primary-blue" />
                    <span className="label-xs text-text-primary">{primaryModel}</span>
                  </div>
                ) : (
                  <div className="label-xs text-text-tertiary">NO DATA</div>
                )}
              </div>

            </div>
          </div>

        </div>
      </div>

      {/* BOTTOM ROW: EVAL + COLLECTIONS */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* RECENT EVAL RUNS */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="title-sm text-text-primary">RECENT EVALUATION RUNS</h2>
            <Link to="/evaluacion" className="label-sm text-text-secondary hover:text-primary-blue">
              VIEW ALL →
            </Link>
          </div>
          <div className="border border-border-subtle bg-surface-container overflow-x-auto">
            {evals.isLoading ? (
              <div className="p-4"><LoadingLine /></div>
            ) : !evals.data || evals.data.length === 0 ? (
              <EmptyState label="NO EVALUATION RUNS" />
            ) : (
              <table className="w-full min-w-[480px]">
                <thead>
                  <tr className="label-sm text-text-secondary border-b border-border-subtle bg-surface-high">
                    <th className="text-left pl-5 py-3 font-normal">RUN_ID</th>
                    <th className="text-left py-3 font-normal">TIMESTAMP</th>
                    <th className="text-right py-3 font-normal">ACCURACY</th>
                    <th className="text-right pr-5 py-3 font-normal">COST</th>
                  </tr>
                </thead>
                <tbody>
                  {evals.data.map((r) => (
                    <tr key={r.run_id} className="body-sm hover:bg-surface-high border-b border-border-subtle/50">
                      <td className="pl-5 py-3 text-text-primary">{r.run_id}</td>
                      <td className="py-3 text-text-secondary tabular-nums">{fmtDate(r.timestamp)}</td>
                      <td className={`py-3 text-right tabular-nums ${accuracyClass(r.accuracy)}`}>
                        {fmtPct(r.accuracy)}
                      </td>
                      <td className="py-3 text-right pr-5 tabular-nums text-error">
                        {fmtCost(r.estimated_cost_usd)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>

        {/* COLLECTIONS */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="title-sm text-text-primary">COLLECTIONS</h2>
            <Link to="/colecciones" className="label-sm text-text-secondary hover:text-primary-blue">
              MANAGE →
            </Link>
          </div>
          <div className="border border-border-subtle bg-surface-container overflow-x-auto">
            {collections.isLoading || documents.isLoading ? (
              <div className="p-4"><LoadingLine /></div>
            ) : collectionRows.length === 0 ? (
              <EmptyState label="NO COLLECTIONS" />
            ) : (
              <table className="w-full min-w-[560px]">
                <thead>
                  <tr className="label-sm text-text-secondary border-b border-border-subtle bg-surface-high">
                    <th className="text-left pl-5 py-3 font-normal">NAME</th>
                    <th className="text-right py-3 font-normal">DOCS</th>
                    <th className="text-right py-3 font-normal">CHUNKS</th>
                    <th className="text-left py-3 pl-5 pr-5 font-normal">DESCRIPTION</th>
                  </tr>
                </thead>
                <tbody>
                  {collectionRows.map((c) => (
                    <tr key={c.id} className="body-sm hover:bg-surface-high border-b border-border-subtle/50">
                      <td className="pl-5 py-3 text-text-primary font-medium">{c.name.toUpperCase()}</td>
                      <td className="py-3 text-right tabular-nums text-text-primary">{fmtNumber(c.docCount)}</td>
                      <td className="py-3 text-right tabular-nums text-text-secondary">{fmtNumber(c.chunkCount)}</td>
                      <td className="py-3 pl-5 pr-5 text-text-tertiary">{truncate(c.description ?? "—", 50)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>

      </div>
    </div>
  );
}

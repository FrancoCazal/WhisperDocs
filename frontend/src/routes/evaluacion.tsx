import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, fmtCost, fmtDate, fmtPct, type EvalReport } from "@/lib/api";
import { PageHeader } from "@/components/ui-kit/PageHeader";
import { Btn } from "@/components/ui-kit/Button";
import { EmptyState, ErrorBanner, LoadingLine } from "@/components/ui-kit/EmptyState";
import { Modal, Field, textareaCls, inputCls } from "@/components/ui-kit/Modal";
import { StatDot } from "@/components/ui-kit/StatDot";

export const Route = createFileRoute("/evaluacion")({
  head: () => ({
    meta: [
      { title: "Evaluation — WhisperDocs" },
      { name: "description", content: "Run and inspect benchmark evaluations of the RAG pipeline." },
    ],
  }),
  component: EvaluacionPage,
});

const STARTER_DATASET = JSON.stringify(
  [
    { question: "What distance metrics does pgvector support?", expected: "cosine, l2, inner_product", difficulty: "easy", category: "in_scope" },
    { question: "How do you create an IVFFlat index in pgvector?", expected: "CREATE INDEX ... USING ivfflat", difficulty: "medium", category: "in_scope" },
    { question: "Difference between selectinload and joinedload", expected: "selectinload uses IN; joinedload uses JOIN", difficulty: "medium", category: "in_scope" },
    { question: "What is the capital of France?", expected: "out of scope", difficulty: "easy", category: "out_of_scope" },
  ],
  null,
  2,
);

function EvaluacionPage() {
  const qc = useQueryClient();
  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const runs = useQuery({ queryKey: ["eval-results", 20], queryFn: () => api.evalResults(20) });

  const [currentRun, setCurrentRun] = useState<EvalReport | null>(null);
  const [open, setOpen] = useState(false);
  const [dataset, setDataset] = useState(STARTER_DATASET);
  const [collId, setCollId] = useState<number | null>(null);
  const [topK, setTopK] = useState(5);
  const [conc, setConc] = useState(4);
  const [formError, setFormError] = useState<string | null>(null);

  const runEval = useMutation({
    mutationFn: () => {
      let parsed: unknown[];
      try {
        parsed = JSON.parse(dataset);
        if (!Array.isArray(parsed)) throw new Error("Dataset must be a JSON array");
      } catch (e) {
        throw new Error(`Invalid dataset JSON: ${e instanceof Error ? e.message : e}`);
      }
      if (collId === null) throw new Error("Select a collection");
      return api.runEval({ dataset: parsed, collection_id: collId, top_k: topK, concurrency: conc });
    },
    onSuccess: (r) => {
      setCurrentRun(r);
      setOpen(false);
      qc.invalidateQueries({ queryKey: ["eval-results"] });
    },
    onError: (e) => setFormError(e instanceof Error ? e.message : String(e)),
  });

  return (
    <div>
      <PageHeader
        title="EVALUATION"
        subtitle="BENCHMARK & TELEMETRY"
        actions={
          <Btn variant="primary" onClick={() => { setFormError(null); setOpen(true); }}>
            + RUN NEW EVALUATION
          </Btn>
        }
      />

      {runs.error && <ErrorBanner message="failed to load evaluation runs" />}

      {/* TOP: current run display */}
      {runEval.isPending ? (
        <div className="border border-border-subtle bg-surface-container p-8 mb-8">
          <div className="label-sm text-text-secondary blink-cursor">EXECUTING EVALUATION</div>
          <div className="space-y-1 mt-4 label-sm text-text-tertiary">
            <div>LOADING DATASET...</div>
            <div>EMBEDDING QUERIES...</div>
            <div>SCORING RESPONSES...</div>
          </div>
        </div>
      ) : currentRun ? (
        <CurrentRun report={currentRun} />
      ) : (
        <div className="border border-dashed border-border-subtle py-12 text-center mb-8">
          <div className="label-sm text-text-tertiary">
            NO ACTIVE RUN · SELECT A HISTORICAL RUN OR EXECUTE A NEW EVALUATION
          </div>
        </div>
      )}

      {/* BOTTOM: historical runs */}
      <h2 className="title-sm text-text-primary mb-3">HISTORICAL RUNS</h2>
      <div className="border border-border-subtle bg-surface-container overflow-x-auto">
        {runs.isLoading ? (
          <div className="p-4"><LoadingLine /></div>
        ) : !runs.data || runs.data.length === 0 ? (
          <EmptyState label="NO HISTORICAL RUNS" />
        ) : (
          <table className="w-full min-w-[640px]">
            <thead>
              <tr className="label-sm text-text-secondary border-b border-border-subtle">
                <th className="text-left pl-6 py-3 font-normal">RUN_ID</th>
                <th className="text-left py-3 font-normal">DATE</th>
                <th className="text-right py-3 font-normal">ACCURACY</th>
                <th className="text-right py-3 font-normal">HALLUCINATION</th>
                <th className="text-center py-3 font-normal w-20">STATUS</th>
                <th className="text-right py-3 font-normal">COST</th>
                <th className="text-right pr-6 py-3 font-normal w-20">ACTIONS</th>
              </tr>
            </thead>
            <tbody>
              {runs.data.map((r) => (
                <tr key={r.run_id} className="body-sm h-9 border-b border-border-subtle/50 hover:bg-surface-high">
                  <td className="pl-6 text-text-primary">{r.run_id}</td>
                  <td className="text-text-secondary tabular-nums">{fmtDate(r.timestamp)}</td>
                  <td
                    className="text-right tabular-nums"
                    style={{ color: accuracyColor(r.accuracy) }}
                  >
                    {fmtPct(r.accuracy)}
                  </td>
                  <td
                    className="text-right tabular-nums"
                    style={{ color: halColor(r.hallucination_rate) }}
                  >
                    {fmtPct(r.hallucination_rate)}
                  </td>
                  <td className="text-center"><StatDot ok={(r.status ?? "complete") === "complete"} /></td>
                  <td className="text-right tabular-nums text-text-primary">{fmtCost(r.estimated_cost_usd)}</td>
                  <td className="text-right pr-6">
                    <button
                      onClick={() =>
                        setCurrentRun({
                          run_id: r.run_id,
                          timestamp: r.timestamp,
                          dataset_size: r.dataset_size,
                          model: r.model,
                          metrics: {
                            total: r.dataset_size,
                            accuracy: r.accuracy,
                            hallucination_rate: r.hallucination_rate,
                            avg_retrieval_precision: 0,
                            avg_retrieval_recall: 0,
                            avg_latency_ms: 0,
                            latency_p50_ms: 0,
                            latency_p95_ms: 0,
                            avg_tokens_input: 0,
                            avg_tokens_output: 0,
                            total_tokens_input: 0,
                            total_tokens_output: 0,
                            estimated_cost_usd: r.estimated_cost_usd,
                          },
                          results: [],
                        })
                      }
                      className="label-sm text-text-primary hover:bg-text-primary hover:text-surface px-2 h-6"
                    >
                      VIEW
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <Modal open={open} onClose={() => setOpen(false)} title="EXECUTE EVALUATION" width="max-w-2xl">
        <Field label="DATASET (JSON ARRAY)">
          <div className="flex justify-end mb-1">
            <button
              onClick={() => setDataset(STARTER_DATASET)}
              className="label-xs text-text-secondary hover:text-primary-blue"
            >
              ↺ LOAD STARTER DATASET
            </button>
          </div>
          <textarea
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            rows={10}
            className={textareaCls}
          />
        </Field>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Field label="COLLECTION">
            <select
              value={collId ?? ""}
              onChange={(e) => setCollId(e.target.value ? Number(e.target.value) : null)}
              className={inputCls}
            >
              <option value="">— SELECT —</option>
              {collections.data?.map((c) => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
          </Field>
          <Field label="TOP_K">
            <input
              type="number" min={1} max={20} value={topK}
              onChange={(e) => setTopK(Number(e.target.value) || 5)}
              className={inputCls}
            />
          </Field>
          <Field label="CONCURRENCY">
            <input
              type="number" min={1} max={16} value={conc}
              onChange={(e) => setConc(Number(e.target.value) || 4)}
              className={inputCls}
            />
          </Field>
        </div>
        {formError && <div className="label-sm text-error mb-3">ERROR: {formError}</div>}
        <div className="flex justify-end gap-2 mt-4">
          <Btn onClick={() => setOpen(false)}>CANCEL</Btn>
          <Btn variant="primary" disabled={runEval.isPending} onClick={() => runEval.mutate()}>
            {runEval.isPending ? "EXECUTING..." : "EXECUTE"}
          </Btn>
        </div>
      </Modal>
    </div>
  );
}

function accuracyColor(a: number) {
  if (a >= 0.8) return "var(--success)";
  if (a < 0.6) return "var(--error)";
  return "var(--text-primary)";
}
function halColor(h: number) {
  if (h < 0.05) return "var(--success)";
  if (h > 0.15) return "var(--error)";
  return "var(--text-primary)";
}
function latColor(ms: number) {
  if (ms < 2000) return "var(--success)";
  if (ms > 3000) return "var(--error)";
  return "var(--text-primary)";
}

function CurrentRun({ report }: { report: EvalReport }) {
  const m = report.metrics;
  return (
    <div className="mb-8">
      <div className="flex items-baseline justify-between mb-2">
        <div className="title-lg text-text-primary">{report.run_id}</div>
        <div className="label-sm text-text-secondary tabular-nums">{fmtDate(report.timestamp)}</div>
      </div>
      <div className="label-sm text-text-secondary mb-6">
        MODEL: <span className="text-text-primary">{report.model}</span>
        <span className="text-text-tertiary"> · </span>
        DATASET_SIZE: <span className="text-text-primary tabular-nums">{report.dataset_size}</span>
        <span className="text-text-tertiary"> · </span>
        STATUS: <span className="text-success">● COMPLETE</span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <BigMetric label="ACCURACY" value={fmtPct(m.accuracy)} color={accuracyColor(m.accuracy)} />
        <BigMetric label="HALLUCINATION RATE" value={fmtPct(m.hallucination_rate)} color={halColor(m.hallucination_rate)} />
        <BigMetric label="RETRIEVAL PRECISION" value={fmtPct(m.avg_retrieval_precision)} />
        <BigMetric label="AVG LATENCY" value={`${Math.round(m.avg_latency_ms)}MS`} color={latColor(m.avg_latency_ms)} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <BreakdownTable title="BY DIFFICULTY" data={m.by_difficulty} />
        <BreakdownTable title="BY CATEGORY" data={m.by_category} />
      </div>
    </div>
  );
}

function BigMetric({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="border border-border-subtle bg-surface-container p-5">
      <div className="label-sm text-text-secondary">{label}</div>
      <div className="display-lg mt-3" style={{ color: color ?? "var(--text-primary)" }}>
        {value}
      </div>
    </div>
  );
}

function BreakdownTable({
  title,
  data,
}: {
  title: string;
  data?: Record<string, { count: number; accuracy: number; hallucination_rate: number }>;
}) {
  return (
    <div>
      <div className="title-sm text-text-primary mb-3">{title}</div>
      <div className="border border-border-subtle bg-surface-container overflow-x-auto">
        {!data || Object.keys(data).length === 0 ? (
          <EmptyState label="NO BREAKDOWN" />
        ) : (
          <table className="w-full min-w-[420px]">
            <thead>
              <tr className="label-sm text-text-secondary border-b border-border-subtle">
                <th className="text-left pl-6 py-2 font-normal">LEVEL</th>
                <th className="text-right py-2 font-normal">COUNT</th>
                <th className="text-right py-2 font-normal">ACCURACY</th>
                <th className="text-right pr-6 py-2 font-normal">HAL</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data).map(([k, v]) => (
                <tr key={k} className="body-sm h-9 hover:bg-surface-high border-b border-border-subtle/50">
                  <td className="pl-6 text-text-primary">{k.toUpperCase()}</td>
                  <td className="text-right tabular-nums text-text-primary">{v.count}</td>
                  <td className="text-right tabular-nums" style={{ color: accuracyColor(v.accuracy) }}>
                    {fmtPct(v.accuracy)}
                  </td>
                  <td className="text-right pr-6 tabular-nums" style={{ color: halColor(v.hallucination_rate) }}>
                    {fmtPct(v.hallucination_rate)}
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

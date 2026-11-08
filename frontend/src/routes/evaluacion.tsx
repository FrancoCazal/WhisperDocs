import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useRef, useState } from "react";
import { api, fmtCost, fmtDate, fmtPct, type EvalRunSummary } from "@/lib/api";
import { parseEvalDataset, STARTER_DATASET } from "@/lib/evalDataset";
import { TOP_K_MAX, TOP_K_MIN, useTopK } from "@/lib/topK";
import { PageHeader } from "@/components/ui-kit/PageHeader";
import { Btn, btnCls } from "@/components/ui-kit/Button";
import {
  EmptyState,
  ErrorBanner,
  LoadingLine,
  emptyActionCls,
} from "@/components/ui-kit/EmptyState";
import { Modal, Field, textareaCls, inputCls } from "@/components/ui-kit/Modal";
import { Elapsed } from "@/components/chat/Trace";
import { RunReport } from "@/components/eval/RunReport";
import { GENERATION_COST_HINT } from "@/components/eval/costHint";
import { toneText } from "@/lib/metricTone";

export const Route = createFileRoute("/evaluacion")({
  head: () => ({
    meta: [
      { title: "Evaluation — WhisperDocs" },
      {
        name: "description",
        content: "Run and inspect benchmark evaluations of the RAG pipeline.",
      },
    ],
  }),
  component: EvaluacionPage,
});

const STARTER_TEXT = JSON.stringify(STARTER_DATASET, null, 2);

// What generating `items` answers should cost, from the latest run that
// recorded a cost. The judge and the embeddings are on top and unmeasured.
function estimateGeneration(runs: EvalRunSummary[] | undefined, items: number) {
  const last = runs?.find((r) => r.dataset_size > 0 && r.estimated_cost_usd > 0);
  if (!last) return null;
  return { usd: (last.estimated_cost_usd / last.dataset_size) * items, from: last };
}

function EvaluacionPage() {
  const qc = useQueryClient();
  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const runs = useQuery({ queryKey: ["eval-results", 20], queryFn: () => api.evalResults(20) });

  // The run on screen: the one picked, else the latest.
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const shownId = selectedId ?? runs.data?.[0]?.run_id ?? null;
  const report = useQuery({
    queryKey: ["eval-run", shownId],
    queryFn: () => api.getEvalRun(shownId!),
    enabled: shownId !== null,
    // A saved report never changes.
    staleTime: Infinity,
  });
  const reportRef = useRef<HTMLDivElement>(null);

  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [running, setRunning] = useState<{ since: number; items: number } | null>(null);
  const [dataset, setDataset] = useState(STARTER_TEXT);
  const [collId, setCollId] = useState<number | null>(null);
  const { topK, defaultTopK, setTopK } = useTopK();
  const [conc, setConc] = useState(4);
  const [formError, setFormError] = useState<string | null>(null);

  const parsed = useMemo(() => parseEvalDataset(dataset), [dataset]);
  const items = parsed.items?.length ?? 0;
  const estimate = estimateGeneration(runs.data, items);

  const runEval = useMutation({
    mutationFn: () => {
      if (!parsed.items) throw new Error("Fix the dataset first");
      if (collId === null) throw new Error("Select a collection");
      return api.runEval({
        dataset: parsed.items,
        collection_id: collId,
        top_k: topK,
        concurrency: conc,
      });
    },
    onMutate: () => {
      setRunning({ since: performance.now(), items });
      setOpen(false);
    },
    onSuccess: (r) => {
      qc.setQueryData(["eval-run", r.run_id], r);
      setSelectedId(r.run_id);
      qc.invalidateQueries({ queryKey: ["eval-results"] });
    },
    onError: (e) => {
      // Back to the form, which says why.
      setFormError(e instanceof Error ? e.message : String(e));
      setOpen(true);
    },
    onSettled: () => {
      setRunning(null);
      setConfirming(false);
    },
  });

  function openForm() {
    setFormError(null);
    setConfirming(false);
    setOpen(true);
  }

  function show(runId: string) {
    setSelectedId(runId);
    reportRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <div>
      <PageHeader
        title="EVALUATION"
        subtitle="BENCHMARK & TELEMETRY"
        actions={
          <Btn variant="primary" onClick={openForm} disabled={runEval.isPending}>
            + RUN NEW EVALUATION
          </Btn>
        }
      />

      {runs.error && (
        <ErrorBanner
          message="failed to load evaluation runs"
          onRetry={() => void runs.refetch()}
          retrying={runs.isFetching}
        />
      )}

      {/* TOP: the run on screen */}
      <div ref={reportRef} className="scroll-mt-16">
        {running ? (
          <div className="border border-border-subtle bg-surface-container p-6 sm:p-8 mb-8">
            <div className="label-sm text-text-primary blink-cursor">
              EVALUATING {running.items} ITEMS · <Elapsed since={running.since} />
            </div>
            <div className="label-xs text-text-tertiary mt-3">
              EACH ITEM IS ANSWERED, THEN JUDGED TWICE. THE REPORT APPEARS WHEN ALL ARE SCORED.
            </div>
          </div>
        ) : report.data ? (
          <RunReport report={report.data} runs={runs.data ?? []} />
        ) : report.error ? (
          <div className="mb-8">
            <ErrorBanner
              message={`failed to load run: ${report.error.message}`}
              onRetry={() => void report.refetch()}
              retrying={report.isFetching}
            />
          </div>
        ) : shownId !== null || runs.isLoading ? (
          <div className="mb-8">
            <LoadingLine />
          </div>
        ) : null}
      </div>

      {/* BOTTOM: historical runs */}
      <h2 className="title-sm text-text-primary mb-3">HISTORICAL RUNS</h2>
      <div className="border border-border-subtle bg-surface-container overflow-x-auto">
        {runs.isLoading ? (
          <div className="p-4">
            <LoadingLine />
          </div>
        ) : !runs.data || runs.data.length === 0 ? (
          <EmptyState label="NO HISTORICAL RUNS">
            <button type="button" onClick={openForm} className={emptyActionCls}>
              SET UP A RUN
            </button>
          </EmptyState>
        ) : (
          <table className="w-full min-w-[640px]">
            <thead>
              <tr className="label-sm text-text-secondary border-b border-border-subtle">
                <th className="text-left pl-6 py-3 font-normal">RUN_ID</th>
                <th className="text-left py-3 font-normal">DATE</th>
                <th className="text-right py-3 font-normal">ITEMS</th>
                <th className="text-right py-3 font-normal">ACCURACY</th>
                <th className="text-right py-3 font-normal">HALLUCINATION</th>
                <th className="text-right py-3 font-normal" title={GENERATION_COST_HINT}>
                  GEN. COST
                </th>
                <th className="text-right pr-6 py-3 font-normal w-20">
                  <span className="sr-only">ACTIONS</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {runs.data.map((r) => {
                const current = r.run_id === shownId;
                return (
                  <tr
                    key={r.run_id}
                    aria-current={current || undefined}
                    className={`body-sm h-9 border-b border-border-subtle/50 hover:bg-surface-high ${
                      current ? "bg-surface-high" : ""
                    }`}
                  >
                    <td className="pl-6 text-text-primary" title={r.run_id}>
                      {r.run_id.slice(0, 8)}
                    </td>
                    <td className="text-text-secondary tabular-nums">{fmtDate(r.timestamp)}</td>
                    <td className="text-right tabular-nums text-text-secondary">
                      {r.dataset_size}
                    </td>
                    <td className={`text-right tabular-nums ${toneText("accuracy", r.accuracy)}`}>
                      {fmtPct(r.accuracy)}
                    </td>
                    <td
                      className={`text-right tabular-nums ${toneText("hallucination", r.hallucination_rate)}`}
                    >
                      {fmtPct(r.hallucination_rate)}
                    </td>
                    <td className="text-right tabular-nums text-text-primary">
                      {fmtCost(r.estimated_cost_usd)}
                    </td>
                    <td className="text-right pr-6">
                      {current ? (
                        <span className="label-sm text-text-tertiary px-2">SHOWN</span>
                      ) : (
                        <button
                          type="button"
                          onClick={() => show(r.run_id)}
                          className={btnCls("ghost", "sm")}
                        >
                          VIEW
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <Modal
        open={open}
        onClose={() => setOpen(false)}
        title="EXECUTE EVALUATION"
        width="max-w-2xl"
      >
        {/* Not a <Field>: a <label> around the button makes a click on the
            label's text press it and replace the user's dataset. */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <label htmlFor="eval-dataset" className="label-sm text-text-secondary">
              DATASET (JSON ARRAY)
            </label>
            <button
              type="button"
              onClick={() => setDataset(STARTER_TEXT)}
              className="label-xs text-text-secondary hover:text-primary-blue"
            >
              ↺ LOAD STARTER DATASET
            </button>
          </div>
          <textarea
            id="eval-dataset"
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            rows={10}
            spellCheck={false}
            aria-invalid={parsed.items === null}
            aria-describedby="eval-dataset-status"
            className={textareaCls}
          />
          <div id="eval-dataset-status" className="mt-2 label-xs">
            {parsed.items ? (
              <span className="text-text-tertiary">
                {parsed.items.length} ITEMS · EACH CALLS THE ANSWER MODEL ONCE AND THE JUDGE TWICE
              </span>
            ) : (
              <ul className="text-error space-y-0.5 normal-case">
                {parsed.errors.slice(0, 6).map((e) => (
                  <li key={e}>{e}</li>
                ))}
                {parsed.errors.length > 6 && <li>…and {parsed.errors.length - 6} more</li>}
              </ul>
            )}
          </div>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Field label="COLLECTION">
            <select
              value={collId ?? ""}
              onChange={(e) => setCollId(e.target.value ? Number(e.target.value) : null)}
              className={inputCls}
            >
              <option value="">— SELECT —</option>
              {collections.data?.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </Field>
          <Field label="TOP_K">
            <input
              type="number"
              min={TOP_K_MIN}
              max={TOP_K_MAX}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              title={`Chunks retrieved for each question. The backend's default is ${defaultTopK}.`}
              className={inputCls}
            />
          </Field>
          <Field label="CONCURRENCY">
            <input
              type="number"
              min={1}
              max={16}
              value={conc}
              onChange={(e) => setConc(Number(e.target.value) || 4)}
              className={inputCls}
            />
          </Field>
        </div>
        {formError && <div className="label-sm text-error my-3">ERROR: {formError}</div>}
        {confirming && parsed.items && collId !== null ? (
          // A run bills every item, so it starts only from here.
          <div className="mt-4 border border-warning/40 p-4">
            <div className="label-sm text-text-primary">
              THIS RUN BILLS {items} ANSWER {items === 1 ? "CALL" : "CALLS"} AND {items * 2} JUDGE
              CALLS
            </div>
            <div className="body-sm text-text-secondary mt-2">
              {estimate ? (
                <>
                  Generating the answers should cost about{" "}
                  <span className="text-text-primary">{fmtCost(estimate.usd)}</span>: the last run
                  cost {fmtCost(estimate.from.estimated_cost_usd)} for {estimate.from.dataset_size}{" "}
                  items. The judge and the embeddings are billed on top and are not measured.
                </>
              ) : (
                <>
                  No earlier run to estimate from. The answers, the judge and the embeddings are all
                  billed.
                </>
              )}
            </div>
            <div className="flex justify-end gap-2 mt-4">
              <Btn onClick={() => setConfirming(false)}>BACK</Btn>
              <Btn
                variant="primary"
                disabled={runEval.isPending}
                onClick={() => {
                  setFormError(null);
                  runEval.mutate();
                }}
              >
                {estimate ? `RUN · ~${fmtCost(estimate.usd)} + JUDGE` : "RUN"}
              </Btn>
            </div>
          </div>
        ) : (
          <div className="flex justify-end gap-2 mt-4">
            <Btn onClick={() => setOpen(false)}>CANCEL</Btn>
            <Btn
              variant="primary"
              disabled={runEval.isPending || parsed.items === null || collId === null}
              onClick={() => {
                setFormError(null);
                setConfirming(true);
              }}
            >
              EXECUTE…
            </Btn>
          </div>
        )}
      </Modal>
    </div>
  );
}

import {
  fmtCost,
  fmtDate,
  fmtNumber,
  fmtPct,
  type EvalBucket,
  type EvalReport,
  type EvalRunSummary,
} from "@/lib/api";
import { EmptyState } from "@/components/ui-kit/EmptyState";
import { ItemResults } from "./ItemResults";
import { RunCompare } from "./RunCompare";
import { GENERATION_COST_HINT } from "./costHint";
import { toneText } from "@/lib/metricTone";

export function RunReport({ report, runs }: { report: EvalReport; runs: EvalRunSummary[] }) {
  const m = report.metrics;
  const judgeErrors = m.judge_errors ?? 0;
  return (
    <div className="mb-8">
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-2">
        <div className="title-lg text-text-primary break-all">{report.run_id}</div>
        <div className="label-sm text-text-secondary tabular-nums">{fmtDate(report.timestamp)}</div>
      </div>
      <div className="label-sm text-text-secondary mb-4">
        MODEL: <span className="text-text-primary">{report.model}</span>
        <span className="text-text-tertiary"> · </span>
        ITEMS: <span className="text-text-primary tabular-nums">{report.dataset_size}</span>
        <span className="text-text-tertiary"> · </span>
        <span title={GENERATION_COST_HINT}>
          GENERATION COST:{" "}
          <span className="text-text-primary">{fmtCost(m.estimated_cost_usd)}</span>
        </span>
      </div>
      {judgeErrors > 0 && (
        <div className="label-sm text-warning border border-warning/40 px-3 py-2 mb-4">
          {judgeErrors} {judgeErrors === 1 ? "ITEM HAS" : "ITEMS HAVE"} A FAILED JUDGE CALL: THEIR
          VERDICTS COUNT AS FAILURES BUT SAY NOTHING ABOUT THE ANSWER.
        </div>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-8">
        <BigMetric
          label="ACCURACY"
          value={fmtPct(m.accuracy)}
          tone={toneText("accuracy", m.accuracy)}
        />
        <BigMetric
          label="HALLUCINATION RATE"
          value={fmtPct(m.hallucination_rate)}
          tone={toneText("hallucination", m.hallucination_rate)}
        />
        <BigMetric
          label="PRECISION@K · RECALL@K"
          value={`${fmtPct(m.avg_retrieval_precision, 0)} · ${fmtPct(m.avg_retrieval_recall, 0)}`}
        />
        <BigMetric
          label="LATENCY AVG · P95"
          value={
            m.latency_p95_ms === undefined
              ? `${fmtNumber(Math.round(m.avg_latency_ms))}MS · —`
              : `${fmtNumber(Math.round(m.avg_latency_ms))} · ${fmtNumber(Math.round(m.latency_p95_ms))}MS`
          }
          tone={toneText("latency", m.avg_latency_ms)}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <BreakdownTable title="BY DIFFICULTY" data={m.by_difficulty} />
        <BreakdownTable title="BY CATEGORY" data={m.by_category} />
      </div>

      <RunCompare report={report} runs={runs} />

      <ItemResults results={report.results} />
    </div>
  );
}

function BigMetric({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="border border-border-subtle bg-surface-container p-4 sm:p-5 min-w-0">
      <div className="label-sm text-text-secondary">{label}</div>
      <div className={`display-md mt-3 tabular-nums break-words ${tone ?? "text-text-primary"}`}>
        {value}
      </div>
    </div>
  );
}

function BreakdownTable({ title, data }: { title: string; data?: Record<string, EvalBucket> }) {
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
                <th className="text-right py-2 font-normal">ITEMS</th>
                <th className="text-right py-2 font-normal">ACCURACY</th>
                <th className="text-right pr-6 py-2 font-normal">HAL</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data).map(([k, v]) => (
                <tr
                  key={k}
                  className="body-sm h-9 hover:bg-surface-high border-b border-border-subtle/50"
                >
                  <td className="pl-6 text-text-primary">{k.replace("_", " ").toUpperCase()}</td>
                  <td className="text-right tabular-nums text-text-primary">{v.total}</td>
                  <td className={`text-right tabular-nums ${toneText("accuracy", v.accuracy)}`}>
                    {fmtPct(v.accuracy)}
                  </td>
                  <td
                    className={`text-right pr-6 tabular-nums ${toneText("hallucination", v.hallucination_rate)}`}
                  >
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

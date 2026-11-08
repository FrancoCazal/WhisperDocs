import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  api,
  fmtCost,
  fmtDate,
  fmtNumber,
  fmtPct,
  type EvalReport,
  type EvalRunSummary,
} from "@/lib/api";
import { diffRuns, trend, type ItemChange, type MetricDelta } from "@/lib/evalDiff";
import { ErrorBanner, LoadingLine } from "@/components/ui-kit/EmptyState";

const short = (runId: string) => runId.slice(0, 8);
// The list says "+00:00" and a full report "Z": compare instants, not strings.
const at = (iso: string) => Date.parse(iso);

/** The shown run against another one: metric moves, and the items that flipped. */
export function RunCompare({ report, runs }: { report: EvalReport; runs: EvalRunSummary[] }) {
  const others = runs.filter((r) => r.run_id !== report.run_id);
  // By default, the run before this one.
  const previous =
    others.find((r) => at(r.timestamp) < at(report.timestamp))?.run_id ?? others[0]?.run_id ?? null;
  const [picked, setPicked] = useState<string | null>(null);
  const otherId = picked !== null && others.some((r) => r.run_id === picked) ? picked : previous;

  const other = useQuery({
    queryKey: ["eval-run", otherId],
    queryFn: () => api.getEvalRun(otherId!),
    enabled: otherId !== null,
    // A saved report never changes.
    staleTime: Infinity,
  });

  if (others.length === 0) return null;
  // Older first, so a regression is always something that got worse over time.
  const pair =
    other.data &&
    (at(other.data.timestamp) <= at(report.timestamp)
      ? { prev: other.data, curr: report }
      : { prev: report, curr: other.data });
  const diff = pair && diffRuns(pair.prev, pair.curr);

  return (
    <section aria-label="Compare runs" className="mb-8">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <h3 className="title-sm text-text-primary">COMPARE</h3>
        <label className="flex items-center gap-2 label-sm text-text-secondary">
          WITH
          <select
            value={otherId ?? ""}
            onChange={(e) => setPicked(e.target.value)}
            className="bg-surface-lowest border border-border-subtle px-2 h-8 text-text-primary body-sm focus:outline-none focus:border-border-focus"
          >
            {others.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {short(r.run_id)} · {fmtDate(r.timestamp)} · {r.dataset_size} ITEMS
              </option>
            ))}
          </select>
        </label>
      </div>

      {other.error ? (
        <ErrorBanner
          message={`failed to load run ${short(otherId ?? "")}: ${other.error.message}`}
        />
      ) : !pair || !diff ? (
        <LoadingLine />
      ) : (
        <>
          <div className="label-xs text-text-tertiary mb-3">
            {short(pair.prev.run_id)} → {short(pair.curr.run_id)} ·{" "}
            {diff.common === 0
              ? "NO ITEMS IN COMMON: ONLY THE TOTALS COMPARE"
              : `${diff.common} ITEMS IN COMMON`}
            {diff.common > 0 &&
              (diff.common < pair.prev.results.length ||
                diff.common < pair.curr.results.length) && (
                <span className="text-warning">
                  {" "}
                  · THE TOTALS COVER DIFFERENT ITEMS ({pair.prev.results.length} VS{" "}
                  {pair.curr.results.length})
                </span>
              )}
          </div>
          <div className="border border-border-subtle bg-surface-container overflow-x-auto mb-4">
            <table className="w-full min-w-[480px]">
              <thead>
                <tr className="label-sm text-text-secondary border-b border-border-subtle">
                  <th className="text-left pl-4 py-2 font-normal">METRIC</th>
                  <th className="text-right py-2 font-normal">{short(pair.prev.run_id)}</th>
                  <th className="text-right py-2 font-normal">{short(pair.curr.run_id)}</th>
                  <th className="text-right pr-4 py-2 font-normal">CHANGE</th>
                </tr>
              </thead>
              <tbody>
                {diff.metrics.map((m) => (
                  <tr key={m.label} className="body-sm border-b border-border-subtle/50">
                    <td className="pl-4 py-2 text-text-secondary">{m.label}</td>
                    <td className="py-2 text-right tabular-nums text-text-secondary">
                      {fmtValue(m, m.prev)}
                    </td>
                    <td className="py-2 text-right tabular-nums text-text-primary">
                      {fmtValue(m, m.curr)}
                    </td>
                    <td className={`py-2 pr-4 text-right tabular-nums ${TREND_TONE[trend(m)]}`}>
                      {fmtChange(m)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {diff.common > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Changes title="REGRESSIONS" tone="text-error" items={diff.regressions} />
              <Changes title="FIXES" tone="text-success" items={diff.fixes} />
            </div>
          )}
        </>
      )}
    </section>
  );
}

const TREND_TONE = {
  better: "text-success",
  worse: "text-error",
  same: "text-text-tertiary",
} as const;

function fmtValue(m: MetricDelta, v: number | null) {
  if (v === null) return "—";
  if (m.kind === "pct") return fmtPct(v);
  if (m.kind === "cost") return fmtCost(v);
  return `${fmtNumber(Math.round(v))}MS`;
}

function fmtChange(m: MetricDelta) {
  if (m.prev === null || m.curr === null) return "—";
  const d = m.curr - m.prev;
  const sign = d > 0 ? "+" : d < 0 ? "−" : "±";
  const abs = Math.abs(d);
  if (m.kind === "pct") return `${sign}${(abs * 100).toFixed(1)} PTS`;
  if (m.kind === "cost") return `${sign}$${abs.toFixed(4)}`;
  return `${sign}${fmtNumber(Math.round(abs))}MS`;
}

function Changes({ title, tone, items }: { title: string; tone: string; items: ItemChange[] }) {
  return (
    <div className="border border-border-subtle bg-surface-container p-4">
      <div className={`label-sm mb-3 ${tone}`}>
        {title} · {items.length}
      </div>
      {items.length === 0 ? (
        <div className="label-xs text-text-tertiary">NONE</div>
      ) : (
        <ul className="space-y-3">
          {items.map((c) => (
            <li key={`${c.id}-${c.check}`}>
              <div className="body-sm text-text-primary">
                <span className="text-text-tertiary tabular-nums mr-2">#{c.id}</span>
                {c.question}
              </div>
              <div className="label-xs text-text-tertiary mt-0.5">{c.check.toUpperCase()}</div>
              <div className="body-sm text-text-secondary mt-1">{c.reasoning}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

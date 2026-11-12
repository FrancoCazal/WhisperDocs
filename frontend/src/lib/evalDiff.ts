// Two eval runs side by side. The API compares them (`GET /eval/diff`), with
// the code `runner diff` runs (app/eval/report.py run_diff); this file only
// reads its answer for the screen.
import type { EvalMetrics, RunDiff } from "@/lib/api";

export type MetricKind = "pct" | "ms" | "cost";

export type MetricDelta = {
  label: string;
  // null: that run's report is older than the metric.
  prev: number | null;
  curr: number | null;
  kind: MetricKind;
  higherIsBetter: boolean;
};

// Each run's version of a set of prompts; null for a report from before them.
export type VersionPair = { prev: string | null; curr: string | null };

/** Both runs name their version and the two differ. A missing version says
 * nothing either way, as in the CLI's diff. */
export function versionChanged(v: VersionPair): boolean {
  return v.prev !== null && v.curr !== null && v.prev !== v.curr;
}

const METRICS: Partial<Record<keyof EvalMetrics, [string, MetricKind, boolean]>> = {
  accuracy: ["ACCURACY", "pct", true],
  hallucination_rate: ["UNSUPPORTED CLAIMS", "pct", false],
  avg_retrieval_precision: ["PRECISION@K", "pct", true],
  avg_retrieval_recall: ["RECALL@K", "pct", true],
  avg_latency_ms: ["LATENCY AVG", "ms", false],
  latency_p50_ms: ["LATENCY P50", "ms", false],
  latency_p95_ms: ["LATENCY P95", "ms", false],
  estimated_cost_usd: ["GENERATION COST", "cost", false],
};

/** The API's metrics, labelled, in its order; one the screen does not know
 * is left out. */
export function metricDeltas(diff: RunDiff): MetricDelta[] {
  return diff.metrics.flatMap((m) => {
    const view = METRICS[m.metric];
    if (!view) return [];
    const [label, kind, higherIsBetter] = view;
    return [{ label, prev: m.previous, curr: m.current, kind, higherIsBetter }];
  });
}

/** A check as the screen names it: the hallucination check passes an
 * answer that is grounded. */
export function checkLabel(check: string): string {
  return check === "hallucination" ? "GROUNDING" : check.toUpperCase();
}

/** A p-value as the CLI prints it. */
export function fmtP(p: number): string {
  return p < 0.001 ? "< 0.001" : p.toFixed(3);
}

/** The API's text in pieces: those between backticks are versions, SHAs or
 * hashes, shown as written. */
export function splitTicks(text: string): { text: string; code: boolean }[] {
  return text
    .split("`")
    .map((part, i) => ({ text: part, code: i % 2 === 1 }))
    .filter((part) => part.text !== "");
}

/** A setting's value as the CLI prints it; null is a setting that does not
 * apply. */
export function fmtSetting(value: string | number | boolean | null): string {
  return value === null ? "—" : String(value);
}

/** "better", "worse" or "same" for a metric's move. */
export function trend(m: MetricDelta): "better" | "worse" | "same" {
  if (m.prev === null || m.curr === null) return "same";
  const d = m.curr - m.prev;
  const noise = m.kind === "pct" ? 0.0005 : m.kind === "ms" ? 0.5 : 0.00005;
  if (Math.abs(d) < noise) return "same";
  return d > 0 === m.higherIsBetter ? "better" : "worse";
}

/** The 95 % Wilson interval of a share of n items, as app/eval/protocol.py
 * computes it. A report saves the share rounded, so the count is the
 * nearest whole number. */
export function wilsonInterval(share: number, n: number): [number, number] {
  if (n <= 0) return [0, 1];
  const z = 1.959964;
  const p = Math.round(share * n) / n;
  const denominator = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denominator;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denominator;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

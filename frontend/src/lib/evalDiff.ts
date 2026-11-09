// Two eval runs side by side: what moved, and which items flipped. The CLI's
// markdown version is `app/eval/report.py` (render_diff_markdown).
import type { EvalMetrics, EvalReport, EvalResult, JudgeVerdict } from "@/lib/api";

export type MetricKind = "pct" | "ms" | "cost";

export type MetricDelta = {
  label: string;
  // null: that run's report is older than the metric.
  prev: number | null;
  curr: number | null;
  kind: MetricKind;
  higherIsBetter: boolean;
};

export type Check = "correctness" | "grounding";

export type ItemChange = {
  id: number;
  question: string;
  check: Check;
  // The newer run's judge reasoning: why it passes or fails now.
  reasoning: string;
};

// Each run's version of a set of prompts; null for a report from before them.
export type VersionPair = { prev: string | null; curr: string | null };

export type RunDiff = {
  metrics: MetricDelta[];
  // Items in both runs: same id and same question.
  common: number;
  regressions: ItemChange[];
  fixes: ItemChange[];
  prompts: VersionPair;
  judgePrompts: VersionPair;
};

/** Both runs name their version and the two differ. A missing version says
 * nothing either way, as in the CLI's diff. */
export function versionChanged(v: VersionPair): boolean {
  return v.prev !== null && v.curr !== null && v.prev !== v.curr;
}

const METRICS: [string, keyof EvalMetrics, MetricKind, boolean][] = [
  ["ACCURACY", "accuracy", "pct", true],
  ["HALLUCINATION RATE", "hallucination_rate", "pct", false],
  ["PRECISION@K", "avg_retrieval_precision", "pct", true],
  ["RECALL@K", "avg_retrieval_recall", "pct", true],
  ["LATENCY AVG", "avg_latency_ms", "ms", false],
  ["LATENCY P95", "latency_p95_ms", "ms", false],
  ["GENERATION COST", "estimated_cost_usd", "cost", false],
];

const CHECKS: [Check, (r: EvalResult) => JudgeVerdict][] = [
  ["correctness", (r) => r.correctness],
  ["grounding", (r) => r.hallucination],
];

function metric(r: EvalReport, key: keyof EvalMetrics): number | null {
  const v = r.metrics[key];
  return typeof v === "number" ? v : null;
}

export function diffRuns(prev: EvalReport, curr: EvalReport): RunDiff {
  const metrics = METRICS.map(([label, key, kind, higherIsBetter]) => ({
    label,
    prev: metric(prev, key),
    curr: metric(curr, key),
    kind,
    higherIsBetter,
  }));

  const before = new Map(prev.results.map((r) => [r.item_id, r]));
  let common = 0;
  const regressions: ItemChange[] = [];
  const fixes: ItemChange[] = [];
  for (const r of curr.results) {
    const p = before.get(r.item_id);
    // Another dataset can reuse an id for another question.
    if (!p || p.question !== r.question) continue;
    common++;
    for (const [check, verdict] of CHECKS) {
      const was = verdict(p);
      const now = verdict(r);
      // A failed judge call is no verdict, so it is no change either.
      if (was.error || now.error || was.passed === now.passed) continue;
      const change = { id: r.item_id, question: r.question, check, reasoning: now.reasoning };
      (now.passed ? fixes : regressions).push(change);
    }
  }
  return {
    metrics,
    common,
    regressions,
    fixes,
    prompts: { prev: prev.prompt_version ?? null, curr: curr.prompt_version ?? null },
    judgePrompts: {
      prev: prev.judge_prompt_version ?? null,
      curr: curr.judge_prompt_version ?? null,
    },
  };
}

/** "better", "worse" or "same" for a metric's move. */
export function trend(m: MetricDelta): "better" | "worse" | "same" {
  if (m.prev === null || m.curr === null) return "same";
  const d = m.curr - m.prev;
  const noise = m.kind === "pct" ? 0.0005 : m.kind === "ms" ? 0.5 : 0.00005;
  if (Math.abs(d) < noise) return "same";
  return d > 0 === m.higherIsBetter ? "better" : "worse";
}

import type { EvalMetrics } from "./api";

/** The unsupported-claims figure a run headlines (audit E5): over the
 * in-scope items when it has any, since the out-of-scope ones are mostly
 * refusals with nothing to support and dilute the rate. The report's
 * `hallucination_rate`, over every item, is what the API and the thresholds
 * read, and the fallback for a run with no in-scope item. */
export function unsupportedClaims(m: EvalMetrics): {
  inScope: boolean;
  rate: number;
  items: number;
} {
  const bucket = m.by_category?.in_scope;
  if (bucket && bucket.total > 0) {
    return { inScope: true, rate: bucket.hallucination_rate, items: bucket.total };
  }
  return { inScope: false, rate: m.hallucination_rate, items: m.total };
}

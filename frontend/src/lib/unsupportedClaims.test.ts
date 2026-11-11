import { describe, expect, it } from "vitest";
import type { EvalMetrics } from "./api";
import { unsupportedClaims } from "./unsupportedClaims";

function metrics(overrides: Partial<EvalMetrics> = {}): EvalMetrics {
  return {
    total: 50,
    accuracy: 0.9,
    hallucination_rate: 0.06,
    avg_retrieval_precision: 0.9,
    avg_latency_ms: 1000,
    avg_tokens_input: 1,
    avg_tokens_output: 1,
    total_tokens_input: 1,
    total_tokens_output: 1,
    estimated_cost_usd: 0,
    ...overrides,
  };
}

describe("the unsupported-claims figure of a run", () => {
  it("is over the in-scope items, which the out-of-scope refusals would dilute", () => {
    const m = metrics({
      by_category: {
        in_scope: { total: 41, accuracy: 0.95, hallucination_rate: 0.0732 },
        out_of_scope: { total: 9, accuracy: 1, hallucination_rate: 0 },
      },
    });
    expect(unsupportedClaims(m)).toEqual({ inScope: true, rate: 0.0732, items: 41 });
  });

  it("falls back to every item when the run has no in-scope item", () => {
    const only = metrics({
      by_category: { out_of_scope: { total: 50, accuracy: 1, hallucination_rate: 0 } },
    });
    expect(unsupportedClaims(only)).toEqual({ inScope: false, rate: 0.06, items: 50 });
    expect(unsupportedClaims(metrics())).toEqual({ inScope: false, rate: 0.06, items: 50 });
    const empty = metrics({
      by_category: { in_scope: { total: 0, accuracy: 0, hallucination_rate: 0 } },
    });
    expect(unsupportedClaims(empty).inScope).toBe(false);
  });
});

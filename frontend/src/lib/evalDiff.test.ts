import { describe, expect, it } from "vitest";
import type { EvalReport } from "./api";
import { diffRuns, versionChanged } from "./evalDiff";

function report(versions: Partial<EvalReport> = {}): EvalReport {
  return {
    run_id: "r",
    timestamp: "2026-11-09T00:00:00Z",
    dataset_size: 0,
    model: "m",
    metrics: {
      total: 0,
      accuracy: 0,
      hallucination_rate: 0,
      avg_retrieval_precision: 0,
      avg_latency_ms: 0,
      avg_tokens_input: 0,
      avg_tokens_output: 0,
      total_tokens_input: 0,
      total_tokens_output: 0,
      estimated_cost_usd: 0,
    },
    results: [],
    ...versions,
  };
}

describe("prompt versions between two runs", () => {
  it("names each run's prompts and the judge's", () => {
    const diff = diffRuns(
      report({ prompt_version: "aaa", judge_prompt_version: "j1" }),
      report({ prompt_version: "bbb", judge_prompt_version: "j1" }),
    );
    expect(diff.prompts).toEqual({ prev: "aaa", curr: "bbb" });
    expect(diff.judgePrompts).toEqual({ prev: "j1", curr: "j1" });
    expect(versionChanged(diff.prompts)).toBe(true);
    expect(versionChanged(diff.judgePrompts)).toBe(false);
  });

  it("calls a run from before the versions unknown, not a change", () => {
    const diff = diffRuns(report(), report({ prompt_version: "bbb", judge_prompt_version: "j2" }));
    expect(diff.prompts).toEqual({ prev: null, curr: "bbb" });
    expect(versionChanged(diff.prompts)).toBe(false);
    expect(versionChanged(diff.judgePrompts)).toBe(false);
    // An API that sends the field as null reads the same as one without it.
    const nulls = diffRuns(report({ prompt_version: null }), report({ prompt_version: "bbb" }));
    expect(versionChanged(nulls.prompts)).toBe(false);
  });
});

import { describe, expect, it } from "vitest";
import type { RunDiff } from "./api";
import {
  checkLabel,
  fmtP,
  fmtSetting,
  metricDeltas,
  splitTicks,
  trend,
  versionChanged,
  wilsonInterval,
} from "./evalDiff";

function diff(metrics: RunDiff["metrics"]): RunDiff {
  return {
    previous: "a",
    current: "b",
    common_items: 0,
    same_items: true,
    warnings: [],
    changed: [],
    metrics,
    checks: [],
    flips: [],
  };
}

describe("prompt versions between two runs", () => {
  it("is a change only when both runs name a version and the two differ", () => {
    expect(versionChanged({ prev: "aaa", curr: "bbb" })).toBe(true);
    expect(versionChanged({ prev: "aaa", curr: "aaa" })).toBe(false);
    // A run from before the versions says nothing either way.
    expect(versionChanged({ prev: null, curr: "bbb" })).toBe(false);
    expect(versionChanged({ prev: "aaa", curr: null })).toBe(false);
  });
});

describe("the API's metrics on screen", () => {
  it("labels them in the API's order and leaves out one the screen does not know", () => {
    const rows = metricDeltas(
      diff([
        { metric: "hallucination_rate", previous: 0.1, current: 0.2 },
        { metric: "accuracy", previous: 0.5, current: 0.75 },
        { metric: "judge_errors", previous: 0, current: 1 },
        { metric: "latency_p95_ms", previous: null, current: 900 },
      ]),
    );
    expect(rows.map((r) => r.label)).toEqual(["HALLUCINATION RATE", "ACCURACY", "LATENCY P95"]);
    expect(rows[1]).toEqual({
      label: "ACCURACY",
      prev: 0.5,
      curr: 0.75,
      kind: "pct",
      higherIsBetter: true,
    });
  });

  it("reads a move up as better or worse by what the metric measures", () => {
    const [hallucination, accuracy, latency] = metricDeltas(
      diff([
        { metric: "hallucination_rate", previous: 0.1, current: 0.2 },
        { metric: "accuracy", previous: 0.5, current: 0.75 },
        { metric: "latency_p95_ms", previous: null, current: 900 },
      ]),
    );
    expect(trend(hallucination)).toBe("worse");
    expect(trend(accuracy)).toBe("better");
    // A metric one run did not measure did not move.
    expect(trend(latency)).toBe("same");
    // Nor does one within rounding.
    expect(trend({ ...accuracy, curr: accuracy.prev! + 0.0001 })).toBe("same");
  });
});

describe("the diff's text", () => {
  it("names the hallucination check by what passing it means", () => {
    expect(checkLabel("hallucination")).toBe("GROUNDING");
    expect(checkLabel("correctness")).toBe("CORRECTNESS");
  });

  it("prints a p-value as the CLI does", () => {
    expect(fmtP(1)).toBe("1.000");
    expect(fmtP(0.0312)).toBe("0.031");
    expect(fmtP(0.0004)).toBe("< 0.001");
  });

  it("shows what a warning wraps in backticks apart from the prose", () => {
    expect(splitTicks("The judge changed: `j1` → `j2`.")).toEqual([
      { text: "The judge changed: ", code: false },
      { text: "j1", code: true },
      { text: " → ", code: false },
      { text: "j2", code: true },
      { text: ".", code: false },
    ]);
    expect(splitTicks("`abc123` is dirty")).toEqual([
      { text: "abc123", code: true },
      { text: " is dirty", code: false },
    ]);
    expect(splitTicks("no code")).toEqual([{ text: "no code", code: false }]);
  });

  it("prints a setting that does not apply as a dash", () => {
    expect(fmtSetting(null)).toBe("—");
    expect(fmtSetting(false)).toBe("false");
    expect(fmtSetting(0)).toBe("0");
    expect(fmtSetting("gemini-3-flash")).toBe("gemini-3-flash");
  });
});

describe("a share's 95% Wilson interval", () => {
  // The values app/eval/protocol.py's wilson_interval gives.
  it("matches the CLI's", () => {
    const close = ([low, high]: [number, number], want: [number, number]) => {
      expect(low).toBeCloseTo(want[0], 9);
      expect(high).toBeCloseTo(want[1], 9);
    };
    close(wilsonInterval(0.96, 50), [0.8653990920638485, 0.9889611157718264]);
    close(wilsonInterval(0, 20), [0, 0.16112516018512965]);
    close(wilsonInterval(1, 20), [0.8388748398148705, 1]);
    // A report saves the share rounded: 0.3333 of 3 items is 1 of them.
    close(wilsonInterval(0.3333, 3), [0.06149194402093078, 0.7923404011921757]);
  });

  it("says nothing of no items", () => {
    expect(wilsonInterval(0.5, 0)).toEqual([0, 1]);
  });
});

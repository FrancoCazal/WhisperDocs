import { describe, expect, it } from "vitest";
import { LATENCY_WARN_MS, metricTone, toneText } from "./metricTone";

describe("metric tones", () => {
  it("puts each limit on the side the dashboard says", () => {
    expect(metricTone("accuracy", 0.8)).toBe("good");
    expect(metricTone("accuracy", 0.6)).toBe("fair");
    expect(metricTone("accuracy", 0.59)).toBe("poor");
    expect(metricTone("hallucination", 0.04)).toBe("good");
    expect(metricTone("hallucination", 0.15)).toBe("fair");
    expect(metricTone("hallucination", 0.16)).toBe("poor");
    expect(metricTone("latency", 1999)).toBe("good");
    expect(metricTone("latency", LATENCY_WARN_MS)).toBe("fair");
    expect(metricTone("latency", LATENCY_WARN_MS + 1)).toBe("poor");
  });

  it("leaves a missing value uncolored", () => {
    expect(toneText("latency", null)).toBe("text-text-primary");
    expect(toneText("latency", undefined)).toBe("text-text-primary");
    expect(toneText("accuracy", Number.NaN)).toBe("text-text-primary");
    expect(toneText("accuracy", 0.9)).toBe("text-success");
  });
});

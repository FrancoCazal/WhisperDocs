// How good a number is, with one set of limits for the whole app: the
// dashboard, the eval pages and the query rows color the same value the same.
export type Metric = "accuracy" | "hallucination" | "latency";
export type Tone = "good" | "fair" | "poor";

// Latency above this is poor: the dashboard's WARN THRESHOLD.
export const LATENCY_WARN_MS = 5000;

// [good when, poor when]; anything between is fair.
const LIMITS: Record<Metric, [(v: number) => boolean, (v: number) => boolean]> = {
  accuracy: [(v) => v >= 0.8, (v) => v < 0.6],
  hallucination: [(v) => v < 0.05, (v) => v > 0.15],
  // Milliseconds.
  latency: [(v) => v < 2000, (v) => v > LATENCY_WARN_MS],
};

export function metricTone(metric: Metric, value: number): Tone {
  const [good, poor] = LIMITS[metric];
  if (good(value)) return "good";
  if (poor(value)) return "poor";
  return "fair";
}

const TEXT: Record<Tone, string> = {
  good: "text-success",
  fair: "text-warning",
  poor: "text-error",
};

/** The text color for `value`; a missing value is neutral. */
export function toneText(metric: Metric, value: number | null | undefined): string {
  return value === null || value === undefined || Number.isNaN(value)
    ? "text-text-primary"
    : TEXT[metricTone(metric, value)];
}

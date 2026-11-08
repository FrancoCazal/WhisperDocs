export const GENERATION_COST_HINT =
  "What generating the answers cost. The judge's calls and the embeddings are billed on top and not counted.";

// Text colors for eval numbers: good, neutral or bad.
export function accuracyTone(a: number) {
  if (a >= 0.8) return "text-success";
  if (a < 0.6) return "text-error";
  return "text-text-primary";
}

export function hallucinationTone(h: number) {
  if (h < 0.05) return "text-success";
  if (h > 0.15) return "text-error";
  return "text-text-primary";
}

export function latencyTone(ms: number) {
  if (ms < 2000) return "text-success";
  if (ms > 3000) return "text-error";
  return "text-text-primary";
}

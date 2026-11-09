// The prompts' version: 12 hex characters of a hash of their text
// (app/agent/prompt_version.py). Eval reports and query log rows carry it.

export const PROMPTS_HINT =
  "The version of the agent's prompts that answered: a hash of their text, so any edit to them changes it. Runs and log rows from before the versions have none.";

/** A version as the reports name it; "—" for one from before them. */
export function fmtVersion(v: string | null | undefined): string {
  return v ?? "—";
}

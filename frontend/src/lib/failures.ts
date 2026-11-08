import { useQuery } from "@tanstack/react-query";
import { api, type ModelStatus } from "./api";

// What each kind of failure says (app/services/llm/errors.py), and what to
// do about it, as a tooltip.
export const ERROR_KIND: Record<string, { label: string; hint: string }> = {
  billing: {
    label: "BILLING OR QUOTA",
    hint: "The provider refused the call over billing or quota: enable billing on the project, or add credit.",
  },
  auth: {
    label: "CREDENTIALS REFUSED",
    hint: "The provider refused the credentials, or they have no access to this model.",
  },
  rate_limit: {
    label: "RATE LIMITED",
    hint: "Over the provider's rate limit; a retry may pass.",
  },
  timeout: { label: "TIMED OUT", hint: "The call took longer than its timeout." },
  unavailable: {
    label: "UNAVAILABLE",
    hint: "The provider or the database could not be reached, or failed on its side.",
  },
  invalid_request: {
    label: "REQUEST REJECTED",
    hint: "The provider rejected the request, for example an unknown model ID.",
  },
  bad_response: {
    label: "UNREADABLE RESPONSE",
    hint: "The model answered, but not in the expected format.",
  },
  unknown: { label: "ERROR", hint: "The API's logs have the exception." },
};

export function errorKind(kind: string | null | undefined) {
  return ERROR_KIND[kind ?? "unknown"] ?? ERROR_KIND.unknown;
}

// Each role a model plays (app/services/model_status.py), and what the app
// does while its calls fail.
export const ROLES: Record<string, { label: string; effect: string }> = {
  classifier: { label: "CLASSIFIER", effect: "THE CHAT SEARCHES THE DOCUMENTS INSTEAD" },
  query_rewrite: { label: "QUERY REWRITE", effect: "SEARCHES USE THE QUESTION AS WRITTEN" },
  rag_generation: { label: "ANSWERS", effect: "ANSWERS FROM THE DOCUMENTS FAIL" },
  clarification: { label: "CLARIFICATION", effect: "A FIXED QUESTION INSTEAD" },
  answer_direct: { label: "DIRECT ANSWERS", effect: "DIRECT ANSWERS FAIL" },
  summarize: { label: "COMPACTING", effect: "LONG THREADS STAY UNCOMPACTED" },
  eval_judge: { label: "EVAL JUDGE", effect: "EVAL RUNS FAIL" },
  embedding: { label: "EMBEDDING", effect: "SEARCHES AND UPLOADS FAIL" },
};

export type FailingGroup = {
  model: string;
  kind: string;
  roles: string[];
  // The latest failure among the group's roles.
  lastFailureAt: string | null;
};

/** The roles whose last call failed, one group per model and kind: an outage
 *  of one model is one notice, however many roles it plays. In the order of
 *  the roles. */
export function failingGroups(status: ModelStatus): FailingGroup[] {
  const groups = new Map<string, FailingGroup>();
  for (const [role, s] of Object.entries(status.roles)) {
    if (s.state !== "failing") continue;
    const kind = s.last_failure_kind ?? "unknown";
    const key = `${s.model}\n${kind}`;
    const group = groups.get(key) ?? { model: s.model, kind, roles: [], lastFailureAt: null };
    group.roles.push(role);
    if (s.last_failure_at && (!group.lastFailureAt || s.last_failure_at > group.lastFailureAt)) {
      group.lastFailureAt = s.last_failure_at;
    }
    groups.set(key, group);
  }
  return [...groups.values()];
}

/** GET /admin/model-status: read from what the API's calls recorded, so
 *  polling it calls no model. A turn that ends refetches it too. */
export function useModelStatus() {
  return useQuery({
    queryKey: ["model-status"],
    queryFn: api.modelStatus,
    refetchInterval: 30_000,
    retry: false,
  });
}

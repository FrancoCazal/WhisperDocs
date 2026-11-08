import { useCallback } from "react";
import { fmtCost, fmtNumber, type TurnError } from "@/lib/api";
import { citedSources } from "./citations";
import { Markdown } from "./Markdown";
import { TurnTrace } from "./Trace";
import { Btn } from "@/components/ui-kit/Button";
import {
  IDLE_TIMEOUT_MS,
  type AssistantTurnData,
  type TurnOutcome,
  type UserTurnData,
} from "./types";

const ACTION_CLASS: Record<string, string> = {
  search_docs: "text-success border-success",
  answer_direct: "text-text-primary border-text-primary",
  ask_clarification: "text-text-tertiary border-text-tertiary",
};

const OUTCOME: Record<TurnOutcome, { badge: string; className: string; note: string | null }> = {
  failed: { badge: "✕ FAILED", className: "text-error border-error", note: null },
  error: { badge: "✕ ERROR", className: "text-error border-error", note: "NOT SAVED" },
  stopped: {
    badge: "■ STOPPED",
    className: "text-text-secondary border-text-secondary",
    note: "PARTIAL ANSWER · NOT SAVED",
  },
  timeout: {
    badge: "⏱ TIMED OUT",
    className: "text-warning border-warning",
    note: `NO RESPONSE FOR ${IDLE_TIMEOUT_MS / 1000} S · PARTIAL ANSWER NOT SAVED`,
  },
};

// What each kind of failure says (app/services/llm/errors.py), and what to
// do about it, as the row's tooltip.
const ERROR_KIND: Record<string, { label: string; hint: string }> = {
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

// The step that failed, and what the turn did instead; null when the answer
// failed with it.
const ERROR_STEP: Record<string, { label: string; fallback: string | null }> = {
  summarize: { label: "COMPACTING", fallback: "HISTORY LEFT AS IT WAS" },
  classifier: { label: "CLASSIFIER", fallback: "SEARCHED THE DOCUMENTS INSTEAD" },
  query_rewrite: { label: "QUERY REWRITE", fallback: "SEARCHED WITH THE QUESTION AS WRITTEN" },
  embedding: { label: "EMBEDDING", fallback: null },
  retrieval: { label: "RETRIEVAL", fallback: null },
  generation: { label: "ANSWER", fallback: null },
  clarification: { label: "CLARIFICATION", fallback: "ASKED A FIXED QUESTION INSTEAD" },
  citation_check: { label: "CITATION CHECK", fallback: "CITATIONS NOT CHECKED" },
};

function TurnErrors({ errors }: { errors: TurnError[] }) {
  return (
    <ul aria-label="Failed calls" className="mt-3 space-y-1">
      {errors.map((e, i) => {
        const step = ERROR_STEP[e.step];
        const kind = ERROR_KIND[e.kind] ?? ERROR_KIND.unknown;
        const fallback = step ? step.fallback : null;
        return (
          <li
            key={i}
            title={kind.hint}
            className={`label-xs ${fallback ? "text-warning" : "text-error"}`}
          >
            {fallback ? "⚠" : "✕"} {step?.label ?? e.step.toUpperCase()} · {kind.label}
            {e.model && <span className="text-text-secondary"> · {e.model}</span>}
            {fallback && <span className="text-text-tertiary"> · {fallback}</span>}
          </li>
        );
      })}
    </ul>
  );
}

export function UserTurn({ turn }: { turn: UserTurnData }) {
  return (
    <div>
      <div className="label-sm text-text-tertiary mb-2">
        {">"} USER · <span className="text-text-secondary">{turn.ts}</span>
      </div>
      <div className="body-read text-text-primary whitespace-pre-wrap">{turn.text}</div>
      <div className="border-b border-border-subtle mt-4" />
    </div>
  );
}

export function AssistantTurn({
  turn,
  canPin,
  pinning,
  pinned,
  onOpenSources,
  onPin,
  onRetry,
}: {
  turn: AssistantTurnData;
  canPin: boolean;
  pinning: boolean;
  // Saved as a note from this tab.
  pinned: boolean;
  // Stable across renders: the answer's markdown is memoized on it.
  onOpenSources: (turn: AssistantTurnData, focus: number | null) => void;
  onPin: () => void;
  // Only the thread's last turn can be retried.
  onRetry?: () => void;
}) {
  const outcome = turn.outcome ? OUTCOME[turn.outcome] : null;
  const answered = outcome === null;
  // Answers made without a model call (out of scope, clarification) have
  // all-zero metadata: "0MS · 0 IN / 0 OUT · $0.0000" reads as a bug.
  const usage = turn.metadata?.model || turn.metadata?.tokens_input ? turn.metadata : null;
  const onCite = useCallback((n: number) => onOpenSources(turn, n), [onOpenSources, turn]);
  const cited = citedSources(turn.text, turn.citations.length).size;
  const actionClass = turn.action
    ? (ACTION_CLASS[turn.action] ?? "text-text-tertiary border-text-tertiary")
    : "text-text-tertiary border-text-tertiary";
  return (
    <div>
      <div className="label-sm text-text-tertiary mb-2 flex flex-wrap items-center gap-2">
        <span>
          {"<"} AGENT · <span className="text-text-secondary">{turn.ts}</span>
        </span>
        {answered && turn.action && (
          <span className={`label-xs px-2 py-0.5 border ${actionClass}`}>
            {turn.action.toUpperCase()}
          </span>
        )}
        {outcome && (
          <span className={`label-xs px-2 py-0.5 border ${outcome.className}`}>
            {outcome.badge}
          </span>
        )}
        {outcome?.note && <span className="label-xs text-text-tertiary">{outcome.note}</span>}
        <span className="flex-1" />
        {answered && canPin && (
          <button
            onClick={onPin}
            disabled={pinning || pinned}
            className={`label-xs ${
              pinned ? "text-primary-blue" : "text-text-tertiary hover:text-primary-blue"
            }`}
          >
            {pinning ? "PINNING..." : pinned ? "★ PINNED" : "★ PIN NOTE"}
          </button>
        )}
      </div>

      {turn.outcome === "failed" || turn.outcome === "error" ? (
        <div className="border border-error/40 bg-error/5 px-4 py-3 body-sm text-error whitespace-pre-wrap">
          {turn.text}
        </div>
      ) : turn.text ? (
        <Markdown text={turn.text} citations={turn.citations} onCite={onCite} />
      ) : (
        <div className="body-sm text-text-tertiary">Stopped before the answer started.</div>
      )}

      {turn.errors && turn.errors.length > 0 && <TurnErrors errors={turn.errors} />}

      {answered && turn.citations.length > 0 && (
        <button
          onClick={() => onOpenSources(turn, null)}
          className="mt-4 label-sm text-text-secondary hover:text-primary-blue"
        >
          ☰ SOURCES [{turn.citations.length}] · {cited} CITED
        </button>
      )}

      {turn.trace && <TurnTrace steps={turn.trace} elapsedMs={turn.elapsedMs} />}

      {onRetry && (
        <Btn size="sm" variant="quiet" onClick={onRetry} className="mt-3">
          ↻ RETRY
        </Btn>
      )}

      {answered && usage && (
        <div className="label-xs text-text-tertiary mt-4 tabular-nums">
          {/* The turn's own time, from sending to the last token, is in its trace. */}
          <span title="The answer's model call only">
            GEN {fmtNumber(usage.latency_ms)} MS
          </span> · {fmtNumber(usage.tokens_input)} IN / {fmtNumber(usage.tokens_output)} OUT
          {usage.model ? ` · ${usage.model}` : ""} · {fmtCost(usage.cost_usd)}
        </div>
      )}
      <div className="border-b border-border-subtle mt-4" />
    </div>
  );
}

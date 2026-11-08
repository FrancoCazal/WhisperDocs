import { COSINE_HINT, fmtCosine, fmtCost } from "@/lib/api";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Markdown } from "./Markdown";
import { TurnTrace } from "./Trace";
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

export function UserTurn({ turn }: { turn: UserTurnData }) {
  return (
    <div>
      <div className="label-sm text-text-tertiary mb-2">
        {">"} USER · <span className="text-text-secondary">{turn.ts}</span>
      </div>
      <div className="body text-text-primary whitespace-pre-wrap">{turn.text}</div>
      <div className="border-b border-border-subtle mt-4" />
    </div>
  );
}

export function AssistantTurn({
  turn,
  canPin,
  pinning,
  onToggle,
  onPin,
  onRetry,
}: {
  turn: AssistantTurnData;
  canPin: boolean;
  pinning: boolean;
  onToggle: () => void;
  onPin: () => void;
  // Only the thread's last turn can be retried.
  onRetry?: () => void;
}) {
  const outcome = turn.outcome ? OUTCOME[turn.outcome] : null;
  const answered = outcome === null;
  // Answers made without a model call (out of scope, clarification) have
  // all-zero metadata: "0MS · 0 IN / 0 OUT · $0.0000" reads as a bug.
  const usage = turn.metadata?.model || turn.metadata?.tokens_input ? turn.metadata : null;
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
            disabled={pinning}
            className="label-xs text-text-tertiary hover:text-primary-blue"
          >
            {pinning ? "PINNING..." : "★ PIN NOTE"}
          </button>
        )}
      </div>

      {turn.outcome === "failed" || turn.outcome === "error" ? (
        <div className="border border-error/40 bg-error/5 px-4 py-3 body-sm text-error whitespace-pre-wrap">
          {turn.text}
        </div>
      ) : turn.text ? (
        <Markdown text={turn.text} citations={turn.citations} />
      ) : (
        <div className="body-sm text-text-tertiary">Stopped before the answer started.</div>
      )}

      {answered && turn.citations.length > 0 && (
        <div className="mt-4">
          <button
            onClick={onToggle}
            className="label-sm text-text-secondary hover:text-primary-blue"
          >
            {turn.expanded ? "▾" : "▸"} CITATIONS [{turn.citations.length}]
          </button>
          {turn.expanded && (
            <div className="mt-3 space-y-2">
              {turn.citations.map((c, idx) => {
                const num = String(idx + 1).padStart(2, "0");
                return (
                  <HoverCard key={`${c.chunk_id}-${idx}`} openDelay={150}>
                    <HoverCardTrigger asChild>
                      <div
                        id={`cite-${turn.ts}-${idx + 1}`}
                        className="border border-border-subtle bg-surface-container-low p-3 cursor-help"
                      >
                        <div className="label-sm flex items-center gap-3">
                          <span className="text-text-primary">[{num}]</span>
                          <span className="text-text-secondary">{c.document_title}</span>
                          <span className="text-text-tertiary">·</span>
                          <span className="text-text-secondary">CHUNK {c.chunk_index}</span>
                          <span className="text-text-tertiary">·</span>
                          <span className="tabular-nums text-text-secondary" title={COSINE_HINT}>
                            {fmtCosine(c.cosine)}
                          </span>
                        </div>
                      </div>
                    </HoverCardTrigger>
                    {c.text_preview && (
                      <HoverCardContent
                        side="top"
                        align="start"
                        className="w-[480px] max-h-[320px] overflow-auto bg-surface-container border border-border-subtle"
                      >
                        <div className="label-xs text-text-tertiary mb-2">
                          {c.document_title} · CHUNK {c.chunk_index}
                        </div>
                        <div className="body-sm text-text-secondary whitespace-pre-wrap">
                          {c.text_preview}
                        </div>
                      </HoverCardContent>
                    )}
                  </HoverCard>
                );
              })}
            </div>
          )}
        </div>
      )}

      {turn.trace && <TurnTrace steps={turn.trace} elapsedMs={turn.elapsedMs} />}

      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-3 label-sm text-text-primary hover:bg-text-primary hover:text-surface px-3 h-7 border border-border-subtle"
        >
          ↻ RETRY
        </button>
      )}

      {answered && usage && (
        <div className="label-xs text-text-tertiary mt-4 tabular-nums">
          {usage.latency_ms}MS · {usage.tokens_input.toLocaleString()} IN /{" "}
          {usage.tokens_output.toLocaleString()} OUT
          {usage.model ? ` · ${usage.model}` : ""} · {fmtCost(usage.cost_usd)}
        </div>
      )}
      <div className="border-b border-border-subtle mt-4" />
    </div>
  );
}

import type { AgentCitation, AgentTurnMetadata } from "@/lib/api";

export type UserTurnData = { kind: "user"; ts: string; text: string };

// No event from the stream for this long ends the turn as "timeout".
export const IDLE_TIMEOUT_MS = 60_000;

// How a turn ended when it did not end with an answer:
// - failed: the agent answered with a failure report, kept in the thread.
// - stopped / timeout: the user stopped the stream, or no event came for
//   IDLE_TIMEOUT_MS; `text` holds the partial answer, which the thread lacks.
// - error: the request or the stream failed; `text` holds the error.
export type TurnOutcome = "failed" | "stopped" | "timeout" | "error";

export type AssistantTurnData = {
  kind: "assistant";
  ts: string;
  text: string;
  action: string | null;
  citations: AgentCitation[];
  // Null for a turn without an answer, and for messages saved without it.
  metadata: AgentTurnMetadata | null;
  outcome?: TurnOutcome;
  // Measured in this tab, so only turns streamed here have them.
  trace?: NodeProgress[];
  elapsedMs?: number;
  expanded?: boolean;
};

export type Turn = UserTurnData | AssistantTurnData;

export type NodeProgress = {
  node: string;
  // "cut": still running when the turn ended without an answer.
  status: "running" | "done" | "cut";
  summary?: { action?: string; chunks?: number };
  startedAt: number;
  ms?: number;
};

export const NODE_LABELS: Record<string, string> = {
  check_compaction: "CHECK MEMORY",
  summarize: "COMPACTING HISTORY",
  decide_action: "CLASSIFYING INTENT",
  search_docs: "RETRIEVING + GENERATING",
  ask_clarification: "ASKING CLARIFICATION",
  answer_direct: "ANSWERING DIRECTLY",
};

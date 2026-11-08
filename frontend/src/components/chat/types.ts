import type { AgentCitation, AgentTurnMetadata } from "@/lib/api";

export type UserTurnData = { kind: "user"; ts: string; text: string };

export type AssistantTurnData = {
  kind: "assistant";
  ts: string;
  text: string;
  action: string | null;
  citations: AgentCitation[];
  metadata: AgentTurnMetadata;
  expanded?: boolean;
};

export type Turn = UserTurnData | AssistantTurnData;

export type NodeProgress = {
  node: string;
  status: "running" | "done";
  summary?: { action?: string; chunks?: number };
};

export const NODE_LABELS: Record<string, string> = {
  check_compaction: "CHECK MEMORY",
  summarize: "COMPACTING HISTORY",
  decide_action: "CLASSIFYING INTENT",
  search_docs: "RETRIEVING + GENERATING",
  ask_clarification: "ASKING CLARIFICATION",
  answer_direct: "ANSWERING DIRECTLY",
};

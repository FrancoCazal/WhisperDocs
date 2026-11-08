import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import { COSINE_HINT, fmtCosine, fmtCost } from "@/lib/api";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { interleaveCitations } from "./citations";
import type { AssistantTurnData, UserTurnData } from "./types";

const ACTION_CLASS: Record<string, string> = {
  search_docs: "text-success border-success",
  answer_direct: "text-text-primary border-text-primary",
  ask_clarification: "text-text-tertiary border-text-tertiary",
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
}: {
  turn: AssistantTurnData;
  canPin: boolean;
  pinning: boolean;
  onToggle: () => void;
  onPin: () => void;
}) {
  const actionClass = turn.action
    ? (ACTION_CLASS[turn.action] ?? "text-text-tertiary border-text-tertiary")
    : "text-text-tertiary border-text-tertiary";
  return (
    <div>
      <div className="label-sm text-text-tertiary mb-2 flex items-center gap-2">
        <span>
          {"<"} AGENT · <span className="text-text-secondary">{turn.ts}</span>
        </span>
        {turn.action && (
          <span className={`label-xs px-2 py-0.5 border ${actionClass}`}>
            {turn.action.toUpperCase()}
          </span>
        )}
        <span className="flex-1" />
        {canPin && (
          <button
            onClick={onPin}
            disabled={pinning}
            className="label-xs text-text-tertiary hover:text-primary-blue"
          >
            {pinning ? "PINNING..." : "★ PIN NOTE"}
          </button>
        )}
      </div>
      <div className="body text-text-primary leading-relaxed markdown-body">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
          components={{
            p: ({ children }) => <p>{interleaveCitations(children, turn.citations)}</p>,
            li: ({ children }) => <li>{interleaveCitations(children, turn.citations)}</li>,
          }}
        >
          {turn.text}
        </ReactMarkdown>
      </div>

      {turn.citations.length > 0 && (
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

      <div className="label-xs text-text-tertiary mt-4 tabular-nums">
        {turn.metadata.latency_ms}MS · {turn.metadata.tokens_input.toLocaleString()} IN /{" "}
        {turn.metadata.tokens_output.toLocaleString()} OUT
        {turn.metadata.model ? ` · ${turn.metadata.model}` : ""} · {fmtCost(turn.metadata.cost_usd)}
      </div>
      <div className="border-b border-border-subtle mt-4" />
    </div>
  );
}

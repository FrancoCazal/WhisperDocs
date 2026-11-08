import { Markdown } from "./Markdown";
import { NODE_LABELS, type NodeProgress } from "./types";

export function StreamingTurn({ progress, text }: { progress: NodeProgress[]; text: string }) {
  return (
    <div className="border-l-2 border-text-primary pl-4 py-2 space-y-2">
      {progress.length === 0 && !text && (
        <div className="label-sm text-text-tertiary">CONNECTING...</div>
      )}
      {progress.map((p, i) => {
        const label = NODE_LABELS[p.node] ?? p.node.toUpperCase();
        const detail = p.summary?.action
          ? ` → ${p.summary.action}`
          : p.summary?.chunks !== undefined
            ? ` · ${p.summary.chunks} chunks`
            : "";
        return (
          <div
            key={`${p.node}-${i}`}
            className={`label-sm ${
              p.status === "running" ? "text-text-primary" : "text-text-tertiary"
            }`}
          >
            {label}
            {detail} {p.status === "running" ? "▌" : "✓"}
          </div>
        );
      })}
      {text && (
        <div>
          <Markdown text={text} />
          <span className="inline-block w-2 h-4 bg-text-primary animate-pulse align-middle" />
        </div>
      )}
    </div>
  );
}

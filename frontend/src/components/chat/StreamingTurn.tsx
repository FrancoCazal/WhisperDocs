import { Markdown } from "./Markdown";
import { Elapsed, TraceSteps } from "./Trace";
import type { NodeProgress } from "./types";

export type LiveTurn = { startedAt: number; progress: NodeProgress[]; text: string };

export function StreamingTurn({ live }: { live: LiveTurn }) {
  return (
    <div className="border-l-2 border-text-primary pl-4 py-2 space-y-2">
      <div className="label-xs text-text-tertiary flex gap-2">
        <span className="flex-1">{live.progress.length === 0 ? "CONNECTING..." : "WORKING"}</span>
        <Elapsed since={live.startedAt} />
      </div>
      <TraceSteps steps={live.progress} />
      {live.text && (
        <div>
          <Markdown text={live.text} />
          <span className="inline-block w-2 h-4 bg-text-primary animate-pulse align-middle" />
        </div>
      )}
    </div>
  );
}

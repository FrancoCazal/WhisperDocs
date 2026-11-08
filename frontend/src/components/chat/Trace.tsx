import { useEffect, useState } from "react";
import { NODE_LABELS, type NodeProgress } from "./types";

function fmtSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} S`;
}

const STATUS_MARK: Record<NodeProgress["status"], string> = { running: "▌", done: "✓", cut: "✕" };

function stepDetail(p: NodeProgress): string {
  if (p.summary?.action) return ` → ${p.summary.action}`;
  if (p.summary?.chunks !== undefined) return ` · ${p.summary.chunks} chunks`;
  return "";
}

export function TraceSteps({ steps }: { steps: NodeProgress[] }) {
  return (
    <ol className="space-y-1">
      {steps.map((p, i) => (
        <li
          key={`${p.node}-${i}`}
          className={`label-sm flex gap-2 ${
            p.status === "running" ? "text-text-primary" : "text-text-tertiary"
          }`}
        >
          <span className="flex-1 min-w-0">
            {NODE_LABELS[p.node] ?? p.node.toUpperCase()}
            {stepDetail(p)} {STATUS_MARK[p.status]}
          </span>
          {p.ms !== undefined && <span className="tabular-nums shrink-0">{fmtSeconds(p.ms)}</span>}
        </li>
      ))}
    </ol>
  );
}

/** Time since `since` (a performance.now() reading), ticking while mounted. */
export function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(since);
  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(id);
  }, []);
  return <span className="tabular-nums">{fmtSeconds(Math.max(0, now - since))}</span>;
}

/** The nodes a finished turn ran, collapsed to one line. */
export function TurnTrace({ steps, elapsedMs }: { steps: NodeProgress[]; elapsedMs?: number }) {
  const [open, setOpen] = useState(false);
  if (steps.length === 0) return null;
  const route = steps.find((p) => p.summary?.action)?.summary?.action;
  const chunks = steps.find((p) => p.summary?.chunks !== undefined)?.summary?.chunks;
  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="label-xs text-text-tertiary hover:text-primary-blue tabular-nums"
      >
        {open ? "▾" : "▸"} TRACE · {steps.length} STEPS
        {route ? ` · ${route}` : ""}
        {chunks !== undefined ? ` · ${chunks} chunks` : ""}
        {elapsedMs !== undefined ? ` · ${fmtSeconds(elapsedMs)}` : ""}
      </button>
      {open && (
        <div className="mt-2 border-l border-border-subtle pl-3">
          <TraceSteps steps={steps} />
        </div>
      )}
    </div>
  );
}

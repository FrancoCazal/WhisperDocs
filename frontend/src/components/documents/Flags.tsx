import { FLAGS_HINT, flagRule } from "@/lib/instructionFlags";

const tagCls = "label-xs text-warning border border-warning/40 px-1.5 py-0.5 whitespace-nowrap";

/** A document's count of chunks with instruction-like text; nothing when none. */
export function FlaggedCount({ count }: { count: number | undefined }) {
  if (!count) return null;
  return (
    <span title={FLAGS_HINT} className={tagCls}>
      <span aria-hidden="true">⚠ </span>
      {count} FLAGGED
      <span className="sr-only"> {count === 1 ? "chunk" : "chunks"}</span>
    </span>
  );
}

/** The rules a chunk's text matched, each with what it matches on hover. */
export function ChunkFlags({ rules }: { rules: string[] | undefined }) {
  if (!rules || rules.length === 0) return null;
  return (
    <span className="flex flex-wrap items-center gap-1.5">
      <span title={FLAGS_HINT} className="label-xs text-warning">
        <span aria-hidden="true">⚠ </span>FLAGGED:
      </span>
      {rules.map((name) => {
        const rule = flagRule(name);
        return (
          <span key={name} title={rule.hint} className={tagCls}>
            {rule.label}
          </span>
        );
      })}
    </span>
  );
}

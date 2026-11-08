import { fmtNumber } from "@/lib/api";

/** A paged list's foot: how much of the total is on screen, and the next page. */
export function LoadMore({
  shown,
  total,
  hasMore,
  loading,
  onLoad,
  className = "border border-t-0 border-border-subtle bg-surface-container",
}: {
  shown: number;
  total: number | null;
  hasMore: boolean;
  loading: boolean;
  onLoad: () => void;
  className?: string;
}) {
  if (!hasMore) return null;
  return (
    <div
      className={`flex flex-wrap items-center justify-between gap-2 px-4 py-2 label-xs text-text-tertiary ${className}`}
    >
      <span className="tabular-nums">
        SHOWING {fmtNumber(shown)}
        {total !== null && ` OF ${fmtNumber(total)}`}
      </span>
      <button
        type="button"
        onClick={onLoad}
        disabled={loading}
        className="label-sm text-text-primary px-2 h-7 border border-border-subtle hover:bg-text-primary hover:text-surface disabled:text-text-tertiary disabled:hover:bg-transparent"
      >
        {loading ? "LOADING..." : "LOAD MORE"}
      </button>
    </div>
  );
}

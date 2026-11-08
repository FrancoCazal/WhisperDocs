import { fmtNumber } from "@/lib/api";
import { Btn } from "@/components/ui-kit/Button";

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
      <Btn size="sm" variant="quiet" onClick={onLoad} disabled={loading}>
        {loading ? "LOADING..." : "LOAD MORE"}
      </Btn>
    </div>
  );
}

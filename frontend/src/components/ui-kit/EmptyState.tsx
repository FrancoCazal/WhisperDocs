import type { ReactNode } from "react";

/** An empty list; `children` is what to do about it (a link or a button). */
export function EmptyState({
  label = "NO DATA AVAILABLE",
  children,
}: {
  label?: string;
  children?: ReactNode;
}) {
  return (
    <div className="border border-border-subtle border-dashed py-10 px-4 text-center label-sm text-text-tertiary">
      <div>{label}</div>
      {children && <div className="mt-3">{children}</div>}
    </div>
  );
}

export function LoadingLine({ label = "LOADING" }: { label?: string }) {
  return <div className="label-sm text-text-secondary blink-cursor">{label}...</div>;
}

/** A failure, said in place. With `onRetry`, a button runs the request again. */
export function ErrorBanner({
  message,
  onRetry,
  retrying = false,
}: {
  message: string;
  onRetry?: () => void;
  retrying?: boolean;
}) {
  return (
    <div
      role="alert"
      className="border border-error/40 bg-error/5 px-4 py-2 label-sm text-error mb-4 flex flex-wrap items-center gap-x-3 gap-y-1"
    >
      <span className="flex-1 min-w-0 break-words">ERROR: {message}</span>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          disabled={retrying}
          className="label-sm text-text-primary hover:text-primary-blue disabled:text-text-tertiary"
        >
          {retrying ? "RETRYING..." : "↻ RETRY"}
        </button>
      )}
    </div>
  );
}

// The call to action inside an EmptyState.
export const emptyActionCls = "label-sm text-text-primary hover:text-primary-blue";

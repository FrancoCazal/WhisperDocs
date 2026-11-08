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

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="border border-error/40 bg-error/5 px-4 py-2 label-sm text-error mb-4">
      ERROR: {message} · RETRY ↻
    </div>
  );
}

// The call to action inside an EmptyState.
export const emptyActionCls = "label-sm text-text-primary hover:text-primary-blue";

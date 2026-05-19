export function EmptyState({ label = "NO DATA AVAILABLE" }: { label?: string }) {
  return (
    <div className="border border-border-subtle border-dashed py-10 text-center label-sm text-text-tertiary">
      {label}
    </div>
  );
}

export function LoadingLine({ label = "LOADING" }: { label?: string }) {
  return (
    <div className="label-sm text-text-secondary blink-cursor">{label}...</div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="border border-error/40 bg-error/5 px-4 py-2 label-sm text-error mb-4">
      ERROR: {message} · RETRY ↻
    </div>
  );
}

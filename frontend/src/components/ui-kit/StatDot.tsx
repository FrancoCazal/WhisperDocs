export function StatDot({ ok = true }: { ok?: boolean }) {
  return (
    <span
      className="inline-block w-2 h-2"
      style={{ backgroundColor: ok ? "var(--success)" : "var(--error)" }}
    />
  );
}

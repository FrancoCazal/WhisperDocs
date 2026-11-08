import { fmtNumber } from "@/lib/api";

/** "4,812 / 5,000" once a text field nears its limit; nothing before. */
export function LengthHint({ length, max }: { length: number; max: number }) {
  if (length < max * 0.9) return null;
  return (
    <span
      aria-live="polite"
      className={`label-xs tabular-nums shrink-0 ${length >= max ? "text-error" : "text-text-tertiary"}`}
    >
      {fmtNumber(length)} / {fmtNumber(max)}
    </span>
  );
}

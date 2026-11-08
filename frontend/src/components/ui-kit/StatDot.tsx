export type StatTone = "success" | "warning" | "error";

const TONE_BG: Record<StatTone, string> = {
  success: "bg-success",
  warning: "bg-warning",
  error: "bg-error",
};

/** A status square. With `label` it says what it means to a screen reader;
 *  without it, it sits next to text that does, and is hidden from one. */
export function StatDot({ tone = "success", label }: { tone?: StatTone; label?: string }) {
  return (
    <span
      className={`inline-block w-2 h-2 ${TONE_BG[tone]}`}
      {...(label ? { role: "img", "aria-label": label, title: label } : { "aria-hidden": true })}
    />
  );
}

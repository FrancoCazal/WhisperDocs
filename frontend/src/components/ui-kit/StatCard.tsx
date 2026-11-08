import type { ReactNode } from "react";

export function StatCard({
  label,
  value,
  valueClass,
  trend,
  trendOk,
}: {
  label: string;
  value: ReactNode;
  valueClass?: string;
  trend?: string;
  trendOk?: boolean;
}) {
  return (
    <div className="border border-border-subtle bg-surface-container p-5 flex flex-col gap-3">
      <div className="label-sm text-text-secondary">{label}</div>
      <div className={`display-md ${valueClass ?? "text-text-primary"}`}>{value}</div>
      {trend && (
        <div className={`label-sm ${trendOk ? "text-success" : "text-text-secondary"}`}>
          {trend}
        </div>
      )}
    </div>
  );
}

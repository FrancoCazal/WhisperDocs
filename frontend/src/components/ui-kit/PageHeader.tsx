import type { ReactNode } from "react";

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between mb-8 pb-4 border-b border-border-subtle">
      <div>
        <h1 className="title-lg text-text-primary">{title}</h1>
        {subtitle && (
          <div className="label-sm text-text-secondary mt-2">{subtitle}</div>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

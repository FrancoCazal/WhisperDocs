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
    // The actions go under the title when they do not fit beside it.
    <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-3 mb-8 pb-4 border-b border-border-subtle">
      <div className="min-w-0">
        <h1 className="title-lg text-text-primary">{title}</h1>
        {subtitle && <div className="label-sm text-text-secondary mt-2">{subtitle}</div>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

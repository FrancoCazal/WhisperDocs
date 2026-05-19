import { useEffect, type ReactNode } from "react";

export function Modal({
  open,
  onClose,
  title,
  children,
  width = "max-w-lg",
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  width?: string;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 px-4"
      onClick={onClose}
    >
      <div
        className={`${width} w-full bg-surface-high border border-border-subtle`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 h-12 border-b border-border-subtle">
          <span className="label-sm text-text-primary">{title}</span>
          <button
            onClick={onClose}
            className="label-sm text-text-secondary hover:text-text-primary"
          >
            ✕ CLOSE
          </button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

export function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="block mb-4">
      <span className="label-sm text-text-secondary block mb-2">{label}</span>
      {children}
    </label>
  );
}

export const inputCls =
  "w-full bg-surface-lowest border-0 border-b border-border-subtle px-3 h-9 text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-border-focus body-sm";

export const textareaCls =
  "w-full bg-surface-lowest border border-border-subtle p-3 text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-border-focus body-sm font-mono";

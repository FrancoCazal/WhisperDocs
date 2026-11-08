import type { ButtonHTMLAttributes, ReactNode } from "react";

// Slate Protocol buttons: rectangular, mono, uppercase, and hovering inverts
// fill and outline. Two heights: sm for toolbars and rows, md for a page's or
// a dialog's actions; icon is md's square. (A link in running text is not a
// button: it has no box, and turns blue under the pointer.)
export type BtnVariant =
  | "primary" // cream fill
  | "secondary" // cream outline
  | "quiet" // hairline outline, for toolbars
  | "ghost" // no outline, for rows
  | "destructive" // error outline
  | "destructive-ghost"; // error text, for rows
export type BtnSize = "sm" | "md" | "icon";

const BASE =
  "inline-flex items-center justify-center gap-1 shrink-0 label-sm border rounded-none shadow-none transition-colors select-none disabled:opacity-40 disabled:cursor-not-allowed";

const SIZES: Record<BtnSize, string> = {
  sm: "h-7 px-2",
  md: "h-9 px-4",
  icon: "h-9 w-9",
};

// `not-disabled:` keeps a disabled button as it is under the pointer; a link
// is never :disabled, so the same classes style links that look like buttons.
const VARIANTS: Record<BtnVariant, string> = {
  primary:
    "bg-text-primary text-surface border-text-primary not-disabled:hover:bg-transparent not-disabled:hover:text-text-primary",
  secondary:
    "bg-transparent border-text-primary text-text-primary not-disabled:hover:bg-text-primary not-disabled:hover:text-surface",
  quiet:
    "bg-transparent border-border-subtle text-text-primary not-disabled:hover:border-text-primary not-disabled:hover:bg-text-primary not-disabled:hover:text-surface",
  ghost:
    "bg-transparent border-transparent text-text-primary not-disabled:hover:bg-text-primary not-disabled:hover:text-surface",
  destructive:
    "bg-transparent border-error/60 text-error not-disabled:hover:border-error not-disabled:hover:bg-error not-disabled:hover:text-surface",
  "destructive-ghost":
    "bg-transparent border-transparent text-error not-disabled:hover:bg-error not-disabled:hover:text-surface",
};

/** A button's classes, for an element that is not a <button>: a link, or a
 *  dialog's action. */
export function btnCls(variant: BtnVariant = "secondary", size: BtnSize = "md", extra = "") {
  return `${BASE} ${SIZES[size]} ${VARIANTS[variant]} ${extra}`;
}

export function Btn({
  variant = "secondary",
  size = "md",
  className = "",
  type = "button",
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: BtnVariant;
  size?: BtnSize;
  children: ReactNode;
}) {
  return (
    <button {...rest} type={type} className={btnCls(variant, size, className)}>
      {children}
    </button>
  );
}

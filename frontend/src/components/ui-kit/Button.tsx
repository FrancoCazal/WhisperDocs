import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "destructive";

export function Btn({
  variant = "secondary",
  className = "",
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; children: ReactNode }) {
  const base =
    "inline-flex items-center justify-center h-9 px-4 label-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed select-none";
  const variants: Record<Variant, string> = {
    primary:
      "bg-primary-blue text-surface border border-primary-blue hover:bg-transparent hover:text-primary-blue",
    secondary:
      "border border-text-primary text-text-primary bg-transparent hover:border-primary-blue hover:text-primary-blue",
    destructive:
      "border border-error text-error bg-transparent hover:bg-error hover:text-surface",
  };
  return (
    <button {...rest} className={`${base} ${variants[variant]} ${className}`}>
      {children}
    </button>
  );
}

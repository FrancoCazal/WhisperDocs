import type { ReactNode } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

// One confirmation for destructive actions, instead of window.confirm(): a
// focus-trapped alertdialog in the console's style. Escape and CANCEL close it.
export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  children,
  confirmLabel = "CONFIRM",
  destructive = false,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  children?: ReactNode;
  confirmLabel?: string;
  destructive?: boolean;
  onConfirm: () => void;
}) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent className="max-w-md bg-surface-high border border-border-subtle shadow-none p-5 gap-5">
        <AlertDialogHeader className="text-left">
          <AlertDialogTitle className="label-sm text-text-primary font-medium">
            {title}
          </AlertDialogTitle>
          {children && (
            <AlertDialogDescription asChild>
              <div className="body-sm text-text-secondary">{children}</div>
            </AlertDialogDescription>
          )}
        </AlertDialogHeader>
        <AlertDialogFooter className="gap-2 sm:gap-3">
          <AlertDialogCancel className="label-sm h-8 px-4 bg-transparent border border-border-subtle text-text-secondary hover:bg-surface-container hover:text-text-primary mt-0">
            CANCEL
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={onConfirm}
            className={
              destructive
                ? "label-sm h-8 px-4 bg-transparent border border-error/60 text-error hover:bg-error hover:text-surface"
                : "label-sm h-8 px-4 bg-transparent border border-text-primary text-text-primary hover:bg-text-primary hover:text-surface"
            }
          >
            {confirmLabel}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

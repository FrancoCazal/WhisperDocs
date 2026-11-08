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
import { btnCls } from "@/components/ui-kit/Button";
import { useReturnFocus } from "@/components/ui-kit/returnFocus";

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
  const returnFocus = useReturnFocus(open);
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent
        onCloseAutoFocus={returnFocus}
        className="max-w-md bg-surface-high border border-border-subtle shadow-none p-5 gap-5"
      >
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
          <AlertDialogCancel className={btnCls("quiet")}>CANCEL</AlertDialogCancel>
          <AlertDialogAction
            onClick={onConfirm}
            className={btnCls(destructive ? "destructive" : "primary")}
          >
            {confirmLabel}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

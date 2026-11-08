import * as DialogPrimitive from "@radix-ui/react-dialog";
import type { ReactNode } from "react";
import { useReturnFocus } from "@/components/ui-kit/returnFocus";

/** A dialog with a form or a task: Radix's, so it is a labelled role="dialog"
 *  that traps the focus and gives it back on close. Escape, a click outside
 *  and ✕ CLOSE close it. */
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
  const returnFocus = useReturnFocus(open);
  return (
    <DialogPrimitive.Root open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/60" />
        {/* No description: the title and the form say what it is for. */}
        <DialogPrimitive.Content
          aria-describedby={undefined}
          onCloseAutoFocus={returnFocus}
          className={`fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2 w-[calc(100%-2rem)] ${width} max-h-[calc(100dvh-2rem)] flex flex-col bg-surface-high border border-border-subtle focus:outline-none`}
        >
          <div className="flex items-center px-5 h-12 border-b border-border-subtle shrink-0 pr-28">
            <DialogPrimitive.Title className="label-sm text-text-primary truncate">
              {title}
            </DialogPrimitive.Title>
          </div>
          <div className="p-5 overflow-auto">{children}</div>
          {/* After the body, so the focus starts on the form, not on CLOSE. */}
          <DialogPrimitive.Close className="absolute top-0 right-5 h-12 label-sm text-text-secondary hover:text-text-primary">
            ✕ CLOSE
          </DialogPrimitive.Close>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
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

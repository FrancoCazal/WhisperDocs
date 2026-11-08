import { useState } from "react";

/**
 * A dialog's onCloseAutoFocus that gives the focus back to what had it when
 * the dialog opened. Radix gives it to a Dialog.Trigger, and these dialogs are
 * opened by their caller's own buttons, so without this the focus is lost.
 */
export function useReturnFocus(open: boolean) {
  const [returnTo, setReturnTo] = useState<HTMLElement | null>(null);
  const [wasOpen, setWasOpen] = useState(open);
  // Noted while rendering the open: the focus has not moved into the dialog yet.
  if (open !== wasOpen) {
    setWasOpen(open);
    if (open && typeof document !== "undefined") {
      const el = document.activeElement;
      setReturnTo(el instanceof HTMLElement && el !== document.body ? el : null);
    }
  }
  return (e: Event) => {
    e.preventDefault();
    if (returnTo?.isConnected) returnTo.focus();
  };
}

import { useId, type ReactNode, type Ref } from "react";
import { LengthHint } from "./LengthHint";

/** A message box that grows with its text (up to ten lines, then scrolls).
 *  Enter sends and Shift+Enter adds a line; Enter while an IME is composing
 *  only confirms the composition. `children` are the send/stop buttons. */
export function Composer({
  value,
  onChange,
  onSubmit,
  onEscape,
  maxLength,
  placeholder,
  label,
  hint = "ENTER SENDS · SHIFT+ENTER ADDS A LINE",
  disabled = false,
  textareaRef,
  children,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onEscape?: () => void;
  maxLength: number;
  placeholder: string;
  label: string;
  hint?: string;
  disabled?: boolean;
  textareaRef?: Ref<HTMLTextAreaElement>;
  children: ReactNode;
}) {
  const hintId = useId();
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
      className="max-w-4xl mx-auto"
    >
      <div className="flex items-end gap-3 border-b border-border-subtle focus-within:border-border-focus pb-2">
        <span aria-hidden="true" className="self-start py-2 text-text-primary body-sm">
          {">"}
        </span>
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") onEscape?.();
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              e.currentTarget.form?.requestSubmit();
            }
          }}
          disabled={disabled}
          maxLength={maxLength}
          placeholder={placeholder}
          aria-label={label}
          aria-describedby={hintId}
          className="flex-1 min-w-0 resize-none field-sizing-content max-h-[calc(10lh+1rem)] overflow-y-auto bg-transparent border-0 outline-none text-text-primary placeholder:text-text-tertiary placeholder:label-sm body-sm py-2 focus-visible:ring-0"
        />
        <LengthHint length={value.length} max={maxLength} />
        {children}
      </div>
      <div id={hintId} className="hidden sm:block mt-1 label-xs text-text-tertiary">
        {hint}
      </div>
    </form>
  );
}

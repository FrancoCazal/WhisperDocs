import { Toaster as Sonner } from "sonner";

type ToasterProps = React.ComponentProps<typeof Sonner>;

// The outcome of an action that leaves no other trace on screen (a delete, a
// save, a failed one). Slate Protocol: a tonal surface, no shadow, mono labels.
const Toaster = (props: ToasterProps) => {
  return (
    <Sonner
      theme="dark"
      position="bottom-right"
      // Above the 28 px status footer.
      offset={40}
      toastOptions={{
        unstyled: true,
        classNames: {
          // The type's colors go by its data-type attribute, which outranks
          // the base text color whatever the stylesheet's order.
          toast:
            "w-[356px] max-w-[calc(100vw-32px)] flex items-start gap-3 px-4 py-3 border border-border-subtle bg-surface-high text-text-primary label-sm data-[type=success]:border-success/50 data-[type=error]:border-error/60 data-[type=error]:text-error",
          title: "label-sm break-words",
          description: "label-xs text-text-tertiary mt-1 break-words",
          success: "[&_[data-icon]]:text-success",
          icon: "mt-0.5",
          actionButton:
            "label-xs px-2 h-6 border border-text-primary text-text-primary hover:bg-text-primary hover:text-surface",
        },
      }}
      {...props}
    />
  );
};

export { Toaster };

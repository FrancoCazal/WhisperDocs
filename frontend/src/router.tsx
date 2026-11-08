import { createRouter, useRouter, type ErrorComponentProps } from "@tanstack/react-router";
import { QueryClient } from "@tanstack/react-query";
import { routeTree } from "./routeTree.gen";
import { Btn, btnCls } from "@/components/ui-kit/Button";

// A route can throw anything, not only an Error.
function DefaultErrorComponent({ error, reset }: ErrorComponentProps) {
  const router = useRouter();
  return (
    <div className="min-h-screen flex items-center justify-center bg-surface px-6">
      <div className="max-w-md w-full border border-error/40 bg-surface-container p-6">
        <div className="label-sm text-error mb-3">SYSTEM_FAULT // UNHANDLED_EXCEPTION</div>
        <pre className="body-sm text-text-primary whitespace-pre-wrap break-words mb-6">
          {error instanceof Error ? error.message : String(error)}
        </pre>
        <div className="flex gap-3">
          <Btn
            variant="primary"
            onClick={() => {
              router.invalidate();
              reset();
            }}
          >
            RETRY ↻
          </Btn>
          <a href="/" className={btnCls("secondary")}>
            RETURN HOME
          </a>
        </div>
      </div>
    </div>
  );
}

export const getRouter = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 30_000,
        refetchOnWindowFocus: false,
        retry: 1,
      },
    },
  });

  const router = createRouter({
    routeTree,
    context: { queryClient },
    scrollRestoration: true,
    // A link's page starts loading when the pointer rests on it or it takes
    // focus. The routes have no loaders: this fetches their code only.
    defaultPreload: "intent",
    defaultPreloadStaleTime: 0,
    defaultErrorComponent: DefaultErrorComponent,
  });

  return router;
};

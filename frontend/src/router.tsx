import { createRouter, useRouter } from "@tanstack/react-router";
import { QueryClient } from "@tanstack/react-query";
import { routeTree } from "./routeTree.gen";

function DefaultErrorComponent({ error, reset }: { error: Error; reset: () => void }) {
  const router = useRouter();
  return (
    <div className="min-h-screen flex items-center justify-center bg-surface px-6">
      <div className="max-w-md w-full border border-error/40 bg-surface-container p-6">
        <div className="label-sm text-error mb-3">SYSTEM_FAULT // UNHANDLED_EXCEPTION</div>
        <pre className="body-sm text-text-primary whitespace-pre-wrap break-words mb-6">
          {error.message}
        </pre>
        <div className="flex gap-3">
          <button
            onClick={() => { router.invalidate(); reset(); }}
            className="h-9 px-4 label-sm bg-text-primary text-surface hover:bg-transparent hover:text-text-primary border border-text-primary"
          >
            RETRY ↻
          </button>
          <a
            href="/"
            className="h-9 px-4 label-sm border border-text-primary text-text-primary inline-flex items-center hover:bg-text-primary hover:text-surface"
          >
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
    defaultPreloadStaleTime: 0,
    defaultErrorComponent: DefaultErrorComponent,
  });

  return router;
};

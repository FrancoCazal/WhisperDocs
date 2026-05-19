import {
  Outlet,
  Link,
  createRootRouteWithContext,
  HeadContent,
  Scripts,
} from "@tanstack/react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Shell } from "@/components/layout/Shell";

import appCss from "../styles.css?url";

function NotFoundComponent() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-surface px-6">
      <div className="max-w-md text-center border border-border-subtle p-8 bg-surface-container">
        <div className="display-md text-text-primary">404</div>
        <div className="label-sm text-error mt-2">PATH_NOT_FOUND</div>
        <p className="body-sm text-text-secondary mt-4">
          The requested route is not registered in the protocol map.
        </p>
        <div className="mt-6">
          <Link
            to="/"
            className="inline-flex items-center justify-center h-9 px-4 label-sm border border-text-primary text-text-primary hover:bg-text-primary hover:text-surface"
          >
            RETURN HOME
          </Link>
        </div>
      </div>
    </div>
  );
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: "WhisperDocs — Backend Engineer's AI Reference" },
      { name: "description", content: "WhisperDocs RAG operations console — grounded answers with citations." },
      { name: "author", content: "WhisperDocs" },
      { property: "og:title", content: "WhisperDocs Console" },
      { property: "og:description", content: "RAG operations console for backend & AI engineers." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
    links: [
      { rel: "stylesheet", href: appCss },
    ],
  }),
  shellComponent: RootShell,
  component: RootComponent,
  notFoundComponent: NotFoundComponent,
});

function RootShell({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body>
        {children}
        <Scripts />
      </body>
    </html>
  );
}

function RootComponent() {
  // queryClient was attached in router context (router.tsx)
  const { queryClient } = Route.useRouteContext();
  return (
    <QueryClientProvider client={queryClient}>
      <Shell />
      {/* Outlet rendered inside Shell */}
      <HiddenOutlet />
    </QueryClientProvider>
  );
}

// Outlet is rendered by the Shell (inside <main>); nothing extra needed here.
function HiddenOutlet() {
  return null;
}

// Re-export Outlet for any direct consumers (kept for tree-shaking safety)
export { Outlet };

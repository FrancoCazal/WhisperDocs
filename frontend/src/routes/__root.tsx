import { createRootRouteWithContext, HeadContent, Scripts } from "@tanstack/react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Shell } from "@/components/layout/Shell";
import { Toaster } from "@/components/ui/sonner";
import { NotFoundPanel } from "@/components/ui-kit/NotFound";

import appCss from "../styles.css?url";

// For a notFound() thrown by a route without its own; a path no route takes
// is the "/$" page, inside the frame.
function NotFoundComponent() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-surface px-6">
      <NotFoundPanel />
    </div>
  );
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: "WhisperDocs — Backend Engineer's AI Reference" },
      {
        name: "description",
        content: "WhisperDocs RAG operations console — grounded answers with citations.",
      },
      { name: "author", content: "WhisperDocs" },
      { property: "og:title", content: "WhisperDocs Console" },
      { property: "og:description", content: "RAG operations console for backend & AI engineers." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
    links: [{ rel: "stylesheet", href: appCss }],
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
      {/* Extensions add attributes to <body> before hydration (e.g. cz-shortcut-listen). */}
      <body suppressHydrationWarning>
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
      {/* The Shell renders the route's Outlet inside <main>. */}
      <Shell />
      <Toaster />
    </QueryClientProvider>
  );
}

import { createFileRoute } from "@tanstack/react-router";
import { NotFoundPanel } from "@/components/ui-kit/NotFound";

// Any path no other route takes, as a page inside the console's frame. The
// root's notFoundComponent replaces the whole frame, which the SPA shell
// holds, so hydrating it failed; a page hydrates against the shell like the
// rest.
export const Route = createFileRoute("/$")({
  head: () => ({
    meta: [{ title: "Not Found — WhisperDocs" }],
  }),
  component: NotFoundPanel,
});

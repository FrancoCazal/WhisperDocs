import { createFileRoute, redirect } from "@tanstack/react-router";
import { NotFoundPanel } from "@/components/ui-kit/NotFound";

// The pages' paths were Spanish until F-6: a bookmark or a saved link to one
// opens the page under its new path.
const RENAMED: Record<string, string> = {
  colecciones: "collections",
  consulta: "query",
  documentos: "documents",
  evaluacion: "evaluation",
  historial: "history",
};

// Any path no other route takes, as a page inside the console's frame. The
// root's notFoundComponent replaces the whole frame, which the SPA shell
// holds, so hydrating it failed; a page hydrates against the shell like the
// rest.
export const Route = createFileRoute("/$")({
  beforeLoad: ({ location }) => {
    const [, first = "", ...rest] = location.pathname.split("/");
    const renamed = RENAMED[first];
    if (renamed) {
      throw redirect({
        href: `/${[renamed, ...rest].join("/")}${location.searchStr}`,
        replace: true,
      });
    }
  },
  head: () => ({
    meta: [{ title: "Not Found — WhisperDocs" }],
  }),
  component: NotFoundPanel,
});

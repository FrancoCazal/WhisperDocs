import { Link } from "@tanstack/react-router";
import { COSINE_HINT, fmtCosine, type AgentCitation } from "@/lib/api";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

// One answer's sources; `focus` is the source an "[n]" click asked for, and
// `cited` the ones the answer cites.
export type SourcesView = {
  citations: AgentCitation[];
  focus: number | null;
  label: string;
  cited: ReadonlySet<number>;
};

function scrollIntoView(el: HTMLElement | null) {
  el?.scrollIntoView({ block: "center" });
}

export function SourcesSheet({ view, onClose }: { view: SourcesView | null; onClose: () => void }) {
  return (
    <Sheet open={view !== null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent
        side="right"
        className="w-full sm:w-[480px] sm:max-w-[480px] bg-surface-low border-l border-border-subtle overflow-auto"
      >
        <SheetHeader>
          <SheetTitle className="label-sm text-text-primary">SOURCES · {view?.label}</SheetTitle>
          <SheetDescription className="label-xs text-text-tertiary normal-case">
            Ranked by retrieval: [n] in the answer is source n. The answer cites{" "}
            {view?.cited.size ?? 0} of the {view?.citations.length ?? 0} retrieved.
          </SheetDescription>
        </SheetHeader>
        <ol className="mt-4 space-y-3">
          {view?.citations.map((c, idx) => {
            const n = idx + 1;
            const focused = n === view.focus;
            const cited = view.cited.has(n);
            return (
              <li
                key={`src-${c.chunk_id}-${idx}`}
                ref={focused ? scrollIntoView : undefined}
                aria-current={focused || undefined}
                className={`border p-3 ${
                  focused
                    ? "border-primary-blue bg-surface-container"
                    : "border-border-subtle bg-surface-container-low"
                }`}
              >
                <div className="label-sm flex items-center justify-between gap-3 mb-2">
                  <span className="text-text-secondary truncate">
                    <span className="text-text-primary">[{n}]</span> {c.document_title}
                  </span>
                  <span className="tabular-nums shrink-0 text-text-secondary" title={COSINE_HINT}>
                    {fmtCosine(c.cosine)}
                  </span>
                </div>
                <div className="label-xs text-text-tertiary mb-2">
                  CHUNK {c.chunk_index} ·{" "}
                  <span className={cited ? "text-primary-blue" : undefined}>
                    {cited ? "CITED" : "NOT CITED"}
                  </span>
                </div>
                {c.text_preview && (
                  <div className="body-sm text-text-secondary whitespace-pre-wrap">
                    {c.text_preview}
                  </div>
                )}
                {/* A new tab, so the conversation stays as it is. */}
                <Link
                  to="/documentos/$id"
                  params={{ id: String(c.document_id) }}
                  search={{ chunk: c.chunk_index }}
                  target="_blank"
                  className="mt-3 inline-block label-xs text-text-secondary hover:text-primary-blue"
                >
                  OPEN IN DOCUMENT ↗
                </Link>
              </li>
            );
          })}
        </ol>
      </SheetContent>
    </Sheet>
  );
}

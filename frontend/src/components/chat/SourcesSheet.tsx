import type { AgentCitation } from "@/lib/api";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";

export function SourcesSheet({ citations }: { citations: AgentCitation[] }) {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <button className="label-sm text-text-primary hover:bg-text-primary hover:text-surface px-2 h-7 border border-border-subtle">
          ☰ SOURCES [{citations.length}]
        </button>
      </SheetTrigger>
      <SheetContent
        side="right"
        className="w-full sm:w-[480px] sm:max-w-[480px] bg-surface-low border-l border-border-subtle overflow-auto"
      >
        <SheetHeader>
          <SheetTitle className="label-sm text-text-primary">
            RETRIEVED CHUNKS · LAST TURN
          </SheetTitle>
        </SheetHeader>
        <div className="mt-4 space-y-3">
          {[...citations]
            .sort((a, b) => b.relevance - a.relevance)
            .map((c, idx) => {
              const rel = c.relevance;
              const relClass =
                rel >= 0.7 ? "text-success" : rel >= 0.5 ? "text-text-tertiary" : "text-error";
              return (
                <div
                  key={`src-${c.chunk_id}-${idx}`}
                  className="border border-border-subtle bg-surface-container-low p-3"
                >
                  <div className="label-sm flex items-center justify-between mb-2">
                    <span className="text-text-secondary truncate">{c.document_title}</span>
                    <span className={`tabular-nums shrink-0 ${relClass}`}>{rel.toFixed(3)}</span>
                  </div>
                  <div className="label-xs text-text-tertiary mb-2">CHUNK {c.chunk_index}</div>
                  {c.text_preview && (
                    <div className="body-sm text-text-secondary whitespace-pre-wrap">
                      {c.text_preview}
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      </SheetContent>
    </Sheet>
  );
}

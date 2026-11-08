import { COSINE_HINT, fmtCosine, type AgentCitation } from "@/lib/api";
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
          {/* In citation order, so [n] in the answer is the n-th source. */}
          {citations.map((c, idx) => {
            return (
              <div
                key={`src-${c.chunk_id}-${idx}`}
                className="border border-border-subtle bg-surface-container-low p-3"
              >
                <div className="label-sm flex items-center justify-between gap-3 mb-2">
                  <span className="text-text-secondary truncate">
                    <span className="text-text-primary">[{idx + 1}]</span> {c.document_title}
                  </span>
                  <span className="tabular-nums shrink-0 text-text-secondary" title={COSINE_HINT}>
                    {fmtCosine(c.cosine)}
                  </span>
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

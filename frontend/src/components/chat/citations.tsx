import type { ReactNode } from "react";
import { fmtCosine, type AgentCitation } from "@/lib/api";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { ChunkPlace } from "@/components/ui-kit/ChunkPlace";

const CITATION_RE = /\[(\d+)\]/g;

/** The sources an answer cites: each n of an "[n]" in its text that names one. */
export function citedSources(text: string, count: number): Set<number> {
  const cited = new Set<number>();
  for (const m of text.matchAll(CITATION_RE)) {
    const n = Number(m[1]);
    if (n >= 1 && n <= count) cited.add(n);
  }
  return cited;
}

// Replaces each "[n]" in the rendered text with a button on citation n: a
// hover shows its preview, a click calls `onCite(n)`.
export function interleaveCitations(
  children: ReactNode,
  citations: AgentCitation[],
  onCite?: (n: number) => void,
): ReactNode {
  const out: ReactNode[] = [];

  function processNode(node: ReactNode, keyPrefix: string) {
    if (typeof node === "string") {
      let lastIdx = 0;
      let match: RegExpExecArray | null;
      const re = new RegExp(CITATION_RE.source, "g");
      while ((match = re.exec(node)) !== null) {
        const n = Number(match[1]);
        if (n < 1 || n > citations.length) continue;
        if (match.index > lastIdx) {
          out.push(node.slice(lastIdx, match.index));
        }
        const c = citations[n - 1];
        out.push(
          <HoverCard key={`${keyPrefix}-cite-${n}-${out.length}`} openDelay={120}>
            <HoverCardTrigger asChild>
              <sup className="px-0.5">
                {/* leading-none: a sup's line-height is 0 (Tailwind's
                    preflight), which left the button no height to click,
                    hover or outline. */}
                <button
                  type="button"
                  onClick={() => onCite?.(n)}
                  aria-label={`Source ${n}: ${c.document_title}`}
                  className="leading-none text-primary-blue font-medium hover:underline focus-visible:outline focus-visible:outline-1 focus-visible:outline-primary-blue"
                >
                  [{n}]
                </button>
              </sup>
            </HoverCardTrigger>
            <HoverCardContent
              side="top"
              align="start"
              className="w-[460px] max-h-[300px] overflow-auto bg-surface-container border border-border-subtle"
            >
              <div className="label-xs text-text-tertiary mb-2">
                {c.document_title} · CHUNK {c.chunk_index}
                <ChunkPlace c={c} /> · {fmtCosine(c.cosine)}
              </div>
              <div className="body-sm text-text-secondary whitespace-pre-wrap">
                {c.text_preview ?? "(no preview)"}
              </div>
              {onCite && (
                <div className="label-xs text-text-tertiary mt-2">CLICK FOR ALL SOURCES</div>
              )}
            </HoverCardContent>
          </HoverCard>,
        );
        lastIdx = match.index + match[0].length;
      }
      if (lastIdx < node.length) out.push(node.slice(lastIdx));
    } else if (Array.isArray(node)) {
      node.forEach((child, i) => processNode(child, `${keyPrefix}-${i}`));
    } else {
      out.push(node);
    }
  }

  processNode(children, "root");
  return out.length === 0 ? children : out;
}

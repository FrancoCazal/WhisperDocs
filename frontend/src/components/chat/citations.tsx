import type { ReactNode } from "react";
import { fmtCosine, type AgentCitation } from "@/lib/api";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";

const CITATION_RE = /\[(\d+)\]/g;

// Replaces each "[n]" in the rendered text with a hover card on citation n.
export function interleaveCitations(children: ReactNode, citations: AgentCitation[]): ReactNode {
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
              <sup className="text-primary-blue cursor-help font-medium px-0.5">[{n}]</sup>
            </HoverCardTrigger>
            <HoverCardContent
              side="top"
              align="start"
              className="w-[460px] max-h-[300px] overflow-auto bg-surface-container border border-border-subtle"
            >
              <div className="label-xs text-text-tertiary mb-2">
                {c.document_title} · CHUNK {c.chunk_index} · {fmtCosine(c.cosine)}
              </div>
              <div className="body-sm text-text-secondary whitespace-pre-wrap">
                {c.text_preview ?? "(no preview)"}
              </div>
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

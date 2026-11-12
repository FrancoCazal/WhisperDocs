import type { ChunkLocation } from "@/lib/api";
import { fmtPages, fmtSection } from "@/lib/chunkLocation";

/** A chunk's pages and section, each after a " · ", for a line that already
 * names the chunk. Nothing when it has neither. The section keeps its case,
 * and its full text is the title. */
export function ChunkPlace({ c }: { c: ChunkLocation }) {
  const pages = fmtPages(c);
  const section = fmtSection(c);
  return (
    <>
      {pages && <> · {pages}</>}
      {section && (
        <>
          {" · "}
          <span className="normal-case" title={c.section ?? undefined}>
            {section}
          </span>
        </>
      )}
    </>
  );
}

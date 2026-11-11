import type { ChunkLocation } from "./api";

// A section label longer than this is cut on screen; the full one is its title.
const MAX_SECTION = 80;

/** "P. 3", or "PP. 3–5" for a chunk across pages; null without pages. */
export function fmtPages(c: ChunkLocation): string | null {
  const start = c.page_start ?? null;
  const end = c.page_end ?? start;
  if (start === null) return null;
  return end !== null && end !== start ? `PP. ${start}–${end}` : `P. ${start}`;
}

/** The section as the screen shows it, "§ Guide › Install", cut past
 * MAX_SECTION characters; null without one. */
export function fmtSection(c: ChunkLocation): string | null {
  const section = c.section?.trim();
  if (!section) return null;
  const shown = section.length > MAX_SECTION ? `${section.slice(0, MAX_SECTION - 1)}…` : section;
  return `§ ${shown}`;
}

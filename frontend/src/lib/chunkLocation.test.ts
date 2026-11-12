import { describe, expect, it } from "vitest";
import { fmtPages, fmtSection } from "./chunkLocation";

describe("where a chunk sits", () => {
  it("names a page, or the range a chunk spans", () => {
    expect(fmtPages({ page_start: 3, page_end: 3 })).toBe("P. 3");
    expect(fmtPages({ page_start: 3, page_end: 5 })).toBe("PP. 3–5");
    expect(fmtPages({ page_start: 4 })).toBe("P. 4");
    expect(fmtPages({ page_start: null, page_end: null })).toBeNull();
    // Citations stored before the fields existed have neither.
    expect(fmtPages({})).toBeNull();
  });

  it("shows the section as it is, cut when it is long", () => {
    expect(fmtSection({ section: "Guide › Install" })).toBe("§ Guide › Install");
    expect(fmtSection({ section: "  " })).toBeNull();
    expect(fmtSection({ section: null })).toBeNull();
    const long = fmtSection({ section: "x".repeat(200) });
    expect(long?.length).toBe(82);
    expect(long?.endsWith("…")).toBe(true);
  });
});

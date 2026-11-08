import { expect, test, type Page } from "@playwright/test";

// iOS zooms the page when a field under 16px takes focus: on a phone, every
// field is 16px. The backend is played with one collection and empty lists.

test.use({ viewport: { width: 390, height: 844 } });

const NOW = "2026-11-08T12:00:00Z";

async function playBackend(page: Page): Promise<string[]> {
  const unexpected: string[] = [];
  const page0 = { items: [], next_cursor: null, total: 0 };
  const answers: Record<string, unknown> = {
    "/health": { status: "ok", database: "ok" },
    "/admin/settings": {
      models: { search_docs: { model: "fake-model", temperature: 0 } },
      retrieval: { top_k: 5 },
      compaction: { token_threshold: 4000, count_threshold: 20, k_recent: 6 },
      embedding: { model: "fake-embedding", dimension: 1536 },
    },
    "/collections": [
      {
        id: 1,
        name: "Docs",
        description: null,
        created_at: NOW,
        document_count: 0,
        chunk_count: 0,
        total_tokens: 0,
      },
    ],
    "/conversations": page0,
    "/notes": [],
    "/query/history": page0,
    "/documents": page0,
    "/documents/jobs": [],
  };
  await page.route(/\/api\/v1\//, async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname.replace(/^.*\/api\/v1/, "");
    if (req.method() === "GET" && path in answers) return route.fulfill({ json: answers[path] });
    unexpected.push(`${req.method()} ${path}`);
    return route.fulfill({ status: 404, json: { detail: "not played" } });
  });
  return unexpected;
}

// The fields each page shows with the backend above, by accessible name.
const PAGES: [string, number][] = [
  ["/chat", 4],
  ["/history", 3],
  ["/documents", 2],
  ["/settings", 1],
];

for (const [path, count] of PAGES) {
  test(`the fields on ${path} are 16px on a phone`, async ({ page }) => {
    const unexpected = await playBackend(page);
    await page.goto(path);
    const fields = page.locator(
      "input:not([type=checkbox]):not([type=file]):not([type=hidden]), select, textarea",
    );
    await expect(fields).toHaveCount(count);
    const sizes = await fields.evaluateAll((els) =>
      els.map((el) => {
        const field = el as HTMLInputElement;
        const name = field.getAttribute("aria-label") ?? field.labels?.[0]?.textContent?.trim();
        return `${name || field.tagName}: ${getComputedStyle(field).fontSize}`;
      }),
    );
    expect(sizes.filter((s) => !s.endsWith(": 16px"))).toEqual([]);
    expect(unexpected).toEqual([]);
  });
}

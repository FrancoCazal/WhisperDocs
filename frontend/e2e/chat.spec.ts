import { expect, test, type Page } from "@playwright/test";

// The chat end to end in a browser: a question, the streamed answer, the
// thread in the URL, and a citation that opens its source. The backend is
// played below.

const THREAD = "5f0c2a8e-1d2b-4c3d-9e4f-a1b2c3d4e5f6";
const NOW = "2026-11-08T12:00:00Z";
const CITATION = {
  chunk_id: 31,
  document_id: 7,
  document_title: "rag-fundamentals.md",
  chunk_index: 2,
  relevance: 0.03,
  cosine: 0.82,
  text_preview: "RAG retrieves passages from your documents and answers from them.",
};
const ANSWER = "RAG answers from retrieved passages [1].";

function sse(events: [string, unknown][]): string {
  return events.map(([e, d]) => `event: ${e}\r\ndata: ${JSON.stringify(d)}\r\n\r\n`).join("");
}

const STREAM = sse([
  ["thread", { thread_id: THREAD }],
  ["node_start", { node: "classify" }],
  ["node_end", { node: "classify", summary: { action: "search_docs" } }],
  ["node_start", { node: "search_docs" }],
  ["token", { node: "search_docs", delta: "RAG answers from retrieved " }],
  ["token", { node: "search_docs", delta: "passages [1]." }],
  ["node_end", { node: "search_docs", summary: { chunks: 1 } }],
  [
    "final",
    {
      thread_id: THREAD,
      action: "search_docs",
      answer: ANSWER,
      citations: [CITATION],
      metadata: {
        model: "fake-model",
        tokens_input: 120,
        tokens_output: 12,
        latency_ms: 900,
        chunks_retrieved: 1,
        cost_usd: 0.0004,
      },
      failed: false,
    },
  ],
]);

/** Answers every API request the chat makes; returns the requests it did
 *  not expect, which a test then requires to be none. */
async function playBackend(page: Page): Promise<string[]> {
  const unexpected: string[] = [];
  const conversation = {
    id: 1,
    thread_id: THREAD,
    user_number: null,
    channel: "web",
    collection_id: null,
    title: "What is RAG?",
    is_saved: false,
    last_active_at: NOW,
    created_at: NOW,
  };
  // Any host: the API's URL comes from VITE_API_BASE_URL.
  await page.route(/\/api\/v1\//, async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname.replace(/^.*\/api\/v1/, "");
    const method = req.method();
    if (method === "POST" && path === "/agent/chat/stream") {
      expect(req.postDataJSON()).toMatchObject({ message: "What is RAG?", thread_id: null });
      return route.fulfill({ status: 200, contentType: "text/event-stream", body: STREAM });
    }
    const page0 = { items: [], next_cursor: null, total: 0 };
    const answers: Record<string, unknown> = {
      "/health": { status: "ok", database: "ok" },
      "/admin/settings": {
        models: { search_docs: { model: "fake-model", temperature: 0 } },
        retrieval: { top_k: 5 },
        compaction: { token_threshold: 4000, count_threshold: 20, k_recent: 6 },
        embedding: { model: "fake-embedding", dimension: 1536 },
      },
      "/collections": [],
      "/conversations": page0,
      [`/conversations/${THREAD}`]: conversation,
      "/notes": [],
      "/query/history": page0,
    };
    if (method === "GET" && path in answers) return route.fulfill({ json: answers[path] });
    unexpected.push(`${method} ${path}`);
    return route.fulfill({ status: 404, json: { detail: "not played" } });
  });
  return unexpected;
}

test("a question streams an answer whose citation opens its source", async ({ page }) => {
  const unexpected = await playBackend(page);
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/chat");

  const composer = page.getByRole("textbox", { name: "Message to agent" });
  await composer.fill("What is RAG?");
  await composer.press("Enter");

  await expect(page.getByText("RAG answers from retrieved passages")).toBeVisible();
  // The question is a turn of its own now, and the composer is free again.
  await expect(page.getByText(/USER ·/)).toBeVisible();
  await expect(composer).toHaveValue("");
  // The new thread takes the URL, so a reload reopens it.
  await expect(page).toHaveURL(new RegExp(`[?&]thread=${THREAD}`));

  const citation = page.getByRole("button", { name: "Source 1: rag-fundamentals.md" });
  await citation.hover();
  await expect(page.getByText("CLICK FOR ALL SOURCES")).toBeVisible();
  await citation.click();
  const sources = page.getByRole("dialog");
  await expect(sources).toContainText("rag-fundamentals.md");
  await expect(sources).toContainText("RAG retrieves passages from your documents");

  expect(unexpected).toEqual([]);
  expect(errors).toEqual([]);
});

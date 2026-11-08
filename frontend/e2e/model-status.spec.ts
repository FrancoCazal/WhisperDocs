import { expect, test, type Page } from "@playwright/test";

// The failing-models notice (BE-14): on every page while a model's last calls
// fail, every role's status in SETTINGS, and a chat turn that ends refetches
// it. The backend is played below; its record is what the API's calls left.

const THREAD = "5f0c2a8e-1d2b-4c3d-9e4f-a1b2c3d4e5f6";
const NOW = new Date().toISOString();
const GEMINI = "gemini-3.8-flash";

function role(state: "ok" | "failing" | "unknown", model = GEMINI) {
  return {
    model,
    state,
    calls: state === "unknown" ? 0 : 2,
    failures_in_a_row: state === "failing" ? 2 : 0,
    last_ok_at: state === "ok" ? NOW : null,
    last_failure_at: state === "failing" ? NOW : null,
    last_failure_kind: state === "failing" ? "billing" : null,
  };
}

/** The record: with `failing`, the classifier and the query rewrite fail
 *  over billing, as on a GCP project without it. */
function modelStatus(failing: boolean) {
  return {
    since: NOW,
    roles: {
      classifier: role(failing ? "failing" : "ok"),
      query_rewrite: role(failing ? "failing" : "ok"),
      rag_generation: role("ok", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
      clarification: role("unknown"),
      answer_direct: role("unknown"),
      summarize: role("unknown", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
      eval_judge: role("unknown"),
      embedding: role("ok", "text-embedding-3-small"),
    },
  };
}

const STREAM = [
  ["thread", { thread_id: THREAD }],
  [
    "final",
    {
      thread_id: THREAD,
      action: "search_docs",
      answer: "An answer from the documents.",
      citations: [],
      metadata: { model: "fake-model", tokens_input: 1, tokens_output: 1, latency_ms: 1 },
      failed: false,
      errors: [{ step: "classifier", kind: "billing", model: GEMINI }],
    },
  ],
]
  .map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`)
  .join("");

async function playBackend(page: Page, state: { failing: boolean }): Promise<string[]> {
  const unexpected: string[] = [];
  const page0 = { items: [], next_cursor: null, total: 0 };
  await page.route(/\/api\/v1\//, async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname.replace(/^.*\/api\/v1/, "");
    if (req.method() === "POST" && path === "/agent/chat/stream") {
      // The turn's classifier call fails: the record says so from now on.
      state.failing = true;
      return route.fulfill({ status: 200, contentType: "text/event-stream", body: STREAM });
    }
    const answers: Record<string, unknown> = {
      "/health": { status: "ok", checks: { database: "ok" } },
      "/admin/settings": {
        models: { classifier: { model: GEMINI, temperature: 0 } },
        retrieval: { top_k: 5 },
        compaction: { token_threshold: 4000, count_threshold: 20, k_recent: 6 },
        embedding: { model: "text-embedding-3-small", dimension: 1536 },
      },
      "/admin/model-status": modelStatus(state.failing),
      "/collections": [],
      "/conversations": page0,
      [`/conversations/${THREAD}`]: {
        id: 1,
        thread_id: THREAD,
        user_number: null,
        channel: "web",
        collection_id: null,
        title: "q",
        is_saved: false,
        last_active_at: NOW,
        created_at: NOW,
      },
      "/notes": [],
      "/query/history": page0,
      "/documents": page0,
      "/documents/jobs": [],
    };
    if (req.method() === "GET" && path in answers) return route.fulfill({ json: answers[path] });
    unexpected.push(`${req.method()} ${path}`);
    return route.fulfill({ status: 404, json: { detail: "not played" } });
  });
  return unexpected;
}

test("a failing model shows on every page, and each role's status in SETTINGS", async ({
  page,
}) => {
  const unexpected = await playBackend(page, { failing: true });
  await page.goto("/documents");

  const notice = page.getByRole("status", { name: "Failing models" });
  await expect(notice).toContainText("CLASSIFIER, QUERY REWRITE FAILING · BILLING OR QUOTA");
  await expect(notice).toContainText(GEMINI);
  await expect(notice).toContainText("THE CHAT SEARCHES THE DOCUMENTS INSTEAD");

  await notice.getByRole("link", { name: /EVERY ROLE'S STATUS/ }).click();
  await expect(page).toHaveURL(/\/settings$/);
  const status = page.locator("section", { hasText: "MODEL STATUS" });
  // A row: its label, exactly, and its value.
  const row = (label: string) =>
    status.locator("div.grid", { has: page.getByText(label, { exact: true }) });
  await expect(row("CLASSIFIER")).toContainText("FAILING · BILLING OR QUOTA · 2 IN A ROW");
  await expect(row("RAG_GENERATION")).toContainText("OK");
  await expect(row("SUMMARIZE")).toContainText("NO CALLS YET");
  expect(unexpected).toEqual([]);
});

test("a chat turn that ends refetches the status, without waiting for the poll", async ({
  page,
}) => {
  const unexpected = await playBackend(page, { failing: false });
  await page.goto("/chat");
  const notice = page.getByRole("status", { name: "Failing models" });
  await expect(page.getByRole("textbox", { name: "Message to agent" })).toBeVisible();
  await expect(notice).toHaveCount(0);

  const composer = page.getByRole("textbox", { name: "Message to agent" });
  await composer.fill("q");
  await composer.press("Enter");
  await expect(page.getByText("An answer from the documents.")).toBeVisible();

  // The poll is every 30 s, past this expect's 15 s.
  await expect(notice).toContainText("CLASSIFIER, QUERY REWRITE FAILING · BILLING OR QUOTA");
  expect(unexpected).toEqual([]);
});

import { expect, test } from "@playwright/test";

// The API key: a backend that runs with one (audit R6), played here, refuses
// the page until the key is set in SETTINGS, and takes it from then on.

const KEY = "test";

const ANSWERS: Record<string, unknown> = {
  "/admin/settings": {
    models: { search_docs: { model: "fake-model", temperature: 0 } },
    retrieval: { top_k: 5 },
    compaction: { token_threshold: 4000, count_threshold: 20, k_recent: 6 },
    embedding: { model: "fake-embedding", dimension: 1536 },
  },
  // The shell's last question.
  "/query/history": { items: [], next_cursor: null, total: 0 },
};

test("a key set in SETTINGS opens an API that asks for one", async ({ page }) => {
  const sent: (string | null)[] = [];
  const unexpected: string[] = [];
  await page.route(/\/api\/v1\//, async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname.replace(/^.*\/api\/v1/, "");
    // The probe needs no key.
    if (path === "/health")
      return route.fulfill({ json: { status: "ok", checks: { database: "ok" } } });
    if (req.method() !== "GET" || !(path in ANSWERS)) {
      unexpected.push(`${req.method()} ${path}`);
      return route.fulfill({ status: 404, json: { detail: "not played" } });
    }
    const key = await req.headerValue("x-api-key");
    sent.push(key);
    if (key !== KEY) {
      return route.fulfill({
        status: 401,
        headers: { "WWW-Authenticate": "ApiKey" },
        json: {
          detail: key ? "The API key is not valid." : "This API needs a key: send it as X-API-Key.",
          code: key ? "api_key_invalid" : "api_key_required",
          request_id: "r-1",
        },
      });
    }
    return route.fulfill({ json: ANSWERS[path] });
  });

  await page.goto("/settings");
  await expect(page.getByText("The API needs a key: set it in SETTINGS.")).toBeVisible();

  const field = page.getByRole("textbox", { name: "API key" });
  await field.fill("wrong");
  await page.getByRole("button", { name: "SAVE" }).click();
  await expect(page.getByText("The API refused the key: check it in SETTINGS.")).toBeVisible();

  await field.fill(KEY);
  await field.press("Enter");
  await expect(page.getByText("fake-model")).toBeVisible();
  // Never shown back: the field is empty, and says a key is set.
  await expect(field).toHaveValue("");
  await expect(field).toHaveAttribute("placeholder", /^SET/);

  // Kept across a reload.
  await page.reload();
  await expect(page.getByText("fake-model")).toBeVisible();

  await page.getByRole("button", { name: "CLEAR" }).click();
  await expect(page.getByText("The API needs a key: set it in SETTINGS.")).toBeVisible();
  expect(sent.filter((k) => k === KEY).length).toBeGreaterThanOrEqual(2);
  expect(unexpected).toEqual([]);
});

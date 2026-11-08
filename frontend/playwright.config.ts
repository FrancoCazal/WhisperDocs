import { defineConfig } from "@playwright/test";

// Browser tests against a backend that each test plays with page.route:
// nothing reaches a real API. They run a dev server of their own, on its own
// port, in the installed Chrome (channel "chrome"; GitHub's runners have it),
// so no browser is downloaded.
const PORT = 5199;

export default defineConfig({
  testDir: "e2e",
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  // The dev server compiles a page's code on its first request: on a cold
  // server, with tests in parallel, a page can take over the default 5 s.
  expect: { timeout: 15_000 },
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    channel: "chrome",
    trace: "retain-on-failure",
  },
  webServer: {
    command: `npm run dev -- --port ${PORT} --strictPort --host 127.0.0.1`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});

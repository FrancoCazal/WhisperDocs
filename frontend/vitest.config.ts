import { defineConfig } from "vitest/config";

// Its own config, not vite.config.ts: the app's plugins (TanStack Start, the
// route generator) have nothing to do in a unit test. The browser test is
// Playwright's (e2e/).
export default defineConfig({
  resolve: { tsconfigPaths: true },
  test: {
    include: ["src/**/*.test.ts"],
    environment: "node",
  },
});

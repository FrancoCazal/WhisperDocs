// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - tanstackStart, viteReact, tailwindcss, tsConfigPaths, cloudflare (build-only),
//     componentTagger (dev-only), VITE_* env injection, @ path alias, React/TanStack dedupe,
//     error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... } }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";

// The default build targets Cloudflare Workers (wrangler.jsonc). The Docker
// image builds for Node instead (BUILD_TARGET=node) and serves the result with
// `vite preview`, which runs dist/server/server.js.
export default defineConfig({
  cloudflare: process.env.BUILD_TARGET === "node" ? false : undefined,
});

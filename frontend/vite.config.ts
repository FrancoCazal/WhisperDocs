import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import viteReact from "@vitejs/plugin-react";

// SPA mode: the build prerenders only the root route, with the router's
// pending component where a page goes, into dist/client/_shell.html, and the
// Docker image serves dist/client as static files, answering every path that
// is not a file with that shell. The pages get their data in the browser
// anyway: the API's URL is the browser's localhost, which a server-side
// render inside a container could not reach. The dev server still renders on
// the server; it is on Vite's default port, 5173, and the container on 8080.
export default defineConfig({
  // The "@/..." imports, from tsconfig.json's paths.
  resolve: { tsconfigPaths: true },
  plugins: [
    tailwindcss(),
    // Before the React plugin: it generates the route tree and splits routes.
    tanstackStart({ spa: { enabled: true } }),
    viteReact(),
  ],
});

import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import viteReact from "@vitejs/plugin-react";

// TanStack Start builds a Node server (dist/server/server.js), which
// `vite preview` serves in the Docker image. The dev server is on Vite's
// default port, 5173: the container takes 8080.
export default defineConfig({
  // The "@/..." imports, from tsconfig.json's paths.
  resolve: { tsconfigPaths: true },
  plugins: [
    tailwindcss(),
    // Before the React plugin: it generates the route tree and splits routes.
    tanstackStart(),
    viteReact(),
  ],
});

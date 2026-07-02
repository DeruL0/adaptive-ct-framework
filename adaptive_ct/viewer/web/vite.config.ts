import { defineConfig } from "vite";

// The Python server hosts the built bundle from ./dist, so assets must be
// referenced with relative paths. During `npm run dev`, /api is proxied to the
// running `python -m adaptive_ct.viewer` instance.
export default defineConfig({
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    assetsDir: "assets",
    target: "es2022",
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
      "/healthz": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },
});

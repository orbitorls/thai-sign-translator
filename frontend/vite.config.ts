import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/translate": "http://127.0.0.1:8000",
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
      "/models": "http://127.0.0.1:8000",
      "/signs": "http://127.0.0.1:8000",
      "/supported-phrases": "http://127.0.0.1:8000",
      "/feedback": "http://127.0.0.1:8000",
      "/privacy": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/model-info": "http://127.0.0.1:8000",
      "/predict": "http://127.0.0.1:8000",
      "/static": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});

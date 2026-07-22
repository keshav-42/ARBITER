import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the API and WebSocket to the FastAPI backend on :8000, so the
// frontend can call same-origin paths without CORS gymnastics during development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});

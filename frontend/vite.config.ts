import { fileURLToPath, URL } from 'node:url'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  // Read .env from the repo root instead of frontend/, so the project has a
  // single env file. Only VITE_-prefixed vars are exposed to the client;
  // ANTHROPIC_API_KEY lives in the same file and never reaches the bundle.
  envDir: fileURLToPath(new URL('..', import.meta.url)),
  plugins: [react(), tailwindcss()],
  // strictPort, because the backend's CORS allowlist names this port exactly.
  // Vite's default is to move to the next free port when 5173 is taken, which
  // turns "something else is on 5173" into a browser Origin the API does not
  // trust -- and that surfaces as `OPTIONS /api/... 400` on every request, with
  // nothing in the log pointing at the port. Failing to start is the better
  // error: it names the actual problem at the moment it happens.
  server: { port: 5173, strictPort: true },
})

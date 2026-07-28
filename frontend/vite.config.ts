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
  server: { port: 5173 },
})

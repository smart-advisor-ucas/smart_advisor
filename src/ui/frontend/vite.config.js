import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // npm run dev: forward API calls to the locally running FastAPI backend
    // so the app can use same-origin (relative) URLs in every environment.
    proxy: {
      '/chat': 'http://127.0.0.1:8000',
      '/voice': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
})

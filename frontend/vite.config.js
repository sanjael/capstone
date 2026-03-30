import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/predict': 'http://localhost:8001',
      '/report': 'http://localhost:8001',
      '/analytics': 'http://localhost:8001',
      '/health': 'http://localhost:8001',
    }
  }
})

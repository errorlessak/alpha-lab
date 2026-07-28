import { tanstackRouter } from '@tanstack/router-plugin/vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

import { fontPreload } from './vite-plugin-font-preload'

// The API runs on loopback (`just serve`). Proxying in dev keeps the browser on
// one origin, so nothing here depends on CORS behaving the same way in
// development as it does in production.
const API_TARGET = process.env.ALPHA_API_URL ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [
    tanstackRouter({ target: 'react', autoCodeSplitting: true }),
    react(),
    tailwindcss(),
    fontPreload(),
  ],
  preview: {
    // Bind the literal loopback address. On Windows `localhost` resolves to ::1
    // first, and a suite whose baseURL is 127.0.0.1 then talks to nothing.
    host: '127.0.0.1',
    port: 4173,
    proxy: {
      '/api': { target: API_TARGET, changeOrigin: true },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        // Server-sent events die behind a buffering proxy, and the job console
        // is the one screen where that failure is invisible until you need it.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache, no-transform'
            }
          })
        },
      },
    },
  },
  build: {
    // §13.5 sets a 200 KB gzipped budget on initial JS. Reporting compressed
    // size on every build makes a regression visible in the build log rather
    // than in a Lighthouse run three days later.
    reportCompressedSize: true,
    // The budget check reads this to know which chunks the entry pulls in.
    manifest: true,
    rollupOptions: {
      output: {
        manualChunks: {
          // The charting libraries are only reached on routes that draw, and
          // route-level code splitting handles the rest.
          charts: ['uplot'],
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
})

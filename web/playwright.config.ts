import { defineConfig, devices } from '@playwright/test'

/**
 * The end-to-end suite (§13.5, §15 Phase 12 exit).
 *
 * Runs against the built app served by Vite preview, proxied to a real API on
 * loopback. Both are started here so `just e2e` is one command — a suite that
 * needs two terminals is a suite that gets skipped.
 *
 * The API must have at least one completed run in its store. `just verify`
 * produces one through `just smoke` before this ever runs.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  timeout: 60_000,
  expect: { timeout: 10_000 },

  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },

  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],

  webServer: [
    {
      // The venv interpreter directly, not `uv run`: `uv` is not on the default
      // PATH on this machine (HANDOFF §3), and a test suite that only starts
      // for people with the right shell profile is a test suite nobody runs.
      //
      // The path is relative to `cwd` below — the repository root — not to this
      // file. Writing it as `../.venv/...` looked right next to `cwd: '..'` and
      // resolved one directory above the repository; it went unnoticed for a
      // phase because `reuseExistingServer` found a server someone had left
      // running.
      command:
        process.platform === 'win32'
          ? '.venv\\Scripts\\python.exe -m uvicorn alphaservice.api.app:app --host 127.0.0.1 --port 8000'
          : '.venv/bin/python -m uvicorn alphaservice.api.app:app --host 127.0.0.1 --port 8000',
      cwd: '..',
      url: 'http://127.0.0.1:8000/api/v1/health',
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: 'npm run preview -- --port 4173 --strictPort',
      url: 'http://127.0.0.1:4173',
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
})

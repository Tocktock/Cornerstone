import { defineConfig } from '@playwright/test'

const databaseUrl =
  process.env.CORNERSTONE_BROWSER_TEST_DATABASE_URL ??
  'postgresql+psycopg://cornerstone:cornerstone@localhost:55432/cornerstone_test'
const backendPort = process.env.CORNERSTONE_BROWSER_BACKEND_PORT ?? '8001'
const frontendPort = process.env.CORNERSTONE_BROWSER_FRONTEND_PORT ?? '4174'
const frontendOrigin = `http://127.0.0.1:${frontendPort}`
const apiBaseUrl = `http://127.0.0.1:${backendPort}/api/v1`

export default defineConfig({
  testDir: './tests/symptoms',
  timeout: 30_000,
  retries: process.env.CI ? 2 : 0,
  use: {
    baseURL: frontendOrigin,
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      command: `zsh -lc "cd .. && . .venv/bin/activate && CORNERSTONE_DATABASE_URL=${databaseUrl} CORNERSTONE_RESET_DATABASE_ON_START=true CORNERSTONE_FIXED_NOW=2026-04-06T09:00:00+09:00 CORNERSTONE_CORS_ORIGINS='[\\\"${frontendOrigin}\\\"]' uvicorn cornerstone.main:app --host 127.0.0.1 --port ${backendPort}"`,
      url: `${apiBaseUrl}/health`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: `VITE_API_BASE_URL=${apiBaseUrl} npx vite --host 127.0.0.1 --port ${frontendPort}`,
      url: frontendOrigin,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
})

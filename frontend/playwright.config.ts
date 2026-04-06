import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { defineConfig } from '@playwright/test'

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const fixtureRoot = process.env.CORNERSTONE_FIXTURE_ROOT ?? path.join(repoRoot, 'backend', 'fixtures')
const workspaceSourceRoot =
  process.env.CORNERSTONE_WORKSPACE_SOURCE_ROOT ??
  path.join(fixtureRoot, 'minimal', 'workspace', 'member-visible')
const personalSourceRoot =
  process.env.CORNERSTONE_PERSONAL_SOURCE_ROOT ??
  path.join(fixtureRoot, 'minimal', 'personal', 'member-private')
const corpusSourceRoot =
  process.env.CORNERSTONE_CORPUS_SOURCE_ROOT ?? path.join(repoRoot, 'sample-data', 'sendy-knowledge')
const databaseUrl =
  process.env.CORNERSTONE_BROWSER_TEST_DATABASE_URL ??
  'postgresql+psycopg://cornerstone:cornerstone@localhost:55433/cornerstone_test'
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
      command: `zsh -lc "cd .. && . .venv/bin/activate && CORNERSTONE_DATABASE_URL=${databaseUrl} CORNERSTONE_FIXTURE_ROOT=${fixtureRoot} CORNERSTONE_WORKSPACE_SOURCE_ROOT=${workspaceSourceRoot} CORNERSTONE_PERSONAL_SOURCE_ROOT=${personalSourceRoot} CORNERSTONE_CORPUS_SOURCE_ROOT=${corpusSourceRoot} CORNERSTONE_RESET_DATABASE_ON_START=true CORNERSTONE_FIXED_NOW=2026-04-06T09:00:00+09:00 CORNERSTONE_CORS_ORIGINS='[\\\"${frontendOrigin}\\\"]' uvicorn cornerstone.main:app --host 127.0.0.1 --port ${backendPort}"`,
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

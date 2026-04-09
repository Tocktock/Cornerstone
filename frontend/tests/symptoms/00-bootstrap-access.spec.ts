import { expect, test } from '@playwright/test'

import {
  actorNamed,
  apiBaseUrl,
  bootstrapViewer,
  captureSnapshot,
  goToRoute,
  switchActor,
  waitForShell,
} from './testkit'

test('backend healthcheck is reachable and actor switching persists across reloads', async ({
  page,
  request,
}, testInfo) => {
  const healthResponse = await request.get(`${apiBaseUrl}/health`)
  expect(healthResponse.ok()).toBeTruthy()
  expect(await healthResponse.json()).toEqual({ status: 'ok' })

  const bootstrap = await bootstrapViewer(request)
  const reviewer = actorNamed(bootstrap, 'Domain Reviewer')
  const admin = actorNamed(bootstrap, 'Workspace Admin')

  await goToRoute(page, '/', 'Workspace overview')
  await expect(
    page.locator('.context-card').filter({ hasText: bootstrap.workspace.context_space_name }).first(),
  ).toBeVisible()

  await switchActor(page, reviewer)
  await goToRoute(page, '/review', 'Review queue')
  await expect(page.getByRole('heading', { name: 'Review access required' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Officialize' }).first()).toBeVisible()

  await switchActor(page, admin)
  await page.goto('/graph')
  await waitForShell(page)
  await expect(page.getByRole('heading', { name: 'Graph slice' })).toBeVisible()
  await expect(page.getByLabel('Switch actor')).toHaveValue(admin.actor_id)

  await page.reload()
  await waitForShell(page)
  await expect(page.getByLabel('Switch actor')).toHaveValue(admin.actor_id)

  await captureSnapshot(page, testInfo, 'bootstrap-access-admin-graph')
})

test('member hides review navigation and direct review visits show a friendly access state', async ({
  page,
  request,
}, testInfo) => {
  const bootstrap = await bootstrapViewer(request)
  const member = actorNamed(bootstrap, 'Member')

  await goToRoute(page, '/', 'Workspace overview')
  await switchActor(page, member)

  await expect(
    page.getByRole('navigation', { name: 'Primary navigation' }).getByText('Review'),
  ).toHaveCount(0)

  await page.goto('/review')
  await waitForShell(page)

  await expect(page.getByRole('heading', { name: 'Review access required' })).toBeVisible()
  await expect(page.getByText(/Use Switch actor to choose/i)).toBeVisible()
  await expect(page.locator('body')).not.toContainText('{"detail"')

  await captureSnapshot(page, testInfo, 'member-review-access-required')
})

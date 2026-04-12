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

  await goToRoute(page, '/', 'Workspace')
  await expect(page.getByRole('navigation', { name: 'Primary navigation' })).toContainText('Explore')

  await switchActor(page, reviewer)
  await goToRoute(page, '/review-studio', 'Review Studio')
  await expect(page.getByRole('heading', { name: 'Review access required' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Officialize' }).first()).toBeVisible()

  await switchActor(page, admin)
  await page.goto('/explore/map')
  await waitForShell(page)
  await expect(page.getByRole('heading', { name: 'Explore Map' })).toBeVisible()
  await expect(page.getByLabel('Switch actor')).toHaveValue(admin.actor_id)

  await page.reload()
  await waitForShell(page)
  await expect(page.getByLabel('Switch actor')).toHaveValue(admin.actor_id)

  await page.goto('/review')
  await waitForShell(page)
  await expect(page).toHaveURL(/\/review-studio$/)
  await expect(page.getByRole('heading', { name: 'Review Studio' })).toBeVisible()

  await captureSnapshot(page, testInfo, 'bootstrap-access-admin-review-studio')
})

test('member hides studio navigation and direct studio visits show a friendly access state', async ({
  page,
  request,
}, testInfo) => {
  const bootstrap = await bootstrapViewer(request)
  const member = actorNamed(bootstrap, 'Member')

  await goToRoute(page, '/', 'Workspace')
  await switchActor(page, member)

  const nav = page.getByRole('navigation', { name: 'Primary navigation' })
  await expect(nav.getByText('Review Studio')).toHaveCount(0)
  await expect(nav.getByText('Source Studio')).toHaveCount(0)

  await page.goto('/review-studio')
  await waitForShell(page)
  await expect(page.getByRole('heading', { name: 'Review access required' })).toBeVisible()
  await expect(page.getByText(/Use Switch actor to choose/i)).toBeVisible()
  await expect(page.locator('body')).not.toContainText('{"detail"')

  await page.goto('/glossary')
  await waitForShell(page)
  await expect(page).toHaveURL(/\/explore\/topics$/)
  await expect(page.getByRole('heading', { name: 'Explore Topics' })).toBeVisible()

  await captureSnapshot(page, testInfo, 'member-review-studio-access-required')
})

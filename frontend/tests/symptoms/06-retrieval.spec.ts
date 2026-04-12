import { expect, test } from '@playwright/test'

import {
  actorNamed,
  bootstrapViewer,
  captureSnapshot,
  goToRoute,
  switchActor,
} from './testkit'

test('workspace search renders structured answer and search result payloads on mobile', async ({
  page,
  request,
}, testInfo) => {
  const bootstrap = await bootstrapViewer(request)
  const reviewer = actorNamed(bootstrap, 'Domain Reviewer')

  await page.setViewportSize({ width: 390, height: 844 })
  await goToRoute(page, '/', 'Workspace')
  await switchActor(page, reviewer)

  const searchInput = page.getByPlaceholder('Search workspace context')
  const inputBox = await searchInput.boundingBox()
  expect(inputBox?.height ?? 0).toBeLessThan(90)

  await searchInput.fill('escalation')
  await page.getByRole('button', { name: 'Run query' }).click()

  await expect(page.getByText('Concepts').nth(0)).toBeVisible()
  await expect(page.getByRole('heading', { name: /\d+ matches/i })).toBeVisible()
  await expect(page.getByText(/verified/i).first()).toBeVisible()

  await captureSnapshot(page, testInfo, 'workspace-escalation-results')
})

test('workspace answers reflect scope-sensitive support disclosure and no-match states', async ({
  page,
  request,
}, testInfo) => {
  const bootstrap = await bootstrapViewer(request)
  const member = actorNamed(bootstrap, 'Member')
  const reviewer = actorNamed(bootstrap, 'Domain Reviewer')

  await goToRoute(page, '/', 'Workspace')
  await switchActor(page, member)

  const searchInput = page.getByPlaceholder('Search workspace context')
  await searchInput.fill('trigger')
  await page.getByRole('button', { name: 'Run query' }).click()

  await expect(page.locator('.reader-primary-panel')).toContainText('restricted support')
  await expect(page.locator('.provenance-strip')).toContainText('0 visible support')
  await expect(page.getByRole('heading', { name: /\d+ matches/i })).toBeVisible()

  await switchActor(page, reviewer)
  await searchInput.fill('trigger')
  await page.getByRole('button', { name: 'Run query' }).click()

  await expect(page.locator('.reader-primary-panel')).toContainText('source backed')
  await expect(page.locator('.provenance-strip')).toContainText('1 visible support')

  await searchInput.fill('totally-unmatched-query')
  await page.getByRole('button', { name: 'Run query' }).click()

  await expect(page.locator('.reader-primary-panel')).toContainText('no_official_match')
  await expect(page.locator('.reader-primary-panel')).toContainText('Try a specific concept or decision title.')

  await captureSnapshot(page, testInfo, 'workspace-trigger-and-no-match')
})

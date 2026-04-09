import { expect, test } from '@playwright/test'

import {
  actorNamed,
  bootstrapViewer,
  captureSnapshot,
  goToRoute,
  switchActor,
} from './testkit'

test('dashboard search renders structured answer and search result payloads on mobile', async ({
  page,
  request,
}, testInfo) => {
  const bootstrap = await bootstrapViewer(request)
  const reviewer = actorNamed(bootstrap, 'Domain Reviewer')

  await page.setViewportSize({ width: 390, height: 844 })
  await goToRoute(page, '/', 'Workspace overview')
  await switchActor(page, reviewer)

  const searchInput = page.getByPlaceholder('Search workspace context')
  const inputBox = await searchInput.boundingBox()
  expect(inputBox?.height ?? 0).toBeLessThan(90)

  await searchInput.fill('escalation')
  await page.getByRole('button', { name: 'Run search' }).click()

  await expect(page.getByText('Concepts').nth(0)).toBeVisible()
  await expect(page.getByText('Cited decisions')).toBeVisible()
  await expect(page.locator('.results-card h3')).toHaveText(/\d+ matches/i)
  await expect(page.getByText(/Verification: verified/i)).toBeVisible()

  await captureSnapshot(page, testInfo, 'dashboard-escalation-results')
})

test('dashboard answers reflect scope-sensitive support disclosure and no-match states', async ({
  page,
  request,
}, testInfo) => {
  const bootstrap = await bootstrapViewer(request)
  const member = actorNamed(bootstrap, 'Member')
  const reviewer = actorNamed(bootstrap, 'Domain Reviewer')

  await goToRoute(page, '/', 'Workspace overview')
  await switchActor(page, member)

  const searchInput = page.getByPlaceholder('Search workspace context')
  await searchInput.fill('trigger')
  await page.getByRole('button', { name: 'Run search' }).click()

  await expect(page.locator('.answer-card')).toContainText('restricted support')
  await expect(page.getByText(/Visible support items: 0/i)).toBeVisible()
  await expect(page.locator('.results-card h3')).toHaveText(/\d+ matches/i)

  await switchActor(page, reviewer)
  await searchInput.fill('trigger')
  await page.getByRole('button', { name: 'Run search' }).click()

  await expect(page.locator('.answer-card')).toContainText('source backed')
  await expect(page.getByText(/Visible support items: 1/i)).toBeVisible()

  await searchInput.fill('totally-unmatched-query')
  await page.getByRole('button', { name: 'Run search' }).click()

  await expect(page.locator('.answer-card').getByRole('heading', { name: 'no_official_match' })).toBeVisible()
  await expect(page.locator('.answer-card')).toContainText('Try a specific concept or decision title.')
  await expect(page.locator('.results-card').getByRole('heading', { name: 'no_official_match' })).toBeVisible()

  await captureSnapshot(page, testInfo, 'dashboard-trigger-and-no-match')
})

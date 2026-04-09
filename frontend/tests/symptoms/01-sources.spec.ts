import { expect, test, type Page } from '@playwright/test'

import {
  actorNamed,
  bootstrapViewer,
  captureSnapshot,
  goToRoute,
  switchActor,
  uniqueName,
} from './testkit'

test('operator can bind preview create and manage Notion connections from the sources page', async ({
  page,
  request,
}, testInfo) => {
  const bootstrap = await bootstrapViewer(request)
  const operator = actorNamed(bootstrap, 'Operator')
  const pageTreeLabel = uniqueName(testInfo, 'Page Tree source')
  const databaseLabel = uniqueName(testInfo, 'Database source')

  await goToRoute(page, '/sources', 'Source status')
  await switchActor(page, operator)

  await expect(page.getByText('Bind Notion and create a source connection')).toBeVisible()
  await expect(page.locator('.status-pill.stale').first()).toBeVisible()
  await expect(page.locator('.status-pill.degraded').first()).toBeVisible()
  await expect(page.locator('.status-pill.paused').first()).toBeVisible()
  await expect(page.locator('.status-pill.removed').first()).toBeVisible()
  await expect(page.locator('.status-pill.active').first()).toBeVisible()

  const firstTime = page.locator('time').first()
  await expect(firstTime).toBeVisible()
  await expect(firstTime).not.toContainText('T')
  const firstLocator = page.locator('.source-locator').first()
  const wrapsWithoutOverflow = await firstLocator.evaluate(
    (element) => element.scrollWidth <= element.clientWidth + 1,
  )
  expect(wrapsWithoutOverflow).toBeTruthy()

  await page.getByRole('button', { name: 'Bind Notion' }).click()
  await expect(page.getByText(/^Ready · /)).toBeVisible()

  await page.getByLabel('Template').selectOption('notion_shared_page_tree')
  await page.getByLabel('Source label').fill(pageTreeLabel)
  await page.getByLabel('Page URL or UUID').fill('11111111-1111-1111-1111-111111111111')
  await page.getByRole('button', { name: 'Preview' }).click()
  await expect(page.locator('.preview-panel')).toContainText('11111111-1111-1111-1111-111111111111')
  await expect(page.locator('.preview-list .list-card').first()).toBeVisible()

  await page.getByLabel('Template').selectOption('notion_shared_database')
  await page.getByLabel('Source label').fill(databaseLabel)
  await page.getByLabel('Database URL or UUID').fill('22222222-2222-2222-2222-222222222222')
  await page.getByRole('button', { name: 'Preview' }).click()
  await expect(page.locator('.preview-panel')).toContainText('22222222-2222-2222-2222-222222222222')
  await expect(page.locator('.preview-list .list-card').first()).toBeVisible()

  await page.getByRole('button', { name: 'Create connection' }).click()
  await expect(page.getByText('Created source connection and started the initial sync.')).toBeVisible()

  const createdCard = sourceCard(page, databaseLabel)
  await expect(createdCard).toBeVisible()

  await createdCard.getByRole('button', { name: 'Show runs' }).click()
  await expect(createdCard.getByText('Recent sync runs')).toBeVisible()
  await expect(createdCard.getByText('initial')).toBeVisible()
  await createdCard.getByRole('button', { name: 'Hide runs' }).click()

  await createdCard.getByRole('button', { name: 'Resync' }).click()
  await expect(page.getByText(`${databaseLabel} sync completed.`)).toBeVisible()
  await createdCard.getByRole('button', { name: 'Show runs' }).click()
  await expect(createdCard.getByText('manual')).toBeVisible()
  await createdCard.getByRole('button', { name: 'Hide runs' }).click()

  await createdCard.getByRole('button', { name: 'Pause' }).click()
  await expect(page.getByText(`${databaseLabel} pause completed.`)).toBeVisible()
  await expect(createdCard.locator('.status-pill.paused')).toBeVisible()

  await createdCard.getByRole('button', { name: 'Resume' }).click()
  await expect(page.getByText(`${databaseLabel} resume completed.`)).toBeVisible()
  await expect(createdCard.locator('.status-pill.active')).toBeVisible()

  await createdCard.getByRole('button', { name: 'Remove' }).click()
  await expect(page.getByText(`${databaseLabel} remove completed.`)).toBeVisible()
  await expect(createdCard.locator('.status-pill.removed')).toBeVisible()

  await captureSnapshot(page, testInfo, 'sources-operator-managed-connection')
})

test('member cannot see connector manager controls on the sources page', async ({
  page,
  request,
}, testInfo) => {
  const bootstrap = await bootstrapViewer(request)
  const member = actorNamed(bootstrap, 'Member')

  await goToRoute(page, '/sources', 'Source status')
  await switchActor(page, member)

  await expect(page.getByText('Bind Notion and create a source connection')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Bind Notion' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Resync' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Pause' })).toHaveCount(0)

  await captureSnapshot(page, testInfo, 'sources-member-readonly')
})

function sourceCard(page: Page, label: string) {
  return page.locator('.source-card').filter({ hasText: label })
}

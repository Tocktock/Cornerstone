import { expect, test, type Page } from '@playwright/test'

test('member hides review navigation and direct review visits show a friendly access state', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('navigation', { name: 'Primary navigation' }).getByText('Review')).toHaveCount(0)

  await page.goto('/review')

  await expect(page.getByRole('heading', { name: 'Review access required' })).toBeVisible()
  await expect(page.getByText(/Use Switch actor to choose/i)).toBeVisible()
  await expect(page.locator('body')).not.toContainText('{"detail"')
})

test('sources page shows current, stale, degraded, paused, and removed symptoms', async ({ page }) => {
  await page.goto('/sources')

  await expect(page.getByRole('heading', { name: 'Source status' })).toBeVisible()
  await expect(page.locator('.status-pill.stale').first()).toBeVisible()
  await expect(page.locator('.status-pill.degraded').first()).toBeVisible()
  await expect(page.locator('.status-pill.paused').first()).toBeVisible()
  await expect(page.locator('.status-pill.removed').first()).toBeVisible()
  await expect(page.locator('.status-pill.active').first()).toBeVisible()
})

test('member glossary discloses restricted support without exposing hidden evidence', async ({ page }) => {
  await page.goto('/glossary')

  await page.getByRole('button', { name: /Private Escalation Trigger/i }).click()
  await expect(page.getByText('restricted support')).toBeVisible()
  await expect(page.getByText('Promoted personal support')).not.toBeVisible()
  await expect(page.getByText('private:')).not.toBeVisible()
})

test('reviewer cannot officialize a cross-domain relation', async ({ page }) => {
  await page.goto('/review')

  await page.getByLabel('Switch actor').selectOption({ label: 'Domain Reviewer (review)' })
  await expect(page.getByRole('button', { name: 'Officialize' })).toBeVisible()
  await page.getByRole('button', { name: 'Officialize' }).click()

  await expect(page.getByText(/does not hold review scope/i)).toBeVisible()
})

test('superseded decisions remain readable with lineage visible', async ({ page }) => {
  await page.goto('/decisions')

  const legacyCard = page.getByRole('button', { name: /Legacy Escalation Routing/i })

  await expect(legacyCard).toBeVisible()
  await expect(legacyCard.getByText('Superseded by')).toBeVisible()
  await expect(legacyCard.getByText('Risk-Based Escalation Routing')).toBeVisible()
})

test('dashboard search uses structured answer content and a normal mobile form', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await page.getByLabel('Switch actor').selectOption({ label: 'Domain Reviewer (review)' })

  const searchInput = page.getByPlaceholder('Search workspace context')
  const inputBox = await searchInput.boundingBox()
  expect(inputBox?.height ?? 0).toBeLessThan(90)

  await page.getByRole('button', { name: 'Run search' }).click()

  await expect(page.getByText('Concepts').nth(0)).toBeVisible()
  await expect(page.getByText('Cited decisions')).toBeVisible()
  await expect(page.getByRole('heading', { name: '4 matches' })).toBeVisible()
})

test('graph explorer groups inbound and outbound relations for the selected concept', async ({ page }) => {
  await page.goto('/graph')
  await page.getByLabel('Switch actor').selectOption({ label: 'Domain Reviewer (review)' })

  await page.getByRole('button', { name: 'Explore Partner SLA' }).click()

  const detailPanel = page.locator('.graph-detail-panel')
  await expect(detailPanel.getByRole('heading', { name: 'Partner SLA' })).toBeVisible()
  await expect(detailPanel.getByRole('heading', { name: /1 relation/ }).first()).toBeVisible()
  await expect(detailPanel.getByText('Platform Circuit Breaker')).toBeVisible()
  await expect(detailPanel.getByText('Ops Playbook')).toBeVisible()
})

test('glossary and decisions show the detail pane before the list on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })

  await page.goto('/glossary')
  await expectDetailPaneBeforeList(page)

  await page.goto('/decisions')
  await expectDetailPaneBeforeList(page)
})

test('sources format timestamps for reading and wrap long locator text without overflow', async ({ page }) => {
  await page.goto('/sources')

  const firstTime = page.locator('time').first()
  await expect(firstTime).toBeVisible()
  await expect(firstTime).not.toContainText('T')
  await expect(page.getByRole('alert').first()).toBeVisible()

  const firstLocator = page.locator('.source-locator').first()
  const wrapsWithoutOverflow = await firstLocator.evaluate(
    (element) => element.scrollWidth <= element.clientWidth + 1,
  )
  expect(wrapsWithoutOverflow).toBeTruthy()
})

test('promoted support provenance does not leak personal origin references', async ({ page }) => {
  await page.goto('/glossary')

  await page.getByRole('button', { name: /VIP Escalation Insight/i }).click()
  await expect(page.getByText('Promoted personal support')).toBeVisible()
  await expect(page.getByText('redacted origin')).toBeVisible()
  await expect(page.getByText('private:')).not.toBeVisible()
})

async function expectDetailPaneBeforeList(page: Page) {
  const detailPane = page.locator('.detail-pane').first()
  const firstListItem = page.locator('.master-list').locator('button').first()

  const detailBox = await detailPane.boundingBox()
  const listBox = await firstListItem.boundingBox()

  expect(detailBox?.y ?? Number.POSITIVE_INFINITY).toBeLessThan(listBox?.y ?? Number.NEGATIVE_INFINITY)
}

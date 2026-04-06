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

test('warm editorial theme replaces the old gradient and shared surfaces use semantic colors', async ({ page }) => {
  await page.goto('/')

  const searchButton = page.getByRole('button', { name: 'Run search' })
  const sidebar = page.locator('.sidebar')
  const searchPanel = page.locator('.dashboard-search-panel')

  const buttonStyles = await searchButton.evaluate((element) => {
    const style = window.getComputedStyle(element)
    return {
      backgroundColor: style.backgroundColor,
      backgroundImage: style.backgroundImage,
      borderRadius: style.borderRadius,
    }
  })
  const sidebarBackground = await sidebar.evaluate((element) => window.getComputedStyle(element).backgroundColor)
  const panelBackground = await searchPanel.evaluate((element) => window.getComputedStyle(element).backgroundColor)

  expect(buttonStyles.backgroundColor).toBe('rgb(212, 154, 87)')
  expect(buttonStyles.backgroundImage).toBe('none')
  expect(buttonStyles.borderRadius).toBe('14px')
  expect(sidebarBackground).toBe('rgb(18, 25, 35)')
  expect(panelBackground).toBe('rgba(24, 33, 44, 0.96)')
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

test('semantic status pills and graph relation colors are visually distinct', async ({ page }) => {
  await page.goto('/sources')

  const activeBackground = await page.locator('.status-pill.active').first().evaluate((element) => {
    return window.getComputedStyle(element).backgroundColor
  })
  const staleBackground = await page.locator('.status-pill.stale').first().evaluate((element) => {
    return window.getComputedStyle(element).backgroundColor
  })
  await page.goto('/glossary')
  await page.getByRole('button', { name: /Private Escalation Trigger/i }).click()

  const restrictedBackground = await page.locator('.status-pill.restricted_support').first().evaluate((element) => {
    return window.getComputedStyle(element).backgroundColor
  })

  expect(activeBackground).not.toBe(staleBackground)
  expect(staleBackground).not.toBe(restrictedBackground)
  expect(activeBackground).not.toBe(restrictedBackground)

  await page.goto('/graph')
  await page.getByLabel('Switch actor').selectOption({ label: 'Domain Reviewer (review)' })
  await page.getByRole('button', { name: 'Explore Partner SLA' }).click()

  const outboundPredicate = await page.locator('.outbound-relations .relation-predicate').first().evaluate((element) => {
    return window.getComputedStyle(element).backgroundColor
  })
  const inboundPredicate = await page.locator('.inbound-relations .relation-predicate').first().evaluate((element) => {
    return window.getComputedStyle(element).backgroundColor
  })

  expect(outboundPredicate).toBe('rgba(116, 174, 221, 0.16)')
  expect(inboundPredicate).toBe('rgba(122, 174, 154, 0.16)')
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

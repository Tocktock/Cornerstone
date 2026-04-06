import { expect, test } from '@playwright/test'

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

  await expect(page.getByText('Legacy Escalation Routing')).toBeVisible()
  await expect(page.getByText('Superseded by Risk-Based Escalation Routing')).toBeVisible()
})

test('dashboard no-match state uses the canonical reason', async ({ page }) => {
  await page.goto('/')

  await page.getByPlaceholder('Search workspace context').fill('totally-unmatched-query')
  await page.getByRole('button', { name: 'Run search' }).click()

  await expect(page.getByRole('heading', { name: 'no_official_match' }).first()).toBeVisible()
})

test('promoted support provenance does not leak personal origin references', async ({ page }) => {
  await page.goto('/glossary')

  await page.getByRole('button', { name: /VIP Escalation Insight/i }).click()
  await expect(page.getByText('Promoted personal support')).toBeVisible()
  await expect(page.getByText('redacted origin')).toBeVisible()
  await expect(page.getByText('private:')).not.toBeVisible()
})

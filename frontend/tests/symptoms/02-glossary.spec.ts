import { expect, test } from '@playwright/test'

import { captureSnapshot, goToRoute } from './testkit'

test('topic reader routes stay presentable on mobile and disclose restricted support without leaks', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 })

  await goToRoute(page, '/explore/topics', 'Explore Topics')
  await expect(page.getByRole('heading', { name: 'Ops Playbook' }).first()).toBeVisible()

  await page.goto('/concepts/private-escalation-trigger')
  await expect(page.locator('.detail-hero h2')).toHaveText('Private Escalation Trigger')
  await expect(page.locator('.detail-hero')).toContainText('sensitive review-only signal')
  await expect(page.locator('.provenance-strip')).toContainText('restricted support present')
  await expect(page.locator('.concept-detail-page')).not.toContainText('private:')
  await expect(page.locator('.concept-detail-page')).not.toContainText('Promoted personal support')

  await page.goto('/concepts/vip-escalation-insight')
  await expect(page.locator('.detail-hero h2')).toHaveText('VIP Escalation Insight')
  await expect(page.locator('.concept-detail-page')).toContainText('Promoted personal support')
  await expect(page.locator('.concept-detail-page')).toContainText('redacted origin')
  await expect(page.locator('.concept-detail-page')).not.toContainText('private:')

  await captureSnapshot(page, testInfo, 'topics-mobile-presentable-reader')
})

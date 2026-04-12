import { expect, test } from '@playwright/test'

import {
  actorNamed,
  bootstrapViewer,
  captureSnapshot,
  goToRoute,
  switchActor,
} from './testkit'

test('map explorer groups inbound and outbound relations with URL-addressable selection', async ({
  page,
  request,
}, testInfo) => {
  const bootstrap = await bootstrapViewer(request)
  const reviewer = actorNamed(bootstrap, 'Domain Reviewer')

  await goToRoute(page, '/explore/map', 'Explore Map')
  await switchActor(page, reviewer)

  await expect(page.getByRole('button', { name: 'Jump to Ops Playbook' })).toBeVisible()
  await page.getByRole('button', { name: 'Explore Partner SLA' }).click()
  await expect(page).toHaveURL(/\/explore\/map\//)

  const detailPanel = page.locator('.graph-detail-panel')
  await expect(detailPanel.getByRole('heading', { name: 'Partner SLA' })).toBeVisible()
  await expect(detailPanel.locator('.outbound-relations .relation-card')).toHaveCount(1)
  await expect(detailPanel.locator('.inbound-relations .relation-card')).toHaveCount(1)
  await expect(detailPanel.locator('.outbound-relations .relation-card')).toContainText('Platform Circuit Breaker')
  await expect(detailPanel.locator('.inbound-relations .relation-card')).toContainText('Ops Playbook')
  await expect(detailPanel.getByText('depends_on')).toHaveCount(2)

  await captureSnapshot(page, testInfo, 'map-partner-sla-detail')
})

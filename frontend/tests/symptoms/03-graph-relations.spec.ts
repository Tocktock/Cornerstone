import { expect, test } from '@playwright/test'

import {
  actorNamed,
  bootstrapViewer,
  captureSnapshot,
  goToRoute,
  switchActor,
} from './testkit'

test('graph explorer groups inbound and outbound relations with distinct visual treatment', async ({
  page,
  request,
}, testInfo) => {
  const bootstrap = await bootstrapViewer(request)
  const reviewer = actorNamed(bootstrap, 'Domain Reviewer')

  await goToRoute(page, '/graph', 'Graph slice')
  await switchActor(page, reviewer)

  await expect(page.getByRole('button', { name: 'Jump to Ops Playbook' })).toBeVisible()
  await page.getByRole('button', { name: 'Explore Partner SLA' }).click()

  const detailPanel = page.locator('.graph-detail-panel')
  await expect(detailPanel.getByRole('heading', { name: 'Partner SLA' })).toBeVisible()
  await expect(detailPanel.locator('.outbound-relations')).toContainText('1 relation')
  await expect(detailPanel.locator('.inbound-relations')).toContainText('1 relation')
  await expect(detailPanel.getByText('Platform Circuit Breaker')).toBeVisible()
  await expect(detailPanel.getByText('Ops Playbook')).toBeVisible()
  await expect(detailPanel.getByText('depends_on')).toHaveCount(2)

  const outboundPredicate = await page
    .locator('.outbound-relations .relation-predicate')
    .first()
    .evaluate((element) => window.getComputedStyle(element).backgroundColor)
  const inboundPredicate = await page
    .locator('.inbound-relations .relation-predicate')
    .first()
    .evaluate((element) => window.getComputedStyle(element).backgroundColor)

  expect(outboundPredicate).toBe('rgba(116, 174, 221, 0.16)')
  expect(inboundPredicate).toBe('rgba(122, 174, 154, 0.16)')

  await captureSnapshot(page, testInfo, 'graph-partner-sla-detail')
})

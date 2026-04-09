import { expect, test } from '@playwright/test'

import {
  captureSnapshot,
  expectDetailPaneBeforeList,
  goToRoute,
} from './testkit'

test('glossary keeps the detail pane first on mobile and discloses restricted support without leaks', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await goToRoute(page, '/glossary', 'Glossary')
  await expectDetailPaneBeforeList(page)
  const detailPanel = page.locator('.glossary-detail-panel')

  await page.getByRole('button', { name: /Ops Playbook/i }).click()
  await expect(detailPanel.getByText(/Support items: 1 · Visible: 1/i)).toBeVisible()
  await expect(detailPanel.getByText('Workspace handbook').first()).toBeVisible()

  await page.getByRole('button', { name: /Private Escalation Trigger/i }).click()
  await expect(detailPanel.getByText(/Support items: 1 · Visible: 0/i)).toBeVisible()
  await expect(page.getByText('restricted support').first()).toBeVisible()
  await expect(detailPanel.getByText('Promoted personal support')).toHaveCount(0)
  await expect(detailPanel.getByText('private:')).toHaveCount(0)

  await page.getByRole('button', { name: /VIP Escalation Insight/i }).click()
  await expect(detailPanel.getByText('Promoted personal support')).toBeVisible()
  await expect(detailPanel.getByText('redacted origin')).toBeVisible()
  await expect(detailPanel.getByText('private:')).toHaveCount(0)

  await captureSnapshot(page, testInfo, 'glossary-mobile-provenance')
})

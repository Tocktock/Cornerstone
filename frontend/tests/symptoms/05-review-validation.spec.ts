import { expect, test } from '@playwright/test'

import {
  actorNamed,
  apiPostAs,
  bootstrapViewer,
  captureSnapshot,
  conceptByName,
  goToRoute,
  supportItemIdsForConcept,
  switchActor,
  uniqueName,
} from './testkit'

test('Review Studio supports valid officialization, invalid scope denial, and queue refresh', async ({
  page,
  request,
}, testInfo) => {
  const bootstrap = await bootstrapViewer(request)
  const member = actorNamed(bootstrap, 'Member')
  const operator = actorNamed(bootstrap, 'Operator')
  const reviewer = actorNamed(bootstrap, 'Domain Reviewer')
  const admin = actorNamed(bootstrap, 'Workspace Admin')

  const partnerSla = await conceptByName(request, member, 'Partner SLA')
  const vipInsight = await conceptByName(request, member, 'VIP Escalation Insight')
  const platformBreaker = await conceptByName(request, reviewer, 'Platform Circuit Breaker')
  const conceptSupportItemIds = await supportItemIdsForConcept(request, member, 'Ops Playbook')
  const platformSupportItemIds = await supportItemIdsForConcept(request, reviewer, 'Platform Circuit Breaker')

  const conceptName = uniqueName(testInfo, 'Draft review concept')
  const sameDomainPredicate = `${uniqueName(testInfo, 'guides')}-predicate`
  const crossDomainPredicate = `${uniqueName(testInfo, 'depends')}-predicate`

  await apiPostAs(request, operator, '/concepts', {
    body: {
      context_space_id: bootstrap.workspace.context_space_id,
      canonical_name: conceptName,
      definition: 'A draft concept created through API setup for review verification.',
      owning_domain: 'sales_ops',
      support_item_ids: conceptSupportItemIds,
    },
  })

  await apiPostAs(request, operator, '/relations', {
    body: {
      context_space_id: bootstrap.workspace.context_space_id,
      subject_concept_id: partnerSla.payload.concept_id,
      predicate: sameDomainPredicate,
      object_concept_id: vipInsight.payload.concept_id,
      description: 'A same-domain review relation created for browser verification.',
      support_item_ids: conceptSupportItemIds,
    },
  })

  await apiPostAs(request, operator, '/relations', {
    body: {
      context_space_id: bootstrap.workspace.context_space_id,
      subject_concept_id: partnerSla.payload.concept_id,
      predicate: crossDomainPredicate,
      object_concept_id: platformBreaker.payload.concept_id,
      description: 'A cross-domain relation that should require workspace review.',
      support_item_ids: platformSupportItemIds,
    },
  })

  await goToRoute(page, '/explore/topics', 'Explore Topics')
  await switchActor(page, member)
  await expect(page.getByRole('heading', { name: new RegExp(conceptName, 'i') })).toHaveCount(0)

  await goToRoute(page, '/review-studio', 'Review Studio')
  await switchActor(page, reviewer)

  const conceptCard = page.locator('.review-item-card').filter({ hasText: conceptName })
  await expect(conceptCard).toBeVisible()
  await conceptCard.getByRole('button', { name: 'Officialize' }).click()
  await expect(page.getByText(new RegExp(`officialize succeeded for ${conceptName}`, 'i'))).toBeVisible()
  await expect(conceptCard).toHaveCount(0)

  const sameDomainLabel = `Partner SLA ${sameDomainPredicate} VIP Escalation Insight`
  const sameDomainCard = page.locator('.review-item-card').filter({ hasText: sameDomainLabel })
  await expect(sameDomainCard).toBeVisible()
  await sameDomainCard.getByRole('button', { name: 'Officialize' }).click()
  await expect(page.getByText(new RegExp(`officialize succeeded for ${sameDomainLabel}`, 'i'))).toBeVisible()
  await expect(sameDomainCard).toHaveCount(0)

  const crossDomainLabel = `Partner SLA ${crossDomainPredicate} Platform Circuit Breaker`
  const crossDomainCard = page.locator('.review-item-card').filter({ hasText: crossDomainLabel })
  await expect(crossDomainCard).toBeVisible()
  await crossDomainCard.getByRole('button', { name: 'Officialize' }).click()
  await expect(page.getByText(/does not hold review scope/i)).toBeVisible()

  await switchActor(page, admin)
  await expect(crossDomainCard).toBeVisible()
  await crossDomainCard.getByRole('button', { name: 'Officialize' }).click()
  await expect(page.getByText(new RegExp(`officialize succeeded for ${crossDomainLabel}`, 'i'))).toBeVisible()
  await expect(crossDomainCard).toHaveCount(0)

  await goToRoute(page, '/explore/topics', 'Explore Topics')
  await expect(
    page
      .locator('.artifact-card')
      .filter({ has: page.locator('h3', { hasText: conceptName }) })
      .first(),
  ).toBeVisible()

  await goToRoute(page, '/explore/map', 'Explore Map')
  await page.getByRole('button', { name: 'Explore Partner SLA' }).click()
  const detailPanel = page.locator('.graph-detail-panel')
  await expect(detailPanel).toContainText('VIP Escalation Insight')
  await expect(detailPanel).toContainText('Platform Circuit Breaker')
  await expect(detailPanel).toContainText(sameDomainPredicate)
  await expect(detailPanel).toContainText(crossDomainPredicate)

  await captureSnapshot(page, testInfo, 'review-studio-valid-and-invalid-actions')
})

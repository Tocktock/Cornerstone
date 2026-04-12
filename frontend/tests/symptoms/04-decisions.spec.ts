import { expect, test } from '@playwright/test'

import type { ContractEnvelope, RelationPayload } from '../../src/types/api'
import {
  actorNamed,
  apiGetAs,
  apiPostAs,
  bootstrapViewer,
  captureSnapshot,
  decisionByTitle,
  goToRoute,
  supportItemIdsForConcept,
  switchActor,
  uniqueName,
} from './testkit'

test('seeded decisions preserve readable lineage and presentable direct routes on mobile', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await goToRoute(page, '/explore/decisions', 'Explore Decisions')

  const legacyCard = page
    .locator('.artifact-card')
    .filter({ has: page.locator('h3', { hasText: 'Legacy Escalation Routing' }) })
  await expect(legacyCard).toBeVisible()
  await expect(legacyCard.getByText('Superseded by')).toBeVisible()
  await expect(legacyCard.getByText('Risk-Based Escalation Routing')).toBeVisible()

  await page.goto('/decisions/risk-based-escalation-routing')
  await expect(page.locator('.detail-hero h2')).toHaveText('Risk-Based Escalation Routing')
  await expect(page.locator('.lineage-rail')).toContainText('Legacy Escalation Routing')
  await expect(page.locator('.decision-detail-page')).toContainText('Promoted personal support')

  await captureSnapshot(page, testInfo, 'decisions-presentable-reader')
})

test('draft decisions can be officialized and extend visible supersession lineage', async ({
  page,
  request,
}, testInfo) => {
  const bootstrap = await bootstrapViewer(request)
  const member = actorNamed(bootstrap, 'Member')
  const operator = actorNamed(bootstrap, 'Operator')
  const admin = actorNamed(bootstrap, 'Workspace Admin')
  const decisionTitle = uniqueName(testInfo, 'Escalation routing update')

  const supportItemIds = await supportItemIdsForConcept(request, member, 'Ops Playbook')
  const currentDecision = await decisionByTitle(request, member, 'Risk-Based Escalation Routing')
  const relations = await apiGetAs<ContractEnvelope<RelationPayload>[]>(request, member, '/relations')
  const officialRelation = relations.find(
    (relation) =>
      relation.payload.subject_concept_ref.resource_label === 'Ops Playbook' &&
      relation.payload.object_concept_ref.resource_label === 'Partner SLA',
  )

  expect(officialRelation, 'Missing seeded official relation.').toBeTruthy()

  const draftDecision = await apiPostAs<ContractEnvelope<{ decision_id: string; public_slug: string; title: string; lifecycle_state: string }>>(
    request,
    operator,
    '/decisions',
    {
      body: {
        context_space_id: bootstrap.workspace.context_space_id,
        title: decisionTitle,
        decision_statement: 'Escalate partner incidents earlier when modern routing indicators align.',
        problem_statement: 'Reviewers need a fresher escalation path than the current decision exposes.',
        rationale: 'The updated routing keeps the playbook aligned with partner and VIP context.',
        constraints: ['Keep existing disclosure controls intact.'],
        impact_summary: 'Makes follow-up routing decisions easier to read in the workspace.',
        owning_domain: 'sales_ops',
        support_item_ids: supportItemIds,
        linked_concept_ids: [currentDecision.payload.linked_concept_refs[0]?.resource_id].filter(Boolean),
        linked_relation_ids: [officialRelation?.payload.relation_id].filter(Boolean),
        supersedes_decision_id: currentDecision.payload.decision_id,
      },
    },
  )

  expect(draftDecision.payload.lifecycle_state).toBe('proposed')

  await goToRoute(page, '/explore/decisions', 'Explore Decisions')
  await switchActor(page, member)
  await expect(page.locator('.artifact-card h3').filter({ hasText: decisionTitle })).toHaveCount(0)

  await goToRoute(page, '/review-studio', 'Review Studio')
  await switchActor(page, admin)

  const decisionCard = page.locator('.review-item-card').filter({ hasText: decisionTitle })
  await expect(decisionCard).toBeVisible()
  await decisionCard.getByRole('button', { name: 'Officialize' }).click()
  await expect(page.getByText(new RegExp(`officialize succeeded for ${decisionTitle}`, 'i'))).toBeVisible()
  await expect(decisionCard).toHaveCount(0)

  await goToRoute(page, '/explore/decisions', 'Explore Decisions')
  await expect(page.locator('.artifact-card h3').filter({ hasText: decisionTitle })).toHaveCount(1)
  const supersededCard = page
    .locator('.artifact-card')
    .filter({ has: page.locator('h3', { hasText: 'Risk-Based Escalation Routing' }) })
  await expect(supersededCard.getByText('Superseded by')).toBeVisible()
  await expect(supersededCard.getByText(decisionTitle)).toBeVisible()

  await page.goto(`/decisions/${draftDecision.payload.public_slug}`)
  await expect(page.locator('.detail-hero h2')).toHaveText(decisionTitle)

  await captureSnapshot(page, testInfo, 'decisions-officialized-draft-lineage')
})

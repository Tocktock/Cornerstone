import { expect, type APIRequestContext, type APIResponse, type Page, type TestInfo } from '@playwright/test'

import type {
  ActorSession,
  ConceptPayload,
  ContractEnvelope,
  DecisionPayload,
  ProvenancePayload,
  RelationPayload,
  ReviewQueueItem,
  ViewerBootstrap,
} from '../../src/types/api'

const backendPort = process.env.CORNERSTONE_BROWSER_BACKEND_PORT ?? '8011'

export const apiOrigin = `http://127.0.0.1:${backendPort}`
export const apiBaseUrl = `${apiOrigin}/api/v1`

type ConsumerScope = 'member' | 'review' | 'admin'

type RequestOptions = {
  body?: unknown
  params?: Record<string, string | undefined>
  requestedScope?: ConsumerScope
}

export async function bootstrapViewer(request: APIRequestContext) {
  return apiGet<ViewerBootstrap>(request, '/bootstrap')
}

export function actorNamed(bootstrap: ViewerBootstrap, displayName: string) {
  const actor = bootstrap.actors.find((candidate) => candidate.display_name === displayName)
  expect(actor, `Missing actor ${displayName}.`).toBeTruthy()
  return actor as ActorSession
}

export async function apiGet<T>(
  request: APIRequestContext,
  path: string,
  options: RequestOptions = {},
) {
  return requestJson<T>(
    await request.get(buildApiUrl(path, options.params, options.requestedScope)),
    `GET ${path}`,
  )
}

export async function apiGetAs<T>(
  request: APIRequestContext,
  actor: ActorSession,
  path: string,
  options: Omit<RequestOptions, 'body'> = {},
) {
  return requestJson<T>(
    await request.get(buildApiUrl(path, options.params, options.requestedScope), {
      headers: actorHeaders(actor),
    }),
    `GET ${path} as ${actor.display_name}`,
  )
}

export async function apiPostAs<T>(
  request: APIRequestContext,
  actor: ActorSession,
  path: string,
  options: RequestOptions = {},
) {
  return requestJson<T>(
    await request.post(buildApiUrl(path, options.params, options.requestedScope), {
      data: options.body,
      headers: actorHeaders(actor),
    }),
    `POST ${path} as ${actor.display_name}`,
  )
}

export async function apiDeleteAs<T>(
  request: APIRequestContext,
  actor: ActorSession,
  path: string,
  options: Omit<RequestOptions, 'body'> = {},
) {
  return requestJson<T>(
    await request.delete(buildApiUrl(path, options.params, options.requestedScope), {
      headers: actorHeaders(actor),
    }),
    `DELETE ${path} as ${actor.display_name}`,
  )
}

export async function goToRoute(page: Page, route: string, heading?: string | RegExp) {
  await page.goto(route)
  await waitForShell(page)
  if (heading) {
    await expect(
      page.getByRole(
        'heading',
        typeof heading === 'string' ? { name: heading, exact: true } : { name: heading },
      ),
    ).toBeVisible()
  }
}

export async function waitForShell(page: Page) {
  await expect(page.locator('.loading-screen')).toHaveCount(0)
  await expect(page.getByRole('navigation', { name: 'Primary navigation' })).toBeVisible()
}

export async function switchActor(page: Page, actor: ActorSession) {
  const selector = page.getByLabel('Switch actor')
  await selector.selectOption(actor.actor_id)
  await expect(selector).toHaveValue(actor.actor_id)
  await expect
    .poll(() => page.evaluate(() => window.localStorage.getItem('cornerstone.actorToken')))
    .toBe(actor.token)
  await expect(page.locator('.loading-screen')).toHaveCount(0)
}

export async function captureSnapshot(page: Page, testInfo: TestInfo, name: string) {
  const fileName = `${sanitize(name)}.png`
  const screenshotPath = testInfo.outputPath(fileName)
  await page.screenshot({ fullPage: true, path: screenshotPath })
  await testInfo.attach(name, {
    path: screenshotPath,
    contentType: 'image/png',
  })
}

export async function expectDetailPaneBeforeList(page: Page) {
  const detailPane = page.locator('.detail-pane').first()
  const firstListItem = page.locator('.master-list').locator('button').first()
  const detailBox = await detailPane.boundingBox()
  const listBox = await firstListItem.boundingBox()

  expect(detailBox?.y ?? Number.POSITIVE_INFINITY).toBeLessThan(
    listBox?.y ?? Number.NEGATIVE_INFINITY,
  )
}

export async function conceptByName(
  request: APIRequestContext,
  actor: ActorSession,
  canonicalName: string,
) {
  const concepts = await apiGetAs<ContractEnvelope<ConceptPayload>[]>(request, actor, '/concepts')
  const concept = concepts.find((item) => item.payload.canonical_name === canonicalName)
  expect(concept, `Missing concept ${canonicalName}.`).toBeTruthy()
  return concept as ContractEnvelope<ConceptPayload>
}

export async function decisionByTitle(
  request: APIRequestContext,
  actor: ActorSession,
  title: string,
) {
  const decisions = await apiGetAs<ContractEnvelope<DecisionPayload>[]>(request, actor, '/decisions')
  const decision = decisions.find((item) => item.payload.title === title)
  expect(decision, `Missing decision ${title}.`).toBeTruthy()
  return decision as ContractEnvelope<DecisionPayload>
}

export async function relationByPredicate(
  request: APIRequestContext,
  actor: ActorSession,
  predicate: string,
) {
  const relations = await apiGetAs<ContractEnvelope<RelationPayload>[]>(request, actor, '/relations')
  const relation = relations.find((item) => item.payload.predicate === predicate)
  expect(relation, `Missing relation with predicate ${predicate}.`).toBeTruthy()
  return relation as ContractEnvelope<RelationPayload>
}

export async function supportItemIdsForConcept(
  request: APIRequestContext,
  actor: ActorSession,
  canonicalName: string,
) {
  const concept = await conceptByName(request, actor, canonicalName)
  const provenance = await apiGetAs<ContractEnvelope<ProvenancePayload>>(
    request,
    actor,
    `/provenance/concept/${concept.payload.concept_id}`,
  )

  return provenance.payload.support_items.map((item) => item.support_item_id)
}

export async function supportItemIdsForDecision(
  request: APIRequestContext,
  actor: ActorSession,
  title: string,
) {
  const decision = await decisionByTitle(request, actor, title)
  const provenance = await apiGetAs<ContractEnvelope<ProvenancePayload>>(
    request,
    actor,
    `/provenance/decision/${decision.payload.decision_id}`,
  )

  return provenance.payload.support_items.map((item) => item.support_item_id)
}

export async function reviewQueue(request: APIRequestContext, actor: ActorSession) {
  return apiGetAs<ReviewQueueItem[]>(request, actor, '/review-queue')
}

export function uniqueName(testInfo: TestInfo, prefix: string, suffix?: string) {
  const base = sanitize(testInfo.title).slice(0, 48)
  return [prefix, base, suffix].filter(Boolean).join(' ')
}

function actorHeaders(actor: ActorSession) {
  return {
    Authorization: `Bearer ${actor.token}`,
  }
}

function buildApiUrl(
  path: string,
  params?: Record<string, string | undefined>,
  requestedScope?: ConsumerScope,
) {
  const url = new URL(`${apiBaseUrl}${path}`)

  if (requestedScope) {
    url.searchParams.set('requested_scope', requestedScope)
  }

  for (const [key, value] of Object.entries(params ?? {})) {
    if (value) {
      url.searchParams.set(key, value)
    }
  }

  return url.toString()
}

async function requestJson<T>(response: APIResponse, label: string) {
  const text = await response.text()
  expect(response.ok(), `${label} failed: ${response.status()} ${text}`).toBeTruthy()
  return JSON.parse(text) as T
}

function sanitize(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '')
}

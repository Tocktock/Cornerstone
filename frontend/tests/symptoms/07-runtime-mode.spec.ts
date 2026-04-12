import { expect, test, type Page } from '@playwright/test'

import { apiBaseUrl, goToRoute, waitForShell } from './testkit'

const workspace = {
  context_space_id: 'ctx-workspace',
  context_space_kind: 'workspace',
  context_space_name: 'Cornerstone Workspace',
} as const

const personalContext = {
  context_space_id: 'ctx-personal',
  context_space_kind: 'personal',
  context_space_name: 'Member Personal Context',
} as const

const memberActor = {
  actor_id: 'actor-member',
  display_name: 'Member',
  base_role: 'member',
  token: 'token-member',
  scoped_capabilities: [{ capability: 'operate', scope: 'workspace' }],
  preferred_consumer_scope: 'member',
} as const

const operatorActor = {
  actor_id: 'actor-operator',
  display_name: 'Operator',
  base_role: 'member',
  token: 'token-operator',
  scoped_capabilities: [
    { capability: 'operate', scope: 'workspace' },
    { capability: 'manage_connectors', scope: 'workspace' },
  ],
  preferred_consumer_scope: 'review',
} as const

test('production awaiting_sources shows manager CTA and source studio onboarding', async ({
  page,
}) => {
  await installRuntimeMock(page, {
    bootstrap: buildBootstrap({
      runtime_mode: 'production',
      workspace_data_state: 'awaiting_sources',
      linked_source_count: 0,
      active_source_count: 0,
      degraded_source_count: 0,
      actors: [operatorActor],
    }),
    workspaceHome: emptyWorkspaceHome(),
    concepts: [],
    decisions: [],
    graph: emptyGraph(),
    sourceConnections: [],
    connectorTemplates: [notionTemplate()],
  })

  await goToRoute(page, '/', 'Workspace')
  await expect(page.getByRole('heading', { name: 'Connect a shared datasource first' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Open Source Studio' })).toBeVisible()

  await page.getByRole('link', { name: 'Open Source Studio' }).click()
  await waitForShell(page)
  await expect(page).toHaveURL(/\/source-studio$/)
  await expect(page.getByText('Connect a live datasource')).toBeVisible()
  await expect(
    page.getByRole('heading', { name: 'Bind Notion and create a production source connection' }),
  ).toBeVisible()
})

test('production awaiting_sources keeps member guidance read-only', async ({ page }) => {
  await installRuntimeMock(page, {
    bootstrap: buildBootstrap({
      runtime_mode: 'production',
      workspace_data_state: 'awaiting_sources',
      linked_source_count: 0,
      active_source_count: 0,
      degraded_source_count: 0,
      actors: [memberActor],
    }),
    workspaceHome: emptyWorkspaceHome(),
    concepts: [],
    decisions: [],
    graph: emptyGraph(),
    sourceConnections: [],
    connectorTemplates: [],
  })

  await goToRoute(page, '/', 'Workspace')
  await expect(page.getByRole('link', { name: 'Open Source Studio' })).toHaveCount(0)
  await expect(page.getByText(/connector manager needs to connect a datasource/i)).toBeVisible()

  await page.goto('/source-studio')
  await waitForShell(page)
  await expect(
    page.getByRole('heading', {
      name: 'A connector manager needs to connect the first datasource',
    }),
  ).toBeVisible()
})

test('production syncing_sources shows first-sync guidance on explore topics', async ({
  page,
}) => {
  await installRuntimeMock(page, {
    bootstrap: buildBootstrap({
      runtime_mode: 'production',
      workspace_data_state: 'syncing_sources',
      linked_source_count: 2,
      active_source_count: 1,
      degraded_source_count: 0,
      actors: [memberActor],
    }),
    workspaceHome: emptyWorkspaceHome(),
    concepts: [],
    decisions: [],
    graph: emptyGraph(),
    sourceConnections: [],
    connectorTemplates: [],
  })

  await goToRoute(page, '/explore/topics', 'Explore Topics')
  await expect(
    page.getByRole('heading', { name: 'Topic index is waiting for first sync' }),
  ).toBeVisible()
  await expect(page.getByText(/2 linked sources are currently being prepared/i)).toBeVisible()
})

test('production degraded keeps published topics visible with recovery cues', async ({
  page,
}) => {
  await installRuntimeMock(page, {
    bootstrap: buildBootstrap({
      runtime_mode: 'production',
      workspace_data_state: 'degraded',
      linked_source_count: 2,
      active_source_count: 1,
      degraded_source_count: 1,
      actors: [memberActor],
    }),
    workspaceHome: emptyWorkspaceHome(),
    concepts: [topicEnvelope()],
    decisions: [],
    graph: emptyGraph(),
    sourceConnections: [],
    connectorTemplates: [],
  })

  await goToRoute(page, '/explore/topics', 'Explore Topics')
  await expect(page.getByRole('heading', { name: 'Some source health is degraded' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Ops Playbook' })).toBeVisible()
})

async function installRuntimeMock(
  page: Page,
  scenario: {
    bootstrap: unknown
    workspaceHome: unknown
    concepts: unknown
    decisions: unknown
    graph: unknown
    sourceConnections: unknown
    connectorTemplates: unknown
  },
) {
  await page.route(`${apiBaseUrl}/**`, async (route) => {
    const url = new URL(route.request().url())

    if (url.pathname.endsWith('/bootstrap')) {
      await route.fulfill(jsonResponse(scenario.bootstrap))
      return
    }
    if (url.pathname.endsWith('/workspace-home')) {
      await route.fulfill(jsonResponse(scenario.workspaceHome))
      return
    }
    if (url.pathname.endsWith('/concepts')) {
      await route.fulfill(jsonResponse(scenario.concepts))
      return
    }
    if (url.pathname.endsWith('/decisions')) {
      await route.fulfill(jsonResponse(scenario.decisions))
      return
    }
    if (url.pathname.endsWith('/graph')) {
      await route.fulfill(jsonResponse(scenario.graph))
      return
    }
    if (url.pathname.endsWith('/source-connections')) {
      await route.fulfill(jsonResponse(scenario.sourceConnections))
      return
    }
    if (url.pathname.endsWith('/connector-templates')) {
      await route.fulfill(jsonResponse(scenario.connectorTemplates))
      return
    }

    await route.fulfill({
      status: 404,
      contentType: 'application/json',
      body: JSON.stringify({ detail: `Unhandled mocked endpoint: ${url.pathname}` }),
    })
  })
}

function buildBootstrap(overrides: Record<string, unknown>) {
  return {
    workspace,
    personal_context: personalContext,
    actors: [memberActor],
    runtime_mode: 'mock',
    workspace_data_state: 'demo_seeded',
    linked_source_count: 4,
    active_source_count: 2,
    degraded_source_count: 1,
    ...overrides,
  }
}

function emptyWorkspaceHome() {
  return envelope('workspace_home', 'get_workspace_home', {
    hero_prompt: 'Ask about official workspace context.',
    featured_answer: null,
    featured_cards: [],
    recent_changes: [],
    freshness_alerts: [],
    review_queue_summary: {
      pending_count: 0,
      review_required_count: 0,
      officialize_ready_count: 0,
    },
    source_health_summary: {
      total_count: 0,
      active_count: 0,
      monitoring_count: 0,
      stale_count: 0,
      degraded_count: 0,
      paused_count: 0,
      removed_count: 0,
    },
  })
}

function emptyGraph() {
  return envelope('graph_slice', 'get_graph_slice', {
    root_concept_refs: [],
    nodes: [],
    edges: [],
  })
}

function topicEnvelope() {
  return envelope('concept', 'get_concept', {
    concept_id: 'concept-ops-playbook',
    public_slug: 'ops-playbook',
    canonical_name: 'Ops Playbook',
    aliases: ['Incident handbook'],
    definition: 'The shared operating handbook for incident response.',
    owning_domain: 'sales_ops',
    review_domain: 'sales_ops',
    lifecycle_state: 'official',
    verification_state: 'verified',
    support_visibility: 'source_backed',
    visible_support_items: [],
    linked_relation_refs: [],
    linked_decision_refs: [],
    provenance_summary: {
      support_item_count: 2,
      visible_support_item_count: 2,
      restricted_support_present: false,
      freshness_state: 'current',
      verification_state: 'verified',
      promotion_lineage_present: false,
    },
  })
}

function notionTemplate() {
  return {
    template_key: 'notion_shared_page_tree',
    provider: 'notion',
    label: 'Notion shared page tree',
    description: 'Sync a shared Notion page tree.',
    scope_kind: 'page_tree',
    default_visibility_class: 'member_visible',
    recommended_sync_interval_seconds: 900,
    preview_required: true,
  }
}

function envelope(responseKind: string, requestIntent: string, payload: unknown) {
  return {
    contract_version: '2026-04-p0',
    response_kind: responseKind,
    request_intent: requestIntent,
    context_space_ref: workspace,
    consumer_scope: 'member',
    payload,
    related_refs: [],
    warnings: [],
  }
}

function jsonResponse(body: unknown) {
  return {
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  }
}

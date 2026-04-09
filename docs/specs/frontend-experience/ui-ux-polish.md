# Frontend Experience UI Polish

## Summary

This spec defines the UI and interaction polish required for the current Cornerstone frontend without changing the underlying backend contracts.

The goal is to make the existing routes readable, responsive, permission-aware, and easier to scan while preserving the current dark visual language.

## Scope and owned behavior

This spec owns:
- shared shell layout on desktop and mobile
- navigation visibility for review-only surfaces
- permission-aware review access messaging
- dashboard answer and search-results presentation
- glossary and decision detail-pane ordering on mobile
- graph-slice exploration presentation
- source-status scanability and timestamp formatting
- frontend focus-visible behavior

## Shared shell requirements

- The shell must keep the current sidebar-orientated identity on desktop.
- Workspace and persona metadata must render as stacked label/value blocks rather than inline text runs.
- Actors without review access must not see `Review` in primary navigation.
- A direct visit to `/review` without review access must render a friendly permission state and must not rely on a raw API error for UX.
- At widths below `1180px`, the shell must compress into a compact header area plus a horizontally scrollable navigation row so route content appears in the first viewport.
- Interactive shell elements must expose visible keyboard focus treatment.

## Route-level requirements

### Dashboard

- The mobile search form must stack normally and keep standard control heights.
- Answer content must use structured sections when `answer_sections` are available.
- Cited refs and follow-up refs must remain visible as secondary context.
- Search results must render in a separate sibling panel with an explicit result count.

### Glossary and Decisions

- Desktop keeps the current master-detail pattern.
- On widths below `960px`, the selected detail/provenance pane must appear before the list of selectable records.
- Selected cards must remain visually distinct.
- Decision supersession must render as explicit lineage, not only muted prose.

### Graph

- The graph route must behave as a structured relationship explorer over the existing graph-slice payload.
- One concept is always selected by default, preferring the first root concept.
- Root concepts must remain visible as quick-jump controls.
- The selected concept must disclose inbound and outbound relations separately.
- The node list must expose direct relation counts.

### Review

- Authorized actors see the review queue with clear kind, domain, disclosure, and action hierarchy.
- Unauthorized actors see a permission-aware explainer that points them to actor switching.

### Sources

- Source cards must prioritize provider, label, state pills, visibility, last success, and the current error.
- Source locators must render as subdued wrapped secondary text.
- Successful sync times must render in readable local datetime text.
- Error text must render as a dedicated alert row.

## Constraints and non-goals

- This change must not modify backend endpoints or payload contracts.
- This change must not introduce a graph visualization dependency.
- This change is a polish pass, not a full visual redesign.

## Verification expectations

- Browser signoff for the current frontend must run against the deterministic Postgres-backed stack, not a mocked transport.
- The Playwright harness must treat backend process readiness separately from API health: the spawned backend must answer a stable root route for boot readiness, and the suite must assert `/api/v1/health` explicitly as an early smoke check.
- Browser symptom coverage must stay split by feature area so failures localize clearly:
  - bootstrap and access
  - sources and connector management
  - glossary and provenance disclosure
  - graph exploration
  - decisions and lineage
  - review queue behavior
  - dashboard retrieval and no-match states
- Browser artifacts for screenshots and traces must land under `output/playwright/`.

## Related docs

- [../retrieval-and-answers/spec.md](../retrieval-and-answers/spec.md)
- [../graph-and-relations/spec.md](../graph-and-relations/spec.md)
- [../review-and-validation/spec.md](../review-and-validation/spec.md)
- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)
- [../../decisions/0012-serving-contract-is-canonical-across-ui-api-and-mcp-surfaces.md](../../decisions/0012-serving-contract-is-canonical-across-ui-api-and-mcp-surfaces.md)

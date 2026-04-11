# Workspace Plus Studios Redesign

## Summary

This spec defines the active frontend experience model for Cornerstone after the Craft-informed redesign pass and the follow-on full-route expressive UI rollout.

The redesign replaces the prior dashboard-plus-utility-route posture with a workspace-plus-studios structure:
- `Workspace` is the default member-facing home
- `Explore` is the shared browse surface for topics, decisions, and map exploration
- `Review Studio` and `Source Studio` are operational spaces with capability-gated navigation

This spec owns the implemented navigation model, route behavior, reader-versus-studio visual split, and the expressive presentation grammar now applied across every current route.

## Scope and owned behavior

This spec owns:
- the top-level route structure and one-release compatibility redirects
- the shared shell posture for reader and studio surfaces
- the reader-mode and studio-mode visual split
- the shared expressive token, spacing, and motion grammar
- workspace-home behavior and its frontend presentation
- explore topics, decisions, and map behavior
- direct concept and decision reading surfaces
- review-studio and source-studio presentation and access behavior
- mobile first-viewport content priority expectations

This spec does not own:
- native application redesign work
- public publishing or branded web-sharing infrastructure
- consumer productivity features such as journaling, task management, or calendar workflows

## Route model

Canonical routes:
- `/`
- `/explore/topics`
- `/explore/decisions`
- `/explore/map`
- `/explore/map/:conceptId`
- `/concepts/:publicSlug`
- `/decisions/:publicSlug`
- `/review-studio`
- `/source-studio`

Compatibility redirects for one release:
- `/glossary` -> `/explore/topics`
- `/graph` -> `/explore/map`
- `/decisions` -> `/explore/decisions`
- `/review` -> `/review-studio`
- `/sources` -> `/source-studio`

## Shared shell requirements

- The app must use a compact top header instead of a persistent left sidebar.
- Primary navigation must emphasize `Workspace` and `Explore` first.
- `Review Studio` and `Source Studio` must appear in primary navigation only when the active actor can use them.
- Workspace metadata and actor switching must live inside a workspace/profile tray rather than consuming the main reading column.
- Reader routes must place the artifact or answer surface in the first viewport before secondary workspace controls.

## Visual system requirements

### Reader mode

- Reader routes must use a light editorial surface family.
- The reader canvas should feel paper-toned and artifact-first rather than operational-first.
- Typography must pair `Newsreader` for display headings with `IBM Plex Sans` for UI/body and `IBM Plex Mono` for metadata.
- Reader routes must use stronger display scale, layered backgrounds, and assertive spacing so the first viewport leads with the primary artifact instead of shell furniture.
- Browse and detail routes must preserve card-to-detail identity through shared artifact, provenance, and section-header primitives.
- Provenance, support disclosure, freshness, and verification states must stay visible without dominating the primary reading column.

### Studio mode

- Studio routes must use a darker operational surface family than reader routes.
- Review and source-management panels must prioritize semantic state colors over decorative brand treatment.
- Studio routes may inherit the same typography and border-radius family as reader routes, but must stay denser, sharper, and more operational in grouping and spacing.
- High-stakes actions must remain visually distinct, especially officialize versus reject paths and healthy monitoring versus degraded intervention states.

### Shared motion grammar

- Hover and focus feedback must stay short and utility-first, targeting roughly `120–180ms`.
- Route-local content entrances may use restrained transitions around `180–240ms`.
- `prefers-reduced-motion` must collapse expressive transitions without breaking layout or state clarity.
- The frontend must not introduce cinematic route choreography, parallax, WebGL, or motion that obscures trust or workflow state.

## Contract dependencies

The frontend depends on the following additive serving-contract behavior:
- `request_intent: get_workspace_home`
- `response_kind: workspace_home`
- `WorkspaceHomePayload` on `/api/v1/workspace-home`
- `DecisionPayload.public_slug` for presentable direct routes

Trust and provenance semantics remain canonical and unchanged:
- `support_visibility`
- `verification_state`
- `freshness_state`
- provenance summary
- lineage semantics

## Route-level requirements

### Workspace

- `Workspace` must be the default landing experience.
- The first interaction must be a hero search/ask surface.
- The first viewport must use one dominant lead lane and one supporting rail so the ask surface and one primary knowledge artifact appear before operational summaries.
- The page must render a featured answer, featured official knowledge cards, recent changes, freshness alerts, and quiet review/source summaries.
- Operational summaries must remain secondary to the artifact and answer surfaces.
- Recent changes must read as an artifact river with direct continuity into detail routes.

### Explore

- `Explore` must expose tabs for `Topics`, `Decisions`, and `Map`.
- Topics and Decisions must behave as one browse family with route-specific emphasis, section intros, lead artifacts, and supporting compact cards.
- Topics must render editorial concept previews with direct links to presentable concept routes.
- Decisions must render lineage-aware decision previews with direct links to presentable decision routes.
- Map must preserve URL-addressable selection, selected-object continuity, and clearly separate inbound from outbound relation storytelling.

### Concept detail

- Direct concept routes must present the canonical definition first.
- Related decisions and relations must be secondary narrative sections.
- Provenance and visible support must appear below the primary narrative rather than before it.
- Provenance, support, and source-origin details may move into structured rails, but they must remain explicit and textual.
- The reading column must avoid operator controls.

### Decision detail

- Direct decision routes must present the decision statement before supporting rationale sections.
- Problem statement, rationale, constraints, impact, and lineage must render in readable story order.
- Supersedes and superseded-by links must remain explicit through a lineage rail or equivalent presentable structure.
- Provenance and support must render in structured rails or grouped side sections rather than as stacked generic cards.
- The reading column must avoid operator controls.

### Review Studio

- Review Studio must remain capability-gated.
- Unauthorized direct visits must show a friendly access state rather than exposing raw API errors.
- Authorized actors must see queue items with clear evidence confidence, disclosure state, and action hierarchy.
- The page must separate the lead queue item from the remaining queue so the next safe action is obvious.
- Officialize and reject paths must remain visibly separate.

### Source Studio

- Source Studio must keep manager versus read-only behavior explicit.
- The bind/create flow must live in a dedicated composer panel instead of competing with the source list.
- Healthy sources and degraded sources must render in separate sections and visibly distinct operational zones.
- Operational state, locator, last-success time, and error rows must stay scannable.
- Summary band, intervention queue, composer/editor area, and healthy-monitoring area must remain visually distinct.

## Mobile and responsiveness

- Mobile reader routes must place the main artifact or answer surface before workspace metadata or secondary controls.
- Direct concept and decision routes must remain readable and presentable on narrow viewports.
- Explore map interactions must remain usable without relying on desktop-only hover behavior.
- Studio routes may stay denser than reader routes, but action labels and state pills must remain readable on mobile.

## Constraints and non-goals

- The redesign must not weaken provenance, disclosure, permission clarity, or lifecycle semantics.
- The redesign must not copy Craft branding or consumer productivity scope.
- The redesign must not require a public publishing system in this pass.
- The redesign must remain additive to the serving contract rather than replacing canonical trust vocabulary.
- The redesign must not require backend, ontology, access, provenance, lineage, or serving-contract mutations beyond the additive changes already owned by this spec.

## Verification expectations

- Contract verification must cover `workspace_home` and `DecisionPayload.public_slug`.
- `npm run build` must pass after the shared token and route composition changes.
- Frontend verification must cover workspace home, explore topics, explore decisions, explore map, direct concept routes, direct decision routes, review-studio access and queue behavior, and source-studio manager versus member behavior.
- Synthetic browser verification must cover desktop and mobile reader routes, studio gating, queue actions, and disclosure visibility.
- Trust and provenance vocabulary must remain explicit in screenshots and browser checks for every redesigned route.
- Browser artifacts for the redesign should continue landing under `output/playwright/`.

## Related docs

- [./craft-ui-ux-benchmark.md](./craft-ui-ux-benchmark.md)
- [./ui-ux-polish.md](./ui-ux-polish.md)
- [./color-system-refresh.md](./color-system-refresh.md)
- [../serving-contract/spec.md](../serving-contract/spec.md)
- [../../decisions/0018-workspace-plus-studios-navigation-separates-reader-and-studio-surfaces.md](../../decisions/0018-workspace-plus-studios-navigation-separates-reader-and-studio-surfaces.md)

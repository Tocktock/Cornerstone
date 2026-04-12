# Craft Docs Design and UI UX Benchmark

## Summary

This spec defines the design and UI UX benchmark for future Cornerstone frontend experience work using Craft Docs as the external reference product.

The benchmark started as a public-source investigation reviewed on 2026-04-09 and was extended on 2026-04-10 with a normalized route-by-route evidence workflow for Cornerstone. It is not a cloning brief. It exists to identify the product-experience qualities Craft executes well and to translate the relevant ones into Cornerstone's knowledge and review context without weakening Cornerstone's trust, provenance, access, or ontology semantics.

## Scope and owned behavior

This spec owns:
- the benchmark dimensions used to evaluate Cornerstone against Craft
- the stable conclusions from the initial Craft investigation
- the phase-two evidence convention, route mapping, and scoring rubric
- the gap framing between current Cornerstone surfaces and the benchmark target
- the design requirements that future frontend changes should satisfy when they claim to move Cornerstone closer to this benchmark

This spec does not own:
- direct visual imitation of Craft branding, illustrations, or marketing art direction
- implementation details for a specific redesign pass
- expansion of Cornerstone into a general personal productivity tool
- changes to serving-contract shapes, enum meanings, provenance semantics, lineage rules, review authorization, or connector-management permissions

## Benchmark source set

### Phase 1 public source set

The phase-one benchmark was derived from the following public sources reviewed on 2026-04-09:

- [Craft homepage](https://www.craft.do/)
- [Craft Write](https://www.craft.do/write)
- [Craft Organize](https://www.craft.do/organize)
- [Craft Customize](https://www.craft.do/customize)
- [Craft Help Center: Write and Edit](https://support.craft.do/en/write-and-edit)
- [Craft Help Center: Journaling](https://support.craft.do/hc/en-us/articles/15335736436380-Journaling)
- [Craft Help Center: Share and Publish](https://support.craft.do/hc/en-us/categories/15274247430684-Share-Publish-and-Collaborate)
- [Craft Help Center: Switching from Notion to Craft](https://support.craft.do/hc/en-us/articles/20070315420956-Switching-from-Notion-to-Craft)
- [App Store listing: Craft: Notes, Documents, AI](https://apps.apple.com/us/app/craft-notes-documents-ai/id1487937127)

The comparison baseline for Cornerstone comes from:

- [frontend/src/components/Layout.tsx](../../../frontend/src/components/Layout.tsx)
- [frontend/src/components/experience.tsx](../../../frontend/src/components/experience.tsx)
- [frontend/src/styles.css](../../../frontend/src/styles.css)
- [frontend/src/pages/WorkspacePage.tsx](../../../frontend/src/pages/WorkspacePage.tsx)
- [frontend/src/pages/ExploreTopicsPage.tsx](../../../frontend/src/pages/ExploreTopicsPage.tsx)
- [frontend/src/pages/ExploreDecisionsPage.tsx](../../../frontend/src/pages/ExploreDecisionsPage.tsx)
- [frontend/src/pages/ExploreMapPage.tsx](../../../frontend/src/pages/ExploreMapPage.tsx)
- [frontend/src/pages/ConceptDetailPage.tsx](../../../frontend/src/pages/ConceptDetailPage.tsx)
- [frontend/src/pages/DecisionDetailPage.tsx](../../../frontend/src/pages/DecisionDetailPage.tsx)
- [frontend/src/pages/ReviewStudioPage.tsx](../../../frontend/src/pages/ReviewStudioPage.tsx)
- [frontend/src/pages/SourceStudioPage.tsx](../../../frontend/src/pages/SourceStudioPage.tsx)

### Phase 2 normalized evidence set

The phase-two benchmark uses one normalized evidence tree and does not treat ad hoc `output/playwright/` artifacts as benchmark source of truth.

- artifact root: [`output/benchmarks/craft-phase-2/`](../../../output/benchmarks/craft-phase-2/)
- manifest: [`output/benchmarks/craft-phase-2/manifest.json`](../../../output/benchmarks/craft-phase-2/manifest.json)
- Cornerstone captures live under:
  - [`output/benchmarks/craft-phase-2/cornerstone/`](../../../output/benchmarks/craft-phase-2/cornerstone/)
- Craft captures live under:
  - [`output/benchmarks/craft-phase-2/craft/`](../../../output/benchmarks/craft-phase-2/craft/)

Execution state on 2026-04-10:
- Cornerstone reader routes have normalized desktop and mobile evidence.
- Cornerstone studio routes have normalized desktop evidence plus one mobile sanity capture each.
- Authenticated Craft interaction coverage is currently blocked because no authenticated Craft session is available in this workspace.
- Public Craft captures remain secondary support only and must not be treated as a substitute for authenticated in-product evidence.

## Phase 2 authenticated benchmark

### Canonical SOT guardrails

Phase two is a UI UX benchmark pass only. It must not change or redefine canonical system behavior owned elsewhere.

Treat the following as fixed inputs, not redesign targets:
- `support_visibility`
- `verification_state`
- `freshness_state`
- provenance summary semantics
- lineage semantics
- review authorization
- connector-management permissions
- serving-contract shapes and enum meanings

Do not edit or redefine canonical behavior owned by:
- [../serving-contract/spec.md](../serving-contract/spec.md)
- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)
- backend ontology, state vocabulary, or domain rules

If a benchmark finding appears to require API, schema, permission, or trust-model changes, it must be recorded as `blocked by canonical SOT` and excluded from the implementation backlog.

### Evidence naming convention

Phase two evidence must use the following stable naming rule:

- path pattern: `{surface}/{device}/{state-or-step}.png`
- reader routes require both `desktop` and `mobile`
- studio routes require `desktop` plus one `mobile` sanity capture
- any Craft artifact that is not captured from an authenticated in-product session must be labeled `public` or `unverified`

### Route mapping and benchmark depth

| Cornerstone surface | Route | Benchmark depth | Craft comparator | Protected invariant |
| --- | --- | --- | --- | --- |
| Workspace | `/` | Deep | Craft workspace or document-home behavior | provenance, trust cues, access semantics |
| Explore Topics | `/explore/topics` | Deep | Craft browse or collection navigation | trust cues remain close to definitions |
| Explore Decisions | `/explore/decisions` | Deep | Craft list-plus-preview reading behavior | lineage and verification remain explicit |
| Explore Map | `/explore/map` | Deep | Craft navigation continuity and object-centric movement, not feature parity | URL-addressable graph semantics |
| Explore Map detail | `/explore/map/:conceptId` | Deep | Craft focused object navigation and relation continuity | relation semantics and trust cues |
| Concept Detail | `/concepts/:publicSlug` | Deep | Craft document read or present surfaces | provenance summary and support visibility |
| Decision Detail | `/decisions/:publicSlug` | Deep | Craft decision-as-document presentation | lineage, provenance, and disclosure |
| Review Studio | `/review-studio` | Light | Craft hierarchy, empty-state quality, and polish only | review authorization and action safety |
| Source Studio | `/source-studio` | Light | Craft hierarchy, empty-state quality, and polish only | connector-management permissions and state semantics |

### Scored heuristic matrix rubric

Each benchmarked route must be scored from `1` to `5` against the benchmark target on:
- first-view focus
- navigation continuity
- visual hierarchy and spacing rhythm
- readability and presentability
- action discoverability
- microinteraction clarity
- transition polish
- empty-state quality
- mobile resilience

Add one non-scored gate:
- trust-cue preservation

Interpretation rules:
- `1` means materially below acceptable baseline
- `3` means functionally acceptable but unpolished
- `5` means clearly intentional and benchmark-level strong
- any route that fails trust-cue preservation cannot generate an implementation item, even if other UX scores are high

Every recorded score must include:
- an evidence artifact link
- a short reason for the score
- the Craft behavior being adapted
- the Cornerstone invariant that must remain unchanged

### Backlog filtering rules

Phase-two findings should produce UI-only backlog items. Valid implementation boundaries are:
- shared shell work in [frontend/src/components/Layout.tsx](../../../frontend/src/components/Layout.tsx)
- reusable reading and browse patterns in [frontend/src/components/experience.tsx](../../../frontend/src/components/experience.tsx)
- route-specific presentation in [frontend/src/pages/](../../../frontend/src/pages/)
- visual grammar work in [frontend/src/styles.css](../../../frontend/src/styles.css)

The phase-two backlog must not include:
- contract changes
- backend model changes
- ontology changes
- permission model changes
- requests to hide or weaken provenance, disclosure, or state vocabulary

## Benchmark conclusions

### 1. Craft is content-first before it is tool-first

Craft consistently leads with the user's artifact and intent:
- notes
- tasks
- calendar
- whiteboards
- daily notes

The product language emphasizes "write", "plan", "organize", and "customize" rather than exposing implementation categories or admin vocabulary first.

Cornerstone requirement:
- knowledge-bearing surfaces must feel primary
- workspace metadata, actor switching, and operator controls must not dominate the first viewport on read-heavy routes

### 2. Craft uses a composable block and page model

Craft presents documents as modular units that can be moved, grouped, embedded, and restyled. Public product copy and help-center guidance repeatedly reinforce blocks, pages, embeds, templates, and whiteboards as one continuous composition model.

Cornerstone requirement:
- future knowledge surfaces should move toward a clearer page-or-module grammar instead of relying only on similarly weighted cards and panels
- evidence, concepts, decisions, and summaries should read as structured content blocks, not only as operational records

### 3. Craft supports multiple parallel organization models

Craft does not force one organizing system. It explicitly supports spaces, folders, tags, and collections, and frames that flexibility as a UX strength.

Cornerstone requirement:
- navigation and information architecture should allow users to move between browse, search, overview, and deep inspection modes without feeling trapped in one rigid route structure
- route boundaries should remain clear, but the experience should feel like one connected workspace

### 4. Craft integrates planning into the writing experience

Tasks, calendar, reminders, and daily notes are presented as native extensions of the same workspace rather than as a detached task product.

Cornerstone requirement:
- action-oriented knowledge states such as review, stale sources, or decision lineage should appear as part of a coherent knowledge flow rather than as visually unrelated utility screens

### 5. Craft treats visual expression as product behavior

Craft gives styling, templates, whiteboards, and publishing-first presentation real product weight. Its public positioning makes visual pleasure and personal expression part of retention, not a decorative afterthought.

Cornerstone requirement:
- future UI work must treat typography, spacing, rhythm, and route-specific visual tone as first-class experience behavior
- the app should not rely on one uniform dark operational card treatment for every route

### 6. Craft is intentionally cross-device and mobile-capable

Craft positions capture, editing, and presentation as consistent across iPhone, iPad, Mac, web, and offline use. Mobile is not described as a degraded companion.

Cornerstone requirement:
- mobile routes should feel like deliberate knowledge workflows, not compressed desktop views
- key tasks should preserve structure, readability, and confidence on first viewport

### 7. Craft closes the loop with polished sharing and publishing

Craft treats collaboration, web publishing, branding, custom domains, and analytics as part of the document experience. The output surface matters as much as the authoring surface.

Cornerstone requirement:
- future member-facing and shareable knowledge views should be designed as presentable artifacts, not only internal operator tools

### 8. Craft positions AI as in-flow assistance, not interface replacement

Public sources frame AI as assistive: help with search, rewriting, summarization, editing, and background work while the core document experience remains legible and user-controlled.

Cornerstone requirement:
- any future AI assistance should preserve content structure, trust cues, and user orientation
- AI must augment the workspace, not obscure provenance or state

## Gap framing against current Cornerstone

### Current strengths to preserve

Cornerstone is already stronger than Craft in some product qualities that must remain visible:
- explicit provenance and trust disclosure
- role-aware access states
- operationally clear review and source-health semantics
- route-level clarity for glossary, graph, decisions, review, and sources

These are differentiators, not liabilities.

### Current gaps

#### Shell posture

Current Cornerstone leads with workspace and actor context in the shell before primary content. This is correct for operator awareness but visually pushes the knowledge artifact below the fold on smaller screens.

#### Visual range

Current Cornerstone uses one strong dark operational system across almost all routes. The palette is calmer than before, but the surface language remains highly uniform compared with Craft's mode-specific emotional range.

#### Content hierarchy

Current routes are readable but many cards share similar weight, spacing, and density. Craft uses larger hierarchy jumps, more whitespace, and stronger composition contrast between overview, detail, and supporting context.

#### Interaction model

Current Cornerstone is route-and-panel driven. Craft feels object-and-canvas driven. Cornerstone exposes well-structured data, but less of it feels like a crafted artifact a person would want to read, present, or revisit.

#### Personal adaptation

Current Cornerstone offers almost no user-controlled visual or structural adaptation outside persona switching. Craft treats templates, styles, and organization patterns as part of the core UX.

## Benchmark requirements for future Cornerstone work

Any future frontend redesign or polish pass that claims alignment with this benchmark must satisfy the following:

- keep provenance, verification, and disclosure visible even as layouts become more editorial
- reduce operator-shell dominance on knowledge-heavy routes
- distinguish knowledge-reading surfaces from operator-management surfaces more clearly
- introduce a stronger content grammar for summaries, evidence, decisions, and relation paths
- increase typographic contrast and whitespace hierarchy without reducing scanability
- allow route-specific visual tone instead of forcing one identical card treatment everywhere
- preserve mobile clarity for first-view content and key actions
- translate Craft's sense of polish and adaptability without copying its consumer-product scope

## Constraints and non-goals

- Cornerstone must not copy Craft branding, illustrations, or signature visual identity
- Cornerstone must not weaken trust semantics, provenance detail, or permission clarity in the name of minimalism
- Cornerstone must not adopt consumer task or journaling scope unless it directly supports the product mission
- Cornerstone must not replace explicit workspace structure with ambiguous aesthetic abstraction

## Related docs

- [./workspace-plus-studios-redesign.md](./workspace-plus-studios-redesign.md)
- [./ui-ux-polish.md](./ui-ux-polish.md)
- [./color-system-refresh.md](./color-system-refresh.md)
- [../../memories/frontend-experience/2026-04-10-phase-2-craft-benchmark-matrix.md](../../memories/frontend-experience/2026-04-10-phase-2-craft-benchmark-matrix.md)
- [../graph-and-relations/spec.md](../graph-and-relations/spec.md)
- [../review-and-validation/spec.md](../review-and-validation/spec.md)
- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)

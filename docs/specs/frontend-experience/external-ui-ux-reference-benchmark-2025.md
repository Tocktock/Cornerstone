# 2025 External UI UX Reference Benchmark

## Summary

This spec defines the external 2025 UI UX reference benchmark that supplements the existing Craft benchmark for Cornerstone.

The benchmark is intentionally product and data UX dominant. It uses a mixed public reference set to improve reader hierarchy, browse continuity, operational density, motion restraint, and data presentation quality without changing Cornerstone's canonical trust, provenance, access, contract, or ontology semantics.

The request summary described a portfolio of 8 sites, but the explicit source list names 9 references. This benchmark follows the explicit source list.

## Scope and owned behavior

This spec owns:
- the curated 2025 external reference portfolio used for Cornerstone UI UX comparison
- the normalized evidence tree and manifest convention
- the flow-analysis model applied to external references and Cornerstone routes
- the route-to-reference mapping used for future UI work
- the filtering rules for a UI-only implementation backlog

This spec does not own:
- product API or schema changes
- access-model changes
- provenance or lineage semantic changes
- trust-state vocabulary changes
- ontology or domain-rule changes
- direct cloning of any external site's brand, motion identity, or editorial voice

## Canonical guardrails

The following are fixed inputs for this benchmark and must not be redesign targets:
- `support_visibility`
- `verification_state`
- `freshness_state`
- provenance summary semantics
- lineage semantics
- review authorization
- connector-management permissions
- serving-contract shapes and enum meanings

Any benchmark finding that appears to require a contract, domain, permission, provenance, or trust-model change must be marked `blocked by canonical SOT` and excluded from implementation backlog work.

Canonical documents that remain authoritative:
- [../serving-contract/spec.md](../serving-contract/spec.md)
- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)

## Reference portfolio

### Track A: 2025-recognized public website references

| Source | Role in benchmark | Basis |
| --- | --- | --- |
| [Dropbox Brand](https://www.dropboxbrand.com/) | systems-and-brand-hub reference | [CSS Design Awards 2025 Website of the Year winners](https://www.cssdesignawards.com/blog/2025-website-of-the-year-winners/430/) |
| [Exat Typeface](https://exat.hottype.co/) | typography-and-CSS craft reference | [CSS Design Awards 2025 Website of the Year winners](https://www.cssdesignawards.com/blog/2025-website-of-the-year-winners/430/) |
| [ComPsych Brand Hub](https://compsych.konpo.co/) | enterprise brand-system and structured-navigation reference | [Konpo case study](https://www.konpo.studio/work/compsych), [CSSDA WOTY nominee page](https://www.cssdesignawards.com/woty2025/sites/compsych-brand-hub/) |
| [Lacoste Members Experience](https://www.awwwards.com/sites/lacoste-members-experience) | immersive navigation and customization proxy reference | [CSSDA WOTD page](https://www.cssdesignawards.com/sites/lacoste-members-experience/46965/) |
| [Sing Sing on A24](https://a24films.com/films/sing-sing/) | mobile-first narrative and cinematic pacing reference | [2025 Webby winners press release](https://www.webbyawards.com/press/press-releases/29th-annual-webby-awards-announce-2025-winners/) |
| [Pinterest Predicts](https://business.pinterest.com/pinterest-predicts/) | editorial data-storytelling reference | [2025 Webby winners press release](https://www.webbyawards.com/press/press-releases/29th-annual-webby-awards-announce-2025-winners/) |

### Track B: product and data references for Cornerstone-fit flows

| Source | Role in benchmark | Basis |
| --- | --- | --- |
| [Linear Inbox docs](https://linear.app/docs/inbox) | keyboard flow, dense action handling, and triage reference | fit-for-purpose product reference |
| [Stripe APIs docs](https://docs.stripe.com/apis) | structured technical data presentation reference | fit-for-purpose product reference |
| [Vercel Observability docs](https://vercel.com/docs/observability) | operational data surfaces and metric hierarchy reference | fit-for-purpose product reference |

## Evidence system

The canonical artifact root for this benchmark is:
- [`output/benchmarks/external-ui-ux-2025/`](../../../output/benchmarks/external-ui-ux-2025/)

The manifest lives at:
- [`output/benchmarks/external-ui-ux-2025/manifest.json`](../../../output/benchmarks/external-ui-ux-2025/manifest.json)

Required path conventions:
- `references/{source_id}/{device}/{flow}.png`
- `cornerstone/{surface}/{device}/{state}.png`

Manifest entry fields:
- `source_id`
- `source_name`
- `source_type`
- `basis`
- `captured_url`
- `device`
- `flow`
- `tags`
- `captured_at`

Allowed `source_type` values:
- `award_public`
- `product_public`
- `unverified_auth`

Execution defaults:
- every external reference should have desktop and mobile capture when public capture is available
- every Cornerstone reader route should have desktop and mobile evidence
- studio routes should have desktop evidence plus one mobile sanity capture
- auth-gated or otherwise unverifiable behavior must be labeled `unverified`, not guessed

## Flow analysis model

Every external reference and every mapped Cornerstone route should be analyzed through the same dimensions:

### CSS and visual grammar
- typography system
- spacing rhythm
- palette logic
- border and radius behavior
- shadow and depth model
- token feel
- responsive shifts

### UI structure
- header and navigation model
- layout composition
- section hierarchy
- browse-to-detail continuity

### Actions and gestures
- hover states
- disclosure patterns
- filtering patterns
- keyboard shortcuts when visible or documented
- context actions
- drag, scrub, or selection behaviors when directly verifiable

### Motion and transition behavior
- entrance patterns
- hover motion
- selected-state continuity
- loading transitions
- route transitions

### Data representation
- cards
- lists
- rails
- charts
- tables
- timelines
- metadata strips
- empty states
- error states
- compare and filter behavior

### Operational fit
Every reference conclusion must classify patterns as:
- `adopt`
- `adapt`
- `reject`

## Route mapping for Cornerstone

| Cornerstone route group | Primary references | Benchmark focus | Protected invariant |
| --- | --- | --- | --- |
| `Workspace` | Dropbox Brand, Pinterest Predicts, Linear Inbox | first-view hierarchy, content-led home, supporting action density, editorial data cues | trust, provenance, and access cues stay explicit |
| `Explore Topics` | Dropbox Brand, ComPsych Brand Hub, Linear Inbox | browse continuity, card-to-detail identity, filters, metadata without clutter | definitions and trust cues remain canonical |
| `Explore Decisions` | Dropbox Brand, ComPsych Brand Hub, Linear Inbox | decision browse continuity, lineage-aware previews, dense-but-readable metadata | lineage and verification remain explicit |
| `Explore Map` | Lacoste Members Experience, Sing Sing, Vercel Observability | object-centric navigation, selected-state continuity, path memory, restrained motion | URL-addressable map semantics remain canonical |
| `Explore Map detail` | Lacoste Members Experience, Sing Sing, Vercel Observability | active-node focus, transition continuity, relation storytelling | relation semantics and trust cues remain explicit |
| `Concept Detail` | Dropbox Brand, Exat Typeface, Stripe APIs docs | typography, narrative reading, detail rails, structured support presentation | provenance summary and support visibility remain explicit |
| `Decision Detail` | Dropbox Brand, Exat Typeface, Stripe APIs docs | readable decision story order, rail structure, support placement | lineage and provenance semantics remain explicit |
| `Review Studio` | Linear Inbox, Vercel Observability | queue hierarchy, action safety, empty-state quality, dense layout polish | review authorization and workflow safety remain unchanged |
| `Source Studio` | Vercel Observability, Stripe APIs docs, Linear Inbox | operational grouping, healthy versus degraded separation, filters, dense data representation | connector permissions and source-state vocabulary remain unchanged |

## Backlog generation rules

The resulting implementation backlog must be UI-only and grouped into:
- shell and navigation
- reader composition
- browse and map continuity
- studio density and data representation
- motion and microinteraction polish

Each backlog item must include:
- target Cornerstone route
- external reference source
- flow being borrowed
- exact frontend boundary
- protected invariant
- evidence required to accept the change

Accepted implementation boundaries:
- [frontend/src/components/Layout.tsx](../../../frontend/src/components/Layout.tsx)
- [frontend/src/components/experience.tsx](../../../frontend/src/components/experience.tsx)
- [frontend/src/pages/](../../../frontend/src/pages/)
- [frontend/src/styles.css](../../../frontend/src/styles.css)

Rejected patterns by default:
- freeform authoring or productivity expansion copied from Craft or other tools
- icon-only trust or status cues
- cinematic motion on operational studio routes
- WebGL-heavy effects on data-dense pages
- any pattern that requires changing serving-contract, access, provenance, or lineage semantics

## Verification requirements

The benchmark is complete only when all of the following are true:
- every reference in scope has at least one documented flow sheet
- every Cornerstone route in scope has a mapped reference bundle
- every reference has `adopt`, `adapt`, and `reject` notes at the flow-analysis level
- every route has UI, action, motion, and data-representation findings
- every backlog item is UI-only and explicitly names its protected invariant
- any auth-gated or otherwise unverifiable flow is marked `unverified`
- no proposed item requires changes to `support_visibility`, `verification_state`, `freshness_state`, provenance semantics, lineage semantics, review authorization, connector permissions, or serving-contract shapes

## Related docs

- [craft-ui-ux-benchmark.md](./craft-ui-ux-benchmark.md)
- [../../memories/frontend-experience/2026-04-10-phase-2-craft-benchmark-matrix.md](../../memories/frontend-experience/2026-04-10-phase-2-craft-benchmark-matrix.md)
- [../serving-contract/spec.md](../serving-contract/spec.md)
- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)

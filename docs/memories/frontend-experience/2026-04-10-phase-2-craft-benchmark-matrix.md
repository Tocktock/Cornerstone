# 2026-04-10 Craft Phase 2 Benchmark Matrix

## Summary

This note records the executed phase-two UI UX benchmark pass for Cornerstone after the workspace-plus-studios redesign.

The phase produced:
- a normalized evidence tree under [`output/benchmarks/craft-phase-2/`](../../../output/benchmarks/craft-phase-2/)
- a route-by-route scored heuristic matrix for all in-scope Cornerstone routes
- a prioritized UI-only backlog for future implementation

This phase did not change canonical serving-contract, access, trust, provenance, or ontology behavior.

## Evidence status

- Manifest: [`output/benchmarks/craft-phase-2/manifest.json`](../../../output/benchmarks/craft-phase-2/manifest.json)
- Cornerstone evidence root: [`output/benchmarks/craft-phase-2/cornerstone/`](../../../output/benchmarks/craft-phase-2/cornerstone/)
- Craft evidence root: [`output/benchmarks/craft-phase-2/craft/`](../../../output/benchmarks/craft-phase-2/craft/)

Authenticated Craft status:
- blocked in this workspace on 2026-04-10 because no authenticated Craft session was available
- public Craft captures are therefore secondary evidence only
- any Craft behavior that depends on authenticated editing, gestures, or motion is marked `unverified`, not guessed

## Route coverage

| Surface | Desktop evidence | Mobile evidence | Coverage note |
| --- | --- | --- | --- |
| Workspace | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/desktop/home.png) | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/mobile/home.png) | Deep reader benchmark |
| Explore Topics | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/desktop/index.png) | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/mobile/index.png) | Deep reader benchmark |
| Explore Decisions | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/desktop/index.png) | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/mobile/index.png) | Deep reader benchmark |
| Explore Map | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/desktop/index.png) | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/mobile/index.png) | Deep reader benchmark |
| Explore Map detail | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/desktop/partner-sla.png) | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/mobile/partner-sla.png) | Deep reader benchmark |
| Concept Detail | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/desktop/ops-playbook.png) | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/mobile/ops-playbook.png) | Deep reader benchmark |
| Decision Detail | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/desktop/legacy-escalation-routing.png) | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/mobile/legacy-escalation-routing.png) | Deep reader benchmark |
| Review Studio | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/desktop/reviewer-queue.png) | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/mobile/reviewer-queue.png) | Light coherence pass |
| Source Studio | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/desktop/operator-managed.png) | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/mobile/operator-managed.png) | Light coherence pass |

Supporting public Craft captures:
- [home desktop](../../../output/benchmarks/craft-phase-2/craft/public-home/desktop/landing.png)
- [home mobile](../../../output/benchmarks/craft-phase-2/craft/public-home/mobile/landing.png)
- [write landing](../../../output/benchmarks/craft-phase-2/craft/public-write/desktop/landing.png)
- [organize landing](../../../output/benchmarks/craft-phase-2/craft/public-organize/desktop/landing.png)
- [customize landing](../../../output/benchmarks/craft-phase-2/craft/public-customize/desktop/landing.png)
- [login gate](../../../output/benchmarks/craft-phase-2/craft/public-login/desktop/gate.png)

## Scored matrix

Scoring scale:
- `1` materially below acceptable baseline
- `3` functionally acceptable but unpolished
- `5` benchmark-level strong and intentional

### Workspace

Craft mapping:
- document-home first view
- low chrome around the primary artifact
- generous spacing and immediate reading posture

Protected invariant:
- keep `support_visibility`, `verification_state`, `freshness_state`, and provenance summary visible

| Dimension | Score | Evidence | Reason | Craft behavior being adapted | Protected invariant |
| --- | --- | --- | --- | --- | --- |
| first-view focus | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/desktop/home.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/mobile/home.png) | The hero query block and featured answer dominate the first view instead of workspace chrome. | Document-home emphasis and generous lead hierarchy. | Trust strip remains visible on the featured answer. |
| navigation continuity | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/desktop/home.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/mobile/home.png) | Header, tray, and read links keep movement coherent, though secondary panels still feel discrete. | Low-friction movement between home and artifact detail. | Access-scoped navigation remains explicit. |
| visual hierarchy and spacing rhythm | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/desktop/home.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/mobile/home.png) | The hero, answer, and recent-changes sections are clearly ranked, but repeated cards flatten the lower half. | Strong vertical rhythm from hero to supporting sections. | Disclosure pills stay present in supporting cards. |
| readability and presentability | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/desktop/home.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/mobile/home.png) | Reader mode feels shareable and artifact-first. | Presentable landing surfaces that still feel like documents. | Provenance remains in-line with reading. |
| action discoverability | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/desktop/home.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/mobile/home.png) | Search, open, and recent artifact actions are clear without becoming operator controls. | Primary actions embedded near the content. | No hidden access escalation. |
| microinteraction clarity | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/desktop/home.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/mobile/home.png) | Buttons and cards are understandable, but interaction states still feel system-like rather than crafted. | Craft-like tactile response is unverified in authenticated product; only clarity is adapted here. | Provenance and state labels remain stable during interaction. |
| transition polish | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/desktop/home.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/mobile/home.png) | Route changes are clean but not especially expressive. | Soft route continuity from home into document detail, authenticated Craft motion unverified. | Navigation state remains explicit. |
| empty-state quality | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/desktop/home.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/mobile/home.png) | Loading and no-result states are serviceable but not yet memorable or instructional. | Intentional empty states with gentle orientation. | No empty state hides trust semantics when content appears. |
| mobile resilience | 4 | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/mobile/home.png) | Mobile keeps the ask surface and answer above secondary controls. | Mobile-first reading posture. | Provenance strip survives mobile compression. |
| trust-cue preservation | pass | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/desktop/home.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/workspace/mobile/home.png) | Reader polish does not remove support, verification, or freshness cues. | Adapt low chrome without hiding trust state. | Preserve current trust vocabulary. |

### Explore Topics

Craft mapping:
- collection browsing with a strong editorial lead item

Protected invariant:
- keep definitions first and trust disclosure close behind

| Dimension | Score | Evidence | Reason | Craft behavior being adapted | Protected invariant |
| --- | --- | --- | --- | --- | --- |
| first-view focus | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/mobile/index.png) | The lead concept creates a readable entry point before the grid. | Editorial collection lead before full browse list. | Definitions and provenance stay adjacent. |
| navigation continuity | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/mobile/index.png) | Explore tabs and topic cards create a clear route family. | Smooth collection-to-document movement. | Explicit route boundaries remain. |
| visual hierarchy and spacing rhythm | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/mobile/index.png) | Lead artifact and grid are visually separated, though card repetition still lowers contrast. | Collection hero plus lighter supporting entries. | Trust cues remain visible in cards. |
| readability and presentability | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/mobile/index.png) | Definitions read cleanly and can plausibly support sharing. | Document-like browsing rather than record tables. | Definition remains canonical first text. |
| action discoverability | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/mobile/index.png) | Open actions are easy to locate and consistent. | Lightweight card actions rather than dense operator menus. | No action hides trust or access state. |
| microinteraction clarity | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/mobile/index.png) | Tabs and cards are understandable, but active and hover states remain conventional. | Crafted browse affordances; authenticated Craft collection interactions are unverified. | Trust labels remain stable under interaction. |
| transition polish | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/mobile/index.png) | Tabs work, but the route family still feels discrete rather than fluid. | Collection continuity into detail pages. | Route clarity remains explicit. |
| empty-state quality | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/mobile/index.png) | No empty collection state is visible here; current fallback pattern is competent but generic. | Calm collection empty states. | Empty-state copy must preserve canonical topic framing. |
| mobile resilience | 4 | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/mobile/index.png) | The lead card survives compression without losing the topic-first posture. | Mobile browse with readable cards. | Trust cues stay visible in mobile cards. |
| trust-cue preservation | pass | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-topics/mobile/index.png) | Reader polish does not suppress provenance or verification cues. | Use editorial browse patterns without hiding trust state. | Preserve current trust vocabulary. |

### Explore Decisions

Craft mapping:
- list-plus-preview reading behavior for artifacts that still preserve lineage

Protected invariant:
- keep lineage and verification legible in the browse surface

| Dimension | Score | Evidence | Reason | Craft behavior being adapted | Protected invariant |
| --- | --- | --- | --- | --- | --- |
| first-view focus | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/mobile/index.png) | The lead decision gives the page a readable top anchor. | Preview-led decision browsing. | Decision statement remains primary. |
| navigation continuity | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/mobile/index.png) | Tabs, cards, and lineage links create a coherent route family. | Artifact browsing that leads naturally into detail. | Lineage is preserved at browse time. |
| visual hierarchy and spacing rhythm | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/mobile/index.png) | Lead preview plus lineage rails create useful contrast, though the card grid still feels repetitive. | Preview first, supporting cards second. | Trust and lineage remain explicit. |
| readability and presentability | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/mobile/index.png) | Decision statements read as artifacts rather than admin records. | Presentable reading posture for decision documents. | No provenance loss. |
| action discoverability | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/mobile/index.png) | Open actions and lineage links are easy to spot. | Lightweight movement between preview and document. | Navigation cannot bypass disclosure. |
| microinteraction clarity | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/mobile/index.png) | Interaction states are clear but not expressive. | Craft-like list and preview tactility is unverified. | Lineage and trust labels remain stable. |
| transition polish | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/mobile/index.png) | The route family is coherent, but movement still feels route-based instead of object-based. | Smooth preview-to-document continuity. | Explicit route boundaries remain. |
| empty-state quality | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/mobile/index.png) | No empty decision set is visible here; fallback treatment is adequate but generic. | Calm empty states for collection reading. | Decision semantics stay intact. |
| mobile resilience | 4 | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/mobile/index.png) | Decision previews remain readable and lineage still fits mobile. | Mobile reading continuity for artifact collections. | Lineage remains explicit on mobile. |
| trust-cue preservation | pass | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-decisions/mobile/index.png) | The surface stays reader-first without hiding trust state. | Adapt presentable decision browsing without softening disclosure. | Preserve trust and lineage vocabulary. |

### Explore Map

Craft mapping:
- object-centric navigation continuity, not whiteboard feature parity

Protected invariant:
- keep URL-addressable graph semantics and explicit relation trust state

| Dimension | Score | Evidence | Reason | Craft behavior being adapted | Protected invariant |
| --- | --- | --- | --- | --- | --- |
| first-view focus | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/mobile/index.png) | The selected node is clear, but the route still opens like a structured inspector rather than a compelling map. | Object-first navigation entry into a connected surface. | Root concept and relation semantics remain explicit. |
| navigation continuity | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/mobile/index.png) | Jump buttons and side index work, but the route feels segmented into chips, cards, and lists. | Fluid movement between connected objects. | URL-addressability cannot be lost. |
| visual hierarchy and spacing rhythm | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/mobile/index.png) | The map has usable hierarchy, but too much neutral space and repeated list treatment lower energy. | More intentional contrast between focus, path, and context. | Relation cards still show trust cues. |
| readability and presentability | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/mobile/index.png) | It is understandable, but not yet a presentable narrative object. | Readable navigation surfaces that feel crafted, not just query output. | Explicit relation predicates remain visible. |
| action discoverability | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/mobile/index.png) | All navigation actions exist, but the primary next move is not visually obvious. | Clear next-step affordances in a connected surface. | No hidden navigation that bypasses context. |
| microinteraction clarity | 2 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/mobile/index.png) | Buttons work, but the route does not yet feel direct-manipulation friendly. | Craft-like navigation tactility is unverified; current target is clearer object transitions. | Relation state labels remain stable. |
| transition polish | 2 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/mobile/index.png) | Changing map focus feels abrupt and route-driven. | Soft continuity between connected nodes, authenticated Craft motion unverified. | URL-state and relation semantics remain explicit. |
| empty-state quality | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/mobile/index.png) | Local empty relation states are plain and informative but not spatially intentional. | Better substate messaging within object navigation. | Empty substates must still describe relation direction accurately. |
| mobile resilience | 3 | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/mobile/index.png) | Mobile remains usable, but the route becomes list-heavy quickly. | Mobile object navigation with a clearer active focus. | URL-based selection remains intact. |
| trust-cue preservation | pass | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/desktop/index.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map/mobile/index.png) | Trust cues remain present on relation cards. | Improve flow without hiding verification and support. | Preserve current relation trust vocabulary. |

### Explore Map detail

Craft mapping:
- focused object reading within a connected navigation system

Protected invariant:
- preserve node identity, directionality, and explicit trust state

| Dimension | Score | Evidence | Reason | Craft behavior being adapted | Protected invariant |
| --- | --- | --- | --- | --- | --- |
| first-view focus | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/desktop/partner-sla.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/mobile/partner-sla.png) | The selected concept is clear, but the view still opens as a dual-list inspector. | Strong object focus within a connected surface. | Node identity and relation direction remain explicit. |
| navigation continuity | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/desktop/partner-sla.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/mobile/partner-sla.png) | Moving to adjacent nodes works, but there is little retained path memory. | Object-to-object continuity in a graph-like surface. | URL-addressable detail state remains. |
| visual hierarchy and spacing rhythm | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/desktop/partner-sla.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/mobile/partner-sla.png) | Focus area, relation columns, and index have similar weight. | Stronger contrast between current node, path, and supporting index. | Trust indicators remain visible on relation cards. |
| readability and presentability | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/desktop/partner-sla.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/mobile/partner-sla.png) | The route is readable but not yet presentation-grade. | Presentable focus view for a connected object. | Relation semantics cannot be abstracted away. |
| action discoverability | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/desktop/partner-sla.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/mobile/partner-sla.png) | Possible next actions exist, but primary and secondary moves look equally weighted. | Better directional cues for next-step navigation. | No hidden access or trust state. |
| microinteraction clarity | 2 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/desktop/partner-sla.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/mobile/partner-sla.png) | Current cards do not convey much tactile hierarchy. | Tighter object navigation feedback; Craft gesture model unverified. | Keep explicit status pills. |
| transition polish | 2 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/desktop/partner-sla.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/mobile/partner-sla.png) | Focus jumps are abrupt. | Smoother selected-object continuity. | Preserve route and selection transparency. |
| empty-state quality | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/desktop/partner-sla.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/mobile/partner-sla.png) | Local empty relation states are clear but visually flat. | Better embedded empty-state treatment. | Directional relation semantics remain explicit. |
| mobile resilience | 3 | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/mobile/partner-sla.png) | Mobile preserves meaning but loses path clarity quickly. | Mobile object navigation with clearer active-node focus. | URL-selected node remains canonical. |
| trust-cue preservation | pass | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/desktop/partner-sla.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/explore-map-detail/mobile/partner-sla.png) | Trust state remains visible on relation cards. | Improve spatial flow without hiding verification state. | Preserve current relation disclosure. |

### Concept Detail

Craft mapping:
- presentable document read surface with support below the main narrative

Protected invariant:
- preserve provenance summary and visible support semantics

| Dimension | Score | Evidence | Reason | Craft behavior being adapted | Protected invariant |
| --- | --- | --- | --- | --- | --- |
| first-view focus | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/desktop/ops-playbook.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/mobile/ops-playbook.png) | Definition-led hero makes the route read as an artifact. | Document-first opening posture. | Canonical definition remains first. |
| navigation continuity | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/desktop/ops-playbook.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/mobile/ops-playbook.png) | Back link, chip references, and source summaries create clear movement. | Document reading with low-friction side references. | Trust and route boundaries remain explicit. |
| visual hierarchy and spacing rhythm | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/desktop/ops-playbook.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/mobile/ops-playbook.png) | Hero, narrative, and secondary provenance are well separated. | Presentable document rhythm. | Support sections remain visible. |
| readability and presentability | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/desktop/ops-playbook.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/mobile/ops-playbook.png) | The route is readable enough for member-facing use. | Shareable-looking document surfaces. | No suppression of provenance. |
| action discoverability | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/desktop/ops-playbook.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/mobile/ops-playbook.png) | Reference chips and back navigation are clear, but support blocks are still passive. | Lightweight in-document navigation. | Explicit support disclosure remains. |
| microinteraction clarity | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/desktop/ops-playbook.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/mobile/ops-playbook.png) | Interactions are clear but not particularly tactile. | Craft-like editing gestures are unverified; target is clearer reading affordances. | Trust labels remain stable. |
| transition polish | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/desktop/ops-playbook.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/mobile/ops-playbook.png) | Entering the route is clean, but section changes are plain. | Calm document transitions. | Route clarity remains explicit. |
| empty-state quality | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/desktop/ops-playbook.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/mobile/ops-playbook.png) | Loading and provenance fallback are competent but generic. | Better document-loading posture. | Fallbacks must not hide provenance. |
| mobile resilience | 4 | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/mobile/ops-playbook.png) | Mobile preserves the artifact-first composition well. | Mobile document reading that stays legible. | Trust strip survives mobile. |
| trust-cue preservation | pass | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/desktop/ops-playbook.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/concept-detail/mobile/ops-playbook.png) | Provenance strip and support blocks remain explicit. | Presentable document styling without removing trust cues. | Preserve support and provenance semantics. |

### Decision Detail

Craft mapping:
- decision-as-document presentation with explicit lineage rail

Protected invariant:
- preserve lineage semantics, provenance summary, and visible support disclosure

| Dimension | Score | Evidence | Reason | Craft behavior being adapted | Protected invariant |
| --- | --- | --- | --- | --- | --- |
| first-view focus | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/desktop/legacy-escalation-routing.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/mobile/legacy-escalation-routing.png) | Decision statement leads cleanly. | Document-style decision opening. | Decision statement and lineage remain explicit. |
| navigation continuity | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/desktop/legacy-escalation-routing.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/mobile/legacy-escalation-routing.png) | Back link, lineage rail, and concept chips create coherent movement. | Smooth reading between related artifacts. | Lineage semantics remain canonical. |
| visual hierarchy and spacing rhythm | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/desktop/legacy-escalation-routing.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/mobile/legacy-escalation-routing.png) | The route reads well, though the sidebar provenance stack still feels dense. | Document rhythm with secondary metadata rail. | Provenance and lineage remain visible. |
| readability and presentability | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/desktop/legacy-escalation-routing.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/mobile/legacy-escalation-routing.png) | The route is close to a presentable decision artifact. | Shareable document feel for decisions. | No trust-cue suppression. |
| action discoverability | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/desktop/legacy-escalation-routing.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/mobile/legacy-escalation-routing.png) | Navigation and linked artifacts are clear, but support cards are still scan-heavy. | In-document navigation with lighter affordances. | Explicit visible support remains. |
| microinteraction clarity | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/desktop/legacy-escalation-routing.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/mobile/legacy-escalation-routing.png) | Interactions are reliable but not refined. | Craft tactile behavior unverified; target is clearer document interactions. | Lineage and trust state stay stable. |
| transition polish | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/desktop/legacy-escalation-routing.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/mobile/legacy-escalation-routing.png) | Route entry is fine, but the page does not yet feel intentionally choreographed. | Calm document transition continuity. | Route clarity remains explicit. |
| empty-state quality | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/desktop/legacy-escalation-routing.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/mobile/legacy-escalation-routing.png) | Loading and provenance fallbacks are functional but generic. | Better document-loading posture. | Fallbacks must preserve decision semantics. |
| mobile resilience | 4 | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/mobile/legacy-escalation-routing.png) | Mobile keeps the statement, lineage, and trust strip readable. | Mobile document reading with retained context. | Lineage and trust survive compression. |
| trust-cue preservation | pass | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/desktop/legacy-escalation-routing.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/decision-detail/mobile/legacy-escalation-routing.png) | The route remains explicit about provenance, lineage, and support. | Adapt presentable decision reading without reducing disclosure. | Preserve lineage and support semantics. |

### Review Studio

Coverage note:
- light benchmark only for hierarchy, transitions, empty states, and polish
- no attempt to import Craft workflow semantics into review actions

Protected invariant:
- review authorization and action safety

| Dimension | Score | Evidence | Reason | Craft behavior being adapted | Protected invariant |
| --- | --- | --- | --- | --- | --- |
| first-view focus | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/desktop/reviewer-queue.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/mobile/reviewer-queue.png) | The queue is clear, but the page is intentionally utilitarian. | Better visual ranking of operational priority. | Review authority stays explicit. |
| navigation continuity | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/desktop/reviewer-queue.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/mobile/reviewer-queue.png) | The studio route is clearly separated from reader routes and action paths are clear. | Calm transition into an operational studio. | Authorization boundary remains visible. |
| visual hierarchy and spacing rhythm | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/desktop/reviewer-queue.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/mobile/reviewer-queue.png) | Panels are readable, but the studio still feels dense. | Better operational spacing without losing scan density. | Queue facts remain explicit. |
| readability and presentability | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/desktop/reviewer-queue.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/mobile/reviewer-queue.png) | Readable for operators, not intended as a presentable artifact. | Cleaner operational reading, not consumer polish. | Keep action semantics explicit. |
| action discoverability | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/desktop/reviewer-queue.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/mobile/reviewer-queue.png) | Officialize and reject are easy to understand. | Strong action separation. | Action safety and review role gating remain. |
| microinteraction clarity | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/desktop/reviewer-queue.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/mobile/reviewer-queue.png) | Buttons are clear, but state feedback is basic. | Cleaner operational feedback, Craft tactility unverified. | No hidden action consequences. |
| transition polish | 2 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/desktop/reviewer-queue.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/mobile/reviewer-queue.png) | Studio transitions are functional but visually abrupt. | Better operational route entry and action feedback. | Explicit studio boundary must remain. |
| empty-state quality | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/desktop/reviewer-queue.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/mobile/reviewer-queue.png) | The queue-clear and access-required states are clear and appropriately scoped. | More intentional operational empties. | Authorization messaging remains explicit. |
| mobile resilience | 3 | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/mobile/reviewer-queue.png) | Mobile works, but the cards compress quickly. | Compact but readable studio views. | Action safety remains explicit on mobile. |
| trust-cue preservation | pass | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/desktop/reviewer-queue.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/review-studio/mobile/reviewer-queue.png) | Studio polish does not weaken review state or disclosure. | Improve hierarchy without softening workflow semantics. | Preserve review authorization and state vocabulary. |

### Source Studio

Coverage note:
- light benchmark only for hierarchy, empty states, and polish

Protected invariant:
- connector-management permissions and source state semantics

| Dimension | Score | Evidence | Reason | Craft behavior being adapted | Protected invariant |
| --- | --- | --- | --- | --- | --- |
| first-view focus | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/desktop/operator-managed.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/mobile/operator-managed.png) | The route clearly communicates operations, but the first view is dense. | Better initial grouping of healthy versus attention work. | Operator permissions remain explicit. |
| navigation continuity | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/desktop/operator-managed.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/mobile/operator-managed.png) | Composer, list, and edit flows are coherent. | Cleaner transition between overview and configuration. | State semantics and permissions remain explicit. |
| visual hierarchy and spacing rhythm | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/desktop/operator-managed.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/mobile/operator-managed.png) | The route is legible, but healthy and degraded states need stronger visual separation. | More intentional grouping and rhythm in operational views. | State pills and source facts remain visible. |
| readability and presentability | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/desktop/operator-managed.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/mobile/operator-managed.png) | Readable for operators; not intended as a presentable artifact. | Cleaner studio composition, not document styling. | Preserve configuration clarity. |
| action discoverability | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/desktop/operator-managed.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/mobile/operator-managed.png) | Create, preview, edit, and sync actions are easy to locate. | Strong action placement near the relevant form or source. | No permission softening. |
| microinteraction clarity | 3 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/desktop/operator-managed.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/mobile/operator-managed.png) | Inputs and buttons are understandable, but the route still feels mechanically dense. | Better feedback during configuration work; Craft gesture model not relevant here. | Source state and permission cues remain explicit. |
| transition polish | 2 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/desktop/operator-managed.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/mobile/operator-managed.png) | The studio is functional but visually abrupt when moving between states. | Smoother studio state changes, not Craft-like document motion. | Preserve explicit operational states. |
| empty-state quality | 4 | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/desktop/operator-managed.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/mobile/operator-managed.png) | Form previews and healthy/attention splits give the route clear no-data and staging behavior. | More intentional studio empty states. | Connector-management boundaries remain clear. |
| mobile resilience | 3 | [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/mobile/operator-managed.png) | Mobile remains usable, but the edit and composer flows are compressed. | Compact studio layout that still preserves meaning. | Permissions and state vocabulary remain explicit on mobile. |
| trust-cue preservation | pass | [desktop](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/desktop/operator-managed.png) / [mobile](../../../output/benchmarks/craft-phase-2/cornerstone/source-studio/mobile/operator-managed.png) | Studio polish does not weaken source state clarity. | Improve hierarchy without hiding source health or permissions. | Preserve connector state and permission semantics. |

## Prioritized UI-only backlog

### Shell and navigation hierarchy

1. Reader header should compress sooner on member routes.
   - Target route: `/`, `/explore/*`, `/concepts/:publicSlug`, `/decisions/:publicSlug`
   - Current symptom: the global header still claims more visual attention than benchmark-level reader chrome after the first viewport
   - Craft pattern being adapted: low-chrome document framing
   - Exact UI layer boundary: [frontend/src/components/Layout.tsx](../../../frontend/src/components/Layout.tsx), [frontend/src/styles.css](../../../frontend/src/styles.css)
   - Protected invariant: actor switching and workspace scope stay accessible in the tray
   - Acceptance evidence required: refreshed desktop and mobile captures showing more vertical room for the primary artifact

2. Explore tabs need stronger route-family continuity.
   - Target route: `/explore/topics`, `/explore/decisions`, `/explore/map`
   - Current symptom: each page feels correct in isolation but the family still feels like separate pages rather than one browse system
   - Craft pattern being adapted: collection navigation with a persistent sense of place
   - Exact UI layer boundary: [frontend/src/components/experience.tsx](../../../frontend/src/components/experience.tsx), [frontend/src/styles.css](../../../frontend/src/styles.css)
   - Protected invariant: explicit route structure must remain visible
   - Acceptance evidence required: updated tab-state captures across all three explore routes

3. Reader routes need a clearer “selected artifact” continuation between browse and detail.
   - Target route: `/explore/topics`, `/explore/decisions`, `/concepts/:publicSlug`, `/decisions/:publicSlug`
   - Current symptom: moving from browse cards into detail loses too much visual continuity
   - Craft pattern being adapted: object continuity between collection and document
   - Exact UI layer boundary: [frontend/src/pages/ExploreTopicsPage.tsx](../../../frontend/src/pages/ExploreTopicsPage.tsx), [frontend/src/pages/ExploreDecisionsPage.tsx](../../../frontend/src/pages/ExploreDecisionsPage.tsx), [frontend/src/pages/ConceptDetailPage.tsx](../../../frontend/src/pages/ConceptDetailPage.tsx), [frontend/src/pages/DecisionDetailPage.tsx](../../../frontend/src/pages/DecisionDetailPage.tsx)
   - Protected invariant: route boundaries and trust cues remain explicit
   - Acceptance evidence required: paired browse/detail captures that preserve stronger visual identity across the transition

### Reader-surface composition

4. Workspace secondary column should stop repeating the same card grammar.
   - Target route: `/`
   - Current symptom: featured cards, search results, and recent changes flatten into repeated card blocks
   - Craft pattern being adapted: mixed composition where hero, summary, and supporting items feel intentionally distinct
   - Exact UI layer boundary: [frontend/src/pages/WorkspacePage.tsx](../../../frontend/src/pages/WorkspacePage.tsx), [frontend/src/components/experience.tsx](../../../frontend/src/components/experience.tsx), [frontend/src/styles.css](../../../frontend/src/styles.css)
   - Protected invariant: support, verification, and freshness cues remain attached to artifacts
   - Acceptance evidence required: new desktop and mobile workspace captures with reduced card repetition

5. Topic browse needs more contrast between the lead concept and the supporting collection.
   - Target route: `/explore/topics`
   - Current symptom: the grid still looks too similar to the lead artifact
   - Craft pattern being adapted: editorial browse lead with lighter secondary entries
   - Exact UI layer boundary: [frontend/src/pages/ExploreTopicsPage.tsx](../../../frontend/src/pages/ExploreTopicsPage.tsx), [frontend/src/components/experience.tsx](../../../frontend/src/components/experience.tsx), [frontend/src/styles.css](../../../frontend/src/styles.css)
   - Protected invariant: definitions and trust strips remain directly visible
   - Acceptance evidence required: updated topic browse captures showing stronger lead-versus-grid contrast

6. Decision detail sidebar should become a cleaner lineage-and-provenance rail.
   - Target route: `/decisions/:publicSlug`
   - Current symptom: lineage and provenance are both present but still feel dense and slightly duplicative
   - Craft pattern being adapted: secondary metadata rail beside a primary narrative column
   - Exact UI layer boundary: [frontend/src/pages/DecisionDetailPage.tsx](../../../frontend/src/pages/DecisionDetailPage.tsx), [frontend/src/components/experience.tsx](../../../frontend/src/components/experience.tsx), [frontend/src/styles.css](../../../frontend/src/styles.css)
   - Protected invariant: lineage semantics and support disclosure remain explicit
   - Acceptance evidence required: refreshed decision detail captures showing a lighter, clearer secondary rail

### Map and browse coherence

7. Explore Map root state should lead with guided relation paths, not only lists.
   - Target route: `/explore/map`
   - Current symptom: the route feels like an inspector with lists instead of a compelling connected surface
   - Craft pattern being adapted: object-centric navigation with a clearer primary next move
   - Exact UI layer boundary: [frontend/src/pages/ExploreMapPage.tsx](../../../frontend/src/pages/ExploreMapPage.tsx), [frontend/src/styles.css](../../../frontend/src/styles.css)
   - Protected invariant: graph semantics and URL-addressable selection remain unchanged
   - Acceptance evidence required: updated map root captures with a stronger focused-path presentation

8. Explore Map detail needs active-path memory.
   - Target route: `/explore/map/:conceptId`
   - Current symptom: selecting adjacent nodes does not preserve enough path context or selected-object continuity
   - Craft pattern being adapted: connected-object navigation with retained orientation
   - Exact UI layer boundary: [frontend/src/pages/ExploreMapPage.tsx](../../../frontend/src/pages/ExploreMapPage.tsx), [frontend/src/components/experience.tsx](../../../frontend/src/components/experience.tsx), [frontend/src/styles.css](../../../frontend/src/styles.css)
   - Protected invariant: relation directionality, trust state, and route transparency remain explicit
   - Acceptance evidence required: before-and-after map-detail captures showing selected-path continuity

### Studio polish and empty states

9. Review Studio needs more intentional queue-state contrast.
   - Target route: `/review-studio`
   - Current symptom: actionable items are clear, but the route lacks visual contrast between calm and urgent states
   - Craft pattern being adapted: clearer operational hierarchy and gentler empty-state treatment
   - Exact UI layer boundary: [frontend/src/pages/ReviewStudioPage.tsx](../../../frontend/src/pages/ReviewStudioPage.tsx), [frontend/src/styles.css](../../../frontend/src/styles.css)
   - Protected invariant: review authorization and action safety remain explicit
   - Acceptance evidence required: updated review-studio captures for populated and clear-queue states

10. Source Studio should separate healthy monitoring from degraded intervention more strongly.
   - Target route: `/source-studio`
   - Current symptom: healthy and attention-needed sources are functionally split but visually too similar
   - Craft pattern being adapted: calmer overview framing with a stronger intervention lane
   - Exact UI layer boundary: [frontend/src/pages/SourceStudioPage.tsx](../../../frontend/src/pages/SourceStudioPage.tsx), [frontend/src/styles.css](../../../frontend/src/styles.css)
   - Protected invariant: connector-management permissions and source state vocabulary remain explicit
   - Acceptance evidence required: updated source-studio captures showing stronger healthy-versus-attention separation

## Excluded from UI backlog

The following ideas were intentionally not turned into implementation items:
- inline block editing, drag-reorder, slash-menu authoring, and gesture-heavy document composition because authenticated Craft authoring behavior is unverified and those ideas drift toward product-scope changes
- hiding provenance, trust, or state cues to create a “cleaner” document because that would be blocked by canonical SOT
- contract or permission changes that would simplify member routes because those are outside the scope of a UI-only benchmark pass

# 0018 - Workspace plus studios navigation separates reader and studio surfaces

- **Status:** Accepted

## Context

Cornerstone's earlier frontend passes improved usability while keeping the original dashboard-plus-utility-route structure. That structure still led with shell posture, exposed operational route categories as peers of knowledge artifacts, and used one dominant dark operational treatment across almost every surface.

The Craft benchmark established a durable product requirement: member-facing knowledge surfaces should feel artifact-first and presentable, while operator-heavy workflows should remain explicit and safe without dominating the default workspace experience.

The product also needed this shift without weakening any canonical trust behavior:
- provenance visibility
- verification state
- freshness state
- support disclosure
- decision lineage

## Decision

- `Workspace` is the canonical default member-facing home route.
- `Explore` is the canonical browse surface for topics, decisions, and map exploration.
- `Review Studio` and `Source Studio` are operational routes, not equal-weight member-facing primary artifacts.
- Concept and decision details use direct presentable routes keyed by stable public slugs.
- Reader-facing routes use a lighter editorial visual family.
- Operational studio routes use a denser darker visual family.
- The shell uses a compact top header with a workspace/profile tray instead of a persistent heavy left sidebar.
- Serving-contract changes for this redesign remain additive: `workspace_home` is introduced and decision payloads gain `public_slug`.

## Consequences

- Member-facing first view now prioritizes artifact and answer surfaces over workspace chrome.
- Studio navigation is capability-gated and hidden from member-only actors.
- Compatibility redirects preserve legacy routes for one release while the new information architecture becomes canonical.
- Future frontend work should classify new surfaces as either reader-first or studio-first before choosing layout and tone.
- Existing trust and provenance semantics remain stable and must stay visible across both mode families.

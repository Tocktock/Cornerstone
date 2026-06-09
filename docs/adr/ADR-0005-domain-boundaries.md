# ADR-0005 - VS-0 Domain Boundaries

**Date:** 2026-06-09
**Status:** Accepted as setup-planning authority; no runtime implementation yet.
**Owner:** JiYong / Tars

## Context

CornerStone users experience one product, but internal ownership boundaries are required for evidence, product meaning, and external action safety.

## Decision

Preserve the three-engine boundary in the scaffold:

| Engine | Owns | Must not own |
|---|---|---|
| Product / Mission / Intelligence | Product meaning, claims, briefs, actions, approvals, workspace state, mission UX | Raw provider credentials, direct provider clients, immutable archive truth without Archive/Evidence |
| Archive / Evidence / KnowledgeBase | Immutable artifacts, hashes, provenance, derived docs/chunks/search, evidence bundles | External provider credentials, final approval authority, autonomous action execution |
| Connector / Provider / Action | Provider access, credentials, source policy, declared external actions, connector audit, retry/quarantine | Product meaning, permanent wiki truth, mission approval policy |

## Setup Rules

- Product code may propose actions; it must not directly mutate external systems.
- Archive/evidence code preserves originals before derived processing.
- Connector access is always mediated through declared capabilities.
- Agent memory is never source of truth.
- Every cross-boundary action must preserve provenance, policy, evidence, and audit refs.

## Consequences

Positive:

- The monorepo can stay one product without collapsing responsibilities.
- Future adapters from existing repos have clear import boundaries.
- External action safety remains explicit from day 0.

Costs:

- Package boundaries and tests must prevent convenience shortcuts.
- Some shared concepts, such as evidence refs and audit refs, need stable cross-package schemas.

## Non-Decision

This ADR does not define database schemas, package APIs, or connector protocols. Those belong to the scaffold implementation contract.

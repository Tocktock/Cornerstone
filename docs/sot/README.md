# CornerStone SoT Bundle

**Date:** 2026-06-08  
**Owner:** JiYong / Tars  
**Status:** Active SoT bundle after product-goal reset

## Why this bundle exists

The product goal changed. The older `project-sot.md` stated it was the “only SoT” and framed the product around Palantir/OpenClaw-class enterprise operational semantics with logistics as the first reference solution. The newer product goal defines CornerStone as a broader living, evidence-first, autonomous operational intelligence platform for personal and organizational knowledge.

This bundle resolves the conflict:

- The new **Product Goal & Direction** document is product authority.
- The new **MUST-PASS Scenario Standard** is acceptance/release authority.
- The older technical SoT is no longer product authority; compatible technical defaults are extracted into `03_TECHNICAL_ARCHITECTURE_DEFAULTS.md`.

## Authority order for CornerStone work

1. System/platform/developer instructions.
2. `AGENTS.md` and repo-local instructions.
3. `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` — product identity and direction.
4. `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` — acceptance scenarios and release gates.
5. `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` — implementation defaults where compatible.
6. Frozen scenario contract for the specific implementation task.
7. Repository code/docs/tests/logs as implementation evidence.

If lower-priority content conflicts with higher-priority content, report the conflict and follow the higher-priority source.

## Active SoT files

| File | Authority |
|---|---|
| `01_PRODUCT_GOAL_AND_DIRECTION.md` | What CornerStone is and where it is going |
| `02_MUST_PASS_SCENARIO_STANDARD.md` | What must pass before a capability/release can claim complete |
| `03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` | Safe technical defaults for the zero-base implementation |
| `04_DOCUMENT_REPLACEMENT_AND_DEPRECATION_PLAN.md` | How to replace/remove conflicting old docs |
| `sot_manifest.yaml` | Machine-readable SoT bundle index |

## New product goal

CornerStone is:

> A living, evidence-first, autonomous operational intelligence platform for personal and organizational knowledge.

It turns fragmented knowledge into:

1. durable evidence;
2. searchable/understandable context;
3. briefs, claims, decisions, and mission contracts;
4. governed actions;
5. learning loops and permanent wiki memory.

## Product non-negotiables

- One product experience, not three visible products.
- Personal-first adoption with organization expansion through explicit promotion.
- Permanent wiki / living memory with memory sovereignty.
- Owner-scoped namespaces for all context.
- Draft → Evidence-backed → Approved trust ladder.
- Orchestrator-led mission-specific agent team.
- Replaceable model brains; CornerStone owns durable value.
- ConnectorHub-mediated provider/action boundary.
- Archive/evidence foundation before generated memory.
- Scenario verification before release claims.

## First implementation target

The first implementation target is VS-0+:

`Personal messy input → immutable artifact → search → evidence-backed brief → draft/evidence-backed claim → action card dry-run → approval/execution → audit`

The full long-term scenario suite remains authoritative, but VS-0+ freezes the first slice.

## Deprecated product framing

The following phrases must not remain as current product authority:

- “This document is the only SoT” when referring to older `project-sot.md`.
- “CornerStone fully integrates Palantir-class Ontology and OpenClaw-class Agent UX” as the product identity.
- “Freight & logistics” as the core product identity or required first market.
- “Enterprise data” as the only target scope.
- Any UX/model that exposes `Cornerstone`, `KnowledgeBase`, and `ConnectorHub` as three products the user must understand.

These can remain as historical context or optional implementation inspiration only.

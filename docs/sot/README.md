# CornerStone SoT Bundle

**Date:** 2026-06-08  
**Owner:** JiYong / Tars  
**Status:** Active SoT bundle after V2 full MUST-PASS handoff

## Why This Bundle Exists

The product goal changed, and the current repository now uses the full V2 AI-agent handoff with the complete MUST-PASS scenario standard embedded and installed.

This bundle resolves authority:

- The **Product Goal & Direction** document is product authority.
- The **MUST-PASS Scenario Standard** is acceptance and release authority.
- The **Full Scenario Matrix** is the generated index for the 206 scenarios.
- The **VS-0 Implementation Contract** is only the first implementation subset.
- The older technical SoT is no longer product authority; compatible technical defaults are preserved only in `03_TECHNICAL_ARCHITECTURE_DEFAULTS.md`.

## Authority Order for CornerStone Work

1. System/platform/developer instructions.
2. Root `AGENTS.md` and repo-local instructions.
3. `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` - product identity and direction.
4. `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` - long-term product acceptance scenarios and release gates.
5. `docs/scenario-contracts/SCENARIO_MATRIX_FULL.md` - scenario index for planning and verification.
6. `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` - first implementation subset when working on VS-0.
7. `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` - implementation defaults where compatible.
8. Frozen scenario contract for the specific implementation task.
9. Repository code/docs/tests/logs as implementation evidence.

If lower-priority content conflicts with higher-priority content, report the conflict and follow the higher-priority source.

## Active SoT Files

| File | Authority |
|---|---|
| `01_PRODUCT_GOAL_AND_DIRECTION.md` | What CornerStone is and where it is going |
| `02_MUST_PASS_SCENARIO_STANDARD.md` | What must pass before a capability/release can claim complete |
| `03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` | Safe technical defaults for the zero-base implementation |
| `04_DOCUMENT_REPLACEMENT_AND_DEPRECATION_PLAN.md` | How to replace/remove conflicting old docs |
| `sot_manifest.yaml` | Machine-readable SoT bundle index |

## Scenario Contract Files

| File | Role |
|---|---|
| `../scenario-contracts/SCENARIO_MATRIX_FULL.md` | Markdown scenario index, 206 scenarios |
| `../scenario-contracts/SCENARIO_MATRIX_FULL.csv` | CSV scenario index, 206 scenarios |
| `../scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` | Strict VS-0 scenario subset, 58 scenarios |
| `../scenario-contracts/SCENARIO_VERIFICATION_REPORT_TEMPLATE.md` | Required report shape for scenario verification |

## New Product Goal

CornerStone is an **Evidence-first Operational Intelligence Platform**.

It turns fragmented knowledge into:

1. durable evidence;
2. searchable and understandable context;
3. briefs, claims, decisions, and mission contracts;
4. governed actions;
5. learning loops and permanent wiki memory.

## Product Non-Negotiables

- One product experience, not three visible products.
- Personal-first adoption with organization expansion through explicit promotion.
- Permanent wiki / living memory with memory sovereignty.
- Owner-scoped namespaces for all context.
- Draft -> Evidence-backed -> Approved trust ladder.
- Orchestrator-led mission-specific agent team.
- Replaceable model brains; CornerStone owns durable value.
- ConnectorHub-mediated provider/action boundary.
- Archive/evidence foundation before generated memory.
- Scenario verification before release claims.

## First Implementation Target

The first implementation target is VS-0:

`Personal messy input -> immutable artifact -> search -> evidence-backed brief -> claim -> action card dry-run/approval/execution -> audit`

The full long-term scenario suite remains authoritative, but VS-0 freezes the first slice.

## Deprecated Product Framing

The following phrases must not remain as current product authority:

- "This document is the only SoT" when referring to older `project-sot.md`.
- "CornerStone fully integrates Palantir-class Ontology and OpenClaw-class Agent UX" as the product identity.
- "Freight & logistics" as the core product identity or required first market.
- "Enterprise data" as the only target scope.
- Any UX/model that exposes `Cornerstone`, `KnowledgeBase`, and `ConnectorHub` as three products the user must understand.

These can remain as historical context or optional implementation inspiration only.

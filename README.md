# CornerStone

**Date:** 2026-06-08  
**Owner:** JiYong / Tars  
**Status:** Documentation authority reset with full AI-agent handoff and MUST-PASS scenarios
**Canonical spelling:** Use **CornerStone** for product/project text.

## Product Definition

CornerStone is an **Evidence-first Operational Intelligence Platform** that becomes the living knowledge and action foundation for a person, team, or organization.

It turns fragmented knowledge into:

1. durable evidence;
2. searchable and understandable context;
3. briefs, claims, decisions, and mission contracts;
4. governed actions;
5. learning loops and permanent wiki memory.

Users should experience one CornerStone product. Internally, the product keeps clear boundaries across the Product / Mission / Intelligence Engine, Archive / Evidence / KnowledgeBase Engine, and Connector / Provider / Action Engine.

## Read Order

1. `docs/handoff/AI_AGENT_HANDOFF_V2_FULL_WITH_MUST_PASS_EMBEDDED.md`
2. `docs/sot/README.md`
3. `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md`
4. `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`
5. `docs/scenario-contracts/SCENARIO_MATRIX_FULL.md`
6. `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md`
7. `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md`
8. `docs/agent/PROJECT_OPERATING_CONSTITUTION.md`
9. `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md`

## Active Authority

| File | Role |
|---|---|
| `docs/sot/README.md` | SoT authority order and index |
| `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` | Canonical product identity and direction |
| `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` | Canonical long-term scenario and release-gate standard |
| `docs/scenario-contracts/SCENARIO_MATRIX_FULL.md` | Scenario index generated from the full standard |
| `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` | Frozen first implementation subset |
| `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` | Compatible technical defaults only; not product authority |
| `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md` | Verification-centered agent workflow |
| `docs/agent/PROJECT_OPERATING_CONSTITUTION.md` | Project operating rules for agents |

## Scenario Counts

- Full scenario standard: 206 scenarios.
- VS-0 implementation subset: 58 scenarios.
- Release rule: no PASS without concrete scenario evidence.

Verify the documentation wiring with:

```sh
scripts/verify_sot_docs.sh
```

## First Implementation Target

VS-0 starts with:

```text
Personal messy input
-> immutable artifact
-> searchable derived representation
-> evidence-backed brief
-> draft/evidence-backed claim
-> action card dry-run
-> approval/execution
-> audit trail
```

The full long-term scenario suite remains authoritative. VS-0 is only the first implementation subset.

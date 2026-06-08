# CornerStone Document Replacement and Deprecation Plan

**Date:** 2026-06-08  
**Owner:** JiYong / Tars  
**Status:** Active documentation reset plan

## 1. Decision

The product goal changed. The documentation set must be reset before implementation starts.

The new source of product truth is:

1. `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md`
2. `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`

The older `project-sot.md` is no longer allowed to claim it is the only SoT. It should be archived as historical input and replaced by the new SoT bundle.

## 2. Replacement map

| Current / old document | New location / treatment | Reason |
|---|---|---|
| `cornerstone_final_product_goal_direction (1).md` | `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` | Becomes canonical product goal/direction |
| `cornerstone_must_pass_scenarios.md` | `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` | Becomes canonical scenario/release standard |
| `project-sot.md` | `docs/archive/superseded/project-sot-2026-02-17.md` | Product framing is superseded; compatible technical defaults extracted into `03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` |
| `CornerStone_Project_Custom_Instructions.md` | `AGENTS.md` and/or `docs/agent/PROJECT_OPERATING_CONSTITUTION.md` | Keep as operating guidance, not product SoT |
| `scenario_first_agent_instruction_final_en.md` | `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md` | Keep as verification process authority |
| Repo READMEs | Replace with short product intro and links to SoT bundle | Prevent drift and old product identity |
| Old roadmap/status docs | Archive or rewrite against new SoT | Avoid conflicting goals |

## 3. Content that must be removed or rewritten

Remove or rewrite current-authority statements that say:

- older `project-sot.md` is the only SoT;
- CornerStone is mainly a Palantir/OpenClaw clone/integration;
- freight/logistics is the core product identity;
- product target is only enterprise data;
- users must understand three products: Cornerstone, KnowledgeBase, ConnectorHub;
- implementation is complete because a roadmap says it is complete;
- source systems can be mutated outside Workflow/Action boundaries;
- generated/agent memory is source of truth.

These statements may be retained only inside `docs/archive/superseded/` with a superseded banner.

## 4. Content to preserve as compatible technical defaults

From older technical docs, preserve these concepts when they support the new product goal:

- Universal Artifact / original preservation.
- Postgres-first durable state.
- RLS/multi-tenancy direction.
- Postgres FTS and pgvector direction.
- OPA/Rego-compatible policy engine.
- Tamper-evident audit direction.
- Evidence Bundle and Claim model.
- ActionCard / dry-run / policy / approval / audit model.
- Default egress deny.
- Prompt-injection defenses.
- Signed/sandboxed tool direction.
- OpenAPI/AsyncAPI/CloudEvents/provenance standards where useful.
- One-command local/on-prem start.

## 5. Repository documentation target layout

```text
CornerStone/
  README.md
  AGENTS.md
  docs/
    sot/
      README.md
      01_PRODUCT_GOAL_AND_DIRECTION.md
      02_MUST_PASS_SCENARIO_STANDARD.md
      03_TECHNICAL_ARCHITECTURE_DEFAULTS.md
      04_DOCUMENT_REPLACEMENT_AND_DEPRECATION_PLAN.md
      sot_manifest.yaml
    scenario-contracts/
      VS0_IMPLEMENTATION_CONTRACT.md
    architecture/
      ONE_PRODUCT_THREE_ENGINES.md
    implementation/
      ZERO_BASE_IMPLEMENTATION_ROADMAP.md
    adr/
      ADR-0001-product-sot-reset.md
    agent/
      SCENARIO_FIRST_AGENT_INSTRUCTION.md
      PROJECT_OPERATING_CONSTITUTION.md
    archive/
      superseded/
        project-sot-2026-02-17.md
```

## 6. Apply checklist

Before coding starts:

- [ ] Add this documentation set to the repo.
- [ ] Archive old `project-sot.md` under `docs/archive/superseded/`.
- [ ] Replace any “only SoT” statement with `docs/sot/README.md` authority order.
- [ ] Update README to say CornerStone is one product with three internal engines.
- [ ] Update root agent instructions to point to the new SoT bundle.
- [ ] Create a VS-0 scenario contract before implementation.
- [ ] Add CI/report placeholder for scenario verification.
- [ ] Mark any unverified implementation claims as documented target, not implemented behavior.

## 7. Do not delete immediately unless required

Prefer archiving over immediate deletion. Old docs are useful as historical evidence and for extracting technical defaults. Delete only after:

- replacement docs are committed;
- links are updated;
- references are verified;
- no active workflow depends on the old path;
- owner approves deletion.

## 8. Final rule

When documents conflict, do not try to merge every sentence. The new product goal wins.

The implementation should move toward:

`Ingest → Understand → Decide → Act → Learn`

with evidence, owner-scoped namespaces, governed autonomy, permanent wiki memory, ConnectorHub-mediated external access, and scenario verification.

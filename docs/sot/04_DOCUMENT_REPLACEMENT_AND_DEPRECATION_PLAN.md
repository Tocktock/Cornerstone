# CornerStone Document Replacement and Deprecation Plan

**Date:** 2026-06-08  
**Owner:** JiYong / Tars  
**Status:** Active documentation reset plan for V2 full MUST-PASS handoff

## 1. Decision

The product goal changed. The documentation set must be reset before implementation starts.

The current source of product truth is:

1. `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md`
2. `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`
3. `docs/scenario-contracts/SCENARIO_MATRIX_FULL.md`
4. `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` for the first implementation subset

The older `project-sot.md` is no longer allowed to claim it is the only SoT. It may remain only as superseded historical evidence with compatible technical defaults extracted into `03_TECHNICAL_ARCHITECTURE_DEFAULTS.md`.

## 2. Replacement Map

| Current / old document | New location / treatment | Reason |
|---|---|---|
| `cornerstone_final_product_goal_direction.md` | `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` | Canonical product goal/direction |
| `cornerstone_must_pass_scenarios.md` | `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` | Canonical long-term scenario/release standard |
| V2 full handoff packet | `docs/handoff/AI_AGENT_HANDOFF_V2_FULL_WITH_MUST_PASS_EMBEDDED.md` | Preserved as the complete AI-agent handoff |
| `SCENARIO_MATRIX_FULL.md` / `.csv` | `docs/scenario-contracts/` | Machine/human scenario index for planning |
| `VS0_IMPLEMENTATION_CONTRACT_STRICT.md` | `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` | Frozen first implementation subset |
| `project-sot.md` | `docs/archive/superseded/project-sot-2026-02-17.md` or historical git record | Product framing is superseded; compatible defaults extracted into `03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` |
| `CornerStone_Project_Custom_Instructions.md` | `AGENTS.md` and/or `docs/agent/PROJECT_OPERATING_CONSTITUTION.md` | Keep as operating guidance, not product SoT |
| `scenario_first_agent_instruction_final_en.md` | `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md` | Keep as verification process authority |
| Repo READMEs | Replace with short product intro and links to SoT bundle | Prevent drift and old product identity |
| Old roadmap/status docs | Archive or rewrite against new SoT | Avoid conflicting goals |

## 3. Content That Must Be Removed or Rewritten

Remove or rewrite current-authority statements that say:

- older `project-sot.md` is the only SoT;
- CornerStone is mainly a Palantir/OpenClaw clone/integration;
- freight/logistics is the core product identity;
- product target is only enterprise data;
- users must understand three products: Cornerstone, KnowledgeBase, ConnectorHub;
- implementation is complete because a roadmap says it is complete;
- source systems can be mutated outside Workflow/Action boundaries;
- generated/agent memory is source of truth.

These statements may be retained only inside `docs/archive/superseded/` or historical git records with a superseded banner.

## 4. Content to Preserve as Compatible Technical Defaults

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

## 5. Repository Documentation Target Layout

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
      SCENARIO_MATRIX_FULL.md
      SCENARIO_MATRIX_FULL.csv
      VS0_IMPLEMENTATION_CONTRACT.md
      SCENARIO_VERIFICATION_REPORT_TEMPLATE.md
    architecture/
      ONE_PRODUCT_THREE_ENGINES.md
    implementation/
      ZERO_BASE_IMPLEMENTATION_ROADMAP.md
    adr/
      ADR-0001-product-sot-reset.md
    agent/
      SCENARIO_FIRST_AGENT_INSTRUCTION.md
      PROJECT_OPERATING_CONSTITUTION.md
    handoff/
      AI_AGENT_HANDOFF_V2_FULL_WITH_MUST_PASS_EMBEDDED.md
```

## 6. Apply Checklist

Before coding starts:

- [x] Add full V2 handoff to the repo.
- [x] Install the full product goal document as canonical SoT.
- [x] Install the full MUST-PASS scenario standard as canonical release-gate authority.
- [x] Install full scenario matrix in Markdown and CSV.
- [x] Install strict VS-0 scenario contract.
- [x] Update README to say CornerStone is one product with three internal engines.
- [x] Update root agent instructions to point to the new SoT bundle.
- [x] Add a local documentation verification script.
- [ ] Archive or preserve any old `project-sot.md` content that is not already preserved in git history.
- [ ] Add CI wiring for `scripts/verify_sot_docs.sh` when the implementation repo is ready for CI.
- [ ] Mark any unverified implementation claims as documented target, not implemented behavior.

## 7. Do Not Delete Immediately Unless Required

Prefer archiving over immediate deletion. Old docs are useful as historical evidence and for extracting technical defaults. Delete only after:

- replacement docs are committed;
- links are updated;
- references are verified;
- no active workflow depends on the old path;
- owner approves deletion.

## 8. Final Rule

When documents conflict, do not try to merge every sentence. The new product goal wins.

The implementation should move toward:

`Ingest -> Understand -> Decide -> Act -> Learn`

with evidence, owner-scoped namespaces, governed autonomy, permanent wiki memory, ConnectorHub-mediated external access, and scenario verification.

## 9. 2026-07-04 Product-Value-First Reset Amendments

Decision record: `docs/adr/ADR-0007-product-value-first-reset.md`. Review basis: live product test + implementation and verification audit.

Replacement map additions:

| Current / old framing | New location / treatment | Reason |
|---|---|---|
| README centered on scenario counts, read-order lists, and ConnectorHub trace-counter paragraphs | Rewritten `README.md` centered on spine, honest state, dormancy register, next external proof point | Front door must tell the product truth, not enumerate apparatus |
| "Evidence-first Operational Intelligence Platform" as primary external framing | Internal category language only; external framing is "briefs with receipts" (`01` Part 0) | Three abstractions; reads as generic platform marketing |
| Structural PASS counts presented as product progress | Two-plane model; Plane 1 supports at most `STRUCTURAL_READY` (`02` §2.4, `05_PRODUCT_VALUE_VERIFICATION_STANDARD.md`) | 27/28 VS4 rows passed while briefs echoed input; counts measured the harness, not value |
| Present-tense capability language for autonomy/agents/brains/learning/wiki | FUTURE-FACING (dormant) labels in `01` §0.3 + section banners; dormancy register in ADR-0007 | No such capability exists; unlabeled direction reads as claims |
| VS sequence continuing into breadth (VS2/VS3/ConnectorHub expansion) | VS5 (real brief) → VS6 (daily loop) → VS7 (wedge); VS1/VS2/VS3/ConnectorHub contracts preserved as dormant history | Value before platform; reactivation by user-evidence pull only |
| Unearned trust labels on templated output | CS-VAL-006 MUST_PASS with recorded open FAIL until VS5 fixes it | Labels that launder unverified output violate the product's own constitution |

New apply checklist (2026-07-04):

- [x] ADR-0007 committed as the direction decision record.
- [x] `05_PRODUCT_VALUE_VERIFICATION_STANDARD.md` installed as Plane 2 acceptance authority (CS-VAL family, verdict ladder).
- [x] `01` Part 0 (spine, claim boundary, section status map) + dormancy banners installed.
- [x] `02` §2.4 (two planes, family activity status) installed.
- [x] VS5/VS6/VS7 contracts + matrices installed.
- [x] README, SoT README, AGENTS.md, `03` model defaults rewritten to match.
- [ ] Fold CS-VAL rows into the generated scenario matrix at the next registry regeneration (code change; scheduled inside VS5).
- [ ] Record `VS4-H01` owner review (entry gate for VS5 external sessions).

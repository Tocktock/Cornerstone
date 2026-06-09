# CornerStone

**Date:** 2026-06-09
**Owner:** JiYong / Tars  
**Status:** Documentation authority reset with full AI-agent handoff, MUST-PASS scenarios, CLI-native gate, local verification plane, and VS-0 scaffold gate
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
6. `docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md`
7. `docs/scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv`
8. `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md`
9. `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md`
10. `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md`
11. `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md`
12. `docs/agent/PROJECT_OPERATING_CONSTITUTION.md`
13. `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md`

## Active Authority

| File | Role |
|---|---|
| `docs/sot/README.md` | SoT authority order and index |
| `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` | Canonical product identity and direction |
| `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` | Canonical long-term scenario and release-gate standard |
| `docs/scenario-contracts/SCENARIO_MATRIX_FULL.md` | Scenario index generated from the full standard |
| `docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md` | Mandatory no-CLI-no-feature-PASS execution gate |
| `docs/scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv` | Required CLI command coverage by feature family |
| `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md` | Local scenario verification, fixture corpus, model harness, validators, and release-gate contract |
| `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md` | Frozen setup-planning contract before VS-0 feature coding |
| `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` | Frozen first implementation subset |
| `docs/verification-reports/template.md` | Required report shape for scenario and CLI verification evidence |
| `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` | Compatible technical defaults only; not product authority |
| `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md` | Verification-centered agent workflow |
| `docs/agent/PROJECT_OPERATING_CONSTITUTION.md` | Project operating rules for agents |

## Scenario Counts

- Full scenario standard: 206 scenarios.
- VS-0 implementation subset: 58 scenarios.
- Release rule: no PASS without concrete scenario evidence.
- CLI-native-first rule: no feature scenario can be marked PASS unless its native `cornerstone ...` CLI path is verified or the item is explicitly classified as a non-feature implementation internal.

Verify the documentation wiring with:

```sh
scripts/verify_sot_docs.sh
scripts/verify_cli_native_first_docs.sh
scripts/verify_local_verification_plane_docs.sh
```

## First Implementation Target

Before feature coding, the VS-0 setup-planning gate is `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md`. It defines the version baseline, monorepo direction, CLI scaffold expectations, verification report shape, and human approval gates for production dependency additions.

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

## CLI Native First

Every product feature must be CLI-native first. UI and API remain important, but a user-visible, operator-visible, admin-visible, API-visible, workflow-visible, connector-visible, verification-visible, or automation-visible capability is not complete until it has a verified native `cornerstone ...` command path.

The release invariant is: **No CLI, no feature PASS.** The CLI must use the same Product / Archive / Connector / Workflow / Policy / Evidence / Audit boundaries as the UI/API and must provide scriptable `--json` output, stable exit codes, workspace/namespace scope, dry-run for mutations, evidence refs, and audit refs.

## Local Verification Plane

Local verification is a product acceptance surface, not an ad-hoc test folder. Scenario `PASS` requires deterministic evidence over product records, policy decisions, workflow/action records, audit events, CLI transcripts, UI traces where relevant, and scenario reports.

The release-facing local proof path is planned around `cornerstone scenario verify <contract> --json`. Local LLMs may help generate outputs, but deterministic validators own `PASS` or `FAIL`.

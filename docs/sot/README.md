# CornerStone SoT Bundle

**Date:** 2026-06-09
**Owner:** JiYong / Tars  
**Status:** Active SoT bundle after V2 full MUST-PASS handoff, design-system contract, VS-0 scaffold gate, local deterministic VS0 product runtime readiness, local VS0 runtime acceptance/hardening evidence, and frozen VS0 evidence-cleanup/interactive-UI-loop scenarios

## Why This Bundle Exists

The product goal changed, and the current repository now uses the full V2 AI-agent handoff with the complete MUST-PASS scenario standard embedded and installed.

This bundle resolves authority:

- The **Product Goal & Direction** document is product authority.
- The **MUST-PASS Scenario Standard** is acceptance and release authority.
- The **Full Scenario Matrix** is the generated index for the 206 scenarios.
- The **CLI Native-First Contract** is the no-CLI-no-feature-PASS execution gate.
- The **Local Verification Plane** defines local scenario verification, fixture corpus, model harness, deterministic validators, CLI-native evidence, and release gating.
- The **Design System Contract** defines the calm workspace/admin visual direction, tokens, component baseline, and design acceptance scenarios.
- The **VS-0 Scaffold Contract** is the setup-planning gate before scaffold or feature coding.
- The **VS-0 Scaffold Readiness Report** is historical scaffold-gate context; current local VS0 runtime readiness evidence lives in the VS0 Product Runtime Readiness implementation report.
- The **VS-0 Implementation Contract** is only the first implementation subset.
- The older technical SoT is no longer product authority; compatible technical defaults are preserved only in `03_TECHNICAL_ARCHITECTURE_DEFAULTS.md`.

## Authority Order for CornerStone Work

1. System/platform/developer instructions.
2. Root `AGENTS.md` and repo-local instructions.
3. `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` - product identity and direction.
4. `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` - long-term product acceptance scenarios and release gates.
5. `docs/scenario-contracts/SCENARIO_MATRIX_FULL.md` - scenario index for planning and verification.
6. `docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md` - mandatory no-CLI-no-feature-PASS execution gate.
7. `docs/scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv` - CLI command coverage by feature family.
8. `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md` - local scenario verification and release-gate plane.
9. `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md` - design-system contract.
10. `docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md` - source design concept.
11. `docs/design/tokens/cornerstone_design_tokens_v0_3.json` - canonical design tokens.
12. `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md` - setup-planning gate before VS-0 scaffold or feature coding.
13. `docs/verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md` - historical scaffold implementation readiness gate.
14. `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` - first implementation subset when working on VS-0.
15. `docs/scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md` - local runtime readiness implementation contract.
16. `docs/verification-reports/VS0_PRODUCT_RUNTIME_READINESS_REPORT_2026-06-11.md` - local runtime readiness evidence.
17. `docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_CONTRACT.md` - local runtime acceptance and hardening criteria.
18. `docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_SCENARIO_FREEZE_REPORT_2026-06-11.md` - scenario-freeze report for the next local runtime acceptance gate.
19. `docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_REPORT_2026-06-11.md` - local runtime acceptance and hardening evidence.
20. `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_CONTRACT.md` - frozen next scenario contract for evidence cleanup and interactive UI workflow proof.
21. `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_MATRIX.csv` - machine-readable EVUX scenario matrix.
22. `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` - implementation defaults where compatible.
23. Frozen scenario contract for the specific implementation task.
24. Repository code/docs/tests/logs as implementation evidence.

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
| `../scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md` | Mandatory no-CLI-no-feature-PASS execution gate |
| `../scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv` | Native CLI command coverage by feature family |
| `../scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md` | Local verification plane for scenario registry, fixture corpus, model harness, validators, transcripts, and gates |
| `../design/DESIGN_SYSTEM_CONTRACT_V0_3.md` | Applied design-system contract |
| `../design/DESIGN_CONCEPT_SYSTEM_V0_3.md` | Source concept, page model, component baseline, and design scenarios |
| `../design/tokens/cornerstone_design_tokens_v0_3.json` | Canonical design tokens |
| `../scenario-contracts/VS0_SCAFFOLD_CONTRACT.md` | Setup-planning contract before VS-0 scaffold or feature coding |
| `../verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md` | Historical readiness report for scaffold implementation gate |
| `../scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` | Strict VS-0 scenario subset, 58 scenarios |
| `../scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md` | Task-scoped VS0 runtime readiness scenarios, 14 rows |
| `../scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_MATRIX.csv` | Machine-readable matrix for the VS0 runtime readiness task contract |
| `../verification-reports/VS0_PRODUCT_RUNTIME_READINESS_SCENARIO_FREEZE_REPORT_2026-06-11.md` | Historical freeze report for VS0 runtime readiness scenarios |
| `../verification-reports/VS0_PRODUCT_RUNTIME_READINESS_REPORT_2026-06-11.md` | Current local deterministic implementation report for VS0 runtime readiness |
| `../scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_CONTRACT.md` | Task-scoped VS0 runtime acceptance and hardening criteria, 9 rows; status belongs to reports |
| `../scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_MATRIX.csv` | Machine-readable matrix for the VS0 runtime acceptance and hardening task contract |
| `../verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_SCENARIO_FREEZE_REPORT_2026-06-11.md` | Scenario-freeze report for the next VS0 runtime acceptance and hardening gate |
| `../verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_REPORT_2026-06-11.md` | Current local deterministic implementation report for VS0 runtime acceptance and hardening |
| `../scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_CONTRACT.md` | Frozen VS0 evidence cleanup and interactive UI loop scenarios, 14 rows |
| `../scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_MATRIX.csv` | Machine-readable matrix for the VS0 evidence cleanup and interactive UI loop task contract |
| `../scenario-contracts/SCENARIO_VERIFICATION_REPORT_TEMPLATE.md` | Required report shape for scenario verification |
| `../verification-reports/template.md` | Required report shape for scaffold, scenario, CLI, and human-required evidence |

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
- CLI-native-first for every product feature: no verified native `cornerstone ...` command path means no feature PASS.
- Local verification is deterministic and evidence-based: LLM output alone never proves PASS.
- Calm workspace design: light-first, evidence-aware, safe-action UI; no dark command-center or admin-first default.

## First Implementation Target

Before feature coding, `../scenario-contracts/VS0_SCAFFOLD_CONTRACT.md` must be accepted as the setup-planning gate.

Before scaffold implementation, `../verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md` must be read as historical gate context. The current local deterministic runtime loop is verified through the VS0 Product Runtime Readiness report; production release, live-provider proof, and human usability acceptance remain out of scope.

The first implementation target is VS-0:

`Personal messy input -> immutable artifact -> search -> evidence-backed brief -> claim -> action card dry-run/approval/execution -> audit`

The full long-term scenario suite remains authoritative, but VS-0 freezes the first slice. Every VS-0 feature must also be operable through the native `cornerstone ...` CLI path and verified with CLI transcript evidence.

The task-scoped runtime scenario contract after the completed local deterministic scaffold proof is `../scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md`. It contains 14 `VS0-RT-*` rows for a runnable local product runtime with CLI, API, and minimal UI parity. It does not change the canonical 206-scenario count or mark production readiness PASS.

The current local acceptance task-scoped scenario contract is `../scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_CONTRACT.md`. It contains 9 `VS0-ACC-*` rows for real browser proof, readiness evidence semantics, connector-call wording, quickstart repeatability, release evidence packaging, and overclaim regression guards. The contract is status-neutral; the local deterministic report passes 7 AI-verifiable rows and keeps 2 human-only rows as `HUMAN_REQUIRED`.

The next frozen scenario contract before implementation is `../scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_CONTRACT.md`. It contains 14 `VS0-EVUX-*` rows: 12 AI-verifiable rows that start `NOT_VERIFIED` and 2 human-only rows that remain `HUMAN_REQUIRED`. It requires actual UI workflow proof for artifact upload/select, search, evidence bundle, claim, action, mock execution, and audit timeline.

## Deprecated Product Framing

The following phrases must not remain as current product authority:

- "This document is the only SoT" when referring to older `project-sot.md`.
- "CornerStone fully integrates Palantir-class Ontology and OpenClaw-class Agent UX" as the product identity.
- "Freight & logistics" as the core product identity or required first market.
- "Enterprise data" as the only target scope.
- Any UX/model that exposes `Cornerstone`, `KnowledgeBase`, and `ConnectorHub` as three products the user must understand.

These can remain as historical context or optional implementation inspiration only.

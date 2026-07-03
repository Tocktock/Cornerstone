# CornerStone SoT Bundle

**Date:** 2026-06-09
**Owner:** JiYong / Tars  
**Status:** Active SoT bundle after V2 full MUST-PASS handoff, design-system contract, VS-0 scaffold gate, local deterministic VS0 product runtime readiness, local VS0 runtime acceptance/hardening evidence, local VS0 evidence-cleanup/interactive-UI-loop evidence, frozen VS0 EVUX clean sign-off governance scenarios, local VS0 operator acceptance UI gate evidence, final VS0 implementation closure for VS1 transition, and frozen VS4 Product Alpha UI Daily Loop documentation contract

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
20. `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_CONTRACT.md` - local evidence cleanup and interactive UI workflow proof criteria.
21. `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_MATRIX.csv` - frozen EVUX scenario matrix.
22. `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_VERIFICATION_MATRIX.csv` - current EVUX verification matrix.
23. `docs/verification-reports/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_REPORT_2026-06-13.md` - local EVUX implementation evidence.
24. `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_CONTRACT.md` - clean sign-off governance criteria for the local EVUX milestone.
25. `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_MATRIX.csv` - machine-readable VS0-GOV scenario matrix.
26. `docs/scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_CONTRACT.md` - operator acceptance UI criteria before full VS-1 implementation.
27. `docs/scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_MATRIX.csv` - machine-readable VS0-UI scenario matrix.
28. `docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REVIEW_2026-06-14.md` - human review record: operator UX acceptance accepted by JiYong/Tars.
29. `docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REPORT_2026-06-14.md` - current local deterministic implementation evidence for VS0 operator UI gate.
30. `docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_HUMAN_REVIEW_2026-06-15.md` - human acceptance evidence for `VS0-UI-H01`.
31. `docs/verification-reports/VS0_IMPLEMENTATION_FINAL_REPORT_2026-06-15.md` - final VS0 implementation closure report for VS1 transition.
32. `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` - implementation defaults where compatible.
33. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md` - Product Alpha UI daily-loop documentation contract.
34. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv` - machine-readable VS4 row inventory.
35. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md` - frozen VS4 Product Alpha Home/Ops Inbox implementation-slice contract.
36. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md` - frozen VS4 Evidence-backed Brief detail implementation-slice contract.
37. Frozen scenario contract for the specific implementation task.
38. Repository code/docs/tests/logs as implementation evidence.

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
| `../scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_MATRIX.csv` | Frozen matrix for the VS0 evidence cleanup and interactive UI loop task contract |
| `../scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_VERIFICATION_MATRIX.csv` | Current verification matrix for the VS0 evidence cleanup and interactive UI loop report |
| `../verification-reports/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_REPORT_2026-06-13.md` | Current local deterministic implementation report for VS0 evidence cleanup and interactive UI loop |
| `../scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_CONTRACT.md` | Frozen VS0 EVUX clean sign-off governance scenarios, 16 rows |
| `../scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_MATRIX.csv` | Machine-readable matrix for the VS0 EVUX clean sign-off governance task contract |
| `../scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_CONTRACT.md` | Frozen VS0 operator acceptance UI gate scenarios, 13 rows |
| `../scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_MATRIX.csv` | Frozen matrix for the VS0 operator acceptance UI gate |
| `../verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REVIEW_2026-06-14.md` | Human review record: operator UX acceptance accepted and full VS-1 unblocked |
| `../verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REPORT_2026-06-14.md` | Current local deterministic implementation evidence for the VS0 operator UI gate |
| `../verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_HUMAN_REVIEW_2026-06-15.md` | Human acceptance evidence for `VS0-UI-H01` |
| `../scenario-contracts/VS1_ONTOLOGY_AUTO_SUGGEST_PROMOTE_CONTRACT.md` | Canonical task-scoped VS-1 natural-language scenario standard for ontology auto-suggest, review, and promote |
| `../scenario-contracts/VS1_ONTOLOGY_AUTO_SUGGEST_PROMOTE_MATRIX.csv` | Machine-readable VS-1 ontology scenario matrix: 22 MUST_PASS, 10 REGRESSION_GUARD, 3 HUMAN_REQUIRED rows |
| `../scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md` | Frozen VS2 policy, tenant isolation, and default egress-deny scenario contract |
| `../scenario-contracts/VS2_POLICY_TENANCY_EGRESS_MATRIX.csv` | Machine-readable VS2 matrix: 70 MUST_PASS, 16 REGRESSION, 7 HUMAN_REQUIRED rows |
| `../verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_STATE_2026-06-19.md` | Baseline inventory, code impact map, implementation notes, and local deterministic VS2 proof links |
| `../verification-reports/VS2_POLICY_TENANCY_EGRESS_IMPLEMENTATION_REPORT_2026-06-19.md` | Superseded local deterministic VS2 implementation report |
| `../verification-reports/VS2_POLICY_TENANCY_EGRESS_SCENARIO_SPECIFIC_REMEDIATION_REPORT_2026-06-19.md` | Current scenario-specific VS2 remediation report: 7 PASS, 79 NOT_VERIFIED, 7 HUMAN_REQUIRED |
| `../scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md` | Frozen VS3 on-prem security closure and trusted extension/connector substrate contract |
| `../scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_MATRIX.csv` | Machine-readable VS3 matrix: 42 MUST_PASS, 8 REGRESSION, 7 HUMAN_REQUIRED rows |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md` | Frozen VS4 Product Alpha UI Daily Loop documentation contract; status-neutral and not implementation evidence |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv` | Machine-readable VS4 matrix: 20 MUST_PASS, 7 REGRESSION_GUARD, 1 HUMAN_REQUIRED row |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md` | Frozen VS4 Product Alpha Home/Ops Inbox shell implementation-slice contract |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md` | Frozen VS4 Evidence-backed Brief detail implementation-slice contract |
| `../verification-reports/VS0_IMPLEMENTATION_FINAL_REPORT_2026-06-15.md` | Final VS0 implementation closure report for VS1 transition |
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

The current local evidence-cleanup and interactive UI loop scenario contract is `../scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_CONTRACT.md`. It contains 14 `VS0-EVUX-*` rows for actual UI workflow proof across artifact upload/select, search, evidence bundle, claim, action, mock execution, and audit timeline. The local deterministic report passes 12 AI-verifiable rows and keeps 2 human-only rows as `HUMAN_REQUIRED`.

The clean sign-off governance scenario contract is `../scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_CONTRACT.md`. It contains 16 `VS0-GOV-*` rows for EVUX matrix/report consistency, status-neutral contract semantics, dirty-worktree metadata, command transcript evidence, release manifest hashing, final-report wording, post-commit rollup behavior, and overclaim/dependency regression guards. Current verification status is produced by `cornerstone scenario verify vs0-evux-governance` and recorded in `../scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_MATRIX.csv` plus `../verification-reports/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_REPORT_2026-06-14.md`; 2 human-only rows remain `HUMAN_REQUIRED` and production release remains false.

The VS0 operator acceptance UI gate is `../scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_CONTRACT.md`. It contains 13 `VS0-UI-*` rows for turning the one-click local EVUX proof into an understandable operator flow across Artifact, Search, Evidence, Claim, Action Card, dry-run, approval, mock execution, and Audit. Current local implementation evidence is recorded in `../verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REPORT_2026-06-14.md`, `../../reports/scenario/vs0-operator-acceptance-ui-2026-06-14.json`, and `../../reports/browser/vs0-operator-acceptance-ui-2026-06-14/`: 12 AI-verifiable rows pass. Human operator UX acceptance is recorded in `../verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_HUMAN_REVIEW_2026-06-15.md`, final closure is recorded in `../verification-reports/VS0_IMPLEMENTATION_FINAL_REPORT_2026-06-15.md`, and full VS-1 main implementation is unblocked. Production release and live-provider readiness remain unclaimed.

The VS2 policy, tenant isolation, and default egress-deny contract is `../scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md`. It contains 93 `VS2-SEC-*` rows: 70 MUST_PASS, 16 REGRESSION, and 7 HUMAN_REQUIRED. The earlier local deterministic proof report is superseded by `../verification-reports/VS2_POLICY_TENANCY_EGRESS_SCENARIO_SPECIFIC_REMEDIATION_REPORT_2026-06-19.md`. Current scenario-specific evidence is recorded in `reports/security/vs2-local-security-proof.json`, `reports/security/vs2-scenario-specific-evidence.json`, `reports/security/vs2-synthetic-world.json`, and `reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json`: 7 PASS, 79 NOT_VERIFIED, and 7 HUMAN_REQUIRED. The native verifier is `cornerstone scenario verify vs2-policy-tenancy-egress --json`. The verifier may not claim VS2 readiness until all 70 MUST_PASS and 16 REGRESSION rows pass with scenario-specific evidence; production security, real IdP, production network, live-provider, human UX, and migration/restore readiness remain blocked by H02-H07 evidence.

The VS3 on-prem security and trusted extension/connector substrate contract is `../scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md`. It contains 57 `VS3-*` rows: 42 MUST_PASS, 8 REGRESSION, and 7 HUMAN_REQUIRED. VS3 starts by reconciling the conflicting VS2 evidence boundary, then carries forward trusted RequestContext, Postgres/RLS, OPA/Rego, default-deny egress, ConnectorHub source safety, Tool SDK/signed registry, Agent Pack activation, operator status, audit, and human-gate evidence. The contract is status-neutral: all AI-verifiable rows start as NOT_RUN, and VS3 may not claim production/on-prem readiness until VS3-H01 through VS3-H07 have dated human/on-prem evidence.

The VS4 Product Alpha UI Daily Loop contract is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`. It contains 28 `VS4-*` rows: 20 MUST_PASS, 7 REGRESSION_GUARD, and 1 HUMAN_REQUIRED. VS4 is the Product Alpha / Daily Loop documentation contract for turning the verified local evidence/action engine into a daily-use product shell around Drop / Ask, Evidence-backed Briefs, Claims, Memory/Wiki, Action Cards, Ops Inbox, Evidence/Audit, and Learn. The parent contract is status-neutral and does not update the canonical 206-scenario matrix or claim implementation evidence. The first implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`, which selects 7 AI-verifiable rows for local Product Alpha Home/Ops Inbox shell proof and leaves full VS4 completion unclaimed. The second implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`, which selects 14 AI-verifiable rows for Evidence-backed Brief detail, claim/memory/action review boundaries, prompt-injection guard, reference-boundary guard, and CLI parity proof. VS3-H01 through VS3-H07 remain conditional deferred gates for production/on-prem/security/live-provider/human acceptance claims and do not block local VS4 documentation or implementation planning.

## Deprecated Product Framing

The following phrases must not remain as current product authority:

- "This document is the only SoT" when referring to older `project-sot.md`.
- "CornerStone fully integrates Palantir-class Ontology and OpenClaw-class Agent UX" as the product identity.
- "Freight & logistics" as the core product identity or required first market.
- "Enterprise data" as the only target scope.
- Any UX/model that exposes `Cornerstone`, `KnowledgeBase`, and `ConnectorHub` as three products the user must understand.

These can remain as historical context or optional implementation inspiration only.

# CornerStone SoT Bundle

**Date:** 2026-07-04
**Owner:** JiYong / Tars  
**Status:** Active SoT bundle after the 2026-07-04 product-value-first reset (`docs/adr/ADR-0007-product-value-first-reset.md`). VS0–VS4 structural evidence is history (`STRUCTURAL_READY`); the active milestone is VS5 (citation-grounded Brief); product-value claims require Plane 2 evidence.

## Why This Bundle Exists

The product goal changed, and the current repository now uses the full V2 AI-agent handoff with the complete MUST-PASS scenario standard embedded and installed. The 2026-07-04 reset added the second verification plane and the VS5–VS7 milestone sequence after a product review proved structural gates alone cannot measure product value.

This bundle resolves authority:

- **ADR-0007** (`docs/adr/ADR-0007-product-value-first-reset.md`) records the active spine (`Drop / Ask -> Brief -> Decision -> Audit`), the scope freeze, and the dormancy register.
- The **Product Value Verification Standard** (`05_PRODUCT_VALUE_VERIFICATION_STANDARD.md`) is the Plane 2 acceptance authority (CS-VAL family, verdict ladder); structural PASS counts alone support at most `STRUCTURAL_READY`.
- The **VS5 / VS6 / VS7 contracts** (`../scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_CONTRACT.md`, `VS6_DAILY_LOOP_CONTRACT.md`, `VS7_WEDGE_VALIDATION_CONTRACT.md`) define the active milestone sequence.

- The **Product Goal & Direction** document is product authority.
- The **MUST-PASS Scenario Standard** is acceptance and release authority.
- The **Full Scenario Matrix** is the generated index for the 216 scenarios, including the ten CS-VAL Plane 2 rows.
- The **CLI Native-First Contract** is the no-CLI-no-feature-PASS execution gate.
- The **Local Verification Plane** defines local scenario verification, fixture corpus, model harness, deterministic validators, CLI-native evidence, and release gating.
- The **Design System Contract** defines the calm workspace/admin visual direction, tokens, component baseline, and design acceptance scenarios.
- The **VS-0 Scaffold Contract** is the setup-planning gate before scaffold or feature coding.
- The **VS-0 Scaffold Readiness Report** is historical scaffold-gate context; current local VS0 runtime readiness evidence lives in the VS0 Product Runtime Readiness implementation report.
- The **VS-0 Implementation Contract** is only the first implementation subset.
- The older technical SoT is no longer product authority; compatible technical defaults are preserved only in `03_TECHNICAL_ARCHITECTURE_DEFAULTS.md`.

## Authority Order for CornerStone Work

**2026-07-04 precedence amendment:** immediately after `AGENTS.md`, the following take precedence over everything below: `docs/adr/ADR-0007-product-value-first-reset.md` (active spine, scope freeze, dormancy register), `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` **Part 0** (binding claim boundary), `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md` (Plane 2 acceptance), and the active milestone contract `docs/scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_CONTRACT.md` (then `VS6_DAILY_LOOP_CONTRACT.md`, `VS7_WEDGE_VALIDATION_CONTRACT.md`). The numbered list below remains the historical authority order for structural work; items covering dormant systems are read as history, not active product authority.

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
37. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md` - frozen VS4 Ask, packs, states, reference alignment, and regression implementation-slice contract.
38. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md` - frozen VS4 human acceptance package and validation contract.
39. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_005_UX_POLISH_LEARN.md` - frozen VS4 UX polish and Learn review implementation-slice contract.
40. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_006_RESPONSIVE_MOBILE_PROOF.md` - frozen VS4 responsive mobile proof implementation-slice contract.
41. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_007_KEYBOARD_FOCUS_REVIEW.md` - frozen VS4 keyboard/focus review implementation-slice contract.
42. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_008_ASK_RESULT_READABILITY.md` - frozen VS4 Ask result readability implementation-slice contract.
43. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_009_CLAIM_ACTION_NAV_DETAIL.md` - frozen VS4 Claim and Action nav-detail implementation-slice contract.
44. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_010_ASK_INJECTION_BOUNDARY.md` - frozen VS4 Ask injection-boundary implementation-slice contract.
45. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_011_OPS_INBOX_TRIAGE_DETAIL.md` - frozen VS4 Ops Inbox triage/detail implementation-slice contract.
46. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_012_ACTION_EXECUTION_BOUNDARY.md` - frozen VS4 Action execution-boundary implementation-slice contract.
47. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_013_DESKTOP_OVERFLOW_CONTAINMENT.md` - frozen VS4 desktop overflow-containment implementation-slice contract.
48. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_014_HUMAN_REVIEW_HANDOFF.md` - frozen VS4 human-review handoff implementation-slice contract.
49. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_015_GATE_INTEGRITY.md` - frozen VS4 scenario-gate integrity implementation-slice contract.
50. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_016_EVIDENCE_AUDIT_DETAIL.md` - frozen VS4 Evidence/Audit detail implementation-slice contract.
51. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_017_USER_DROP_ASK_SOURCE.md` - frozen VS4 user Drop/Ask source implementation-slice contract.
52. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_018_DROP_ASK_TRUST_BOUNDARY.md` - frozen VS4 Drop/Ask trust-boundary implementation-slice contract.
53. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_019_INTERACTIVE_OPS_INBOX.md` - frozen VS4 interactive Ops Inbox implementation-slice contract.
54. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_020_RUNTIME_BACKED_OPS_INBOX.md` - frozen VS4 runtime-backed Ops Inbox implementation-slice contract.
55. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_021_RUNTIME_LOOP_COHERENCE.md` - frozen VS4 runtime loop coherence implementation-slice contract.
56. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_022_RETURN_TO_WORK_LINEAGE_GUARD.md` - frozen VS4 return-to-work lineage guard implementation-slice contract.
57. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_023_REPORT_PACKAGE_INTEGRITY.md` - frozen VS4 report/package integrity implementation-slice contract.
58. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_024_ACTIVE_REPORT_PACKAGE_COHERENCE.md` - frozen VS4 active report/package coherence implementation-slice contract.
59. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_025_OPS_INBOX_JOURNEY_TIMELINE.md` - frozen VS4 Ops Inbox journey-timeline implementation-slice contract.
60. `docs/verification-reports/VS4_PRODUCT_ALPHA_CLOSURE_CHECKPOINT_2026-07-04.md` - local AI-verifiable VS4 Product Alpha closure checkpoint with `VS4-H01` still human-required.
61. Frozen scenario contract for the specific implementation task.
62. Repository code/docs/tests/logs as implementation evidence.

If lower-priority content conflicts with higher-priority content, report the conflict and follow the higher-priority source.

## Active SoT Files

| File | Authority |
|---|---|
| `01_PRODUCT_GOAL_AND_DIRECTION.md` | What CornerStone is and where it is going (Part 0 is the binding claim boundary) |
| `02_MUST_PASS_SCENARIO_STANDARD.md` | Plane 1: what must pass structurally before a capability/release can claim `STRUCTURAL_READY` |
| `05_PRODUCT_VALUE_VERIFICATION_STANDARD.md` | Plane 2: what must pass before any product-value claim (CS-VAL, verdict ladder) |
| `03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` | Safe technical defaults for the zero-base implementation |
| `04_DOCUMENT_REPLACEMENT_AND_DEPRECATION_PLAN.md` | How to replace/remove conflicting old docs |
| `sot_manifest.yaml` | Machine-readable SoT bundle index |

## Scenario Contract Files

| File | Role |
|---|---|
| `../scenario-contracts/SCENARIO_MATRIX_FULL.md` | Markdown scenario index, 216 scenarios |
| `../scenario-contracts/SCENARIO_MATRIX_FULL.csv` | CSV scenario index, 216 scenarios |
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
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md` | Frozen VS4 Ask, general-purpose packs, page states, reference alignment, and regression implementation-slice contract |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md` | Frozen VS4 human acceptance package and validation contract; keeps `VS4-H01` HUMAN_REQUIRED |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_005_UX_POLISH_LEARN.md` | Frozen VS4 UX polish and Learn review contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_006_RESPONSIVE_MOBILE_PROOF.md` | Frozen VS4 responsive mobile proof contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_007_KEYBOARD_FOCUS_REVIEW.md` | Frozen VS4 keyboard/focus review contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_008_ASK_RESULT_READABILITY.md` | Frozen VS4 Ask result readability contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_009_CLAIM_ACTION_NAV_DETAIL.md` | Frozen VS4 Claim and Action nav-detail contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_010_ASK_INJECTION_BOUNDARY.md` | Frozen VS4 Ask injection-boundary contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_011_OPS_INBOX_TRIAGE_DETAIL.md` | Frozen VS4 Ops Inbox triage/detail contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_012_ACTION_EXECUTION_BOUNDARY.md` | Frozen VS4 Action execution-boundary contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_013_DESKTOP_OVERFLOW_CONTAINMENT.md` | Frozen VS4 desktop overflow-containment contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_014_HUMAN_REVIEW_HANDOFF.md` | Frozen VS4 human-review handoff contract; keeps `VS4-H01` HUMAN_REQUIRED |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_015_GATE_INTEGRITY.md` | Frozen VS4 scenario-gate integrity contract; keeps the canonical VS4 matrix unchanged and rejects overclaims or weak evidence |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_016_EVIDENCE_AUDIT_DETAIL.md` | Frozen VS4 Evidence/Audit detail contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_017_USER_DROP_ASK_SOURCE.md` | Frozen VS4 user Drop/Ask source contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_018_DROP_ASK_TRUST_BOUNDARY.md` | Frozen VS4 Drop/Ask trust-boundary contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_019_INTERACTIVE_OPS_INBOX.md` | Frozen VS4 interactive Ops Inbox contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_020_RUNTIME_BACKED_OPS_INBOX.md` | Frozen VS4 runtime-backed Ops Inbox contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_021_RUNTIME_LOOP_COHERENCE.md` | Frozen VS4 runtime loop coherence contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_022_RETURN_TO_WORK_LINEAGE_GUARD.md` | Frozen VS4 return-to-work lineage guard contract; keeps the canonical VS4 matrix unchanged |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_023_REPORT_PACKAGE_INTEGRITY.md` | Frozen VS4 report/package integrity contract; keeps filtered slice evidence separate from the full H01 package report |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_024_ACTIVE_REPORT_PACKAGE_COHERENCE.md` | Frozen VS4 active report/package coherence contract; keeps active slice metadata and H01 package review inputs aligned |
| `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_025_OPS_INBOX_JOURNEY_TIMELINE.md` | Frozen VS4 Ops Inbox journey-timeline contract; turns selected work into a runtime-backed staged daily loop |
| `../verification-reports/VS4_PRODUCT_ALPHA_CLOSURE_CHECKPOINT_2026-07-04.md` | VS4 Product Alpha closure checkpoint; records local AI-verifiable readiness and keeps `VS4-H01` human-required |
| `../verification-reports/VS0_IMPLEMENTATION_FINAL_REPORT_2026-06-15.md` | Final VS0 implementation closure report for VS1 transition |
| `../scenario-contracts/SCENARIO_VERIFICATION_REPORT_TEMPLATE.md` | Required report shape for scenario verification |
| `../verification-reports/template.md` | Required report shape for scaffold, scenario, CLI, and human-required evidence |

## Current Product Direction (2026-07-04)

The active spine is **`Drop / Ask -> Evidence-backed Brief -> Decision -> Audit`** — externally: *briefs with receipts*. The long-term ambition (durable evidence, understandable context, governed actions, learning loops, permanent memory) is preserved as labeled future-facing direction in `01_PRODUCT_GOAL_AND_DIRECTION.md`; "Evidence-first Operational Intelligence Platform" is internal category language, not user-facing copy.

Current honest state: the structural substrate is verified (`STRUCTURAL_READY`); model-backed understanding does not exist yet and is the VS5 build; four CS-VAL rows are recorded as open FAILs until VS5 evidence flips them; no external user has used the product yet. The next proof point is the VS5 stranger test.

## Product Non-Negotiables

Active now:

- One product experience, not three visible products.
- Personal-first adoption with organization expansion through explicit promotion.
- Owner-scoped namespaces for all context (data model; multi-tenant enforcement dormant until real tenants).
- Draft -> Evidence-backed -> Approved trust ladder — **labels must be earned by citation-integrity checks (CS-VAL-006); unearned labels are a violation, not a feature.**
- Archive/evidence foundation before generated memory.
- Scenario verification before release claims — **both planes: structural (Plane 1) and product value (Plane 2); Plane 1 alone supports at most `STRUCTURAL_READY`.**
- CLI-native-first for every product feature: no verified native `cornerstone ...` command path means no feature PASS.
- Local verification: deterministic validators own structural PASS; LLM judges are advisory only and never flip a row; humans own subjective quality PASS.
- Local-first model stack: Ollama `ornith:9b` (generation) and `qwen3-embedding:0.6b` (embeddings) are the default assumption; `ornith:35b` is opt-in for explicitly named larger-model tests; external providers are optional and named per-scenario.
- Calm workspace design: light-first, evidence-aware, safe-action UI; no dark command-center or admin-first default; internal tokens never as user-facing copy.

Preserved direction, dormant until user-evidence pull (ADR-0007; disposition in VS7):

- Permanent wiki / living memory with memory sovereignty.
- Orchestrator-led mission-specific agent team.
- Multi-brain routing/ensembles (the *replaceable brain* principle stands; replaceable never means omittable).
- ConnectorHub-mediated provider/action boundary (the mediation principle governs any future action path; the hub expansion itself is dormant).

## Active Implementation Target and Historical Structural Record

The active implementation target after the 2026-07-04 reset is **VS5**:

`Drop / Ask -> citation-grounded Brief -> direct Ask answer -> earned trust labels -> external stranger test`

The frozen contract is `../scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_CONTRACT.md`; the matrix is `../scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_MATRIX.csv`. VS5 remains blocked from external sessions until `VS4-H01` is recorded, and completion claims require Plane 2 evidence from `05_PRODUCT_VALUE_VERIFICATION_STANDARD.md`.

The following VS0-VS4 entries are retained as the historical structural record. They are still useful when touching the substrate they froze, but they are not the current product target.

Before historical VS0 scaffold work, `../scenario-contracts/VS0_SCAFFOLD_CONTRACT.md` and `../verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md` define the closed setup-planning gate. The deterministic runtime loop is verified through the VS0 Product Runtime Readiness report; production release, live-provider proof, and human usability acceptance remain out of scope.

The historical VS0 implementation slice was:

`Personal messy input -> immutable artifact -> search -> evidence-backed brief -> claim -> action card dry-run/approval/execution -> audit`

The full long-term scenario suite remains authoritative, but VS0 froze the first structural slice. Every VS0 feature also had to be operable through the native `cornerstone ...` CLI path and verified with CLI transcript evidence.

The task-scoped runtime scenario contract after the completed local deterministic scaffold proof is `../scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md`. It contains 14 `VS0-RT-*` rows for a runnable local product runtime with CLI, API, and minimal UI parity. It did not change the then-current 206-scenario count and does not mark production readiness PASS; the current matrix is 216 after the VS5 CS-VAL fold-in.

The historical local acceptance task-scoped scenario contract is `../scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_CONTRACT.md`. It contains 9 `VS0-ACC-*` rows for real browser proof, readiness evidence semantics, connector-call wording, quickstart repeatability, release evidence packaging, and overclaim regression guards. The contract is status-neutral; the local deterministic report passes 7 AI-verifiable rows and keeps 2 human-only rows as `HUMAN_REQUIRED`.

The current local evidence-cleanup and interactive UI loop scenario contract is `../scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_CONTRACT.md`. It contains 14 `VS0-EVUX-*` rows for actual UI workflow proof across artifact upload/select, search, evidence bundle, claim, action, mock execution, and audit timeline. The local deterministic report passes 12 AI-verifiable rows and keeps 2 human-only rows as `HUMAN_REQUIRED`.

The clean sign-off governance scenario contract is `../scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_CONTRACT.md`. It contains 16 `VS0-GOV-*` rows for EVUX matrix/report consistency, status-neutral contract semantics, dirty-worktree metadata, command transcript evidence, release manifest hashing, final-report wording, post-commit rollup behavior, and overclaim/dependency regression guards. Historical verification status is produced by `cornerstone scenario verify vs0-evux-governance` and recorded in `../scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_MATRIX.csv` plus `../verification-reports/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_REPORT_2026-06-14.md`; 2 human-only rows remain `HUMAN_REQUIRED` and production release remains false.

The VS0 operator acceptance UI gate is `../scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_CONTRACT.md`. It contains 13 `VS0-UI-*` rows for turning the one-click local EVUX proof into an understandable operator flow across Artifact, Search, Evidence, Claim, Action Card, dry-run, approval, mock execution, and Audit. Historical local implementation evidence is recorded in `../verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REPORT_2026-06-14.md`, `../../reports/scenario/vs0-operator-acceptance-ui-2026-06-14.json`, and `../../reports/browser/vs0-operator-acceptance-ui-2026-06-14/`: 12 AI-verifiable rows pass. Human operator UX acceptance is recorded in `../verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_HUMAN_REVIEW_2026-06-15.md`, final closure is recorded in `../verification-reports/VS0_IMPLEMENTATION_FINAL_REPORT_2026-06-15.md`, and full VS-1 main implementation is unblocked. Production release and live-provider readiness remain unclaimed.

The VS2 policy, tenant isolation, and default egress-deny contract is `../scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md`. It contains 93 `VS2-SEC-*` rows: 70 MUST_PASS, 16 REGRESSION, and 7 HUMAN_REQUIRED. The earlier local deterministic proof report is superseded by `../verification-reports/VS2_POLICY_TENANCY_EGRESS_SCENARIO_SPECIFIC_REMEDIATION_REPORT_2026-06-19.md`. Current scenario-specific evidence is recorded in `reports/security/vs2-local-security-proof.json`, `reports/security/vs2-scenario-specific-evidence.json`, `reports/security/vs2-synthetic-world.json`, and `reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json`: 7 PASS, 79 NOT_VERIFIED, and 7 HUMAN_REQUIRED. The native verifier is `cornerstone scenario verify vs2-policy-tenancy-egress --json`. The verifier may not claim VS2 readiness until all 70 MUST_PASS and 16 REGRESSION rows pass with scenario-specific evidence; production security, real IdP, production network, live-provider, human UX, and migration/restore readiness remain blocked by H02-H07 evidence.

The VS3 on-prem security and trusted extension/connector substrate contract is `../scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md`. It contains 57 `VS3-*` rows: 42 MUST_PASS, 8 REGRESSION, and 7 HUMAN_REQUIRED. VS3 starts by reconciling the conflicting VS2 evidence boundary, then carries forward trusted RequestContext, Postgres/RLS, OPA/Rego, default-deny egress, ConnectorHub source safety, Tool SDK/signed registry, Agent Pack activation, operator status, audit, and human-gate evidence. The contract is status-neutral: all AI-verifiable rows start as NOT_RUN, and VS3 may not claim production/on-prem readiness until VS3-H01 through VS3-H07 have dated human/on-prem evidence.

The VS4 Product Alpha UI Daily Loop contract is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`. It contains 28 `VS4-*` rows: 20 MUST_PASS, 7 REGRESSION_GUARD, and 1 HUMAN_REQUIRED. VS4 is the Product Alpha / Daily Loop documentation contract for turning the verified local evidence/action engine into a daily-use product shell around Drop / Ask, Evidence-backed Briefs, Claims, Memory/Wiki, Action Cards, Ops Inbox, Evidence/Audit, and Learn. The parent contract is status-neutral and does not update the canonical 206-scenario matrix or claim implementation evidence. The first implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`, which selects 7 AI-verifiable rows for local Product Alpha Home/Ops Inbox shell proof and leaves full VS4 completion unclaimed. The second implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`, which selects 14 AI-verifiable rows for Evidence-backed Brief detail, claim/memory/action review boundaries, prompt-injection guard, reference-boundary guard, and CLI parity proof. The third implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md`, which selects the remaining 6 AI-verifiable rows for Ask-to-work-item, three general-purpose packs, required page states, Home/Search/Artifact reference alignment, and fresh VS0/VS1 regression proof. The fourth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md`, which prepares `VS4-H01` review package, blank reviewer template, and structural validator while keeping human UX acceptance unclaimed. The fifth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_005_UX_POLISH_LEARN.md`, which strengthens product-language, Learn-review, and progressive proof-detail readiness without adding canonical matrix rows. The sixth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_006_RESPONSIVE_MOBILE_PROOF.md`, which adds deterministic narrow-viewport browser proof and responsive containment markers without adding canonical matrix rows. The seventh implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_007_KEYBOARD_FOCUS_REVIEW.md`, which adds deterministic keyboard/focus review proof without adding canonical matrix rows. The eighth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_008_ASK_RESULT_READABILITY.md`, which makes the Ask result readable before raw refs while preserving evidence refs in progressive detail without adding canonical matrix rows. The ninth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_009_CLAIM_ACTION_NAV_DETAIL.md`, which turns the normal `Claims` and `Actions` nav destinations into product-ready review pages without adding canonical matrix rows. The tenth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_010_ASK_INJECTION_BOUNDARY.md`, which proves unsafe Ask text cannot approve claims, approve memory, execute actions, call providers, change policy, or expand authority without adding canonical matrix rows. The eleventh implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_011_OPS_INBOX_TRIAGE_DETAIL.md`, which adds Ops Inbox triage lanes and selected-item detail for returning daily work without adding canonical matrix rows. The twelfth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_012_ACTION_EXECUTION_BOUNDARY.md`, which freezes Action Card unauthorized approval/execution denial, safety-envelope evidence, audit refs, and zero provider/writeback side-effect requirements without adding canonical matrix rows. The thirteenth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_013_DESKTOP_OVERFLOW_CONTAINMENT.md`, which contains desktop Product Alpha overflow by wrapping long policy, safety-envelope, evidence, and audit tokens without adding canonical matrix rows. The fourteenth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_014_HUMAN_REVIEW_HANDOFF.md`, which makes the `VS4-H01` human-review handoff visible in Product Alpha Home/Ops Inbox while keeping acceptance `HUMAN_REQUIRED`. The fifteenth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_015_GATE_INTEGRITY.md`, which adds VS4-specific scenario-gate integrity checks for overclaims, human-required status, reference-image boundaries, CLI parity, negative evidence, and source-tree metadata without adding canonical matrix rows. The sixteenth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_016_EVIDENCE_AUDIT_DETAIL.md`, which makes Evidence/Audit a product-readable detail surface with source, provenance, safety, activity, audit verification, and Learn linkage without adding canonical matrix rows. The seventeenth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_017_USER_DROP_ASK_SOURCE.md`, which makes the primary Drop/Ask path preserve user-pasted source text and create the Evidence-backed Brief from that source instead of a fixed fixture, with CLI `artifact ingest --text` parity and no canonical matrix rows. The eighteenth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_018_DROP_ASK_TRUST_BOUNDARY.md`, which hardens Drop/Ask trust boundaries so user-pasted and conversation text stay untrusted through HTTP/API intake and same-checksum CLI dedupe, with structured unsafe-promotion denial and no canonical matrix rows. The nineteenth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_019_INTERACTIVE_OPS_INBOX.md`, which makes Ops Inbox lane/item selection interactive and adds CLI selected-item plus loop-view parity without adding canonical matrix rows. The twentieth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_020_RUNTIME_BACKED_OPS_INBOX.md`, which refreshes the Ops Inbox from runtime Brief, Claim, Memory/Wiki, and Action records without adding canonical matrix rows. The twenty-first implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_021_RUNTIME_LOOP_COHERENCE.md`, which makes Learn runtime-backed through trajectory/lesson refs, normalizes Action approval copy, and binds active proof paths to the active slice without adding canonical matrix rows. The twenty-second implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_022_RETURN_TO_WORK_LINEAGE_GUARD.md`, which makes return-to-work Loop View reject missing, cross-scope, or mismatched refs before presenting one evidence-backed journey without adding canonical matrix rows. The twenty-third implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_023_REPORT_PACKAGE_INTEGRITY.md`, which keeps focused slice reports separate from the canonical full VS4 report used by `VS4-H01` human-package evidence without adding canonical matrix rows. The twenty-fourth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_024_ACTIVE_REPORT_PACKAGE_COHERENCE.md`, which makes active report metadata, focused Slice 024 report paths, proof-boundary markers, and H01 package review inputs agree without adding canonical matrix rows. The twenty-fifth implementation slice is `../scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_025_OPS_INBOX_JOURNEY_TIMELINE.md`, which turns Ops Inbox selected work into a runtime-backed staged Journey Timeline with safe loop-recovery states and no canonical matrix rows. `../verification-reports/VS4_PRODUCT_ALPHA_CLOSURE_CHECKPOINT_2026-07-04.md` records local AI-verifiable closure readiness and recommends pausing new VS4 slices unless a concrete blocker appears. VS3-H01 through VS3-H07 remain conditional deferred gates for production/on-prem/security/live-provider/human acceptance claims and do not block local VS4 documentation or implementation planning.

## Deprecated Product Framing

The following phrases must not remain as current product authority:

- "This document is the only SoT" when referring to older `project-sot.md`.
- "CornerStone fully integrates Palantir-class Ontology and OpenClaw-class Agent UX" as the product identity.
- "Freight & logistics" as the core product identity or required first market.
- "Enterprise data" as the only target scope.
- Any UX/model that exposes `Cornerstone`, `KnowledgeBase`, and `ConnectorHub` as three products the user must understand.

These can remain as historical context or optional implementation inspiration only.

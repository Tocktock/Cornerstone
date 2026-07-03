# CornerStone

**Date:** 2026-06-09
**Owner:** JiYong / Tars
**Status:** Documentation authority reset with full AI-agent handoff, MUST-PASS scenarios, CLI-native gate, local verification plane, design-system contract, VS-0 scaffold gate, no-dependency scaffold CLI bootstrap, local deterministic VS-0 product runtime readiness, local VS0 runtime acceptance/hardening evidence, local VS0 evidence-cleanup/interactive-UI-loop evidence, frozen VS0 EVUX clean sign-off governance scenarios, local VS0 operator acceptance UI gate evidence, and final VS0 implementation closure for VS1 transition
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
9. `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`
10. `docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md`
11. `docs/design/tokens/cornerstone_design_tokens_v0_3.json`
12. `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md`
13. `docs/verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md`
14. `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md`
15. `docs/scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md`
16. `docs/verification-reports/VS0_PRODUCT_RUNTIME_READINESS_SCENARIO_FREEZE_REPORT_2026-06-11.md`
17. `docs/verification-reports/VS0_PRODUCT_RUNTIME_READINESS_REPORT_2026-06-11.md`
18. `docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_CONTRACT.md`
19. `docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_SCENARIO_FREEZE_REPORT_2026-06-11.md`
20. `docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_REPORT_2026-06-11.md`
21. `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_CONTRACT.md`
22. `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_MATRIX.csv`
23. `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_VERIFICATION_MATRIX.csv`
24. `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_CONTRACT.md`
25. `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_MATRIX.csv`
26. `docs/scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_CONTRACT.md`
27. `docs/scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_MATRIX.csv`
28. `docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REVIEW_2026-06-14.md`
29. `docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REPORT_2026-06-14.md`
30. `docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_HUMAN_REVIEW_2026-06-15.md`
31. `docs/verification-reports/VS0_IMPLEMENTATION_FINAL_REPORT_2026-06-15.md`
32. `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md`
33. `docs/agent/PROJECT_OPERATING_CONSTITUTION.md`
34. `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md`
35. `docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md`
36. `docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_MATRIX.csv`
37. `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_STATE_2026-06-19.md`
38. `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_VERIFICATION_REPORT_2026-06-28.md`
39. `docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_CONTRACT.md`
40. `docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_MATRIX.csv`
41. `docs/verification-reports/CONNECTOR_HUB_ENGINEERING_TRAIL_INDEX_2026-06-24.md`
42. `docs/verification-reports/CONNECTOR_HUB_HUMAN_GATES_PREPARATION_REPORT_2026-06-24.md`
43. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
44. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
45. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`
46. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`
47. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md`
48. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md`
49. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_005_UX_POLISH_LEARN.md`
50. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_006_RESPONSIVE_MOBILE_PROOF.md`
51. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_007_KEYBOARD_FOCUS_REVIEW.md`
52. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_008_ASK_RESULT_READABILITY.md`
53. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_009_CLAIM_ACTION_NAV_DETAIL.md`
54. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_010_ASK_INJECTION_BOUNDARY.md`
55. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_011_OPS_INBOX_TRIAGE_DETAIL.md`
56. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_012_ACTION_EXECUTION_BOUNDARY.md`
57. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_013_DESKTOP_OVERFLOW_CONTAINMENT.md`
58. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_014_HUMAN_REVIEW_HANDOFF.md`
59. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_015_GATE_INTEGRITY.md`
60. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_016_EVIDENCE_AUDIT_DETAIL.md`
61. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_017_USER_DROP_ASK_SOURCE.md`
62. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_018_DROP_ASK_TRUST_BOUNDARY.md`

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
| `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md` | Applied design-system contract for UI implementation |
| `docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md` | Source design concept and page/component guidance |
| `docs/design/tokens/cornerstone_design_tokens_v0_3.json` | Canonical design tokens |
| `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md` | Frozen setup-planning contract before VS-0 feature coding |
| `docs/verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md` | Current implementation gate: scaffold next, product features blocked |
| `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` | Frozen first implementation subset |
| `docs/scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md` | Verified task-scoped scenarios for runnable local VS0 product runtime readiness |
| `docs/verification-reports/VS0_PRODUCT_RUNTIME_READINESS_SCENARIO_FREEZE_REPORT_2026-06-11.md` | Historical scenario-freeze report for VS0 product runtime readiness |
| `docs/verification-reports/VS0_PRODUCT_RUNTIME_READINESS_REPORT_2026-06-11.md` | Current implementation report for local VS0 product runtime readiness |
| `docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_CONTRACT.md` | Frozen task-scoped acceptance criteria for turning local VS0 runtime readiness into an operator-reviewable local release candidate; PASS/FAIL status belongs to reports |
| `docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_MATRIX.csv` | Machine-readable matrix for the VS0 runtime acceptance and hardening task contract |
| `docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_SCENARIO_FREEZE_REPORT_2026-06-11.md` | Scenario-freeze report for the next VS0 runtime acceptance and hardening gate |
| `docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_REPORT_2026-06-11.md` | Current local deterministic implementation report for VS0 runtime acceptance and hardening |
| `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_CONTRACT.md` | Frozen next task-scoped scenarios for evidence cleanup and real interactive UI workflow proof |
| `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_MATRIX.csv` | Frozen machine-readable matrix for the VS0 evidence cleanup and interactive UI loop contract |
| `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_VERIFICATION_MATRIX.csv` | Machine-readable current verification matrix for the VS0 evidence cleanup and interactive UI loop report |
| `docs/verification-reports/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_REPORT_2026-06-13.md` | Current local deterministic implementation report for VS0 evidence cleanup and interactive UI loop |
| `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_CONTRACT.md` | Frozen task-scoped criteria for VS0 EVUX clean sign-off governance |
| `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_MATRIX.csv` | Machine-readable verification matrix for the VS0 EVUX clean sign-off governance contract |
| `docs/verification-reports/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_REPORT_2026-06-14.md` | Final local governance sign-off report for VS0 EVUX evidence cleanup |
| `docs/scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_CONTRACT.md` | Frozen task-scoped criteria for human-understandable VS0 operator UI acceptance before full VS-1 |
| `docs/scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_MATRIX.csv` | Frozen machine-readable matrix for the VS0 operator acceptance UI gate |
| `docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REVIEW_2026-06-14.md` | Current human review record: operator UX acceptance accepted; full VS-1 unblocked |
| `docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REPORT_2026-06-14.md` | Current local deterministic implementation evidence for the VS0 operator UI gate |
| `docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_HUMAN_REVIEW_2026-06-15.md` | Human acceptance evidence for `VS0-UI-H01` |
| `docs/verification-reports/VS0_IMPLEMENTATION_FINAL_REPORT_2026-06-15.md` | Final VS0 implementation closure report for VS1 transition |
| `docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md` | Frozen VS2 policy, tenant isolation, and default egress-deny scenario contract |
| `docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_MATRIX.csv` | Machine-readable VS2 scenario matrix: 70 MUST_PASS, 16 REGRESSION, 7 HUMAN_REQUIRED rows |
| `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_STATE_2026-06-19.md` | Baseline inventory, impact map, implementation notes, and local deterministic VS2 proof links |
| `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_IMPLEMENTATION_REPORT_2026-06-19.md` | Superseded local deterministic VS2 implementation report |
| `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_SCENARIO_SPECIFIC_REMEDIATION_REPORT_2026-06-19.md` | Superseded scenario-specific VS2 remediation report |
| `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_VERIFICATION_REPORT_2026-06-28.md` | Current generated VS2 verification status and supersession boundary |
| `docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_CONTRACT.md` | Frozen ConnectorHub adoption contract; implementation status belongs to ConnectorHub reports and manifests |
| `docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_MATRIX.csv` | Machine-readable ConnectorHub scenario matrix; current row status is reported by ConnectorHub verification artifacts |
| `docs/verification-reports/CONNECTOR_HUB_ENGINEERING_TRAIL_INDEX_2026-06-24.md` | ConnectorHub engineering trail index, source reconciliation, scenario-result links, verifier commands, and remaining proof surfaces |
| `docs/verification-reports/CONNECTOR_HUB_HUMAN_GATES_PREPARATION_REPORT_2026-06-24.md` | Human-gate preparation report for H01-H07 packages, templates, structural validation, execution order, and no-PASS boundary |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md` | Frozen VS4 Product Alpha UI Daily Loop contract |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv` | Machine-readable VS4 matrix: 20 MUST_PASS, 7 REGRESSION_GUARD, 1 HUMAN_REQUIRED row |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md` | Frozen VS4 Slice 001 Product Alpha Home/Ops Inbox shell implementation contract |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md` | Frozen VS4 Slice 002 Evidence-backed Brief detail implementation contract |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md` | Frozen VS4 Slice 003 Ask, general-purpose packs, page states, reference alignment, and regression implementation contract |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md` | Frozen VS4 Slice 004 human acceptance package and validation contract; keeps `VS4-H01` HUMAN_REQUIRED |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_005_UX_POLISH_LEARN.md` | Frozen VS4 Slice 005 UX polish and Learn review contract; no new canonical matrix rows |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_006_RESPONSIVE_MOBILE_PROOF.md` | Frozen VS4 Slice 006 responsive mobile proof contract; no new canonical matrix rows |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_007_KEYBOARD_FOCUS_REVIEW.md` | Frozen VS4 Slice 007 keyboard/focus review contract; no new canonical matrix rows |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_008_ASK_RESULT_READABILITY.md` | Frozen VS4 Slice 008 Ask result readability contract; no new canonical matrix rows |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_009_CLAIM_ACTION_NAV_DETAIL.md` | Frozen VS4 Slice 009 Claim and Action nav-detail contract; no new canonical matrix rows |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_010_ASK_INJECTION_BOUNDARY.md` | Frozen VS4 Slice 010 Ask injection-boundary contract; no new canonical matrix rows |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_011_OPS_INBOX_TRIAGE_DETAIL.md` | Frozen VS4 Slice 011 Ops Inbox triage/detail contract; no new canonical matrix rows |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_012_ACTION_EXECUTION_BOUNDARY.md` | Frozen VS4 Slice 012 Action execution-boundary contract; no new canonical matrix rows |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_013_DESKTOP_OVERFLOW_CONTAINMENT.md` | Frozen VS4 Slice 013 desktop overflow-containment contract; no new canonical matrix rows |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_014_HUMAN_REVIEW_HANDOFF.md` | Frozen VS4 Slice 014 human-review handoff contract; keeps `VS4-H01` HUMAN_REQUIRED |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_015_GATE_INTEGRITY.md` | Frozen VS4 Slice 015 scenario-gate integrity contract; no new canonical matrix rows |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_016_EVIDENCE_AUDIT_DETAIL.md` | Frozen VS4 Slice 016 Evidence/Audit detail contract; no new canonical matrix rows |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_017_USER_DROP_ASK_SOURCE.md` | Frozen VS4 Slice 017 user Drop/Ask source contract; no new canonical matrix rows |
| `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_018_DROP_ASK_TRUST_BOUNDARY.md` | Frozen VS4 Slice 018 Drop/Ask trust-boundary contract; no new canonical matrix rows |
| `docs/verification-reports/template.md` | Required report shape for scenario and CLI verification evidence |
| `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` | Compatible technical defaults only; not product authority |
| `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md` | Verification-centered agent workflow |
| `docs/agent/PROJECT_OPERATING_CONSTITUTION.md` | Project operating rules for agents |

## Scenario Counts

- Full scenario standard: 206 scenarios.
- VS-0 implementation subset: 58 scenarios.
- VS0 runtime readiness overlay: 14 task-scoped scenarios, already accepted for local deterministic runtime readiness.
- VS0 runtime acceptance/hardening overlay: 9 task-scoped scenarios; reports show 7 AI-verifiable PASS, 2 HUMAN_REQUIRED, production release still false.
- VS0 evidence cleanup and interactive UI loop overlay: 14 task-scoped scenarios; reports show 12 AI-verifiable PASS, 2 HUMAN_REQUIRED, production release still false.
- VS0 EVUX clean sign-off governance overlay: 16 task-scoped scenarios; verifier path `cornerstone scenario verify vs0-evux-governance`, 14 AI-verifiable rows, 2 HUMAN_REQUIRED, production release still false.
- VS0 operator acceptance UI gate overlay: 13 task-scoped scenarios; local verifier path `cornerstone scenario verify vs0-operator-acceptance-ui`, 12 AI-verifiable PASS, 1 human-only row accepted by JiYong/Tars, full VS-1 main implementation unblocked.
- VS2 policy/tenancy/egress overlay: 93 task-scoped scenarios; verifier path `cornerstone scenario verify vs2-policy-tenancy-egress`; current generated status is recorded in `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_VERIFICATION_REPORT_2026-06-28.md`.
- ConnectorHub adoption overlay: 47 task-scoped scenarios; verifier path `cornerstone scenario verify connector-contract-adapter`; current generated status is recorded in `reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json` and `docs/verification-reports/CONNECTOR_HUB_ENGINEERING_TRAIL_INDEX_2026-06-24.md`.
- VS4 Product Alpha UI Daily Loop overlay: 28 task-scoped scenarios; verifier path `cornerstone scenario verify vs4-product-alpha-ui-daily-loop`. Slice 001 selects 7 AI-verifiable rows for the local Product Alpha Home/Ops Inbox shell. Slice 002 adds 14 AI-verifiable rows for the Evidence-backed Brief detail, claim/memory/action review boundaries, prompt-injection guard, reference-boundary guard, and CLI parity. Slice 003 adds the remaining 6 AI-verifiable rows for Ask, packs, page states, reference alignment, and VS0/VS1 regression. Slice 004 prepares the `VS4-H01` human acceptance package and validator without marking human UX accepted. Slice 005 strengthens product-language, Learn-review, and progressive proof-detail readiness without adding canonical matrix rows. Slice 006 adds deterministic narrow-viewport browser proof and responsive containment markers without adding canonical matrix rows. Slice 007 adds deterministic keyboard/focus review proof without adding canonical matrix rows. Slice 008 makes the Ask result readable before raw refs while preserving evidence refs in progressive detail without adding canonical matrix rows. Slice 009 turns the normal `Claims` and `Actions` nav destinations into product-ready review pages without adding canonical matrix rows. Slice 010 proves unsafe Ask text cannot approve claims, approve memory, execute actions, call providers, change policy, or expand authority without adding canonical matrix rows. Slice 011 adds Ops Inbox triage lanes and selected-item detail for returning daily work without adding canonical matrix rows. Slice 012 freezes the Action execution-boundary plan for denied unauthorized approval/execution, safety-envelope evidence, audit refs, and zero provider/writeback side effects without adding canonical matrix rows. Slice 013 contains desktop Product Alpha overflow by wrapping long policy/safety/evidence tokens without adding canonical matrix rows. Slice 014 makes the `VS4-H01` human-review handoff visible in Product Alpha Home/Ops Inbox while keeping acceptance HUMAN_REQUIRED. Slice 015 adds VS4-specific scenario-gate integrity checks for overclaims, human-required status, reference-image boundaries, CLI parity, negative evidence, and source-tree metadata without adding canonical matrix rows. Slice 016 makes Evidence/Audit a product-ready detail surface with source, provenance, safety, activity, audit verification, and Learn linkage without adding canonical matrix rows. Slice 017 makes the primary Drop/Ask path preserve user-pasted source text and create the Evidence-backed Brief from that source instead of a fixed fixture, with CLI `artifact ingest --text` parity and no canonical matrix rows. Slice 018 hardens Drop/Ask trust boundaries so user-pasted and conversation text stay untrusted through HTTP/API intake and same-checksum CLI dedupe, with structured unsafe-promotion denial and no canonical matrix rows. Full VS4 remains incomplete until `VS4-H01` has human acceptance evidence.
- Release rule: no PASS without concrete scenario evidence.
- CLI-native-first rule: no feature scenario can be marked PASS unless its native `cornerstone ...` CLI path is verified or the item is explicitly classified as a non-feature implementation internal.

Verify the documentation wiring with:

```sh
scripts/verify_sot_docs.sh
scripts/verify_cli_native_first_docs.sh
scripts/verify_local_verification_plane_docs.sh
scripts/verify_design_system_docs.sh
scripts/verify_vs0_scaffold_readiness_docs.sh
```

Verify the current scaffold bootstrap with:

```sh
export PATH="$PWD:$PATH"
cornerstone --help
cornerstone version --json
cornerstone health --json
cornerstone ready --json        # local_scenario_ready=true, vs0_runtime_ready=true, production_release_ready=false
cornerstone runtime serve --port 8787
cornerstone scenario list --set full --json
cornerstone scenario coverage --json
python3 scripts/verify_scenario_matrix.py
cornerstone scenario verify vs0-scaffold --json
cornerstone scenario verify vs0-fixtures --corpus fixtures/vs0 --model-provider local_test --json
cornerstone scenario verify vs0-artifacts --json
cornerstone scenario verify vs0-security --json
cornerstone scenario verify vs0-search-evidence --json
cornerstone scenario verify vs0-search-understanding --json
cornerstone scenario verify vs0-namespace-isolation --json
cornerstone scenario verify vs0-audit-ledger --json
cornerstone scenario verify vs0-universal-core --json
cornerstone scenario verify vs0-claim-evidence --json
cornerstone scenario verify vs0-security-policy --json
cornerstone scenario verify vs0-regression-guardrails --json
cornerstone scenario verify vs0-briefing --json
cornerstone scenario verify vs0-mission-action --json
cornerstone scenario verify vs0-detail-surfaces --json
cornerstone scenario verify vs0-conversation-onboarding --json
cornerstone scenario verify vs0-product-loop-identity --json
cornerstone scenario verify vs0-memory-truth-boundary --json
cornerstone scenario verify vs0-product-domain-readiness --json
cornerstone scenario verify vs0-tenant-security-boundary --json
cornerstone scenario verify vs0-product-runtime --json
cornerstone scenario verify vs0-runtime-acceptance --json
cornerstone scenario verify full-claim-collaboration --json
cornerstone scenario verify full-memory-wiki --json
cornerstone scenario verify full-learning-experience --json
cornerstone scenario verify full-understanding-ontology --json
cornerstone scenario verify full-extension-ecosystem --json
cornerstone scenario verify full-agent-orchestration --json
cornerstone scenario verify full-brain-routing --json
cornerstone scenario verify full-security-operations --json
cornerstone scenario verify full-namespace-governance --json
cornerstone scenario verify full-mission-control-autonomy-lifecycle --json
make verify-local-fast
make verify-vs0-runtime
make verify-vs0-acceptance
make verify-vs0-evux
make verify-vs4-product-alpha-shell
make verify-vs4-product-alpha-slice-003
make verify-vs4-product-alpha-human-package
make verify-vs4-product-alpha-responsive-mobile
make verify-vs4-product-alpha-keyboard-focus
make verify-vs4-product-alpha-ask-readability
make verify-vs4-product-alpha-decision-pages
make verify-vs4-product-alpha-ask-injection-boundary
make verify-vs4-product-alpha-ops-inbox-triage
make verify-vs4-product-alpha-action-execution-boundary
make verify-vs4-product-alpha-desktop-overflow
make verify-vs4-product-alpha-human-review-handoff
make verify-vs4-product-alpha-gate-integrity
make verify-vs4-product-alpha-evidence-audit-detail
make verify-vs4-product-alpha-user-drop-ask-source
make verify-vs4-product-alpha-drop-ask-trust-boundary
```

The current scaffold CLI can verify scaffold readiness, scenario registry coverage, deterministic local fixture-validator readiness, the first CLI-native artifact preservation slice, the first redaction/prompt-injection safety slice, the first search/evidence-bundle/draft-claim/evidence-viewer slice, the first deterministic search-understanding slice, the first owner/namespace isolation slice, the first tamper-evident audit-ledger slice, the first universal non-logistics core slice, the first claim evidence-gating slice, the first default-deny egress/sandbox policy slice, the first regression guardrail summary slice, the first evidence-backed briefing slice, the first Mission Goal Contract / Action Card / dry-run / approval / mocked connector-action safety slice, the first detail-surface slice, the first conversation-onboarding slice, the first product-loop identity slice with durable memory and learning records, the first memory truth-boundary slice, the first product/domain/Autopilot-readiness slice, the first deterministic tenant/security boundary slice for namespace promotion, access policy evaluation, and personal-to-organization memory leakage prevention, the first local VS-0 product runtime loop with CLI/API/minimal UI parity, the first full-suite claim-collaboration slice for Knowledge Capsules, Decision Cards, corrections, and trust-state-aware shared views, the first full-suite memory/wiki slice for permanent wiki views, memory sovereignty controls, correction, rollback/forget, freshness warnings, poisoning quarantine, explainable memory use, product-learning isolation, namespace-local adaptation, and memory export, the first full-suite learning/experience slice for Mission Trajectory Ledger, Experience Library, recommendations, lesson promotion/control, local adaptation, connected outcomes, metrics, and export, the first full-suite understanding/ontology slice for draft structure suggestions, promoted draft ontology items, operational maps, contradictions, stale-context warnings, versioned ontology changes, and unknown-domain draft handling, the first full-suite extension ecosystem slice for local Agent Pack registry/import, install-vs-activation, explicit grants, certification, ConnectorHub mediation, version pin/update/rollback, untrusted/direct-provider denial, and emergency patch policy, the first full-suite agent-orchestration slice for Orchestrator-led mission traces, visible specialist role contracts/cards, evidence-labeled outputs, direct mutation denial, brain replacement, versioned contract changes, prompt authority denial, failure diagnosis, Agent Pack grant enforcement, and replay without hidden chain-of-thought, the first full-suite brain-routing slice for replaceable model brains, policy-aware routing, provider override denial, namespace-local Brain Performance Ledger learning, ensemble gating, LLM-as-judge support limits, objective/owner outcome precedence, disagreement escalation, calibration tracking, and provider-switch evidence continuity, the first full-suite security-operations slice for ConnectorHub credential custody, sensitive-change stop-and-ask gates, explicit human-required reporting, backup/restore rehearsal, helpful failures, action idempotency, retention transparency, operator status, and release-report evidence checks, the first full-suite namespace-governance slice for owner-scoped archive namespaces, classification-aware access, promotion modes, product-learning boundaries, cross-tenant isolation, namespace audit export, retention dry-run, and recovery, and the first full-suite mission-control/autonomy-lifecycle slice for Mission Control, product loop visibility, source-system boundaries, personal-to-organization path, ConnectorHub action trace, autonomy revoke/escalation/outcome/AAR/audit/metrics/reversibility, and repo-split UX guardrails. These commands verify the frozen AI-verifiable scenario rows through deterministic local CLI evidence and local API/UI evidence where claimed; they do not claim live provider integration, production tenant isolation, production release readiness, or human-only operational proof.

## First Implementation Target

Before feature coding, the VS-0 setup-planning gate is `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md`. It defines the version baseline, monorepo direction, CLI scaffold expectations, verification report shape, and human approval gates for production dependency additions.

Before scaffold implementation, read `docs/verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md`. That report remains historical scaffold-gate context. Current local deterministic runtime evidence is recorded in `docs/verification-reports/VS0_PRODUCT_RUNTIME_READINESS_REPORT_2026-06-11.md`; it does not claim production release readiness.

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

The current local acceptance criteria are in `docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_CONTRACT.md`. Local evidence is recorded in `docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_REPORT_2026-06-11.md`, `reports/scenario/vs0-runtime-acceptance-2026-06-11.json`, `reports/browser/vs0-runtime-acceptance-2026-06-11/`, and `reports/release/vs0-runtime-acceptance-2026-06-11/`. The contract is status-neutral; PASS/FAIL status belongs to those reports. It must not be treated as production release, live-provider, or human UX acceptance.

The current local evidence-cleanup and interactive UI loop criteria are in `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_CONTRACT.md`. Local evidence is recorded in `docs/verification-reports/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_REPORT_2026-06-13.md`, `reports/scenario/vs0-evux-2026-06-13.json`, `reports/browser/vs0-evux-2026-06-13/`, `reports/quickstart/vs0-evux-quickstart.json`, and `reports/release/vs0-evux-2026-06-13/`. It proves local/mock EVUX only; production release, live-provider readiness, and human UX acceptance remain unclaimed.

The clean sign-off governance criteria are in `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_CONTRACT.md`. They add `VS0-GOV-*` rows for matrix/report consistency, dirty-worktree metadata semantics, command transcript evidence, release manifest hashing, post-commit rollup behavior, and overclaim/dependency regression guards. Current PASS/HUMAN_REQUIRED status is recorded by `cornerstone scenario verify vs0-evux-governance`, `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_MATRIX.csv`, and `docs/verification-reports/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_REPORT_2026-06-14.md`; production release, live-provider readiness, and human usability acceptance remain unclaimed.

The VS0 operator acceptance UI gate is in `docs/scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_CONTRACT.md`. It freezes the narrow UI acceptance slice required before full VS-1: the local VS0 UI must expose Artifact, Search, Evidence, Claim, Action Card, dry-run, approval, mock execution, and Audit as understandable operator steps. Current local implementation evidence is recorded in `docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REPORT_2026-06-14.md`, `reports/scenario/vs0-operator-acceptance-ui-2026-06-14.json`, and `reports/browser/vs0-operator-acceptance-ui-2026-06-14/`: 12 AI-verifiable rows pass. Human operator UX acceptance is recorded in `docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_HUMAN_REVIEW_2026-06-15.md`, final closure is recorded in `docs/verification-reports/VS0_IMPLEMENTATION_FINAL_REPORT_2026-06-15.md`, and full VS-1 main implementation is unblocked. Production release and live-provider readiness remain unclaimed.

The VS2 policy, tenant isolation, and default egress-deny contract is in `docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md` with a machine-readable matrix at `docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_MATRIX.csv`. Baseline inventory and implementation notes are in `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_STATE_2026-06-19.md`; the earlier remediation report is superseded for current generated status by `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_VERIFICATION_REPORT_2026-06-28.md`. Current evidence is in `reports/security/vs2-local-security-proof.json`, `reports/security/vs2-scenario-specific-evidence.json`, `reports/security/vs2-synthetic-world.json`, and `reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json`. The native verifier is `cornerstone scenario verify vs2-policy-tenancy-egress --json`; production security, real IdP, production network, live provider, human UX, and production-like migration/restore claims remain separate human/external gates.

## VS0 Runtime Acceptance Quickstart

This quickstart is local-only. It uses fixture data, mocked ConnectorHub-style action execution, and zero real external HTTP calls.

```sh
export PATH="$PWD:$PATH"

# Terminal 1: local API/UI runtime
cornerstone runtime serve --port 8787

# Terminal 2: repeat the local VS0 loop with native CLI paths
cornerstone ready --json
cornerstone artifact ingest fixtures/vs0/packs/01_artifact_basic/input.txt --state-dir data/quickstart-vs0 --json
cornerstone search query alpha-evidence-anchor --state-dir data/quickstart-vs0 --json
cornerstone evidence bundle create --search-snapshot-id <search_snapshot_id> --state-dir data/quickstart-vs0 --json
cornerstone claim create --evidence-bundle-id <evidence_bundle_id> --statement "The Alpha evidence anchor is ready for local VS0 acceptance." --state-dir data/quickstart-vs0 --json
cornerstone claim approve <claim_id> --state-dir data/quickstart-vs0 --json
cornerstone mission create --goal "Complete local VS0 acceptance through governed mock action" --claim-id <claim_id> --state-dir data/quickstart-vs0 --json
cornerstone mission activate <mission_id> --mode autopilot --state-dir data/quickstart-vs0 --json
cornerstone action propose --mission-id <mission_id> --claim-id <claim_id> --goal "Record local acceptance status" --action-kind external_writeback --risk high --connector mock_connector --target mock://vs0-runtime/acceptance --state-dir data/quickstart-vs0 --json
cornerstone action dry-run <action_id> --state-dir data/quickstart-vs0 --json
cornerstone action approve <action_id> --state-dir data/quickstart-vs0 --json
cornerstone action execute <action_id> --state-dir data/quickstart-vs0 --json
cornerstone audit verify --state-dir data/quickstart-vs0 --json

# Release-facing local acceptance gate
cornerstone scenario verify vs0-runtime-acceptance --output reports/scenario/vs0-runtime-acceptance-2026-06-11.json --json
cornerstone release evidence collect --scope vs0-runtime-acceptance --json

# EVUX verification gate
cornerstone scenario verify vs0-evux --json --output reports/scenario/vs0-evux-2026-06-13.json
cornerstone scenario gate reports/scenario/vs0-evux-2026-06-13.json --json
cornerstone quickstart verify vs0-evux --json --output reports/quickstart/vs0-evux-quickstart.json
cornerstone release evidence collect --scope vs0-evux --json
```

The quickstart commands above are operator guidance. Placeholder IDs are for manual walkthrough only. Scenario `PASS` requires a generated quickstart transcript with generated IDs, exit codes, evidence refs, and audit refs; README token presence alone is not proof.

The quickstart does not verify production tenant isolation, live provider execution, or human usability acceptance. Those remain explicit `HUMAN_REQUIRED` rows.

EVUX evidence outputs:

```text
reports/scenario/vs0-evux-2026-06-13.json
reports/browser/vs0-evux-2026-06-13/
reports/quickstart/vs0-evux-quickstart.json
reports/release/vs0-evux-2026-06-13/manifest.json
docs/verification-reports/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_REPORT_2026-06-13.md
```

## CLI Native First

Every product feature must be CLI-native first. UI and API remain important, but a user-visible, operator-visible, admin-visible, API-visible, workflow-visible, connector-visible, verification-visible, or automation-visible capability is not complete until it has a verified native `cornerstone ...` command path.

The release invariant is: **No CLI, no feature PASS.** The CLI must use the same Product / Archive / Connector / Workflow / Policy / Evidence / Audit boundaries as the UI/API and must provide scriptable `--json` output, stable exit codes, workspace/namespace scope, dry-run for mutations, evidence refs, and audit refs.

Connector Hub adoption starts with the scenario-first ConnectorPort contract slice in `docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_CONTRACT.md`. The CH-0 delivery units plus the CH-1 `CS-CH-007` Projection Delivery archive unit, `CS-CH-008` durable-ack unit, `CS-CH-009` retry/quarantine unit, `CS-CH-010` dedupe/version-lineage unit, `CS-CH-011` Source Policy field/body enforcement unit, `CS-CH-012` Evidence Bundle promotion unit, `CS-CH-013` temporary raw-access unit, `CS-CH-014` untrusted connector-content guard, `CS-CH-034` owner/namespace scope isolation unit, `CS-CH-035` credential custody guard, `CS-CH-036` current VS2 local default-deny egress topology guard, `CS-CH-037` audit-correlation guard, CH-2 `CS-CH-015` selected GitHub repository boundary, `CS-CH-016` source-control Projection family ingestion, `CS-CH-017` incremental sync idempotency, `CS-CH-018` GitHub content restriction and secret hygiene, `CS-CH-019` GitHub zero-write guard, `CS-CH-020` GitHub provider failure-state handling, CH-3 `CS-CH-021` macOS capture consent/permission gating, `CS-CH-022` bounded activity-session projection, `CS-CH-023` owner-scoped Watch Rule lifecycle, `CS-CH-024` explicit Chrome active-tab capture, `CS-CH-025` allowlist-based Chrome auto capture, `CS-CH-026` sensitive Chrome page block/degrade policy, `CS-CH-027` capture lifecycle controls, `CS-CH-028` Watch Result truth separation, and CH-4 `CS-CH-029` through `CS-CH-033` governed Action units are verified with `cornerstone scenario verify connector-contract-adapter --json` and can be replayed independently with `--scenario CS-CH-001`, `--scenario CS-CH-002`, `--scenario CS-CH-003`, `--scenario CS-CH-004`, `--scenario CS-CH-005`, `--scenario CS-CH-006`, `--scenario CS-CH-007`, `--scenario CS-CH-008`, `--scenario CS-CH-009`, `--scenario CS-CH-010`, `--scenario CS-CH-011`, `--scenario CS-CH-012`, `--scenario CS-CH-013`, `--scenario CS-CH-014`, `--scenario CS-CH-015`, `--scenario CS-CH-016`, `--scenario CS-CH-017`, `--scenario CS-CH-018`, `--scenario CS-CH-019`, `--scenario CS-CH-020`, `--scenario CS-CH-021`, `--scenario CS-CH-022`, `--scenario CS-CH-023`, `--scenario CS-CH-024`, `--scenario CS-CH-025`, `--scenario CS-CH-026`, `--scenario CS-CH-027`, `--scenario CS-CH-028`, `--scenario CS-CH-029`, `--scenario CS-CH-030`, `--scenario CS-CH-031`, `--scenario CS-CH-032`, `--scenario CS-CH-033`, `--scenario CS-CH-034`, `--scenario CS-CH-035`, `--scenario CS-CH-036`, `--scenario CS-CH-037`, `--scenario CS-CH-038`, `--scenario CS-CH-039`, or `--scenario CS-CH-040`. This remains local fixture and current local VS2 topology evidence only; live-provider, physical-device, rendered Watch Result UI, human UX/privacy, destructive-action UX approval, production secret-backend custody, production RLS/topology, production network-control review, and production security claims stay separate.

The Connector Hub human gates (`CS-CH-H01`, `CS-CH-H02`, `CS-CH-H03`, `CS-CH-H04`, `CS-CH-H05`, `CS-CH-H06`, and `CS-CH-H07`) are prepared by `cornerstone connector human-gate package --scenario <id> --json` with matching template readiness, review order, dependencies, stop/reject conditions, `Senior Review Perspectives`, `scenario_delivery_unit_plan`, and blank proposed reviewer-record templates that include a typed evidence-packet manifest skeleton with required-evidence labels and allowed redaction statuses, `reviewer_checklist`, `record_template_output_command`, `validation_command`, `validation_output_command`, plus a top-level `summary` and `final_verdict=HUMAN_REQUIRED` mirror for operator scripts; the plan maps senior-perspective research, implementation approach definition, smallest complete rehearsal, remediation/refactor, structural validation, result documentation, and dependency-aware next-gate movement without collecting approval or enabling PASS/product claims; the blank reviewer template can be written directly with `cornerstone connector human-gate package --scenario <id> --json --record-template-output <reviewer-record-template.json>` and remains preparation data until a human completes it; the checklist maps required fields, senior-review perspective findings, evidence-packet rows, dependency refs, and validation commands without collecting approval; checked structurally by `cornerstone connector human-gate validate-record --scenario <id> --record-file <json> --json` or `cornerstone connector human-gate validate-record --scenario <id> --record-file <json> --json --output <redacted-validation-envelope.json>` for required fields, redacted senior-review perspective findings, evidence-packet manifest coverage, required-evidence label matching, allowed redaction statuses, dependency refs to existing structurally valid ACCEPT `connector_human_gate_record_validation:<id>` artifacts, ISO-8601 timezone-aware `review_timestamp`, decision values, and sensitive markers without persisting the submitted record body, raw record path, decision value, senior-review finding text, or manifest values, while the optional redacted validation envelope mirrors top-level `summary` and `final_verdict=HUMAN_REQUIRED` without promoting any H row; summarized by `cornerstone connector human-gate report --json --output reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json` with execution queue, template-structure readiness counters, senior-review-perspective readiness, senior-review finding completion counts, evidence-packet manifest completion counts, validation coverage summary fields, row-level `scenario_delivery_unit_plan`, row-level `record_template_output_command`, row-level `record_validation_output_command`, depends-on human-gate record validation readiness fields, dependency unlock counts, and a top-level `summary` plus `final_verdict=HUMAN_REQUIRED` mirror for operator scripts; sequenced by `cornerstone connector human-gate next --json` so operators can see the first uncompleted dependency-ready H gate plus blocked dependency reasons, `next_record_template_output_command`, `next_record_validation_output_command`, `next_reviewer_checklist`, `next_scenario_delivery_unit_plan`, `next_remaining_human_evidence_summary`, and, when H04 is next, H04-only local-baseline comparison inputs, required human delta, and recommended preflight command counts plus structured preflight command-plan rows with expected report paths; compacted by `cornerstone connector human-gate validation-handoff --json --output reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json` into an ordered validation handoff manifest with H-row validation state, dependency blockers, reviewer commands, latest validation refs and redacted latest-validation issue summaries when present, H04 local-baseline summary, H04 preflight command-plan rows, and zero approval/PASS/product-claim counters; stored as durable readiness artifact `reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json`; stored as durable next-selector artifact `reports/scenario/connectorhub-human-gate-next-2026-06-24.json`; stored as durable validation-handoff artifact `reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json`; stored as durable per-gate package artifacts `reports/scenario/connectorhub-human-gate-package-cs-ch-h01-2026-06-24.json` through `reports/scenario/connectorhub-human-gate-package-cs-ch-h07-2026-06-24.json`; and paired with the matching `docs/verification-reports/CONNECTOR_HUB_CS_CH_H0*_HUMAN_REVIEW_TEMPLATE_2026-06-24.md` template. For `CS-CH-H04`, package, readiness rows, the first next-gate selector, and the validation-handoff row expose local baseline review inputs or summary counts with current local VS2 and ConnectorHub dependency report refs, hashes, status summaries, required human delta, recommended local preflight commands, and `recommended_preflight_command_plan` rows; package/readiness expose first-class `local_baseline_preflight_bundle` mirrors, next exposes first-class `next_local_baseline_preflight_bundle`, and validation handoff exposes first-class `local_baseline_preflight_bundle`; this is review input only with `acceptance_sufficient=false`, `product_claim_allowed=false`, and `pass_claim_allowed=false`, and each individual baseline report row repeats `review_input_only=true`, `acceptance_sufficient=false`, `product_claim_allowed=false`, `pass_claim_allowed=false`, and `claim_boundary=h04_local_baseline_snapshot_is_review_input_not_human_acceptance`, while each preflight command-plan row repeats `review_input_only=true`, `acceptance_sufficient=false`, `product_claim_allowed=false`, `pass_claim_allowed=false`, and `claim_boundary=h04_local_baseline_preflight_is_review_input_not_human_acceptance`. Structurally valid `REJECT` records remain validation evidence but do not unlock dependent H gates. Those packages, validations, next-gate selections, validation handoffs, and reports are non-mutating preparation evidence only; live accounts, physical devices, browser privacy review, production-like VS2 proof, live non-GitHub Action execution, UX/trust study, and recovery exercise evidence still require dated human records before any H row can become `PASS`. The engineering trail is checked by `make verify-connectorhub-engineering-trail`, which verifies the 40 AI PASS result docs, 7 human-required templates, package/readiness/next/validation-handoff/validate-record source surfaces, the pinned human package JSONs, the pinned human readiness JSON, the pinned human next-selector JSON, the pinned human validation-handoff JSON, aggregate ConnectorHub report, and stale-metadata guards.

Human-gate packages and rollups also expose `scenario_delivery_unit_plan_summary`, while the next selector exposes `next_scenario_delivery_unit_plan_summary`; their top-level `summary` fields include `scenario_delivery_unit_plan_ready`, `scenario_delivery_unit_plan_ready_count=7`, `scenario_delivery_unit_plan_missing_count=0`, `next_scenario_delivery_unit_plan_ready`, lifecycle-step counts, senior-review perspective counts, and false product/PASS/approval/dependency-unlock plan flags for scriptable scenario delivery-unit checks.

H04 `local_baseline_preflight_bundle` and `next_local_baseline_preflight_bundle` aliases also repeat `recommended_preflight_command_plan_schema_version`, `recommended_preflight_command_plan_count`, and `recommended_preflight_command_plan` so operator scripts can consume the same plan field names from either the local-baseline review-input object or the first-class bundle. The H04 preflight-bundle top-level summary also mirrors `preflight_bundle_report_id`, `local_baseline_review_inputs_schema_version`, `local_baseline_acceptance_sufficient`, `local_baseline_product_claim_allowed`, `local_baseline_pass_claim_allowed`, `required_human_delta`, `recommended_preflight_commands`, `recommended_preflight_command_count`, `recommended_preflight_command_plan_schema_version`, `recommended_preflight_command_plan_count`, and `recommended_preflight_command_plan`; this remains comparison input only and does not execute preflight commands or change any `HUMAN_REQUIRED` gate.

Each evidence-packet validation summary mirrors filename-only packet validation aliases and schema-version refs without exposing packet contents, collecting approval, or changing any `HUMAN_REQUIRED` gate.

Each evidence-packet record-draft summary mirrors packet validation lineage and draft safety flags while keeping `draft_record_included_in_summary=false`, raw packet content persistence false, human-decision recording false, and every human gate `HUMAN_REQUIRED`.

Each evidence-packet scaffold summary mirrors template filenames, template hashes, and write metadata while keeping `template_contents_included_in_summary=false` and every human gate `HUMAN_REQUIRED`.

Human-gate reviewer-record templates, validation envelopes, readiness rows, and the next selector now expose `redaction_guidance` or `next_record_redaction_guidance` so operators can see the field-level safety contract before filling a record: raw secrets, raw provider payloads, raw evidence values, raw record bodies, and raw record paths are disallowed; senior-review findings and evidence-packet manifest values are not persisted by the validator; dependency refs must point at structurally valid ACCEPT `connector_human_gate_record_validation:<id>` artifacts. This guidance is operator preparation only and does not promote any H row.

The human-gate next selector also exposes active-gate validation correction metadata when a reviewer record has already been structurally checked: `next_latest_record_validation_ref`, `next_record_validation_status`, `next_latest_record_validation_dependency_unlock_allowed`, and redacted `next_latest_record_validation_issue_summary`. These fields help the operator fix or consciously reject the current gate without exposing raw record bodies, raw paths, human decision values, senior-review finding text, or evidence-packet manifest values, and the issue summary explicitly keeps `structural_validation_is_human_acceptance=false`, `human_acceptance_requires_owner_promotion=true`, and the full-goal completion boundary.

The scenario delivery-unit manifest `reports/scenario/connectorhub-scenario-delivery-unit-manifest-2026-06-24.json` records 40 closed AI delivery units with exact closed AI and excluded human scenario ID sets, exact focused verify/gate commands, focused report and gate facts, machine-readable decision/research/lifecycle coverage facts, top-level `scenario_delivery_loop_trace`, top-level `focused_report_evidence_trace`, top-level `focused_report_command_exit_code_trace`, top-level `focused_report_readiness_dimension_trace`, top-level `research_perspective_trace`, top-level `research_perspective_decision_bridge_trace`, top-level `implementation_solution_trace`, top-level `refactor_hardening_trace`, top-level `documented_result_trace`, top-level `adoption_contribution_trace`, result-doc closure flags, the local fixture/current local VS2 topology claim boundary, and false product/live/human/production claim flags. The delivery-loop trace preserves `scenario_delivery_loop_trace_count=40`, `scenario_delivery_loop_trace_step_entry_total=280`, `scenario_delivery_loop_trace_move_next_ready_count=39`, and `scenario_delivery_loop_trace_full_goal_completion_allowed_count=0`; this is local AI delivery-process evidence for research, approach, smallest solution, refactor, verification, documentation, and adoption, not human/release completion proof. The focused evidence trace preserves `focused_report_evidence_trace_count=40`, `focused_report_evidence_trace_ready_count=40`, and `focused_report_evidence_trace_full_goal_completion_allowed_count=0`; this is local focused-report evidence, command, negative-evidence, and gate-status trace data, not live-provider, human-acceptance, production, release, or full-goal completion proof. The focused command-exit trace preserves `focused_report_command_exit_code_trace_count=40`, `focused_report_command_exit_code_trace_entry_total=280`, and `focused_report_command_exit_code_trace_full_goal_completion_allowed_count=0`; this is stable local CLI transcript distribution evidence, not live-provider, human-acceptance, production, release, or full-goal completion proof. The readiness trace preserves `focused_report_readiness_dimension_trace_count=40`, `focused_report_readiness_dimension_trace_entry_total=320`, `focused_report_readiness_dimension_trace_human_required_total=80`, `focused_report_readiness_dimension_trace_not_verified_total=120`, and `focused_report_readiness_dimension_trace_full_goal_completion_allowed_count=0`; this is a focused-report status split, not live-provider, human-acceptance, production, or full-goal completion proof. The research trace preserves `research_perspective_trace_count=40`, `research_perspective_trace_ready_count=40`, `research_perspective_trace_perspective_entry_total=200`, and `research_perspective_trace_full_goal_completion_allowed_count=0`; this is local result-doc senior-perspective coverage, not external senior-review completion proof. The research decision-bridge trace preserves `research_perspective_decision_bridge_trace_count=40`, `research_perspective_decision_bridge_trace_entry_total=200`, `research_perspective_decision_bridge_trace_dimension_link_total=440`, and `research_perspective_decision_bridge_trace_full_goal_completion_allowed_count=0`; this is local decision-trace grounding, not product or human acceptance proof. The implementation trace preserves `implementation_solution_trace_count=40`, `implementation_solution_trace_ready_count=40`, `implementation_solution_trace_approach_entry_total=40`, `implementation_solution_trace_smallest_solution_entry_total=40`, and `implementation_solution_trace_full_goal_completion_allowed_count=0`; this is local result-doc evidence that each delivery unit defined its implementation approach and smallest complete solution, not live-provider, release, or full-goal completion proof. The refactor trace preserves `refactor_hardening_trace_count=40`, `refactor_hardening_trace_ready_count=40`, `refactor_hardening_trace_text_entry_total=40`, and `refactor_hardening_trace_full_goal_completion_allowed_count=0`; this is local result-doc evidence that each delivery unit recorded focused hardening against its proof boundary, not live-provider, release, or full-goal completion proof. The documented-result trace preserves `documented_result_trace_count=40`, `documented_result_trace_ready_count=40`, `documented_result_trace_text_entry_total=40`, `documented_result_trace_required_term_total=280`, and `documented_result_trace_full_goal_completion_allowed_count=0`; this is local result-doc evidence that each scenario result was recorded before moving to the next unit, not live-provider, release, or full-goal completion proof. The adoption contribution trace preserves `adoption_contribution_trace_count=40`, `adoption_contribution_trace_ready_count=40`, `adoption_contribution_trace_text_entry_total=40`, `adoption_contribution_trace_unique_concept_count=40`, `adoption_contribution_trace_surface_count=5`, and `adoption_contribution_trace_full_goal_completion_allowed_count=0`; this is local result-doc evidence that each scenario result names its ConnectorHub concept and CornerStone adoption surface, not live-provider, release, or full-goal completion proof. The Must Pass delivery-unit audit preserves `must_pass_delivery_unit_audit_decision_rationale_trace_ready_count=34`, `must_pass_delivery_unit_audit_research_decision_bridge_trace_ready_count=34`, `must_pass_delivery_unit_audit_implementation_trace_ready_count=34`, and `must_pass_delivery_unit_audit_documented_result_trace_ready_count=34`, making decision rationale, senior-research grounding, implementation approach/smallest solution, and result documentation explicit for every Must-Pass row without expanding the local proof claim. It also preserves an enriched `human_required_work_mapping` with `human_required_work_mapping_trace_ready_count=7`, `human_required_work_mapping_required_human_field_total=72`, `human_required_work_mapping_required_evidence_total=29`, `human_required_work_mapping_why_ai_cannot_verify_present_count=7`, `human_required_work_mapping_required_human_action_present_count=7`, `human_required_work_mapping_expected_evidence_summary_present_count=7`, `human_required_work_mapping_source_release_impact_present_count=7`, `human_required_work_mapping_source_requirement_link_total=17`, and zero product/PASS/approval claim counters for H01-H07; this is operator-preparation trace only, not human acceptance. `make generate-connectorhub-engineering-trail-manifest` refreshes the pinned human-gate package, readiness, next-selector, and validation-handoff artifacts before writing manifests, and `make verify-connectorhub-engineering-trail` checks the result.

The source open-question trace preserves `source_open_question_operator_handoff_ready_count=10`, `source_open_question_completion_blocked_until_human_accept_count=10`, `source_open_question_next_human_gate_unique_count=6`, `source_open_question_operator_package_command_count=10`, and `source_open_question_operator_validation_output_command_count=10`, mapping the ten unresolved source decisions to the earliest applicable human gate in the H04 -> H07 -> H01 -> H02 -> H03 -> H05 -> H06 review order. This is operator handoff metadata only; it does not resolve owner decisions, collect acceptance, or promote any H row.

The application-guide freeze-decision trace preserves `application_guide_freeze_decision_owner_handoff_ready_count=10`, `application_guide_freeze_decision_human_gate_handoff_ready_count=9`, `application_guide_freeze_decision_owner_only_count=1`, `application_guide_freeze_decision_completion_blocked_until_owner_freeze_count=10`, `application_guide_freeze_decision_next_human_gate_unique_count=5`, `application_guide_freeze_decision_operator_package_command_count=9`, and `application_guide_freeze_decision_operator_validation_output_command_count=9`, mapping unresolved implementation-freeze decisions to owner artifacts and, where applicable, the earliest H-gate handoff. This is owner-review preparation only; it does not record owner freeze, release approval, or product acceptance.

The application-guide risk-mitigation trace preserves `application_guide_risk_mitigation_operator_handoff_ready_count=10`, `application_guide_risk_mitigation_acceptance_blocked_until_human_evidence_count=10`, `application_guide_risk_mitigation_next_human_gate_unique_count=4`, `application_guide_risk_mitigation_operator_package_command_count=10`, and `application_guide_risk_mitigation_operator_validation_output_command_count=10`, mapping each guide risk to the earliest applicable H-gate evidence handoff. This is mitigation-review preparation only; it does not verify the mitigation, collect acceptance, or promote any H row.

The application-guide adoption-conclusion trace preserves `application_guide_adoption_conclusion_operator_handoff_ready_count=3`, `application_guide_adoption_conclusion_release_acceptance_blocked_until_human_evidence_count=3`, `application_guide_adoption_conclusion_final_adoption_blocked_until_release_acceptance_count=3`, `application_guide_adoption_conclusion_next_human_gate_unique_count=1`, `application_guide_adoption_conclusion_operator_package_command_count=3`, and `application_guide_adoption_conclusion_operator_validation_output_command_count=3`, mapping each strategic adoption conclusion to the earliest applicable H04 evidence handoff. This is strategy-review preparation only; it does not grant release acceptance or final ConnectorHub adoption.

The application-guide integration-architecture trace preserves `application_guide_integration_architecture_operator_handoff_ready_count=6`, `application_guide_integration_architecture_runtime_blocked_until_runtime_evidence_count=6`, `application_guide_integration_architecture_release_blocked_until_human_evidence_count=6`, `application_guide_integration_architecture_next_human_gate_unique_count=3`, `application_guide_integration_architecture_operator_package_command_count=6`, and `application_guide_integration_architecture_operator_validation_output_command_count=6`, mapping each architecture guidance section to its earliest applicable H-gate evidence handoff. This is architecture-review preparation only; it does not prove runtime implementation, release acceptance, or final ConnectorHub adoption.

The application-guide verification-plane trace preserves `application_guide_verification_plane_operator_handoff_ready_count=6`, `application_guide_verification_plane_native_verifier_command_total=5`, `application_guide_verification_plane_release_evidence_item_total=11`, `application_guide_verification_plane_negative_counter_item_total=12`, `application_guide_verification_plane_human_required_gate_item_total=7`, `application_guide_verification_plane_recommended_sequence_step_total=5`, `application_guide_verification_plane_runtime_claim_allowed_count=0`, `application_guide_verification_plane_release_claim_allowed_count=0`, `application_guide_verification_plane_human_acceptance_claim_allowed_count=0`, and `application_guide_verification_plane_completion_blocked_until_release_evidence_count=6`. This is verification-plane handoff metadata only; it does not prove runtime behavior, release acceptance, or human gate completion.

The application-guide source-register trace preserves `application_guide_source_register_trace_ready_count=5`, `application_guide_source_register_operator_handoff_ready_count=5`, `application_guide_source_register_cornerstone_authority_source_total=7`, `application_guide_source_register_connectorhubkit_evidence_total=20`, `application_guide_source_register_user_instruction_total=3`, `application_guide_source_register_external_primary_reference_total=2`, `application_guide_source_register_glossary_term_total=11`, `application_guide_source_register_dependency_adoption_claim_allowed_count=0`, `application_guide_source_register_live_reference_claim_allowed_count=0`, `application_guide_source_register_runtime_claim_allowed_count=0`, `application_guide_source_register_release_claim_allowed_count=0`, `application_guide_source_register_human_acceptance_claim_allowed_count=0`, and `application_guide_source_register_completion_blocked_until_source_review_count=5`. This is source-register handoff metadata only; it does not treat ConnectorHubKit evidence, external references, or glossary terms as dependency adoption, live-reference proof, runtime behavior, release acceptance, or human acceptance.

The application-guide scenario-detail trace preserves `application_guide_scenario_detail_trace_ready_count=40`, `application_guide_scenario_detail_operator_handoff_ready_count=40`, `application_guide_scenario_detail_title_match_count=40`, `application_guide_scenario_detail_phase_match_count=40`, `application_guide_scenario_detail_type_match_count=40`, `application_guide_scenario_detail_additional_pass_claim_allowed_count=0`, `application_guide_scenario_detail_runtime_claim_allowed_count=0`, `application_guide_scenario_detail_release_claim_allowed_count=0`, and `application_guide_scenario_detail_human_acceptance_claim_allowed_count=0`. This is source-to-matrix traceability and operator handoff metadata only; it does not create additional PASS evidence beyond the focused local result docs, and it does not prove runtime behavior, release acceptance, or human acceptance.

The source context-and-assumptions trace preserves `source_context_assumptions_section_count=7`, `source_context_assumptions_item_total=29`, `source_context_assumptions_trace_ready_count=7`, `source_context_assumptions_operator_handoff_ready_count=7`, `source_context_assumptions_assumption_item_total=8`, `source_context_assumptions_status_neutral_rule_item_total=2`, `source_context_assumptions_owner_freeze_recorded_count=0`, `source_context_assumptions_product_claim_allowed_count=0`, `source_context_assumptions_runtime_claim_allowed_count=0`, `source_context_assumptions_release_claim_allowed_count=0`, `source_context_assumptions_human_acceptance_claim_allowed_count=0`, and `source_context_assumptions_completion_blocked_until_owner_freeze_count=7`. This is front-matter and assumption handoff metadata only; it does not record owner freeze, product verification, runtime behavior, release acceptance, or human acceptance.

The source initial-scenario trace preserves `source_initial_scenario_row_count=47`, `source_initial_scenario_trace_ready_count=47`, `source_initial_scenario_operator_handoff_ready_count=47`, `source_initial_scenario_ai_row_count=40`, `source_initial_scenario_human_required_count=7`, `source_initial_scenario_source_row_pass_claim_allowed_count=0`, `source_initial_scenario_product_claim_allowed_count=0`, `source_initial_scenario_runtime_claim_allowed_count=0`, `source_initial_scenario_release_claim_allowed_count=0`, `source_initial_scenario_human_acceptance_claim_allowed_count=0`, and `source_initial_scenario_completion_claim_allowed_count=0`. This is source-to-matrix reconciliation only; the source document's status-neutral rows do not create PASS, product, runtime, release, human-acceptance, or completion proof.

The source-required implementation trace preserves `source_required_implementation_total_count=47`, `source_required_implementation_ai_owned_count=40`, `source_required_implementation_human_gate_owned_count=7`, `source_required_implementation_operator_handoff_ready_count=47`, `source_required_implementation_human_evidence_blocked_count=7`, `source_required_implementation_pass_claim_allowed_count=0`, `source_required_implementation_runtime_claim_allowed_count=0`, `source_required_implementation_release_claim_allowed_count=0`, and `source_required_implementation_human_acceptance_claim_allowed_count=0`. This is owner-split source work mapping only; it does not prove release readiness, runtime behavior, or human-gate acceptance.

The source architecture contract trace preserves `source_architecture_contract_section_count=11`, `source_architecture_contract_required_term_total=100`, `source_architecture_contract_trace_ready_count=11`, `source_architecture_contract_operator_handoff_ready_count=11`, `source_architecture_contract_runtime_claim_allowed_count=0`, `source_architecture_contract_release_claim_allowed_count=0`, `source_architecture_contract_human_acceptance_claim_allowed_count=0`, and `source_architecture_contract_completion_blocked_until_runtime_evidence_count=11`. This is implementation-document architecture traceability and operator handoff metadata only; it does not prove runtime behavior, release acceptance, or human acceptance.

The source verification-plan trace preserves `source_verification_plan_section_count=10`, `source_verification_plan_item_total=77`, `source_verification_plan_trace_ready_count=10`, `source_verification_plan_operator_handoff_ready_count=10`, `source_verification_plan_negative_counter_item_total=12`, `source_verification_plan_human_required_gate_item_total=7`, `source_verification_plan_not_yet_verifiable_item_total=4`, `source_verification_plan_completion_criteria_item_total=9`, `source_verification_plan_runtime_claim_allowed_count=0`, `source_verification_plan_release_claim_allowed_count=0`, `source_verification_plan_human_acceptance_claim_allowed_count=0`, and `source_verification_plan_completion_blocked_until_runtime_or_release_evidence_count=10`. This is source verification-plane handoff metadata only; it does not prove planned verifiers have run, release evidence exists, or human acceptance is complete.

`make verify-connector-contract-adapter` refreshes the current VS2 local proof and VS2 scenario report before focused ConnectorHub replay, then replays the matrix-backed 40 AI-owned ConnectorHub scenarios as independent focused verify/gate pairs, runs the aggregate verify/gate pair, and runs `python3 -m unittest tests.scenario.test_connectorhub_cli`.

## Local Verification Plane

Local verification is a product acceptance surface, not an ad-hoc test folder. Scenario `PASS` requires deterministic evidence over product records, policy decisions, workflow/action records, audit events, CLI transcripts, UI traces where relevant, and scenario reports.

The release-facing local proof path is planned around `cornerstone scenario verify <contract> --json`. Local LLMs may help generate outputs, but deterministic validators own `PASS` or `FAIL`.

## Design System

CornerStone's design doctrine is **Calm Surface. Deep Evidence. Safe Action.**

The default product surface is a light, calm universal workspace for drop, search, ask, recent work, and quiet knowledge states. Admin surfaces use the same calm shell but expose connectors, policies, roles, namespaces, and audit logs only in admin context.

The default theme is light. Do not make the first screen a chatbot, admin dashboard, dark command center, connector panel, or ontology setup surface.

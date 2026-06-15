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

## Local Verification Plane

Local verification is a product acceptance surface, not an ad-hoc test folder. Scenario `PASS` requires deterministic evidence over product records, policy decisions, workflow/action records, audit events, CLI transcripts, UI traces where relevant, and scenario reports.

The release-facing local proof path is planned around `cornerstone scenario verify <contract> --json`. Local LLMs may help generate outputs, but deterministic validators own `PASS` or `FAIL`.

## Design System

CornerStone's design doctrine is **Calm Surface. Deep Evidence. Safe Action.**

The default product surface is a light, calm universal workspace for drop, search, ask, recent work, and quiet knowledge states. Admin surfaces use the same calm shell but expose connectors, policies, roles, namespaces, and audit logs only in admin context.

The default theme is light. Do not make the first screen a chatbot, admin dashboard, dark command center, connector panel, or ontology setup surface.

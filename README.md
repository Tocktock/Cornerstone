# CornerStone

**Date:** 2026-06-09
**Owner:** JiYong / Tars  
**Status:** Documentation authority reset with full AI-agent handoff, MUST-PASS scenarios, CLI-native gate, local verification plane, design-system contract, VS-0 scaffold gate, VS-0 scaffold readiness report, and no-dependency scaffold CLI bootstrap
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
15. `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md`
16. `docs/agent/PROJECT_OPERATING_CONSTITUTION.md`
17. `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md`

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
scripts/verify_design_system_docs.sh
scripts/verify_vs0_scaffold_readiness_docs.sh
```

Verify the current scaffold bootstrap with:

```sh
export PATH="$PWD:$PATH"
cornerstone --help
cornerstone version --json
cornerstone health --json
cornerstone ready --json        # exits 4 until product runtime exists
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
make verify-local-fast
```

The current scaffold CLI can verify scaffold readiness, scenario registry coverage, deterministic local fixture-validator readiness, the first CLI-native artifact preservation slice, the first redaction/prompt-injection safety slice, the first search/evidence-bundle/draft-claim/evidence-viewer slice, the first deterministic search-understanding slice, the first owner/namespace isolation slice, the first tamper-evident audit-ledger slice, the first universal non-logistics core slice, the first claim evidence-gating slice, the first default-deny egress/sandbox policy slice, the first regression guardrail summary slice, and the first evidence-backed briefing slice. It does not claim the full VS-0 product loop is implemented or passing.

## First Implementation Target

Before feature coding, the VS-0 setup-planning gate is `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md`. It defines the version baseline, monorepo direction, CLI scaffold expectations, verification report shape, and human approval gates for production dependency additions.

Before scaffold implementation, read `docs/verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md`. The current gate allows VS-0 scaffold foundation work only after preflight and approval. It does not allow VS-0 product-feature implementation or any claim that local verification is implemented.

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

## Design System

CornerStone's design doctrine is **Calm Surface. Deep Evidence. Safe Action.**

The default product surface is a light, calm universal workspace for drop, search, ask, recent work, and quiet knowledge states. Admin surfaces use the same calm shell but expose connectors, policies, roles, namespaces, and audit logs only in admin context.

The default theme is light. Do not make the first screen a chatbot, admin dashboard, dark command center, connector panel, or ontology setup surface.

# Connector Hub CS-CH-014 Verification Report - 2026-06-23

## Result

- Scenario set: `connector-contract-adapter`
- Filtered scenario: `CS-CH-014`
- Filtered status: `success`
- Filtered PASS rows: `1`
- Filtered blocking rows: `0`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Unfiltered connector-contract-adapter status: `success`
- Unfiltered verified rows: `40 PASS, 0 blocking`
- Verified base commit: `cb682e9`
- Worktree dirty at verification: `true`

## What Changed

- Added a malicious Connector Projection fixture for prompt/tool/action/memory/egress manipulation text.
- Added `cs.connector_untrusted_content_review.v1` review records linked to Artifact, Delivery Receipt, Projection Snapshot, Evidence Link, and audit refs.
- Added `cornerstone connector untrusted-content review --delivery-receipt-id ... --json`.
- Extended connector Evidence Bundle output with trust-boundary coverage for untrusted content.
- Extended `connector-contract-adapter` verification, Make gating, and unittest coverage with filtered `CS-CH-014`.

## Scenario Verification

| ID | Type | Status | Evidence | Notes |
|---|---|---|---|---|
| CS-CH-014 | REGRESSION_GUARD | PASS | `reports/scenario/connector-contract-adapter-cs-ch-014-2026-06-23.json` | Untrusted connector content can be archived, searched, quoted, and cited only as evidence; prompt/tool/action/egress/memory/policy manipulation attempts are blocked with zero side effects and audit refs. |

## Evidence Summary

Filtered report:

```text
reports/scenario/connector-contract-adapter-cs-ch-014-2026-06-23.json
status=success
scenario_count=1
pass=1
blocking=0
product_feature_claims=LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING
untrusted_state_dir=tmp/scenario/connector-contract-adapter-untrusted-content-24472
prompt_injection_delivery_path=fixtures/connectorhub/deliveries/github_issue_projection_delivery_prompt_injection.json
untrusted_delivery_receipt_id=cdelrec_5cdf75cee183d874
untrusted_artifact_id=art_2d5ff8553f984bbf
untrusted_review_id=cuntrust_811a74a4c8657f32
untrusted_evidence_bundle_id=evb_40681647adcfa137
untrusted_claim_id=claim_39f2568f2a08c368
untrusted_memory_quarantine_id=memq_f91a388453478f13
```

Unfiltered report:

```text
reports/scenario/connector-contract-adapter-2026-06-23.json
status=success
scenario_count=40
pass=40
not_verified=0
blocking=0
product_feature_claims=LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING
```

CS-CH-014 checks recorded in the filtered report:

```text
untrusted_contract_validate_exit_zero=true
untrusted_setup_plan_exit_zero=true
untrusted_delivery_process_exit_zero=true
untrusted_review_show_exit_zero=true
untrusted_evidence_bundle_exit_zero=true
untrusted_claim_create_exit_zero=true
untrusted_audit_verify_exit_zero=true
untrusted_records_persisted=true
artifact_and_source_labeled_untrusted=true
unsafe_instruction_blocked=true
content_treated_as_evidence_only=true
review_negative_counters_zero=true
bundle_trust_boundary_coverage=true
evidence_item_distinguishes_untrusted_review=true
claim_quotes_instruction_without_authority=true
agent_prompt_authority_denied=true
memory_promotion_quarantined=true
egress_denied_without_http_call=true
zero_action_workflow_memory_side_effects=true
raw_provider_payload_not_exposed=true
evidence_refs_present=true
audit_refs_present=true
zero_provider_internals=true
zero_secret_findings=true
no_unauthorized_marker=true
```

Negative evidence:

```text
tool_calls_from_untrusted_connector_content=0
action_cards_from_untrusted_connector_content=0
workflow_runs_from_untrusted_connector_content=0
connector_actions_from_untrusted_connector_content=0
provider_calls_from_untrusted_connector_content=0
shell_calls_from_untrusted_connector_content=0
external_calls_from_untrusted_connector_content=0
memory_promotions_from_untrusted_connector_content=0
policy_overrides_from_untrusted_connector_content=0
authority_expansions_from_untrusted_connector_content=0
unauthorized_provider_calls=0
provider_credentials_exposed=0
raw_provider_payloads_exposed=0
production_readiness_overclaims=0
```

## Implementation Decision

The smallest complete solution treats connector content as evidence with an explicit safety review, not as executable or authoritative instruction. Delivery processing records unsafe instruction findings and zero side-effect counters in `cs.connector_untrusted_content_review.v1`, then Evidence Bundle assembly carries those trust-boundary facts into Product evidence.

The scenario uses existing Product guardrails for the dependent surfaces: Agent prompt authority is denied by policy, unsafe memory promotion is quarantined, default egress is denied without a network call, and state scans verify no ActionCard, WorkflowRun, or trusted memory record was created.

## Commands Run

| Command | Result |
|---|---|
| `python3 -m compileall packages/cornerstone_cli` | PASS |
| `python3 -m unittest tests.scenario.test_connectorhub_cli.ConnectorHubCliTests.test_connector_untrusted_content_cannot_direct_agents_or_actions_cs_ch_014` | PASS |
| `python3 -m unittest tests.scenario.test_connectorhub_cli.ConnectorHubCliTests.test_connectorhub_scenario_list_and_filtered_verify` | PASS |
| `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-014 --json --output reports/scenario/connector-contract-adapter-cs-ch-014-2026-06-23.json` | PASS; report status `success`, 1 PASS, 0 blocking |
| `cornerstone scenario verify connector-contract-adapter --json --output reports/scenario/connector-contract-adapter-2026-06-23.json` | PASS; report status `success`, 40 PASS, 0 blocking |
| `make verify-connector-contract-adapter` | PASS; full gate status `success`, 40 PASS, 0 blocking, Connector Hub unittest suite 21 tests OK |

## Proof Surface

- `proof_surface`: `local_fixture`
- `claim_boundary`: deterministic local fixture evidence only; no live-provider production or human-acceptance claim

## Senior Engineering Decision Trail

- Product value: `CS-CH-014` advances Connector Hub adoption in CornerStone by proving `Untrusted connector content cannot direct agents or actions` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Connector content remains untrusted evidence and cannot trigger authority tool action memory promotion policy override or egress`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-014` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-014`, phase `CH-1`, related requirements `IR-10;IR-11;IR-18`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter-cs-ch-014-2026-06-23.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-014 --json`; the expected method is `Prompt-injection delivery review evidence-bundle claim agent memory and egress tests`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-014` as the independent delivery unit for `Untrusted connector content cannot direct agents or actions`.
- Implementation approach: use `Prompt-injection delivery review evidence-bundle claim agent memory and egress tests` against matrix row `CS-CH-014`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Connector content remains untrusted evidence and cannot trigger authority tool action memory promotion policy override or egress` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter-cs-ch-014-2026-06-23.json` as the acceptance record.
- Refactor and hardening: `CS-CH-014` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter-cs-ch-014-2026-06-23.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-014` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `untrusted connector-content authority guard` into the CornerStone adoption surface `Durable evidence archive policy audit and safety guardrails`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Boundaries

This report claims local fixture proof for `CS-CH-014`.

It does not claim:

- live browser/document-provider prompt-injection variants beyond the local CS-CH-014 fixture;
- rendered UI/API proof;
- live GitHub read-only readiness;
- production tenancy, RLS, OPA, egress topology, backup/restore, or release readiness.

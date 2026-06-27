# Connector Hub CS-CH-011 Verification Report - 2026-06-23

## Result

- Scenario set: `connector-contract-adapter`
- Filtered scenario: `CS-CH-011`
- Filtered status: `success`
- Filtered PASS rows: `1`
- Filtered blocking rows: `0`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Unfiltered connector-contract-adapter status: `success`
- Unfiltered verified rows: `40 PASS, 0 blocking`
- Verified base commit: `cb682e9`
- Worktree dirty at verification: `true`

## What Changed

- Added Projection Source Policy decision records with schema `cs.connector_projection_policy_decision.v1`.
- Added a pre-archive Source Policy enforcement gate for Projection Delivery payload fields, selected resources, allowed paths, raw-access denial, and max-content size.
- Added forbidden full-body and narrowed max-content fixtures for local negative proof.
- Extended `connector-contract-adapter` scenario verification and Make gating with filtered `CS-CH-011`.
- Added negative evidence counters for forbidden Source Policy field leaks and raw content policy leaks.

## Scenario Verification

| ID | Type | Status | Evidence | Notes |
|---|---|---|---|---|
| CS-CH-011 | MUST_PASS | PASS | `reports/scenario/connector-contract-adapter-cs-ch-011-2026-06-23.json` | Allowed preview fields are normalized and archived with a policy decision; forbidden full-body fields and over-limit payloads are blocked before Artifact, receipt, current-version, or Product state creation. |

## Evidence Summary

Filtered report:

```text
reports/scenario/connector-contract-adapter-cs-ch-011-2026-06-23.json
status=success
scenario_count=1
pass=1
blocking=0
product_feature_claims=LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING
policy_state_dir=tmp/scenario/connector-contract-adapter-policy-38982
policy_allowed_decision_id=cpdec_87c854bb31298cbb
policy_forbidden_decision_id=cpdec_8417d652b464a398
policy_oversized_decision_id=cpdec_673c50e87a050134
policy_allowed_artifact_id=art_79b05a96a3bcc811
policy_allowed_delivery_receipt_id=cdelrec_e06f82e5dcdf0049
policy_narrowed_source_policy_id=cspol_8c99d67836148d8c
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

CS-CH-011 checks recorded in the filtered report:

```text
policy_contract_validate_exit_zero=true
policy_setup_plan_exit_zero=true
policy_allowed_process_exit_zero=true
allowed_decision_persisted=true
allowed_projection_normalized=true
allowed_summary_preserves_restriction=true
allowed_artifact_and_receipt_persisted=true
forbidden_body_rejected=true
forbidden_decision_persisted=true
forbidden_full_body_value_not_persisted=true
narrowed_source_policy_persisted=true
narrowed_policy_applies_to_subsequent_delivery=true
oversized_decision_persisted=true
blocked_deliveries_created_no_artifacts_or_receipts=true
policy_decision_count=true
raw_content_not_persisted=true
policy_decision_refs_present=true
evidence_refs_present=true
audit_refs_present=true
policy_audit_verify_exit_zero=true
zero_provider_internals=true
zero_secret_findings=true
```

Negative evidence:

```text
forbidden_source_policy_field_leaks=0
raw_content_policy_leaks=0
unauthorized_provider_calls=0
provider_credentials_exposed=0
raw_provider_payloads_exposed=0
production_readiness_overclaims=0
```

## Implementation Decision

The smallest complete solution makes Source Policy enforcement a pre-archive Product safety gate. Allowed deliveries are normalized to the permitted field set, then archived through the existing immutable Artifact and receipt path. Rejected deliveries still persist a policy decision and audit evidence, but they do not create Artifact, receipt, Projection snapshot, content-version, current-version, or Product truth records.

The policy decision record is separate from the Artifact so future Postgres/RLS storage can map it to a durable policy-decision table with delivery, setup, Source Policy, evidence, and audit references. This keeps enforcement observable without storing raw provider payloads or forbidden full-body content in Product state.

## Commands Run

| Command | Result |
|---|---|
| `python3 -m compileall packages/cornerstone_cli` | PASS |
| `python3 -m unittest tests.scenario.test_connectorhub_cli.ConnectorHubCliTests.test_connector_delivery_enforces_source_policy_restrictions_cs_ch_011` | PASS |
| `python3 -m unittest tests.scenario.test_connectorhub_cli.ConnectorHubCliTests.test_connectorhub_scenario_list_and_filtered_verify` | PASS |
| `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-011 --json --output reports/scenario/connector-contract-adapter-cs-ch-011-2026-06-23.json` | PASS; report status `success`, 1 PASS, 0 blocking |
| `cornerstone scenario verify connector-contract-adapter --json --output reports/scenario/connector-contract-adapter-2026-06-23.json` | PASS; report status `success`, 40 PASS, 0 blocking |
| `make verify-connector-contract-adapter` | PASS; full gate status `success`, 40 PASS, 0 blocking, Connector Hub unittest suite 21 tests OK |

## Proof Surface

- `proof_surface`: `local_fixture`
- `claim_boundary`: deterministic local fixture evidence only; no live-provider production or human-acceptance claim

## Senior Engineering Decision Trail

- Product value: `CS-CH-011` advances Connector Hub adoption in CornerStone by proving `Enforce field and body restrictions from Source Policy` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Disallowed fields are rejected or stripped before durable Product state with policy evidence`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-011` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-011`, phase `CH-1`, related requirements `IR-08;IR-09`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter-cs-ch-011-2026-06-23.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-011 --json`; the expected method is `Projection policy validator tests`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-011` as the independent delivery unit for `Enforce field and body restrictions from Source Policy`.
- Implementation approach: use `Projection policy validator tests` against matrix row `CS-CH-011`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Disallowed fields are rejected or stripped before durable Product state with policy evidence` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter-cs-ch-011-2026-06-23.json` as the acceptance record.
- Refactor and hardening: `CS-CH-011` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter-cs-ch-011-2026-06-23.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-011` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `Source Policy field and body enforcement` into the CornerStone adoption surface `Durable evidence archive policy audit and safety guardrails`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Boundaries

This report claims local fixture proof for `CS-CH-011`.

It does not claim:

- live browser/document-provider prompt-injection variants beyond the local CS-CH-014 fixture;
- rendered UI/API proof;
- live GitHub read-only readiness;
- production tenancy, RLS, OPA, egress, backup/restore, or release readiness.

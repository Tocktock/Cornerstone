# Connector Hub CS-CH-009 Verification Report - 2026-06-23

## Result

- Scenario set: `connector-contract-adapter`
- Filtered scenario: `CS-CH-009`
- Filtered status: `success`
- Filtered PASS rows: `1`
- Filtered blocking rows: `0`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Unfiltered connector-contract-adapter status: `success`
- Unfiltered verified rows: `40 PASS, 0 blocking`
- Verified base commit: `cb682e9`
- Worktree dirty at verification: `true`

## What Changed

- Added native CLI path `cornerstone connector delivery process --file ... --contract-id ... --failure-mode transient --json`.
- Added native quarantine paths `cornerstone connector quarantine list --json` and `cornerstone connector quarantine replay --quarantine-id ... --json`.
- Added deterministic retry state for transient Delivery failures.
- Added deterministic quarantine state for poison Delivery failures at the configured threshold.
- Added malformed Projection Delivery fixture `fixtures/connectorhub/deliveries/github_issue_poison_projection_delivery.json`.
- Extended `connector-contract-adapter` scenario verification and Make gating with filtered `CS-CH-009`.

## Scenario Verification

| ID | Type | Status | Evidence | Notes |
|---|---|---|---|---|
| CS-CH-009 | MUST_PASS | PASS | `reports/scenario/connector-contract-adapter/scenarios/CS-CH-009.json` | Transient Delivery failures retry with bounded backoff, poison Delivery reaches quarantine at the configured threshold with safe diagnostics, unrelated healthy Delivery still archives and acknowledges, and replay preserves failure evidence. |

## Evidence Summary

Filtered report:

```text
reports/scenario/connector-contract-adapter/scenarios/CS-CH-009.json
status=success
scenario_count=1
pass=1
blocking=0
product_feature_claims=LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING
transient_retry_state=cret_87c854bb31298cbb
poison_retry_state=cret_4c6baad8f5c3e1ac
poison_quarantine=cquar_3751cdaf2a1d7a5d
state_dir=tmp/scenario/connector-contract-adapter-retry-82649
```

Unfiltered report:

```text
reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json
status=success
scenario_count=40
pass=40
not_verified=0
blocking=0
product_feature_claims=LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING
```

CS-CH-009 checks recorded in the filtered report:

```text
retry_contract_validate_exit_zero=true
retry_setup_plan_exit_zero=true
transient_first_exit_runtime_failure=true
transient_first_retry_scheduled=true
transient_second_exit_runtime_failure=true
transient_bounded_backoff=true
transient_retry_state_persisted=true
healthy_stream_continues=true
poison_first_exit_runtime_failure=true
poison_first_retry_scheduled=true
poison_second_exit_runtime_failure=true
poison_second_retry_scheduled=true
poison_third_exit_connector_unavailable=true
poison_quarantined_at_threshold=true
poison_retry_and_quarantine_persisted=true
quarantine_safe_diagnostics=true
unrelated_streams_continue=true
quarantine_list_exit_zero=true
quarantine_replay_exit_zero=true
evidence_refs_present=true
audit_refs_present=true
audit_verify_exit_zero=true
zero_provider_internals=true
zero_secret_findings=true
```

Negative evidence:

```text
infinite_retry_loops=0
queue_wide_blockage=0
raw_payload_in_quarantine_output=0
unauthorized_provider_calls=0
provider_credentials_exposed=0
raw_provider_payloads_exposed=0
production_readiness_overclaims=0
```

## Implementation Decision

The smallest complete solution keeps retry/quarantine state separate from the healthy archive/ack path. Transient failure simulation records bounded retry attempts with deterministic backoff and no acknowledgement. Malformed poison delivery uses the same retry policy, then moves to quarantine at the contract threshold with only safe metadata, redacted error codes, source-health impact, and replay linkage.

This preserves CS-CH-007 and CS-CH-008 semantics: healthy deliveries still archive immutable Artifacts and acknowledge after commit, while failed deliveries retain operational evidence without storing raw provider payloads or blocking unrelated streams.

## Commands Run

| Command | Result |
|---|---|
| `python3 -m compileall packages/cornerstone_cli` | PASS |
| `python3 -m unittest tests.scenario.test_connectorhub_cli` | PASS, 19 tests |
| `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-009 --json` | PASS; report status `success`, 1 PASS, 0 blocking |
| `make verify-connector-contract-adapter` | PASS; full report status `success`, 40 PASS, 0 blocking |

## Proof Surface

- `proof_surface`: `local_fixture`
- `claim_boundary`: deterministic local fixture evidence only; no live-provider production or human-acceptance claim

## Senior Engineering Decision Trail

- Product value: `CS-CH-009` advances Connector Hub adoption in CornerStone by proving `Retry transient failures and quarantine poison deliveries` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Bounded retry handles transient failures and poison deliveries enter quarantine safely`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-009` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-009`, phase `CH-1`, related requirements `IR-07;IR-17`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-009.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-009 --json`; the expected method is `Retry clock and malformed payload tests`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-009` as the independent delivery unit for `Retry transient failures and quarantine poison deliveries`.
- Implementation approach: use `Retry clock and malformed payload tests` against matrix row `CS-CH-009`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Bounded retry handles transient failures and poison deliveries enter quarantine safely` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-009.json` as the acceptance record.
- Refactor and hardening: `CS-CH-009` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter/scenarios/CS-CH-009.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-009` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `bounded Delivery retry and quarantine` into the CornerStone adoption surface `Durable evidence archive policy audit and safety guardrails`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Boundaries

This report claims local fixture proof for `CS-CH-009`.

It does not claim:

- changed-content lineage;
- rendered UI/API proof;
- live GitHub read-only readiness;
- production tenancy, RLS, OPA, egress, backup/restore, or release readiness.

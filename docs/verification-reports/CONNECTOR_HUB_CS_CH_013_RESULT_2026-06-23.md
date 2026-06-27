# Connector Hub CS-CH-013 Verification Report - 2026-06-23

## Result

- Scenario set: `connector-contract-adapter`
- Filtered scenario: `CS-CH-013`
- Filtered status: `success`
- Filtered PASS rows: `1`
- Filtered blocking rows: `0`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Unfiltered connector-contract-adapter status: `success`
- Unfiltered verified rows: `40 PASS, 0 blocking`
- Verified base commit: `cb682e9`
- Worktree dirty at verification: `true`

## What Changed

- Added a temporary raw-access request/result lifecycle behind the ConnectorPort boundary.
- Added `cs.connector_raw_access_request.v1`, `cs.connector_raw_access_grant.v1`, and `cs.connector_raw_access_export.v1` records.
- Added CLI support for `cornerstone connector raw-access request`, `read`, `revoke`, and `export`.
- Added a fixture contract that declares `temporary_scoped` raw access with purpose, TTL, max-read, redaction, and revocation limits.
- Extended `connector-contract-adapter` scenario verification and Make gating with filtered `CS-CH-013`.

## Scenario Verification

| ID | Type | Status | Evidence | Notes |
|---|---|---|---|---|
| CS-CH-013 | MUST_PASS | PASS | `reports/scenario/connector-contract-adapter-cs-ch-013-2026-06-23.json` | Temporary raw access is denied by default; declared grants are purpose-bound, human-approved, expiring, read-counted, redacted, scoped, revocable, audited, and exported only as metadata without raw content or reusable handles. |

## Evidence Summary

Filtered report:

```text
reports/scenario/connector-contract-adapter-cs-ch-013-2026-06-23.json
status=success
scenario_count=1
pass=1
blocking=0
product_feature_claims=LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING
raw_access_state_dir=tmp/scenario/connector-contract-adapter-raw-access-90025
raw_access_contract_path=fixtures/connectorhub/contracts/github_raw_access_contract.json
raw_access_request_id=crawreq_fa8639235310f556
raw_access_grant_id=crawgrant_e5242a9565875ffb
raw_access_expiry_grant_id=crawgrant_ec661010d6306344
raw_access_revoke_grant_id=crawgrant_9a8de0871b3abc91
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

CS-CH-013 checks recorded in the filtered report:

```text
raw_default_contract_validate_exit_zero=true
raw_default_setup_plan_exit_zero=true
raw_default_source_policy_denied=true
raw_default_request_denied=true
raw_allowed_contract_validate_exit_zero=true
raw_allowed_setup_plan_exit_zero=true
raw_allowed_policy_limits_persisted=true
raw_ttl_boundary_denied=true
raw_read_limit_boundary_denied=true
raw_grant_exit_zero=true
raw_records_persisted=true
raw_metadata_export_safe=true
raw_read_once_decrements=true
raw_read_limit_exhaustion_denied=true
raw_expiry_denied=true
raw_revoke_exit_zero=true
raw_revoked_read_denied=true
raw_audit_refs_present=true
raw_audit_verify_exit_zero=true
raw_no_provider_payload_or_handle_leak=true
zero_provider_internals=true
zero_secret_findings=true
```

Negative evidence:

```text
reusable_raw_access_handles=0
raw_access_read_limit_bypasses=0
raw_access_expiry_bypasses=0
raw_access_revocation_bypasses=0
raw_access_payload_or_handle_leaks=0
unauthorized_provider_calls=0
provider_credentials_exposed=0
raw_provider_payloads_exposed=0
production_readiness_overclaims=0
```

## Implementation Decision

The smallest complete solution treats raw access as a mediated, short-lived ConnectorHub grant instead of a provider handle or Product-side data copy. A request must point at one EvidenceRef and match a contract plus Source Policy that explicitly declare `temporary_scoped` raw access. The allowed path also requires human approval, purpose, classification, TTL, and read limit checks before a grant record can exist.

Reads decrement the durable grant counter and re-check expiry, exhaustion, and revocation every time. Exports return metadata only, including opaque-reference fingerprints and audit refs, while raw content, provider payloads, reusable handles, and handle material stay out of Product records and reports.

## Commands Run

| Command | Result |
|---|---|
| `python3 -m compileall packages/cornerstone_cli` | PASS |
| `python3 -m unittest tests.scenario.test_connectorhub_cli.ConnectorHubCliTests.test_connector_raw_access_is_denied_and_tightly_bounded_cs_ch_013` | PASS |
| `python3 -m unittest tests.scenario.test_connectorhub_cli.ConnectorHubCliTests.test_connectorhub_scenario_list_and_filtered_verify` | PASS |
| `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-013 --json --output reports/scenario/connector-contract-adapter-cs-ch-013-2026-06-23.json` | PASS; report status `success`, 1 PASS, 0 blocking |
| `cornerstone scenario verify connector-contract-adapter --json --output reports/scenario/connector-contract-adapter-2026-06-23.json` | PASS; report status `success`, 40 PASS, 0 blocking |
| `make verify-connector-contract-adapter` | PASS; full gate status `success`, 40 PASS, 0 blocking, Connector Hub unittest suite 21 tests OK |

## Proof Surface

- `proof_surface`: `local_fixture`
- `claim_boundary`: deterministic local fixture evidence only; no live-provider production or human-acceptance claim

## Senior Engineering Decision Trail

- Product value: `CS-CH-013` advances Connector Hub adoption in CornerStone by proving `Temporary raw access is denied and tightly bounded` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Default raw request is denied and any allowed grant is scoped expiring counted redacted and revocable`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-013` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-013`, phase `CH-1`, related requirements `IR-09;IR-11`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter-cs-ch-013-2026-06-23.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-013 --json`; the expected method is `Denial TTL max-read and revocation tests`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-013` as the independent delivery unit for `Temporary raw access is denied and tightly bounded`.
- Implementation approach: use `Denial TTL max-read and revocation tests` against matrix row `CS-CH-013`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Default raw request is denied and any allowed grant is scoped expiring counted redacted and revocable` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter-cs-ch-013-2026-06-23.json` as the acceptance record.
- Refactor and hardening: `CS-CH-013` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter-cs-ch-013-2026-06-23.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-013` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `temporary scoped raw-access control` into the CornerStone adoption surface `Durable evidence archive policy audit and safety guardrails`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Boundaries

This report claims local fixture proof for `CS-CH-013`.

It does not claim:

- live browser/document-provider prompt-injection variants beyond the local CS-CH-014 fixture;
- rendered UI/API proof;
- live-provider raw-access semantics;
- live GitHub read-only readiness;
- production tenancy, RLS, OPA, egress, backup/restore, or release readiness.

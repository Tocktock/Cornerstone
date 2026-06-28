# Connector Hub CS-CH-006 Verification Report - 2026-06-23

## Result

- Scenario set: `connector-contract-adapter`
- Filtered scenario: `CS-CH-006`
- Filtered status: `success`
- Filtered PASS rows: `1`
- Filtered blocking rows: `0`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Unfiltered connector-contract-adapter status: `success`
- Unfiltered verified rows: `40 PASS, 0 blocking`
- Worktree dirty at verification: `true`

## Scenario Contract

`CS-CH-006` proves that credential and permission gaps produce owner-safe status output without leaking provider internals.

Expected behavior:

- setup planning against a permission-gap Provider Pack returns blocked with exit code `7`;
- blocked state uses reason code `CS_CONNECTOR_PERMISSION_REQUIRED`;
- status explanation includes cause, impact, and safe resolution steps;
- redaction flags show no token, secret, raw provider response, path, or handle exposure;
- provider-call ledger remains zero;
- audit refs exist;
- secret/provider-internal scans remain clean.

## What Changed

- Added permission-gap local Provider Pack `local_source_control_permission_gap.v1`.
- Added `status_explanation` to Setup Result for permission and required-capability gaps.
- Added redaction flags to owner-safe connector status output.
- Extended `connector-contract-adapter` verification to run CS-CH-006 as a real scenario row.
- Extended `tests/scenario/test_connectorhub_cli.py` to assert permission-specific reason code, safe explanation, redaction flags, zero provider calls, and audit refs.


## Scenario Verification

| ID | Type | Status | Evidence | Notes |
|---|---|---|---|---|
| CS-CH-006 | MUST_PASS | PASS | `reports/scenario/connector-contract-adapter/scenarios/CS-CH-006.json` | Status explains cause impact and safe resolution without exposing secrets paths handles or raw responses |

## Evidence Summary

Filtered report:

```text
reports/scenario/connector-contract-adapter/scenarios/CS-CH-006.json
status=success
scenario_count=1
pass=1
blocking=0
product_feature_claims=LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING
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

Verifier checks recorded in the filtered report:

```text
permission_gap_exit_connector_unavailable=true
audit_verify_exit_zero=true
permission_gap_setup_persisted=true
permission_gap_source_policy_persisted=true
setup_status_blocked=true
setup_readiness_blocked=true
activation_state_permission_required=true
blocked_reason_code_permission_required=true
status_explanation_reason_specific=true
status_explanation_has_cause_impact_resolution=true
status_explanation_owner_safe=true
redaction_flags_false=true
zero_provider_calls_before_activation=true
evidence_refs_present=true
audit_refs_present=true
zero_provider_internals=true
zero_secret_findings=true
```

Negative evidence:

```text
unauthorized_provider_calls=0
provider_credentials_exposed=0
raw_provider_payloads_exposed=0
ownerless_contracts=0
production_readiness_overclaims=0
```

## Failure Loop

The first blocked setup path had a generic missing-capability reason. That was insufficient for a credential or permission gap because the owner needs a safe cause, impact, and resolution without seeing provider credentials, raw handles, local paths, or raw provider responses.

Fix: permission-gap Provider Packs now produce a specific blocked state and `status_explanation` with false redaction exposure flags. The verifier scans command output and state for provider internals and secrets.

## Commands Run

| Command | Result |
|---|---|
| `python3 -m compileall packages/cornerstone_cli` | PASS |
| `python3 -m unittest tests.scenario.test_connectorhub_cli` | PASS, 10 tests |
| `make verify-connector-contract-adapter` | PASS |
| `cornerstone scenario verify connector-contract-adapter --json --output reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json` | PASS; report status `success`, 40 PASS, 0 blocking |

## Proof Surface

- `proof_surface`: `local_fixture`
- `claim_boundary`: deterministic local fixture evidence only; no live-provider production or human-acceptance claim

## Senior Engineering Decision Trail

- Product value: `CS-CH-006` advances Connector Hub adoption in CornerStone by proving `Explain credential and permission gaps without secrets` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Status explains cause impact and safe resolution without exposing secrets paths handles or raw responses`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-006` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-006`, phase `CH-0`, related requirements `IR-09;IR-12;ER-05`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-006.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-006 --json`; the expected method is `Error-shape tests plus CLI status contract plus secret scan`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-006` as the independent delivery unit for `Explain credential and permission gaps without secrets`.
- Implementation approach: use `Error-shape tests plus CLI status contract plus secret scan` against matrix row `CS-CH-006`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Status explains cause impact and safe resolution without exposing secrets paths handles or raw responses` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-006.json` as the acceptance record.
- Refactor and hardening: `CS-CH-006` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter/scenarios/CS-CH-006.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-006` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `credential and permission gap explanation without secrets` into the CornerStone adoption surface `ConnectorPort setup and versioned provider-pack foundation`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Boundaries

This report claims only local fixture proof for `CS-CH-006`.

It does not claim:

- rendered UI/API credential-gap proof;
- live provider permission flow;
- live GitHub read-only readiness;
- production tenancy, RLS, OPA, egress, backup/restore, or release readiness.

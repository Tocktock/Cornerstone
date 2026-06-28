# Connector Hub CS-CH-002 Verification Report - 2026-06-23

## Result

- Scenario set: `connector-contract-adapter`
- Filtered scenario: `CS-CH-002`
- Filtered status: `success`
- Filtered PASS rows: `1`
- Filtered blocking rows: `0`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Unfiltered connector-contract-adapter status: `success`
- Unfiltered verified rows: `40 PASS, 0 blocking`
- Worktree dirty at verification: `true`

## Scenario Contract

`CS-CH-002` proves that a required connector capability cannot silently activate when no permitted Provider Pack mapping can supply it.

Expected behavior:

- setup planning returns a blocked state and non-zero CLI exit code;
- the blocked reason is stable and machine-readable;
- safe resolution guidance is present;
- no delivery stream is created;
- no provider call is made before activation;
- blocked setup is persisted with audit evidence.

## What Changed

- Added missing-required fixture `fixtures/connectorhub/contracts/github_required_missing_contract.json`.
- Added explicit blocked setup metadata in `packages/cornerstone_cli/connector.py`:
  - `activation_state=blocked_required_capability_missing`
  - `blocked_reason_code=CS_CONNECTOR_REQUIRED_CAPABILITY_MISSING`
  - `activation_blockers`
  - `activation_guidance`
  - `delivery_streams`
- Extended `connector-contract-adapter` verification to run CS-CH-002 as a real scenario row.
- Extended `tests/scenario/test_connectorhub_cli.py` to assert exit code `7`, blocked setup, stable gap reason, safe guidance, empty delivery streams, zero provider calls, audit refs, and persisted setup result.


## Scenario Verification

| ID | Type | Status | Evidence | Notes |
|---|---|---|---|---|
| CS-CH-002 | MUST_PASS | PASS | `reports/scenario/connector-contract-adapter/scenarios/CS-CH-002.json` | Activation is blocked with stable reason code guidance no stream and no provider call |

## Evidence Summary

Filtered report:

```text
reports/scenario/connector-contract-adapter/scenarios/CS-CH-002.json
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
contract_validate_exit_zero=true
setup_plan_exit_connector_unavailable=true
audit_verify_exit_zero=true
scope_complete=true
contract_persisted=true
setup_result_persisted=true
source_policy_persisted=true
setup_status_blocked=true
setup_readiness_blocked=true
activation_denied=true
activation_state_stable=true
required_capabilities_unavailable=true
blocked_reason_code_stable=true
cli_error_code_stable=true
gap_reason_code_stable=true
safe_resolution_guidance=true
no_delivery_streams=true
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

The first implementation already had a partial blocked setup path, but the warning text treated all gaps as optional and there was no explicit stream evidence. That was insufficient for `CS-CH-002` because required missing capabilities must produce stable blocked activation evidence and prove that no Delivery stream was created.

Fix: setup planning now separates required blockers from optional gaps, emits stable blocked metadata, and derives `delivery_streams` only when activation is allowed.

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

- Product value: `CS-CH-002` advances Connector Hub adoption in CornerStone by proving `Required capability missing blocks activation` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Activation is blocked with stable reason code guidance no stream and no provider call`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-002` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-002`, phase `CH-0`, related requirements `IR-03;IR-08;IR-12`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-002.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-002 --json`; the expected method is `Invalid contract fixture plus activation negative test`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-002` as the independent delivery unit for `Required capability missing blocks activation`.
- Implementation approach: use `Invalid contract fixture plus activation negative test` against matrix row `CS-CH-002`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Activation is blocked with stable reason code guidance no stream and no provider call` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-002.json` as the acceptance record.
- Refactor and hardening: `CS-CH-002` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter/scenarios/CS-CH-002.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-002` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `required capability activation blocking` into the CornerStone adoption surface `ConnectorPort setup and versioned provider-pack foundation`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Boundaries

This report claims only local fixture proof for `CS-CH-002`.

It does not claim:

- optional degraded mode;
- UI/API parity;
- live GitHub read-only readiness;
- macOS or Chrome capture readiness;
- side-effecting connector Action readiness;
- production tenancy, RLS, OPA, egress, backup/restore, or release readiness.

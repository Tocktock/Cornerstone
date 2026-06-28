# Connector Hub CS-CH-004 Verification Report - 2026-06-23

## Result

- Scenario set: `connector-contract-adapter`
- Filtered scenario: `CS-CH-004`
- Filtered status: `success`
- Filtered PASS rows: `1`
- Filtered blocking rows: `0`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Unfiltered connector-contract-adapter status: `success`
- Unfiltered verified rows: `40 PASS, 0 blocking`
- Worktree dirty at verification: `true`

## Scenario Contract

`CS-CH-004` proves that owner Source Policy confirmation creates an immutable audited snapshot and that attempted broadening cannot silently succeed.

Expected behavior:

- owner confirmation or narrowing override returns exit code `0`;
- a new Source Policy snapshot is persisted;
- the snapshot records owner confirmation, diff hashes, narrowed fields, and compatibility decision;
- `constraints_never_broadened_silently` remains true;
- attempted broadening is denied with a stable error and audit ref;
- denied broadening creates no Source Policy snapshot;
- no provider internals or secrets appear in output/state.

## What Changed

- Added native CLI command `cornerstone connector source-policy confirm`.
- Added Source Policy normalization and narrowing-only compatibility checks in `packages/cornerstone_cli/connector.py`.
- Added auditable broadening denial without policy snapshot creation.
- Extended `connector-contract-adapter` verification to run CS-CH-004 as a real scenario row.
- Extended `tests/scenario/test_connectorhub_cli.py` to assert owner override, immutable source policy persistence, diff/narrowing fields, denial of broadening, and denial audit refs.


## Scenario Verification

| ID | Type | Status | Evidence | Notes |
|---|---|---|---|---|
| CS-CH-004 | MUST_PASS | PASS | `reports/scenario/connector-contract-adapter/scenarios/CS-CH-004.json` | Immutable Source Policy snapshot is stored and audited without silent broadening |

## Evidence Summary

Filtered report:

```text
reports/scenario/connector-contract-adapter/scenarios/CS-CH-004.json
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
source_policy_confirm_exit_zero=true
broadening_denial_exit_policy_denied=true
audit_verify_exit_zero=true
confirmed_policy_persisted=true
confirmed_policy_id_changed=true
owner_confirmed=true
confirmation_kind_override=true
confirmed_by_owner=true
silent_broadening_false=true
constraints_never_broadened_silently=true
compatibility_decision_allows_narrowing=true
narrowed_fields_recorded=true
allowed_paths_narrowed=true
max_content_bytes_narrowed=true
retention_days_narrowed=true
diff_hashes_present=true
broadening_denied_stable_error=true
broadening_denial_no_policy_snapshot=true
broadening_denial_audited=true
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
source_policy_broadening_silent_successes=0
production_readiness_overclaims=0
```

## Failure Loop

Before this unit, setup planning produced a Source Policy snapshot but there was no owner confirmation or override lifecycle. That left no explicit state transition for owner-reviewed constraints and no negative proof that broader paths/resources would be rejected.

Fix: `source-policy confirm` now persists a new immutable snapshot for owner-confirmed/narrowed policies, records policy hashes and narrowed fields, and rejects broadening with a stable error plus audit event.

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

- Product value: `CS-CH-004` advances Connector Hub adoption in CornerStone by proving `Owner confirms or overrides Source Policy` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Immutable Source Policy snapshot is stored and audited without silent broadening`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-004` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-004`, phase `CH-0`, related requirements `IR-04;IR-08;IR-17`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-004.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-004 --json`; the expected method is `Policy normalization plus owner action test plus state diff`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-004` as the independent delivery unit for `Owner confirms or overrides Source Policy`.
- Implementation approach: use `Policy normalization plus owner action test plus state diff` against matrix row `CS-CH-004`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Immutable Source Policy snapshot is stored and audited without silent broadening` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-004.json` as the acceptance record.
- Refactor and hardening: `CS-CH-004` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter/scenarios/CS-CH-004.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-004` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `owner-confirmed Source Policy snapshot` into the CornerStone adoption surface `ConnectorPort setup and versioned provider-pack foundation`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Boundaries

This report claims only local fixture proof for `CS-CH-004`.

It does not claim:

- rendered owner Source Policy override UI;
- provider-swap invariance;
- live GitHub read-only readiness;
- macOS or Chrome capture readiness;
- side-effecting connector Action readiness;
- production tenancy, RLS, OPA, egress, backup/restore, or release readiness.

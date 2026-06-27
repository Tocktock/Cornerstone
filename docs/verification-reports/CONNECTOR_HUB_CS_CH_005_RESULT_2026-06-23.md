# Connector Hub CS-CH-005 Verification Report - 2026-06-23

## Result

- Scenario set: `connector-contract-adapter`
- Filtered scenario: `CS-CH-005`
- Filtered status: `success`
- Filtered PASS rows: `1`
- Filtered blocking rows: `0`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Unfiltered connector-contract-adapter status: `success`
- Unfiltered verified rows: `40 PASS, 0 blocking`
- Worktree dirty at verification: `true`

## Scenario Contract

`CS-CH-005` proves that equivalent Provider Packs can be swapped without changing Product logic.

Expected behavior:

- setup planning with the alternate Provider Pack returns `ready`;
- setup/source provider refs change;
- Source Policy identity changes with selected provider pack;
- Product handler contract hash remains unchanged;
- Product projection contract remains unchanged;
- Product preview object remains unchanged;
- no provider call is made before activation;
- no provider internals or secrets appear in output/state.

## What Changed

- Added alternate local Provider Pack `local_source_control_readonly_alt.v1`.
- Added `--provider-pack-id` to `cornerstone connector setup plan`.
- Added provider-pack identity to Source Policy snapshot identity.
- Added provider-neutral `product_handler_contract`, `product_projection_contract`, and `product_object_preview` fields to Setup Result.
- Extended `connector-contract-adapter` verification to run CS-CH-005 as a real regression row.
- Extended `tests/scenario/test_connectorhub_cli.py` to compare default and alternate setup plans.


## Scenario Verification

| ID | Type | Status | Evidence | Notes |
|---|---|---|---|---|
| CS-CH-005 | REGRESSION_GUARD | PASS | `reports/scenario/connector-contract-adapter/scenarios/CS-CH-005.json` | Provider swap changes setup/source refs only while product handlers stay unchanged |

## Evidence Summary

Filtered report:

```text
reports/scenario/connector-contract-adapter/scenarios/CS-CH-005.json
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
provider_swap_setup_exit_zero=true
audit_verify_exit_zero=true
provider_swap_setup_persisted=true
provider_swap_source_policy_persisted=true
provider_refs_changed=true
source_policy_refs_changed=true
product_handler_contract_unchanged=true
product_projection_contract_unchanged=true
product_object_preview_unchanged=true
handler_contract_hash_present=true
provider_swap_readiness_ready=true
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

The first provider-swap run failed because Source Policy identity did not include selected Provider Pack identity. That meant the default and alternate policies could share the same `source_policy_id` even though provider refs differed.

Fix: setup planning now includes `selected_provider_pack_id` in Source Policy identity and snapshot state. The regression verifier checks that provider refs and source policy refs change while Product handler/projection contracts remain unchanged.

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

- Product value: `CS-CH-005` advances Connector Hub adoption in CornerStone by proving `Swap providers without changing Product logic` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Provider swap changes setup/source refs only while product handlers stay unchanged`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-005` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-005`, phase `CH-0`, related requirements `IR-02;IR-08;IR-15`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-005.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-005 --json`; the expected method is `Provider-swap integration test`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-005` as the independent delivery unit for `Swap providers without changing Product logic`.
- Implementation approach: use `Provider-swap integration test` against matrix row `CS-CH-005`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Provider swap changes setup/source refs only while product handlers stay unchanged` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-005.json` as the acceptance record.
- Refactor and hardening: `CS-CH-005` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter/scenarios/CS-CH-005.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-005` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `provider-pack swap behind Product-invariant contracts` into the CornerStone adoption surface `ConnectorPort setup and versioned provider-pack foundation`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Boundaries

This report claims only local fixture proof for `CS-CH-005`.

It does not claim:

- live provider substitution;
- credential or permission gap UX;
- rendered UI/API provider-swap proof;
- live GitHub read-only readiness;
- production tenancy, RLS, OPA, egress, backup/restore, or release readiness.

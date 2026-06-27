# Connector Hub CS-CH-010 Verification Report - 2026-06-23

## Result

- Scenario set: `connector-contract-adapter`
- Filtered scenario: `CS-CH-010`
- Filtered status: `success`
- Filtered PASS rows: `1`
- Filtered blocking rows: `0`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Unfiltered connector-contract-adapter status: `success`
- Unfiltered verified rows: `40 PASS, 0 blocking`
- Verified base commit: `cb682e9`
- Worktree dirty at verification: `true`

## What Changed

- Added Connector-owned delivery idempotency state keyed by source external ID, Projection type, and source content hash.
- Added Connector content-version records, current-version pointer records, and predecessor Artifact/version links.
- Added native lineage read path `cornerstone connector lineage show --contract-id ... --source-external-id ... --json`.
- Added duplicate and changed-content fixtures for provider-event replay, unchanged-content replay, and one-byte changed source content.
- Extended `connector-contract-adapter` scenario verification and Make gating with filtered `CS-CH-010`.

## Scenario Verification

| ID | Type | Status | Evidence | Notes |
|---|---|---|---|---|
| CS-CH-010 | MUST_PASS | PASS | `reports/scenario/connector-contract-adapter-cs-ch-010-2026-06-23.json` | Repeated provider events and unchanged source content resolve to one logical intake record; changed source content creates a second version linked to the predecessor Artifact/version; lineage query shows one current truth and immutable historical evidence. |

## Evidence Summary

Filtered report:

```text
reports/scenario/connector-contract-adapter-cs-ch-010-2026-06-23.json
status=success
scenario_count=1
pass=1
blocking=0
product_feature_claims=LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING
lineage_state_dir=tmp/scenario/connector-contract-adapter-lineage-13165
first_dedupe_state=cdedup_77d6504108a4ac76
changed_dedupe_state=cdedup_8b57e7d4c7791b43
first_content_version=cver_44d528603180d861
changed_content_version=cver_336c8632f16d807e
content_source_key=csrc_2eccb4f7972935e3
first_artifact=art_79b05a96a3bcc811
changed_artifact=art_7a2e724f88ecb1f4
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

CS-CH-010 checks recorded in the filtered report:

```text
lineage_contract_validate_exit_zero=true
lineage_setup_plan_exit_zero=true
first_process_exit_zero=true
first_dedupe_state_canonical=true
first_content_version_created=true
duplicate_provider_event_exit_zero=true
duplicate_provider_event_one_logical_record=true
unchanged_content_event_exit_zero=true
unchanged_content_one_logical_record=true
changed_content_exit_zero=true
changed_content_new_version=true
changed_artifact_lineage_linked=true
durable_lineage_records_persisted=true
lineage_query_exit_zero=true
lineage_query_versions=true
one_current_logical_truth=true
historical_evidence_not_mutated=true
reconcile_two_logical_versions=true
audit_verify_exit_zero=true
evidence_refs_present=true
audit_refs_present=true
zero_provider_internals=true
zero_secret_findings=true
```

Negative evidence:

```text
duplicate_active_connector_truth=0
immutable_history_mutations=0
unauthorized_provider_calls=0
provider_credentials_exposed=0
raw_provider_payloads_exposed=0
production_readiness_overclaims=0
```

## Implementation Decision

The smallest complete solution keeps transport idempotency and source-content versioning explicit at the ConnectorPort boundary. The local Archive layer still preserves immutable Artifact bytes, while Connector state decides whether a Delivery represents a duplicate logical intake or a new version of the same source object.

The idempotency key uses scope, contract version, source external ID, Projection type, and source content hash. That means a repeated provider event and a later provider event with unchanged source content both resolve to the existing receipt/Artifact. A changed source-content hash creates a new content version linked to the previous content version and Artifact, then advances a separate current-version pointer without mutating the predecessor record.

## Commands Run

| Command | Result |
|---|---|
| `python3 -m compileall packages/cornerstone_cli` | PASS |
| `python3 -m unittest tests.scenario.test_connectorhub_cli.ConnectorHubCliTests.test_connector_delivery_dedupes_and_versions_changed_content_cs_ch_010` | PASS |
| `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-010 --json` | PASS; report status `success`, 1 PASS, 0 blocking |
| `cornerstone scenario verify connector-contract-adapter --json` | PASS; report status `success`, 40 PASS, 0 blocking |
| `make verify-connector-contract-adapter` | PASS; full gate status `success`, 40 PASS, 0 blocking, Connector Hub unittest suite 21 tests OK |
| `cornerstone connector report-lint --report reports/scenario/connector-contract-adapter-2026-06-23.json --json` | PASS; lint status `pass`, overclaims `0` |
| `scripts/verify_sot_docs.sh` | PASS |
| `scripts/verify_cli_native_first_docs.sh` | PASS |
| `CORNERSTONE_SKIP_VS2_REGRESSION_TESTS=1 python3 -m unittest tests.scenario.test_scaffold_cli tests.scenario.test_connectorhub_cli` | PASS; 65 tests OK, 5 skipped |
| `git diff --check` | PASS |

## Proof Surface

- `proof_surface`: `local_fixture`
- `claim_boundary`: deterministic local fixture evidence only; no live-provider production or human-acceptance claim

## Senior Engineering Decision Trail

- Product value: `CS-CH-010` advances Connector Hub adoption in CornerStone by proving `Deduplicate provider events and version changed content` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Duplicates resolve to one logical record and changed content creates version lineage`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-010` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-010`, phase `CH-1`, related requirements `IR-05;IR-07`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter-cs-ch-010-2026-06-23.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-010 --json`; the expected method is `Dedupe and changed-content fixture tests plus lineage query`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-010` as the independent delivery unit for `Deduplicate provider events and version changed content`.
- Implementation approach: use `Dedupe and changed-content fixture tests plus lineage query` against matrix row `CS-CH-010`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Duplicates resolve to one logical record and changed content creates version lineage` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter-cs-ch-010-2026-06-23.json` as the acceptance record.
- Refactor and hardening: `CS-CH-010` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter-cs-ch-010-2026-06-23.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-010` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `delivery idempotency and content version lineage` into the CornerStone adoption surface `Durable evidence archive policy audit and safety guardrails`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Boundaries

This report claims local fixture proof for `CS-CH-010`.

It does not claim:

- live browser/document-provider prompt-injection variants beyond the local CS-CH-014 fixture;
- rendered UI/API proof;
- live GitHub read-only readiness;
- production tenancy, RLS, OPA, egress, backup/restore, or release readiness.

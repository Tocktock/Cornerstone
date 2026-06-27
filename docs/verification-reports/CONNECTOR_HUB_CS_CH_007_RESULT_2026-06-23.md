# Connector Hub CS-CH-007 Verification Report - 2026-06-23

## Result

- Scenario set: `connector-contract-adapter`
- Filtered scenario: `CS-CH-007`
- Filtered status: `success`
- Filtered PASS rows: `1`
- Filtered blocking rows: `0`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Unfiltered connector-contract-adapter status: `success`
- Unfiltered verified rows: `40 PASS, 0 blocking`
- Verified base commit: `cb682e9`
- Worktree dirty at verification: `true`

## What Changed

- Added native CLI path `cornerstone connector delivery ingest --file ... --contract-id ... --json`.
- Added deterministic delivery fixture `fixtures/connectorhub/deliveries/github_issue_projection_delivery.json`.
- Extended the local ConnectorRuntime to archive a valid Projection Delivery through Artifact ingest.
- Persisted a ConnectorDeliveryReceipt, ProjectionSnapshot, ConnectorEvidenceLink, and connector provenance on the Artifact.
- Extended `connector-contract-adapter` scenario verification and Make gating with filtered `CS-CH-007`.

## Scenario Verification

| ID | Type | Status | Evidence | Notes |
|---|---|---|---|---|
| CS-CH-007 | MUST_PASS | PASS | `reports/scenario/connector-contract-adapter-cs-ch-007-2026-06-23.json` | Valid app-scoped Projection Delivery is archived as an immutable scoped Artifact with exact envelope bytes, receipt, Projection snapshot, Source Policy link, EvidenceRef metadata, and audit refs before any Product interpretation or acknowledgement. |

## Evidence Summary

Filtered report:

```text
reports/scenario/connector-contract-adapter-cs-ch-007-2026-06-23.json
status=success
scenario_count=1
pass=1
blocking=0
product_feature_claims=LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING
artifact=art_79b05a96a3bcc811
delivery_receipt=cdelrec_00a47062114cabc9
projection_snapshot=cproj_3fa6d3eaf056ed7f
evidence_link=celink_b7518cfe401b6896
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

CS-CH-007 checks recorded in the filtered report:

```text
delivery_ingest_exit_zero=true
artifact_show_exit_zero=true
scope_complete=true
delivery_receipt_persisted=true
projection_snapshot_persisted=true
evidence_link_persisted=true
artifact_record_persisted=true
original_envelope_bytes_persisted=true
artifact_checksum_matches_delivery_file=true
artifact_identity_linked=true
projection_and_delivery_linked=true
source_policy_and_setup_linked=true
evidence_ref_is_metadata_with_artifact=true
product_interpretation_after_archive_only=true
ack_not_claimed_for_cs_ch_007=true
raw_provider_payload_not_stored=true
zero_provider_calls_during_ingest=true
zero_provider_internals=true
zero_secret_findings=true
```

Negative evidence:

```text
unauthorized_provider_calls=0
provider_credentials_exposed=0
raw_provider_payloads_exposed=0
ownerless_connector_artifacts=0
projection_envelope_checksum_mismatches=0
product_interpretation_before_archive_commit=0
projection_acknowledgements_before_cs_ch_008=0
production_readiness_overclaims=0
```

## Implementation Decision

The smallest complete solution reuses `LocalRuntimeStore.ingest_artifact` for the immutable byte-preservation surface instead of adding a parallel connector-only object store. Connector-specific state is additive: the Artifact remains the original evidence object, while receipt, snapshot, evidence-link, and audit records explain how the Projection Delivery entered CornerStone.

Ack, retry, quarantine, replay dedupe, changed-content versioning, and prompt-injection authority checks remain separate future scenario units.

## Commands Run

| Command | Result |
|---|---|
| `python3 -m compileall packages/cornerstone_cli` | PASS |
| `python3 -m unittest tests.scenario.test_connectorhub_cli` | PASS, 11 tests |
| `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-007 --json` | PASS; report status `success`, 1 PASS, 0 blocking |
| `make verify-connector-contract-adapter` | PASS; full report status `success`, 40 PASS, 0 blocking |

## Proof Surface

- `proof_surface`: `local_fixture`
- `claim_boundary`: deterministic local fixture evidence only; no live-provider production or human-acceptance claim

## Senior Engineering Decision Trail

- Product value: `CS-CH-007` advances Connector Hub adoption in CornerStone by proving `Convert a Projection into an immutable Artifact` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Exact Projection envelope is preserved as scoped Artifact linked to policy evidence and audit refs`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-007` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-007`, phase `CH-1`, related requirements `IR-04;IR-05;IR-06;IR-07`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter-cs-ch-007-2026-06-23.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-007 --json`; the expected method is `Fixture delivery integration test`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-007` as the independent delivery unit for `Convert a Projection into an immutable Artifact`.
- Implementation approach: use `Fixture delivery integration test` against matrix row `CS-CH-007`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Exact Projection envelope is preserved as scoped Artifact linked to policy evidence and audit refs` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter-cs-ch-007-2026-06-23.json` as the acceptance record.
- Refactor and hardening: `CS-CH-007` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter-cs-ch-007-2026-06-23.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-007` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `Projection Delivery to immutable Artifact archive` into the CornerStone adoption surface `Durable evidence archive policy audit and safety guardrails`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Boundaries

This scenario-specific report claims local fixture proof for `CS-CH-007`. The unfiltered connector-contract-adapter report separately includes the `CS-CH-008` durable-ack proof and the `CS-CH-009` retry/quarantine proof.

It does not claim:

- retry, quarantine, replay, or dedupe correctness;
- rendered UI/API proof;
- live GitHub read-only readiness;
- production tenancy, RLS, OPA, egress, backup/restore, or release readiness.

# Connector Hub CS-CH-020 Verification Report - 2026-06-23

## Summary

- Scenario: `CS-CH-020`
- Type: `MUST_PASS`
- Status: `PASS`
- Filtered scenario report: `reports/scenario/connector-contract-adapter/scenarios/CS-CH-020.json`
- Aggregate scenario report: `reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Proof boundary: local deterministic fixture only; this does not claim live GitHub App revocation, live repository deletion, live rate-limit headers, webhook retry delivery, alert delivery, rendered UI/browser proof, or production egress/security readiness.

## Scenario Result

| ID | Type | Status | Evidence | Result |
|---|---|---|---|---|
| `CS-CH-020` | `MUST_PASS` | `PASS` | `reports/scenario/connector-contract-adapter/scenarios/CS-CH-020.json` | GitHub provider failures become stable local states: rate limits schedule bounded retry and freshness delay, revoked permissions create a permanent setup gap and suspended stream state, repository removal stops future ingestion and marks the source unavailable, transient transport failures retry safely, and existing evidence remains preserved with Search/Claim warnings. |

## Proof Surface

- `proof_surface`: `local_fixture`
- `claim_boundary`: deterministic local fixture evidence only; no live-provider production or human-acceptance claim

## Senior Engineering Decision Trail

- Product value: `CS-CH-020` advances Connector Hub adoption in CornerStone by proving `Handle GitHub rate limits revoked permissions and repository removal` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Failures become stable states with recovery guidance and freshness metadata`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-020` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-020`, phase `CH-2`, related requirements `ER-06;IR-07;IR-08;IR-17`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-020.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-020 --json`; the expected method is `Failure fixture and reconnect tests`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-020` as the independent delivery unit for `Handle GitHub rate limits revoked permissions and repository removal`.
- Implementation approach: use `Failure fixture and reconnect tests` against matrix row `CS-CH-020`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Failures become stable states with recovery guidance and freshness metadata` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-020.json` as the acceptance record.
- Refactor and hardening: `CS-CH-020` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter/scenarios/CS-CH-020.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-020` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `provider failure state and freshness handling` into the CornerStone adoption surface `Selected GitHub read-only connected source`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Out Of Scope

- Live GitHub App installation revocation or permission-change proof.
- Live GitHub repository removal, rename, transfer, or archival proof.
- Live rate-limit header handling and real scheduler retry execution.
- Alert notification delivery and rendered Mission Control, Search, Claim, or Setup UI proof.
- Automatic reconnection or repository substitution.
- Production RLS, OPA, network egress, backup/restore, or release-readiness claims.

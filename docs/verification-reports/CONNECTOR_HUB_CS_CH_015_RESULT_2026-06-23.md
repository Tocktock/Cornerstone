# Connector Hub CS-CH-015 Verification Report - 2026-06-23

## Summary

- Scenario: `CS-CH-015`
- Type: `MUST_PASS`
- Status: `PASS`
- Filtered scenario report: `reports/scenario/connector-contract-adapter/scenarios/CS-CH-015.json`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Proof boundary: local deterministic fixture only; this does not claim live GitHub App readiness.

## Scenario Result

| ID | Type | Status | Evidence | Result |
|---|---|---|---|---|
| `CS-CH-015` | `MUST_PASS` | `PASS` | `reports/scenario/connector-contract-adapter/scenarios/CS-CH-015.json` | A three-repository GitHub-style installation fixture exposes only the owner-selected repository to CornerStone Product state. The selected repository Delivery archives and acknowledges; the unselected repository Delivery is denied before Artifact, receipt, or ack creation. Direct GitHub write and silent selected-resource broadening are denied. |

## Proof Surface

- `proof_surface`: `local_fixture`
- `claim_boundary`: deterministic local fixture evidence only; no live-provider production or human-acceptance claim

## Senior Engineering Decision Trail

- Product value: `CS-CH-015` advances Connector Hub adoption in CornerStone by proving `Connect only explicitly selected GitHub repositories` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Only selected repositories are visible and ingestible with no organization-wide fallback`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-015` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-015`, phase `CH-2`, related requirements `ER-06;IR-04;IR-08;IR-09`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-015.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-015 --json`; the expected method is `Selected-resource GitHub fixture and zero-write checks`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-015` as the independent delivery unit for `Connect only explicitly selected GitHub repositories`.
- Implementation approach: use `Selected-resource GitHub fixture and zero-write checks` against matrix row `CS-CH-015`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Only selected repositories are visible and ingestible with no organization-wide fallback` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-015.json` as the acceptance record.
- Refactor and hardening: `CS-CH-015` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter/scenarios/CS-CH-015.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-015` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `selected GitHub repository Source Policy boundary` into the CornerStone adoption surface `Selected GitHub read-only connected source`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Out Of Scope

- Live GitHub App installation and permission review.
- Live provider read rehearsal and redacted external call logs.
- GitHub repository/commit/change/file Projection family coverage beyond this selected-repository boundary.
- GitHub rate limits, revoked permissions, repository removal, and live sync freshness.
- Production RLS, OPA, network egress, backup/restore, or release-readiness claims.

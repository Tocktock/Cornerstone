# Connector Hub CS-CH-017 Verification Report - 2026-06-23

## Summary

- Scenario: `CS-CH-017`
- Type: `MUST_PASS`
- Status: `PASS`
- Filtered scenario report: `reports/scenario/connector-contract-adapter/scenarios/CS-CH-017.json`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Proof boundary: local deterministic fixture only; this does not claim live GitHub webhook, HMAC signature, polling API cursor, pagination, rate-limit, or external sync readiness.

## Scenario Result

| ID | Type | Status | Evidence | Result |
|---|---|---|---|---|
| `CS-CH-017` | `MUST_PASS` | `PASS` | `reports/scenario/connector-contract-adapter/scenarios/CS-CH-017.json` | Invalid webhook metadata is denied before commit, webhook/poll overlap dedupes to one logical result, a post-commit/pre-cursor gap is detected before replay and cleared by replay, crash-after-cursor replay is safe, and out-of-order source revisions preserve one current source-control truth. |

## Proof Surface

- `proof_surface`: `local_fixture`
- `claim_boundary`: deterministic local fixture evidence only; no live-provider production or human-acceptance claim

## Senior Engineering Decision Trail

- Product value: `CS-CH-017` advances Connector Hub adoption in CornerStone by proving `Incremental GitHub sync is idempotent` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Cursor webhook and replay handling do not duplicate truth or skip changed content`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-017` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-017`, phase `CH-2`, related requirements `ER-06;IR-07`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-017.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-017 --json`; the expected method is `Incremental sync and replay tests`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-017` as the independent delivery unit for `Incremental GitHub sync is idempotent`.
- Implementation approach: use `Incremental sync and replay tests` against matrix row `CS-CH-017`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Cursor webhook and replay handling do not duplicate truth or skip changed content` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-017.json` as the acceptance record.
- Refactor and hardening: `CS-CH-017` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter/scenarios/CS-CH-017.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-017` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `incremental sync cursor and replay idempotency` into the CornerStone adoption surface `Selected GitHub read-only connected source`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Out Of Scope

- Live GitHub App webhook delivery or HMAC verification against a real secret.
- Live GitHub polling API cursors, pagination, rate limits, revoked permissions, repository removal, and external call logs.
- Large, binary, secret-bearing, or out-of-policy file restrictions, covered by `CS-CH-018`.
- GitHub write-path denial hardening, covered by `CS-CH-019`.
- Rate limits, revoked permissions, repository removal, and live freshness states, covered by `CS-CH-020`.
- Production RLS, OPA, network egress, backup/restore, or release-readiness claims.

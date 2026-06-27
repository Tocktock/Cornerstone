# Connector Hub CS-CH-021 Verification Report - 2026-06-23

## Summary

- Scenario: `CS-CH-021`
- Type: `MUST_PASS`
- Status: `PASS`
- Filtered scenario report: `reports/scenario/connector-contract-adapter/scenarios/CS-CH-021.json`
- Aggregate scenario report: `reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json`
- Product claim: `LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING`
- Proof boundary: local deterministic metadata-only fixture. This does not claim physical macOS permission prompt proof, first real activity sample proof, pause/revoke device behavior, rendered UI/browser proof, human privacy acceptance, or production egress/security readiness.

## Scenario Result

| ID | Type | Status | Evidence | Result |
|---|---|---|---|---|
| `CS-CH-021` | `MUST_PASS` | `PASS` | `reports/scenario/connector-contract-adapter/scenarios/CS-CH-021.json` | macOS WatchAgent capture stays off by default: permission probes are metadata-only, permission without consent is blocked, consent without permission is blocked, both gates create a ready state without starting capture, and negative evidence records zero samples, screenshots, hidden startup capture, cross-namespace capture, external calls, or provider mutations. |

## Proof Surface

- `proof_surface`: `local_fixture`
- `claim_boundary`: deterministic local fixture evidence only; no live-provider production or human-acceptance claim

## Senior Engineering Decision Trail

- Product value: `CS-CH-021` advances Connector Hub adoption in CornerStone by proving `macOS capture is off until explicit consent and permission` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Capture remains off until user consent and platform permission are both present`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-021` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-021`, phase `CH-3`, related requirements `ER-05;IR-14`, `proof_surface=local_fixture`, `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`, and evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-021.json` rather than informal assistant confidence.
- Reliability: replayable local fixture CLI verification and durable local state serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: deterministic fixture execution instead of live-provider calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-021 --json`; the expected method is `Permission probe and consent tests`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future scoped durable ConnectorHub records with owner, namespace, workspace, evidence, audit, and policy boundaries preserved.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-021` as the independent delivery unit for `macOS capture is off until explicit consent and permission`.
- Implementation approach: use `Permission probe and consent tests` against matrix row `CS-CH-021`, preserving `proof_surface=local_fixture` and `claim_boundary=deterministic local fixture evidence only; no live-provider production or human-acceptance claim`.
- Smallest complete solution: deliver `Capture remains off until user consent and platform permission are both present` through a deterministic local fixture path behind the native ConnectorHub CLI and scenario verifier, with the evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-021.json` as the acceptance record.
- Refactor and hardening: `CS-CH-021` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter/scenarios/CS-CH-021.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_fixture` guard, and claim-boundary guard `deterministic local fixture evidence only; no live-provider production or human-acceptance claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-021` is recorded as `PASS` only on `local_fixture` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `macOS WatchAgent consent and permission gate` into the CornerStone adoption surface `Opt-in WatchAgent and Chrome connected evidence`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Out Of Scope

- Real macOS permission prompts, Accessibility or Screen Recording state, and physical-device behavior.
- First real activity sample, pause/revoke lifecycle, retention deletion, export, or local agent startup.
- Chrome active-tab or auto-capture behavior.
- Rendered setup/privacy UI and human privacy acceptance.
- Production RLS, OPA, network egress, backup/restore, or release-readiness claims.

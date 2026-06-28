# Connector Hub CS-CH-036 Verification Report - 2026-06-23

## Summary

- Scenario: `CS-CH-036`
- Type: `MUST_PASS`
- Status: `PASS`
- Proof surface: current reusable VS2 local security proof scoped to the CS-CH-036 egress rows, Docker internal-network topology, controlled provider sink, governed egress proxy, default-deny egress proof, local range report, filtered scenario gate, aggregate scenario gate, and focused CLI regression test.
- Filtered report: `reports/scenario/connector-contract-adapter/scenarios/CS-CH-036.json`
- Aggregate report: `reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json`
- Last evidence refresh: 2026-06-25

## Scenario Result

| Scenario | Type | Status | Evidence |
|---|---|---|---|
| `CS-CH-036` | `MUST_PASS` | `PASS` | ConnectorHub adopts the current VS2 local default-deny egress proof as a guarded dependency. API, worker, and tool-runtime containers cannot reach the controlled provider directly by HTTP or socket; the provider receives zero direct-attempt requests; the governed egress proxy reaches the provider through the approved path; redirect, DNS rebinding, reserved destination, protocol normalization, sandbox/proxy/subprocess, untrusted-content, fail-closed, audit-correlation, and no-secret checks pass. |

Important boundary: the full `cornerstone security vs2-local-proof --json` and full `vs2-policy-tenancy-egress` reuse verifier now exit `0` for the local AI-verifiable VS2 surface. They still contain seven `HUMAN_REQUIRED` rows, so CS-CH-036 does not promote production network readiness, live-provider readiness, independent security review, or human UX acceptance. CS-CH-036 consumes the current reusable proof only for required egress rows `VS2-SEC-051`, `VS2-SEC-052`, `VS2-SEC-057`, `VS2-SEC-058`, `VS2-SEC-059`, `VS2-SEC-063`, and `VS2-SEC-064`, all of which are `PASS`.

## Decision Trail

- Product value: ConnectorHub can be adopted without giving Product, agents, workers, or tool runtime arbitrary outbound provider access.
- Domain correctness: provider/network access is a governed ConnectorHub capability path with declared provider, capability, action context, policy decision, evidence refs, and audit refs.
- Architecture: the Connector Hub verifier consumes the current reusable VS2 local proof instead of adding a shallow application-level mock counter.
- Data contracts: the scenario report exposes proof paths, source-fingerprint digests, required VS2 scenario ids, Docker topology, provider request counts, egress checks, and negative counters.
- Reliability: proof freshness is enforced through `source_fingerprint` validation; stale VS2 reports are rejected instead of treated as reusable.
- Security: direct HTTP/socket attempts are denied, provider request counts stay zero after direct attempts, bypass classes are guarded, secrets and raw payloads are absent from audit evidence, and production topology is not overclaimed.
- Observability: filtered and aggregate reports expose `egress_topology_checks`, `egress_topology_negative_evidence`, network-boundary checks, provider counts, and reusable-proof diagnostics.
- Performance: the local proof reuses the VS2 verifier artifacts after regeneration; it avoids per-scenario live-provider calls and keeps final PASS deterministic.
- Testability: CS-CH-036 is covered by filtered ConnectorHub verification, scenario gate, aggregate verification, aggregate gate, a focused CLI regression test, and the broader scenario-list regression.
- Maintainability: ConnectorHub references the VS2 proof as a dependency boundary, so future stronger egress proofs can replace the underlying VS2 evidence without changing the ConnectorHub scenario contract.
- Migration feasibility: the local Docker proof maps to future production firewall/proxy/service-mesh and operator-review evidence; those production surfaces remain outside this local PASS.

## Verification Evidence

Commands:

```bash
python3 -m py_compile packages/cornerstone_cli/scenarios.py tests/scenario/test_connectorhub_cli.py
cornerstone security vs2-local-proof --json
cornerstone scenario verify vs2-policy-tenancy-egress --reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json --output reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json
python3 -m unittest tests.scenario.test_connectorhub_cli.ConnectorHubCliTests.test_connectorhub_default_deny_egress_topology_cs_ch_036
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-036.json
cornerstone scenario gate reports/scenario/connector-contract-adapter/scenarios/CS-CH-036.json --json
python3 -m unittest tests.scenario.test_connectorhub_cli.ConnectorHubCliTests.test_connectorhub_scenario_list_and_filtered_verify
cornerstone scenario verify connector-contract-adapter --json --output reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json
cornerstone scenario gate reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json --json
```

Filtered report facts observed from `reports/scenario/connector-contract-adapter/scenarios/CS-CH-036.json`:

```text
status=success
scenario_count=1
pass=1
blocking=0
fail=0
not_verified=0
human_required=0
product_feature_claims=LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING
```

Required VS2 rows:

```text
VS2-SEC-051=PASS
VS2-SEC-052=PASS
VS2-SEC-057=PASS
VS2-SEC-058=PASS
VS2-SEC-059=PASS
VS2-SEC-063=PASS
VS2-SEC-064=PASS
```

Full VS2 proof boundary:

```text
reports/security/vs2-local-security-proof.json status=success
summary.pass=86
summary.fail=0
summary.blocking=0
summary.human_required=7
proof_hash=02b7baa735174dd3c929d45f699dae32db28ef3d5ecfdb37198890b169a73c45
CS-CH-036 scoped reuse_errors=0
```

Key `egress_topology_checks`:

```text
vs2_local_proof_reusable_current=True
vs2_required_egress_rows_pass=True
vs2_scenario_report_rows_pass=True
egress_proof_status_passed=True
local_range_status_passed=True
app_worker_tool_direct_http_socket_blocked=True
provider_zero_requests_after_direct_attempts=True
governed_proxy_reaches_provider=True
network_membership_isolated=True
default_denied_before_sink_call=True
redirect_dns_protocol_and_socket_bypass_guarded=True
proxy_subprocess_and_untrusted_bypass_guarded=True
egress_audit_and_policy_logs_correlate=True
secrets_and_payloads_not_exposed=True
fail_closed_without_fallback=True
production_topology_not_claimed=True
```

Network-boundary facts:

```text
service_members=api,worker,tool_runtime,egress_proxy
provider_members=egress_proxy,provider
host_network=False
privileged=False
published_ports=False
provider_requests_before_direct_attempts=0
provider_requests_after_direct_attempts=0
provider_requests_after_governed_proxy=1
total_sink_request_count=1
default_denied_sink_calls=0
redirect_denied_hop_trap_calls=0
```

Negative evidence:

```text
egress_topology_vs2_reuse_errors=0
egress_topology_missing_required_vs2_rows=0
egress_topology_failed_required_vs2_rows=0
egress_topology_direct_http_socket_bypass_allowed=0
egress_topology_provider_requests_after_direct_attempts=0
egress_topology_default_denied_sink_calls=0
egress_topology_redirect_denied_hop_trap_calls=0
egress_topology_sensitive_headers_forwarded_to_denied_hop=0
egress_topology_raw_credentials_exposed=0
egress_topology_raw_payloads_in_audit=0
egress_topology_production_topology_overclaimed=0
```

## Completion Notes

- Added CS-CH-036 ConnectorHub verification by requiring current reusable VS2 local proof, required VS2 egress rows, local Docker network-boundary checks, egress proof checks, and negative counters.
- Added `CS-CH-036` to the Connector Hub application matrix, contract proof surface, independent proof commands, Make target, aggregate report, and focused regression coverage.
- Regenerated VS2 proof before claiming ConnectorHub PASS so stale source fingerprints and evidence hashes cannot satisfy the scenario.
- Local proof does not claim production network gateway/firewall/service-mesh enforcement, live providers, production traffic, independent security review, penetration testing, or human operator approval.

## Proof Surface

- `proof_surface`: `local_vs2_topology`
- `claim_boundary`: current reusable local VS2 topology evidence only; no production network readiness claim

## Senior Engineering Decision Trail

- Product value: `CS-CH-036` advances Connector Hub adoption in CornerStone by proving `Enforce default-deny egress around ConnectorHub and tools` as a user-visible connected-source capability inside one CornerStone product, not as a separate ConnectorHub surface.
- Domain correctness: the accepted outcome is `Unauthorized egress attempts are blocked and logged under current VS2 local topology; production network topology NOT_VERIFIED`; anything outside that observable behavior remains outside this scenario's PASS claim.
- Architecture: implementation stays behind native `cornerstone connector ...` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036` paths, preserving Product / Archive / Connector / Policy / Evidence / Audit boundaries.
- Data contracts: the result is bound to matrix row `CS-CH-036`, phase `CH-1`, related requirements `IR-10;IR-18`, `proof_surface=local_vs2_topology`, `claim_boundary=current reusable local VS2 topology evidence only; no production network readiness claim`, and evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-036.json` rather than informal assistant confidence.
- Reliability: the current reusable VS2 local topology proof and its source-fingerprint guard serve as the acceptance surface for this independent delivery unit.
- Security: provider credentials, raw provider payloads, unauthorized provider calls, live-provider readiness, human-acceptance, and production-readiness claims remain excluded unless explicitly evidenced elsewhere.
- Observability: evidence refs, audit refs, negative counters, filtered scenario reports, and the aggregate connector scenario report are the trace surfaces for review.
- Performance: reusing current VS2 topology artifacts instead of making per-scenario live network calls keeps the scenario proof bounded and repeatable; this local PASS makes no production latency, throughput, or scalability claim.
- Testability: the focused verifier command is `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036 --json`; the expected method is `Current reusable VS2 local proof with Docker internal-network boundary controlled provider sink governed proxy redirect DNS sandbox audit and no-secret checks`.
- Maintainability: the scenario remains an independent delivery unit in the matrix, report, and verifier so future changes can rerun or update it without depending on another scenario's prose report.
- Migration feasibility: the local proof maps to future production firewall, proxy, service-mesh, and operator-review evidence while keeping those production surfaces outside the local PASS claim.

## Scenario Lifecycle Trail

- Research perspectives: senior product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration reviewers converged on `CS-CH-036` as the independent delivery unit for `Enforce default-deny egress around ConnectorHub and tools`.
- Implementation approach: use `Current reusable VS2 local proof with Docker internal-network boundary controlled provider sink governed proxy redirect DNS sandbox audit and no-secret checks` against matrix row `CS-CH-036`, preserving `proof_surface=local_vs2_topology` and `claim_boundary=current reusable local VS2 topology evidence only; no production network readiness claim`.
- Smallest complete solution: deliver `Unauthorized egress attempts are blocked and logged under current VS2 local topology; production network topology NOT_VERIFIED` through the reusable local VS2 topology proof plus ConnectorHub scenario gating for the egress-specific row, with the evidence artifact `reports/scenario/connector-contract-adapter/scenarios/CS-CH-036.json` as the acceptance record.
- Refactor and hardening: `CS-CH-036` was folded into the matrix, focused report `reports/scenario/connector-contract-adapter/scenarios/CS-CH-036.json`, result document, aggregate report, stale-metadata guard, `proof_surface=local_vs2_topology` guard, and claim-boundary guard `current reusable local VS2 topology evidence only; no production network readiness claim` so this independent delivery unit cannot depend on ad hoc prose or a broader ConnectorHub claim.
- Verification result: `CS-CH-036` is recorded as `PASS` only on `local_vs2_topology` evidence; live-provider, human-acceptance, and production claims remain outside this result unless the claim boundary explicitly allows them.
- Documented result: this report records the scenario outcome, evidence path, proof surface, decision trail, lifecycle trail, and out-of-scope boundary before the next scenario is treated as complete.
- ConnectorHub adoption contribution: it turns `default-deny egress around ConnectorHub and tools` into the CornerStone adoption surface `Durable evidence archive policy audit and safety guardrails`, keeping provider internals behind ConnectorPort/evidence/audit/policy boundaries and preserving the local proof boundary.

## Out Of Scope

- Live provider readiness, external account permission review, and live provider call ledgers.
- Physical-device macOS behavior, real Chrome browser privacy acceptance, and human UX/trust acceptance.
- Production PostgreSQL/RLS, OPA, network egress, backup/restore, audit-integrity, or release-readiness claims.
- Side-effecting live external mutations except through separately approved human-required gates.

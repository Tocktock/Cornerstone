# VS3 Egress and Sandbox Checkpoint - 2026-06-29

**Status:** VS3-4 local egress/sandbox slice PASS.
**Scope:** `VS3-EGR-001` through `VS3-EGR-006` only.
**Proof boundary:** Local process-level runtime guard, localhost controlled sinks, URL/redirect/DNS normalization fixtures, sandbox negative suite, component-outage fail-closed fixtures, and untrusted-content authority-denial fixtures.

This checkpoint does not claim VS3-P, production/on-prem readiness, real firewall/proxy/service-mesh enforcement, real IdP readiness, live-provider readiness, migration/restore readiness, independent security review, or human security acceptance.

## Slice Contract

Goal:

- Verify default-deny egress and sandbox behavior through native `cornerstone ... --json` commands.
- Capture negative evidence for zero forbidden sink contact, zero sandbox bypass, zero fallback direct connection, zero untrusted-content authority, and zero production/readiness overclaim.
- Keep real network and human security review as `HUMAN_REQUIRED`.

Selected scenarios:

| Scenario | Status in this checkpoint | Required proof surface |
|---|---|---|
| `VS3-EGR-001` | PASS | No-grant runtime/network denial; forbidden sink receives zero requests and bytes. |
| `VS3-EGR-002` | PASS | Governed ConnectorHub-like allow path reaches allowed sink exactly once. |
| `VS3-EGR-003` | PASS | URL normalization, redirect, fake-DNS, IPv4/IPv6, and reserved-address variants deny before contact. |
| `VS3-EGR-004` | PASS | Sandbox denies direct socket, proxy env, alternate DNS, protocols, subprocess, shell, filesystem, and env access. |
| `VS3-EGR-005` | PASS | Egress/proxy/sandbox/runtime component outage degrades readiness and fails closed. |
| `VS3-EGR-006` | PASS | Untrusted artifact, connector payload, web page, and tool output cannot create egress/action/policy/tool authority. |

Full VS3 mapping remains the frozen 57-row inventory: 42 `MUST_PASS`, 8 `REGRESSION`, and 7 `HUMAN_REQUIRED`. Non-egress rows are outside this checkpoint except where the aggregate scenario report is cited as supporting local scenario-gate context.

## Implementation Delta

- `tests/scenario/test_scaffold_cli.py` now asserts the VS3 egress/sandbox native CLI proof fields instead of only checking command success.
- New assertions cover:
  - local-only proof boundary and `real_network=HUMAN_REQUIRED`;
  - runtime denial before socket open, not application skip-only behavior;
  - forbidden sink zero requests/bytes;
  - allowed sink exactly one request;
  - redirect hop denied without contacting the forbidden sink;
  - URL-normalization cases denied without contact;
  - sandbox matrix denies with zero host operations;
  - outage matrix fails closed with degraded readiness and no fallback direct connection;
  - untrusted-content matrix creates zero egress, approvals, policy changes, and tool executions.

## Command Evidence

Focused compile:

```text
python3 -m compileall tests/scenario/test_scaffold_cli.py
Compiling 'tests/scenario/test_scaffold_cli.py'...
exit=0
```

Focused tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_egress_sandbox_proof_is_local_and_negative_evidence_backed \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_egress_and_sandbox_cli_paths_are_native \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_closes_local_dev_ai_rows

Ran 3 tests in 30.885s
OK
```

Direct CLI probes:

```text
cornerstone security vs3-egress-sandbox \
  --reuse-vs2-local-range-report reports/security/vs2-local-range.json \
  --json
exit=0
status=success
schema_version=cs.vs3_egress_sandbox_proof.v0
proof_boundary.surface=local_process_controlled_sink_and_sandbox_fixture
proof_boundary.real_network=HUMAN_REQUIRED
proof_boundary.vs3_p=NOT_CLAIMED
evidence_refs=2
audit_refs=21
policy_decision_refs=24
```

```text
cornerstone egress test --profile vs3 --json
exit=0
status=success
schema_version=cs.cli.v0
egress_test_schema_version=cs.vs3_egress_test.v0
vs3_egress_sandbox_schema_version=cs.vs3_egress_sandbox_proof.v0
proof_boundary.real_network=HUMAN_REQUIRED
proof_boundary.vs3_p=NOT_CLAIMED
forbidden_sink.requests=0
forbidden_sink.bytes=0
allowed_sink.requests=1
redirect_denied_hop_contacted=false
evidence_refs=2
audit_refs=21
policy_decision_refs=24
```

```text
cornerstone sandbox verify --json
exit=0
status=success
schema_version=cs.cli.v0
sandbox_verify_schema_version=cs.vs3_sandbox_verify.v0
vs3_egress_sandbox_schema_version=cs.vs3_egress_sandbox_proof.v0
proof_boundary.real_network=HUMAN_REQUIRED
proof_boundary.vs3_p=NOT_CLAIMED
sandbox_matrix=10
outage_matrix=4
untrusted_content_matrix=4
evidence_refs=2
audit_refs=21
policy_decision_refs=24
```

Egress/sandbox proof report:

```text
reports/security/vs3-egress-sandbox-proof.json
status=success
schema_version=cs.vs3_egress_sandbox_proof.v0
scenario_status:
  VS3-EGR-001=PASS
  VS3-EGR-002=PASS
  VS3-EGR-003=PASS
  VS3-EGR-004=PASS
  VS3-EGR-005=PASS
  VS3-EGR-006=PASS
checks:
  vs3_egr_001_no_grant_denied_by_runtime_boundary=true
  vs3_egr_002_one_approved_connectorhub_call=true
  vs3_egr_003_redirect_dns_url_bypass_denied=true
  vs3_egr_004_sandbox_denies_undeclared_access=true
  vs3_egr_005_outage_fail_closed=true
  vs3_egr_006_untrusted_content_no_authority=true
  no_overclaim_or_secret_leak=true
```

Negative evidence:

```text
forbidden_sink_requests=0
forbidden_sink_bytes=0
duplicate_allowed_sink_requests=0
denied_address_contact_count=0
direct_socket_successes=0
fallback_direct_connections=0
untrusted_content_egress_calls=0
untrusted_content_action_approvals=0
untrusted_content_policy_changes=0
untrusted_content_tool_executions=0
raw_secret_leaks=0
production_network_claimed=0
vs3_p_claimed=0
```

Aggregate scenario verification:

```text
cornerstone scenario verify vs3-onprem-trusted-extension --json \
  --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit=0
schema_version=cs.vs3_onprem_trusted_extension.v0
scenario_result_count=57
status_counts: PASS=50, HUMAN_REQUIRED=7
type_counts: MUST_PASS=42, REGRESSION=8, HUMAN_REQUIRED=7
claim_boundaries.vs3_p=NOT_CLAIMED
claim_boundaries.production_onprem=NOT_CLAIMED
claim_boundaries.security_acceptance=NOT_CLAIMED
```

Aggregate scenario gate:

```text
cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit=0
status=success
schema_version=cs.cli.v0
errors=[]
warnings=[]
claim_boundaries.vs3_p=NOT_CLAIMED
```

## Scenario Evidence Mapping

| Scenario | Evidence refs |
|---|---|
| `VS3-EGR-001` | `reports/security/vs3-egress-sandbox-proof.json`, `cornerstone security vs3-egress-sandbox --json`, `controlled_sink:forbidden`, `reports/security/vs2-local-range.json` |
| `VS3-EGR-002` | `reports/security/vs3-egress-sandbox-proof.json`, `cornerstone egress test --profile vs3 --json`, `controlled_sink:allowed`, `reports/security/vs2-local-range.json` |
| `VS3-EGR-003` | `reports/security/vs3-egress-sandbox-proof.json`, `cornerstone egress test --profile vs3 --json`, `url_normalization_matrix`, `controlled_sink:redirector`, `reports/security/vs2-local-range.json` |
| `VS3-EGR-004` | `reports/security/vs3-egress-sandbox-proof.json`, `cornerstone sandbox verify --json`, `sandbox_matrix`, `reports/security/vs2-local-range.json` |
| `VS3-EGR-005` | `reports/security/vs3-egress-sandbox-proof.json`, `cornerstone sandbox verify --json`, `outage_matrix`, `reports/security/vs2-local-range.json` |
| `VS3-EGR-006` | `reports/security/vs3-egress-sandbox-proof.json`, `cornerstone egress test --profile vs3 --json`, `untrusted_content_matrix`, `reports/security/vs2-local-range.json` |

## Human Required

Still `HUMAN_REQUIRED` and not converted to PASS:

- Real firewall, proxy, service mesh, DNS, and on-prem network validation.
- Human network/security operator review.
- Independent security review.
- Human security acceptance.
- VS3-P release approval.

## Deliberately Not Done

- No real firewall/proxy/service-mesh or production/on-prem network was exercised.
- No live provider was called.
- No independent penetration test or security acceptance was claimed.
- No external credentials were used.

## Decision

`VS3-EGR-001` through `VS3-EGR-006` are locally AI-verifiable and PASS for the VS3-4 local/dev proof surface. Continue to the next slice only with the same proof-boundary separation.

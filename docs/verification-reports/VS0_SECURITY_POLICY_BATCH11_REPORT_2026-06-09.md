# VS-0 Security Policy Batch 11 Report - 2026-06-09

Status: PASS for the first default-deny egress and sandbox policy slice only.
Scope: `CS-SEC-002` and `CS-SEC-003`.

This report does not mark RBAC/ABAC, policy-denial UI/API examples, production sandboxing, connector capability grants, or real external action execution as complete.

## Frozen Goal

Make the frozen scenario set pass through evidence-backed batches without adding unrelated features. This batch verifies only deterministic policy-denial checks: egress is denied by default without making a network call, and undeclared shell/filesystem/environment/host access is denied without executing the requested host operation.

## Assumptions

- `cornerstone egress test` and `cornerstone sandbox test` are local policy-evaluation commands, not real network or host access commands.
- Exit code `8` represents a policy/unsafe-operation denial for these checks.
- Policy decisions and audit events are sufficient scaffold evidence for the frozen denial-test rows.

## Out Of Scope

- `CS-SEC-001`: first-use upload/search/brief quickstart remains `NOT_VERIFIED` because brief generation is incomplete.
- `CS-SEC-004`: RBAC/ABAC access-control matrix remains `NOT_VERIFIED`.
- `CS-SEC-005`: policy-denial UI/API examples remain `NOT_VERIFIED`.
- Production sandbox isolation, connector capability grants, real network egress, and external action execution.

## Checklist

- [x] Frozen `CS-SEC-002` and `CS-SEC-003` wording inspected.
- [x] CLI-native egress denial command added.
- [x] CLI-native sandbox denial command added.
- [x] Policy decisions include reason and resolution path.
- [x] Audit events are recorded for policy denials.
- [x] Matrix PASS rows backed by a JSON report artifact.
- [x] Unit test added for report shape and negative evidence.
- [x] Audit verifier updated for policy denial events.
- [x] No destructive action, external call, secret access, tenant/security mutation, new dependency, or production state change.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| CS-SEC-002 | MUST_PASS | PASS | `reports/scenario/vs0-security-policy-2026-06-09.json`, egress denial policy/audit transcript |
| CS-SEC-003 | MUST_PASS | PASS | `reports/scenario/vs0-security-policy-2026-06-09.json`, shell/filesystem/environment/host sandbox denial policy/audit transcripts |

## Human Required

No human-required item was introduced for this batch. Production sandbox and connector policy review remains outside the current scaffold scope.

## Command Evidence

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-security-policy --json --output reports/scenario/vs0-security-policy-2026-06-09.json
# status: success
# scenario_set: vs0-security-policy
# summary.pass: 2
# summary.blocking: 0
# summary.product_feature_claims: PARTIAL_VS0_SECURITY_POLICY_ONLY
# security_policy_evidence.egress_exit_code: 8
# security_policy_evidence.egress_external_http_calls: 0
# security_policy_evidence.egress_policy: default_egress_deny
# security_policy_evidence.sandbox_cases: sandbox_environment, sandbox_filesystem, sandbox_host, sandbox_shell
# negative_evidence.external_http_calls: 0
# negative_evidence.egress_allowed: 0
# negative_evidence.host_operations_executed: 0
# negative_evidence.shell_commands_executed: 0
# negative_evidence.filesystem_reads: 0
# negative_evidence.environment_reads: 0
# negative_evidence.sandbox_access_allowed: 0
```

## Evidence Summary

- `cornerstone egress test --url https://example.invalid/blocked --json` exits `8` with `CS_EGRESS_DENIED`.
- The egress decision policy is `default_egress_deny`, includes a resolution path, records an audit event, and reports `external_http_calls=0`.
- `cornerstone sandbox test` denies undeclared `shell`, `filesystem`, `environment`, and `host` capabilities.
- Each sandbox denial exits `8` with `CS_SANDBOX_ACCESS_DENIED`, includes a resolution path, records an audit event, and reports zero host/shell/filesystem/environment operations.
- Audit verification succeeds for the generated policy-denial state.

## Matrix Update

`docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv` marks only `CS-SEC-002` and `CS-SEC-003` as `PASS` in this batch.

## Gaps

- `CS-SEC-004` remains `NOT_VERIFIED`; RBAC/ABAC access-control matrix is not implemented.
- `CS-SEC-005` remains `NOT_VERIFIED`; UI/API denial examples are not implemented.
- Production sandboxing and connector capability enforcement remain future hardening work.

## Risks

- These are deterministic policy-denial tests, not proof of OS-level sandbox isolation.
- Future tool execution must route through this policy surface or an equivalent stricter policy/audit path before any host or egress capability can be allowed.

# CornerStone VS-0 Tenant/Security Boundary Preflight Report

Summary:
- Verdict: needs explicit approval before runtime or scenario-verifier code changes.
- Scope: remaining VS-0 namespace promotion, RBAC/ABAC access-control, and personal-to-organization leak-prevention scenarios.
- Date: 2026-06-09
- Owner: JiYong / Tars
- Baseline commit: `ac8bb09`

## Goal Freeze

Make CornerStone pass the frozen scenario suite, not add unrelated features.

The immediate remaining VS-0 scope is limited to these currently `NOT_VERIFIED` AI-owned rows:

- `CS-NS-004` - explicit promotion with provenance.
- `CS-SEC-004` - RBAC/ABAC enforcement.
- `CS-REG-006` - personal context must not leak into organization context.

The full long-term objective remains all 206 scenarios in `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`.
This report does not narrow that objective; it freezes the next approval-gated batch only.

## Assumptions

- Current scaffold state is the source of truth until changed and verified.
- The remaining VS-0 rows require tenant/security-boundary semantics.
- Tenant/security semantics require explicit human approval before code changes.
- The next implementation should stay local and deterministic:
  no production tenant mutation, no live auth provider, no external network call, no secret access, no new production dependency, and no real connector execution.
- Native `cornerstone ... --json` command paths must exist before any product scenario can be marked `PASS`.
- Deterministic validators, audit records, policy decision records, evidence refs, and CLI transcripts decide PASS; local LLM output may be smoke evidence only, not the judge.
- Sub-agents were not used for this preflight because the available delegation tool requires explicit user authorization for sub-agent work.

## Out Of Scope

- Broad authorization architecture migration.
- Production RBAC, ABAC, ReBAC, OIDC, SSO, IAM, or tenant-management integration.
- Adding Cedar, OPA, OpenFGA, SpiceDB, or other runtime dependencies in this batch.
- External provider calls, real connector writes, real secrets, real PII, or live organization data.
- UI/API feature expansion beyond CLI-native local verification needs.
- Marking the remaining rows `PASS` without concrete scenario reports and matrix updates.

## Scenario Verification

| ID | Type | Expected Result | Verification Method | Evidence | Status |
|---|---|---|---|---|---|
| `CS-NS-004` | MUST_PASS | Promoted item receives organization ownership/scope, keeps provenance to the original personal item if allowed, carries evidence, and records audit. | `cornerstone scenario verify vs0-tenant-security-boundary --json` after approval and implementation. | Required: promotion transcript, source/target records, provenance refs, evidence refs, audit refs. | NOT_VERIFIED |
| `CS-SEC-004` | MUST_PASS | Access is enforced by role, attributes, namespace, classification, mission authority, and policy. | `cornerstone scenario verify vs0-tenant-security-boundary --json` after approval and implementation. | Required: access-control matrix, allow/deny policy decisions, audit refs, zero unauthorized reads/actions. | NOT_VERIFIED |
| `CS-REG-006` | REGRESSION_GUARD | Personal memory is not used in organization answers/actions without explicit promotion or permission. | `cornerstone scenario verify vs0-tenant-security-boundary --json` after approval and implementation. | Required: before-promotion org answer denial/insufficient evidence, after-promotion provenance-bounded answer, audit refs. | NOT_VERIFIED |

## CLI Parity Summary

| Feature / Scenario | CLI Command(s) | JSON Schema | Exit-Code Tests | Evidence/Audit Refs | Same Backend Path | Status |
|---|---|---|---|---|---|---|
| Namespace promotion | Proposed: `cornerstone namespace promote --source-kind memory --source-id <id> --target-owner-id local-org --target-namespace-id organization --target-workspace-id ops --mode copy_with_provenance --json` | Proposed: `cs.namespace_promotion.v0` | Required: success, not-found, scope-denied, evidence-required, policy-denied. | Required in payload and report. | Must use `LocalRuntimeStore`, scoped records, evidence, and audit. | NOT_VERIFIED |
| Access evaluation | Proposed: `cornerstone access evaluate --principal-role <role> --action <action> --resource-kind <kind> --resource-id <id> --classification <level> --mission-authority <state> --json` | Proposed: `cs.access_decision.v0` | Required: allow and deny cases with stable exit codes. | Required in payload and report. | Must use the same local policy decision record shape as action/egress/sandbox policy. | NOT_VERIFIED |
| Cross-namespace memory boundary | Proposed: `cornerstone memory answer --question <q> --json` or verifier-only command if product command is not approved. | Proposed: `cs.memory_answer.v0` | Required: personal-to-org leak denied before promotion; bounded use after explicit promotion. | Required in payload and report. | Must read only active scope plus explicitly promoted/referenced records. | NOT_VERIFIED |

## Research Basis

The safest best-fit option for the current scaffold is a small deterministic local policy evaluator, not a new policy-engine dependency.

- NIST SP 800-162 defines ABAC around subject, object, operation, environment, and policy attributes. The local matrix should explicitly include subject role/attributes, resource namespace/classification, action, mission authority, and policy.
- AWS Cedar and Amazon Verified Permissions are a strong future direction because Cedar supports RBAC/ABAC, local policy testing, auditability, and Apache-2.0 open source libraries, but adding it now would be a new production dependency and needs approval.
- Zanzibar, OpenFGA, and SpiceDB show the dominant fine-grained authorization pattern for relationship and namespace boundaries. They are useful references for future ReBAC/tuple storage, but not necessary for the local VS-0 deterministic proof.
- OPA decision logs show the right audit shape: every policy decision should have a traceable decision id, input/result context, and sensitive-data masking.
- W3C PROV-O is the right provenance vocabulary reference: promotion records should preserve source entity, target entity, activity, agent/owner, and evidence lineage.

## Implementation Checklist After Approval

- Add a local promotion record type with:
  source item ref, target item ref, mode, source scope, target scope, provenance, evidence refs, audit ref, policy decision ref, and created timestamp.
- Add a `namespace promote` CLI path that refuses implicit cross-namespace access and only creates a target organization record through explicit promotion.
- Add a deterministic access decision evaluator with:
  role, subject attributes, resource namespace, classification, action, mission authority, policy id, decision, reason, resolution path, and audit event.
- Add a cross-namespace memory answer/verifier path that:
  excludes personal memory in org scope before promotion and permits only the promoted/ref-copied record after explicit provenance is present.
- Add a scenario verifier for all three rows, likely `vs0-tenant-security-boundary`.
- Update `README.md`, `scripts/verify_scaffold_cli.sh`, scenario tests, JSON scenario reports, and `SCENARIO_VERIFICATION_MATRIX.csv` only after PASS evidence exists.
- Commit immediately after the batch has PASS evidence.

## Failure Reverse Engineering

| Scenario | Expected | Actual / Missing Evidence | First Failing Layer | Root Cause | Fix or Blocker | Re-verification Plan |
|---|---|---|---|---|---|---|
| `CS-NS-004` | Explicit promotion with org ownership/scope, provenance, evidence, and audit. | No current explicit promotion command or promotion report exists. | CLI/runtime surface. | Existing namespace work verifies isolation, not governed promotion. | Approval required before tenant/security code change. | Run promotion command and verifier after approved implementation. |
| `CS-SEC-004` | Matrix tests for role, attributes, namespace, classification, mission authority, and policy. | Existing policy tests cover egress, sandbox, action mode, and mission action policy, not full RBAC/ABAC matrix. | Policy evaluator/verifier surface. | No deterministic access matrix command/report yet. | Approval required before security semantics change. | Run access matrix verifier with allow/deny cases and audit refs. |
| `CS-REG-006` | Personal memory not used in organization context without explicit promotion or permission. | Existing namespace search isolation and memory truth-boundary tests do not prove org answer/action behavior across memory promotion. | Memory answer/verifier surface. | Missing before/after promotion leak test. | Approval required before tenant/security code change. | Run org answer/action leak test before and after explicit promotion. |

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| H-TSB-001 | Tenant/security semantics are an explicit stop-and-ask gate. | Approve local deterministic tenant/security implementation for `CS-NS-004`, `CS-SEC-004`, and `CS-REG-006`. | Written approval in the thread. | Blocks the remaining VS-0 scenario PASS work. |
| H-TSB-002 | Future production-grade auth engine selection affects dependency, operations, and security posture. | Decide later whether to adopt Cedar, OPA, OpenFGA, SpiceDB, or a custom policy service for non-scaffold runtime. | ADR or product architecture decision. | Not required for local VS-0 scaffold PASS if no new dependency is added. |

## Current Evidence Snapshot

- `git status --porcelain=v1`: clean before this report.
- `git log -1 --oneline`: `ac8bb09 Add VS-0 memory truth boundary verification`.
- Current matrix: full suite `62 PASS / 144 NOT_VERIFIED`; VS-0 `55 PASS / 3 NOT_VERIFIED`.
- Remaining VS-0 rows: `CS-NS-004`, `CS-SEC-004`, `CS-REG-006`.
- Most recent full fast gate observed before this report:
  `make verify-local-fast` passed docs, matrix, scaffold CLI, and 29 tests.
- Local Ollama availability observed before this report:
  installed models include `nemotron3:33b`, `qwen3-embedding:0.6b`, and `qwen3.6:27b`.

## Risks

- Implementing tenant/security behavior without approval would violate the project gate.
- Using a narrow mock and then marking broad production auth behavior PASS would overclaim.
- Adding a policy-engine dependency now would increase supply-chain and maintenance risk without being necessary for VS-0 local proof.
- Promotion must not become implicit sharing; the verifier must prove zero unauthorized personal-memory use before explicit promotion.
- Policy/audit logs must avoid secrets or sensitive input leakage.

## Verdict

- AI-verifiable scope: needs-follow-up.
- Human/release gate: approval required before runtime/scenario-verifier changes.
- Scenario PASS claim: not made by this report.

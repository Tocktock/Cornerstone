# VS-0 Tenant Security Boundary Batch 20 Report - 2026-06-09

Status: PASS for deterministic CLI-native tenant/security boundary scaffold only.
Scope: `CS-NS-004`, `CS-SEC-004`, `CS-REG-006`.

This report does not mark production UI runtime, production API runtime, real identity provider integration, production tenant migration, real connector execution, or full 206-scenario completion as complete.

## Frozen Goal

Make the frozen scenario set pass through evidence-backed batches without adding unrelated features. This batch verifies explicit namespace promotion with provenance, local deterministic RBAC/ABAC access decisions, and prevention of personal-memory leakage into organization answers before explicit promotion.

## Research Checkpoint

- NIST SP 800-162 defines ABAC as authorization based on subject, object, operation, and environmental attributes evaluated against policy: <https://csrc.nist.gov/pubs/sp/800/162/upd2/final>
- Zanzibar-style authorization stores permissions and performs authorization checks from stored relationship data: <https://pdos.csail.mit.edu/6.824/papers/zanzibar.pdf>
- W3C PROV frames provenance records as the basis for traceability and trustworthiness: <https://www.w3.org/TR/prov-overview/>
- Open Policy Agent is a CNCF-graduated general-purpose policy engine for policy-as-code, but it would be a new production dependency for this repo: <https://www.cncf.io/projects/open-policy-agent-opa/>

Best fit for this batch remains the existing deterministic local runtime. It adds auditable policy decision records and namespace-promotion provenance directly to the no-dependency scaffold instead of installing a policy engine or changing production tenant/auth behavior.

## Assumptions

- Native CLI JSON is the scaffold verification surface until production UI/API surfaces exist.
- Option 1 approval allows local deterministic tenant/security semantics only.
- Promotion is explicit copy-with-provenance from personal scope to organization scope.
- Access decisions are deterministic local scaffold records, not production authz decisions.

## Out Of Scope

- Production tenant mutation, live identity provider integration, live secrets, real connector execution, real network egress, new policy-engine dependency, and production UI/API walkthrough.
- Full 206-scenario completion remains out of scope for this batch.

## Checklist

- [x] Frozen `CS-NS-004`, `CS-SEC-004`, and `CS-REG-006` wording inspected.
- [x] README read before coding.
- [x] Research checkpoint completed for ABAC/ReBAC/provenance/policy-as-code.
- [x] Personal owner-approved memory remains private before promotion.
- [x] Organization answer before promotion returns insufficient evidence and uses no personal memory refs.
- [x] Direct organization-scope read of personal memory is denied.
- [x] Explicit namespace promotion creates an organization-scoped memory copy with source provenance, evidence refs, policy refs, and audit refs.
- [x] Organization answer after promotion uses only the promoted organization-scoped memory.
- [x] Access matrix covers role, attributes, namespace, classification, mission authority, and policy allow/deny decisions.
- [x] Matrix PASS rows backed by a JSON report artifact.
- [x] Unit and shell-gate coverage added.
- [x] No destructive action, external call, secret access, production tenant/security mutation, new dependency, or production state change.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| CS-NS-004 | MUST_PASS | PASS | `reports/scenario/vs0-tenant-security-boundary-2026-06-09.json`, `namespace promote` transcript |
| CS-SEC-004 | MUST_PASS | PASS | `reports/scenario/vs0-tenant-security-boundary-2026-06-09.json`, seven-case `access evaluate` matrix |
| CS-REG-006 | REGRESSION_GUARD | PASS | `reports/scenario/vs0-tenant-security-boundary-2026-06-09.json`, pre/post promotion memory-answer transcripts |

## Human Required

No human-required item was introduced for this local batch. Production tenant/security rollout remains human-required in a later batch and would need real identity-provider, tenant, admin-role, audit-retention, and rollback evidence before production PASS.

## Command Evidence

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-tenant-security-boundary --json --output reports/scenario/vs0-tenant-security-boundary-2026-06-09.json
# status: success
# scenario_set: vs0-tenant-security-boundary
# summary.pass: 3
# summary.blocking: 0
# summary.product_feature_claims: PARTIAL_VS0_TENANT_SECURITY_BOUNDARY_ONLY
# tenant_security_evidence.pre_promotion_answer_status: insufficient_evidence
# tenant_security_evidence.post_promotion_answer_status: answered
# tenant_security_evidence.post_promotion_used_promoted_memory: true
# tenant_security_evidence.direct_cross_scope_read_exit_code: 6
# tenant_security_evidence.promotion_mode: copy_with_provenance
# tenant_security_evidence.promotion_policy: local_rbac_abac_matrix
# tenant_security_evidence.access_matrix_case_count: 7
# tenant_security_evidence.access_allow_count: 3
# tenant_security_evidence.access_deny_count: 4
# negative_evidence.pre_promotion_personal_memory_used: 0
# negative_evidence.post_promotion_used_source_memory_directly: 0
# negative_evidence.unauthorized_access_allowed: 0
# negative_evidence.real_external_http_calls: 0
# negative_evidence.secret_reads: 0
```

## Evidence Summary

- `memory answer` in organization scope before promotion returns `insufficient_evidence` with no used memory refs and a deny policy record.
- `memory show` from organization scope against the personal memory exits 6 with `CS_SCOPE_DENIED`.
- `namespace promote` creates a target organization memory, links to the source memory, keeps source evidence refs, emits a policy decision, and records `namespace.promotion.created`.
- `memory answer` in organization scope after promotion returns `answered`, uses the promoted organization memory, and records the promotion ref.
- `access evaluate` records deterministic allow/deny decisions for org admin, org member, org approver, personal user, restricted classification, configuration, and mission-authority cases.

## Matrix Update

`docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv` now marks `CS-NS-004`, `CS-SEC-004`, and `CS-REG-006` as `PASS`.

Current full matrix after this batch:

- `PASS`: 65
- `NOT_VERIFIED`: 141
- `FAIL`: 0
- `NOT_RUN`: 0

Current VS-0 subset after this batch:

- `PASS`: 58
- `NOT_VERIFIED`: 0

## Gaps

- Full 206-scenario PASS remains incomplete.
- Production UI/API product surfaces remain missing; this batch uses CLI-native JSON as scaffold evidence.
- The local deterministic policy matrix is not a replacement for production identity provider, tenant model, or durable policy service design.

## Risks

- Future production authz must preserve the same evidence, policy-decision, and audit semantics while replacing the local deterministic evaluator.
- Relationship-heavy authorization may need ReBAC/Zanzibar-style modeling later; this batch deliberately avoids adding that dependency or schema surface.
- A future UI/API implementation must continue to deny implicit cross-namespace memory use and expose promotion provenance without leaking source-scope content.

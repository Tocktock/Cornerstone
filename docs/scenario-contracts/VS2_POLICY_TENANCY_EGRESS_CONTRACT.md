# Summary

This document freezes a proposed task-scoped scenario contract for **CornerStone VS2 — Policy, Tenant Isolation, and Default Egress Deny**.

The current active repository does not contain an active VS2 contract. The term **VS2** is therefore resolved, as an explicit assumption, to the compatible milestone in the legacy product SoT:

```text
PostgreSQL RLS tenant isolation
+ OPA/Rego policy enforcement
+ default egress deny
```

The contract contains:

- **70 MUST_PASS** scenarios;
- **16 REGRESSION** scenarios;
- **7 HUMAN_REQUIRED** scenarios;
- **93 total** scenario rows.

All AI-verifiable scenarios begin as `NOT_RUN`. No implementation or runtime PASS is claimed by this document.

## Goal

Replace the current local deterministic security scaffold with a smallest complete, locally verifiable VS2 security slice that proves:

1. tenant isolation is enforced by real PostgreSQL RLS and hardened database roles;
2. a real OPA/Rego policy control plane governs data, tools, models, actions, connectors, egress, and sensitive administration;
3. outbound network access is denied at a real runtime/network boundary unless a declared, tenant-scoped, policy-approved capability permits it;
4. gateway, service, database, tool/runtime, ConnectorHub, audit, CLI, API, and UI behavior remain consistent;
5. VS0 and VS1 continue to pass;
6. local integration evidence is not overclaimed as production, live-provider, penetration-tested, or human-accepted security.

## Source Basis

- User-provided `test_scenario_based_implementation_prompt.md`.
- User-provided legacy `project-sot.md` (2026-02-17), specifically the VS2 milestone and security defaults.
- User-provided CornerStone Project Operating Constitution and Scenario-First Agent Instruction.
- `Tocktock/Cornerstone`: active product SoT, 206-scenario standard, root AGENTS, technical defaults, VS1 contract/report, current CLI and Makefile, and VS0/full security reports.
- `Tocktock/KnowledgeBase`: README and AGENTS describing the Postgres Archive boundary, current scaffold status, read-only sources, redaction, hashing, and evidence storage.
- `Tocktock/Connector-Hub`: README and internal CornerStone contract defining provider/credential/action ownership and policy/approval handoff.
- Official PostgreSQL Row Security Policies documentation and official OPA documentation for Rego testing, bundles, decision logs/status, REST decisions, and service hardening.

# Problem Definition

## 1. Product and milestone confusion

The legacy SoT defines VS2, but the active repository has moved to a newer product authority and currently exposes a canonical VS1 contract without a VS2 successor. Without a new task-scoped contract, engineers could implement different meanings of “VS2” or treat historical milestone text as current implementation evidence.

**Root cause:** milestone naming survived a product-goal reset, while acceptance criteria were not re-frozen under the active scenario-first and CLI-native-first rules.

## 2. Security semantics are not yet equivalent to security enforcement

Current reports demonstrate useful deterministic denial semantics, reason text, audit references, tenant fixtures, and negative counters. Those reports explicitly scope themselves to local scaffold behavior. They do not prove:

- actual PostgreSQL RLS;
- an actual OPA decision service or Rego bundle lifecycle;
- runtime/OS-level network containment;
- production identities, tenants, provider credentials, or network controls.

**Root cause:** the initial implementation optimized for deterministic scenario scaffolding, while VS2 requires moving enforcement into durable database, policy, and runtime boundaries.

## 3. Domain-boundary risk

VS2 crosses three engines:

- CornerStone product owns trusted request context, product policy, approvals, workflows, UX, and product audit.
- KnowledgeBase/Archive owns immutable originals, evidence, redaction, provenance, and durable archive concepts.
- ConnectorHub owns provider access, credential custody, declared actions, provider execution, connector audit, retry, quarantine, and raw-access control.

**Root cause:** if a shared security contract is not explicit, product code may duplicate provider credentials, ConnectorHub may be treated as product approval, or archive/search paths may bypass tenant policy.

## 4. Validation gaps

VS2 must reject missing or forged scope, undefined policies, malformed inputs, stale decisions, invalid bundles, unsafe destinations, cross-tenant keys, and ownerless migrations. Frontend controls cannot provide security.

**Root cause:** current scope arguments and local fixtures are useful test inputs but are not a trusted production identity boundary.

## 5. Data-consistency risks

RLS correctness can fail through database role bypass, connection pools, background workers, joins, search, caches, object/blob links, views/functions, unique/FK constraints, migration backfills, and backup/export paths.

**Root cause:** tenant isolation is a system invariant, not a WHERE clause on a small set of API queries.

## 6. Downstream execution risks

A policy allow at dry-run can become stale before execution. An allowlisted URL can redirect, re-resolve, use proxy variables, or invoke a direct socket. A prompt or provider payload can attempt to expand authority.

**Root cause:** policy decisions, approvals, runtime capability enforcement, ConnectorHub execution, and network controls can drift unless linked by immutable context, revision, evidence, and audit references.

## SoT vs Repository Gap Map

| Area | Documented Target | Observed Repository Reality | VS2 Contract Response |
| --- | --- | --- | --- |
| Milestone identity | The February legacy SoT names VS2 as OPA + RLS + default egress deny. | The active repository has a canonical VS1 contract but no active VS2 contract; `Makefile` has targets through VS1 only. | Freeze this document as the task-scoped VS2 scenario contract before implementation. |
| Tenant isolation | Active technical defaults call for Postgres RLS plus service policy. | Current reports prove synthetic/local tenant semantics; they explicitly do not prove production RLS or durable storage enforcement. | Introduce a real Postgres integration profile, hardened roles, RLS inventory, and two-tenant database tests. |
| Policy engine | Legacy VS2 names OPA; active defaults require OPA/Rego-compatible policy. | Current RBAC/ABAC checks use a deterministic local evaluator; repository search did not locate an active Rego/OPA implementation. | Add a versioned OPA contract, real policy process, Rego tests, bundle lifecycle, and fail-closed adapter. |
| Egress | Default egress deny is a non-negotiable safety default. | Current egress verification intentionally makes zero network calls and is not proof of OS/runtime network isolation. | Add a real sandbox/network-boundary test against a controlled local egress sink, including bypass cases. |
| Cross-repository boundary | CornerStone owns policy/approval; KnowledgeBase owns archive/evidence; ConnectorHub owns provider access/credentials/actions. | The current repositories provide local scaffolds and handoff contracts, but no verified unified VS2 enforcement plane. | Use shared scope/decision/envelope contracts; do not duplicate provider credentials or archive truth in product code. |
| Release evidence | Scenario PASS requires concrete CLI/API/UI/DB/runtime evidence. | Existing full-matrix PASS rows are scoped to deterministic local scaffold reports, not production VS2 controls. | Create a dedicated `vs2-policy-tenancy-egress` verifier, report, gate, evidence manifest, and regression reruns. |

# Requirements

## Explicit Requirements

| ID | Explicit Requirement |
| --- | --- |
| ER-01 | Enforce tenant isolation in PostgreSQL with database-level Row-Level Security (RLS), not only application filtering. |
| ER-02 | Use OPA/Rego policy decisions to govern data, tool, action, model, connector, and administrative access. |
| ER-03 | Deny external egress by default; permit exceptions only through declared, policy-approved capabilities. |
| ER-04 | Apply RBAC plus ABAC using role, attributes, tenant, namespace, classification, mission/workspace authority, and risk. |
| ER-05 | Enforce policy at gateway, service, and tool/runtime boundaries; no single enforcement point is sufficient. |
| ER-06 | Return helpful denials with cause and safe resolution guidance, without leaking protected information. |
| ER-07 | Record policy, access, RLS, tool, connector, egress, approval, execution, and security-change events in auditable evidence. |
| ER-08 | Keep provider access and credential custody behind ConnectorHub; CornerStone owns product policy, approvals, workflows, and evidence. |
| ER-09 | Keep every truth-bearing object explicitly tenant-, owner-, namespace-, and where relevant workspace-scoped. |
| ER-10 | Provide CLI-native parity for every VS2 product behavior, alongside API and UI evidence. |
| ER-11 | Preserve verified VS0 and VS1 behavior while adding VS2 controls. |
| ER-12 | Treat uploaded, connected, retrieved, and generated content as untrusted evidence; it cannot expand authority or trigger egress/actions. |
| ER-13 | Require dry-run, current policy evaluation, and approval where required before risky or external execution. |

## Implicit Requirements

| ID | Implicit Requirement |
| --- | --- |
| IR-01 | Authoritative tenant/role/attribute context must be derived from trusted identity and membership state, never accepted from caller-controlled fields. |
| IR-02 | Tenant context must survive concurrency, retries, workers, and connection pooling without cross-request contamination. |
| IR-03 | The runtime database role must be structurally unable to bypass RLS; privileged maintenance must be separate and governed. |
| IR-04 | Migrations and CI need a complete inventory so new tables/views/functions cannot silently omit tenant policy. |
| IR-05 | Search, caches, object/blob access, cursors, idempotency, deduplication, exports, and metrics must be tenant-scoped. |
| IR-06 | Policy input/output and PolicyDecision records need versioned schemas, stable reason codes, decision IDs, revisions, and redaction. |
| IR-07 | Policy bundles, caches, outages, revocations, and component readiness must fail safely and remain observable. |
| IR-08 | Default egress deny must be proved at a real runtime/network boundary, not by an application branch that simply avoids making a call. |
| IR-09 | Execution must re-evaluate policy and impact after dry-run to prevent time-of-check/time-of-use authorization drift. |
| IR-10 | Security logs, errors, metrics, and reports must be useful without becoming a side channel or secret store. |
| IR-11 | Postgres/OPA adoption requires reversible migration, compatibility, backup/restore, and explicit approval gates. |
| IR-12 | Live provider, real IdP, production network, and subjective UX claims require separate human/external evidence. |
| IR-13 | Schema, policy, audit, and release gates must remain future-proof as tables and capabilities are added. |

## Assumptions

| ID | Assumption Requiring Confirmation |
| --- | --- |
| A-01 | “VS2” is interpreted as the compatible legacy milestone: Policy (OPA) + multi-tenancy (Postgres RLS) + default egress deny. The active repository currently has no task-scoped VS2 contract. |
| A-02 | The intended AI-verifiable VS2 target is a local/on-prem integration profile using a real PostgreSQL process, a real OPA/Rego evaluation process, and a deterministic local egress test harness. This is not a production-readiness claim. |
| A-03 | The local CI gate does not require public Internet, paid model credentials, live provider credentials, or a production IdP. |
| A-04 | Synthetic trusted principals and memberships may be used for automated tests; real OIDC/SSO mapping remains HUMAN_REQUIRED. |
| A-05 | The smallest safe migration keeps the current local scaffold behind compatibility interfaces while moving the VS0/VS1 durable security slice to Postgres; it does not merge all three repositories. |
| A-06 | Actual production dependency addition, auth/tenant model changes, and durable-state migration require owner approval before implementation. |
| A-07 | Allowed egress in automated tests targets only controlled local mock endpoints. Live external provider execution remains HUMAN_REQUIRED. |

## Out of Scope Before Implementation

- Full public-cloud/Kubernetes production hardening, certification, or government accreditation.
- Production OIDC/SSO implementation and enterprise group governance.
- Live external writeback or provider E2E without approved credentials and human evidence.
- Full Agent Pack registry, signing/SLSA/Rekor/TUF work (the later extension/supply-chain slice).
- A broad repository merge or replacement of ConnectorHub/KnowledgeBase ownership boundaries.
- Full ReBAC/Zanzibar implementation; VS2 must remain compatible with future relationship-based authorization.
- Destructive production migration, deletion, or retention execution without separate approval.
- Claiming production penetration-test, network-control, live-provider, or human UX acceptance from local deterministic evidence.

# Test Scenarios

## Status and PASS Rules

- `PASS` requires the stated behavior plus the required evidence generated by an executed check.
- Source inspection, design intent, existing local-scaffold reports, or “should work” are not sufficient.
- `REGRESSION` is the task-level label corresponding to the project's canonical `REGRESSION_GUARD`.
- `HUMAN_REQUIRED` is used only where owner approval, real production topology, real credentials, independent security review, or subjective human judgment is genuinely required.
- The scenario matrix is the implementation contract. Scenario IDs may not be silently removed after implementation starts.

| Scenario ID | Priority | Related Requirement | Given | When | Then / Expected Behavior | Implementation Area | Verification | Required Evidence | Initial Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VS2-SEC-001 | MUST_PASS | ER-04, ER-05, ER-10; IR-01 | The same authenticated principal, tenant, namespace, resource, and action are available through CLI, API, and UI. | The operation is submitted through each surface. | All surfaces derive the same trusted request context, receive the same policy outcome and stable error semantics, and emit linked policy/audit references. | Shared RequestContext, policy client, CLI adapter, API middleware, UI error mapping. | Automated CLI/API/browser integration test using one fixture principal and normalized input comparison. | Three transcripts, normalized request-context digest, matching decision outcome/revision, exit/status codes, audit references. | NOT_RUN |
| VS2-SEC-002 | MUST_PASS | ER-01, ER-04; IR-01 | A caller is authenticated for tenant A and supplies tenant B, owner, role, or classification values in a header, body, query, or CLI flag. | The caller invokes a protected read, write, policy, tool, or action operation. | Authoritative scope comes only from trusted identity/membership state; forged attributes are ignored or rejected, tenant B is not accessed, and the attempt is audited. | Identity-to-scope resolver, input allowlist, anti-confused-deputy validation. | Parameterized API/CLI tests for every client-controlled scope field and direct service invocation. | Denied response, zero tenant-B rows/objects, policy decision, audit event, negative-evidence counters. | NOT_RUN |
| VS2-SEC-003 | MUST_PASS | ER-01, ER-02, ER-03; IR-01, IR-06 | A protected request has missing, malformed, expired, conflicting, or unresolvable principal/tenant context. | The request reaches the gateway, service, worker, or tool entry point. | The system fails closed before protected database access or egress, returns a stable helpful error, and records no secret-bearing input. | Request validation, principal resolver, policy fallback, worker/tool guards. | Unit and integration matrix for null, empty, malformed, expired, and conflicting context. | Stable error codes, zero DB/egress calls, sanitized policy/audit records. | NOT_RUN |
| VS2-SEC-004 | MUST_PASS | ER-01, ER-09; IR-03, IR-04 | A truth-bearing object is created or migrated: Artifact, DerivedRepresentation, SearchSnapshot, EvidenceBundle, Claim, ontology object/link, ActionCard, WorkflowRun, PolicyDecision, or AuditEvent. | The object is persisted. | Required tenant_id, namespace_id, owner_id, and applicable workspace/classification fields are non-null, validated, immutable where required, and visible in API/CLI evidence. | Domain models, database constraints, serializer validation, migration checks. | Schema tests plus create/read tests for every active durable object type. | Schema inventory, constraint output, representative rows/API payloads, failed-null inserts. | NOT_RUN |
| VS2-SEC-005 | MUST_PASS | ER-04, ER-05; IR-07 | A principal's membership, role, attribute, capability, or workspace access is revoked. | The principal repeats a previously allowed request, including through a cached session or long-running worker. | The new request is denied within the documented revocation window; stale authorization state cannot preserve access. | Membership store, token/session versioning, policy cache invalidation, worker lease checks. | Integration test with allow → revoke → retry across API, CLI, service, and worker paths. | Before/after decisions, cache revision/expiry, denial audit, zero post-revocation access. | NOT_RUN |
| VS2-SEC-006 | MUST_PASS | ER-02, ER-06, ER-07; IR-06 | Any protected operation is evaluated. | OPA and the CornerStone policy adapter return a result. | A canonical PolicyDecision records allow/deny/escalate, reason codes, safe resolution guidance, policy path, bundle revision/hash, decision_id, input digest, timestamps, tenant/namespace, and evidence/audit refs. | PolicyDecision schema, OPA result adapter, persistence, redaction. | Schema contract tests and golden JSON tests for allow, deny, escalate, undefined, and error results. | PolicyDecision JSON fixtures, OPA decision_id/revision, audit linkage, redaction proof. | NOT_RUN |
| VS2-SEC-007 | MUST_PASS | ER-01, ER-09; IR-02, IR-03 | Tenant A and tenant B each have rows in every active tenant-bearing table. | The application database role executes SELECT as tenant A. | Only tenant-A rows are visible; tenant-B rows and existence metadata are absent. | Postgres RLS policies, transaction-local tenant context, repository adapters. | Database integration matrix across every RLS table using the real application role. | SQL transcript, row counts, pg_policies inventory, zero tenant-B identifiers. | NOT_RUN |
| VS2-SEC-008 | MUST_PASS | ER-01; IR-03 | The application role is scoped to tenant A. | It attempts INSERT with tenant B, UPDATE a row into tenant B, or DELETE tenant-B data. | INSERT/UPDATE are rejected by WITH CHECK or equivalent; cross-tenant DELETE affects zero rows or is denied; no partial mutation occurs. | Command-specific RLS policies, immutable tenant key constraints, transaction handling. | Database tests for INSERT/UPDATE/DELETE, including bulk statements and RETURNING. | SQLSTATE/neutral API error, unchanged before/after snapshots, rollback evidence, audit event. | NOT_RUN |
| VS2-SEC-009 | MUST_PASS | ER-01; IR-03, IR-10 | Tenant A queries tables containing tenant-B data through joins, subqueries, aggregates, counts, groupings, existence checks, and pagination. | The query is executed through supported repository and API paths. | Results, counts, cursors, totals, and timing/error text do not reveal tenant-B data or existence. | RLS-safe queries, tenant-scoped pagination/cursors, neutral errors. | Parameterized SQL/API integration tests including empty-result and boundary-page cases. | Result snapshots, count/cursor assertions, no foreign IDs, sanitized errors. | NOT_RUN |
| VS2-SEC-010 | MUST_PASS | ER-01, ER-09; IR-05 | FTS, vector/semantic search, ontology search, autocomplete, and saved search snapshots contain content from two tenants. | Tenant A searches using terms unique to tenant B. | No tenant-B result, snippet, score, suggestion, facet count, object, or snapshot reference is returned. | Tenant-bearing search tables/indexes, RLS-safe search queries, tenant-scoped embeddings/cache. | End-to-end search tests with unique canary terms in each tenant. | Search/API transcripts, query plans or SQL trace, zero canary leakage, snapshot scope. | NOT_RUN |
| VS2-SEC-011 | MUST_PASS | ER-01, ER-08, ER-09; IR-05 | Tenant B owns an original blob, derived representation, EvidenceBundle item, or storage URI referenced by an identifier guessed by tenant A. | Tenant A requests the object, download, signed URL, or evidence traversal. | Access is denied without revealing content or sensitive metadata; storage access is bound to the authorized tenant and object. | Artifact service policy, tenant-scoped storage keys/signing, evidence traversal checks. | API/object-store adapter tests using guessed IDs and cross-tenant evidence links. | Denied responses, zero bytes returned, storage access log, policy/audit refs. | NOT_RUN |
| VS2-SEC-012 | MUST_PASS | ER-01, ER-07; IR-05, IR-10 | Audit events, policy decisions, operator metrics, exports, and status records exist for multiple tenants. | A tenant-scoped user queries or exports them. | Only authorized tenant records and aggregated non-sensitive metrics are visible; system-wide access requires explicit privileged policy. | RLS/policy for audit and policy tables, tenant-aware metrics/export layer. | API/DB tests for tenant user, tenant admin, and system operator fixtures. | Scoped exports, role matrix, denial logs, no cross-tenant event IDs. | NOT_RUN |
| VS2-SEC-013 | MUST_PASS | ER-01; IR-03 | The application connects to Postgres using its normal runtime role. | Role and table security attributes are inspected and protected queries run. | The role is not superuser, table owner, or BYPASSRLS; protected tables have RLS enabled and forced where the design requires owner protection. | Dedicated migration owner and application roles, FORCE ROW LEVEL SECURITY, grants. | Database metadata assertions over pg_roles, pg_class, pg_policies, and ownership. | Role inventory showing rolsuper=false and rolbypassrls=false; relrowsecurity/relforcerowsecurity report. | NOT_RUN |
| VS2-SEC-014 | MUST_PASS | ER-01, ER-07; IR-03 | An operator needs cross-tenant maintenance, recovery, or investigation. | A break-glass or maintenance path is requested. | Normal app credentials cannot perform it; the privileged path requires explicit authorization, purpose, time bound, audit, and safe output handling. | Separate maintenance role/workflow, approval record, scoped admin commands. | Integration rehearsal with denied app role and approved synthetic break-glass role. | Approval record, role assumption/revocation timestamps, accessed row counts, audit trail. | NOT_RUN |
| VS2-SEC-015 | MUST_PASS | ER-01; IR-02 | A pooled database connection previously served tenant A. | The same physical connection is reused for tenant B, on success, error, timeout, cancellation, and rollback paths. | Tenant context is transaction-local and reset; tenant B never sees tenant A state or rows. | Connection checkout/checkin hooks, SET LOCAL context, transaction discipline. | Pool stress test that alternates tenants and injects failures/cancellations. | Connection IDs with tenant sequence, zero canary leakage, reset assertions. | NOT_RUN |
| VS2-SEC-016 | MUST_PASS | ER-01; IR-02, IR-10 | Concurrent tenant-A and tenant-B requests use the same service and pool. | Reads and writes interleave under load. | Each request remains isolated; no cross-request context, cache, result, or audit correlation is mixed. | Async context propagation, pool discipline, tenant-scoped caches. | Concurrent integration test with unique canaries and randomized scheduling. | Load transcript, per-request trace IDs, zero foreign canaries, no mixed audit refs. | NOT_RUN |
| VS2-SEC-017 | MUST_PASS | ER-01, ER-05; IR-02 | A background job, retry, scheduled task, extraction worker, or workflow continuation processes tenant-owned work. | The job executes with valid, missing, stale, or tampered scope metadata. | Valid jobs run only in their tenant; missing/tampered jobs fail closed or quarantine; retries preserve scope and idempotency. | Signed/validated job envelope, worker RequestContext, quarantine and retry policy. | Worker integration tests for valid, missing, changed, replayed, and cross-tenant envelopes. | Job envelope digest, worker decision, quarantine/retry record, scoped DB/audit trace. | NOT_RUN |
| VS2-SEC-018 | MUST_PASS | ER-01; IR-03, IR-04 | Protected data is accessed through SQL views, functions, stored procedures, ORM raw SQL, or security-definer code. | The application role invokes each supported path. | No path bypasses tenant isolation; unsafe security-definer or owner-context access is rejected by CI or explicitly governed. | Database function/view security review, restricted EXECUTE grants, repository lint. | Database integration tests plus static inventory of views/functions/security-definer objects. | Object inventory, grants, cross-tenant tests, lint report. | NOT_RUN |
| VS2-SEC-019 | MUST_PASS | ER-01; IR-03, IR-10 | Tenant A attempts writes that collide with tenant-B unique keys or reference tenant-B foreign keys. | The database validates constraints. | Errors do not reveal tenant-B values/existence; keys are tenant-scoped where required, and unsafe cross-tenant references are impossible. | Composite tenant keys, tenant-aware foreign keys, neutral error translation. | DB/API tests for duplicate and foreign-key boundary cases. | Schema definitions, neutral errors, unchanged data, no foreign identifiers. | NOT_RUN |
| VS2-SEC-020 | MUST_PASS | ER-01, ER-09; IR-04, IR-11 | Existing local/scaffold records or imported archive rows are migrated to the VS2 schema. | A row has known, missing, ambiguous, or invalid tenant/namespace ownership. | Known rows migrate deterministically; unknown/ambiguous rows are quarantined or block migration; no ownerless global truth is created. | Backfill mapping, quarantine table/report, NOT NULL/foreign-key constraints. | Migration tests on clean, mixed, malformed, duplicate, and rollback fixtures. | Migration report with counts/checksums, quarantine reasons, rollback transcript. | NOT_RUN |
| VS2-SEC-021 | MUST_PASS | ER-01; IR-04, IR-13 | A migration adds or changes a tenant-bearing table, partition, materialized view, or policy. | CI runs the schema security inventory. | The gate fails unless tenant columns, RLS enable/force state, command policies, grants, and test coverage are declared. | Schema linter/inventory script, migration checklist, CI gate. | Failing fixture migration, passing corrected migration, machine-readable inventory. | Schema inventory report, expected failing gate, passing migration evidence. | NOT_RUN |
| VS2-SEC-022 | MUST_PASS | ER-04, ER-09; IR-05 | Two namespaces/workspaces exist inside the same tenant, such as Personal and Organization. | A principal accesses, searches, or uses context across them. | RLS tenant access alone is insufficient; service/OPA policy enforces namespace, owner, membership, classification, and explicit promotion rules. | Namespace policy input, service authorization, scoped queries. | Integration matrix for same-tenant personal/org/team/project access. | Allow/deny decisions, zero implicit context use, provenance on allowed promotion. | NOT_RUN |
| VS2-SEC-023 | MUST_PASS | ER-01, ER-09; IR-05 | A user attempts copy, reference, share, or promotion from tenant A to tenant B. | The transfer request is evaluated. | The operation is denied at product, policy, and storage layers; no target row, link, blob, or metadata is created. | Cross-tenant invariant in domain service, OPA policy, RLS constraints. | API/CLI/DB test for every transfer mode. | Denied policy decision, zero target records, audit event, negative-evidence counters. | NOT_RUN |
| VS2-SEC-024 | MUST_PASS | ER-01; IR-05, IR-07 | Caches, idempotency keys, deduplication indexes, locks, rate limits, and saved cursors are used by multiple tenants. | Two tenants submit identical IDs/content/keys or concurrent operations. | Keys are tenant-scoped unless intentionally global and non-sensitive; one tenant cannot read, suppress, replay, or infer another tenant's operation. | Tenant-prefixed cache/idempotency keys, composite indexes, scoped locks. | Collision and replay tests across two tenants. | Key snapshots/digests, independent results, zero cross-tenant cache hits. | NOT_RUN |
| VS2-SEC-025 | MUST_PASS | ER-01, ER-07; IR-11 | Tenant-scoped export and privileged full backup/restore paths exist. | Exports and a local backup→restore rehearsal run. | Tenant export contains only authorized rows; privileged backup is complete, restore preserves RLS/policies/audit, and filtered backups cannot silently omit protected rows. | Export service, backup role/procedure, restore verification, policy migration. | Automated local backup/restore and per-tenant export tests with row/hash counts. | Backup manifest, restored counts/hashes, RLS inventory before/after, tenant export diff. | NOT_RUN |
| VS2-SEC-026 | MUST_PASS | ER-02, ER-04, ER-05; IR-06 | A protected data, tool, action, connector, model, policy-admin, or memory operation is requested. | The product constructs an OPA input document. | Input follows one versioned schema containing trusted subject, tenant/namespace/workspace, resource, action, classification, mission authority, risk, data scope, connector/tool capability, and environment fields. | Versioned PolicyInput schema, builders per enforcement point, schema validation. | Contract tests and golden inputs for every operation family. | Schema file, valid/invalid fixtures, input digest and source-of-attribute map. | NOT_RUN |
| VS2-SEC-027 | MUST_PASS | ER-02, ER-04 | A principal has the required role and membership for a low-risk operation. | OPA evaluates the operation. | The request is allowed only for the declared resource/action/scope and the allow decision is auditable. | Rego RBAC rules and role data. | OPA unit tests plus product integration allow cases. | opa test JSON, allow decision with bundle revision, downstream success/audit. | NOT_RUN |
| VS2-SEC-028 | MUST_PASS | ER-02, ER-04, ER-06 | A principal lacks a required role or approval authority. | The principal reads, writes, approves, executes, or configures a protected resource. | OPA denies with a non-leaking reason and safe resolution path; downstream mutation/egress does not occur. | Rego deny rules, reason mapping, enforcement adapter. | Role/action access-control matrix tests. | Denied decisions, zero side effects, stable error and audit references. | NOT_RUN |
| VS2-SEC-029 | MUST_PASS | ER-02, ER-04 | Role alone would allow an operation, but attributes such as namespace, classification, mission authority, workspace mode, environment, or risk disallow it. | OPA evaluates the request. | ABAC conditions deny or escalate it; a matching allowed attribute set succeeds. | Rego ABAC rules, classification/risk taxonomy, mission/workspace inputs. | Data-driven OPA tests at each attribute boundary. | Allow/deny/escalate matrix, reason codes, policy coverage report. | NOT_RUN |
| VS2-SEC-030 | MUST_PASS | ER-02, ER-03; IR-06 | A resource type, action, tool, connector capability, model, or policy path is unknown or returns undefined. | The enforcement adapter evaluates it. | The normalized result is deny, never implicit allow, with an unknown-policy reason and resolution. | Default-deny Rego rules and undefined-result adapter. | OPA/product tests for typos, new enum values, missing rule paths, and undefined output. | Denied outputs, no downstream calls, audit decision. | NOT_RUN |
| VS2-SEC-031 | MUST_PASS | ER-02, ER-06; IR-06 | OPA input is partial, malformed, over-specified, wrong-versioned, or contains unexpected authoritative attributes. | The policy endpoint receives it. | Schema validation fails closed; errors identify safe corrective action without echoing secrets or protected values. | Strict input schema, version negotiation, unknown-field policy, redaction. | Fuzz/property tests and boundary fixtures. | 400/deny transcripts, zero evaluation/side effects as appropriate, sanitized logs. | NOT_RUN |
| VS2-SEC-032 | MUST_PASS | ER-02, ER-03; IR-07 | The OPA service is running in the local/on-prem profile. | A local authorized client and an unauthorized/remote client access its APIs. | OPA is bound to the intended interface, protected by the chosen authentication/transport controls, and management/data APIs are not anonymously exposed. | OPA hardened configuration, network binding, client authentication/authorization. | Container/network tests from allowed and disallowed peers. | OPA config, listening sockets, successful authorized call, denied unauthorized call. | NOT_RUN |
| VS2-SEC-033 | MUST_PASS | ER-05 | A request reaches the API gateway. | Gateway policy denies it. | The request does not reach the service handler, database, tool runtime, model router, or connector. | Gateway middleware/enforcement point and trace propagation. | Integration test with handler/DB/tool call counters. | Gateway decision, zero downstream counters, trace/audit record. | NOT_RUN |
| VS2-SEC-034 | MUST_PASS | ER-05 | A caller bypasses or originates behind the gateway and invokes a protected service method. | Service-layer policy denies it. | No protected read, mutation, tool, model, or connector operation occurs. | Service authorization decorator/interceptor and repository guards. | Direct service/API-internal integration tests. | Service decision, zero repository/tool/connector calls, audit event. | NOT_RUN |
| VS2-SEC-035 | MUST_PASS | ER-03, ER-05 | A tool or Agent Pack receives a request that gateway/service policy allowed but runtime capability policy denies. | The tool attempts filesystem, environment, host, shell, or network access. | The runtime blocks the undeclared capability and records negative evidence; upstream allow cannot bypass runtime enforcement. | Capability sandbox, tool policy adapter, runtime audit. | Real sandbox integration test for each capability class. | Runtime denial, zero operations/bytes/calls, policy and audit refs. | NOT_RUN |
| VS2-SEC-036 | MUST_PASS | ER-01, ER-02, ER-05 | Service policy mistakenly allows or is bypassed for a cross-tenant database query. | The application role executes the query. | Postgres RLS still prevents access, proving defense in depth; the anomaly is observable. | RLS independent of service decision, anomaly logging. | Fault-injection integration test that stubs an allow at service layer. | Service allow + DB zero rows/denial, anomaly audit/metric. | NOT_RUN |
| VS2-SEC-037 | MUST_PASS | ER-02, ER-05, ER-13 | An ActionCard or WorkflowRun requests internal mutation, risky action, or external side effect. | Policy evaluates dry-run, approval, and execution phases. | Allowed low-risk behavior follows mission/workspace authority; high-risk behavior escalates/requires approval; execution rechecks policy and cannot bypass Workflow/Action. | Action policy entrypoints, risk rules, workflow gate. | Action lifecycle integration matrix. | Dry-run, policy decisions, approval state, execution/denial, audit timeline. | NOT_RUN |
| VS2-SEC-038 | MUST_PASS | ER-02, ER-05 | A workspace policy allows or disallows model providers based on classification or on-prem requirements. | The model router selects a provider. | Disallowed providers are blocked or safely rerouted with explanation; no request/data is sent to them. | Policy-aware model router, provider capability metadata. | Router integration test with local_test and denied external provider fixtures. | Routing decision, zero disallowed provider calls, explanation and audit refs. | NOT_RUN |
| VS2-SEC-039 | MUST_PASS | ER-02, ER-08 | A workflow, agent, or extension requests provider data, credentials, raw access, or an external action. | Policy and ConnectorHub capability checks run. | Only declared ConnectorHub-mediated capabilities proceed; raw credentials/provider clients/direct API handles remain inaccessible; CornerStone retains product approval ownership. | ConnectorHub handoff envelope, capability policy, credential references only. | Contract tests with declared/undeclared capability and raw-secret probes. | Setup/preflight result, policy decision, zero raw secrets, connector and CornerStone audit refs. | NOT_RUN |
| VS2-SEC-040 | MUST_PASS | ER-02; IR-06 | The Rego policy bundle contains all VS2 entrypoints. | CI runs policy tests. | All positive, negative, malformed, boundary, and regression cases pass; no-test execution fails; a machine-readable coverage report is produced and reviewed against declared entrypoints. | Rego unit/data-driven tests, entrypoint manifest, CI commands. | Run `opa test ... --fail-on-empty --coverage --format=json` plus bundle build/check. | OPA test JSON, coverage JSON, entrypoint coverage map, exit code 0. | NOT_RUN |
| VS2-SEC-041 | MUST_PASS | ER-02, ER-07; IR-07 | A valid new policy bundle is published. | OPA downloads and activates it while requests continue. | Activation is atomic; decisions identify the active revision; no request observes a partial mix; the previous revision remains traceable. | Versioned bundle build/service, revision metadata, status monitoring. | Concurrent evaluation during bundle update. | OPA status before/after, revision transition, decision sample set, zero mixed/undefined results. | NOT_RUN |
| VS2-SEC-042 | MUST_PASS | ER-02, ER-07; IR-07 | A bundle is malformed, fails Rego tests/compile, has incompatible schema, or cannot activate. | The update is attempted. | Activation fails, the last known good bundle remains active (or protected operations fail closed if none exists), and status/error is visible without exposing secrets. | Pre-activation validation, persisted last-known-good, readiness/status integration. | Bundle failure injection tests including first-start failure. | OPA status/error, unchanged active revision, decisions under failure, operator alert. | NOT_RUN |
| VS2-SEC-043 | MUST_PASS | ER-02, ER-03, ER-05; IR-07 | OPA is unavailable, slow, returns invalid output, or times out. | A protected read, write, tool, action, connector, or model request is made. | The operation fails closed or follows an explicitly documented safe read-only fallback; readiness degrades, and no risky side effect/egress occurs. | Policy client timeout/circuit breaker, fail-closed adapter, readiness. | Fault-injection tests for connection refused, timeout, 5xx, malformed JSON. | Stable denial/error, zero side effects, readiness/metric/audit evidence. | NOT_RUN |
| VS2-SEC-044 | MUST_PASS | ER-02; IR-07 | An allow decision is cached and policy/membership is then revoked or bundle revision changes. | The same operation is retried. | The stale allow is not reused beyond the documented bound; cache keys include tenant, principal, resource, action, and revision. | Revision-aware decision cache and invalidation. | Allow→revoke/update→retry tests across tenants. | Cache keys/digests, new decision revision, denial, zero stale allows. | NOT_RUN |
| VS2-SEC-045 | MUST_PASS | ER-06, ER-07; IR-06, IR-10 | Policy decisions contain potentially sensitive input fields. | Decision logging and CornerStone audit mirroring run. | Each event has decision_id, policy path/revision, tenant-safe correlation, and masked/redacted sensitive fields; logs are linkable but not a secret store. | OPA decision logging/masking policy, audit adapter, redaction. | Secret-bearing fixture through allow and deny decisions. | Decision log with masked fields, CornerStone audit ref, zero secret scanner findings. | NOT_RUN |
| VS2-SEC-046 | MUST_PASS | ER-02, ER-07; IR-11 | A user proposes a policy, role, tenant-isolation, egress, retention, or audit configuration change. | The change is reviewed/applied. | It is represented as a sensitive governed change with diff, impact, tests, policy decision, authorized approval, version, rollback, and audit; direct mutation is blocked. | Policy-admin ActionCard/Workflow, bundle pipeline, rollback metadata. | Integration test for draft→dry-run→approve→activate→rollback and direct bypass denial. | Action/workflow records, bundle revisions, approvals, before/after decisions, rollback transcript. | NOT_RUN |
| VS2-SEC-047 | MUST_PASS | ER-06 | A user is denied data, action, model, connector, tool, egress, or policy-admin access. | The error is shown through API, CLI, and UI. | The response states safe cause, impact, retry/request-access/change-scope/approver resolution where applicable, and what remains preserved, without exposing protected data. | Reason-code catalog, error translation, UI denial component. | Golden API/CLI/browser tests for major denial classes. | Response snapshots/screenshots, stable codes, matching decision/audit refs, leak scan. | NOT_RUN |
| VS2-SEC-048 | MUST_PASS | ER-02, ER-06; IR-06 | Policy input/output is at, below, and above configured size/depth/list limits, or contains unknown enum values. | The request is evaluated. | At/below-limit requests behave deterministically; over-limit/unsupported requests fail safely with helpful errors, bounded resource use, and no partial side effect. | Documented policy limits, parser/schema guards, time/memory limits. | Boundary-value and property/fuzz tests using configuration-derived N-1/N/N+1 cases. | Limit configuration, test corpus, resource metrics, stable failures. | NOT_RUN |
| VS2-SEC-049 | MUST_PASS | ER-02, ER-04; IR-06 | Multiple policy rules apply, including an allow and a restrictive deny/escalation. | The aggregate decision is computed. | The documented conservative precedence is deterministic; restrictive tenant/security/risk rules cannot be accidentally overridden by a permissive rule. | Decision aggregation policy, explicit deny/escalate precedence. | OPA data-driven conflict tests. | Conflict matrix, final reason list, bundle revision, no accidental allow. | NOT_RUN |
| VS2-SEC-050 | MUST_PASS | ER-02, ER-05, ER-10 | The same policy input is evaluated at gateway, service, tool runtime, and native CLI. | All enforcement points evaluate it against the same active revision. | Materially equivalent decisions are produced; any mismatch fails closed, emits an anomaly, and cannot lead to a side effect. | Shared schema/client, revision pinning/correlation, mismatch guard. | Cross-enforcement conformance test and deliberate mismatch injection. | Decision comparison report, mismatch denial, anomaly audit/metric. | NOT_RUN |
| VS2-SEC-051 | MUST_PASS | ER-03, ER-05; IR-08 | A real tool/runtime process has no egress grant and a local egress-sink endpoint is reachable at the network layer. | The process attempts an outbound connection using supported HTTP and socket paths. | The connection is blocked by the runtime/network boundary, not merely skipped by application logic; the sink records zero requests/bytes. | Sandbox/network policy or enforced egress proxy, denial adapter. | Container/process integration test with local egress sink and packet/request counters. | Denied tool result, sink access log with zero requests, network counter, policy/audit refs. | NOT_RUN |
| VS2-SEC-052 | MUST_PASS | ER-02, ER-03, ER-08 | A tool/workflow has a declared ConnectorHub capability, tenant/workspace authorization, allowed mock destination, and required approval. | It calls the local mock provider through the governed egress path. | Exactly the declared call succeeds; evidence, policy, approval, connector result, and audit are linked; no undeclared call occurs. | ConnectorHub-mediated egress adapter, allow policy, mock provider. | End-to-end local integration test. | One sink request, sanitized request metadata, Action/Workflow/Connector result, decision/audit refs. | NOT_RUN |
| VS2-SEC-053 | MUST_PASS | ER-01, ER-03; IR-05 | Tenant A has an egress allow rule and tenant B does not. | Both tenants request the same destination/capability. | Tenant A follows its governed path; tenant B is denied; allowlist/cache/config does not bleed across tenants. | Tenant-scoped egress policy data and cache keys. | Two-tenant egress integration test. | A success/B denial, separate decisions, sink count exactly one. | NOT_RUN |
| VS2-SEC-054 | MUST_PASS | ER-03; IR-08 | An egress policy declares allowed scheme, hostname, port, method, path/capability, and data scope. | Calls vary each field and use URL normalization variants. | Only exact normalized permitted combinations pass; case, trailing-dot, encoded path, alternate port, wildcard, or method variations cannot broaden access. | Canonical URL/destination parser, structured allow rules. | Table-driven boundary tests. | Per-case decisions, sink logs, normalization output, zero unexpected calls. | NOT_RUN |
| VS2-SEC-055 | MUST_PASS | ER-03; IR-08 | A caller targets IP literals, loopback, private, link-local, multicast, Unix sockets, cloud metadata ranges, or a public name resolving to them. | The tool attempts connection without a purpose-specific internal capability. | The runtime denies the destination before sending data and records a safe reason. | Resolved-address validation, reserved-range policy, internal-capability model. | IPv4/IPv6 destination matrix against local fixtures. | Denied decisions, zero sink/metadata requests, sanitized destination class. | NOT_RUN |
| VS2-SEC-056 | MUST_PASS | ER-03; IR-08 | An allowed hostname resolves first to an allowed address and then to a denied/private address, or resolves to mixed addresses. | Connection is attempted across resolution/retry. | Resolved addresses are validated at connection time; denied addresses are never contacted; policy cannot be bypassed by DNS rebinding. | Resolver integration, address pinning/revalidation, connection guard. | Deterministic fake-DNS rebinding test. | DNS transcript, selected/blocked addresses, zero denied-address connections. | NOT_RUN |
| VS2-SEC-057 | MUST_PASS | ER-03; IR-08 | An allowed endpoint redirects to a denied host, scheme, port, or path. | The HTTP client follows redirects. | Every hop is re-authorized; the denied hop is blocked, redirect limits apply, and no sensitive header is forwarded across unauthorized boundaries. | Redirect-aware governed HTTP client. | Mock redirect-chain tests including loops and cross-host redirects. | Hop-by-hop decisions, sink logs, header redaction, bounded redirect count. | NOT_RUN |
| VS2-SEC-058 | MUST_PASS | ER-03, ER-05; IR-08 | A tool attempts to bypass governed HTTP using proxy environment variables, direct sockets, alternate DNS, WebSocket/FTP/SMTP, subprocess, or a bundled client. | The process runs in the VS2 sandbox. | Undeclared protocols and paths remain blocked at the network/capability boundary; no arbitrary shell/host privilege is gained. | Network namespace/proxy enforcement, environment sanitization, syscall/capability restrictions. | Adversarial sandbox integration suite. | Zero unauthorized connections/processes, denied capability records, runtime audit. | NOT_RUN |
| VS2-SEC-059 | MUST_PASS | ER-03, ER-12 | An Artifact, web page, connector payload, tool output, or prompt contains instructions to call a URL, exfiltrate data, change policy, or approve an action. | The content is processed by an agent/workflow. | It remains untrusted evidence; it cannot create authority, egress grants, policy changes, approvals, or tool/action execution. | Untrusted-content labeling, agent/tool boundary, policy gate. | Prompt-injection red-team fixtures across artifact and connector paths. | Zero tool/action/egress calls, blocked-attempt audit, untrusted label/evidence refs. | NOT_RUN |
| VS2-SEC-060 | MUST_PASS | ER-03, ER-08; IR-10 | An allowed external call needs provider authentication or contains sensitive payload fields. | The governed ConnectorHub path prepares and executes it. | Agents/product outputs never receive raw credentials; least data is sent; denied calls send nothing; logs/decisions/audit mask secrets and sensitive payload. | ConnectorHub credential custody, request projection/redaction, log masking. | Secret-canary integration tests for allowed and denied calls. | Sink metadata without raw secret, zero secret scanner findings, credential reference only. | NOT_RUN |
| VS2-SEC-061 | MUST_PASS | ER-02, ER-03, ER-13; IR-09 | An Action dry-run predicts external calls and is then approved or delayed. | Policy, allowlist, data, approval, or destination changes before execution. | Execution detects stale dry-run state, re-evaluates current policy and impact, and blocks or requires renewed approval; it never relies solely on the old allow. | Dry-run fingerprint, policy revision binding, execution recheck. | Allow-at-dry-run → revoke/change → execute test. | Dry-run and execution decision revisions, stale-state denial, zero sink calls. | NOT_RUN |
| VS2-SEC-062 | MUST_PASS | ER-03, ER-07; IR-09 | An allowed external operation times out, returns retryable failure, or receives duplicate execution requests. | Workflow retry logic runs. | Timeout/retry limits are bounded, idempotency is tenant-scoped, and duplicate real/mock side effects are prevented or explicitly compensated. | Workflow retry/timeout/idempotency, ConnectorHub action keys. | Mock provider fault and duplicate-request tests. | Attempt timeline, one side effect maximum, idempotency key scope, compensation/audit record. | NOT_RUN |
| VS2-SEC-063 | MUST_PASS | ER-03, ER-07; IR-10 | Allowed and denied egress attempts occur. | Operators inspect policy, connector, runtime, and CornerStone audit/metrics. | Records correlate tenant-safe trace, decision, action/workflow, connector capability, normalized destination class, outcome, and byte/call counts without storing secrets or raw payloads. | Egress audit schema, metrics, trace correlation, redaction. | Integration plus log/metric contract tests. | Correlated records, counter assertions, decision IDs, secret-scan report. | NOT_RUN |
| VS2-SEC-064 | MUST_PASS | ER-03, ER-05; IR-07, IR-08 | The egress controller/proxy/sandbox policy component is unavailable or misconfigured. | A tool attempts network, shell, filesystem, environment, or host access. | Protected capabilities fail closed; readiness reports the degraded component; no fallback direct connection or host access occurs. | Fail-closed sandbox wiring, readiness dependencies, no direct-client fallback. | Stop/disable controller and repeat capability tests. | Degraded readiness, denied operations, zero sink/host counters, operator guidance. | NOT_RUN |
| VS2-SEC-065 | MUST_PASS | ER-06, ER-10 | A major VS2 allow, deny, missing-context, policy-unavailable, RLS, and egress failure is exercised. | The behavior is observed through CLI JSON/text, HTTP API, and UI. | Machine-readable schemas, exit/status codes, user messages, evidence_refs, policy_decision_refs, and audit_refs remain semantically consistent. | CLI/API/UI response contracts and error catalog. | Golden contract tests and browser proof. | CLI/API payloads, screenshots/DOM snapshot, schema validation, matching references. | NOT_RUN |
| VS2-SEC-066 | MUST_PASS | ER-07 | VS2 generates access, denial, policy update, RLS anomaly, egress, approval, connector, and rollback events. | Audit verification and deliberate tampering run. | Required events are append-only/tamper-evident, queryable by tenant/action/decision, and modification/removal is detected. | Audit event contract, hash-chain/checkpoint integration, verifier. | Audit contract tests and tamper fixture. | Event inventory, verification success, tamper detection failure evidence. | NOT_RUN |
| VS2-SEC-067 | MUST_PASS | ER-07; IR-10 | An operator monitors Postgres, OPA, egress control, policy denials, RLS anomalies, retries, and audit integrity. | Status, metrics, logs, and traces are queried. | Health/readiness distinguishes component states, metrics are actionable and tenant-safe, and alerts can identify failure without exposing protected data. | Health/readiness, metrics, trace IDs, operator status surface. | Component fault-injection and observability contract tests. | Status JSON/dashboard snapshot, metrics samples, alert events, leak scan. | NOT_RUN |
| VS2-SEC-068 | MUST_PASS | ER-01, ER-02; IR-11 | VS2 database/policy/runtime changes are deployed over the existing local scaffold and VS1 data. | Forward migration, compatibility run, failed migration, and rollback are exercised. | The smallest compatible adapter path preserves existing artifacts/evidence/claims/ontology/audit; rollback is deterministic; no destructive migration runs without approval. | Postgres store adapter, migration versions, compatibility layer, rollback scripts. | Fresh and upgrade-path integration tests with checksums and existing VS0/VS1 fixtures. | Before/after object counts/hashes, regression reports, rollback transcript, no data loss. | NOT_RUN |
| VS2-SEC-069 | MUST_PASS | ER-01, ER-02, ER-03, ER-10 | A new evaluator uses the documented local/on-prem VS2 quickstart without live provider credentials. | They start CornerStone, Postgres, OPA, and the local egress test harness. | Readiness succeeds only when required components and active policy/RLS inventory are valid; a tenant-isolation and egress-deny smoke flow completes. | Compose/local profile, migrations, seed policy, health/readiness, quickstart verifier. | Executable clean-environment quickstart. | Command transcript with exit codes, component versions, policy revision, RLS inventory, generated IDs, elapsed time. | NOT_RUN |
| VS2-SEC-070 | MUST_PASS | ER-07, ER-10, ER-11 | A feature/milestone report claims VS2 completion. | The native scenario verifier and gate run against the exact code/tree state. | All applicable MUST_PASS and REGRESSION rows have concrete evidence; human rows are explicit; report metadata binds to verified revision/tree; no production/live overclaim is present. | Native `cornerstone scenario verify vs2-policy-tenancy-egress`, gate, Make target, release evidence package. | Scenario report schema validation, gate command, report review. | Machine-readable report, command transcript, evidence manifest/hashes, base/final revision metadata, verdict. | NOT_RUN |
| VS2-SEC-R01 | REGRESSION | ER-11; CS-REG-001, CS-REG-002 | VS2 changes are applied. | The existing VS0 Artifact→Search→Evidence→Claim→Action→Audit gates run. | The VS0 local product loop remains green and evidence/action safety is unchanged. | Compatibility adapters and regression suite. | Run existing VS0 EVUX/runtime/operator gates. | Existing scenario reports with exit code 0 and no weakened assertions. | NOT_RUN |
| VS2-SEC-R02 | REGRESSION | ER-11; CS-REG-016, CS-REG-018 | VS2 changes are applied over the accepted local VS1 ontology slice. | The VS1 suggest/review/promote suite runs. | Ontology suggestions remain draft until explicit promotion; search/profile/claim/action/audit integration remains green under tenant policy. | VS1 data adapter and policy-aware ontology services. | Run `make verify-vs1-ontology` plus cross-tenant ontology tests. | VS1 report, zero auto-promotions/cross-tenant promotions, policy refs. | NOT_RUN |
| VS2-SEC-R03 | REGRESSION | CS-NS-003, CS-REG-006 | Personal context exists in a user's personal namespace. | An organization request is answered or acted on without explicit promotion/permission. | Personal context is not used, cited, searched, or exposed. | Namespace policy and retrieval guards. | Known-canary personal→organization leak test. | Zero personal refs/canary, denial or insufficient-evidence response. | NOT_RUN |
| VS2-SEC-R04 | REGRESSION | CS-REG-007 | Organization context exists and the user opens a personal workspace. | A related personal query/action runs without an allowed reference. | Organization context is not used or exposed. | Reverse namespace guards. | Organization→personal canary leak test. | Zero organization refs/canary, policy/audit outcome. | NOT_RUN |
| VS2-SEC-R05 | REGRESSION | CS-NS-008, CS-REG-008 | Product-learning or benchmark processes run. | They request raw personal/organization truth without opt-in/approved redaction. | Access is denied and no hidden memory/truth rewrite occurs. | Product-learning namespace policy. | Policy/data-access test. | Denied decision, zero source rows and memory writes. | NOT_RUN |
| VS2-SEC-R06 | REGRESSION | CS-REG-013, CS-SEC-007 | A prompt/document claims the agent now has admin, tenant, connector, tool, or egress authority. | The agent processes it. | Authority remains defined by trusted role, workspace, mission, connector capability, and policy state. | Prompt-injection and authority guards. | Red-team fixture suite. | Zero authority changes/tool calls/egress, blocked-attempt audit. | NOT_RUN |
| VS2-SEC-R07 | REGRESSION | CS-EXT-014, CS-EXT-015, CS-REG-014 | An Agent Pack or product handler tries to use provider credentials or direct provider networking. | Validation/runtime executes. | The pack/path is blocked or quarantined; ConnectorHub remains the provider boundary. | Pack validation, import/runtime restrictions, egress policy. | Static and runtime direct-provider tests. | Validation failure/quarantine, zero credential/network exposure. | NOT_RUN |
| VS2-SEC-R08 | REGRESSION | ER-05, CS-AUTO-020 | A developer adds a direct repository/database/writeback path that skips Workflow/Action or service policy. | CI and integration tests run. | The bypass is rejected by architecture lint, service guard, RLS, or runtime enforcement and cannot mutate state. | Boundary lint, repository interfaces, RLS. | Deliberate bypass fixture/test. | Gate failure or runtime denial, unchanged state, audit/anomaly evidence. | NOT_RUN |
| VS2-SEC-R09 | REGRESSION | CS-REG-017 | A new tool, model, workflow, policy entrypoint, connector capability, or table is introduced. | Audit contract tests run. | All required critical events and references remain present; coverage cannot silently drop. | Audit event manifest and coverage gate. | Mutation test that omits an event. | Failing omission proof, passing corrected event inventory. | NOT_RUN |
| VS2-SEC-R10 | REGRESSION | CS-REG-018 | Configuration is missing, reset, partially upgraded, or a new workspace/tenant is created. | Security defaults initialize. | Egress is denied, policy is default-deny, app role cannot bypass RLS, high-risk actions require approval, and shell/host access remains blocked. | Secure defaults and bootstrap validation. | Fresh/reset/partial-config integration tests. | Default config snapshot, denial suite, RLS/role inventory. | NOT_RUN |
| VS2-SEC-R11 | REGRESSION | IR-07, CS-REG-018 | Policy or membership is revoked after an allow was cached. | Requests continue during cache/bundle transition. | No stale allow produces access or side effect beyond the frozen revocation bound. | Cache and revision guard. | High-concurrency revocation test. | Decision revision sequence, zero post-revocation successes. | NOT_RUN |
| VS2-SEC-R12 | REGRESSION | CS-SEC-008 | Cross-tenant, malformed, secret-bearing, and denied requests generate errors/logs/traces/screenshots/reports. | Leak scanners and snapshot tests run. | No secret, protected payload, raw credential, or foreign-tenant identifier appears unnecessarily. | Central redaction and safe error schemas. | Canary/secret scanner across generated outputs. | Zero scanner findings or reviewed allowlist with justification. | NOT_RUN |
| VS2-SEC-R13 | REGRESSION | IR-04, CS-REG-018 | A new durable tenant-bearing table or partition is added without RLS/policy coverage. | Schema CI runs. | The build fails before merge/release. | Schema/RLS inventory gate. | Negative migration fixture. | Expected failing gate and remediation evidence. | NOT_RUN |
| VS2-SEC-R14 | REGRESSION | IR-02 | Tenant tests run in parallel or retry after failures. | Test fixtures, pools, caches, queues, and reports share infrastructure. | Test state remains isolated; one test/tenant cannot contaminate another or create false PASS. | Unique test tenant IDs, cleanup, deterministic concurrency harness. | Parallel/repeat test execution. | Repeatable report hashes/semantics, zero cross-fixture canaries. | NOT_RUN |
| VS2-SEC-R15 | REGRESSION | CS-PROD-001, CS-REG-019 | Policy, tenant, and connector controls are added. | A normal user navigates the product. | Users still experience one CornerStone product; repo/engine names are not required mental models, while admin security detail remains available. | Product navigation and progressive-disclosure UX. | Automated nav/DOM check plus human review in H06. | UI map/screenshots, absence of required repo-name workflow. | NOT_RUN |
| VS2-SEC-R16 | REGRESSION | CS-SEC-020, CS-REG-020 | Reports, README, UI, or release metadata describe VS2. | A reviewer compares claims with evidence. | Local/integration proof is not described as production, live-provider, penetration-tested, or human-accepted without separate evidence. | Claim vocabulary guard and release-report validator. | Static/report lint and evidence manifest review. | Zero overclaim findings; explicit scope/human-required table. | NOT_RUN |
| VS2-SEC-H01 | HUMAN_REQUIRED | Project stop-and-ask gate | VS2 proposes an actual OPA production dependency, Postgres/RLS tenant model, authz semantics, and migration of durable state. | Implementation is ready to start. | JiYong/Tars or authorized owner approves the architecture, dependency, migration scope, rollback plan, and security ownership before code changes that cross the gate. | Architecture/ADR and approval workflow. | Human architecture/security review. | Dated APPROVE/REJECT record with approved scope, exceptions, rollback owner, and dependency decision. | HUMAN_REQUIRED |
| VS2-SEC-H02 | HUMAN_REQUIRED | ER-01, ER-02; production gate | A production-like deployment contains multiple real or representative tenants. | An independent security reviewer performs tenant-isolation and policy-bypass testing. | No cross-tenant data/metadata leak or policy bypass is found, or findings are remediated and re-tested. | Production-like environment and security test plan. | Independent penetration/security review. | Signed report, test scope/topology, findings, remediation, re-test evidence. | HUMAN_REQUIRED |
| VS2-SEC-H03 | HUMAN_REQUIRED | ER-04; production authn gate | A real OIDC/SSO or enterprise identity provider is selected. | Groups, roles, attributes, revocation, and tenant memberships are mapped. | Authorized stakeholders confirm mappings and revocation behavior; no synthetic-fixture claim is promoted to real IdP readiness. | Identity provider integration and mapping policy. | Approved real-IdP test. | Redacted login/mapping/revocation transcript and owner/security approval. | HUMAN_REQUIRED |
| VS2-SEC-H04 | HUMAN_REQUIRED | ER-03; production network gate | The target production/on-prem network topology, DNS, proxy, firewall, service mesh, and sandbox are available. | A network/security operator verifies deny-by-default and approved exceptions outside the local harness. | All undeclared paths are blocked and approved paths are traceable under the real topology. | Production network controls. | Network control review and egress test. | Topology diagram, firewall/proxy/network-policy evidence, packet/log transcript, approval. | HUMAN_REQUIRED |
| VS2-SEC-H05 | HUMAN_REQUIRED | ER-08; live provider gate | Authorized live ConnectorHub credentials and provider permissions are available. | A low-risk read-only or explicitly approved test action is executed. | Credential custody, declared capability, source policy, product approval, egress, result, evidence, and audit boundaries hold without secret exposure. | Live ConnectorHub integration. | Approved live-provider rehearsal. | Redacted provider transcript, approval, ConnectorHub/CornerStone audit refs, result/evidence refs. | HUMAN_REQUIRED |
| VS2-SEC-H06 | HUMAN_REQUIRED | ER-06, CS-SEC-005, CS-REG-019 | A human operator uses tenant/workspace switching, denial guidance, policy admin, egress dry-run, approval, and audit surfaces. | The operator completes representative tasks. | They accept or reject whether boundaries, causes, risks, and recovery paths are understandable and not misleading. | Admin/operator UX. | Human usability/trust review. | ACCEPT/REJECT note, screenshots/recording, task outcomes, confusion/issues list. | HUMAN_REQUIRED |
| VS2-SEC-H07 | HUMAN_REQUIRED | IR-11; production migration gate | Real production-like data, backup tooling, retention rules, and rollback authority are available. | The approved VS2 migration and rollback/restore drill runs. | Data/evidence/audit integrity, tenant ownership, RLS policies, and service recovery meet the approved plan. | Production migration/backup/restore process. | Human-supervised non-destructive rehearsal. | Signed transcript, before/after counts/hashes, RLS/policy inventory, restore/rollback result. | HUMAN_REQUIRED |

# Implementation Plan

The primary scenario table maps every scenario to a minimum implementation area. Execute the following phases in order, with each phase independently verifiable.

| Phase | Scenario Mapping | Smallest Independently Verifiable Implementation | Phase Evidence |
| --- | --- | --- | --- |
| P0 — Approval and freeze | H01; all | Adopt this contract; record owner decision on actual OPA dependency, Postgres/RLS migration, authz ownership, compatibility, rollback, and evidence paths. No sensitive implementation starts before approval. | Approved ADR/decision plus frozen scenario IDs. |
| P1 — Shared security contracts | 001–006, 026, 047–050, 065 | Add versioned `RequestContext`, `PolicyInput`, `PolicyDecision`, reason/error catalog, trusted attribute-source map, and shared CLI/API/UI adapters. Reject caller-controlled authoritative scope. | Unit/schema/golden tests and cross-surface normalized transcripts. |
| P2 — Postgres security substrate | 007–025, 036, 068 | Introduce a Postgres runtime adapter for the active VS0/VS1 durable slice; separate owner/migration/app/maintenance roles; add tenant/owner/namespace constraints, RLS policies, FORCE RLS where applicable, tenant-aware keys, migration/quarantine, pool hooks, inventory gate, and backup/export checks. | Real two-tenant DB integration report, role/RLS inventory, migration and rollback evidence. |
| P3 — OPA/Rego control plane | 026–050 | Run an actual OPA process in the local profile; implement Rego entrypoints for data, action, model, connector, tool, egress, and policy administration; add strict schemas, tests, bundles, status, decision logging/masking, cache/revocation, and fail-closed behavior. | OPA test/coverage JSON, bundle build/activation/failure transcripts, decision logs and status. |
| P4 — Egress and capability enforcement | 035, 051–064 | Route tool/provider networking through an enforced sandbox/proxy/network boundary; default deny; validate normalized destinations and resolved addresses; re-authorize redirects; sanitize environment; prevent direct sockets/protocol bypass; use a controlled local sink/DNS/redirect fixture. | Network/sink/packet proof with zero denied calls and exactly expected allowed calls. |
| P5 — Workflow and ConnectorHub handoff | 037–039, 052, 060–063 | Extend ActionCard/Workflow and ConnectorHub envelopes with tenant/namespace, policy decision/revision, approval, evidence, idempotency, expected external calls, data scope, execution result, and audit refs. Keep credentials/provider clients inside ConnectorHub. | End-to-end local mock action transcript and secret-scanner proof. |
| P6 — Audit, observability, and UX | 045–047, 063, 065–067 | Mirror OPA/runtime/connector decisions into CornerStone audit; preserve tamper evidence; expose helpful denial/status surfaces; add tenant-safe metrics/traces and progressive disclosure. | Audit contract/tamper tests, status/metrics report, browser/CLI/API proof. |
| P7 — Native verification and regressions | 069–070, R01–R16 | Add `cornerstone scenario verify vs2-policy-tenancy-egress`, a scenario gate, Make target, deterministic fixtures, negative counters, exact revision metadata, and rerun VS0/VS1 gates. | Machine-readable scenario report, evidence manifest, command transcript, all regression reports. |
| P8 — Human and production gates | H01–H07 | Run only after approvals/environment availability. Keep local/integration, product-accepted, live-provider, and production-ready claims separate. | Signed human/security/network/IdP/provider/UX/migration evidence. |

## Canonical Contracts to Add

The smallest coherent VS2 implementation should introduce versioned contracts equivalent to:

```text
TrustedPrincipal
MembershipSnapshot
RequestContext
PolicyInput
PolicyDecision
PolicyReason
PolicyBundleRevision
TenantScope
NamespaceScope
ConnectorCapabilityRequest
EgressIntent
ActionDryRunFingerprint
SecurityAuditEvent
```

Every contract that can influence access or execution must carry explicit scope, provenance/revision, timestamps, and audit correlation.

## Persistence and Migration Strategy

1. Preserve existing originals/evidence before migration.
2. Add a Postgres-backed runtime adapter behind an interface; retain the current local scaffold only as a compatibility/test mode.
3. Create separate migration-owner, application, and maintenance roles.
4. Add tenant/namespace/owner columns and constraints before enabling app writes.
5. Backfill only deterministic ownership; quarantine unknown/ambiguous rows.
6. Enable and force RLS as required; create command-specific `USING` and `WITH CHECK` policies.
7. Validate every active table, view, function, partition, index, FK, and unique constraint.
8. Cut reads/writes to Postgres only after two-tenant and rollback evidence passes.
9. Keep a reversible migration and exact before/after counts/checksums.

## Policy Strategy

1. Define the versioned input/output schemas before Rego.
2. Implement default deny and conservative aggregation.
3. Add operation-family entrypoints for data, action/workflow, model routing, connector capability, tool/runtime, egress, and sensitive administration.
4. Test all entrypoints with `opa test`, fail on an empty test set, and publish machine-readable coverage.
5. Activate versioned bundles atomically; retain or persist a last-known-good revision.
6. Bind/protect the OPA API; do not expose unauthenticated management/data APIs.
7. Record decision IDs, bundle revisions, safe reasons, and masked logs.
8. Fail closed on undefined, malformed, unavailable, stale, or mismatched decisions.

## Egress Strategy

1. Use a mandatory runtime/network boundary, not application-only URL branching.
2. Route permitted provider access through ConnectorHub-mediated capabilities.
3. Normalize and validate scheme, hostname, port, method, path/capability, data scope, resolved address, and every redirect hop.
4. Deny private/reserved/metadata destinations unless an explicit purpose-specific internal capability permits them.
5. Sanitize proxy/environment state and prevent direct socket/protocol/subprocess bypass.
6. Re-evaluate current policy, approval, destination, and dry-run fingerprint immediately before execution.
7. Prove denial using a controlled local sink with zero observed requests/bytes.
8. Prove the allowed path with exactly one expected local mock call and complete evidence/audit linkage.

# Verification Plan

## Verification Layers

| Verification Layer | Required Checks | Scenario Coverage |
| --- | --- | --- |
| Policy unit | Rego data-driven allow/deny/escalate/malformed/conflict tests; `--fail-on-empty`; machine-readable coverage. | 040–050 |
| Domain/unit | Request context, schema, reason mapping, tenant-scoped keys, URL normalization, redaction, dry-run fingerprints. | 001–006, 024, 026, 047–061 |
| Postgres integration | Real app role, RLS SELECT/INSERT/UPDATE/DELETE, joins/search, pool reuse, workers, views/functions, constraints, migrations, backup/restore. | 007–025, 036, 068 |
| OPA integration | REST decision calls, secured interface, active revision, atomic update, invalid bundle, timeout/unavailable, decision logs/status. | 032–050 |
| Runtime/network | Local egress sink, fake DNS, redirect chain, direct sockets/protocols, proxy env, controller outage, capability sandbox. | 035, 051–064 |
| API/CLI/UI | Same principal and operation across surfaces; stable schemas/codes/references; browser proof of helpful denials and admin/status surfaces. | 001, 047, 065, 069 |
| End-to-end | Tenant-scoped Artifact/Search/Claim/Ontology/Action/Connector/Audit flow with one allowed mock egress and denied adversarial paths. | 010–012, 037–039, 052, 060–070 |
| Regression | Existing VS0/VS1 gates, namespace leak canaries, prompt injection, provider bypass, audit coverage, secure reset, overclaim lint. | R01–R16 |
| Human/external | Architecture approval, production security/network/IdP/provider/migration review, subjective UX acceptance. | H01–H07 |

## Target Commands

These are **proposed implementation targets**. They were not run while authoring this contract and may require path/name adjustment after the repository layout is frozen.

```bash
# Proposed commands; these do not exist or have not been run yet.
docker compose -f compose.vs2.yml config
docker compose -f compose.vs2.yml up -d postgres opa egress-sink fake-dns

opa test policies/vs2 --fail-on-empty --coverage --format=json       > reports/policy/vs2-opa-test.json

python -m pytest       tests/vs2/test_request_context.py       tests/vs2/test_postgres_rls.py       tests/vs2/test_policy_integration.py       tests/vs2/test_egress_runtime.py       tests/vs2/test_vs2_api_cli_ui.py

PATH="$PWD:$PATH" cornerstone scenario verify       vs2-policy-tenancy-egress       --json       --output reports/scenario/vs2-policy-tenancy-egress-YYYY-MM-DD.json

PATH="$PWD:$PATH" cornerstone scenario gate       reports/scenario/vs2-policy-tenancy-egress-YYYY-MM-DD.json       --json

make verify-vs0-evux
make verify-vs0-operator-ui
make verify-vs1-ontology
make verify-vs2-security
```

## Postgres State Checks

```sql
-- Proposed evidence query: protected table and policy inventory.
SELECT
  n.nspname AS schema_name,
  c.relname AS relation_name,
  c.relrowsecurity,
  c.relforcerowsecurity,
  pg_get_userbyid(c.relowner) AS owner
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p', 'v', 'm')
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY 1, 2;

SELECT rolname, rolsuper, rolbypassrls
FROM pg_roles
WHERE rolname IN ('cornerstone_app', 'cornerstone_migrator', 'cornerstone_maintenance');

SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual, with_check
FROM pg_policies
ORDER BY schemaname, tablename, policyname;
```

## Required Evidence Manifest

| Evidence Class | Required Artifact |
| --- | --- |
| Scenario report | `reports/scenario/vs2-policy-tenancy-egress-YYYY-MM-DD.json` |
| Command transcript | `reports/release/vs2-policy-tenancy-egress-YYYY-MM-DD/command-transcript.json` |
| Postgres RLS inventory | `reports/db/vs2-rls-inventory.json` |
| Postgres two-tenant proof | `reports/db/vs2-tenant-isolation.json` |
| Migration/rollback proof | `reports/db/vs2-migration-rollback.json` |
| OPA tests and coverage | `reports/policy/vs2-opa-test.json`, `vs2-opa-coverage.json` |
| OPA bundle/status proof | `reports/policy/vs2-bundle-lifecycle.json` |
| Egress network proof | `reports/network/vs2-egress-proof.json` plus sink/DNS/redirect logs |
| Secret/leak scan | `reports/security/vs2-output-leak-scan.json` |
| API/CLI/browser proof | `reports/browser/vs2-policy-tenancy-egress/` and normalized API/CLI transcripts |
| Audit integrity proof | `reports/audit/vs2-audit-integrity.json` |
| Regression reports | Existing VS0/VS1 reports generated against the same final tree |
| Human evidence | `reports/release/vs2-policy-tenancy-egress/human-*.md` as applicable |

## Automatically Verifiable

All `MUST_PASS` and `REGRESSION` scenarios are intended to be AI-verifiable in a local integration environment with PostgreSQL, OPA, and deterministic mock network fixtures. Unavailable dependencies or skipped checks remain `NOT_RUN`/`NOT_VERIFIED`; they do not become human items merely because they are difficult.

## Human Required

`VS2-SEC-H01` through `VS2-SEC-H07` require owner, security, infrastructure, IdP, live-provider, migration, or subjective UX evidence. Their required actions and evidence are defined in the scenario table.

## Not Yet Verifiable

At document creation time:

- no VS2 implementation was performed;
- no Postgres RLS migration or role inventory was run;
- no OPA/Rego bundle was built or tested;
- no real runtime/network egress attempt was executed;
- no VS2 CLI/API/UI verifier exists;
- no VS2 regression suite was run;
- all AI-verifiable rows remain `NOT_RUN`.

# Completion Criteria

1. Every one of the 70 `MUST_PASS` rows is `PASS` with concrete evidence bound to the exact verified code/tree and environment.
2. Every one of the 16 `REGRESSION` rows is `PASS` with concrete evidence.
3. No AI-verifiable row is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN` when claiming AI-verifiable VS2 completion.
4. Every `PASS` includes an executed verification method and artifact; design text, source review, mocks without boundary proof, or narrative confidence are insufficient.
5. The Postgres gate covers every active tenant-bearing durable relation, not a hand-selected example table.
6. The egress gate includes real blocked connection attempts at the sandbox/network boundary and zero-call negative evidence from a controlled sink.
7. The policy gate uses a real OPA/Rego process and records decision IDs and active bundle revisions; the existing deterministic local evaluator alone is insufficient.
8. VS0 and VS1 regression reports are regenerated against the same final tree.
9. H01 approval is required before sensitive implementation begins. H02–H07 remain explicit and block only the corresponding production/live/human claim unless an approved release policy says otherwise.
10. The final verdict is no stronger than the weakest required scenario and must distinguish local integration, product acceptance, live-provider readiness, and production readiness.

## Claim Levels

- **VS2 scenario contract complete:** this document and matrix are reviewed/frozen.
- **VS2 AI-verifiably complete:** all 70 MUST_PASS and 16 REGRESSION rows pass locally with exact-tree evidence.
- **VS2 product-accepted:** AI-verifiable completion plus accepted H06 UX/trust review.
- **VS2 live-provider ready:** AI-verifiable completion plus H05.
- **VS2 production security ready:** AI-verifiable completion plus applicable H01–H07 and a release decision.
- **VS2 done:** only the strongest claim whose prerequisites all have evidence.

# Risks

| Risk ID | Risk | Mitigation / Scenario Coverage |
| --- | --- | --- |
| R-01 | RLS bypass through owner/superuser/BYPASSRLS, security-definer functions, or unsafe maintenance credentials. | Separate roles; FORCE RLS; metadata gate; direct-bypass tests. |
| R-02 | Connection-pool, async-context, cache, or worker contamination leaks tenant state. | Transaction-local context; tenant-scoped keys; concurrency/failure injection. |
| R-03 | Unique/FK constraints, counts, errors, timing, metrics, or logs become metadata side channels. | Tenant-aware keys; neutral errors; canary/leak tests; redaction. |
| R-04 | Application-level policy gives false confidence while direct DB/network paths remain open. | Triple enforcement; RLS independent test; real sandbox/network sink proof. |
| R-05 | OPA outage, invalid bundle, stale cache, or undefined result creates accidental allow. | Default deny; last-known-good; fail-closed adapter; revision-aware cache; readiness. |
| R-06 | Egress allowlists are bypassed by DNS rebinding, redirects, proxy variables, direct sockets, alternate protocols, or private-address resolution. | Resolved-address validation; hop re-authorization; network boundary; adversarial suite. |
| R-07 | CornerStone duplicates ConnectorHub credential/provider logic or KnowledgeBase archive truth. | Enforce contract handoffs and ownership boundaries; architecture lint/review. |
| R-08 | Postgres migration breaks current local VS0/VS1 behavior or loses provenance/audit. | Adapter-first migration; checksums; quarantine; compatibility and rollback tests. |
| R-09 | Decision/audit logs expose secrets or protected tenant metadata. | OPA masking plus CornerStone redaction; output-wide secret/canary scanning. |
| R-10 | Local deterministic PASS is misrepresented as production or live-provider security. | Explicit claim taxonomy, human gates, release-report overclaim validator. |
| R-11 | Policy/RLS complexity harms first-run usability or normal-user navigation. | Progressive disclosure; helpful denials; preserve one-product UX; human H06. |
| R-12 | Adding actual OPA/Postgres/network dependencies broadens operational burden. | Owner approval, minimal local profile, health/readiness, pinned versions, rollback. |

# Open Questions

| ID | Open Question |
| --- | --- |
| OQ-01 | Will VS2 standardize on an actual OPA sidecar/service for local/on-prem, or permit an embedded/compiled Rego runtime? This contract assumes a real OPA process for acceptance. |
| OQ-02 | Which durable tables are currently authoritative when implementation starts, and which compatibility adapter owns migration from `LocalRuntimeStore`? The acceptance rule remains: every active tenant-bearing table is covered. |
| OQ-03 | What trusted local identity source supplies principal/membership context before OIDC/SSO exists? CLI flags may request a context but cannot be authoritative by themselves. |
| OQ-04 | Which runtime/network mechanism will enforce egress: sandbox network namespace, mandatory proxy, service mesh, or another capability boundary? Application-only URL checks are insufficient. |
| OQ-05 | What are the frozen revocation/cache TTL, policy input size/depth, redirect, retry, timeout, and payload limits? Scenario 048/057/062 require N-1/N/N+1 tests once configured. |
| OQ-06 | Who owns break-glass authorization and audit review, and which operations are permitted under it? |
| OQ-07 | Which classifications, risk levels, mission authorities, workspace modes, and policy reason codes are canonical for VS2? |
| OQ-08 | Does the first VS2 release require tenant-scoped object storage/signed URL enforcement, or only local content-addressed storage? Scenario 011 requires equivalent behavior in either case. |
| OQ-09 | What exact local performance/SLO budget must Postgres RLS and OPA meet? This document requires measurement but does not invent a threshold. |

# Tool / Process Evidence

## Inputs Inspected

- Uploaded test-scenario implementation prompt.
- Uploaded legacy SoT, Project Operating Constitution, and Scenario-First Agent Instruction.
- Current Cornerstone product SoT, scenario standard, root agent instructions, technical defaults, VS1 contract/report, Makefile, CLI/scenario code, and security/namespace reports.
- KnowledgeBase README/AGENTS.
- Connector-Hub README and internal CornerStone integration contract.
- Official PostgreSQL and OPA documentation.

## Current Behavior Reverse-Engineered

- The current Cornerstone CLI identifies itself as a `local_scaffold` and exposes caller-supplied scope arguments for deterministic local checks.
- Current security reports explicitly limit PASS to deterministic local policy/tenant behavior.
- The current egress verifier proves that no network call is attempted; it does not prove network-layer blocking.
- The current tenant/RBAC/ABAC verifier explicitly says it is not a production identity, tenant, RLS, or durable policy service.
- The active VS1 contract explicitly excludes production Postgres/RLS/OPA migration.
- The inspected Makefile provides targets through VS1 and no VS2 target.
- ConnectorHub's internal contract assigns provider credentials/execution to ConnectorHub and product policy/approval to CornerStone.
- KnowledgeBase documents Postgres Archive concepts but the inspected material does not provide VS2 RLS evidence.

## Artifacts Created

- `CornerStone_VS2_Policy_Tenancy_Egress_Test_Scenarios.md`
- `CornerStone_VS2_Scenario_Matrix.csv`
- `CornerStone_VS2_Scenario_Matrix.json`

## Commands / Checks Run

- Repository and document inspection through the GitHub connector.
- Official PostgreSQL/OPA documentation inspection.
- Structural validation of this matrix: unique IDs, allowed priorities, required fields, and counts.

## Checks Not Run

All runtime, database, OPA, network, API, CLI, UI, migration, regression, and human checks described by the scenario matrix.

# Failure Reverse Engineering

## Current Verification Gap

**Expected for VS2:** actual RLS, actual OPA/Rego, and actual runtime/network default-deny evidence.

**Observed:** deterministic local policy and namespace semantics, with reports explicitly limiting their claim.

**First missing enforcement layers:**

1. Postgres role/RLS enforcement;
2. OPA service/bundle enforcement;
3. runtime/network egress containment;
4. unified trusted identity/scope propagation;
5. dedicated VS2 verifier and evidence package.

**Root cause:** VS2 was defined in the legacy milestone plan but was not re-frozen as an active implementation contract after the product reset; current work intentionally stopped at local scaffold and VS1 scope.

**Fix:** adopt this contract, obtain H01 approval, implement phases P1–P7, and verify every scenario against the final tree.

**Re-verification:** run the proposed VS2 gate plus all VS0/VS1 regressions. Until then, every AI-verifiable scenario remains `NOT_RUN`.

# Verification Gaps

- 70 MUST_PASS: `NOT_RUN`.
- 16 REGRESSION: `NOT_RUN`.
- 7 HUMAN_REQUIRED: pending human/external evidence.
- No implementation status can be inferred from this contract.
- Existing local-scaffold PASS reports remain valuable baseline evidence but are insufficient for VS2 completion.

# Final Verdict

- **Contract-authoring scope:** done — a comprehensive, scenario-first VS2 contract and machine-readable matrix were produced.
- **VS2 AI-verifiable implementation scope:** needs-follow-up — all runtime scenarios are `NOT_RUN`.
- **Human/release gate:** blocked before sensitive implementation by `VS2-SEC-H01`; production/live/human claims additionally require applicable H02–H07.
- **Confidence:** 0.93 for scenario coverage and source alignment; lower confidence remains around unresolved architecture choices listed in Open Questions.

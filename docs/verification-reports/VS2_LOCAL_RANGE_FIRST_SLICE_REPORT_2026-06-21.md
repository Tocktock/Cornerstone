# VS2 Local Range First Slice Report - 2026-06-21

## Summary

The attached review rejected the previous VS2 local proof as too synthetic: reusable validators and generated evidence could not prove the real CornerStone path.

This update adds a first production-flow local range:

```text
host-side verifier
-> native cornerstone CLI subprocess
-> local gateway/API/browser-visible HTML
-> trusted session and membership resolver
-> stale/revoked membership retry through API, CLI, browser, service, and worker
-> signed worker envelope validation, quarantine, replay guard, and worker audit
-> tenant-scoped operation key collision, replay, suppression, and cursor/cache evidence
-> real OPA HTTP decision
-> revision-aware product policy-decision cache exercised against real `v1` and `v2` OPA decisions
-> high-concurrency post-revocation stale-allow retry across API, service, and worker
-> leak/output-safety regression over generated reports, denied responses, UI denial snapshots, and decision logs
-> namespace-leak regression for personal/organization workspace boundaries
-> ConnectorHub/direct-provider, direct-writeback bypass, and secure-default regressions
-> schema-security gate regression for unprotected tenant-bearing table fixtures
-> overclaim regression over README, current-state report, local first-slice report, and operator boundary metadata
-> break-glass maintenance guard with normal app denial, approval-only synthetic path, time-bound scope, counts-only output, and audit
-> raw product-learning guard denying personal/org truth writes and hidden memory synthesis without approval
-> policy limit matrix for bounded input size, JSON depth, role count, enum validation, and pre-OPA rejection
-> cross-entrypoint policy conformance for gateway, service, tool runtime, and native CLI revision mismatch fail-closed behavior
-> real PostgreSQL 16 migrations, RLS, app role, and audit rows
-> ConnectorHub-mediated external Action dry-run, approval, execution, and audit
-> governed egress proxy
-> local mock provider sink
-> Docker-network direct egress denial probe
-> local adversarial egress probes for exact destination parsing, reserved-address denial, DNS rebinding, redirect re-authorization, timeout/retry/idempotency, audit correlation, and fail-closed controller outage
-> real pg_dump/pg_restore into a second local PostgreSQL database
-> VS0/VS1-shaped upgrade-path fixture migration, failed migration, rollback, and no-data-loss check
-> append-only PostgreSQL audit hash-chain fixture, verifier role, and tamper probes
```

The local range, OPA CI proof, OPA bundle lifecycle proof, adversarial egress slice, namespace-leak regressions, ConnectorHub/direct-provider boundary regressions, schema-security gate regression, break-glass guard, product-learning guard, policy-limit matrix, policy-conformance matrix, leak/output-safety regression, overclaim regression, and fresh regression reruns verify all 86 AI-verifiable rows. The 7 remaining rows are external or human gates and stay `HUMAN_REQUIRED`.

## New Command

```bash
make verify-vs2-local-range
```

Equivalent direct command:

```bash
PATH="$PWD:$PATH" cornerstone security vs2-local-range --json
```

Primary generated evidence:

- `reports/security/vs2-local-range.json`
- `reports/security/vs2-local-range-command.json`
- `reports/network/vs2-egress-proof.json`
- `reports/policy/vs2-opa-test.json`
- `reports/policy/vs2-opa-coverage.json`
- `reports/security/vs2-regression-proof.json`
- `reports/vs2/regression/vs0-product-runtime.json`
- `reports/vs2/regression/vs0-runtime-acceptance.json`
- `reports/vs2/regression/vs0-evux.json`
- `reports/vs2/regression/vs0-operator-ui.json`
- `reports/vs2/regression/vs1-ontology.json`

## Current Scenario Result

Fresh `cornerstone scenario verify vs2-policy-tenancy-egress --json` now reports:

| Status | Count |
|---|---:|
| PASS | 86 |
| NOT_VERIFIED | 0 |
| HUMAN_REQUIRED | 7 |
| FAIL | 0 |
| NOT_RUN | 0 |

Verified rows:

- `VS2-SEC-001`
- `VS2-SEC-002`
- `VS2-SEC-003`
- `VS2-SEC-004`
- `VS2-SEC-005`
- `VS2-SEC-006`
- `VS2-SEC-007`
- `VS2-SEC-008`
- `VS2-SEC-009`
- `VS2-SEC-010`
- `VS2-SEC-011`
- `VS2-SEC-012`
- `VS2-SEC-013`
- `VS2-SEC-014`
- `VS2-SEC-015`
- `VS2-SEC-016`
- `VS2-SEC-017`
- `VS2-SEC-018`
- `VS2-SEC-019`
- `VS2-SEC-020`
- `VS2-SEC-021`
- `VS2-SEC-022`
- `VS2-SEC-023`
- `VS2-SEC-024`
- `VS2-SEC-025`
- `VS2-SEC-026`
- `VS2-SEC-027`
- `VS2-SEC-028`
- `VS2-SEC-029`
- `VS2-SEC-030`
- `VS2-SEC-031`
- `VS2-SEC-032`
- `VS2-SEC-033`
- `VS2-SEC-034`
- `VS2-SEC-035`
- `VS2-SEC-036`
- `VS2-SEC-037`
- `VS2-SEC-038`
- `VS2-SEC-039`
- `VS2-SEC-040`
- `VS2-SEC-041`
- `VS2-SEC-042`
- `VS2-SEC-043`
- `VS2-SEC-044`
- `VS2-SEC-045`
- `VS2-SEC-046`
- `VS2-SEC-047`
- `VS2-SEC-048`
- `VS2-SEC-049`
- `VS2-SEC-050`
- `VS2-SEC-051`
- `VS2-SEC-052`
- `VS2-SEC-053`
- `VS2-SEC-054`
- `VS2-SEC-055`
- `VS2-SEC-056`
- `VS2-SEC-057`
- `VS2-SEC-058`
- `VS2-SEC-059`
- `VS2-SEC-060`
- `VS2-SEC-061`
- `VS2-SEC-062`
- `VS2-SEC-063`
- `VS2-SEC-064`
- `VS2-SEC-065`
- `VS2-SEC-066`
- `VS2-SEC-067`
- `VS2-SEC-068`
- `VS2-SEC-069`
- `VS2-SEC-070`
- `VS2-SEC-R01`
- `VS2-SEC-R02`
- `VS2-SEC-R03`
- `VS2-SEC-R04`
- `VS2-SEC-R05`
- `VS2-SEC-R06`
- `VS2-SEC-R07`
- `VS2-SEC-R08`
- `VS2-SEC-R09`
- `VS2-SEC-R10`
- `VS2-SEC-R11`
- `VS2-SEC-R12`
- `VS2-SEC-R13`
- `VS2-SEC-R14`
- `VS2-SEC-R15`
- `VS2-SEC-R16`

## Evidence Boundary

The local range proves:

- CLI/API/browser-visible context parity for the first artifact-read flow.
- Caller-forged tenant/role fields are denied.
- Missing or bad sessions fail before DB or egress calls.
- Durable object contract rows for all active object tables are persisted and visible through API and CLI evidence.
- Required tenant/namespace/owner/workspace/classification fields are non-null where applicable; audit events are treated as classification-not-applicable.
- Failed-null inserts and cross-scope mutation attempts are denied for the object-contract rows.
- Tenant-A object access paths over tenant-mixed blobs, derived rows, evidence bundles, download probes, signed-URL probes, and evidence traversal probes are checked through API and CLI.
- Tenant-B guessed object/download/signed-URL/evidence traversal attempts return neutral not-found-or-denied responses, zero bytes, no signed URL, no storage access, no content, and no sensitive metadata; the storage access log contains only authorized tenant-A reads.
- Tenant-scoped observability paths over tenant-mixed audit events, policy decisions, operator metrics, status records, and exports are checked through API and CLI.
- Tenant user and tenant-admin fixtures can query only their authorized tenant-scoped observability records; system-wide export/status access is denied without explicit privileged policy and records a denial event.
- Tenant-scoped observability exports include only authorized row refs and payload hashes; aggregate metrics omit tenant IDs, principal IDs, tenant-B record IDs, and beta canaries.
- A single PostgreSQL backend connection is reused across tenant-alpha success, reset, tenant-beta success, duplicate-key error, statement-timeout cancellation, rollback, and reset paths.
- Transaction-local tenant context is reset after success, error, timeout, and rollback paths; tenant-beta reads on the reused connection see zero tenant-alpha rows, IDs, or canaries.
- Sixteen concurrent host-side HTTP requests exercise tenant-alpha and tenant-beta artifact reads through the same local gateway/API service, with deterministic staggered scheduling and interleaved completion.
- Concurrent tenant-alpha and tenant-beta reads each return their own context digest, artifact tenant, canary, policy trace ID, and persisted audit ref; no response contains the other tenant's artifact ID or canary.
- The concurrent audit transcript records 16 distinct PostgreSQL audit rows whose trace IDs and decision IDs match the originating request, with no alpha/beta audit-ref overlap.
- Signed worker envelopes are exercised with valid, missing-scope, tampered-scope, stale-revision, cross-tenant-payload, invalid-signature, and replay cases.
- The valid worker job completes once; missing, tampered, stale, cross-tenant, invalid-signature, and replay envelopes are quarantined or rejected with stable reasons and no egress.
- Worker job and audit records are persisted in PostgreSQL for stateful worker outcomes; replay creates a separate quarantine record without re-reading the protected artifact.
- Tenant-alpha and tenant-beta use identical cache, idempotency, dedupe, lock, rate-limit, and cursor operation keys without cross-tenant read visibility.
- Duplicate/replay insertion for tenant-alpha is rejected by the tenant-scoped key, without echoing tenant-beta identifiers; tenant-alpha update/delete attempts against tenant-beta operation keys affect zero rows.
- Per-tenant operation-key evidence contains only the requesting tenant's rows, operation keys, and canaries.
- A previously allowed principal is denied after stale session-version and membership revocation checks.
- Revocation is observed through API, CLI, browser-visible HTML, direct service invocation, and signed worker job quarantine.
- Revoked/stale retries record denial audit refs and do not return artifacts or perform new egress/provider calls.
- OPA produces a real allow/deny decision over the product request.
- PostgreSQL RLS hides tenant-B rows from tenant-A.
- Cross-tenant insert/update attempts are denied or affect zero rows.
- Tenant-A read paths over tenant-mixed durable tables are checked through API and CLI for select, counts, existence checks, groupings, joins, subqueries, boundary pagination, and guessed foreign IDs.
- Tenant-B rows, tenant IDs, beta object IDs, and beta canaries are absent from tenant-A read-matrix result snapshots, with neutral not-found-or-denied metadata for guessed foreign IDs.
- Tenant-A search paths over tenant-mixed search snapshots and ontology objects are checked through API and CLI for full-text search, suggestions, facets, saved snapshot refs, object refs, and semantic-cache refs.
- Tenant-B-only search terms return no result, snippet, score, suggestion, facet count, object ref, snapshot ref, or cache ref to tenant A; search index inventory and RLS state are captured.
- Raw SQL repository reads, a security-invoker view, and a security-invoker function are exercised through API and CLI while preserving tenant isolation.
- An intentionally unsafe security-definer artifact function is present in the database inventory but denied to the normal app role, with no foreign identifiers or canaries echoed.
- Tenant-scoped unique and foreign-key collision paths are exercised through API, CLI, and PostgreSQL: the same artifact key can exist independently in tenant-alpha and the control tenant, same-tenant duplicate insert fails with a neutral conflict, and cross-scope artifact reference insert fails through a tenant-aware composite foreign key.
- The artifact-reference table has RLS enabled and forced, uses composite tenant-aware foreign keys for source and target artifacts, and the request-scope collision evidence contains no control-tenant IDs, owners, workspace, or canaries.
- A host-side migration matrix runs through API and native CLI, asks real OPA before execution, seeds known, missing-tenant, ambiguous-owner, invalid-namespace, duplicate-ID, and cross-tenant-reference legacy fixtures, and migrates only the known rows into tenant-scoped product storage.
- Malformed, ambiguous, duplicate, and cross-tenant-reference fixture rows are written to `cs.migration_quarantine` with machine-readable reasons, digest-only legacy references, content checksums, and no ownerless global truth.
- The migration evidence includes product/quarantine checksums, request-scope product verification, schema/RLS inventory, and a rollback transaction that leaves no durable rollback fixture table behind.
- A host-side upgrade-path matrix runs through API and native CLI, asks real OPA before execution, seeds VS0/VS1-shaped Artifact, EvidenceBundle, Claim, ontology, search, and audit fixtures, and snapshots before/after counts and content hashes.
- The upgrade-path forward migration adds a compatibility column without changing fixture counts or content hashes; compatibility reads under RLS verify the preserved artifact, evidence, claim, ontology, search, and audit rows.
- A deliberately bad migration fails validation and leaves no bad constraint; rollback removes upgrade columns while preserving the original counts and hashes.
- A high-risk destructive migration request is denied by OPA without approval and no destructive database statement is attempted.
- A host-side audit-integrity matrix runs through API and native CLI, asks real OPA before verification, seeds the required access, denial, policy update, RLS anomaly, egress, approval, connector, rollback, and audit verification events into PostgreSQL, and verifies the clean hash chain through a separate `cornerstone_auditor` role.
- The normal app role can insert and scoped-read audit rows, but PostgreSQL denies direct update and delete attempts against audit events.
- Deliberate audit tamper fixtures for event modification, event deletion, fake insertion, event reordering, and previous-hash modification all fail deterministic hash-chain verification.
- A host-side schema-security gate creates a deliberately bad migration fixture and a corrected fixture inside rolled-back PostgreSQL transactions, then inventories tenant scope columns, RLS enabled/FORCE state, policies, app grants, and declared test coverage.
- The bad fixture fails on every required schema-security surface; the corrected fixture passes with machine-readable inventory, and rollback leaves no fixture tables behind.
- App/security roles are not superuser or `BYPASSRLS`; protected tables force RLS.
- Audit rows are persisted in PostgreSQL.
- A negative control that disables RLS reveals tenant-B data and is detected by the test.
- ActionCard/Workflow dry-run, approval, execution, policy recheck, and audit linkage are observed.
- A declared ConnectorHub capability reaches the local mock provider exactly once through the governed proxy.
- Tenant A egress succeeds while tenant B is denied for the same provider/capability without another provider call.
- Provider access uses a credential reference only; raw credentials are not exposed to product output or the provider request.
- A stale dry-run probe is denied without sending an extra provider request.
- Protected API, worker, and tool-runtime containers on the service network cannot reach a provider-only network over HTTP or raw sockets, while the governed proxy can reach that same provider.
- Exact egress destination variations for host, port, scheme, method, path-prefix shadowing, and encoded traversal are attempted through the governed client; all non-declared normalized variants are denied before network and the declared sink receives exactly one request.
- Reserved egress destinations for loopback literal, localhost, cloud metadata IPv4, private IPv4, IPv6 loopback, and multicast IPv4 are attempted; all are denied before network and the trap sink records zero requests.
- A deterministic fake-DNS rebinding fixture exercises allowed-then-loopback, public-plus-metadata, and public-plus-IPv6-loopback answer sets; reserved answers are denied with zero denied-address connections.
- Redirect probes exercise a permitted redirect source that points at a denied target plus a bounded redirect loop; every hop is re-authorized, the denied target is not contacted, and no sensitive header reaches the trap sink.
- Sandbox bypass probes attempt proxy environment variables, direct raw socket, alternate DNS, WebSocket, FTP, SMTP, subprocess curl, bundled HTTP client, shell, and host filesystem access through the local capability boundary; every attempt is denied before network/process/host execution, trap counters stay unchanged, and denied capability audit records are emitted.
- Prompt-injection fixtures for artifact, web page, connector payload, tool output, and prompt sources are processed as untrusted evidence; every embedded URL, policy-change, approval, tool, egress-grant, and authority-grant instruction is attempted and denied before network or mutation, with retained evidence refs and blocked-attempt audit records.
- A regression fixture where an untrusted document claims admin, tenant, connector, tool, and egress authority is processed; every authority claim is attempted, denied before mutation, and audited while trusted authority remains bound to request context, policy, and approval state.
- A flaky local provider exercises a timed-out operation, a retryable `503 -> 200` operation, and a duplicate request with the same tenant-scoped idempotency key; retry count is bounded, the duplicate is suppressed before a second network call, and side effects are limited to one per tenant-scoped key.
- Egress audit records now correlate tenant-safe trace ID, decision ID, connector capability, destination class, outcome, byte count, and call count for allowed, denied, redirect-denied, and controller-outage attempts without raw payloads or raw credentials.
- A local egress-controller outage probe denies the operation before network, reports degraded readiness, and records zero direct-client fallback calls to the provider sink.
- A privileged local backup is produced with `pg_dump`, restored into a second PostgreSQL container, and then rechecked for row counts, policies, RLS, audit hashes, and tenant-scoped export behavior.
- A same-tenant personal/organization fixture exercises implicit cross-namespace access denial, RLS-hidden organization rows from the personal workspace, and explicit promotion into the organization workspace with provenance and audit refs.
- Cross-tenant copy, reference, share, and promotion attempts are executed through the app role; OPA denies the target tenant, every DB transfer attempt fails, zero target records are created, and the denial is audited.
- A fault-injected service-allow path attempts a tenant-B read through `cornerstone_app`; Postgres RLS still returns zero foreign rows and records anomaly audit plus tenant-scoped operator metric evidence.
- OPA CI now records real `opa test --fail-on-empty --coverage --format=json`, detailed `opa test --fail-on-empty --format=json`, `opa check`, `opa build`, and empty-test negative-control command transcripts; the coverage report is machine-readable, required policy entrypoints are reviewed, and named positive, negative, malformed, boundary, and regression policy tests are observed.
- OPA bundle lifecycle proof now builds versioned `v1` and `v3` bundles with real `opa build --revision`, starts OPA with a local bundle server, observes 100 concurrent `v1` decisions, publishes a malformed `v2` bundle and observes `bundle_error` while `v1` remains active, starts OPA from a malformed bundle and observes fail-closed undefined decisions, then publishes valid `v3` while requests continue and observes only known `v1`/`v3` revisions with all post-activation decisions on `v3`.
- A versioned `cs.policy_input.vs2.v1` schema artifact exists at `config/vs2/policy_input_schema.v1.json`; gateway, service, tool-runtime, action-card, connector, model-router, policy-admin, and memory PolicyInput builders validate against it, valid cases call real local OPA, malformed fixtures are rejected before OPA, and input digests/source-of-attribute maps are retained.
- A real `data.system.log.mask` Rego policy exists at `policies/vs2/system_log_mask.rego`; a canary-bearing OPA decision request is collected only after masking, the collector entry has no canary, and the masked decision log remains linked to policy decision and CornerStone audit refs.
- A versioned reason-code catalog exists at `config/vs2/reason_code_catalog.v1.json`; data, tool, action, connector, model, egress, and policy-admin denial decisions translate into stable safe responses, browser-style denial UI snapshots, matching decision/audit refs, and no protected data echo.
- A real OPA enforcement matrix exercises low-risk allow with downstream read/audit, role-denied write with no non-audit side effects, ABAC boundary denies plus a matching allowed set, malformed/wrong-version/over-specified input fail-closed cases, OPA timeout/connection-refused/500/malformed/undefined/revision-mismatch fail-closed cases, deny-precedence conflict cases, and unknown policy-path default deny with stable safe denial responses.
- A revision-aware product policy-decision cache is exercised through the local gateway path: the initial artifact-read allow is obtained from real `v1` OPA, the same-revision retry hits the cache, a `v2` OPA revision update changes the cache key and forces a real denial instead of reusing the stale allow, the legacy no-revision key is shown to collide as a negative control, tenant/principal/resource/action/revision key material is recorded, and audit refs are persisted.
- A post-revocation stale-allow regression probe starts from a cached real OPA allow, records same-revision cache reuse, applies a `v2` denial revision and membership revocation, then runs 24 concurrent API, direct service, and signed-worker retries; every retry is denied or quarantined with zero artifact return, zero stale policy-decision reuse, denial audit refs, and no additional provider or egress call.
- A leak/output-safety regression validates the generated output leak scan plus local-range negative-output checks: denied/error surfaces hide foreign tenant values, secret canaries reach pre-mask policy input but not collected logs, credential references stay non-raw, worker/error payloads do not leak protected data, and denial UI snapshots use stable safe reason codes.
- A schema-security gate regression creates a bad tenant-bearing table fixture and verifies it fails for missing tenant scope columns, RLS/FORCE state, command policy, app grants, and test coverage declaration; the corrected fixture passes with machine-readable inventory and rollback leaves no fixture tables.
- A policy limit matrix uses `config/vs2/policy_limits.v1.json` to prove below-limit and at-limit requests reach real local OPA, while oversized, over-deep, over-role, and unknown-enum requests are rejected before OPA with bounded resource use and no protected side effects.
- A break-glass maintenance probe proves normal app credentials cannot bypass tenant scope, missing approval is denied by OPA, the synthetic approved path is purpose/time/scope bounded, output is counts-only, and the approval is audited.
- A product-learning guard proves raw personal and organization truth/memory writes are denied without governed authority, no hidden memory/truth rows are created, and denials are audited.
- A policy-conformance matrix proves gateway, service, tool runtime, and native CLI use the same policy input digest and equivalent decisions, while active-revision mismatch fails closed with anomaly audit/metric evidence and no protected side effects.
- An overclaim regression scans VS2-facing report and README text for stale row counts and unqualified production/live-provider/penetration-test/human-acceptance claims, while requiring explicit 86 PASS / 0 NOT_VERIFIED / 7 HUMAN_REQUIRED counts and non-production boundary language.
- Fresh VS0 product-runtime, runtime-acceptance, EVUX, and operator-acceptance scenario verifiers are rerun into `reports/vs2/regression/*` and gated from those generated files.
- Fresh VS1 ontology suggest/promote scenario verifier output is rerun into `reports/vs2/regression/vs1-ontology.json` and gated from that generated file.
- Namespace-leak regressions use the same-tenant namespace local-range probe to prove implicit personal-to-organization use is denied, personal workspace reads do not expose organization rows, organization context is distinct, and explicit promotion records provenance and audit refs.
- ConnectorHub/direct-provider, direct-writeback bypass, and secure-default regressions use Docker network isolation, governed proxy/provider evidence, sandbox bypass denial, service-allow RLS defense-in-depth, cross-tenant transfer denial, app-role hardening, default-deny OPA behavior, and fail-closed egress/controller evidence.

Still not claimed:

- Full VS2 local readiness.
- Production security.
- Live-provider readiness.
- Production IdP readiness.
- Independent penetration-test completion.
- Human UX acceptance.
- Human operator approval for the fresh VS0 UI regression outputs.
- Production migration, backup, or rollback readiness.
- Independent penetration-test coverage beyond the local adversarial sandbox and Docker-network probes.

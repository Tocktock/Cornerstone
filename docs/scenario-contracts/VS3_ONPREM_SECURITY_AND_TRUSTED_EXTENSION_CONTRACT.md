# CornerStone VS3 On-Prem Security and Trusted Extension Contract

**Date:** 2026-06-29 KST
**Owner:** JiYong / Tars
**Status:** Frozen task-scoped scenario contract for VS3 planning and implementation.
**Matrix:** `docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_MATRIX.csv`
**Canonical milestone name:** VS3 - Verified On-Prem Control Plane + Trusted Extension/Connector Substrate

## Summary

VS3 is not a clean jump from VS2 into Tool SDK and signed registry work.

VS3 must first close the unresolved VS2 evidence boundary, then prove the substrate needed for trusted extensions, ConnectorHub-backed sources, WatchAgent/browser capture, and Agent Pack activation.

The safe milestone interpretation is:

```text
VS3 = VS2 production/on-prem assurance closure
    + trusted ConnectorHub/source substrate
    + trusted Tool SDK / signed registry / Agent Pack baseline
```

This document is a scenario contract, not an implementation report. All AI-verifiable rows start as `NOT_RUN`. Human and external rows remain `HUMAN_REQUIRED` until dated, redacted, reviewer-owned evidence exists.

## Source Basis

- User-provided VS3 planning brief: `/Users/jiyong/.codex/attachments/9886a460-9488-4752-85fd-ce9d0e57b4c1/pasted-text.txt`.
- `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md`: CornerStone is one evidence-first operational intelligence product with three internal engines.
- `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`: canonical long-term scenarios, especially `CS-EXT-*`, `CS-SEC-*`, and `CS-REG-*`.
- `docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md`: no CLI, no feature PASS.
- `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md`: deterministic local scenario verification and negative evidence requirements.
- `docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md`: VS2 policy, tenant isolation, and default egress-deny scenario base.
- `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_STATE_2026-06-19.md`: records one optimistic local generated PASS signal.
- `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_FINAL_REPORT_2026-06-20.md`: rejects the broad local-readiness claim and records VS2 readiness as remediation-required.
- `reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json`: current optimistic generated report signal, 86 PASS and 7 HUMAN_REQUIRED.
- `reports/scenario/vs2-policy-tenancy-egress-final.json`: corrective report signal, 86 NOT_VERIFIED and 7 HUMAN_REQUIRED.
- `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`: operator/admin UX must stay calm, progressive, and evidence-aware.
- `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md`: compatible defaults for Postgres-first durable state, RLS, OPA/Rego, audit, sandboxed tools, and ConnectorHub-mediated capabilities.

## Current Evidence Boundary

The repository contains two conflicting VS2 status signals that VS3 must reconcile before enabling trusted tools, ConnectorHub live flows, or signed Agent Pack activation:

| Evidence artifact | Current signal | VS3 interpretation |
|---|---|---|
| `reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json` | `status=success`, 86 PASS, 7 HUMAN_REQUIRED, `LOCAL_VS2_AI_VERIFIED_HUMAN_GATES_PENDING` | Useful local generated evidence, but unsafe as a final VS2 readiness claim until scenario-specific execution gaps are reconciled. |
| `reports/scenario/vs2-policy-tenancy-egress-final.json` | `status=failed`, 86 NOT_VERIFIED, 7 HUMAN_REQUIRED, `LOCAL_VS2_READINESS_REJECTED_REMEDIATION_REQUIRED` | Current conservative blocker signal for VS3-0 unless superseded by a new canonical report with exact evidence. |
| `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_FINAL_REPORT_2026-06-20.md` | Rejects the broad local-readiness claim and lists missing execution surfaces | Must be treated as the safer status boundary until VS3-0 proves otherwise. |
| `reports/security/vs2-production-like-integration-2026-06-27.json` | Local production-like Docker rehearsal | Valuable rehearsal only. It is not production, real IdP, real network, live provider, penetration-test, or migration-readiness evidence. |

## Goal

Prove that CornerStone can operate a verified on-prem control plane and trusted extension/connector substrate where:

1. VS2 evidence status is canonical and non-contradictory.
2. Request context is derived from trusted identity and membership, not caller-provided scope.
3. Postgres/RLS, OPA/Rego, default-deny egress, audit, and migration paths are real product substrates, not loose fixture claims.
4. ConnectorHub remains the only provider/credential/action boundary.
5. Read-only source connectors, WatchAgent/browser capture, Tool SDK, signed registry, and Agent Packs cannot expand authority without explicit workspace or mission activation grants.
6. CLI, API, UI, policy, audit, evidence, and operator status expose the same safety truth.
7. Local/dev assurance and production/on-prem readiness are separate completion claims.

## Success Criteria

VS3-L, the local/dev assurance claim, is allowed only when:

- all AI-owned VS3 `MUST_PASS` and `REGRESSION` rows are `PASS`;
- every PASS row has concrete evidence from executed checks;
- VS0 and VS1 regression gates are freshly run against the same source tree;
- VS2 carry-over rows are either superseded by VS3 rows or have fresh canonical evidence;
- no VS2 or VS3 report contradicts the final milestone claim;
- no production, real IdP, real network, live-provider, independent security review, or real migration/restore claim is made.

VS3-P, the production/on-prem readiness candidate claim, is allowed only when:

- VS3-L is complete;
- VS3-H01 through VS3-H07 have dated, redacted, signed evidence;
- independent security review has no open release-blocking findings;
- real IdP, real network, live provider, migration/restore, and operator UX evidence are attached;
- the release report distinguishes local, production-like rehearsal, on-prem candidate, and production claims.

## Constraints

### Product and UX

- Preserve one CornerStone product experience. Connector, policy, tool, registry, and audit controls may be visible in admin/operator contexts, but they must not become the normal-user product identity.
- Use calm, precise, evidence-aware language. Do not call local rehearsal production readiness.
- Keep extension and connector activation understandable: installed is not activated, activated is not unlimited, and approved is not permanent.

### Data and State

- Every truth-bearing object remains tenant-, owner-, namespace-, and where relevant workspace-scoped.
- Original artifacts and ConnectorHub projections must commit before derived or generated outputs can claim durable truth.
- Migration, backup, restore, rollback, and quarantine paths are part of the release gate, not afterthoughts.

### Permission and Security

- Default deny is the normal state for policy, egress, tool runtime, direct provider access, extension activation, and sensitive admin changes.
- Caller-controlled scope, role, classification, connector capability, model policy, file/network/env grants, and action authority are not trusted.
- Prompt or connector content is untrusted evidence and cannot create authority.
- Secrets must not appear in CLI output, logs, screenshots, reports, durable generated memory, or audits except through approved redacted references.

### Compatibility

- VS3 must preserve the accepted VS0 and VS1 product loops.
- VS3 must carry forward VS2 controls and evidence gaps explicitly instead of hiding them behind new Tool SDK or registry work.
- ConnectorHub and KnowledgeBase boundaries remain internal engines, not user-facing product fragmentation.

### Operational

- Local/CI verification must not require live provider credentials, real IdP accounts, production access, broad network access, or paid model providers.
- Human/on-prem rows require human evidence and cannot become AI PASS.
- Long-running or destructive migration drills require separate owner approval before execution.

## Architecture Direction

VS3 uses the existing one-product, three-engine model:

| Engine | VS3 ownership |
|---|---|
| Product / Mission / Intelligence Engine | Trusted RequestContext, mission/workspace authority, claims, action cards, approvals, policy decisions, operator UX, pack activation grants, scenario reports. |
| Archive / Evidence / KnowledgeBase Engine | Immutable artifacts, evidence bundles, hashes, redaction, provenance, derived representations, searchable source material, backup/restore evidence. |
| Connector / Provider / Action Engine | Provider access, credential custody, Source Policy, projections, declared actions, connector audit, retry/quarantine, live-provider execution, SDK/control bridge. |

The minimum VS3 domain objects are:

- `TrustedPrincipal`
- `MembershipSnapshot`
- `RequestContext`
- `PolicyInput`
- `PolicyDecision`
- `RlsScope`
- `EgressDecision`
- `ConnectorCapability`
- `SourcePolicySnapshot`
- `ProjectionDeliveryRecord`
- `ToolPackage`
- `ToolManifest`
- `ToolSignature`
- `SbomRef`
- `TrustedRegistryEntry`
- `PackInstallRecord`
- `PackActivationGrant`
- `HumanGateEvidencePackage`
- `OperatorStatusSnapshot`

## Implementation Guidance

### VS3-0 - Evidence reconciliation and contract freeze

- Create one canonical current VS2/VS3 evidence status before product work continues.
- Reject or supersede contradictory VS2 reports through a machine-readable reconciliation report.
- Add static report lint for overclaim language, empty evidence artifacts, stale report metadata, and conflicting `product_feature_claims`.
- Do not start Tool SDK, registry, live ConnectorHub, or trusted Agent Pack activation work until this gate has a PASS report.

### VS3-1 - Production-like RequestContext and identity boundary

- Implement `TrustedPrincipal`, `MembershipSnapshot`, and `RequestContext` as service-layer authority.
- Reject forged tenant, role, classification, connector capability, egress grant, and workspace scope before DB access or egress.
- Make CLI/API/UI produce comparable context digests and policy decisions.
- Keep real OIDC/SSO mapping as `HUMAN_REQUIRED` until redacted real IdP evidence exists.

### VS3-2 - Durable Postgres/RLS upgrade path

- Move RLS from proof harness to active durable object families.
- Inventory every table, view, function, materialized view, partition, trigger, FK, unique constraint, and security-definer path.
- Quarantine ownerless or ambiguous migration rows.
- Test pool reset, background workers, retries, backup, restore, and rollback.

### VS3-3 - OPA/Rego control plane hardening

- Version policy input and output schemas.
- Store `PolicyDecision` with revision, reason code, input digest, decision ID, tenant/namespace, safe resolution guidance, and audit refs.
- Harden OPA HTTP access, bundle activation, rollback, decision-log masking, cache invalidation, and fail-closed behavior.

### VS3-4 - Realistic egress boundary and tool runtime sandbox

- Enforce deny by runtime/network boundary, not only by an application branch.
- Test DNS rebinding, redirects, reserved addresses, proxy environment variables, direct sockets, subprocesses, alternate protocols, and controller outage.
- Require controlled sink evidence: denied sink receives zero requests, allowed sink receives exactly one approved call.

### VS3-5 - ConnectorHub boundary and read-only source confidence

- Keep provider credentials outside Product code.
- Commit projection to immutable Artifact before ack.
- Mirror ConnectorHub audit into CornerStone audit.
- Prove GitHub read-only, WatchAgent/macOS/Chrome capture consent, source-policy scoping, retry/quarantine, and prompt-injection resistance with fixtures.
- Keep live credentials and real provider rehearsals as human/on-prem gates.

### VS3-6 - Tool SDK and signed registry baseline

- Use WASM-first or equivalently sandboxed local tool packages for the first baseline.
- Require manifest, capabilities, file/network/env/model grants, ConnectorHub requirements, risk, version, evaluation rubric, signature, SBOM, and registry metadata.
- Separate install from activation.
- Allow execution only after explicit workspace or mission activation grant.
- Verify update diff, evaluation result, pinned version, rollback, and no undeclared access.

### VS3-7 - Operator UX, audit, and human gate packages

- Add admin/operator status surfaces for Postgres, OPA, egress, ConnectorHub, tool runtime, registry, audit, and migration.
- Make denial guidance inspectable without leaking protected data.
- Generate human-gate evidence packages, but keep the rows `HUMAN_REQUIRED` until signed evidence is uploaded or committed.

## CLI Parity

Every VS3 product feature must have native CLI parity before PASS.

Planned command coverage:

| Feature family | Required command examples |
|---|---|
| VS3 scenario gate | `cornerstone scenario verify vs3-onprem-trusted-extension --json`; `cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-YYYY-MM-DD.json --json` |
| Evidence reconciliation | `cornerstone security vs3-evidence-reconcile --json`; `cornerstone release report-lint --scope vs3 --json` |
| Identity/context | `cornerstone principal context resolve --json`; `cornerstone access check --json` |
| RLS/migration | `cornerstone tenant rls-inventory --json`; `cornerstone backup create --json`; `cornerstone restore verify --json` |
| OPA/policy | `cornerstone policy evaluate --input <file> --json`; `cornerstone policy bundle activate --dry-run --json` |
| Egress/sandbox | `cornerstone egress test --profile vs3 --json`; `cornerstone sandbox verify --json` |
| ConnectorHub | `cornerstone connector source-policy show --json`; `cornerstone connector projection verify --json`; `cornerstone connector action dry-run --json` |
| Tool/pack registry | `cornerstone pack install --dry-run --json`; `cornerstone pack activate --dry-run --json`; `cornerstone pack rollback --json`; `cornerstone tool verify --json` |
| Operator status | `cornerstone observe status --scope vs3 --json`; `cornerstone audit verify --json` |

Required CLI evidence for PASS:

- command, arguments, start/end timestamp, exit code, and timeout state;
- valid `--json` output with schema version;
- tenant/owner/namespace/workspace where relevant;
- evidence refs, policy decision refs, and audit refs;
- safe text output for operators;
- stable error exit codes for denial, missing evidence, human-required, egress denial, and secret redaction failures.

## Scenario Inventory

Counts:

| Type | Count |
|---|---:|
| MUST_PASS | 42 |
| REGRESSION | 8 |
| HUMAN_REQUIRED | 7 |
| Total | 57 |

All AI-owned rows begin as `NOT_RUN`. Human-owned rows begin as `HUMAN_REQUIRED`.

| ID | Type | Phase | Trigger / Action | Expected Behavior | Verification Steps | Required Evidence | Pass / Fail Criteria | Owner | Initial Status |
|---|---|---|---|---|---|---|---|---|---|
| VS3-GATE-001 | MUST_PASS | VS3-0 | A reviewer compares all VS2 final/current reports and scenario JSON artifacts. | One canonical current status exists; conflicting VS2 claims are rejected, superseded, or reconciled with exact report paths and hashes. | Run VS3 evidence reconciliation; inspect VS2 JSON summaries; lint claim strings. | Reconciliation JSON, report hashes, rejected/superseded artifact list, final product claim string. | PASS only if one canonical status remains and no report can be used to claim broader readiness; FAIL if contradictory PASS/NOT_VERIFIED signals remain unclassified. | AI | NOT_RUN |
| VS3-GATE-002 | MUST_PASS | VS3-0 | VS3 contract and matrix are frozen. | Markdown and CSV exist, row counts match, every row has expected behavior, verification, evidence, pass/fail criteria, owner, and initial status. | Run docs verifier and matrix structural checks. | `scripts/verify_sot_docs.sh`, matrix row-count output, duplicate-ID check. | PASS only if matrix and contract are internally consistent; FAIL on duplicate IDs, missing criteria, or status-bearing contract claims. | AI | NOT_RUN |
| VS3-GATE-003 | MUST_PASS | VS3-0 | A VS3 local/dev report is generated. | Report distinguishes `VS3-L` local/dev assurance from `VS3-P` production/on-prem candidate and cannot overclaim real IdP, live provider, real network, migration, or human acceptance. | Run static report lint over docs, reports, README, and UI copy touched by VS3. | Overclaim lint report, negative evidence counters, reviewed allowlist if any. | PASS only if forbidden overclaim strings are absent or explicitly historical; FAIL if local proof is described as production/on-prem/live readiness. | AI | NOT_RUN |
| VS3-GATE-004 | MUST_PASS | VS3-0 | A VS3 verifier or Make target is introduced. | `cornerstone scenario verify vs3-onprem-trusted-extension --json` emits status, counts, per-row evidence, human rows, and gate metadata. | Execute dry-run/list/coverage path before full implementation; validate JSON schema. | CLI transcript, JSON schema validation, coverage matrix. | PASS only if CLI is native and status-neutral before implementation; FAIL if raw scripts replace CLI parity. | AI | NOT_RUN |
| VS3-CTX-001 | MUST_PASS | VS3-1 | CLI, API, UI, worker, and tool runtime receive the same operation. | All surfaces derive RequestContext from trusted identity/membership and produce matching context digests and policy outcomes. | Run normalized fixture across all surfaces. | CLI/API/UI/worker/tool transcripts, context digest, policy decision ID, audit refs. | PASS only if no surface accepts caller-controlled scope; FAIL on digest mismatch or missing audit refs. | AI | NOT_RUN |
| VS3-CTX-002 | MUST_PASS | VS3-1 | Caller supplies forged tenant, owner, namespace, role, classification, egress, connector, or tool grant. | Forged fields are ignored or denied before protected DB access, egress, connector call, or tool execution. | Parameterized negative tests for each field and entry point. | Denied responses, zero protected DB rows touched, zero egress, policy/audit refs. | PASS only if every forged-authority path is denied or neutralized; FAIL if any forged value influences authority. | AI | NOT_RUN |
| VS3-CTX-003 | MUST_PASS | VS3-1 | Membership or activation grant is revoked after a prior allow. | New requests deny within the documented revocation window, including cached sessions, workers, and tool runtimes. | Allow -> revoke -> retry test across API, CLI, worker, and pack/tool path. | Before/after decision revisions, cache invalidation record, denial audit, zero post-revocation side effects. | PASS only if stale allow cannot produce data access or side effects; FAIL on stale success beyond documented bound. | AI | NOT_RUN |
| VS3-CTX-004 | MUST_PASS | VS3-1 | RequestContext is missing, malformed, expired, conflicted, or unresolvable. | Protected operations fail closed with helpful redacted errors and no downstream DB/egress/tool/provider access. | Fault matrix over gateway, service, worker, CLI, and tool entry points. | Stable exit/status codes, zero downstream counters, sanitized logs, audit refs. | PASS only if no protected boundary is reached; FAIL if any path falls back open. | AI | NOT_RUN |
| VS3-CTX-005 | MUST_PASS | VS3-1 | Mission/workspace authority differs from tenant-level membership. | Tenant membership alone is insufficient; mission/workspace policy controls memory, connector, model, tool, and action use. | Same-tenant personal/org/project fixture with allowed and denied operations. | Policy decisions, zero implicit context use, allowed promotion provenance, audit refs. | PASS only if namespace/workspace/mission scope is enforced above tenant RLS; FAIL on implicit cross-context use. | AI | NOT_RUN |
| VS3-RLS-001 | MUST_PASS | VS3-2 | Active VS0/VS1 durable object families are persisted. | Required tenant, owner, namespace, workspace, classification, provenance, and audit fields exist and are non-null where required. | Schema inventory plus create/read/null-insert tests. | Schema report, failed-null insert transcript, representative API/CLI payloads. | PASS only if every active truth-bearing table is scoped; FAIL on ownerless global truth or nullable required scope. | AI | NOT_RUN |
| VS3-RLS-002 | MUST_PASS | VS3-2 | Tenant A and tenant B data exist in every protected table. | Application role can read only authorized tenant rows; tenant-B data and existence metadata remain hidden from tenant A. | Two-tenant DB integration matrix using real app role. | SQL transcript, `pg_policies` inventory, row counts, zero foreign canaries. | PASS only if SELECT/count/join/search paths hide foreign data; FAIL on row, count, ID, cursor, or timing/error leak. | AI | NOT_RUN |
| VS3-RLS-003 | MUST_PASS | VS3-2 | Application role attempts cross-tenant INSERT, UPDATE, DELETE, bulk mutation, and RETURNING. | RLS/WITH CHECK/constraints deny or affect zero unauthorized rows without partial mutation. | Database mutation matrix with rollback assertions. | SQLSTATE or neutral API error, before/after snapshots, audit/anomaly event. | PASS only if unauthorized mutations have no effect; FAIL on partial cross-tenant write or revealing error. | AI | NOT_RUN |
| VS3-RLS-004 | MUST_PASS | VS3-2 | Pool, retry, worker, scheduled job, and cancellation paths reuse infrastructure. | Tenant context is transaction-local or otherwise reset; no request contaminates another request or report. | Pool stress test with alternating tenants and injected errors/timeouts. | Connection IDs, tenant sequence, reset assertions, parallel report hash stability. | PASS only if repeated parallel tests show zero contamination; FAIL on leaked context or false PASS. | AI | NOT_RUN |
| VS3-RLS-005 | MUST_PASS | VS3-2 | Migration imports known, missing, ambiguous, invalid, and duplicate ownership rows. | Known rows migrate deterministically; ambiguous rows quarantine or block; rollback is deterministic; no ownerless truth is created. | Forward migration, failed migration, quarantine, rollback, and restored-read tests. | Migration report, counts, checksums, quarantine reasons, rollback transcript. | PASS only if data integrity and scope are preserved; FAIL on silent default tenant assignment or destructive migration. | AI | NOT_RUN |
| VS3-RLS-006 | MUST_PASS | VS3-2 | Backup, tenant export, full restore, and filtered restore paths run. | Backup/restore preserves artifacts, evidence, claims, ontology, policy decisions, audit integrity, RLS policies, and tenant boundaries. | Local backup -> restore -> verify suite with two tenants. | Backup manifest, restored hashes/counts, RLS inventory before/after, audit verify output. | PASS only if restored system reproduces evidence and audit safely; FAIL on missing rows, broken policies, or leaked tenant export. | AI | NOT_RUN |
| VS3-OPA-001 | MUST_PASS | VS3-3 | Any protected operation constructs policy input. | Policy input schema includes trusted subject, scope, resource, action, classification, mission authority, connector/tool capability, model policy, risk, data scope, and environment. | Contract tests and golden fixtures for every operation family. | Schema file, valid/invalid fixture results, source-of-attribute map, input digest. | PASS only if unknown or caller-authoritative fields fail closed; FAIL on schema drift or unvalidated authority. | AI | NOT_RUN |
| VS3-OPA-002 | MUST_PASS | VS3-3 | OPA/Rego evaluates allow, deny, escalate, undefined, malformed, and conflict cases. | PolicyDecision records action, reason codes, safe resolution, bundle revision/hash, decision ID, input digest, tenant/namespace, evidence refs, and audit refs. | `opa test`, HTTP decision tests, golden JSON validation. | OPA JSON output, coverage, PolicyDecision fixtures, audit linkage, redaction proof. | PASS only if decisions are deterministic and default deny; FAIL on implicit allow or missing reason/audit refs. | AI | NOT_RUN |
| VS3-OPA-003 | MUST_PASS | VS3-3 | OPA service, management APIs, and decision APIs are accessed from allowed and disallowed peers. | OPA is bound to intended interfaces, authenticated/authorized, and management APIs are not anonymously exposed. | Network/container tests from allowed and denied peers. | OPA config, listening sockets, authorized success, unauthorized denial, audit/status refs. | PASS only if unauthorized access is denied; FAIL on anonymous policy or data API exposure. | AI | NOT_RUN |
| VS3-OPA-004 | MUST_PASS | VS3-3 | Bundle update is valid, invalid, incompatible, stale, or rollback-triggering. | Activation is atomic; invalid bundles fail without replacing last known good; first-start failure fails closed and is visible. | Bundle lifecycle tests and rollback injection. | Active revision before/after, invalid-bundle error, degraded readiness, policy decisions under failure. | PASS only if no invalid policy becomes active; FAIL on stale permissive allow or hidden degraded state. | AI | NOT_RUN |
| VS3-OPA-005 | MUST_PASS | VS3-3 | Policy logs include sensitive input or denied data. | Decision logging and audit mirror mask secrets/protected values while preserving correlation. | Secret-canary fixture through allow, deny, and malformed paths. | Decision log sample, CornerStone audit refs, zero secret scanner findings. | PASS only if logs are useful and redacted; FAIL on raw secret or protected payload leak. | AI | NOT_RUN |
| VS3-EGR-001 | MUST_PASS | VS3-4 | Tool/runtime has no egress grant and a reachable forbidden sink exists. | Runtime/network boundary blocks connection; forbidden sink records zero requests and zero bytes. | Container/process integration test with sink counters. | Denied tool result, sink logs, network counters, policy/audit refs. | PASS only if no packet/request reaches forbidden sink; FAIL if app merely skips call without runtime proof. | AI | NOT_RUN |
| VS3-EGR-002 | MUST_PASS | VS3-4 | Tool/workflow has declared ConnectorHub capability, current approval, policy allow, and mock provider destination. | Exactly one approved call reaches allowed sink; evidence, connector result, policy, approval, and audit are linked. | End-to-end local controlled sink test. | One sink request, sanitized metadata, Action/Workflow/Connector result, decision/audit refs. | PASS only if exactly the declared call occurs; FAIL on missing call, duplicate call, or undeclared call. | AI | NOT_RUN |
| VS3-EGR-003 | MUST_PASS | VS3-4 | Destination uses URL normalization variants, alternate ports, methods, redirect hops, or DNS rebinding. | Every normalized destination and redirect hop is re-authorized; denied addresses are never contacted. | Table-driven URL, redirect, fake-DNS, IPv4/IPv6 tests. | Per-case decisions, DNS transcript, redirect hop decisions, sink logs, zero denied-address contact. | PASS only if all bypass variants deny safely; FAIL on broadened allowlist or header leak. | AI | NOT_RUN |
| VS3-EGR-004 | MUST_PASS | VS3-4 | Tool attempts direct sockets, proxy env, alternate DNS, WebSocket/FTP/SMTP, subprocess, shell, host filesystem, or env access. | Undeclared protocols and host access are blocked by sandbox/capability boundary. | Adversarial sandbox suite. | Zero unauthorized connections/processes/files/env reads, denied capability records, runtime audit. | PASS only if no undeclared access succeeds; FAIL on arbitrary host/shell access. | AI | NOT_RUN |
| VS3-EGR-005 | MUST_PASS | VS3-4 | Egress controller, proxy, sandbox policy, or runtime policy component is down or misconfigured. | Protected capabilities fail closed; readiness degrades; no fallback direct connection or host access occurs. | Disable component and rerun allowed/denied cases. | Degraded readiness, denied operations, zero sink/host counters, operator guidance. | PASS only if outage denies safely; FAIL on fallback open or misleading ready state. | AI | NOT_RUN |
| VS3-EGR-006 | MUST_PASS | VS3-4 | Untrusted artifact, connector payload, web page, or tool output asks the system to call a URL, approve an action, or change policy. | Content remains untrusted evidence and cannot create egress grants, action approvals, policy changes, or tool execution. | Prompt-injection fixtures across artifact and connector paths. | Zero tool/action/egress calls, blocked-attempt audit, untrusted label, evidence refs. | PASS only if malicious content produces no authority; FAIL on content-driven authority expansion. | AI | NOT_RUN |
| VS3-CON-001 | MUST_PASS | VS3-5 | ConnectorHub delivers a projection from a read-only source. | CornerStone commits immutable Artifact and evidence metadata before ack; crash/retry cannot acknowledge uncommitted source truth. | Projection delivery crash/retry fixture. | Artifact ID/hash/provenance, delivery attempt log, ack-after-commit proof, audit refs. | PASS only if ack follows durable commit; FAIL on lost projection or ack-before-commit. | AI | NOT_RUN |
| VS3-CON-002 | MUST_PASS | VS3-5 | GitHub connector fixture is installed and used. | Connector is read-only: zero write mappings, zero mutation commands, write attempts denied or quarantined. | Static mapping scan plus runtime write-denial tests. | Capability manifest, denied write transcript, zero external mutation counters, audit refs. | PASS only if no write-capable path is exposed; FAIL on mutation mapping or direct write client. | AI | NOT_RUN |
| VS3-CON-003 | MUST_PASS | VS3-5 | Connector credentials, tokens, provider payloads, or credential-bearing URLs appear in allowed/denied paths. | Credentials stay in ConnectorHub; Product outputs use credential references only and redact sensitive payload fields. | Secret canary scan across CLI, logs, screenshots, reports, audit, and durable state. | Zero secret scanner findings, credential-ref-only payload, redacted sink metadata. | PASS only if raw secrets never appear; FAIL on raw credential exposure. | AI | NOT_RUN |
| VS3-CON-004 | MUST_PASS | VS3-5 | Source Policy changes for connector, workspace, repo, browser scope, or capture mode. | SourcePolicySnapshot is tenant/workspace scoped, auditable, revocable, and enforced on next delivery/capture. | Policy update -> delivery/capture -> revoke -> retry fixture. | Snapshot diff, before/after delivery decisions, revoked denial, audit refs. | PASS only if source policy cannot bleed across scopes; FAIL on stale or cross-scope delivery. | AI | NOT_RUN |
| VS3-CON-005 | MUST_PASS | VS3-5 | WatchAgent/macOS/Chrome capture is enabled in local fixture mode. | Capture is consented, bounded, pauseable, revocable, scope-visible, and summary-only where raw capture is disallowed. | Fixture runtime and browser/CLI checks with pause/revoke. | Consent record, scope config, capture summary, pause/revoke transcript, zero disallowed raw output. | PASS only if capture obeys explicit scope and stop controls; FAIL on silent or unbounded capture. | AI/local plus physical-device human where needed | NOT_RUN |
| VS3-CON-006 | MUST_PASS | VS3-5 | Connector delivery fails, duplicates, returns stale data, or contains prompt injection. | Retry/quarantine is idempotent, evidence-safe, and cannot create memory, policy, action, egress, or authority without product approval. | Fault injection and malicious payload fixture. | Retry/quarantine record, zero unauthorized side effects, evidence refs, audit refs. | PASS only if failed/malicious delivery remains contained; FAIL on duplicate truth or side effect. | AI | NOT_RUN |
| VS3-TOOL-001 | MUST_PASS | VS3-6 | Local sample tool package is built. | Package contains manifest, capabilities, file/network/env/model grants, ConnectorHub requirements, risk, version, evaluation rubric, signature, SBOM, and provenance metadata. | Build and manifest/schema verification. | Package artifact, manifest JSON, signature, SBOM, schema validation, audit refs. | PASS only if required metadata exists and validates; FAIL on missing grants, signature, or SBOM. | AI | NOT_RUN |
| VS3-TOOL-002 | MUST_PASS | VS3-6 | Tool package is registered in local trusted registry. | Registry accepts only trusted signed package metadata and rejects unsigned, tampered, stale, revoked, or unknown-source packages. | Registry positive/negative package tests. | Acceptance record, rejection transcripts, signature verification output, registry audit. | PASS only if tampered/untrusted packages are rejected; FAIL on silent trust or unsigned install. | AI | NOT_RUN |
| VS3-TOOL-003 | MUST_PASS | VS3-6 | Pack/tool is installed but not activated. | Install makes package available but gives no mission/workspace authority, connector access, model route, egress, file/env grant, or action capability. | Installed-inactive execution attempts across surfaces. | Install record, denied execution transcripts, zero side effects, audit refs. | PASS only if installed package cannot act; FAIL on install-as-activation. | AI | NOT_RUN |
| VS3-TOOL-004 | MUST_PASS | VS3-6 | Namespace owner activates a pack/tool for a workspace or mission. | Activation grants are explicit, least-privilege, pinned, reversible, auditable, and tied to policy decision and RequestContext. | Activation dry-run, approval, execution, revoke test. | Grant record, policy decision, audit refs, capability list, revocation transcript. | PASS only if only granted capabilities work; FAIL on ungranted capability or hidden activation. | AI | NOT_RUN |
| VS3-TOOL-005 | MUST_PASS | VS3-6 | Activated tool attempts undeclared file, env, network, shell, model, connector, or memory access. | Sandbox denies undeclared access and records negative evidence without leaking secrets. | Runtime sandbox negative suite. | Denied records, zero file/env/network/process access counters, audit refs, secret scan. | PASS only if every undeclared access is denied; FAIL on sandbox bypass. | AI | NOT_RUN |
| VS3-TOOL-006 | MUST_PASS | VS3-6 | Pack update is available. | Update shows diff of role contract, capabilities, playbooks, model policy, connector requirements, risk, evaluation results, migration notes, and rollback path before activation. | Update dry-run and evaluation gate tests. | Diff artifact, evaluation result, blocked/approved update transcript, audit refs. | PASS only if behavior-changing update cannot silently activate; FAIL on unreviewed update. | AI | NOT_RUN |
| VS3-TOOL-007 | MUST_PASS | VS3-6 | Active pack is rolled back or emergency-patched. | Rollback returns to previous pinned version; emergency patch is policy-governed, audited, and does not expand authority without review. | Rollback and emergency-patch simulation. | Previous/new version records, rollback transcript, policy decision, affected mission list. | PASS only if rollback is deterministic and auditable; FAIL on irreversible or authority-expanding patch. | AI | NOT_RUN |
| VS3-OBS-001 | MUST_PASS | VS3-7 | Operator opens VS3 status. | Status distinguishes Postgres/RLS, OPA, egress, ConnectorHub, tool runtime, registry, audit, backup/restore, migration, and human gates. | Fault-injection status tests and CLI/API/UI comparison. | `observe status` JSON, UI/DOM snapshot, component fault results, audit refs. | PASS only if degraded components are visible and scoped; FAIL on misleading green state. | AI | NOT_RUN |
| VS3-OBS-002 | MUST_PASS | VS3-7 | Policy, RLS, egress, connector, tool, action, migration, and human-gate events occur. | Audit records are append-only, tamper-evident, tenant-safe, queryable, and link evidence/policy/action refs. | Audit contract tests and tamper fixture. | Event inventory, hash/checkpoint verification, tamper detection failure, query transcript. | PASS only if required event classes are covered; FAIL on missing or mutable critical event. | AI | NOT_RUN |
| VS3-OBS-003 | MUST_PASS | VS3-7 | Human-gate package is generated for each VS3-H row. | Package contains scope, why AI cannot verify, required human action, expected evidence, redaction rules, release impact, and blank approval/rejection record. | Generate and validate evidence packages. | Seven package files, schema validation, redaction guidance, no PASS status. | PASS only if packages prepare evidence without marking H rows PASS; FAIL if generated package is treated as human evidence. | AI | NOT_RUN |
| VS3-REG-001 | REGRESSION | Final gate | VS3 changes are applied. | VS0 Artifact -> Search -> Evidence -> Claim -> Action -> Audit loop still passes on the same tree. | Rerun accepted VS0 gates. | Scenario reports, command transcripts, zero weakened assertions. | PASS only if fresh VS0 reports pass; FAIL on stale report reuse or regression. | AI | NOT_RUN |
| VS3-REG-002 | REGRESSION | Final gate | VS3 changes are applied over accepted VS1 ontology slice. | VS1 suggestions remain draft until promotion; search/profile/claim/action/audit integration remains green under VS3 policy/RLS boundaries. | Rerun VS1 gate and cross-scope ontology tests. | VS1 scenario report, zero auto-promotions, policy/audit refs. | PASS only if fresh VS1 report passes; FAIL on auto-promotion or cross-scope leakage. | AI | NOT_RUN |
| VS3-REG-003 | REGRESSION | Final gate | Prompt or connector content claims new authority over tenant, connector, egress, tool, model, memory, policy, or action. | Authority remains governed by RequestContext, Mission/Agent/Pack contracts, ConnectorHub capability, and policy. | Red-team fixture suite. | Zero authority changes, zero tool/action/egress calls, blocked-attempt audit. | PASS only if prompt cannot expand authority; FAIL on content-driven grant. | AI | NOT_RUN |
| VS3-REG-004 | REGRESSION | Final gate | New pack, connector, model, policy, table, workflow, or tool is introduced. | Audit and scenario coverage cannot silently drop; missing coverage fails gate before release claim. | Coverage and audit mutation tests. | Failing omission fixture, passing corrected inventory, matrix coverage output. | PASS only if coverage gates detect omissions; FAIL on silent scenario disappearance. | AI | NOT_RUN |
| VS3-REG-005 | REGRESSION | Final gate | Reports, README, UI, help, or release metadata describe VS3. | Claims do not describe local/dev proof as production, real IdP, live provider, penetration-tested, human-accepted, or migration-ready. | Static overclaim lint and evidence manifest review. | Zero overclaim findings, historical-claim allowlist if needed, human-required table. | PASS only if wording is no stronger than evidence; FAIL on unqualified readiness claim. | AI | NOT_RUN |
| VS3-REG-006 | REGRESSION | Final gate | A normal user opens CornerStone after VS3 admin/security work. | Product still appears as one calm CornerStone workspace; connector/tool/policy admin detail is progressively disclosed in admin context. | UI/nav review and browser/DOM check. | Screenshots/DOM, nav map, absence of repo-name mental model. | PASS only if normal flow remains product-first; FAIL if first value is connector/tool admin setup. | AI plus human UX in H06 | NOT_RUN |
| VS3-REG-007 | REGRESSION | Final gate | New dependencies, signing tools, sandbox runtime, or registry infrastructure are proposed. | Supply-chain change is justified, pinned, scanned, and approved where production dependency or security-sensitive gate applies. | Dependency diff, lockfile review, approval-gate check. | Diff, version pins, security notes, H01 approval if required. | PASS only if no unapproved production dependency enters; FAIL on broad/churn dependency change. | AI/Human where approval required | NOT_RUN |
| VS3-REG-008 | REGRESSION | Final gate | Security defaults initialize in fresh, reset, partial-upgrade, or new workspace states. | Default egress deny, default policy deny, no arbitrary shell, RLS enforced, activation inactive, and high-risk actions require approval. | Fresh/reset/partial-config integration suite. | Default config snapshot, denial suite, role/policy inventory, audit refs. | PASS only if conservative defaults hold everywhere; FAIL on permissive default. | AI | NOT_RUN |
| VS3-H01 | HUMAN_REQUIRED | Human gate | VS3 architecture adds or changes authz, tenant isolation, durable migration, OPA, sandbox, registry, signing, or production dependencies. | JiYong/Tars or authorized owner approves architecture, dependency scope, migration/rollback plan, and security ownership. | Human architecture/security review. | Dated APPROVE/REJECT record with scope, exceptions, rollback owner, dependency decision. | Remains HUMAN_REQUIRED until signed owner evidence exists; blocks VS3-P and security-sensitive implementation gates as applicable. | Human | HUMAN_REQUIRED |
| VS3-H02 | HUMAN_REQUIRED | Human gate | Representative production/on-prem topology is ready for adversarial review. | Independent reviewer validates tenant, policy, connector, tool, extension, and egress bypass resistance. | Independent security review and retest. | Signed review, test scope/topology, findings, remediation, retest evidence. | Remains HUMAN_REQUIRED until independent evidence exists; blocks VS3-P. | Human | HUMAN_REQUIRED |
| VS3-H03 | HUMAN_REQUIRED | Human gate | Real OIDC/SSO or enterprise identity provider is selected. | Real user/group/role/attribute/revocation mapping is validated without promoting synthetic fixture proof. | Human-approved real IdP test. | Redacted login, mapping, revocation transcript, owner/security approval. | Remains HUMAN_REQUIRED until real IdP evidence exists; blocks VS3-P. | Human | HUMAN_REQUIRED |
| VS3-H04 | HUMAN_REQUIRED | Human gate | Target on-prem network, DNS, proxy, firewall, service mesh, and sandbox are available. | Real topology enforces deny-by-default and approved exceptions outside local harness. | Network/security operator review. | Topology diagram, firewall/proxy/network-policy evidence, packet/log transcript, approval. | Remains HUMAN_REQUIRED until real topology evidence exists; blocks VS3-P. | Human | HUMAN_REQUIRED |
| VS3-H05 | HUMAN_REQUIRED | Human gate | Live ConnectorHub credentials and provider permissions are available. | Low-risk read-only or explicitly approved live provider rehearsal succeeds without secret exposure. | Approved live-provider rehearsal. | Redacted provider transcript, approval, ConnectorHub/CornerStone audit refs, result/evidence refs. | Remains HUMAN_REQUIRED until live-provider evidence exists; blocks live readiness claim. | Human | HUMAN_REQUIRED |
| VS3-H06 | HUMAN_REQUIRED | Human gate | Human operator uses denial, audit, connector, source-policy, tool registry, activation, and migration/status UX. | Operator accepts or rejects whether boundaries, causes, risks, and recovery paths are understandable and not misleading. | Human operator UX/trust review. | ACCEPT/REJECT note, screenshots/recording, task outcomes, issue list if rejected. | Remains HUMAN_REQUIRED until human acceptance exists; blocks UX acceptance claim. | Human | HUMAN_REQUIRED |
| VS3-H07 | HUMAN_REQUIRED | Human gate | Real or representative production-like data, retention rules, backup tooling, and rollback authority are available. | Migration, backup, restore, rollback, quarantine, RLS, policy, and audit drill passes under approved data rules. | Human-supervised non-destructive drill. | Signed transcript, before/after counts/hashes, restore/rollback result, RLS/policy inventory. | Remains HUMAN_REQUIRED until signed drill evidence exists; blocks migration/readiness claim. | Human | HUMAN_REQUIRED |

## Proof Surface Matrix

| Surface | Required before VS3-L | Required before VS3-P |
|---|---|---|
| Local docs/contract | Contract, matrix, row-count and duplicate-ID checks pass. | Same as VS3-L plus human evidence attached to final report. |
| CLI | Native CLI transcripts for all AI-owned product features. | Same, against approved production/on-prem candidate environment where safe. |
| API/service | Shared RequestContext, policy, audit, and workflow handlers verified. | Same with real IdP and approved topology evidence. |
| Database | Docker/local Postgres RLS and migration/backup/restore rehearsal. | Human-supervised production-like migration/restore drill. |
| OPA/Rego | Local OPA tests, HTTP decisions, bundle lifecycle, fail-closed proof. | Independent security review and approved operational deployment plan. |
| Egress/sandbox | Controlled sink, DNS/redirect/socket/proxy/subprocess negative evidence. | Real network/firewall/proxy/service mesh evidence. |
| ConnectorHub | Fixture provider and read-only source proof with no credential exposure. | Live provider rehearsal with redacted evidence. |
| Tool/pack registry | Local signed/SBOM package, install, activation, execution, update, rollback proof. | Signing-root, key rotation, registry governance, and emergency process approval. |
| UI/operator | Objective UI/status/DOM checks for admin/operator surfaces. | Human operator UX acceptance. |
| Reports | No overclaim, exact source-tree metadata, evidence refs, audit refs, human-required rows. | Same plus signed human/on-prem evidence. |

## Human Required

Human-required rows are release-critical for VS3-P and must not be converted to AI PASS by generated templates, local fixtures, or report placeholders.

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS3-H01 | Security-sensitive architecture, dependencies, authz, tenant isolation, migration, and rollback ownership need owner approval. | Review and approve/reject VS3 architecture and implementation gates. | Dated signed approval/rejection record. | Blocks security-sensitive implementation and VS3-P. |
| VS3-H02 | Independent bypass review requires a human reviewer and representative topology. | Perform or commission security review and retest. | Signed review, findings, remediation, retest evidence. | Blocks VS3-P. |
| VS3-H03 | Real IdP state, user/group mapping, and revocation are external. | Run redacted real OIDC/SSO mapping and revocation test. | Login/mapping/revocation transcript and approval. | Blocks real identity readiness. |
| VS3-H04 | Real network controls require on-prem/firewall/proxy/service-mesh access. | Validate egress deny and allowed exceptions in target topology. | Topology, logs, packet/proxy evidence, approval. | Blocks on-prem network readiness. |
| VS3-H05 | Live provider rehearsal requires real credentials and external account permissions. | Execute approved live provider rehearsal. | Redacted provider transcript and audit/evidence refs. | Blocks live ConnectorHub readiness. |
| VS3-H06 | Operator trust and UX acceptance are subjective. | Complete walkthrough and accept/reject UX. | Acceptance note, screenshots/recording, issue list. | Blocks human UX acceptance claim. |
| VS3-H07 | Real migration/restore requires real or representative data, retention rules, and rollback authority. | Run supervised non-destructive migration/backup/restore drill. | Signed transcript, counts/hashes, restore/rollback result. | Blocks migration/restore readiness claim. |

## Out of Scope Before This Contract Is Implemented

- Public marketplace distribution.
- Third-party pack trust beyond local trusted/internal registry fixtures.
- Production deployment or production migration.
- Real IdP readiness.
- Live provider readiness.
- Penetration-test completion.
- Legal/compliance certification.
- Full cloud/Kubernetes production hardening.
- Broad repo merge across Cornerstone, KnowledgeBase, and Connector-Hub.
- Autonomous destructive or external writeback without separate action/workflow approval.

## Failure Reverse Engineering Rule

For any VS3 `FAIL`, `NOT_VERIFIED`, or `NOT_RUN` row in an implementation report, the report must include:

```markdown
Scenario:
Expected:
Actual / Missing Evidence:
First Failing Layer:
Root Cause:
Why It Was Missed:
Fix or Blocker:
Re-verification Plan:
Related Regressions:
Updated Status:
```

## Definition of VS3 PASS

A VS3 scenario can be marked PASS only when:

1. The observable behavior occurred.
2. The named verification command or check actually ran.
3. Evidence exists as command output, scenario report, JSON artifact, API response, browser/DOM proof, DB transcript, audit record, or human approval record.
4. Evidence is tied to the exact source tree or environment boundary being claimed.
5. Required negative evidence is present for safety scenarios.
6. CLI parity exists for product features.
7. Human-only rows remain `HUMAN_REQUIRED` until real human evidence exists.

Do not mark PASS from:

- scenario row existence alone;
- local fixture setup alone;
- generated wrapper reports without scenario-specific behavior;
- stale VS2 reports;
- screenshots without state/audit/evidence backing;
- a human-gate template;
- live-provider, production, real-network, or real-IdP claims without corresponding human/on-prem evidence.

## Next Implementation Slice Recommendation

Start VS3 with VS3-0 only:

1. implement `cornerstone security vs3-evidence-reconcile --json`;
2. implement `cornerstone scenario verify vs3-onprem-trusted-extension --dry-run --json` or equivalent coverage/list path;
3. add report lint for VS2/VS3 claim conflicts and overclaims;
4. produce a canonical VS2 carry-over status report;
5. only then start VS3-1 RequestContext work.

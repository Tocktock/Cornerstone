# CornerStone ConnectorHub Test Scenario-Based Implementation Document

**Date:** 2026-06-22
**Owner:** JiYong / Tars
**Recommended product slice:** v3 “Connected Evidence”
**Document status:** Implementation contract candidate - ready for owner freeze; product scenarios are not yet verified.

# Summary

This document turns the ConnectorHub investigation and the stakeholder decisions in the conversation into an engineer-executable implementation contract. The recommended v3 base is **connected evidence**, not a visible “Connector Hub product”: CornerStone owns product meaning, durable evidence, claims, memory, missions, policy, approvals, and user experience; ConnectorHub owns provider access, credentials, Source Policy enforcement, Projections, Delivery, declared connector execution, retry/quarantine, and connector audit.

The smallest complete release sequence is:

```text
Capability Contract -> Setup Result -> Owner-confirmed Source Policy
-> Projection Delivery -> immutable Artifact -> Evidence Bundle
-> Capture Inbox / Watch Result / Brief / Claim
-> optional ActionCard dry-run and ConnectorHub preflight
-> governed execution -> outcome re-ingest -> audit and learning
```

The recommended **v3 base scope** includes CH-0 through CH-3: contract adapter, durable read ingestion, GitHub read-only, and consent-driven macOS/Chrome WatchAgent. CH-4 live external execution is a separately gated extension and is not required for the GitHub read-only slice.

## Source basis and authority discipline

- Stakeholder conversation: comprehensive ConnectorHub analysis; practical CornerStone application; scenario guide as the primary output; v3 connector/WatchAgent direction; GitHub read-only.
- Attached test-scenario implementation prompt and Scenario-First Agent Instruction.
- Existing ConnectorHub application guide generated from the earlier investigation.
- Current repository evidence, especially `Connector-Hub/docs/internal-cornerstone-connectorhub-contract.md`, `Connector-Hub/connectorhubkit/`, `Connector-Hub/character_watch_agent/`, `Cornerstone/docs/architecture/ONE_PRODUCT_THREE_ENGINES.md`, and the current VS2 remediation report.
- The older `project-sot.md` is historical where it conflicts with current product authority; documented targets are not treated as implemented behavior.

## Frozen implementation scope

**In scope for recommended v3 base:** contract and Setup Result integration; Source Policy; durable Projection ingestion; Artifact/Evidence handoff; retry/quarantine; GitHub selected-repository read-only sync; macOS and Chrome capture with explicit consent; Watch Rules; Capture Inbox; privacy lifecycle; namespace, audit, and native CLI/API/UI parity.

**Out of scope before implementation:** GitHub mutation of any kind; public connector marketplace; production release claims; unbounded autonomy; silent cross-namespace promotion; arbitrary shell/network access; live external writeback in the v3 base; broad repository merge; replacing current storage without a migration plan; package publishing, tags, or release operations.

# Problem Definition

## Product confusion

ConnectorHubKit is a strong internal substrate, but exposing it as a second user-facing product would violate CornerStone’s one-product direction. The root cause is that the existing repositories have independently useful surfaces and terminology. Without a CornerStone-owned integration layer, repository boundaries can leak into navigation, setup, support, and mental models.

## Domain boundary issues

ConnectorHubKit owns provider safety while CornerStone owns meaning and durable product state. The current gap is not a missing provider client; it is the absence of a verified CornerStone adapter that translates App Requirements, Setup Result, Source Policy, Projection Delivery, EvidenceRef, Action Preflight, and connector audit into CornerStone domain objects without bypassing Archive, Policy, Workflow, or Audit boundaries.

## Validation gaps

ConnectorHubKit validates local App Requirements and Provider Packs, but CornerStone does not yet have a frozen connector capability contract, activation gate, defense-in-depth Projection validator, or exact scenario verifier for the integrated path. The root risk is accepting a technically valid Hub payload that is invalid for the active owner, namespace, security classification, retention policy, or product trust state.

## Data consistency risks

Delivery acknowledgement, immutable Artifact creation, evidence linking, cursor advancement, deduplication, and audit correlation span multiple components. A naive poll-and-ack implementation can acknowledge before durable commit, create duplicate truth on retry, lose lineage on changed content, or mistake connector metadata for the immutable original.

## Downstream read and execution risks

Connector Projections can feed search, briefs, claims, memory, Watch Results, and Actions. If downstream readers do not preserve source restrictions, trust state, scope, and evidence coverage, derived outputs can become unsupported or cross-boundary truth. For actions, ConnectorHub preflight alone is insufficient: CornerStone policy, evidence, approval, and WorkflowRun authority must remain decisive.

## Privacy, security, and operational risks

macOS and Chrome capture can be useful but can also feel like surveillance. Provider content is untrusted, credentials and raw payloads are sensitive, and external network access must remain policy-controlled. Production connected sources are additionally blocked by currently unverified RequestContext, RLS, OPA, egress, and recovery paths.

## Verification and claim risk

ConnectorHubKit has substantial deterministic fixture evidence, while live providers, physical-device behavior, subjective trust, and production security require separate proof. The root cause of overclaim risk is collapsing “schema exists,” “fixture passed,” “live provider worked,” and “production-ready” into one status. This document keeps them separate.

# Requirements

## Explicit Requirements

- **ER-01:** Investigate ConnectorHubKit in depth, including its concepts, architecture, contracts, runtime behavior, and implementation approach. _Basis: Direct stakeholder request._
- **ER-02:** Explain how ConnectorHubKit can be adapted to CornerStone in a practical, implementation-oriented way. _Basis: Direct stakeholder request._
- **ER-03:** Make the scenario guide the primary deliverable, with concrete use cases, workflows, edge cases, and verifiable implementation scenarios. _Basis: Direct stakeholder request._
- **ER-04:** Use ConnectorHub as the foundation for a future CornerStone connected-source release, including connectors and WatchAgent capabilities. _Basis: Direct stakeholder decision in the conversation._
- **ER-05:** Support permissioned macOS activity collection and Chrome extension integration through the connector boundary. _Basis: Direct stakeholder examples._
- **ER-06:** Keep GitHub strictly read-only. No issue, comment, label, merge, push, branch, release, or settings mutation belongs in this scope. _Basis: Direct stakeholder constraint._
- **ER-07:** Produce an engineer-executable, test-scenario-based implementation document rather than a descriptive summary. _Basis: Attached implementation-document prompt._
- **ER-08:** Every scenario must identify verification method and concrete evidence required for PASS. _Basis: Attached implementation-document prompt and scenario-first instruction._
- **ER-09:** Do not claim implementation completion while required scenarios remain unverified or human-gated. _Basis: Attached implementation-document prompt and scenario-first instruction._

## Implicit Requirements

- **IR-01:** Users experience one CornerStone product; ConnectorHub remains an internal engine and is not exposed as a separate product model. _Basis: CornerStone one-product boundary._
- **IR-02:** Integrate through a CornerStone-owned adapter/port. Do not merge repositories or import provider SDKs into Product logic. _Basis: Connector ownership boundary and adapter rule._
- **IR-03:** A versioned capability contract and Setup Result act as the activation gate for every connector-backed feature. _Basis: ConnectorHub App Requirements and Setup Result model._
- **IR-04:** Every connector contract, connection, Delivery, Artifact, Watch Rule, Action, and audit reference is explicitly scoped by tenant, owner, namespace, and workspace. _Basis: CornerStone namespace and tenancy rules._
- **IR-05:** A Projection Delivery is acknowledged only after the exact delivered representation and its provenance are durably committed as a CornerStone Artifact or equivalent immutable intake record. _Basis: Archive-first rule plus Delivery semantics._
- **IR-06:** ConnectorHub EvidenceRef metadata is provenance input, not the immutable original by itself; CornerStone assembles the Evidence Bundle. _Basis: Internal ConnectorHub-to-CornerStone contract._
- **IR-07:** Delivery handling is idempotent, retryable, and quarantine-aware; duplicate or changed source content preserves lineage. _Basis: ConnectorHub Delivery and Archive requirements._
- **IR-08:** Source Policy is durable, owner-visible, versioned, and enforced again at the CornerStone boundary. _Basis: Source Policy and defense-in-depth requirements._
- **IR-09:** Provider credentials, clients, raw local paths, raw provider payloads, and raw access handles remain inside ConnectorHub and never appear in product outputs. _Basis: ConnectorHub app-facing prohibitions._
- **IR-10:** All connector and browser content is untrusted evidence and cannot instruct agents, expand authority, or trigger tools/actions. _Basis: Prompt-injection and untrusted-content rules._
- **IR-11:** External execution must pass ActionCard dry-run, ConnectorHub preflight, policy, evidence, approval, idempotency, execution, and audit handoff. _Basis: Action Safety Envelope._
- **IR-12:** Every product capability has native CornerStone CLI, API, and UI paths with stable IDs, exit codes, evidence refs, policy refs, and audit refs. _Basis: CLI-native-first and product parity rule._
- **IR-13:** Fixture proof, live-provider proof, human acceptance, and production readiness are distinct claims with distinct evidence. _Basis: Release truth standard._
- **IR-14:** WatchAgent is opt-in, local-first, privacy-bounded, pausable, revocable, retention-aware, exportable, and partially deletable where policy permits. _Basis: WatchAgent and privacy requirements._
- **IR-15:** Contracts, Projection schemas, Provider Packs, and migration behavior are versioned and pinned to prevent silent behavior changes. _Basis: Versioning and compatibility constraints._
- **IR-16:** Existing VS-0/VS-1 behavior and first-value onboarding must continue to work without connector setup. _Basis: Regression and onboarding constraints._
- **IR-17:** Connector-level audit events are correlated into the CornerStone audit lifecycle without weakening redaction or tamper checks. _Basis: Audit boundary._
- **IR-18:** Production connected-source release remains blocked until scenario-specific RequestContext, RLS, OPA, egress, backup/restore, and related VS2 checks are genuinely verified. _Basis: Current CornerStone VS2 remediation status._

## Assumptions

- **A-01 - Assumption:** Product release label “v3” is separate from scenario-slice identifiers such as VS-0/VS-1/VS-2, avoiding collision with the historical VS-3 Tool SDK milestone.
- **A-02 - Assumption:** The smallest first implementation may use an in-process or local ConnectorHub adapter, provided the port keeps process/transport replacement possible.
- **A-03 - Assumption:** GitHub authentication will use a least-privilege mechanism such as a selected-repository GitHub App or equivalent, but the exact mechanism is not frozen by the conversation.
- **A-04 - Assumption:** CornerStone archives the exact ConnectorHub Projection envelope bytes and links ConnectorHub provenance; raw provider payloads remain inside ConnectorHub unless a separately approved raw-access design exists.
- **A-05 - Assumption:** The first slice may preserve current local runtime storage while defining Postgres/RLS-compatible domain contracts and migrations.
- **A-06 - Assumption:** GitHub remains read-only for the entire program described here. Governed write scenarios apply only to separately approved, non-GitHub connectors.
- **A-07 - Assumption:** WatchAgent data remains local by default and capture is disabled until explicit user consent and platform/browser permission are present.
- **A-08 - Assumption:** No new production dependency, production credential, or external mutation is authorized by this document.

# Test Scenarios

Scenario rows are status-neutral acceptance criteria. At document generation time, none of the product implementation rows is marked PASS. Human-only rows require real accounts, devices, subjective review, or production-like environments.

| Scenario ID | Priority | Given | When | Then | Related requirement | Verification | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CS-CH-001 | MUST_PASS | An owner is in one tenant/namespace/workspace and submits a versioned connector capability contract. | CornerStone validates and registers the contract through its ConnectorPort. | A scoped Setup Result is persisted with readiness, mappings, Source Policy, warnings, verification refs, and no provider internals. | ER-01, ER-02, IR-02, IR-03, IR-04, IR-09 | Schema unit test; CLI/API integration test; durable-state and secret scan. | Contract fixture, Setup Result JSON, scoped DB rows, audit refs, zero-secret scan. |
| CS-CH-002 | MUST_PASS | A contract marks a capability required and no permitted Provider Pack or valid connection can supply it. | Setup planning and activation are attempted. | Activation is blocked with stable reason code, safe resolution guidance, no Delivery stream, and no provider call. | IR-03, IR-08, IR-12 | Invalid/partial contract fixture; activation API/CLI negative test. | Blocked Setup Result, non-zero exit code, denial audit, provider-call ledger=0. |
| CS-CH-003 | MUST_PASS | Only an optional capability is unavailable. | The contract is planned and activated. | Activation succeeds as ready-with-gaps; unavailable UI/actions are disabled without blocking unrelated capabilities. | IR-03, IR-16 | Setup fixture; API and UI degraded-mode test. | Ready-with-gaps record, warning, active streams for available capabilities, regression transcript. |
| CS-CH-004 | MUST_PASS | Setup recommends one provider/source policy and the owner has authority to confirm or override it. | The owner confirms or selects another declared compatible source. | A new immutable Source Policy snapshot is stored, constraints never broaden silently, and the change is audited. | IR-04, IR-08, IR-17 | Policy normalization unit test; API/UI owner action test; state diff. | Before/after Source Policy snapshots, policy decision, owner identity, audit event. |
| CS-CH-005 | REGRESSION | Two Provider Packs expose the same Common Capability and Projection contract. | Source Policy switches the selected provider. | Product handlers and downstream domain logic continue unchanged; only setup/source references change. | IR-02, IR-08, IR-15 | Provider-swap integration test against common fixtures. | Same product object output, changed provider refs, unchanged handler code hash, audit diff. |
| CS-CH-006 | MUST_PASS | A credential, scope, platform permission, or selected-resource gap exists. | The user opens setup/status surfaces. | CornerStone explains cause, impact, and safe resolution without exposing tokens, paths, handles, or raw provider responses. | IR-09, IR-12, ER-05 | Error-shape unit tests; CLI/API/UI snapshot; secret fixtures. | Redacted status payload, reason code, resolution steps, secret/path scan=0 findings. |
| CS-CH-007 | MUST_PASS | A valid app-scoped Projection Delivery arrives. | The Connector ingress handler processes it. | The exact envelope is preserved as an immutable scoped Artifact/intake record, linked to Projection, Source Policy, EvidenceRef, and connector audit refs. | IR-04, IR-05, IR-06, IR-07 | Integration test with fixture delivery; database/object-store checks. | Artifact checksum, stored envelope bytes, provenance links, scope fields, audit event. |
| CS-CH-008 | MUST_PASS | A Delivery can be redelivered and the process may fail between receipt and commit. | The handler crashes before commit or after commit but before acknowledgement. | No acknowledgement occurs before durable commit; redelivery yields one logical Artifact and acknowledgement only after transaction success. | IR-05, IR-07 | Fault-injection integration test at transaction boundaries. | Crash transcripts, DB rows, unique-key result, ack timeline, acknowledged_without_artifact=0. |
| CS-CH-009 | MUST_PASS | A handler or provider fails transiently, repeatedly, or with malformed payload. | Delivery retry policy is applied. | Transient failures retry with bounds/backoff; poison deliveries enter quarantine with safe diagnostics; unrelated streams continue. | IR-07, IR-17 | Retry clock test; malformed payload integration test; queue state check. | Attempt counts, retry schedule, quarantine item, redacted error, audit events. |
| CS-CH-010 | MUST_PASS | The same provider event or unchanged source content is received multiple times, then later changes. | Deliveries are processed. | Duplicates resolve to one logical intake record; changed content creates a version linked to its predecessor and source revision. | IR-05, IR-07 | Dedupe and one-byte/change fixture tests; lineage query. | Unique dedupe key, content hashes, version lineage, no duplicate search truth. |
| CS-CH-011 | MUST_PASS | Source Policy limits metadata/body fields, content size, selected resources, paths, or raw access. | A Projection arrives with fields beyond the allowed contract. | Disallowed fields are rejected or stripped before durable Product state; a policy decision and helpful error are recorded. | IR-08, IR-09 | Property-based validator tests; oversized/extra-field fixtures. | Normalized Projection, excluded-field list, policy decision, raw-content persistence=0. |
| CS-CH-012 | MUST_PASS | An Artifact was created from a Projection and ConnectorHub supplied EvidenceRef metadata. | A brief, claim, or action requests evidence. | CornerStone assembles an Evidence Bundle containing Artifact, Delivery, Setup Result, Source Policy, EvidenceRef, query/snapshot, and audit refs; EvidenceRef alone is never treated as the original. | IR-06, IR-17 | Evidence assembly unit/integration test; claim creation test. | Evidence Bundle JSON, artifact checksum, connector refs, claim trust state, audit refs. |
| CS-CH-013 | MUST_PASS | Raw access is absent from the contract or declared with strict TTL/read limits. | An app or operator requests raw evidence. | Default request is denied; an allowed grant is purpose-bound, expiring, counted, redacted, scoped, and revocable. | IR-09, IR-11 | Denial test; TTL/max-read boundary tests; log/UI scan. | Denied/granted decisions, expiry proof, read counter, revoked handle, no handle in logs. |
| CS-CH-014 | REGRESSION | Connector content includes instructions to ignore policy, reveal secrets, or trigger a tool/action. | The Projection is archived, summarized, searched, or used by an agent. | Content remains untrusted evidence; no authority changes, tool calls, connector actions, or external egress are triggered. | IR-10, IR-11, IR-18 | Prompt-injection fixture through ingest/search/agent/action paths. | Untrusted label, blocked tool/action records, external calls=0, audit/policy evidence. |
| CS-CH-015 | MUST_PASS | A GitHub read-only connector is installed for selected repositories. | The owner chooses repositories and activates sync. | Only explicitly selected repositories are visible and ingestible; organization-wide or unselected repository access is denied. | ER-06, IR-04, IR-08, IR-09 | Selected-resource integration test with allowed and denied repo fixtures. | Installation scope, selected repo list, denied access result, provider-call ledger. |
| CS-CH-016 | MUST_PASS | A selected repository contains repository metadata, commits, pull requests, issues, and allowed file snapshots. | Initial or incremental read sync runs. | Versioned source-control Projections map into immutable Artifacts and searchable evidence with source revision and repository provenance. | ER-06, IR-05, IR-06, IR-07 | Fixture sync integration tests per Projection type; search/evidence tests. | Projection fixtures, Artifact records, source refs, search snapshot, Evidence Bundle. |
| CS-CH-017 | MUST_PASS | Polling and webhook signals may overlap, arrive out of order, or repeat. | Incremental GitHub synchronization processes them. | Provider event IDs, cursors, source revisions, and content hashes produce idempotent ordered state without missed or duplicated logical content. | ER-06, IR-07 | Out-of-order/replay/cursor integration tests. | Cursor snapshots, dedupe records, reconciliation report, no duplicate Artifact truth. |
| CS-CH-018 | MUST_PASS | Repository content may be large, binary, secret-bearing, generated, or outside allowed paths. | The GitHub mapper evaluates the content. | Policy skips, truncates, redacts, or quarantines content according to path, size, type, and secret rules; source remains read-only. | ER-06, IR-08, IR-09, IR-10 | Boundary fixtures for bytes, binary type, secret markers, paths, and prompt injection. | Decision records, redacted Artifact, skipped/quarantine state, secret scan=0. |
| CS-CH-019 | REGRESSION | Any Product, agent, pack, CLI, API, or provider path attempts a GitHub mutation. | A write-like operation is requested or an undeclared endpoint is reached. | The operation is denied before network execution; no GitHub write capability, action schema, token scope, route, or UI control exists. | ER-06, IR-09, IR-10, IR-11 | Static capability/permission scan; API/CLI negative tests; controlled egress ledger. | actions=[] contract, absence scan, denied result, github_write_calls=0. |
| CS-CH-020 | MUST_PASS | GitHub rate limits, revoked installation permission, repository removal, or transient transport errors occur. | Sync or status runs. | CornerStone preserves existing evidence, reports freshness/setup gaps, retries safely when appropriate, stops removed scopes, and never fabricates current data. | ER-06, IR-07, IR-08, IR-17 | Failure simulations and stale-data API/UI tests. | Rate-limit record, retry time, revoked/removed state, freshness marker, audit events. |
| CS-CH-021 | MUST_PASS | WatchAgent is installed on a supported or unsupported host without consent/permission. | Collection starts or status is checked. | No activity is captured until explicit consent and required platform permissions exist; unsupported/missing states are explained safely. | ER-05, IR-14 | Fixture unit tests plus supported-host integration harness; physical proof deferred. | Consent record, permission state, capture_before_consent=0, status/audit output. |
| CS-CH-022 | MUST_PASS | Permissioned activity samples include app switches, idle gaps, duplicates, and low-information noise. | Sessionization runs. | Samples become bounded ActivitySession Projections with deterministic boundaries, confidence, caveats, and no unsupported intent claim. | ER-05, IR-07, IR-14 | Deterministic timeline unit tests with boundary values and idle windows. | Input samples, resulting sessions, confidence/caveats, dedupe and retention records. |
| CS-CH-023 | MUST_PASS | An owner wants monitoring for a project, source, condition, and time window. | The owner creates, pauses, resumes, edits, or deletes a Watch Rule. | The rule is explicit, scoped, source-limited, reviewable, audited, and cannot broaden capture without a new confirmation. | ER-04, ER-05, IR-04, IR-14 | Domain/API/UI lifecycle tests; cross-scope denial tests. | WatchRule versions, policy snapshot, owner action, audit timeline. |
| CS-CH-024 | MUST_PASS | The Chrome extension is loaded and the user invokes capture on the active tab. | The extension requests temporary active-tab access and submits a bounded payload to the local backend. | Only the active page is considered; policy is rechecked server-side; summary/evidence is created or safely blocked without storing raw HTML/text. | ER-05, IR-10, IR-14 | Extension unit tests, mocked Chrome API tests, backend contract tests, manual browser proof. | Permission event, payload, backend decision, summary/evidence, raw_text/html_stored=false. |
| CS-CH-025 | MUST_PASS | Auto capture is configured for approved domains/source packs with browser permission and matching backend consent/config versions. | A page-load, URL-change, or tab-activate trigger occurs. | Capture occurs only on an active allowed page within throttle/session limits; consent/config mismatch or missing permission blocks it. | ER-05, IR-08, IR-14 | Extension state-machine tests; backend consent/config/idempotency tests. | Consent and config snapshots, permission status, history/timeline, unapproved_domain_captures=0. |
| CS-CH-026 | MUST_PASS | A page is sensitive, contains password/payment/secret fields, is browser-internal/private, or is an unknown editable surface. | Capture policy evaluates it. | The page is blocked or degraded to domain/hash-only metadata; no forbidden content is sent or stored, and the reason is visible. | ER-05, IR-09, IR-10, IR-14 | Policy unit tests and browser/backend fixtures for every sensitive class. | Decision code, degraded payload, history item, raw persistence=0, UI explanation. |
| CS-CH-027 | MUST_PASS | A user has collected WatchAgent/Chrome state. | The user pauses/revokes sources, changes retention, exports data, dismisses/saves results, or deletes eligible local state. | Collection stops promptly; decisions persist; exports are scoped/redacted; deletion follows retention/audit policy and is explained. | IR-04, IR-14, IR-17 | Lifecycle API/UI tests; state and file checks; retention boundary tests. | Pause/revoke timestamps, export bundle, deletion receipt, retained-audit explanation. |
| CS-CH-028 | MUST_PASS | One or more captured signals support a potential interpretation and next step. | CornerStone creates a Watch Result. | The result visibly separates Observation, Inference, Evidence/Caveats, and Proposed Action or Memory; unsupported inference remains Draft/Hypothesis. | ER-03, IR-06, IR-10, IR-14 | Domain mapping and UI snapshot/browser test; evidence coverage check. | WatchResult JSON, evidence refs, confidence/caveats, trust state, UI proof. |
| CS-CH-029 | MUST_PASS | A separately approved non-GitHub connector action is proposed from a claim or mission. | CornerStone runs ActionCard dry-run and ConnectorHub Action Preflight. | One combined review shows diff/impact, provider support, permissions, Source Policy, risk, idempotency, expected calls, evidence, and approval need; no side effect occurs. | IR-11, IR-12, IR-18 | Dry-run/preflight integration test; call ledger assertion. | ActionCard, preflight record, policy input, expected calls, real calls=0, audit refs. |
| CS-CH-030 | MUST_PASS | A side-effecting connector action lacks evidence, policy allow, authorized approval, required permission, or idempotency key. | Execution is requested. | Execution is denied with a precise cause and resolution; ConnectorHub cannot infer product approval and Product cannot infer connector permission. | IR-11, IR-18 | Combinatorial negative tests for each missing gate. | Denial decisions, zero external calls, audit events, stable exit codes. |
| CS-CH-031 | MUST_PASS | All gates pass for a declared, separately approved non-GitHub action. | WorkflowRun invokes ConnectorHub execution once. | An Action Result is linked to WorkflowRun and audit, external outcome is re-ingested as evidence, and no duplicate side effect occurs. | IR-11, IR-17, IR-18 | Fixture integration test; live human gate for real provider. | Execution request/result, idempotency record, provider receipt, outcome Artifact/Evidence Bundle, audit chain. |
| CS-CH-032 | REGRESSION | An agent, Product module, or extension attempts an undeclared action or direct provider/credential access. | The bypass path is invoked. | Backend enforcement denies it, records the attempt, and exposes no provider client or secret even if UI checks are bypassed. | IR-02, IR-09, IR-10, IR-11 | Static dependency scan; direct-call negative tests; sandbox/egress test. | Denied response, audit event, provider calls=0, credential exposure=0. |
| CS-CH-033 | MUST_PASS | A side-effecting Action is retried because of timeout, duplicate request, or ambiguous provider response. | The same or conflicting idempotency key is submitted. | Same key returns the existing result; conflicting intent is rejected; compensation/rollback expectation is visible when atomic rollback is impossible. | IR-07, IR-11 | Idempotency concurrency tests; timeout/ambiguous-result simulation. | Single provider effect, existing result linkage, conflict denial, compensation plan. |
| CS-CH-034 | MUST_PASS | Two owners/workspaces use connector capabilities with similar external IDs. | They register, poll, search, watch, or execute. | All objects and results remain scoped; no ownerless or cross-namespace Delivery, evidence, memory, or action is returned or executed. | IR-04, IR-18 | Cross-tenant/namespace integration matrix and DB/RLS tests. | Denied cross-scope reads/actions, scoped rows, policy decisions, leak counter=0. |
| CS-CH-035 | REGRESSION | Provider credentials exist or rotate/revoke. | Product, logs, reports, UI, exports, errors, and durable state are inspected. | Only credential references/fingerprints/status are visible; raw secrets and handles never cross the ConnectorHub boundary. | IR-09, IR-13 | Seeded-secret scan over API/CLI/UI/log/state/export artifacts. | Scan report=0 findings, rotation/revocation event, safe connection status. |
| CS-CH-036 | MUST_PASS | ConnectorHub, a tool runtime, or untrusted content attempts an external call outside an explicitly allowed provider route. | Network access is attempted, including redirects, alternate addresses, proxy, subprocess, or protocol variants. | Default-deny egress blocks the call, logs policy reason, and no alternate route bypasses the gateway. | IR-10, IR-18 | Production-like network integration suite; local unit mocks are insufficient for final PASS. | Blocked sink traces, policy decision, DNS/redirect/proxy/subprocess results, egress calls=0. |
| CS-CH-037 | MUST_PASS | Connector setup, delivery, evidence access, policy, action, retry, quarantine, or credential events occur. | CornerStone records the corresponding lifecycle. | Connector event IDs correlate to CornerStone audit events and affected objects without copying secrets/raw payloads; integrity verification still works. | IR-17, IR-18 | Audit contract tests; correlation query; tamper test. | Connector event, CornerStone event, correlation ID, object refs, integrity verification output. |
| CS-CH-038 | MUST_PASS | A contract, Projection schema, SDK protocol, or Provider Pack version changes. | A workspace plans or applies an upgrade. | Pinned versions remain active until reviewed; diff/compatibility/migration plan is shown; incompatible changes block activation; rollback is available. | IR-15, IR-16 | Contract compatibility fixtures; upgrade/downgrade migration tests. | Version pins, diff, migration record, canary result, rollback transcript. |
| CS-CH-039 | REGRESSION | Connector capabilities are added to navigation, onboarding, CLI, and help. | A normal user completes first value and connected-source setup. | The user sees one CornerStone product and plain product concepts; repo/package names and connector internals remain admin details. | IR-01, IR-12, IR-16 | UI/navigation copy scan; onboarding browser test; human UX gate. | First-value transcript without connector setup, connected-source walkthrough, forbidden-name scan. |
| CS-CH-040 | REGRESSION | Fixture tests pass but live credentials, physical devices, human review, or production topology are absent. | A report or release claim is generated. | The report labels fixture/local evidence accurately, keeps live/production items unverified or human-required, and never upgrades readiness by implication. | ER-08, ER-09, IR-13, IR-18 | Report-lint tests and evidence manifest review. | Scenario statuses, human-required table, negative overclaim counter, exact commit/tree metadata. |
| CS-CH-H01 | HUMAN_REQUIRED | A real GitHub account/organization and least-privilege installation are available. | An authorized owner installs the connector for selected repositories and runs a read-only rehearsal. | Read Projections and audit evidence are produced for selected repos, and an independent call/permission review shows zero write capability or write calls. | ER-06, IR-13 | Human/external account rehearsal. | Redacted permissions, selected repos, call ledger, Deliveries, audit refs, zero-write proof. |
| CS-CH-H02 | HUMAN_REQUIRED | A supported physical Mac is available. | A human grants, pauses, revokes, and observes activity permission behavior. | Capture follows visible consent and permission state and stops on pause/revoke without unexpected observation. | ER-05, IR-14 | Physical-device walkthrough. | Recording/screenshots, status transcript, first sample, pause/revoke timestamps. |
| CS-CH-H03 | HUMAN_REQUIRED | Chrome extension and local backend are installed in a real browser profile. | A human reviews setup, active-tab capture, allowlist auto capture, sensitive blocking, pause, and revoke. | The permission/privacy experience is understandable and behavior matches the contract. | ER-05, IR-14 | Human browser/privacy review. | Recording/screenshots, permission pages, timeline, issues, accept/reject note. |
| CS-CH-H04 | HUMAN_REQUIRED | An integrated production-like environment with trusted RequestContext, PostgreSQL/RLS, OPA, and network controls exists. | The corrected security/tenancy/egress scenario suite runs end to end. | Connector operations enforce scope, policy, egress, backup/restore, and audit under real topology. | IR-04, IR-18 | Human-authorized production-like verification. | Scenario-specific DB/policy/network transcripts, reports, backup/restore evidence. |
| CS-CH-H05 | HUMAN_REQUIRED | A separately approved non-GitHub provider and reversible test target exist. | An authorized reviewer approves and executes one live declared Action. | The provider state changes exactly once and all safety-envelope, result, compensation, and audit evidence is captured. | IR-11, IR-13, IR-18 | Human-authorized external mutation. | Approval, redacted request/result, provider receipt/state, idempotency and audit evidence. |
| CS-CH-H06 | HUMAN_REQUIRED | A representative user can use the integrated CornerStone UI. | The user completes Connected Sources and Capture Inbox first-use tasks. | The flow is understandable, trustworthy, and does not require knowledge of internal repositories or connector architecture. | IR-01, IR-14, IR-16 | Human usability/trust study. | Timed task, notes, recording/screenshots, acceptance decision and issue list. |
| CS-CH-H07 | HUMAN_REQUIRED | Production-like durable storage and connector state are available. | An operator performs backup, restore, cursor reconciliation, and audit verification. | Artifacts, evidence, connector cursors, search, quarantine, and audit integrity are recoverable without duplicate or lost logical state. | IR-07, IR-17, IR-18 | Human/operator recovery exercise. | Backup/restore logs, reconciled cursors, replay results, audit verification. |

# Implementation Plan

## Scenario-first implementation batches

| Batch | Scenarios | Smallest complete deliverable | Release posture |
| --- | --- | --- | --- |
| CH-0 - Contract adapter foundation | CS-CH-001-006, CS-CH-038-040 | ConnectorPort, contract registry, Setup Result and Source Policy snapshots, activation state, native CLI/API/UI, fixture provider. | Local deterministic fixture only. |
| CH-1 - Durable connected evidence | CS-CH-007-014, CS-CH-034-037 | Transactional Delivery inbox, Projection-to-Artifact mapping, Evidence Bundle handoff, retries/quarantine, namespace and audit enforcement. | Local fixture; no live-provider claim. |
| CH-2 - GitHub read-only | CS-CH-015-020 | Source-control capability family, selected-repository read-only Provider Pack, incremental sync, content restrictions, zero-write guard. | Live read requires CS-CH-H01; writes remain forbidden. |
| CH-3 - Watch connected evidence | CS-CH-021-028 | macOS consent/permissions, sessionization, Watch Rules, Chrome capture, Capture Inbox, privacy lifecycle, Observation/Inference/Proposal UX. | Physical-device and browser acceptance require H02/H03/H06. |
| CH-4 - Governed Actions (separate extension) | CS-CH-029-033 | ActionCard + ConnectorHub preflight, safety-envelope validation, separately approved non-GitHub execution, outcome re-ingest. | Outside recommended v3 base; blocked by VS2 remediation and H04/H05. |

## Scenario-to-work mapping

| Scenario ID | Phase | Required implementation |
| --- | --- | --- |
| CS-CH-001 | CH-0 | Create ConnectorCapabilityContract, ConnectorSetupResult, ConnectorPort, validation service, persistence, CLI/API commands, activation review UI, and audit event. |
| CS-CH-002 | CH-0 | Add required/optional coverage evaluation, activation state machine, stable errors, UI guidance, and negative provider-call instrumentation. |
| CS-CH-003 | CH-0 | Model optional gaps, feature flags derived from Setup Result, helpful UI labels, and partial-capability regression tests. |
| CS-CH-004 | CH-0 | Add SourcePolicySnapshot, confirmation/override commands, compatibility validation, diff view, and audit correlation. |
| CS-CH-005 | CH-0 | Define provider-neutral Projection interfaces, adapter contract tests, and provider swap fixtures. |
| CS-CH-006 | CH-0 | Normalize setup gaps, add redaction layer at adapter boundary, safe status DTO, operator UI, and observability event. |
| CS-CH-007 | CH-1 | Create ConnectorDeliveryReceipt, projection artifact mapping, content hashing, immutable storage write, provenance model, and ingest audit. |
| CS-CH-008 | CH-1 | Use transactional inbox/outbox pattern, durable processing state, commit-after-write callback, idempotent ack worker, and crash metrics. |
| CS-CH-009 | CH-1 | Add delivery state machine, retry policy, quarantine store, replay command, operator status, and failure counters. |
| CS-CH-010 | CH-1 | Persist provider event ID, delivery idempotency key, content hash, source revision, lineage links, and reconciliation job. |
| CS-CH-011 | CH-1 | Implement ProjectionPolicyValidator, size/path/resource checks, allowlist serialization, and field-level observability. |
| CS-CH-012 | CH-1 | Add ConnectorEvidenceLink mapper, EvidenceBundle builder extensions, provenance viewer, and evidence coverage validation. |
| CS-CH-013 | CH-1 | Define raw-access request/result port, policy bridge, secure handle store, TTL enforcement, count enforcement, revocation, and safe UI. |
| CS-CH-014 | CH-1 | Tag connector artifacts untrusted, separate data/instructions, enforce tool/action policy server-side, and add red-team fixtures. |
| CS-CH-015 | CH-2 | Add source-control capability family, GitHub read-only Provider Pack, selected-resource policy, connection status, and repo selector UI. |
| CS-CH-016 | CH-2 | Define source_control.repository/commit/change/issue/file_snapshot schemas, mappers, hash rules, and derived text handlers. |
| CS-CH-017 | CH-2 | Add cursor and webhook receipt models, dedupe keys, monotonic/reconciliation logic, and sync lag metrics. |
| CS-CH-018 | CH-2 | Implement path/size/type policy, secret scanner, redaction, binary metadata-only mode, and helpful reason codes. |
| CS-CH-019 | CH-2 | Keep GitHub manifest read-only, omit write scopes/actions, add explicit deny policy and regression scan across source, routes, CLI, and UI. |
| CS-CH-020 | CH-2 | Map provider failures to stable states, add stale/freshness metadata, cursor recovery, reconnect guidance, and alerts. |
| CS-CH-021 | CH-3 | Add WatchSourceConsent, permission probe adapter, off-by-default state, source toggles, and setup diagnostics. |
| CS-CH-022 | CH-3 | Implement sample schema, sessionizer, idle/noise thresholds, versioned algorithm metadata, and session metrics. |
| CS-CH-023 | CH-3 | Create WatchRule/WatchRuleVersion, condition validator, lifecycle API/CLI/UI, scope enforcement, and rule evaluation trace. |
| CS-CH-024 | CH-3 | Adopt activeTab capture contract, bounded payload schema, local endpoint authentication/consent check, server revalidation, and Capture Inbox mapping. |
| CS-CH-025 | CH-3 | Implement two-sided consent handshake, domain/source-pack rules, trigger metadata, throttling, idempotency, heartbeat, and diagnostics. |
| CS-CH-026 | CH-3 | Port sensitive-page policy, keep backend authoritative, define degraded schemas, and add reason-to-guidance mapping. |
| CS-CH-027 | CH-3 | Add source lifecycle, decision store, retention job, export service, delete/disable semantics, and privacy settings UI. |
| CS-CH-028 | CH-3 | Define WatchObservation, WatchInference, WatchResult, trust-state rules, evidence viewer, and separate UI sections. |
| CS-CH-029 | CH-4 | Create ConnectorActionPreflight port, ActionCard merge service, expected-impact schema, and review UI/CLI/API. |
| CS-CH-030 | CH-4 | Enforce safety envelope server-side, validate refs and approver authority, bridge OPA/RequestContext, and add denial reason catalog. |
| CS-CH-031 | CH-4 | Implement execution adapter, WorkflowRun linkage, result mapper, outcome re-ingest, and provider receipt persistence. |
| CS-CH-032 | CH-4 | Add import/dependency guard tests, capability allowlist, server-side authorization, and provider-call gateway enforcement. |
| CS-CH-033 | CH-4 | Persist idempotency scope/digest, add execution lock and reconciliation, model compensation expectation, and expose retry status. |
| CS-CH-034 | CH-1 | Require RequestContext at every port, include scope in keys and schemas, enforce DB/RLS/policy, and test all entry points. |
| CS-CH-035 | CH-1 | Use credential-ref DTOs, central redactor, structured logging allowlist, export filters, and rotation status model. |
| CS-CH-036 | CH-1 | Place ConnectorHub behind egress gateway, route through policy, prohibit arbitrary clients, and add topology-level tests/metrics. |
| CS-CH-037 | CH-1 | Add audit ingest/correlation schema, redacted metadata mapper, integrity-link fields, query UI/API, and missing-event contract tests. |
| CS-CH-038 | CH-0 | Create version registry, compatibility matrix, migration hooks, dual-read window where needed, and rollback metadata. |
| CS-CH-039 | CH-0 | Expose Connected Sources/Capture Inbox/Watch Rules within CornerStone; keep connector diagnostics in admin detail; preserve onboarding regression. |
| CS-CH-040 | CH-0 | Add claim-level evidence classifier, report schema/linter, readiness gates, and release wording tests. |
| CS-CH-H01 | Human | Prepare runbook and evidence collector; no automated PASS without actual account evidence. |
| CS-CH-H02 | Human | Prepare signed/local build and runbook; capture physical evidence. |
| CS-CH-H03 | Human | Prepare manual script and evidence checklist; do not substitute mocked extension tests. |
| CS-CH-H04 | Human | Complete VS2 remediation and execute the integrated environment suite. |
| CS-CH-H05 | Human | Prepare safe target and rollback/compensation plan; GitHub remains excluded. |
| CS-CH-H06 | Human | Prepare task script, fixture workspace, and scoring rubric. |
| CS-CH-H07 | Human | Implement runbook and recovery tooling, then perform operator exercise. |

## Domain model

Implement the following CornerStone-owned records. Every truth-bearing record carries `tenant_id`, `owner_id`, `namespace_id`, `workspace_id`, timestamps, security classification, trust state where applicable, and audit references:

- `ConnectorCapabilityContract` and immutable `ConnectorCapabilityContractVersion`.
- `ConnectorSetupResultSnapshot`, `ConnectorConnectionRef`, and `ConnectorSourcePolicySnapshot`.
- `ConnectorDeliveryReceipt`, `ProjectionSnapshot`, `ConnectorProcessingAttempt`, and `ConnectorQuarantineItem`.
- `Artifact`/immutable connector intake record plus `ConnectorEvidenceLink` and `EvidenceBundle` extensions.
- `ConnectorSyncCursor`, `ProviderEventReceipt`, `SourceRevision`, and content-version lineage.
- `WatchSourceConsent`, `WatchRule`, `WatchRuleVersion`, `ActivitySample`, `ActivitySession`, `WatchObservation`, `WatchInference`, and `WatchResult`.
- `ConnectorActionPreflight`, `ConnectorExecutionReceipt`, and `ConnectorAuditCorrelation` for the separately gated Action slice.

## Canonical port and contract

Create a CornerStone-owned `ConnectorPort` interface. Product and agent code depend only on this port, never on Provider Packs or provider SDKs. A minimal interface is:

```python
class ConnectorPort(Protocol):
    def validate_contract(self, contract: ConnectorCapabilityContract) -> ValidationResult: ...
    def plan_setup(self, contract: ConnectorCapabilityContract) -> ConnectorSetupResultSnapshot: ...
    def activate(self, contract_version_id: str, source_policy_id: str, request_context: RequestContext) -> ActivationResult: ...
    def poll_deliveries(self, activation_id: str, request_context: RequestContext) -> list[ProjectionDelivery]: ...
    def acknowledge_delivery(self, delivery_id: str, durable_receipt_id: str, request_context: RequestContext) -> AckResult: ...
    def fail_delivery(self, delivery_id: str, reason_code: str, request_context: RequestContext) -> FailureResult: ...
    def preflight_action(self, request: ConnectorActionRequest, request_context: RequestContext) -> ConnectorActionPreflight: ...
    def execute_action(self, approved_envelope: ApprovedConnectorActionEnvelope, request_context: RequestContext) -> ConnectorExecutionReceipt: ...
```

The first adapter is `connectorhub_client.adapters.connectorhubkit`. It may be in-process initially, but no Product code may assume the transport.

## Source adapter and input mapping

- Map the CornerStone contract into ConnectorHub App Requirements while retaining CornerStone scope and policy references outside the Hub-owned schema.
- Map Setup Result into an immutable CornerStone snapshot and activation decision input.
- Map each Projection envelope to a canonical byte representation before hashing and storing.
- Preserve `delivery_id`, `projection_id`, `provider_pack_id`, `provider_event_id`, `source_revision`, `source_summary`, `evidence_ref_id`, schema versions, and content restrictions.
- For GitHub, add new read-only Common Capabilities and versioned Projections rather than overloading communication/document capabilities.
- For WatchAgent, reuse activity/page-summary Projections through the same generic Delivery path; proof apps receive no privileged bypass.

## Validation and normalization

- Validate contract schema, version compatibility, Common Capability, Projection type, Action declarations, selected resources, retention, body restrictions, and raw-access posture before activation.
- Revalidate every received envelope against the pinned contract and active Source Policy.
- Reject unknown extra fields unless they belong to a declared, versioned provider extension explicitly requested by the contract.
- Normalize stable reason codes and user-safe resolution guidance.
- Treat all provider/browser text as untrusted evidence; never pass it as system/tool instruction.

## Persistence and snapshot strategy

Use a transactional inbox/outbox pattern. The atomic durable unit should include the Delivery receipt, exact Projection bytes/hash, Artifact/intake record, provenance/evidence links, source cursor candidate, and audit event. Only after commit may the acknowledgement worker acknowledge the Delivery. A unique key should cover at least the scoped activation plus provider event/delivery identity; a separate content hash governs unchanged-content deduplication.

Changed content creates a new version linked to the previous Artifact/source revision. Cursor advancement is committed with the processed receipt or reconciled from receipts after restart. Quarantine stores only safe metadata and references to Hub-controlled diagnostics, never raw secrets or provider payloads.

## Proposed API behavior

The following is a proposed CornerStone surface, not a claim that these routes already exist:

```text
POST /connector-contracts
POST /connector-contracts/{id}/validate
POST /connector-contracts/{id}/plan
POST /connector-contracts/{id}/activate
GET  /connected-sources
GET  /connected-sources/{id}/status
POST /connected-sources/{id}/poll
GET  /connector-deliveries/{id}
GET  /connector-quarantine
POST /connector-quarantine/{id}/replay
POST /watch-rules
PATCH /watch-rules/{id}
GET  /capture-inbox
POST /capture-inbox/{id}/save|dismiss|restore|feedback
POST /actions/{id}/connector-preflight
POST /actions/{id}/execute  # separately gated, non-GitHub only
```

Every route derives scope from trusted `RequestContext`; user-supplied scope fields are treated as requested context, not authority.

## Proposed native CLI behavior

```text
cornerstone connector contract validate --file ... --json
cornerstone connector setup plan --contract-id ... --json
cornerstone connector source confirm --setup-id ... --json
cornerstone connector activate --contract-id ... --json
cornerstone connector status --activation-id ... --json
cornerstone connector poll --activation-id ... --json
cornerstone connector quarantine list|replay ... --json
cornerstone watch rule create|pause|resume|show ... --json
cornerstone watch inbox list|save|dismiss|feedback ... --json
cornerstone scenario verify connector-contract-adapter --json
cornerstone scenario verify connector-read-ingestion --json
cornerstone scenario verify github-read-only --json
cornerstone scenario verify watch-connected-evidence --json
cornerstone scenario verify connector-governed-action --json
```

CLI responses include stable exit codes, scope, object IDs, evidence refs, audit refs, policy decision refs, negative-evidence counters, and no secrets.

## UI behavior

- **Connected Sources:** purpose, selected resources, readiness, Source Policy, permissions, freshness, last sync, and setup gaps.
- **Capture Inbox:** Observation, Inference, Evidence/Caveats, Proposed next step, save/dismiss/feedback, source and freshness.
- **Watch Rules:** explicit sources, conditions, scope, schedule/window, status, pause/resume, and change history.
- **Privacy controls:** source toggles, domain/source-pack controls, retention, export, delete/disable, and permission diagnostics.
- **ActionCard:** combined Product dry-run and ConnectorHub preflight. GitHub never displays write actions.
- Internal ConnectorHub details are available in an admin drill-down, not required for ordinary first value.

## Downstream reader behavior

- Search, brief, claim, memory, ontology, Watch, and action readers require matching owner/namespace/workspace scope.
- Every generated output carries the active Source Policy/content restriction and evidence lineage.
- Evidence-backed trust states remain `Draft -> Evidence-backed -> Approved`; connector output does not auto-promote truth.
- A connector deletion or permission loss does not erase already retained immutable evidence unless retention policy explicitly permits deletion; freshness and source availability are updated.
- Action outcomes are re-ingested as new evidence rather than mutating history in place.

## Migration and backward compatibility

- Preserve existing upload/search/claim/action local flows and first-value onboarding without connector setup.
- Introduce connector tables and ports additively; do not replace current working storage in the first batch.
- Treat existing WatchAgent proof-app state as importable evidence/fixture data, not canonical CornerStone truth.
- Pin contract and Provider Pack versions; support explicit migration/canary/rollback records.
- During schema transitions, dual-read or reprocess from immutable Projection Artifacts rather than silently rewriting evidence.

## Error handling and observability

Use stable reason codes for invalid contract, required capability missing, source permission gap, stale credential, source removed, Projection invalid, policy restriction, duplicate, retry scheduled, quarantine, cross-scope denial, action gate missing, and egress denial. Each error states cause, preserved safe state, retry/resolution path, and audit reference.

Metrics/logs should cover setup readiness, sync lag, delivery age, retries, quarantine depth, dedupe rate, acknowledgement latency, freshness, policy denials, capture blocks/degrades, redaction findings, external call counts, and audit-correlation gaps. Logs are structured and allowlisted; raw provider payloads are not logged.

# Verification Plan

## Automatically verifiable items

1. **Contract unit suite:** JSON/YAML schema, required/optional capabilities, version compatibility, provider extensions, Action declarations, retention and raw-access boundaries.
2. **Adapter integration suite:** App Requirements/Setup Result mapping, provider-neutral swap, scoped activation, gap handling, redaction, and deterministic fixtures.
3. **Durable-ingress suite:** exact envelope hashing, transactional commit/ack, crash injection, duplicate/reordered Delivery, changed-content lineage, retries, and quarantine.
4. **Evidence/downstream suite:** Artifact and Evidence Bundle linkage, search reproducibility, claim trust states, Watch Result evidence, and source restriction propagation.
5. **GitHub read-only suite:** selected repositories, Projection families, pagination/cursor/replay, size/path/binary/secret policy, revoked permission, removed repository, rate limit, and zero-write static/runtime guards.
6. **Watch suite:** consent off-by-default, sample sessionization, Watch Rule lifecycle, Chrome bounded payload, two-sided consent, sensitive block/degrade, throttling, Capture Inbox decisions, retention/export/delete.
7. **Action suite:** combined dry-run/preflight, missing-gate denial, idempotency/concurrency, result/outcome mapping, direct-provider bypass denial. This suite remains fixture-only until the human live-action gate.
8. **Security/regression suite:** RequestContext scope, cross-tenant/namespace denial, seeded-secret scans, prompt injection, egress topology, audit correlation/integrity, version migration, one-product UX, and report-overclaim lint.

## Proposed database and state checks

- No ownerless connector rows.
- One logical processed record per scoped provider event/idempotency key.
- Every acknowledged Delivery references a committed immutable Artifact/intake record.
- Every connector-derived Evidence Bundle links its Artifact, Delivery, Source Policy, Setup Result, and EvidenceRef metadata.
- No credentials, raw local paths, provider raw payload markers, or raw access handles appear in durable Product state.
- GitHub contract and Provider Pack contain no write Action or write permission.
- Watch capture records have consent/config/policy references and `raw_text_stored=false`, `raw_html_stored=false` where required.

## Required evidence package per scenario run

- Frozen scenario contract/matrix and exact source commit/tree/dirty state.
- Contract fixture, Setup Result, Source Policy, Projection, Delivery, Artifact, Evidence Bundle, and relevant action/audit records.
- CLI/API transcripts with stable exit codes and object/evidence/audit/policy references.
- Database/object-store state checks and hash manifest.
- UI/browser proof only for rows claiming UI behavior.
- Secret/redaction scan, provider/network call ledger, and negative-evidence counters.
- Failure reverse-engineering record for FAIL/NOT_VERIFIED/NOT_RUN rows.
- Human evidence attached only to HUMAN_REQUIRED rows.

## Required negative-evidence counters

```text
unauthorized_provider_calls = 0
provider_credentials_exposed = 0
raw_provider_payloads_exposed = 0
cross_namespace_deliveries = 0
acknowledged_without_artifact = 0
undeclared_actions_executed = 0
github_write_calls = 0
unapproved_egress_calls = 0
capture_before_consent = 0
unapproved_domain_captures = 0
raw_browser_text_persisted = 0
production_readiness_overclaims = 0
```

## Human-required checks

| ID | Why automation is insufficient | Required human action | Expected evidence | Release impact |
| --- | --- | --- | --- | --- |
| CS-CH-H01 | Requires a real GitHub account/organization and external permission state. | Run selected-repository read-only rehearsal. | Redacted permissions, selected repos, calls, Deliveries, audits, zero-write proof. | Blocks live GitHub readiness. |
| CS-CH-H02 | Requires a physical supported Mac and OS permission prompts. | Grant, pause, revoke, and observe capture. | Recording/screenshots and status/sample evidence. | Blocks physical-device Watch readiness. |
| CS-CH-H03 | Requires real Chrome permission UX and subjective privacy review. | Review manual/auto capture, sensitive block, pause/revoke. | Recording/screenshots, timeline, accept/reject note. | Blocks Chrome/privacy acceptance. |
| CS-CH-H04 | Requires production-like RLS/OPA/network/backup topology. | Run corrected integrated security suite. | Scenario-specific DB/policy/network/recovery evidence. | Blocks production connected-source release. |
| CS-CH-H05 | Mutates an external non-GitHub system and needs authorization. | Approve and execute a reversible live declared Action. | Approval, provider receipt/state, idempotency, audit. | Blocks live Action readiness. |
| CS-CH-H06 | Usability/trust is subjective. | Complete Connected Sources/Capture Inbox task study. | Timed task, notes, recording, decision. | Blocks human UX acceptance. |
| CS-CH-H07 | Requires production-like recovery infrastructure. | Perform backup/restore/reconciliation exercise. | Recovery logs, cursor reconciliation, evidence/audit replay. | Blocks production operations readiness. |

## Not-yet-verifiable items

- Live GitHub and browser/device claims until H01-H03 evidence exists.
- Production scope/policy/egress/recovery claims until corrected VS2 and H04/H07 evidence exists.
- Live side-effecting Action claims until H05 evidence exists.
- Human trust/usability claims until H06 evidence exists.

# Completion Criteria

The implementation can be considered complete for a frozen batch only when:

1. Every applicable `MUST_PASS` row is `PASS` with concrete scenario-specific evidence.
2. Every applicable `REGRESSION` row is `PASS` with concrete evidence.
3. No AI-verifiable row is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.
4. Every `HUMAN_REQUIRED` row states the required action, expected evidence, and release impact; release claims do not exceed completed human gates.
5. Native CLI/API/UI parity exists for each claimed Product feature.
6. The evidence package contains exact source state, transcripts, object/state checks, audit refs, negative counters, and failure analysis.
7. GitHub write capability and calls remain zero.
8. Production readiness is not claimed until VS2 scenario-specific security evidence and required human gates are complete.
9. The final verdict equals the weakest applicable scenario result.

A batch may be locally fixture-ready while live or production release remains blocked. These are separate verdicts.

# Risks

| Risk | Failure mode | Mitigation / required guard |
| --- | --- | --- |
| Boundary erosion | Provider SDKs or credentials may leak into Product code because direct integration feels faster. | Enforce ConnectorPort imports, dependency scans, app-facing DTO allowlists, and CS-CH-032/035 regressions. |
| Evidence inconsistency | Acknowledging a Delivery before durable Artifact commit can lose evidence or create duplicate truth. | Use transactional inbox/outbox, content hashes, unique keys, lineage, and fault-injection tests. |
| Surveillance perception | WatchAgent may feel invasive even if technically permissioned. | Off by default, explicit rules, observation/inference separation, visible pause/revoke/delete, human privacy review. |
| Read-only drift | GitHub scope could expand from evidence ingestion into automation or mutation. | Empty Action contract, no write scopes/routes/UI, egress deny, static scans, and zero-write call ledger. |
| Security overclaim | Fixture or wrapper tests may be mistaken for production scope, policy, and network proof. | Exact scenario validators, evidence classifier, report linter, and CS-CH-040 release guard. |
| Contract/version drift | Provider or schema changes can silently alter behavior. | Pinned versions, compatibility matrix, migration reports, canary verification, rollback. |
| Operational overload | Connector failures, retries, quarantine, and setup gaps may overwhelm normal users. | Plain-language product surfaces, admin drill-down, actionable errors, health/freshness metrics. |
| Cross-namespace leakage | External IDs and cached projections may collide across owners/workspaces. | Trusted RequestContext, scoped keys, RLS/policy, cross-scope tests, no ownerless records. |

# Open Questions

- Which GitHub authentication model is preferred: GitHub App, fine-grained personal token for local-only use, or another approved mechanism?
- Which source-control Projection types are mandatory for the first release: repository, commit, pull request/change, issue, discussion, release, and/or file snapshot?
- Should ConnectorHub run in-process for the first slice or as a separate local service behind a stable transport contract?
- What exact bytes constitute the CornerStone immutable connector Artifact: canonical Projection envelope only, an exported provider snapshot, or both when policy permits?
- What default retention periods apply to GitHub evidence, macOS samples, Activity Sessions, Chrome summaries, Capture Inbox state, and connector audit records?
- How should existing WatchAgent/Daily Review local state be migrated or imported into CornerStone without treating proof-app state as canonical truth?
- Is the Chrome extension a first-party CornerStone companion, an Agent Pack component, or a separately versioned provider component?
- Does product v3 end at connected evidence and Watch Results, or should a separately approved non-GitHub Action be included after VS2 remediation?
- Which current CornerStone persistence path is acceptable for the first implementation before Postgres/RLS is fully verified?
- Who is the authorized owner for Source Policy confirmation, connector activation, privacy defaults, and human release gates?

# Final Verdict

**Document verdict:** The scenario-based implementation contract is complete enough for owner review and scenario freeze.

**Product implementation verdict:** `NOT_VERIFIED`. This document does not claim that the CornerStone ConnectorHub adapter, GitHub read-only connector, integrated WatchAgent, or production security path has been implemented or passed.

**AI-verifiable scope:** needs-follow-up until the selected batch is implemented and every applicable MUST_PASS/REGRESSION row passes with evidence.

**Human/release gate:** needs-human-verification for the applicable H01-H07 rows.

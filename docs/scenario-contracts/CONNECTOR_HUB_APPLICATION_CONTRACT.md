# Connector Hub Application Scenario Contract

**Date:** 2026-06-23
**Owner:** JiYong / Tars
**Status:** Frozen for CH-0 independent delivery units `CS-CH-001` through `CS-CH-006` and `CS-CH-038` through `CS-CH-040`, plus the CH-1 `CS-CH-007` Projection Delivery archive unit, `CS-CH-008` durable-ack unit, `CS-CH-009` retry/quarantine unit, `CS-CH-010` dedupe/version-lineage unit, `CS-CH-011` Source Policy field/body enforcement unit, `CS-CH-012` Evidence Bundle promotion unit, `CS-CH-013` temporary raw-access unit, `CS-CH-014` untrusted connector-content guard, `CS-CH-034` owner/namespace scope isolation, `CS-CH-035` credential custody regression guard, `CS-CH-036` current VS2 local default-deny egress topology guard, `CS-CH-037` audit-correlation guard, CH-2 `CS-CH-015` selected GitHub repository boundary, `CS-CH-016` source-control Projection family ingestion, `CS-CH-017` incremental sync idempotency, `CS-CH-018` GitHub content restriction and secret hygiene, `CS-CH-019` GitHub zero-write guard, `CS-CH-020` GitHub provider failure-state handling, CH-3 `CS-CH-021` macOS capture consent/permission gating, `CS-CH-022` bounded activity-session projection, `CS-CH-023` owner-scoped Watch Rule lifecycle, `CS-CH-024` explicit Chrome active-tab capture, `CS-CH-025` allowlist-based Chrome auto capture, `CS-CH-026` sensitive Chrome page block/degrade policy, `CS-CH-027` capture lifecycle controls, `CS-CH-028` Watch Result truth separation, CH-4 `CS-CH-029` ActionCard/preflight integration, `CS-CH-030` action safety gates, `CS-CH-031` declared Action execution, `CS-CH-032` action bypass denial, and `CS-CH-033` idempotent retries; current matrix state is 40 AI-owned PASS rows plus 7 explicit `HUMAN_REQUIRED` human/external gates, with no generic status-neutral backlog rows in this contract.

## Goal

Validate and evolve the Connector Hub application for CornerStone through independent scenario-first delivery units.

The verified local fixture units are:

```text
CS-CH-001 -> Register a connector-backed CornerStone capability
CS-CH-002 -> Required capability missing blocks activation
CS-CH-003 -> Optional capability degrades gracefully
CS-CH-004 -> Owner confirms or overrides Source Policy
CS-CH-005 -> Swap providers without changing Product logic
CS-CH-006 -> Explain credential and permission gaps without secrets
CS-CH-007 -> Convert a Projection into an immutable Artifact
CS-CH-008 -> Acknowledge only after durable archive commit
CS-CH-009 -> Retry transient failures and quarantine poison deliveries
CS-CH-010 -> Deduplicate provider events and version changed content
CS-CH-011 -> Enforce field and body restrictions from Source Policy
CS-CH-012 -> Promote EvidenceRef metadata into an Evidence Bundle
CS-CH-013 -> Temporary raw access is denied and tightly bounded
CS-CH-014 -> Untrusted connector content cannot direct agents or actions
CS-CH-015 -> Connect only explicitly selected GitHub repositories
CS-CH-016 -> Ingest repository, commit, change, issue, and file-snapshot Projections
CS-CH-017 -> Incremental GitHub sync is idempotent
CS-CH-018 -> Apply GitHub content restrictions and secret hygiene
CS-CH-019 -> Deny every GitHub write path
CS-CH-020 -> Handle GitHub rate limits, revoked permissions, and repository removal
CS-CH-021 -> macOS capture is off until explicit consent and permission
CS-CH-022 -> Activity samples become bounded privacy-safe sessions
CS-CH-023 -> Watch Rules are owner-scoped, versioned, and auditable
CS-CH-024 -> Chrome active-tab capture is explicit, bounded, and backend-validated
CS-CH-025 -> Chrome auto capture is allowlisted, consented, and summary-only
CS-CH-026 -> Sensitive Chrome pages are blocked or degraded before capture content
CS-CH-027 -> Capture lifecycle controls pause, revoke, retain, export, review, and delete eligible local state
CS-CH-028 -> Watch Results separate observed facts, inference, caveats, and proposed next steps
CS-CH-029 -> Combine ActionCard dry-run with ConnectorHub preflight
CS-CH-030 -> Require evidence, policy, and authorized approval
CS-CH-031 -> Execute a declared Action and re-ingest its outcome
CS-CH-032 -> Deny undeclared Actions and direct provider bypass
CS-CH-033 -> Make retries idempotent and expose compensation
CS-CH-034 -> Bind every connector app, Delivery, and Watch to owner and namespace
CS-CH-035 -> Keep credentials exclusively inside ConnectorHub
CS-CH-036 -> Enforce default-deny egress around ConnectorHub and tools
CS-CH-037 -> Correlate connector audit with CornerStone audit
CS-CH-038 -> Version contracts pin Provider Packs and migrate safely
CS-CH-039 -> Present one CornerStone product, not a ConnectorHub sub-product
CS-CH-040 -> Separate fixture proof from live and production claims
```

These units prove the first CornerStone-owned ConnectorPort boundary: a versioned connector capability contract can be validated and registered, then converted into a scoped Setup Result and Source Policy snapshot without leaking provider internals or making provider calls before activation. They also prove that a missing required capability blocks activation with stable guidance, no delivery stream, and no provider call, while a missing optional capability keeps available streams enabled and exposes disabled-surface metadata for the unavailable surface. Owner Source Policy confirmation persists immutable narrowed snapshots and denies attempted broadening with audit evidence. Provider-swap proof shows that provider refs can change while the Product handler contract, projection contract, and preview object remain unchanged. Permission-gap proof adds owner-safe cause, impact, resolution, and redaction metadata without exposing credentials or raw provider details. The CS-CH-007 proof archives a valid app-scoped Projection Delivery as an immutable scoped Artifact with exact envelope bytes, delivery receipt, Projection snapshot, Source Policy link, EvidenceRef metadata, and audit refs before any Product interpretation or acknowledgement. The CS-CH-008 proof adds a local inbox/outbox ack boundary: simulated pre-commit crash sends no ack and writes no durable delivery state, simulated post-commit/pre-ack crash leaves a pending ack outbox, and redelivery reuses one logical Artifact before acknowledging after commit. The CS-CH-009 proof adds bounded retry state and safe quarantine: transient failures schedule deterministic backoff without acknowledgement, poison deliveries reach quarantine with redacted diagnostics at the configured threshold, healthy deliveries on unrelated streams continue, and replay requests link to the quarantine record without erasing failure evidence. The CS-CH-010 proof adds connector-owned idempotency and version lineage: repeated provider events or unchanged source content resolve to one logical intake record, changed content creates a linked version with source revision evidence, and lineage query output separates current truth from immutable historical evidence. The CS-CH-011 proof adds Source Policy enforcement at the Projection boundary: permitted metadata/body-preview fields pass with a policy decision record, forbidden full-body fields are blocked before Artifact/receipt/current-state creation, narrowed max-content limits apply to subsequent deliveries, and negative evidence scans show zero raw/full-body leakage into durable state. The CS-CH-012 proof promotes connector EvidenceRef metadata through a normal CornerStone Evidence Bundle that links Artifact, Delivery, Setup Result, Source Policy, EvidenceRef, query/search snapshot, claim, policy, and audit refs; EvidenceRef-only and zero-evidence approval paths are denied, so connector metadata is never treated as original truth by itself. The CS-CH-013 proof denies raw access by default, allows only declared Source Policy grants that are purpose-bound, human-approved, TTL-limited, read-counted, redacted, scoped, revocable, and audited, and exports only metadata without raw content or reusable handles. The CS-CH-014 proof labels connector content as untrusted evidence, allows it to be quoted and cited only through evidence paths, records blocked unsafe instructions, and proves zero tool calls, action cards, workflow runs, provider calls, shell calls, external HTTP calls, memory promotions, policy overrides, or authority expansion. The CS-CH-015 proof adds the first CH-2 selected GitHub repository boundary: a local installation fixture exposes three repositories, only one owner-selected opaque source ref is visible and ingestible, unselected repository events are denied before Artifact/receipt/ack creation, direct GitHub writes are denied, and silent selection broadening is blocked. The CS-CH-016 proof expands that selected repository into provider-neutral source-control evidence: repository, commit, change, issue, and allowed file-snapshot Projections archive as immutable Artifacts, acknowledge after commit, create source-revisioned content versions, and assemble searchable Evidence Bundles without requiring GitHub-specific Product fields. The CS-CH-017 proof adds incremental sync idempotency: invalid webhook boundary metadata is denied before delivery commit, webhook/poll overlap resolves to one logical result, post-commit/pre-cursor gaps are detected by reconciliation before replay, replay advances the cursor only after durable commit, and out-of-order source revisions preserve lineage without duplicate active truth. The CS-CH-018 proof adds GitHub content restriction and secret hygiene: token-like excerpts are redacted before Artifact creation, binary and oversized file content becomes metadata-only Product state, forbidden/generated paths are skipped before Artifact or receipt creation, private-key material is quarantined, and state-wide scans prove zero raw sensitive or out-of-policy markers. The CS-CH-019 proof hardens GitHub as a strictly read-only source: write Action declarations are rejected before contract persistence, Provider Packs expose zero write mappings, the Product CLI exposes no GitHub mutation commands, controlled egress and direct runtime write attempts are denied, and negative evidence records zero write calls or provider mutations. The CS-CH-020 proof turns provider failure fixtures into stable recovery states: rate limits create bounded retry/freshness delay metadata, revoked permissions create a permanent setup gap and suspended streams, repository removal stops future ingestion and records source unavailability, transient transport errors retry safely, and existing evidence remains preserved with stale/unavailable warnings. The CS-CH-021 proof adds the first CH-3 WatchAgent privacy boundary: local macOS activity capture is disabled by default, permission probes are metadata-only, consent and platform permission are distinct inspectable records, permission-only and consent-only states stay blocked, both gates create only a ready state, and negative evidence records zero samples, screenshots, hidden startup capture, cross-namespace capture, external calls, or provider mutations. The CS-CH-022 proof turns permissioned foreground-app samples into privacy-safe bounded ActivitySession projections: duplicates are deduped, idle gaps and project hints split sessions, low-information noise is filtered with metrics, app switches remain observed facts, confidence/caveats stay visible, and unsupported intent is never stored as observed truth. The CS-CH-023 proof creates an owner-scoped Product/Mission Watch Rule with explicit sources, connector contract refs, Source Policy refs, allowed outputs, and no external action authority; missing source readiness blocks activation while ready sources activate, pause/resume/delete transitions are audited, edits create a second version without broadening scope, and evaluation traces remain pinned to the original active version. Version-upgrade proof keeps pinned versions active, blocks incompatible Provider Packs, and records rollback metadata. Product-surface proof keeps Connected Sources inside one CornerStone experience. Report-lint proof keeps local fixture readiness separate from live-provider, physical-device, human UX/privacy, publishing, and production security readiness.

The CS-CH-024 proof adds explicit Chrome active-tab capture as a local CLI/runtime fixture: no-consent capture is denied, popup/browser-internal capture is denied without summary or inbox creation, allowed active-tab capture is server-revalidated, and only summary/inbox records with hashes and metadata persist while raw text, raw HTML, cookies, storage, screenshots, form values, browser history, broad `<all_urls>` permission, external calls, and provider mutations remain absent.

The CS-CH-025 proof adds allowlist-based Chrome auto capture as a local CLI/runtime fixture: auto capture is denied before explicit `chrome_auto_capture` source consent and confirmed config, blocked when site/source-pack/version/throttle/session checks fail, allowed only for an active allowlisted page with matching browser permission, and duplicate idempotency keys create no second summary. Durable state persists config, trigger, policy, summary, and Capture Inbox metadata only while raw browser data, unapproved-domain text, broad `<all_urls>` permission, external calls, provider mutations, and Artifacts remain absent.

The CS-CH-026 proof adds sensitive Chrome page block/degrade policy as a local CLI/runtime fixture: password, payment, private-account, browser-internal, unsupported-scheme, and false-safe token-like pages are blocked; mail compose and oversized pages degrade to hash-only metadata. Backend revalidation preserves or increases client preflight restrictions, rejects malicious false-safe payloads, writes owner-visible history guidance, and creates no searchable Artifact, Capture Inbox item, raw browser data, model-send side effect, external call, or provider mutation.

The CS-CH-027 proof adds capture lifecycle controls as a local CLI/runtime fixture: seeded WatchAgent and Chrome capture state can be paused, resumed, revoked, retention-adjusted, exported, reviewed, and deletion-dry-run/executed through native `cornerstone connector capture lifecycle ...` commands. Pause and revoke persist decisions and deny new sample attempts before sample creation; exports are source-scoped and redacted; save/dismiss review decisions persist; deletion receipts explain deleted, disabled, retained, anonymized, and audit-retained state without promising full erasure or deleting audit records.

The CS-CH-028 proof adds Watch Result truth separation as a local CLI/runtime fixture: `cornerstone watch result build` persists source-backed Observations, draft/hypothesis Inferences, Evidence/Caveats, and a non-executing proposal as separate records; `correct`, `approve-memory`, and `review` commands prove correction changes inference history without mutating immutable observations, low-confidence memory approval is denied, and owner review saves only draft non-executing state.

The CS-CH-029 through CS-CH-033 proofs add the first governed ConnectorHub Action lane: ActionCard dry-runs are joined with connector preflight, execution requires evidence/policy/authorized approval/idempotency, declared supportdesk Actions execute once and re-ingest outcomes as evidence, undeclared/direct-provider bypasses are denied, and same-key retries return existing results while conflicting intent is blocked before a second provider effect.

The CS-CH-034 proof adds owner/namespace/workspace scope isolation across connector app setup, Source Policy, Delivery receipt, Artifact, Evidence Bundle, Watch Result, ActionCard, approval/preflight, audit, and cross-scope denial paths. It proves denied cross-scope setup, Delivery, evidence, Watch, and Action attempts disclose only the requested resource scope and create zero other-scope rows, ownerless rows, WorkflowRuns, Action Results, provider receipts, provider mutations, external calls, or credential leaks in the deterministic local fixture.

The CS-CH-035 proof adds a local credential-custody regression guard: ConnectorHub credential lifecycle commands record status, rotation, and revocation with only opaque refs, fingerprints, safe status, evidence refs, and audit refs; a seeded raw canary remains absent from stdout and durable state; static Product/runtime scans show no provider auth imports; provider internals, raw handles, auth headers, credential-bearing URLs, Product secret writes, external calls, provider mutations, and credential leaks all remain zero.

The CS-CH-036 proof reuses the current VS2 local security proof as a ConnectorHub adoption guard: Docker internal networks separate API, worker, tool runtime, governed egress proxy, and controlled provider sink; direct API/worker/tool HTTP and socket attempts to the provider fail; the provider receives zero requests after direct attempts; the governed proxy reaches the provider exactly through the approved path; redirect, DNS rebinding, reserved destination, protocol normalization, direct socket, sandbox/proxy/subprocess, untrusted-content, fail-closed, audit-correlation, and no-secret checks pass. This is current local VS2 topology evidence only; production network-control review remains separate.

The CS-CH-037 proof adds the local ConnectorAuditBridge adoption guard: setup, policy, delivery, evidence, retry, quarantine, action, and credential lifecycle connector events are correlated to CornerStone audit events with stable event IDs, affected object refs, hash-chain metadata, zero copied raw payloads, zero copied secrets, and a tamper test that proves audit verification fails closed.

## Sources

- `/Users/jiyong/Downloads/CornerStone_ConnectorHub_Application_Guide (1).md`
- `/Users/jiyong/Downloads/CornerStone_ConnectorHub_Test_Scenario_Implementation_Document.md`
- `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md`
- `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`
- `docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md`
- `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md`
- `docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md`
- `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_STATE_2026-06-19.md`

## Scope

In scope for CH-0 `CS-CH-001` through `CS-CH-006` and `CS-CH-038` through `CS-CH-040`, plus CH-1 `CS-CH-007`, `CS-CH-008`, `CS-CH-009`, `CS-CH-010`, `CS-CH-011`, `CS-CH-012`, `CS-CH-013`, `CS-CH-014`, `CS-CH-034`, `CS-CH-035`, `CS-CH-036`, `CS-CH-037`, CH-2 `CS-CH-015`, `CS-CH-016`, `CS-CH-017`, `CS-CH-018`, `CS-CH-019`, `CS-CH-020`, CH-3 `CS-CH-021`, `CS-CH-022`, `CS-CH-023`, `CS-CH-024`, `CS-CH-025`, `CS-CH-026`, `CS-CH-027`, `CS-CH-028`, and CH-4 `CS-CH-029`, `CS-CH-030`, `CS-CH-031`, `CS-CH-032`, `CS-CH-033`:

- CornerStone-owned connector contract validation and draft registration.
- Local deterministic fixture adapter behind a ConnectorPort-style boundary.
- Scoped Setup Result persistence.
- Scoped Source Policy snapshot persistence.
- Required capability coverage evaluation.
- Blocked activation state with stable reason code and safe resolution guidance.
- Negative evidence for zero delivery streams and zero provider calls before activation.
- Optional capability coverage evaluation.
- Ready-with-gaps state with enabled available capability streams.
- Disabled-surface metadata for unavailable optional capabilities.
- Owner-confirmed Source Policy snapshots.
- Narrowing-only Source Policy override.
- Audited broadening denial without creating a new policy snapshot.
- Provider Pack selection during setup planning.
- Product handler/projection contract invariance across equivalent Provider Packs.
- Owner-safe credential/permission gap status explanation.
- Redaction flags proving no token, secret, raw provider response, path, or handle exposure.
- Projection Delivery fixture ingest for `CS-CH-007`.
- Exact Projection envelope preservation as an immutable scoped Artifact.
- ConnectorDeliveryReceipt, ProjectionSnapshot, ConnectorEvidenceLink, Source Policy link, EvidenceRef metadata, and audit refs for `CS-CH-007`.
- Durable ack outbox and replay-safe acknowledgement for `CS-CH-008`.
- Fault injection before archive commit and after archive commit before acknowledgement.
- Redelivery reconciliation proving one logical Artifact and no acknowledged-without-artifact state.
- Negative evidence for no Product interpretation before archive commit, no acknowledgement before durable commit, and no duplicate downstream effects.
- Bounded retry state for transient Delivery failures in `CS-CH-009`.
- Deterministic retry schedule and attempt history from the active contract retry policy.
- Poison Delivery quarantine after the configured threshold.
- Safe quarantine diagnostics with redacted error, source/capability impact, replay link, and no raw provider payload.
- Unrelated healthy Delivery processing while a poison Delivery is quarantined.
- Connector-owned delivery idempotency key for repeated provider events and unchanged source content in `CS-CH-010`.
- Source external ID, source revision, source content hash, content-version record, current-version pointer, and predecessor links for changed content in `CS-CH-010`.
- Lineage query CLI output proving one current logical truth, inspectable predecessor versions, and immutable historical evidence for `CS-CH-010`.
- Projection Source Policy enforcement decision for `CS-CH-011`.
- Permitted-field normalization before Product durable state for `CS-CH-011`.
- Rejection of forbidden full-body/raw fields before Artifact, receipt, current-version, or Product state creation for `CS-CH-011`.
- Narrowed Source Policy max-content limit application to subsequent deliveries for `CS-CH-011`.
- Negative evidence for zero forbidden full-body/raw content leaks in local connector state for `CS-CH-011`.
- Connector-derived Evidence Bundle assembly for `CS-CH-012`.
- Search/query snapshot creation from committed connector evidence for `CS-CH-012`.
- Evidence-backed Claim creation and approval using the connector Evidence Bundle for `CS-CH-012`.
- Denial of EvidenceRef-only bundle creation and zero-evidence Claim approval for `CS-CH-012`.
- Negative evidence for zero EvidenceRef-only approved truth and zero inaccessible phantom evidence for `CS-CH-012`.
- Default raw-access denial for contracts and Source Policies that do not explicitly declare temporary scoped raw access for `CS-CH-013`.
- Declared temporary scoped raw-access request and grant metadata for `CS-CH-013`, including purpose, classification, TTL, read count, human approval, evidence ref, Source Policy, and audit refs.
- Raw-access metadata-only export, read-count exhaustion, deterministic expiry, and revocation for `CS-CH-013`.
- Negative evidence for zero reusable raw handles, zero raw provider payloads, zero raw content copies, and zero raw handle leakage for `CS-CH-013`.
- Untrusted connector content review for `CS-CH-014`, including source trust label, Artifact trust state, unsafe-instruction detection, blocked attempts, and evidence-only handling flags.
- Evidence Bundle trust-boundary coverage for `CS-CH-014`, proving quoted connector instructions are not system instructions or policy authority.
- Agent prompt-authority denial, memory quarantine, and default egress denial commands for `CS-CH-014`.
- Negative evidence for zero tool calls, action cards, workflow runs, connector actions, provider calls, shell calls, external HTTP calls, memory promotions, policy overrides, and authority expansions from untrusted connector content.
- Selected GitHub repository Source Policy scope for `CS-CH-015`, including three-repository installation fixture count, one selected opaque source ref, namespace-scoped selection version, and no organization/account-wide fallback.
- Processing of one selected-repository Projection Delivery for `CS-CH-015`.
- Rejection of one unselected-repository Projection Delivery for `CS-CH-015` before Artifact, receipt, or acknowledgement creation.
- Direct GitHub write denial and Source Policy selection-broadening denial for `CS-CH-015`.
- Negative evidence for zero unselected-repository Artifacts, zero unselected-repository receipts, zero unselected-repository acknowledgements, zero organization-wide fallback, zero GitHub write permissions, zero GitHub write calls, and zero silent selected-repository broadening.
- Selected-repository source-control Projection family ingest for `CS-CH-016`, including repository, commit, change, issue, and allowed file-snapshot fixtures.
- Immutable Artifact, delivery receipt, Projection snapshot, evidence link, ack outbox, content version, search snapshot, and Evidence Bundle persistence for every `CS-CH-016` Projection type.
- Source revision, source external ID, source ref, selected repository provenance, and repository scope checks for every `CS-CH-016` Projection type.
- Negative evidence for zero missing source-control Artifacts, zero missing searchable evidence, zero missing source revisions, zero Product-required provider-specific fields, zero raw provider payload leaks, and zero acknowledgements before archive commit.
- Local incremental sync fixture flow for `CS-CH-017`, including invalid webhook denial, valid webhook receipt, overlapping poll observation, crash/replay, cursor reconciliation, and out-of-order webhook observation.
- Provider-event key composition for `CS-CH-017` from provider installation, repository, object, action, and source revision.
- Connector sync signal receipt, sync cursor, sync reconciliation, delivery receipt, Artifact, content-version, and lineage evidence for `CS-CH-017`.
- Negative evidence for zero duplicate logical Artifacts, zero cursor advancement before durable commit, zero missed cursor receipts after replay, zero duplicate product events, zero source-revision lineage gaps, and zero unverified webhook commits.
- Content restriction decision records for `CS-CH-018`, including redact, metadata-only, skip, and quarantine actions.
- GitHub file content fixtures for token-like text, binary content, oversized content, forbidden paths, generated paths, and private-key material.
- Redacted sanitized Artifact input for permitted secret-like excerpts.
- Metadata-only Artifact input for binary and oversized file content.
- Pre-Artifact skip for forbidden and generated paths.
- Quarantine before Artifact, receipt, or acknowledgement creation for private-key material.
- Negative evidence for zero raw sensitive marker leaks, zero imports outside allowed paths, zero generated Artifacts, zero binary raw-content imports, zero large-file silent truncation, and zero private-material Artifacts.
- GitHub/source-control write Action declaration rejection for `CS-CH-019`.
- Static Provider Pack, active contract, CLI command, and runtime source scan for zero GitHub write surfaces.
- Controlled local egress matrix covering issue creation, comments, labels, file writes, pull-request merge, branch creation, repository settings update, and comment deletion.
- Runtime direct-write denial for every controlled GitHub write operation with zero external HTTP calls and zero provider mutations.
- Negative evidence for zero source-control Actions declared, zero Provider Pack write mappings, zero accepted write contracts, zero exposed GitHub write CLI commands, zero allowed write egress, zero write endpoint literals in runtime command sources, zero GitHub write calls, zero GitHub write permissions requested, and zero provider mutations.
- GitHub provider failure-state fixtures for `CS-CH-020`, including rate limit, revoked permission, repository removal, and transient transport failure.
- Stable health/setup/freshness state records for `CS-CH-020`, including retry schedule, permanent setup gap, stream suspension, source unavailability, owner action requirement, and new-verification requirement.
- Existing-evidence preservation for `CS-CH-020`, including stale/unavailable Search and Claim warning metadata without silent data deletion or fabricated fresh sync claims.
- Negative evidence for zero tight retry loops, zero silent data deletion, zero fresh-current claims while suspended, zero future ingestion from removed repositories, zero reconnect without owner action, zero provider mutations, and zero external GitHub calls.
- Metadata-only macOS capture permission probe for `CS-CH-021`.
- Explicit Watch source consent record for `CS-CH-021`.
- Capture guard decision records for no-consent/no-permission, permission-only, consent-only, and both-gates-present states.
- Setup diagnostics for `CS-CH-021`, including data categories, sampling interval, privacy mode, retention, namespace, pause/delete guidance, and human-required physical-device proof.
- Negative evidence for zero capture before consent, zero capture before platform permission, zero samples before both gates, zero hidden startup capture, zero cross-namespace capture, zero screenshots, zero raw window titles, zero keystrokes, zero clipboard values, zero browser history, zero external HTTP calls, and zero provider mutations.
- Permissioned activity sample batch fixture for `CS-CH-022`, including app switches, duplicates, idle gaps, low-information noise, sparse data, source refs, retention, and privacy flags.
- Deterministic ActivitySession sessionization for `CS-CH-022`, including idle-gap threshold, sample interval, algorithm version, input metrics, filtered-sample records, bounded session IDs, source sample IDs, confidence scores, and caveats.
- ActivitySession projection records for `CS-CH-022`, including observed facts, app/domain categories, project hints, source coverage, evidence refs, audit refs, and explicit separation between observations and inference.
- Negative evidence for zero unsupported intent claims, zero inference stored as observed fact, zero raw window titles, zero full URLs, zero keystrokes, zero clipboard values, zero screenshots, zero cookies, zero browser history, zero external HTTP calls, zero provider mutations, and zero Artifacts created by sessionization.
- Watch Rule definition fixtures for `CS-CH-023`, including goal, explicit sources, match criteria, schedule, sensitivity, namespace/scope, retention, allowed outputs, connector contract refs, and Source Policy refs.
- `WatchRule`, `WatchRuleVersion`, `WatchRulePolicyDecision`, and `WatchRuleEvaluationTrace` records for `CS-CH-023`.
- Missing-source and ready-source activation policy decisions for `CS-CH-023`.
- Pause, resume, reversible delete, and lifecycle audit refs for `CS-CH-023`.
- Versioned edit diff for `CS-CH-023`, with the active version retained and the prior evaluation trace pinned to the original active version.
- Negative evidence for zero ownerless/global Watch Rules, zero cross-namespace lifecycle mutations, zero authority expansion from rule text, zero external actions, zero provider mutations, zero external HTTP calls, zero unconfirmed broadening, and zero Artifacts created by Watch Rule evaluation.
- Chrome active-tab capture fixtures for `CS-CH-024`, including an allowed active-tab payload and a popup/browser-internal blocked payload.
- `ChromeActiveTabPermissionEvent`, `ChromeActiveTabPayload`, `ChromeActiveTabPolicyDecision`, `ChromeActiveTabCaptureSummary`, and `CaptureInboxItem` records for `CS-CH-024`.
- No-consent denial, popup/browser-internal denial, and allowed summary-only capture policy decisions for `CS-CH-024`.
- Backend policy revalidation for `CS-CH-024`, including source scope, owner consent, activeTab temporary access, user gesture, explicit confirmation, active-page scope, bounded payload, and raw browser data absence.
- Negative evidence for zero broad `<all_urls>` permission, zero capture without gesture or confirmation, zero popup capture, zero non-active-tab capture, zero backend policy bypass, zero blocked-page text persistence, zero raw text/HTML, zero cookies, zero storage, zero screenshots, zero form values, zero browser history, zero external calls, zero provider mutations, and zero Artifacts created by active-tab capture.
- Chrome auto-capture config and trigger fixtures for `CS-CH-025`, including no-config, blocked, allowed, and duplicate idempotency paths.
- `ChromeAutoCaptureConfig`, `ChromeAutoCaptureTrigger`, `ChromeAutoCapturePolicyDecision`, `ChromeAutoCaptureSummary`, and `CaptureInboxItem` records for `CS-CH-025`.
- Explicit `chrome_auto_capture` source consent, two-sided config readiness, denied no-config trigger, denied blocked trigger, allowed summary-only trigger, and denied duplicate trigger policy decisions for `CS-CH-025`.
- Backend policy revalidation for `CS-CH-025`, including owner rule, site allowlist, source-pack allowlist, browser host permission, consent/config versions, allowed trigger type, active allowed page, throttle, session limit, idempotency, bounded payload, and raw browser data absence.
- Negative evidence for zero capture without owner rule, zero capture without site/source-pack/browser permission, zero consent/config version mismatch bypasses, zero unapproved-domain captures, zero inactive-tab captures, zero throttle/session/idempotency bypasses, zero raw browser data, zero external calls, zero provider mutations, and zero Artifacts created by auto capture.
- Chrome sensitive-page policy fixtures for `CS-CH-026`, including password, payment, token-like false-safe, mail compose, private account, browser-internal, unsupported-scheme, and oversized page cases.
- `ChromeSensitivePagePolicyDecision`, `ChromeSensitivePageDegradedPayload`, and `ChromeSensitivePageHistoryItem` records for `CS-CH-026`.
- Backend policy revalidation for `CS-CH-026`, including client block preservation, backend false-safe detection, sensitive signal classification, block/degrade decisions, hash-only degraded payloads, and owner-visible safe manual alternative guidance.
- Negative evidence for zero client block downgrades, zero false-safe bypasses, zero blocked/degraded raw text persistence, zero raw HTML, zero cookies, zero storage, zero screenshots, zero form values, zero browser history, zero full URLs/origins/titles, zero model sends, zero searchable content Artifacts, zero Capture Inbox items, zero external calls, and zero provider mutations.
- Capture lifecycle fixture for `CS-CH-027`, including seeded WatchAgent source state, Chrome auto-capture state, one Watch Rule target, one global collection target, eligible derived local state, audit/evidence-retained state, anonymizable candidate state, and pending capture results.
- `CaptureLifecycleSourceState`, `CaptureLifecycleDecision`, `CaptureLifecycleExport`, `CaptureLifecycleDeletionReceipt`, and `CaptureResultReview` records for `CS-CH-027`.
- Pause, resume, revoke, retention, sample-attempt, source-scoped export, result save/dismiss, delete dry-run, and authorized local fixture delete execution commands for `CS-CH-027`.
- Negative evidence for zero samples collected while paused or revoked, zero configuration deletion on pause, zero unscoped exports, zero raw content/browser payload/credential export, zero misleading delete-everything claims, zero audit record deletion, zero unauthorized delete execution, zero external calls, and zero provider mutations.
- Watch Result fixture for `CS-CH-028`, including source-backed observations, low-confidence and unsupported inferences, evidence/caveats, alternatives, and one non-executing proposal.
- `WatchObservation`, `WatchInference`, `WatchResult`, `WatchResultCorrection`, and `WatchResultReview` records for `CS-CH-028`.
- Watch Result build, correction, low-confidence memory approval denial, owner review, and audit verification commands for `CS-CH-028`.
- Negative evidence for zero inferred intent labeled as observed fact, zero inference stored as observed fact, zero unsupported or low-confidence inference approval, zero observation mutation by correction, zero direct proposal execution, zero direct Action/Claim/Mission creation, zero workflow starts, zero raw content, zero external calls, and zero provider mutations.
- ActionCard dry-run plus ConnectorHub preflight integration for `CS-CH-029`.
- Action execution safety gate denial matrix for `CS-CH-030`, including missing evidence, missing policy allow, missing approval, stale preflight, cross-namespace scope, and idempotency gaps.
- Declared supportdesk Action execution and outcome re-ingest for `CS-CH-031`, including WorkflowRun, Action Result, provider receipt, connected outcome, outcome Artifact, and outcome Evidence Bundle records.
- Undeclared Action, direct provider writeback, credential-boundary, and malicious provider-client Agent Pack bypass denial for `CS-CH-032`.
- Scoped idempotency replay, conflicting-intent denial, durable retry metadata, and visible compensation expectation for `CS-CH-033`.
- Owner/namespace/workspace scope binding for `CS-CH-034` across connector contract, Setup Result, Source Policy, Delivery receipt, Artifact, Evidence Bundle, Watch Result, Claim, Mission, ActionCard, preflight, approval, and audit refs.
- Cross-scope denial proof for `CS-CH-034`, including setup, Delivery processing, Evidence Bundle assembly, Watch Result review, and Action execution attempts with `CS_SCOPE_DENIED`.
- Negative evidence for `CS-CH-034`: zero cross-scope object returns, zero other-scope rows, zero ownerless connector rows, zero cross-scope Action execution, zero WorkflowRuns, zero Action Results, zero provider receipts, zero external calls, zero provider mutations, zero provider internals, and zero credential leaks.
- ConnectorHub-only credential lifecycle records for `CS-CH-035`, including status, rotate, revoke, credential ref, fingerprint, safe connection state, and audit refs.
- Seeded raw credential canary scan for `CS-CH-035` across CLI stdout and durable local state.
- Static Product/runtime provider-auth import scan for `CS-CH-035`.
- Negative evidence for `CS-CH-035`: zero raw credential canary output/state leaks, zero raw secret values, zero raw handles, zero auth headers, zero credential-bearing URLs, zero Product secret writes, zero provider auth imports, zero provider internals, zero external calls, and zero provider mutations.
- Current VS2 local proof reuse for `CS-CH-036`, including source-fingerprint validation, required VS2 egress scenario rows, proof hash, and reusable evidence paths.
- Local Docker internal-network topology for `CS-CH-036`, including API, worker, tool runtime, governed egress proxy, controlled provider sink, direct HTTP/socket denial, governed proxy success, and provider request counts.
- Negative evidence for `CS-CH-036`: zero proof reuse errors, zero missing/failed required VS2 egress rows, zero direct HTTP/socket bypass, zero provider requests after direct attempts, zero denied-hop sink/trap calls, zero sensitive headers forwarded to denied hops, zero raw credentials, zero raw audit payloads, and zero production topology overclaim.
- Connector audit correlation for `CS-CH-037`, including setup, policy, delivery, evidence, retry, quarantine, action, and credential lifecycle event families.
- Stable correlation IDs, connector event IDs, CornerStone audit event IDs, affected object refs, event hashes, previous hashes, and scope consistency for `CS-CH-037`.
- Negative evidence for `CS-CH-037`: zero missing required event families, zero uncorrelated connector events, zero duplicate correlation IDs, zero scope mismatches, zero raw payload or secret leaks, zero audit-integrity errors, and zero tamper-detection failures.
- Provider Pack upgrade compatibility plans.
- Incompatible Provider Pack activation blocking.
- Rollback metadata for planned upgrades.
- Connected Sources product-surface audit.
- Fixture/live/production readiness dimension linting.
- Negative evidence for fixture evidence overclaim prevention.
- Native CLI path with JSON output.
- Scenario verifier for `connector-contract-adapter`.
- Negative evidence for provider calls, provider internals, secrets, and ownerless records.

Out of scope for these units:

- Owner Source Policy override UI.
- Rendered UI/API degraded-mode proof. `CS-CH-003` currently verifies the CLI/state data contract that UI/API should consume.
- Rendered UI/API credential-gap status proof. `CS-CH-006` currently verifies the CLI/state data contract that UI/API should consume.
- Rendered Watch Rule UI/API lifecycle proof. `CS-CH-023` currently verifies the CLI/runtime data contract and durable lifecycle audit only.
- Rendered Watch Result UI/API proof. `CS-CH-028` currently verifies the CLI/runtime data contract, durable correction/review state, and negative safety counters only.
- Rendered Chrome extension/browser privacy proof. `CS-CH-024`, `CS-CH-025`, `CS-CH-026`, and `CS-CH-027` currently verify the CLI/runtime data contract, backend policy validation, and local lifecycle state contract only.
- Rendered Connected Sources browser walkthrough. `CS-CH-039` currently verifies the product walkthrough and product-surface data contract only.
- Live browser/document-provider prompt-injection variants beyond the verified local GitHub Projection fixture in `CS-CH-014`.
- Live-provider raw-access semantics; local `CS-CH-013` verifies the fixture data contract only.
- Live GitHub App installation, live selected-repository API enumeration, live repository permission state, and redacted external call logs. `CS-CH-015` verifies deterministic local fixture semantics only; `CS-CH-H01` remains required for live GitHub read-only rehearsal.
- Live GitHub repository, commit, pull-request, issue, file-content, webhook, pagination, rate-limit, and incremental cursor sync semantics. `CS-CH-016`, `CS-CH-017`, `CS-CH-018`, and `CS-CH-020` verify deterministic local fixture semantics only; `CS-CH-H01` remains required for broader source-control proof.
- Live GitHub App installation permission review for `CS-CH-019`; local proof verifies fixture manifests, contracts, CLI/runtime surfaces, and denied mutation attempts only.
- Live GitHub account access.
- Live macOS or Chrome capture. `CS-CH-021` verifies local metadata-only consent/permission guard semantics, `CS-CH-022` verifies local deterministic activity-session fixture semantics only, `CS-CH-023` verifies local Watch Rule lifecycle semantics only, `CS-CH-024` verifies local active-tab payload validation only, `CS-CH-025` verifies local auto-capture policy validation only, `CS-CH-026` verifies local sensitive-page policy validation only, `CS-CH-027` verifies local lifecycle state/export/deletion-receipt semantics only, and `CS-CH-028` verifies local Watch Result truth-separation semantics only; `CS-CH-H02` and `CS-CH-H03` remain required for physical-device and browser/privacy acceptance.
- Destructive-action Product approval UX for capture deletion. `CS-CH-027` verifies only deterministic local `--execute --authorized` receipt semantics; full ActionCard/dry-run/approval UI remains future Product governance work.
- Side-effecting connector Actions.
- Production RequestContext, RLS, OPA, network egress, backup/restore, or release readiness claims. `CS-CH-034` verifies deterministic local scope propagation and denial semantics only; production DB RLS/topology remains future VS2/live proof. `CS-CH-035` verifies local ConnectorHub credential-custody metadata and canary scans only; selected production secret backend custody and operational access controls remain future proof. `CS-CH-036` verifies the current local VS2 Docker/network-boundary proof only; production network-control review, firewall/proxy/service-mesh evidence, and operator approval remain future proof. `CS-CH-037` verifies local audit-correlation contract and tamper semantics only; production audit-retention policy and external SIEM export remain future proof.

2026-06-27 owner substitution decision for H04:

- JiYong/Tars observed the served local UI only states: `Local VS1 proof only. This page does not claim production readiness, live connector readiness, or human acceptance.`
- JiYong/Tars cannot personally verify H04 required human delta items 2 through 4 and approved replacing those manual review actions with local integration-test evidence for local readiness only.
- The approved local replacement commands are `cornerstone scenario verify vs2-policy-tenancy-egress --reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json` and `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036 --json`.
- This substitution does not satisfy production-like topology item 1, does not create the dated ACCEPT/REJECT decision item 5, and does not change H04 `HUMAN_REQUIRED`, `acceptance_sufficient=false`, `product_claim_allowed=false`, or `pass_claim_allowed=false`.

## Scenario Ledger

The full Connector Hub scenario ledger is in:

```text
docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_MATRIX.csv
```

Current status:

- `CS-CH-001`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-002`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-003`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-004`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-005`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-006`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-007`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-008`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-009`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-010`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-011`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-012`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-013`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-014`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-015`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-016`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-017`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-018`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-019`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-020`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-021`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-022`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-023`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-024`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-025`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-026`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-027`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-028`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-029`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-030`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-031`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-032`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-033`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-034`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-035`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-036`: AI-verifiable via filtered `connector-contract-adapter` scenario report backed by current reusable VS2 local egress proof.
- `CS-CH-037`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-038`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-039`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-040`: AI-verifiable via filtered `connector-contract-adapter` scenario report.
- `CS-CH-H01` through `CS-CH-H07`: `HUMAN_REQUIRED`.
- Human-gate preparation packages: `cornerstone connector human-gate package --scenario <CS-CH-H01..CS-CH-H07> --json`; these record required human evidence, the matching review-template path and structure readiness, review order, dependency gates, stop/reject condition, `scenario_delivery_unit_plan`, `scenario_delivery_unit_plan_summary`, blank `proposed_record_template` reviewer-record JSON skeleton including a typed evidence-packet manifest skeleton with `required_evidence` labels and `allowed_redaction_statuses=redacted,public_safe,no_sensitive_material`, field-level `redaction_guidance`, `reviewer_checklist`, `validation_command`, `validation_output_command` for `--output <redacted-validation-envelope.json>`, `record_template_output_command` for `--record-template-output <reviewer-record-template.json>` to write only the blank reviewer template, a top-level CLI envelope `summary`/`final_verdict=HUMAN_REQUIRED` mirror for operator scripting, and zero package-side provider mutations but do not mark any human row `PASS`. The `scenario_delivery_unit_plan` maps senior-perspective research, implementation approach definition, smallest complete rehearsal, remediation/refactor, structural verification, result documentation, and dependency-aware next-gate movement while keeping product/PASS/approval flags false. The package `summary` exposes `scenario_delivery_unit_plan_ready=true`, `scenario_delivery_unit_plan_lifecycle_step_count=7`, senior-review perspective count, and false product/PASS/approval/dependency-unlock plan flags so operator scripts can reject incomplete delivery-unit preparation without parsing the nested lifecycle plan. For `CS-CH-H04` only, the package also exposes `local_baseline_review_inputs` with current local VS2 and ConnectorHub dependency report refs, hashes, status summaries, recommended local preflight commands, and a structured `recommended_preflight_command_plan` with expected report paths, plus a first-class `local_baseline_preflight_bundle` alias that mirrors `local_baseline_review_inputs.preflight_bundle` and repeats `recommended_preflight_command_plan_schema_version`, `recommended_preflight_command_plan_count`, and `recommended_preflight_command_plan`; this is review input only with `acceptance_sufficient=false`, `product_claim_allowed=false`, and `pass_claim_allowed=false`, each individual baseline report row repeats `review_input_only=true`, `acceptance_sufficient=false`, `product_claim_allowed=false`, `pass_claim_allowed=false`, and `claim_boundary=h04_local_baseline_snapshot_is_review_input_not_human_acceptance`, and each preflight command-plan row repeats the same no-claim flags with `claim_boundary=h04_local_baseline_preflight_is_review_input_not_human_acceptance`.
- Human-gate proposed-record validation: `cornerstone connector human-gate validate-record --scenario <CS-CH-H01..CS-CH-H07> --record-file <json> --json`, optionally with `--output <redacted-validation-envelope.json>`; this checks required fields, decision values, redacted senior-review perspective findings, evidence-packet manifest coverage including matching required-evidence labels and allowed redaction statuses, plus unique evidence refs, dependency gate refs to existing structurally valid ACCEPT `connector_human_gate_record_validation:<id>` artifacts, ISO-8601 timezone-aware `review_timestamp` format, and sensitive-marker findings without persisting the record body, raw record path, submitted decision value, senior-review finding text, or evidence-packet manifest values, while returning `redaction_guidance` so operators can correct unsafe reviewer records without exposing raw values. Duplicate evidence-packet refs are reported in validation issue summaries as `duplicate_evidence_packet_manifest_refs` with short SHA-256 fingerprints only; raw evidence refs remain omitted from validator, next-selector, and validation-handoff outputs. The optional validation envelope mirrors top-level `summary.validation_id`, `summary`, and `final_verdict=HUMAN_REQUIRED` without promoting the scenario out of `HUMAN_REQUIRED`; validation issue summaries also include `structural_validation_is_human_acceptance=false`, `human_acceptance_requires_owner_promotion=true`, and `completion_claim_boundary=connectorhub_full_goal_requires_dated_accept_records_for_all_human_external_gates`. Structurally valid `REJECT` records remain validation evidence but set `dependency_unlock_allowed_by_validator=false` and do not unlock dependent H gates.
- Human-gate readiness report: `cornerstone connector human-gate report --json --output reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json` plus `docs/verification-reports/CONNECTOR_HUB_HUMAN_GATES_PREPARATION_REPORT_2026-06-24.md`; this audits H01-H07 package, execution queue, template, template-structure readiness, blank `proposed_record_template` shape, embedded `reviewer_checklist`, embedded `scenario_delivery_unit_plan`, row-level `scenario_delivery_unit_plan`, row-level `scenario_delivery_unit_plan_summary`, row-level `record_template_output_command`, row-level `record_validation_output_command`, H04-only `local_baseline_review_inputs`, H04-only first-class `local_baseline_preflight_bundle` mirroring `local_baseline_review_inputs.preflight_bundle` with the same `recommended_preflight_command_plan` alias fields, existing proposed-record validation coverage including `senior_review_perspective_findings_complete_count` and `evidence_packet_manifest_complete_count`, and depends-on human-gate record validation readiness fields such as `depends_on_human_gate_record_validation_status` and `depends_on_human_gates_missing_structurally_valid_record_validation` while keeping the weakest applicable result as `HUMAN_REQUIRED`. The readiness report exposes `scenario_delivery_unit_plan_ready_count=7`, `scenario_delivery_unit_plan_missing=[]`, and zero plan-level product/PASS/approval/dependency-unlock allowances. The standard `cs.cli.v0` envelope also mirrors `summary.report_id`, `summary.readiness_report_id`, `final_verdict=HUMAN_REQUIRED`, top-level `summary` with scenario counts, validation counts, delivery-unit plan readiness counters, zero PASS/product-claim allowances, and `product_feature_claims=CONNECTOR_HUB_HUMAN_GATES_PREPARED_HUMAN_EVIDENCE_REQUIRED` for operator scripting.
- Human-gate next selector: `cornerstone connector human-gate next --json`; this selects the first uncompleted dependency-ready H gate, lists ready and blocked gate IDs, exposes missing dependency unlock validations, emits `readiness_report_ref`, exposes `next_record_template_output_command`, exposes `next_record_validation_output_command`, exposes `next_record_redaction_guidance`, exposes `next_reviewer_checklist`, exposes `next_scenario_delivery_unit_plan`, exposes `next_scenario_delivery_unit_plan_summary`, exposes `next_remaining_human_evidence_summary` with required fields/evidence/release-impact/stop-reject details and `claim_boundary=remaining_human_evidence_summary_is_operator_input_not_acceptance`, and exposes `next_latest_record_validation_issue_summary` plus latest validation ref/dependency-unlock status when the active next gate already has a structural validation; issue summaries explicitly say structural validation is not human acceptance and owner promotion remains required; it mirrors `summary.next_id`, `summary.readiness_report_id`, `next_scenario_delivery_unit_plan_ready`, lifecycle-step count, senior-review perspective count, active-gate validation count/status, issue-summary presence, remaining human field/evidence counts, and false plan-level product/PASS/approval/dependency-unlock flags into the top-level `summary`; when H04 is next, it mirrors `next_local_baseline_review_inputs`, `next_required_human_delta`, `next_recommended_preflight_commands`, `next_recommended_preflight_command_plan`, and first-class `next_local_baseline_preflight_bundle` as comparison input with `acceptance_sufficient=false`, where `next_local_baseline_preflight_bundle` mirrors `next_local_baseline_review_inputs.preflight_bundle` with the same `recommended_preflight_command_plan` alias fields; the pinned operator sequencing artifact is `reports/scenario/connectorhub-human-gate-next-2026-06-24.json`; it keeps `final_verdict=HUMAN_REQUIRED` without collecting approval, calling live providers, or promoting any row to `PASS`.
- Human-gate validation handoff: `cornerstone connector human-gate validation-handoff --json --output reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json`; this compacts readiness into `schema_version=cs.connector_human_gate_validation_handoff.v1` with ordered `scenario_validation_handoff_rows`, dependency blockers, reviewer commands, validation status, latest `connector_human_gate_record_validation:<id>` refs when present, redacted `cs.connector_human_gate_validation_issue_summary.v1` correction/dependency summaries when a validation exists, H04 local-baseline summary counts, H04 preflight command-plan rows, H04 first-class `local_baseline_preflight_bundle` mirroring `local_baseline_review_input_summary.preflight_bundle` with the same `recommended_preflight_command_plan` alias fields, row-level `remaining_human_evidence_summary` objects that preserve required human fields, required evidence labels, release impact, stop/reject rule, reviewer-template command, redacted validation-output command, and `claim_boundary=remaining_human_evidence_summary_is_operator_input_not_acceptance`, plus zero approval/PASS/product-claim counters. The handoff emits a `readiness_report_ref`, mirrors `summary.handoff_id`, `summary.readiness_report_id`, and `summary.operator_rule`, remains `final_verdict=HUMAN_REQUIRED`, and is an operator trail only, not approval or PASS evidence.
- Human-gate source-requirement handoff metadata: every package, readiness row, next-selector output, and validation-handoff row exposes matrix-derived `source_requirement_ids`, `source_requirement_count`, `source_requirement_status=human_external_pending`, and `source_requirement_claim_boundary=human_gate_preparation_does_not_close_source_requirements`; the validation handoff also exposes the unique human-pending source requirement IDs `ER-05`, `ER-06`, `IR-01`, `IR-04`, `IR-07`, `IR-11`, `IR-13`, `IR-14`, `IR-16`, `IR-17`, and `IR-18`. This is operator traceability only. It does not close source requirements, collect approval, or promote any H row to `PASS`.
- Engineering-trail verifier: `make verify-connectorhub-engineering-trail`; this checks 40 AI PASS result docs, 7 human templates, the aggregate ConnectorHub report, and stale metadata guards.

## Source Requirement Coverage

The `related_requirements` column in `docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_MATRIX.csv` is the machine-readable mapping from the source implementation document's `ER-*` and `IR-*` requirements to Connector Hub scenario rows. `make verify-connectorhub-engineering-trail` fails if a source requirement is unmapped, if a scenario row omits the mapping, or if an unknown requirement ID appears.

The matrix also carries `proof_surface` and `claim_boundary` fields so each scenario keeps its evidence plane explicit:

| Proof surface | Applies to | Claim boundary |
|---|---|---|
| `local_fixture` | AI-verifiable Connector Hub rows except `CS-CH-036` | Deterministic local fixture evidence only; no live-provider, production, or human-acceptance claim. |
| `local_vs2_topology` | `CS-CH-036` | Current reusable local VS2 topology evidence only; no production network readiness claim. |
| `human_required` | `CS-CH-H01` through `CS-CH-H07` | Human/external evidence required; no AI PASS claim. |

Every AI-owned PASS result document must include a `Senior Engineering Decision Trail` covering these dimensions before the scenario can be treated as a complete delivery unit:

- Product value
- Domain correctness
- Architecture
- Data contracts
- Reliability
- Security
- Observability
- Performance
- Testability
- Maintainability
- Migration feasibility

The trail must tie the decision to the scenario row, evidence artifact, proof surface, and claim boundary. It is not a substitute for verification evidence; `make verify-connectorhub-engineering-trail` enforces the presence of all dimensions across the 40 AI-owned result documents.

Every AI-owned PASS result document must also include a `Scenario Lifecycle Trail` so each scenario remains an independent delivery unit instead of a bulk implementation note. The lifecycle trail must document:

- Research perspectives from product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration viewpoints.
- Implementation approach selected for the specific scenario.
- Smallest complete solution actually delivered.
- Refactor and hardening work after the smallest solution.
- Verification result with the exact proof surface and evidence artifact.
- Documented result and ConnectorHub adoption contribution.

This lifecycle trail does not upgrade live-provider, human, or production readiness. Those remain governed by `proof_surface`, `claim_boundary`, and the H01-H07 human-required gates.

Per-scenario result documents must remain self-contained. A result document may mention the aggregate report as supporting evidence, but its `Scenario Verification` section must reference only its own scenario row, and it must not contain a stale `Next Scenario` handoff. The delivery order is governed by the scenario matrix and proof gates, not by prose pointers embedded in older result reports.

The readable entry point for the full engineering trail is `docs/verification-reports/CONNECTOR_HUB_ENGINEERING_TRAIL_INDEX_2026-06-24.md`. The machine-readable scenario closure manifest is `reports/scenario/connectorhub-scenario-delivery-unit-manifest-2026-06-24.json`, and the machine-readable integrity manifest is `reports/scenario/connectorhub-engineering-trail-manifest-2026-06-24.json`. `make verify-connectorhub-engineering-trail` fails if the index omits any AI-owned result document, any human-required template/package, the aggregate report, the scenario delivery-unit manifest, or the required proof-boundary wording; it also fails if the integrity manifest omits or has a stale hash for any required trail artifact.

| Requirement | Covering scenario rows |
|---|---|
| `ER-01` | `CS-CH-001` |
| `ER-02` | `CS-CH-001` |
| `ER-03` | `CS-CH-028` |
| `ER-04` | `CS-CH-023` |
| `ER-05` | `CS-CH-006, CS-CH-021, CS-CH-022, CS-CH-023, CS-CH-024, CS-CH-025, CS-CH-026, CS-CH-H02, CS-CH-H03` |
| `ER-06` | `CS-CH-015, CS-CH-016, CS-CH-017, CS-CH-018, CS-CH-019, CS-CH-020, CS-CH-H01` |
| `ER-07` | `CS-CH-040` |
| `ER-08` | `CS-CH-040` |
| `ER-09` | `CS-CH-040` |
| `IR-01` | `CS-CH-039, CS-CH-H06` |
| `IR-02` | `CS-CH-001, CS-CH-005, CS-CH-032` |
| `IR-03` | `CS-CH-001, CS-CH-002, CS-CH-003` |
| `IR-04` | `CS-CH-001, CS-CH-004, CS-CH-007, CS-CH-015, CS-CH-023, CS-CH-027, CS-CH-034, CS-CH-H04` |
| `IR-05` | `CS-CH-007, CS-CH-008, CS-CH-010, CS-CH-016` |
| `IR-06` | `CS-CH-007, CS-CH-012, CS-CH-016, CS-CH-028` |
| `IR-07` | `CS-CH-007, CS-CH-008, CS-CH-009, CS-CH-010, CS-CH-016, CS-CH-017, CS-CH-020, CS-CH-022, CS-CH-033, CS-CH-H07` |
| `IR-08` | `CS-CH-002, CS-CH-004, CS-CH-005, CS-CH-011, CS-CH-015, CS-CH-018, CS-CH-020, CS-CH-025` |
| `IR-09` | `CS-CH-001, CS-CH-006, CS-CH-011, CS-CH-013, CS-CH-015, CS-CH-018, CS-CH-019, CS-CH-026, CS-CH-032, CS-CH-035` |
| `IR-10` | `CS-CH-014, CS-CH-018, CS-CH-019, CS-CH-024, CS-CH-026, CS-CH-028, CS-CH-032, CS-CH-036` |
| `IR-11` | `CS-CH-013, CS-CH-014, CS-CH-019, CS-CH-029, CS-CH-030, CS-CH-031, CS-CH-032, CS-CH-033, CS-CH-H05` |
| `IR-12` | `CS-CH-002, CS-CH-006, CS-CH-029, CS-CH-039` |
| `IR-13` | `CS-CH-035, CS-CH-040, CS-CH-H01, CS-CH-H05` |
| `IR-14` | `CS-CH-021, CS-CH-022, CS-CH-023, CS-CH-024, CS-CH-025, CS-CH-026, CS-CH-027, CS-CH-028, CS-CH-H02, CS-CH-H03, CS-CH-H06` |
| `IR-15` | `CS-CH-005, CS-CH-038` |
| `IR-16` | `CS-CH-003, CS-CH-038, CS-CH-039, CS-CH-H06` |
| `IR-17` | `CS-CH-004, CS-CH-009, CS-CH-012, CS-CH-020, CS-CH-027, CS-CH-031, CS-CH-037, CS-CH-H07` |
| `IR-18` | `CS-CH-014, CS-CH-029, CS-CH-030, CS-CH-031, CS-CH-034, CS-CH-036, CS-CH-037, CS-CH-040, CS-CH-H04, CS-CH-H05, CS-CH-H07` |

## CS-CH-001 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-001` |
| Type | `MUST_PASS` |
| Trigger / Action | Owner submits a versioned connector capability contract for a scoped workspace. |
| Expected Result | CornerStone validates and registers the contract through ConnectorPort and persists a scoped Setup Result with mappings, Source Policy, warnings, verification refs, and audit refs. |
| Affected Layers | CLI, ConnectorPort adapter, local durable state, audit, scenario verifier. |
| Verification Method | Schema/unit checks, CLI integration, durable state inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | Contract fixture, Setup Result JSON, Source Policy JSON, audit refs, zero provider call ledger, zero secret/internal findings. |
| Owner | AI |

## CS-CH-002 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-002` |
| Type | `MUST_PASS` |
| Trigger / Action | Owner submits a versioned connector capability contract where a required capability has no compatible Provider Pack mapping. |
| Expected Result | Setup planning persists a blocked Setup Result with stable reason code, safe resolution guidance, no delivery streams, zero provider calls, and audit refs. |
| Affected Layers | CLI, ConnectorPort adapter, setup planning state machine, local durable state, audit, scenario verifier. |
| Verification Method | Missing-required fixture, CLI negative setup test, scenario verifier, durable state inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | Missing-required contract fixture, blocked Setup Result JSON, Source Policy JSON, exit code `7`, audit refs, provider-call ledger `0`, empty delivery streams. |
| Owner | AI |

## CS-CH-003 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-003` |
| Type | `MUST_PASS` |
| Trigger / Action | Owner submits a versioned connector capability contract where required capabilities are available but an optional capability has no compatible Provider Pack mapping. |
| Expected Result | Setup planning succeeds with `ready_with_gaps`, available capability streams remain enabled, unavailable optional surfaces are disabled with stable metadata, and provider calls remain zero. |
| Affected Layers | CLI, ConnectorPort adapter, setup planning state machine, local durable state, audit, scenario verifier. |
| Verification Method | Optional-missing fixture, CLI setup test, scenario verifier, durable state inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | Optional-missing contract fixture, ready-with-gaps Setup Result JSON, enabled delivery stream, disabled surface record, audit refs, provider-call ledger `0`. |
| Owner | AI |

## CS-CH-004 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-004` |
| Type | `MUST_PASS` |
| Trigger / Action | Owner confirms the recommended Source Policy or submits a narrower compatible override. |
| Expected Result | A new immutable Source Policy snapshot is stored with owner confirmation, compatibility decision, diff hashes, audit refs, and `constraints_never_broadened_silently=true`; attempted broadening is denied and audited without creating a policy snapshot. |
| Affected Layers | CLI, ConnectorPort adapter, Source Policy normalization, local durable state, audit, scenario verifier. |
| Verification Method | Policy normalization, owner override command, broadening-denial negative command, durable state inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | Confirmed Source Policy JSON, source policy diff, denial transcript for broadening attempt, audit refs, zero secret/internal findings. |
| Owner | AI |

## CS-CH-005 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-005` |
| Type | `REGRESSION_GUARD` |
| Trigger / Action | Setup planning is run with an alternate compatible Provider Pack that exposes the same common capabilities and projection contracts. |
| Expected Result | Provider and source policy refs change, but Product handler contract, projection contract, preview object, and provider-free boundaries remain unchanged. |
| Affected Layers | CLI, ConnectorPort adapter, provider-pack selection, setup planning state, scenario verifier. |
| Verification Method | Provider-swap integration test comparing default and alternate setup plans. |
| Evidence Required | Default and alternate Setup Result JSON, changed provider refs, unchanged handler contract hash, unchanged projection contract, unchanged product preview, audit refs. |
| Owner | AI |

## CS-CH-006 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-006` |
| Type | `MUST_PASS` |
| Trigger / Action | Setup planning is run against a Provider Pack that requires a missing read-only selected-repository permission. |
| Expected Result | Setup is blocked with a specific permission reason, owner-safe cause/impact/resolution text, false redaction exposure flags, audit refs, zero provider calls, and no provider internals. |
| Affected Layers | CLI, ConnectorPort adapter, setup status contract, local durable state, audit, scenario verifier. |
| Verification Method | Permission-gap Provider Pack fixture, CLI status contract test, scenario verifier, secret/provider-internal scan. |
| Evidence Required | Blocked Setup Result JSON, status explanation, redaction flags, audit refs, zero provider-call ledger, zero secret/internal findings. |
| Owner | AI |

## CS-CH-007 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-007` |
| Type | `MUST_PASS` |
| Trigger / Action | A valid app-scoped Connector Projection Delivery arrives for a previously validated contract and ready Setup Result. |
| Expected Result | CornerStone archives the exact delivery envelope as an immutable scoped Artifact, then stores a ConnectorDeliveryReceipt, ProjectionSnapshot, ConnectorEvidenceLink, Source Policy link, EvidenceRef metadata, checksum, storage ref, and audit refs before any Product interpretation or acknowledgement. |
| Affected Layers | CLI, ConnectorPort adapter, Artifact ingest, local durable state, provenance links, audit, scenario verifier. |
| Verification Method | Fixture delivery integration test, persisted original-byte checksum check, artifact show, durable receipt/snapshot/evidence-link inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | Delivery fixture, Artifact JSON, original stored envelope bytes, delivery receipt JSON, Projection snapshot JSON, evidence link JSON, Source Policy and Setup Result refs, audit refs, zero pre-archive interpretation, zero acknowledgement. |
| Owner | AI |

## CS-CH-008 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-008` |
| Type | `MUST_PASS` |
| Trigger / Action | A Delivery can be redelivered and the handler may crash before durable commit or after durable commit but before acknowledgement. |
| Expected Result | No acknowledgement occurs before durable Artifact/receipt/outbox commit; a post-commit redelivery reuses one logical Artifact and sends acknowledgement only through the committed outbox. |
| Affected Layers | CLI, ConnectorPort adapter, Artifact ingest, delivery receipt state, ack outbox, local durable state, audit, scenario verifier. |
| Verification Method | Fault-injection integration test for pre-commit crash, post-commit/pre-ack crash, redelivery, duplicate redelivery, reconciliation, durable state inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | Crash transcripts, one delivery receipt, one ack outbox, one Artifact record, stored envelope bytes, ack timeline, reconciliation JSON, acknowledged_without_artifact=0, duplicate logical artifacts=0, audit refs. |
| Owner | AI |

## CS-CH-009 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-009` |
| Type | `MUST_PASS` |
| Trigger / Action | A Delivery handler or provider fails transiently, repeatedly, or with a malformed/poison payload while other connector streams remain healthy. |
| Expected Result | Transient failures create bounded retry state with deterministic backoff and no acknowledgement; poison deliveries reach quarantine at the configured threshold with safe diagnostics; unrelated healthy deliveries continue to archive and acknowledge; replay requests link to the quarantine record without erasing failure evidence. |
| Affected Layers | CLI, ConnectorPort adapter, retry state, quarantine store, delivery processing, ack boundary, local durable state, audit, scenario verifier. |
| Verification Method | Retry-clock integration test, malformed payload integration test, queue-state check, healthy-stream continuation check, quarantine replay check, audit verification, secret/provider-internal scan. |
| Evidence Required | Attempt counts, retry schedule, retry state JSON, quarantine item JSON, redacted error, source-health impact, healthy Delivery Artifact/ack refs, replay attempt, audit events, zero infinite retry loop, zero queue-wide blockage, zero raw payload in operator output. |
| Owner | AI |

## CS-CH-010 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-010` |
| Type | `MUST_PASS` |
| Trigger / Action | The same provider event or unchanged source object content is delivered multiple times, then the same source object later changes. |
| Expected Result | Repeated provider events or unchanged source content resolve to one logical intake record with one active Artifact/search truth; changed content creates a new content-version record linked to the predecessor Artifact/version and source revision without mutating historical evidence. |
| Affected Layers | CLI, ConnectorPort adapter, delivery idempotency, content hash computation, content-version lineage store, current-version pointer, Artifact ingest, ack outbox, audit, scenario verifier. |
| Verification Method | Duplicate provider-event fixture, unchanged-content/new-event fixture, one-byte changed-content fixture, lineage query command, durable state inspection, reconciliation, audit verification, secret/provider-internal scan. |
| Evidence Required | Provider event IDs, delivery idempotency key, source external ID, source revision, content hashes, dedupe state JSON, content-version records, current-version pointer, lineage query JSON, two versioned Artifacts after the changed fixture, one current truth, no duplicate active truth, no mutation of immutable historical evidence. |
| Owner | AI |

## CS-CH-011 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-011` |
| Type | `MUST_PASS` |
| Trigger / Action | Projection Delivery processing receives allowed preview fields, a forbidden full-body field, and a later delivery after the owner narrows Source Policy content limits. |
| Expected Result | Allowed deliveries are normalized and archived with a policy decision record; forbidden full-body/raw fields and over-limit payloads are blocked before durable Product state, with helpful errors, audit refs, and zero forbidden content leakage. |
| Affected Layers | CLI, ConnectorPort adapter, Source Policy enforcement, Projection normalization, Artifact ingest, delivery receipt state, audit, scenario verifier. |
| Verification Method | Allowed-preview fixture, forbidden full-body fixture, narrowed max-content fixture, durable-state inspection, policy decision inspection, audit verification, forbidden-content scan, secret/provider-internal scan. |
| Evidence Required | Projection policy decision JSON, Source Policy snapshot refs, normalized projection payload, rejected-delivery transcripts, zero new Artifact/receipt for rejected deliveries, audit refs, forbidden marker leak count `0`, raw-content leak count `0`. |
| Owner | AI |

## CS-CH-012 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-012` |
| Type | `MUST_PASS` |
| Trigger / Action | Product asks to use connector EvidenceRef metadata from a committed Projection Delivery in a brief, Claim, or action-review context. |
| Expected Result | CornerStone assembles a normal Evidence Bundle that links the immutable Artifact, Connector Delivery Receipt, Setup Result, Source Policy, EvidenceRef metadata, query/search snapshot, policy decision, and audit refs; EvidenceRef metadata alone is denied as truth, and zero-evidence Claim approval remains blocked. |
| Affected Layers | CLI, ConnectorPort adapter, Evidence Bundle store, search snapshot store, Claim workflow, audit, scenario verifier. |
| Verification Method | Valid delivery-to-bundle integration test, Evidence Bundle show command, Claim create/approve command, EvidenceRef-only negative command, zero-evidence Claim approval negative command, audit verification, secret/provider-internal scan. |
| Evidence Required | Evidence Bundle JSON, search snapshot JSON, connector evidence-bundle link schema, Artifact checksum, Delivery Receipt refs, Setup Result refs, Source Policy refs, EvidenceRef metadata, Claim refs, audit refs, EvidenceRef-only denial transcript, zero-evidence Claim approval denial transcript, raw-provider payload count `0`, phantom-evidence count `0`. |
| Owner | AI |

## CS-CH-013 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-013` |
| Type | `MUST_PASS` |
| Trigger / Action | Product or operator requests temporary raw access to one connector evidence ref for a declared purpose. |
| Expected Result | Undeclared raw access is denied by default; declared access requires Source Policy permission and human approval, issues only an opaque internal grant reference, expires by TTL, enforces max reads, supports revocation, returns metadata-only exports, and never copies raw content or reusable handles into Product records, reports, logs, screenshots, or audit payloads. |
| Affected Layers | CLI, ConnectorPort adapter, Source Policy bridge, raw-access request store, raw-access grant store, read mediation, metadata export, audit, scenario verifier. |
| Verification Method | Default-denial request, declared raw-access contract setup, TTL-boundary denial, max-read-boundary denial, granted request, metadata-only export, read-count exhaustion, deterministic expiry, revocation, durable state inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | Denied request transcript, active raw-access request JSON, active grant JSON, Source Policy limits, metadata export JSON, read result JSON, exhausted/expired/revoked denial transcripts, audit refs, reusable raw-handle counter `0`, raw payload/content/handle leak counter `0`. |
| Owner | AI |

## CS-CH-014 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-014` |
| Type | `REGRESSION_GUARD` |
| Trigger / Action | Connector Projection content contains instructions to ignore policy, reveal secrets, invoke tools, call external URLs, approve actions, or promote content into memory. |
| Expected Result | Projection content remains untrusted evidence. It may be archived, searched, quoted, summarized, and cited only as evidence; it cannot become system instruction, policy authority, role authority, a tool call, an ActionCard, a WorkflowRun, external egress, trusted memory, or a policy override. |
| Affected Layers | CLI, ConnectorPort adapter, Artifact safety metadata, untrusted-content review store, Evidence Bundle trust-boundary coverage, Claim creation, Agent policy, Memory quarantine, Egress policy, audit, scenario verifier. |
| Verification Method | Prompt-injection Projection fixture, delivery processing, untrusted-content review show command, Evidence Bundle assembly, boundary-aware Claim creation, prompt-authority denial, memory quarantine, egress denial, durable-state side-effect scan, audit verification, secret/provider-internal scan. |
| Evidence Required | Malicious Projection fixture, Artifact trust label, `cs.connector_untrusted_content_review.v1` JSON, blocked attempt list, Evidence Bundle trust-boundary coverage, evidence-backed Claim statement distinguishing quoted instruction from system instruction, `CS_AGENT_POLICY_DENIED` transcript, memory quarantine JSON, `CS_EGRESS_DENIED` transcript, zero action/workflow/memory side-effect counts, audit refs. |
| Owner | AI |

## CS-CH-015 Contract

Goal:

Prove the first CH-2 GitHub selected-repository boundary with deterministic local fixtures: a GitHub-style installation may expose multiple repositories to the connector layer, but CornerStone Product state can see and ingest only owner-selected repositories.

Constraints:

- Product / UX: normal Product surfaces must expose selected connected sources, not an organization-wide GitHub import.
- Data / State: Source Policy stores opaque selected source refs, selected-resource version metadata, and no credentials or raw provider payloads.
- Permission / Security: unselected repository events are denied before Artifact, receipt, ack, or Product state creation; GitHub writes remain unavailable.
- Compatibility / Format: source-control capability and Projection contracts remain provider-neutral and use the existing ConnectorPort adapter path.
- Operational / Environment: proof is local deterministic fixture evidence only; live GitHub App installation and call logs stay `HUMAN_REQUIRED`.

Assumptions:

- The local `provider_resource_catalog` represents repositories visible to an installed read-only GitHub App fixture.
- `github:repo:owner/project-alpha` is the only owner-selected repository for this scenario.

Out of scope before coding:

- Live GitHub API enumeration, live installation permission review, pagination, webhooks, rate-limit behavior, repository removal, and real external call logs.
- Source-control Projection family expansion beyond the existing issue Projection fixture.
- GitHub write action implementation.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-015` |
| Type | `MUST_PASS` |
| Trigger / Action | Owner connects a GitHub-style read-only source where the installation fixture exposes three repositories and the owner selects only one repository for a workspace. |
| Expected Result | Setup persists a namespace-scoped, versioned selected-resource scope with one visible selected repository, no organization/account-wide fallback, no credential exposure, and no write permission. A selected-repository Delivery is archived and acknowledged. An unselected-repository Delivery is denied before Artifact, receipt, or acknowledgement creation. Direct GitHub write and silent selection broadening are denied with audit refs. |
| Affected Layers | CLI, ConnectorPort adapter, Setup Result, Source Policy, Projection policy decision, Delivery archive/ack path, direct-write policy, audit, scenario verifier. |
| Verification Method | Selected-repositories contract fixture, selected-repo Delivery fixture, unselected-repo Delivery fixture, direct-write denial command, Source Policy broadening-denial command, durable-state count inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | `cs.connector_selected_resource_scope.v1` JSON, selected Delivery receipt/Artifact/ack, denied unselected Projection policy decision with `CS_CONNECTOR_SOURCE_POLICY_RESOURCE_DENIED`, `CS_DIRECT_WRITE_DENIED` transcript, `CS_CONNECTOR_SOURCE_POLICY_BROADENING_DENIED` transcript, negative counters for zero unselected Artifacts/receipts/acks, zero org-wide fallback, zero GitHub write permission/calls, zero silent broadening, audit refs. |
| Owner | AI |

## CS-CH-016 Contract

Goal:

Prove selected-repository read-only source-control Projection family ingestion with deterministic fixtures: repository, commit, change or pull-request, issue, and allowed file snapshot Projections become immutable CornerStone evidence with source revisions and repository provenance.

Constraints:

- Product / UX: Product logic consumes provider-neutral `source_control.*` Projections, not GitHub schemas.
- Data / State: every Projection preserves exact envelope bytes plus Artifact, receipt, Projection snapshot, evidence link, content version, search snapshot, and Evidence Bundle refs.
- Permission / Security: read-only capability only, raw access denied, no provider tokens, no raw provider payload, no write path, and file snapshot stays inside allowed selected-resource paths.
- Reliability: each Delivery acknowledges only after durable archive commit and persists source revision/content version state.
- Operational / Environment: proof is local deterministic fixture evidence only; live GitHub sync, webhook, and API proof remains `HUMAN_REQUIRED`.

Assumptions:

- The local selected repository is `github:repo:owner/project-alpha`.
- Pull-request style provider data is represented by provider-neutral `source_control.change.v1`.
- The file snapshot fixture represents an allowed `docs/**` path, not arbitrary repository content.

Out of scope before coding:

- Live GitHub API calls, webhooks, pagination, rate limits, revoked permissions, repository removal, and external call logs.
- Large, binary, secret-bearing, or out-of-policy file behavior, now covered by `CS-CH-018`.
- Local incremental cursor ordering, replay, and changed-source synchronization, now covered by `CS-CH-017`; live GitHub sync remains out of scope for `CS-CH-016`.
- GitHub writes, which remain denied and are covered separately by `CS-CH-019`.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-016` |
| Type | `MUST_PASS` |
| Trigger / Action | Initial read sync processes repository, commit, change, issue, and allowed file-snapshot Projection fixtures for one selected repository. |
| Expected Result | All five provider-neutral Projection types archive as immutable Artifacts, acknowledge after durable commit, persist source revision/repository provenance/content version records, and assemble searchable Evidence Bundles without requiring GitHub-specific Product fields. |
| Affected Layers | CLI, ConnectorPort adapter, Source Policy, Delivery archive/ack, content-version state, Evidence Bundle/search snapshot, audit, scenario verifier. |
| Verification Method | Five Projection Delivery fixtures, delivery process commands, Evidence Bundle creation per receipt, durable-state count inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | Five delivery receipts, five Artifacts, five Projection snapshots, five evidence links, five ack outbox records, five content-version records, five search snapshots, five Evidence Bundles, source_revision/source_external_id/source_ref checks, zero raw/provider-specific requirements/calls/leaks. |
| Owner | AI |

## CS-CH-017 Contract

Goal:

Prove deterministic local incremental sync idempotency for a selected GitHub-style source: webhook redelivery, scheduled polling overlap, crash/replay, and out-of-order source revisions must not duplicate product truth or skip changed content.

Constraints:

- Product / UX: CornerStone sees provider-neutral source-control truth and sync freshness evidence, not GitHub webhook internals.
- Domain / Data: the provider-event key is derived from provider installation, repository, object, action, and source revision; webhook and poll observations for the same revision share one logical identity.
- Reliability: cursors advance only after durable Delivery processing; a post-commit/pre-cursor crash is detectable by reconciliation before replay and cleared by replay without duplicate Artifact truth.
- Security: webhook origin and signature verification are explicit Connector-boundary evidence. Invalid webhook metadata is denied before delivery receipt, Artifact, or cursor creation.
- Observability: sync signal receipts, cursor history, reconciliation output, sync lag metrics, evidence refs, and audit refs are persisted.
- Performance / Migration: local proof uses deterministic fixture state and stable cursor IDs so the model can migrate to live GitHub pagination/webhook runners later without changing Product truth semantics.
- Operational / Environment: proof is local deterministic fixture evidence only; live GitHub webhook signatures, polling API cursors, pagination, rate limits, and external call logs remain `HUMAN_REQUIRED`.

Assumptions:

- The local selected repository is `github:repo:owner/project-alpha`.
- The fixture cursor value is ISO-8601 timestamp text, so lexicographic monotonic comparison is deterministic for this local proof.
- The webhook verification booleans in fixtures represent the Connector boundary decision record, not a live cryptographic GitHub signature check.

Out of scope before coding:

- Live GitHub webhook delivery, HMAC verification against a real secret, polling API pagination, rate-limit handling, revoked permissions, and repository removal.
- Large/binary/secret-bearing content restrictions, now covered by `CS-CH-018`.
- GitHub writes, which remain denied and are covered separately by `CS-CH-019`.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-017` |
| Type | `MUST_PASS` |
| Trigger / Action | A selected GitHub-style source sends one valid webhook, an overlapping poll event for the same source revision, a changed source revision that crashes after commit but before cursor advancement, replay of that changed revision, a crash after cursor advancement, replay after cursor advancement, and an out-of-order older webhook. |
| Expected Result | Invalid webhook metadata is denied before commit. Valid webhook/poll overlap produces one logical receipt and Artifact. Reconciliation detects the post-commit/pre-cursor gap before replay. Replay advances the cursor only after durable commit. Crash after cursor advancement is replay-safe. Out-of-order observation preserves current cursor and source-revision lineage. Final reconciliation has zero duplicate logical Artifacts, zero cursor-before-commit, zero missed receipts, zero duplicate active truth, and zero raw/provider secret leakage. |
| Affected Layers | CLI, ConnectorPort adapter, sync signal receipt, sync cursor, sync reconciliation, Delivery archive/ack path, dedupe/content-version lineage, audit, scenario verifier. |
| Verification Method | Bad-webhook fixture, valid webhook fixture, duplicate poll fixture, changed Delivery fixture with `after_commit_before_cursor` and `after_cursor` fault modes, out-of-order webhook fixture, sync reconciliation before and after replay, lineage query, durable-state count inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | `cs.connector_sync_signal_receipt.v1`, `cs.connector_sync_cursor.v1`, `cs.connector_sync_reconciliation.v1`, provider-event key parts, cursor history, sync lag metrics, two delivery receipts, two Artifacts, two content versions, lineage output with one current truth, bad-webhook denial transcript, final negative counters for zero duplicate logical Artifacts, cursor-before-commit, missed cursor receipts, duplicate product events, source-revision lineage gaps, and unverified webhook commits. |
| Owner | AI |

## CS-CH-018 Contract

Goal:

Prove deterministic local GitHub content restriction and secret hygiene before durable Product state: permitted file excerpts may be used only after redaction, large or binary files become metadata-only, forbidden/generated paths are skipped, and private-key material is quarantined.

Constraints:

- Product / UX: Product state may explain partial or skipped content, but must not silently treat restricted content as complete truth.
- Domain / Data: content restriction is a Connector/Source Policy decision, linked to the Projection Delivery, setup result, policy decision, Artifact or quarantine state, evidence refs, and audit refs.
- Security: token-like strings and private-key material must not appear in generated outputs, reports, durable connector state, audit payloads, or screenshots.
- Reliability: skip and quarantine actions happen before Artifact, receipt, ack, content-version, or current-truth creation.
- Observability: each restriction records a stable decision id, action, partial status, reason code, safe normalized payload, marker scan metadata, and negative evidence counters.
- Performance / Migration: local proof uses deterministic decisions that map to future indexed policy-decision rows by scope, contract version, delivery id, path, and content status.
- Operational / Environment: proof is local deterministic fixture evidence only; live GitHub content scanning, API pagination, rate limits, and external secret-scanner integrations remain `HUMAN_REQUIRED`.

Assumptions:

- The local selected repository is `github:repo:owner/project-alpha`.
- The Source Policy allows `docs/**` and `README.md`, with `max_content_bytes=200000` and `content_mode=metadata_and_markdown_excerpt`.
- The private-key fixture is a canary for sensitive private material and must be quarantined rather than redacted into Product state.

Out of scope before coding:

- Live GitHub blob fetches, Git LFS, submodules, generated-file detection beyond local deterministic path patterns, and external secret scanner services.
- Human review UX for quarantined repository content.
- GitHub writes, which remain denied and are covered separately by `CS-CH-019`.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-018` |
| Type | `MUST_PASS` |
| Trigger / Action | Six selected-repository file-snapshot Projection fixtures are processed: token-like Markdown excerpt, binary file, oversized file, forbidden path, generated path, and private-key material. |
| Expected Result | Token-like excerpts are redacted into sanitized Artifact input. Binary and oversized files create metadata-only Artifacts and receipts. Forbidden and generated paths are skipped before Artifact/receipt/ack creation. Private-key material is quarantined before Product state. All decisions are linked to Source Policy and audit evidence, with zero raw sensitive or out-of-policy marker leaks. |
| Affected Layers | CLI, ConnectorPort adapter, Source Policy gate, content restriction decision store, Delivery archive/ack path, quarantine state, Artifact ingest, audit, scenario verifier. |
| Verification Method | Six file-snapshot fixtures, delivery process commands, durable-state count inspection, content restriction decision inspection, quarantine inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | `cs.connector_content_restriction_decision.v1`, three delivery receipts, three Artifacts, three ack outboxes, one quarantine record, six decision records, redacted normalized payload, metadata-only normalized payloads, skip transcripts, quarantine transcript, final negative counters for zero raw sensitive marker leaks, imports outside allowed paths, generated Artifacts, binary raw-content imports, large-file silent truncations, and private-material Artifacts. |
| Owner | AI |

## CS-CH-019 Contract

Goal:

Prove deterministic local GitHub zero-write posture for Connector Hub adoption: GitHub remains a read-only source, and every Action, CLI, runtime, Provider Pack, and controlled egress surface rejects source-control mutation.

Constraints:

- Product / UX: normal users and admins may connect/read selected GitHub sources, but the Product CLI must not expose GitHub comment, label, merge, push, issue-create, file-write, release, workflow-dispatch, or settings-update commands.
- Domain / Data: connector capability contracts may describe source-control read needs and Projection delivery only; GitHub/source-control write Actions are invalid and must not be registered.
- Architecture: Provider Pack manifests remain transport-replaceable and read-only; the Product cannot depend on provider SDK mutation helpers.
- Security: write HTTP methods and mutation endpoints are denied locally with `external_http_calls=0`, `provider_mutations=0`, no direct provider handles, and no credentials or secrets in state or reports.
- Observability: denied write contracts, static guard output, direct runtime denials, negative counters, and audit verification must be present in the filtered scenario report.
- Performance / Migration: local proof uses deterministic provider-pack, contract, CLI, and egress scans that can map to future Provider Pack tables, Action declaration constraints, egress policy rows, and audit joins.
- Operational / Environment: this is local fixture proof only. Live GitHub App manifest permission review remains `HUMAN_REQUIRED` under `CS-CH-H01`.

Assumptions:

- The selected source is `github:repo:owner/project-alpha`.
- GitHub is only a source connector for this phase; governed writeback scenarios are future CH-4 work and must not be implemented through GitHub in this scope.
- The local guard matrix covers representative mutation classes: issue creation, comments, labels, file writes, pull-request merge, branch creation, repository settings update, and deletion.

Out of scope before coding:

- Live GitHub permission attestation, installation review, or external provider mutation smoke tests.
- Implementing governed GitHub write Actions.
- UI/browser proof that no hidden GitHub write controls render.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-019` |
| Type | `MUST_PASS` |
| Trigger / Action | A negative GitHub contract attempts to declare write Actions, the static guard scans Provider Pack/contract/CLI/runtime surfaces, and direct runtime write attempts are executed for multiple GitHub mutation operations. |
| Expected Result | Write Action declarations are rejected before persistence; Provider Packs and active contracts expose no write mappings; product CLI exposes no GitHub mutation commands; controlled egress and direct-write attempts are denied with zero external HTTP calls and zero provider mutations; audit verification passes. |
| Affected Layers | CLI, ConnectorPort contract validator, Provider Pack fixture registry, runtime direct-write policy, egress guard report, local durable audit state, scenario verifier. |
| Verification Method | Negative write-action contract fixture, `cornerstone connector github-write-guard --json`, operation-specific `cornerstone connector direct-write-test --provider github --operation ... --json`, durable-state inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | `CS_CONNECTOR_GITHUB_WRITE_ACTION_DENIED` validation error, `cs.connector_github_write_guard.v1`, controlled egress attempt matrix, direct-write denial payloads, audit refs, and negative counters for zero source-control Actions declared, Provider Pack write mappings, provider mutations, GitHub write calls, write permissions requested, exposed write CLI commands, accepted write contracts, allowed write egress, and runtime write endpoint literals. |
| Owner | AI |

## CS-CH-020 Contract

Goal:

Prove deterministic local GitHub provider failure handling for Connector Hub adoption: source availability failures become stable health/setup/freshness states, prior evidence is preserved, and CornerStone never claims suspended or removed sources are current.

Constraints:

- Product / UX: Mission Control, Search, Claims, and setup diagnostics must have data-contract fields that distinguish delayed, suspended, unavailable, and transient states; local proof verifies the JSON contract, not rendered browser UI.
- Domain / Data: provider failures attach to scope, contract id, source ref, setup/source policy evidence, failure fixture, stream state, freshness state, recovery path, and audit refs.
- Architecture: failure-state handling remains behind ConnectorPort and does not introduce Product dependency on GitHub SDK objects, raw provider responses, credentials, or direct GitHub calls.
- Security: revoked permissions and repository removal must suspend/stop future ingestion without deleting prior Artifacts; no secrets, provider handles, external HTTP calls, or provider mutations may appear in state.
- Reliability: rate limits and transient transport failures must produce bounded retry schedules; revoked permissions require owner action and new verification; removed repositories stop future ingestion.
- Observability: every failure mode records stable reason codes, health/setup/freshness metadata, owner recovery guidance, warning metadata for Search/Claim surfaces, and negative evidence counters.
- Performance / Migration: local proof uses deterministic failure fixtures that can map to future provider health rows, source availability rows, retry queue rows, and audit joins.
- Operational / Environment: this is local fixture proof only. Live GitHub App revocation, live repository removal, live rate-limit headers, and real alert delivery remain `HUMAN_REQUIRED` under `CS-CH-H01`.

Assumptions:

- The selected source is `github:repo:owner/project-alpha`.
- Existing evidence for the selected repository has already been archived through the local Projection Delivery path before provider failure simulation.
- Reconnection or repository re-selection is an owner action and requires a new verification record before ingestion resumes.

Out of scope before coding:

- Live GitHub API failure reproduction, OAuth/App installation revocation, webhook retry delivery, repository rename migration, and alert notification delivery.
- Rendered UI/browser proof for Health, Setup Result, Mission Control, Search, and Claims warning states.
- Automatic reconnection, repository substitution, or any provider-side write/mutation.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-020` |
| Type | `MUST_PASS` |
| Trigger / Action | Four local GitHub provider failure fixtures are simulated against a selected source after baseline evidence exists: `rate_limit`, `permission_revoked`, `repository_removed`, and `transient_transport`. |
| Expected Result | Rate limits produce a bounded retry schedule, visible freshness delay, and no tight retry loop. Revoked permissions produce a permanent setup gap and suspend affected streams. Repository removal stops future ingestion and records source unavailability. Transient transport failures retry safely without permanent setup gaps. Existing Artifacts remain evidence with stale/unavailable Search and Claim warnings, and reconnection/re-selection requires owner action plus new verification. |
| Affected Layers | CLI, ConnectorPort adapter, provider failure-state store, setup/freshness state contract, Delivery evidence preservation, audit, scenario verifier. |
| Verification Method | Baseline selected-source delivery processing, four local failure-state simulation commands, durable-state count inspection, warning metadata inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | `cs.connector_provider_failure_state.v1`, four failure state records, rate-limit retry time, revoked-permission setup gap, repository-removed source unavailable state, transient retry state, preserved Artifact/receipt counts, Search/Claim warning metadata, owner-action/new-verification recovery metadata, audit refs, and negative counters for zero silent data deletion, fresh claims while suspended, tight retry loops, removed-repository future ingestion, reconnect without owner action, provider mutations, and external GitHub calls. |
| Owner | AI |

## CS-CH-021 Contract

Goal:

Prove deterministic local macOS WatchAgent capture gating for Connector Hub adoption: no activity capture occurs until explicit owner consent and platform permission are both present, and local fixture proof cannot be mistaken for physical-device production proof.

Constraints:

- Product / UX: setup diagnostics must explain data categories, sampling interval, privacy mode, retention, namespace, and pause/delete controls before any source can be enabled.
- Domain / Data: consent and platform permission are distinct inspectable records; neither record alone starts collection.
- Architecture: capture gating remains behind ConnectorPort and does not introduce OS-specific capture APIs into Product/Mission code.
- Security: no screenshots, raw window titles, keystrokes, clipboard values, browser history, external calls, provider mutations, or cross-namespace capture may appear in local proof.
- Observability: every permission probe, consent record, and guard decision must include evidence refs, audit refs, stable reason codes, and negative counters.
- Operational / Environment: this is local metadata-only fixture proof. Physical Mac permission prompts, first real sample, pause/revoke behavior, and human privacy acceptance remain `HUMAN_REQUIRED` under `CS-CH-H02`.

Assumptions:

- The local source id is `macos_activity`.
- Platform permission state is fixture input for AI verification and is never treated as production proof.
- Both gates being present makes the source ready for future collection, but the guard-evaluate command itself must not create samples or Artifacts.

Out of scope before coding:

- Real macOS APIs, system permission prompts, screenshots, active app/window capture, background agent startup, pause/revoke lifecycle, rendered UI/browser proof, and human privacy acceptance.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-021` |
| Type | `MUST_PASS` |
| Trigger / Action | A local macOS WatchAgent source is probed or evaluated before consent, after permission only, after consent only, and after both consent and fixture permission are present. |
| Expected Result | Capture remains blocked until consent and platform permission are both active. Permission probes are metadata-only. Consent records do not grant platform permission. The both-gates-present state is ready but does not start capture or create samples. Unsupported or missing states explain safe resolution. |
| Affected Layers | CLI, ConnectorPort adapter, Watch source consent store, permission probe store, capture guard decision store, audit, scenario verifier. |
| Verification Method | `connector capture permission probe`, `connector capture consent granted`, `connector capture guard evaluate`, state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | `cs.connector_capture_permission_probe.v1`, `cs.connector_watch_source_consent.v1`, `cs.connector_capture_guard_decision.v1`, two permission probes, one consent record, four guard decisions, setup diagnostics, audit refs, and negative counters for zero capture before consent/platform permission, zero samples before both gates, zero hidden startup capture, zero cross-namespace capture, zero screenshots/window titles/keystrokes/clipboard/browser history, zero external calls, and zero provider mutations. |
| Owner | AI |

## CS-CH-022 Contract

Goal:

Prove deterministic local ActivitySession projection for Connector Hub adoption: permissioned foreground-app observations are transformed into bounded, privacy-safe work-session candidates without surfacing raw event noise or fabricating user intent.

Constraints:

- Product / UX: default surfaces should show bounded sessions with confidence, caveats, source coverage, and privacy state rather than raw foreground-app event streams.
- Domain / Data: observed facts, inferred work mode, confidence, caveats, source refs, and retention state must remain separate fields.
- Architecture: sessionization remains behind ConnectorPort and persists connector-owned records; Product/Mission code receives ActivitySession projections rather than app-specific capture samples.
- Security: domain-only and privacy-safe modes must not reconstruct raw titles, full URLs, keystrokes, clipboard values, screenshots, cookies, or browser history.
- Observability: the sessionization record must expose algorithm version, thresholds, deterministic metrics, dedupe/idle/noise records, evidence refs, audit refs, and negative counters.
- Operational / Environment: this is local deterministic fixture proof. Physical device activity capture, Chrome privacy review, pause/revoke behavior, and human acceptance remain `HUMAN_REQUIRED` under `CS-CH-H02` and `CS-CH-H03`.

Assumptions:

- The input sample batch has already passed the `CS-CH-021` consent and platform-permission gates.
- The deterministic fixture is sufficient to prove the sessionization data contract, not real user behavior.
- Project hints are treated as weak grouping signals, not evidence of user intent.

Out of scope before coding:

- Real macOS APIs, browser APIs, screenshots, raw titles, full URLs, keystrokes, clipboard values, cookies, browser history, rendered UI/browser proof, human privacy acceptance, and mission/memory promotion of accepted sessions.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-022` |
| Type | `MUST_PASS` |
| Trigger / Action | A permissioned local activity sample batch with app switches, duplicates, idle gaps, low-information noise, sparse data, source refs, and privacy flags is passed to the sessionizer. |
| Expected Result | CornerStone persists a sanitized sample batch, a versioned sessionization record, and bounded ActivitySession projections with deterministic boundaries, confidence, caveats, source coverage, evidence refs, and audit refs. Duplicates are deduped, idle gaps split sessions, low-information noise is filtered with visible metrics, and unsupported intent is not stored as observed truth. |
| Affected Layers | CLI, ConnectorPort adapter, activity sample batch store, ActivitySession projection store, audit, scenario verifier. |
| Verification Method | `connector capture sessionize`, focused CLI regression test, filtered scenario verifier, aggregate scenario verifier, state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | `cs.connector_activity_sample_batch.v1`, `cs.connector_activity_sessionization.v1`, `cs.activity_session_projection.v1`, one sample batch, one sessionization record, three bounded ActivitySession projections, input metrics, filtered-sample records, source refs, retention record, audit refs, and negative counters for zero unsupported intent, zero inference-as-observed-fact, zero raw titles/full URLs/keystrokes/clipboard/screenshots/cookies/browser history, zero external calls, zero provider mutations, and zero Artifacts from sessionization. |
| Owner | AI |

## CS-CH-023 Contract

Goal:

Prove the first Product/Mission Watch Rule lifecycle for Connector Hub adoption: an owner explicitly defines what sources to watch, what match criteria apply, what outputs are allowed, and which version is active, without granting implicit capture expansion or external Action authority.

Constraints:

- Product / UX: Watch Rules must be understandable as owner-scoped Product/Mission automation intent, not provider-native scheduler internals.
- Domain / Data: source refs, connector contract refs, Source Policy refs, allowed outputs, retention, active version, pending version, policy decisions, and evaluation traces must remain explicit durable fields.
- Architecture: Watch Rule lifecycle remains behind ConnectorPort and cannot bypass Workflow/Action governance; evaluation may produce a Watch Result trace but not an external action.
- Security: rule text cannot grant provider mutation, external action execution, hidden capture, cross-namespace access, ownerless/global scope, or capture broadening without explicit confirmation.
- Observability: every create, activation decision, pause, resume, edit, delete, cross-scope denial, and evaluation path must expose evidence refs, audit refs, stable reason codes, version ids, and negative counters.
- Operational / Environment: this is local deterministic lifecycle proof. Live Watch Rule evaluation against real connected sources, rendered UI/API lifecycle proof, browser/device capture, notifications, and human privacy acceptance remain outside this AI-only proof.

Assumptions:

- The Watch Rule source refs point at local fixture source contracts and Source Policies already proved by earlier Connector Hub units.
- A missing-source activation denial is a successful safety result, not a scenario failure.
- Edits that narrow or tighten match criteria can create a pending draft version without replacing the active version.

Out of scope before coding:

- Real scheduler loops, live provider reads, browser or macOS capture, rendered UI/API proof, notifications, Workflow execution, Action card creation, external provider mutation, and physical deletion of historical Watch Rule state.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-023` |
| Type | `MUST_PASS` |
| Trigger / Action | An owner creates a scoped Watch Rule, attempts activation while a declared source is missing, activates after sources are ready, evaluates the active version, pauses/resumes/deletes the rule, edits it into a second version, and attempts cross-namespace access. |
| Expected Result | CornerStone persists an owner-scoped Watch Rule, version records, policy decisions, and an evaluation trace. Missing-source activation is denied and leaves the rule in draft. Ready activation pins version 1 as active. Evaluation traces version 1. Pause/resume/delete transitions are audited. Edit creates version 2 without broadening and keeps the original active version until explicit activation. Cross-namespace show is denied. No external action, provider mutation, hidden capture, ownerless/global rule, or Artifact is created. |
| Affected Layers | Native CLI, ConnectorPort adapter, Watch Rule store, Watch Rule version store, Watch Rule policy decision store, Watch Rule evaluation trace store, audit, scenario verifier. |
| Verification Method | `watch rule create`, `watch rule activate`, `watch rule evaluate`, `watch rule pause`, `watch rule resume`, `watch rule edit`, `watch rule show`, `watch rule delete`, focused CLI regression test, filtered scenario verifier, aggregate scenario verifier, state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | `cs.watch_rule.v1`, `cs.watch_rule_version.v1`, `cs.watch_rule_policy_decision.v1`, `cs.watch_rule_evaluation_trace.v1`, one Watch Rule, two Watch Rule versions, four policy decisions, one evaluation trace, active/pending version ids, lifecycle statuses, cross-scope denial transcript, audit refs, and negative counters for zero ownerless/global rules, zero cross-namespace lifecycle mutation, zero authority expansion, zero external Actions, zero provider mutations, zero external HTTP calls, zero unconfirmed broadening, and zero Artifacts from Watch Rule evaluation. |
| Owner | AI |

## CS-CH-024 Contract

Goal:

Prove explicit Chrome active-tab capture for Connector Hub adoption: the owner deliberately captures the current browser page through a bounded activeTab-style payload, and CornerStone creates summary-only Capture Inbox evidence only after backend policy revalidation.

Constraints:

- Product / UX: opening a popup or extension surface alone must never capture; capture requires explicit user gesture and confirmation.
- Domain / Data: permission event, sanitized payload, policy decision, summary, and inbox item must remain separate durable records with scope, source refs, evidence refs, audit refs, hashes, and raw-data flags.
- Architecture: the backend revalidates the extension payload instead of trusting the browser-side preflight result; Capture Inbox output is not an Artifact or Claim by itself.
- Security: activeTab temporary access cannot become broad `<all_urls>` access; raw HTML, raw text, cookies, local/session storage, screenshots, form values, and browser history are not stored.
- Observability: no-consent denial, popup/browser-internal denial, allowed capture, durable counts, negative counters, provider-internal findings, and audit verification must be visible in the filtered report.
- Operational / Environment: this is local deterministic fixture proof. Real unpacked-extension behavior, Chrome permission UI, browser privacy review, and human acceptance remain `HUMAN_REQUIRED`.

Assumptions:

- The local fixture payload represents the browser extension handoff after a deliberate owner action.
- `chrome_active_tab` is a declared local source id with explicit owner consent before the allowed capture path.
- The text clip may be used transiently to derive a summary, but the clip itself is not durable state.

Out of scope before coding:

- Real Chrome extension packaging, browser UI screenshots, live activeTab permission events, live webpage capture, allowlist auto-capture, pause/revoke UI, Evidence Bundle promotion, Claim creation, and human browser privacy acceptance.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-024` |
| Type | `MUST_PASS` |
| Trigger / Action | Owner attempts active-tab capture before consent, records explicit `chrome_active_tab` source consent, attempts popup/browser-internal capture, then captures an allowed active tab with user gesture, confirmation, activeTab temporary permission, active-page scope, and bounded payload. |
| Expected Result | No-consent capture is denied by backend policy. Popup/browser-internal capture is denied and creates no summary or inbox item. Allowed capture persists a permission event, sanitized payload, allow policy decision, summary-only capture record, and pending Capture Inbox item. Durable state stores only hashes, metadata, counts, policy decisions, evidence refs, and audit refs; it stores no raw text, raw HTML, cookies, storage, screenshots, form values, browser history, broad permission, external call, provider mutation, or Artifact. |
| Affected Layers | Native CLI, ConnectorPort adapter, active-tab payload validation, capture policy decision store, Capture Inbox store, audit, scenario verifier. |
| Verification Method | `connector capture browser active-tab`, `connector capture consent granted --source-id chrome_active_tab`, focused CLI regression test, filtered scenario verifier, aggregate scenario verifier, state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | `cs.connector_chrome_active_tab_permission_event.v1`, `cs.connector_chrome_active_tab_payload.v1`, `cs.connector_chrome_active_tab_policy_decision.v1`, `cs.connector_chrome_active_tab_capture_summary.v1`, `cs.capture_inbox_item.v1`, one consent record, two permission events, two sanitized payloads, three policy decisions, one summary, one inbox item, zero Artifacts, no-consent denial reason, popup/browser-internal denial reasons, allowed backend checks, audit refs, and negative counters for zero broad permission, zero capture without gesture/confirmation, zero popup capture, zero non-active-tab capture, zero backend bypass, zero blocked-page text persistence, zero raw browser data, zero external calls, and zero provider mutations. |
| Owner | AI |

## CS-CH-025 Contract

Goal:

Prove allowlist-based Chrome auto capture for Connector Hub adoption: background-style browser capture is allowed only when both sides of the consent boundary are present, the active page is allowlisted, the source pack is allowed, browser permission is specific, and backend policy creates only summary/inbox review state.

Constraints:

- Product / UX: auto capture must be owner-controlled by explicit source consent plus a confirmed Watch/source-pack/site rule; an extension-side trigger is never authority by itself.
- Domain / Data: config, trigger, policy decision, summary, and inbox item must remain separate durable records with scope, source id, version ids, hashes, evidence refs, audit refs, and raw-data flags.
- Architecture: backend policy revalidates the trigger against the latest persisted config and source consent; Capture Inbox output is not an Artifact, Claim, Workflow, or Action.
- Security: site allowlist, source-pack allowlist, browser host permission, consent/config version, active-page scope, throttle, session limit, idempotency, and raw-browser-data checks are all denial gates.
- Observability: no-config denial, blocked-trigger diagnostics, allowed trigger, duplicate denial, durable counts, negative counters, provider-internal findings, and audit verification must be visible in the filtered report.
- Operational / Environment: this is local deterministic fixture proof. Real Chrome extension background behavior, permission UI, browser privacy review, and human acceptance remain `HUMAN_REQUIRED`.

Assumptions:

- The local trigger fixture represents the browser extension handoff after an owner-enabled auto-capture rule observes an active tab event.
- `chrome_auto_capture` is a declared local source id with explicit owner consent before config and allowed trigger processing.
- A text clip may be used transiently to derive a summary, but the clip itself and blocked-site origin text are not durable state.

Out of scope before coding:

- Real Chrome extension packaging, live browser host-permission UI, live webpage capture, pause/revoke UI, sensitive-page degradation beyond the fixture policy reasons, Evidence Bundle promotion, Claim creation, Workflow execution, Action card creation, and human browser privacy acceptance.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-025` |
| Type | `MUST_PASS` |
| Trigger / Action | Owner attempts auto-capture before consent/config, records explicit `chrome_auto_capture` source consent, records a two-sided auto-capture config, submits a blocked trigger, submits an allowed trigger, then resubmits the allowed trigger as a duplicate. |
| Expected Result | No-config trigger is denied. Config persists only when owner rule, site allowlist, source-pack allowlist, and browser permission are present. Blocked trigger is denied with diagnostics and creates no summary or inbox item. Allowed trigger persists one sanitized trigger, allow policy decision, summary-only capture record, and pending Capture Inbox item. Duplicate trigger is denied by idempotency and creates no second summary or inbox item. Durable state stores only hashes, metadata, counts, policy decisions, evidence refs, and audit refs; it stores no raw text, raw HTML, cookies, storage, screenshots, form values, browser history, blocked-site origin text, broad permission, external call, provider mutation, or Artifact. |
| Affected Layers | Native CLI, ConnectorPort adapter, auto-capture config validation, auto-capture trigger validation, capture policy decision store, Capture Inbox store, audit, scenario verifier. |
| Verification Method | `connector capture browser auto-config`, `connector capture browser auto-capture`, `connector capture consent granted --source-id chrome_auto_capture`, focused CLI regression test, filtered scenario verifier, aggregate scenario verifier, state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | `cs.connector_chrome_auto_capture_config.v1`, `cs.connector_chrome_auto_capture_trigger.v1`, `cs.connector_chrome_auto_capture_policy_decision.v1`, `cs.connector_chrome_auto_capture_summary.v1`, `cs.capture_inbox_item.v1`, one consent record, one config record, two persisted trigger records, four policy decisions, one summary, one inbox item, zero Artifacts, no-config denial reasons, blocked-trigger denial reasons, allowed backend checks, duplicate idempotency denial, audit refs, and negative counters for zero capture without owner rule/site/source-pack/browser permission, zero version mismatch bypass, zero unapproved-domain capture, zero inactive-tab capture, zero throttle/session/idempotency bypass, zero raw browser data, zero external calls, and zero provider mutations. |
| Owner | AI |

## CS-CH-026 Contract

Goal:

Prove sensitive Chrome page capture policy for Connector Hub adoption: sensitive browser surfaces are blocked or degraded before capture content becomes searchable evidence, and backend revalidation can maintain or increase the client preflight restriction.

Constraints:

- Product / UX: blocked and degraded attempts must be visible to the owner as safe history guidance, not silent failure.
- Domain / Data: sensitive-page policy decisions, degraded payloads, and history items must remain separate durable records with scope, source id, hashes, evidence refs, audit refs, and raw-data absence flags.
- Architecture: browser preflight is untrusted input; backend policy can preserve a client block, increase a false-safe allow to block, or degrade only to metadata/hash output.
- Security: password, payment, token-like, private-account, browser-internal, unsupported-scheme, raw-browser-data, and false-safe signals are block gates; compose/unknown-editable and oversized pages are degradation gates.
- Observability: the filtered report must expose policy statuses, durable counts, negative counters, provider-internal findings, and audit verification for every case.
- Operational / Environment: this is local deterministic fixture proof. Real Chrome extension page classification, browser UI, screenshots/recordings, and human browser privacy acceptance remain `HUMAN_REQUIRED`.

Assumptions:

- The fixture represents the browser extension handoff after local preflight classification has produced a block, degrade, or allow recommendation.
- Backend policy treats all fixture cases as sensitive and therefore creates no Artifact, Capture Inbox item, or model-send path.
- Degraded output may persist only policy-approved metadata and hashes, never raw text, full URL/origin/title, raw HTML, cookies, storage, screenshots, form values, or browser history.

Out of scope before coding:

- Real Chrome extension classification, live browser page capture, rendered UI screenshots, pause/revoke lifecycle, Evidence Bundle promotion, Claim creation, Workflow execution, Action card creation, and human browser privacy acceptance.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-026` |
| Type | `MUST_PASS` |
| Trigger / Action | Owner/browser fixture evaluates password, payment, token-like false-safe, mail compose, private-account, browser-internal, unsupported-scheme, and oversized Chrome page cases through `connector capture browser sensitive-policy`. |
| Expected Result | Password, payment, token-like false-safe, private-account, browser-internal, and unsupported-scheme cases are blocked. Mail compose and oversized-page cases are degraded to hash-only metadata. Backend policy never downgrades a client block, blocks malicious false-safe sensitive payloads, persists one policy decision and one history item per case, persists degraded payloads only for degraded cases, and creates no Artifact, Capture Inbox item, raw browser persistence, model-send side effect, external call, or provider mutation. |
| Affected Layers | Native CLI, ConnectorPort adapter, sensitive-page policy validation, degraded metadata store, history item store, audit, scenario verifier. |
| Verification Method | `connector capture browser sensitive-policy`, focused CLI regression test, filtered scenario verifier, aggregate scenario verifier, state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | `cs.connector_chrome_sensitive_page_policy_decision.v1`, `cs.connector_chrome_sensitive_page_degraded_payload.v1`, `cs.connector_chrome_sensitive_page_history_item.v1`, eight policy decisions, two degraded payloads, eight history items, zero Capture Inbox items, zero Artifacts, block/degrade reason codes, backend false-safe block check, client block preservation check, audit refs, and negative counters for zero block downgrade, zero false-safe bypass, zero raw browser persistence, zero model sends, zero searchable content Artifacts, zero Capture Inbox items, zero external calls, and zero provider mutations. |
| Owner | AI |

## CS-CH-027 Contract

Goal:

Prove capture lifecycle controls for Connector Hub adoption: a user can pause, resume, revoke, retain, export, review, and delete eligible local capture state while immutable evidence and audit obligations remain explicit.

Constraints:

- Product / UX: lifecycle controls must be owner-visible decisions, not hidden side effects; deletion must explain what is deleted, disabled, retained, or anonymized.
- Domain / Data: lifecycle state, decisions, export bundles, deletion receipts, and result reviews must remain separate durable records with scope, evidence refs, audit refs, retention metadata, and negative counters.
- Architecture: pause/revoke state is a capture policy gate; new sample attempts must check lifecycle state before sample creation.
- Security: exports are scoped and redacted; deletion never promises full erasure of immutable evidence/audit obligations and never deletes audit records.
- Observability: the filtered report must expose lifecycle statuses, durable counts, negative counters, provider-internal findings, and audit verification.
- Operational / Environment: this is local deterministic fixture proof. Real Chrome/macOS device behavior, rendered UI/API controls, destructive-action ActionCard approval UX, and human privacy acceptance remain `HUMAN_REQUIRED` or `NOT_VERIFIED`.

Assumptions:

- The fixture represents already-collected WatchAgent and Chrome capture metadata, not live collection.
- Local delete execution is constrained to deterministic fixture state and is explicitly gated by `--execute --authorized`.
- Retained audit/evidence records are represented by retention classes and receipts, not by deleting actual user data.

Out of scope before coding:

- Real device capture, rendered UI screenshots, browser extension pause/revoke controls, live provider calls, Product ActionCard approval UI, production retention policy, and physical deletion of real user data.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-027` |
| Type | `MUST_PASS` |
| Trigger / Action | Owner/fixture seeds collected WatchAgent and Chrome state, pauses one source, resumes it, pauses one Watch Rule and all collection, revokes Chrome capture authority, changes retention, exports scoped state, saves/dismisses capture results, dry-runs deletion, then executes authorized local fixture deletion. |
| Expected Result | Pause and revoke decisions persist and deny subsequent sample attempts before sample creation; configuration is retained; retention changes persist; export is scoped to the requested source and redacted; save/dismiss result reviews persist; delete dry-run explains delete/disable/retain/anonymize boundaries; authorized local delete removes only eligible derived state, disables affected source state, retains audit/evidence obligations, and records a deletion receipt. |
| Affected Layers | Native CLI, ConnectorPort adapter, capture lifecycle state store, lifecycle decision store, export store, deletion receipt store, result review store, audit, scenario verifier. |
| Verification Method | `connector capture lifecycle ...` commands, focused CLI regression test, filtered scenario verifier, aggregate scenario verifier, durable state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | `cs.connector_capture_lifecycle_source_state.v1`, `cs.connector_capture_lifecycle_decision.v1`, `cs.connector_capture_lifecycle_export.v1`, `cs.connector_capture_lifecycle_deletion_receipt.v1`, `cs.connector_capture_result_review.v1`, four source states, at least eight decisions, one export bundle, two deletion receipts, two result reviews, zero capture sample files from denied attempts, retained-audit explanation, audit refs, and negative counters for zero paused/revoked samples, zero unscoped/raw exports, zero misleading delete-everything claim, zero audit deletion, zero external calls, and zero provider mutations. |
| Owner | AI |

## CS-CH-028 Contract

Goal:

Prove Watch Result truth separation for Connector Hub adoption: CornerStone can turn connector-backed captured signals into a Watch Result that keeps observed facts, inference, evidence/caveats, and proposed next steps separate, reviewable, and non-executing.

Constraints:

- Product / UX: Watch Result must explain what was observed, what CornerStone hypothesizes, what evidence and caveats support it, and what optional next step is proposed without presenting inference as truth.
- Domain / Data: `WatchObservation`, `WatchInference`, `WatchResult`, correction, and review records must remain separate durable records with scope, evidence refs, audit refs, trust state, and negative counters.
- Architecture: Connector Hub delivers source evidence only; Product/Intelligence creates the inference candidate and proposal record over that evidence.
- Security: proposed actions or memory updates must not execute directly from a Watch Result; low-confidence or unsupported inference stays Draft/Hypothesis and requires owner review before memory, Claim, Mission, or ActionCard promotion.
- Observability: the filtered report must expose section-separation checks, trust states, review outcomes, durable counts, negative counters, provider-internal findings, and audit verification.
- Operational / Environment: this is local deterministic fixture proof. Rendered UI/browser proof, human trust-language acceptance, live capture, model-provider behavior, and production policy enforcement remain `HUMAN_REQUIRED` or `NOT_VERIFIED`.

Assumptions:

- The fixture represents already-committed captured signal metadata and evidence refs, not live capture or live model inference.
- Deterministic local inference is a fixture-owned candidate generator used to verify contracts; it is not a claim about LLM quality.
- User correction changes inference/review history but never mutates immutable observed evidence.

Out of scope before coding:

- Rendered UI screenshots, browser walkthrough, real model calls, live WatchAgent/Chrome capture, durable memory promotion, Claim creation, Mission creation, ActionCard creation, external provider calls, workflow execution, and production policy gates.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-028` |
| Type | `MUST_PASS` |
| Trigger / Action | Owner/fixture provides captured ActivitySession and Capture Inbox signal refs that support a possible interpretation and next step; CornerStone builds a Watch Result, applies a correction, attempts low-confidence memory approval, and reviews the result. |
| Expected Result | Watch Result contains distinct observation, inference, evidence/caveats, and proposed next-step sections; observed records contain only source-backed facts and evidence refs; inference records carry hypothesis confidence, model/version, caveats, alternatives, and Draft/Hypothesis trust state; proposals remain non-executing and declare authority/risk/ActionCard requirements; correction creates history over inference/review state without mutating observation records; low-confidence memory approval is denied until evidence/review is present. |
| Affected Layers | Native CLI, Product Watch Result builder, observation store, inference store, Watch Result store, correction store, review store, audit, scenario verifier. |
| Verification Method | `watch result build`, `watch result correct`, `watch result review`, and `watch result approve-memory` commands, focused CLI regression test, filtered scenario verifier, aggregate scenario verifier, durable state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | `cs.watch_observation.v1`, `cs.watch_inference.v1`, `cs.watch_result.v1`, `cs.watch_result_correction.v1`, `cs.watch_result_review.v1`, at least two observations, at least two inferences including one low-confidence hypothesis, one Watch Result, one correction, one review, one denied memory approval attempt, preserved observation hash, section labels, evidence refs, caveats, alternatives, proposed action/memory metadata, audit refs, and negative counters for zero inferred intent labeled observed, zero proposal execution, zero direct memory approval, zero ActionCard creation, zero provider mutation, and zero external call. |
| Owner | AI |

CH-4 phase name override note:

The source implementation batch label is `Governed Actions (separate extension)`, while the current CornerStone phase plan uses `Governed Actions for separately approved connectors`. This is a scope clarification: the same `CS-CH-029` through `CS-CH-033` delivery units remain mapped to CH-4, but the current name preserves that action execution must be non-GitHub, separately approved, and blocked until corrected VS2/durable audit and human live-action proof are available. This note is traceability only; it does not add release readiness, live-provider proof, human approval, UI proof, or production proof.

## CS-CH-029 Contract

Goal:

Prove the first CH-4 governed-action integration for Connector Hub adoption: CornerStone can combine a Product ActionCard dry-run with a ConnectorHub Action Preflight for a separately approved non-GitHub connector action, while preserving distinct ownership, review requirements, and zero side effects.

Constraints:

- Product / UX: the combined review must show Product impact and connector feasibility together, including diff, expected impact, risk, approval need, evidence, provider support, permission state, Source Policy, input-schema status, idempotency, expected calls, and zero real calls.
- Domain / Data: ActionCard dry-run remains the Product safety envelope; ConnectorHub preflight is an input to that envelope and cannot count as approval or execution.
- Architecture: preflight must use a ConnectorPort-style record and must not expose provider clients, credentials, handles, or provider SDK objects to Product code.
- Security: GitHub/source-control read-only connectors have no action path and cannot be admitted to this scenario; every denied preflight blocks execution and includes a safe resolution path.
- Observability: preflight records and combined reviews must expose case status, denial reason codes, expected calls, real calls, policy/evidence/audit refs, negative counters, provider-internal findings, and audit verification.
- Operational / Environment: this is local deterministic fixture proof. Live provider credentials, real external writeback, human approval, rendered UI/API review, and production policy enforcement remain `HUMAN_REQUIRED` or `NOT_VERIFIED`.

Assumptions:

- The fixture represents a non-GitHub connector action declaration such as a support-ticket update; it does not use a live provider.
- Product can create an ActionCard dry-run from an evidence-backed Claim and Mission before ConnectorHub preflight runs.
- Preflight may return `allow` for connector feasibility while Product approval is still pending.

Out of scope before coding:

- Real external provider mutation, real credential use, live provider permission checks, rendered UI review, user approval, WorkflowRun execution, outcome re-ingest, rollback execution, and production OPA/RLS/network egress enforcement.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-029` |
| Type | `MUST_PASS` |
| Trigger / Action | An evidence-backed Claim and Mission propose a non-GitHub external writeback ActionCard; CornerStone runs Product dry-run and ConnectorHub Action Preflight for allowed, undeclared, unsupported, missing-permission, invalid-input, missing-idempotency, and GitHub read-only cases. |
| Expected Result | The allowed case creates one combined review with Product dry-run diff/impact, policy decision, connector provider support, Source Policy, permission state, input-schema result, risk, idempotency key, expected provider calls, evidence refs, audit refs, and approval status. Denied cases block execution with stable reason codes and resolution paths. Preflight creates no execution result, no WorkflowRun, no provider mutation, no external HTTP call, no credential exposure, and no approval. GitHub read-only action attempts are denied before ConnectorHub action preflight can become allowed. |
| Affected Layers | Native CLI, Product ActionCard dry-run, ConnectorHub Action Preflight port, combined review store, action card persistence, audit, scenario verifier. |
| Verification Method | `action propose`, `action dry-run`, `connector action-preflight run`, `action execute`, focused CLI regression test, filtered scenario verifier, aggregate scenario verifier, durable state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | `cs.action_card.v0`, `cs.action_dry_run.v0`, `cs.connector_action_preflight.v1`, `cs.connector_action_preflight_review.v1`, one allowed preflight, five non-GitHub denial preflights, one GitHub read-only denial preflight, preflight refs stored on the ActionCard, combined review sections, product policy refs, Source Policy refs, evidence refs, audit refs, expected call ledger, real calls `0`, external HTTP calls `0`, provider mutations `0`, execution results `0`, WorkflowRuns `0`, credential leaks `0`, and negative counters for zero dry-run execution, zero preflight-as-approval, zero GitHub action admission, zero direct provider access, zero secrets, and zero provider internals. |
| Owner | AI |

## CS-CH-030 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-030` |
| Type | `MUST_PASS` |
| Goal | Prove the first CH-4 server-side Action Safety Envelope: a technically supported side-effecting non-GitHub connector Action cannot execute unless evidence, Product policy, authorized approval, current ConnectorHub preflight, connector permission, idempotency, scope, and workspace state are all valid at execution time. |
| Constraints | ConnectorHub preflight cannot infer Product approval; Product approval cannot infer connector permission; every denied execution must return a stable reason code, resolution path, audit event, and zero side effects; invalid approver attempts must be denied; stale preflight refs and stale action-binding inputs must be denied; wrong namespace execution must remain scope-denied; local fixture proof must not create Action Result, WorkflowRun, external HTTP call, provider mutation, real provider call, or credential exposure. |
| Assumptions | The local non-GitHub `supportdesk` connector remains a deterministic fixture; `owner` and scoped owner id are authorized local approvers; mutation cases are injected by changing persisted local fixture state before calling the real `action execute` path; successful external execution and outcome re-ingest are deferred to `CS-CH-031`. |
| Out of Scope | Live provider mutation, real credential use, rendered approval UI/API, actual WorkflowRun execution, outcome re-ingest, compensation/rollback execution, production OPA/RLS/network egress enforcement, and human live approval proof. |

| Field | Value |
|---|---|
| Trigger / Action | A side-effecting supportdesk connector Action has been proposed from an evidence-backed Claim/Mission and preflighted; execution is requested while one safety gate is missing, stale, unauthorized, or out of scope. |
| Expected Result | Execution is denied with the precise missing-gate reason and safe resolution. Missing evidence, denied Product policy/locked workspace, missing authorized approval, unauthorized approver, missing connector permission, missing idempotency, stale target/preflight binding, stale connector binding, and wrong namespace all block before execution. Denials persist Action Safety Envelope evidence and audit refs with zero external calls, provider mutations, real provider calls, execution results, or WorkflowRuns. |
| Affected Layers | Native CLI, Product ActionCard execution, Action Safety Envelope validator, ConnectorHub preflight ref validation, approval authority model, workspace mode policy, audit, scenario verifier. |
| Verification Method | `action propose`, `connector action-preflight run`, `action approve`, `action execute`, workspace lock/unlock commands, focused CLI regression test, filtered scenario verifier, aggregate scenario verifier, durable state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | `cs.action_card.v0`, `cs.connector_action_preflight.v1`, `cs.action_safety_envelope.v0` denial records, stable reason codes for missing evidence, Product policy denial, missing approval, unauthorized approver, missing connector permission, missing idempotency, stale target/preflight binding, stale connector binding, and scope denial; audit refs; safety envelope refs; execution result count `0`; WorkflowRun count `0`; external HTTP calls `0`; provider mutations `0`; real provider calls `0`; credential leaks `0`; and negative counters for zero Product-approved-without-evidence, zero ConnectorHub-permission-as-approval, zero unauthorized approval, zero stale-preflight execution, zero cross-namespace execution, zero side effects, and zero provider internals. |
| Owner | AI |

## CS-CH-031 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-031` |
| Type | `MUST_PASS` |
| Goal | Prove the first CH-4 successful governed Action execution path: a declared, evidence-backed, separately approved non-GitHub connector Action executes through ConnectorHub once, creates durable execution/result/receipt records, and re-ingests the external outcome as mission evidence. |
| Constraints | Execution must use the same ActionCard, current ConnectorHub preflight binding, Product policy allow, authorized approval, idempotency key, and selected Provider Pack validated by `CS-CH-030`; the local fixture may simulate provider mutation but must record zero direct Product/agent provider calls, zero real external HTTP calls, zero credential exposure, zero raw provider payload persistence, and zero duplicate side effects on replay. |
| Assumptions | The local `supportdesk` connector remains the deterministic non-GitHub fixture for governed action execution; live provider execution requires explicit human approval and redacted provider proof; full retry/conflict/compensation semantics are deferred to `CS-CH-033`; GitHub/source-control remains read-only. |
| Out of Scope | Real provider credentials, real external network mutation, live human approval proof, rendered WorkflowRun UI/API, destructive rollback execution, ambiguous-result reconciliation, concurrent idempotency locks, and production OPA/RLS/network egress enforcement. |

| Field | Value |
|---|---|
| Trigger / Action | All CS-CH-030 gates pass for a declared supportdesk ticket-update Action and `cornerstone action execute <action_id> --json` is invoked, then invoked again as an idempotency replay. |
| Expected Result | The first execution returns a `WorkflowRun`, `Action Result`, provider receipt, idempotency record, outcome Artifact, outcome Evidence Bundle, connected outcome, and audit refs. The replay returns the existing Action Result/provider receipt/idempotency record with zero duplicate side effect. |
| Affected Layers | Native CLI, Product ActionCard execution, Action Safety Envelope validator, ConnectorHub execution adapter, WorkflowRun persistence, Action Result mapper, provider receipt persistence, Artifact/Evidence Bundle ingestion, connected outcome recording, idempotency replay, audit. |
| Verification Method | `action propose`, `connector action-preflight run --case-id allowed`, `action approve`, two `action execute` calls, focused CLI regression test, filtered scenario verifier, aggregate scenario verifier, durable state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | `cs.workflow_run.v0`, `cs.action_result.v0`, `cs.connector_provider_receipt.v1`, `cs.connector_action_idempotency.v1`, `cs.artifact.v0` outcome Artifact, `cs.evidence_bundle.v0` outcome bundle, `cs.connected_outcome.v0`, preflight/policy/approval/result/receipt/audit refs, one WorkflowRun, one Action Result, one provider receipt, one idempotency record, one connected outcome, one outcome Artifact, replayed result/receipt IDs, duplicate side effects `0`, direct provider calls `0`, external HTTP calls `0`, credential leaks `0`, and raw provider payload persistence `false`. |
| Owner | AI |

## CS-CH-032 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-032` |
| Type | `REGRESSION_GUARD` |
| Goal | Prove the CH-4 bypass boundary: undeclared connector Actions, direct Product/agent/provider access, embedded provider clients, and credential-owning Agent Pack logic are denied by backend/runtime enforcement even if UI omission or review screens are bypassed. |
| Constraints | ConnectorPort exposes only declared capabilities; ConnectorHub preflight must deny undeclared or unsupported Actions before provider calls; Product/agent direct provider writeback must return a stable denial, policy/audit refs, and zero calls/mutations; Agent Pack registry validation must quarantine provider clients, extension-owned credentials, direct API writeback, and raw secret access; output/state must expose no provider client, raw credential value, or secret-like value. |
| Assumptions | The local `supportdesk` connector preflight fixture is the non-GitHub action fixture; the direct-provider Agent Pack fixture is intentionally malicious and may persist its forbidden-runtime booleans as quarantine evidence; static import proof scans Product package imports for direct provider SDK/network clients; production egress topology proof remains covered by VS2 and `CS-CH-036`. |
| Out of Scope | Real provider credentials, real external network mutation, production network gateway enforcement, penetration testing, browser/UI-only omission proof, GitHub write-path coverage already owned by `CS-CH-019`, and idempotency retry/compensation semantics owned by `CS-CH-033`. |

| Field | Value |
|---|---|
| Trigger / Action | An evidence-backed supportdesk ActionCard is preflighted with an undeclared `support.ticket.delete` case, the owner approves the ActionCard anyway, execution is requested, a direct provider SDK-style write test is invoked, and a malicious Agent Pack manifest with provider clients/credentials/direct write/raw secret access is imported. |
| Expected Result | ConnectorHub preflight returns `deny` with `CS_CONNECTOR_ACTION_PREFLIGHT_UNDECLARED_ACTION`; approved execution remains backend-denied with `CS_ACTION_PREFLIGHT_NOT_ALLOWED`; direct provider writeback returns `CS_DIRECT_WRITE_DENIED`; credential boundary proof shows ConnectorHub custody only; Agent Pack import is quarantined; audit verification passes; Action Result and WorkflowRun counts remain `0`; direct provider calls, external HTTP calls, provider mutations, real provider calls, provider clients exposed, credential values exposed, and undeclared actions executed all remain `0`. |
| Affected Layers | Native CLI, ConnectorHub action preflight, Product Action Safety Envelope, direct provider bypass guard, Agent Pack registry validation/quarantine, credential boundary proof, static import scan, audit, scenario verifier. |
| Verification Method | `connector action-preflight run --case-id undeclared_action`, `action approve`, `action execute`, `connector direct-write-test`, `connector credential-boundary-test`, `pack import` with malicious direct-provider fixture, static direct provider SDK import scan over `packages/cornerstone_cli`, filtered scenario verifier, aggregate scenario verifier, durable state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | Denied `cs.connector_action_preflight.v1` with undeclared reason code, blocked `cs.action_safety_envelope.v0`, denied direct-write payload, safe `cs.connector_credential_boundary_test.v0`, quarantined `cs.agent_pack_quarantine.v0`, policy refs, audit refs, static import findings `[]`, provider-internal findings `[]`, Action Result count `0`, WorkflowRun count `0`, quarantine count `>=1`, and negative counters for undeclared actions executed, direct provider calls, external HTTP calls, provider mutations, real provider calls, provider clients exposed, credential values exposed, and malicious pack activations all `0`. |
| Owner | AI |

## CS-CH-033 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-033` |
| Type | `MUST_PASS` |
| Goal | Prove the first CH-4 idempotent retry and compensation-visibility contract: side-effecting ConnectorHub Action retries use a stable idempotency scope and request digest, same-key/same-intent retries return the existing result, same-key/different-intent retries are rejected before provider calls, and compensation expectations are visible without executing hidden rollback. |
| Constraints | The idempotency record must be keyed by owner/workspace scope, ConnectorHub/provider-pack scope, and idempotency key; the request digest must cover action type, target, source policy, required permission, selected resources, and input fingerprint; process-restart replay must be proven through durable state; ambiguous-response/timeout simulation must not create a second Action Result, WorkflowRun, provider receipt, provider mutation, or outcome Artifact; conflicting intent must return a stable denial and audit refs; compensation is a visible expectation/candidate only, never automatic execution. |
| Assumptions | The local `supportdesk` connector fixture simulates timeout/ambiguous-provider-response semantics through durable idempotency/reconciliation metadata; each CLI invocation is a separate process and can prove process-restart replay against persisted state; live provider retry behavior requires the human live non-GitHub Action gate; production distributed locks are outside this local fixture proof. |
| Out of Scope | Real provider timeout, real provider duplicate response, actual distributed locking, live external mutation, automatic compensation execution, rollback mutation, production queue retry workers, production OPA/RLS/network egress enforcement, UI/API rendering of reconciliation state, and human live provider rehearsal. |

| Field | Value |
|---|---|
| Trigger / Action | A declared supportdesk ticket-update Action executes successfully with an idempotency key; execution is invoked again from a separate CLI process with the same key/request digest; then another approved Action attempts to reuse the same idempotency key for a different target/input digest. |
| Expected Result | First execution creates exactly one WorkflowRun, Action Result, provider receipt, idempotency record, connected outcome, outcome Artifact, and outcome Evidence Bundle. Same-key/same-digest retry returns the existing result and increments retry/reconciliation metadata with duplicate side effect `0`. Same-key/different-digest retry is denied with `CS_ACTION_IDEMPOTENCY_CONFLICT`, records conflict evidence/audit refs, and does not create another WorkflowRun, Action Result, provider receipt, outcome Artifact, or provider effect. Compensation expectation is visible and `automatic_compensation_executed=false`. |
| Affected Layers | Native CLI, Product ActionCard execution, Action Safety Envelope, ConnectorHub execution adapter, idempotency persistence, request-digest calculation, provider receipt/result mapper, compensation expectation model, audit, scenario verifier. |
| Verification Method | `action propose`, `connector action-preflight run --case-id allowed`, `action approve`, first `action execute`, process-restart replay `action execute`, second ActionCard with `allowed_conflicting_intent` preflight using the same idempotency key, conflict `action execute`, focused CLI regression test, filtered scenario verifier, aggregate scenario verifier, durable state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | `cs.connector_action_idempotency.v1` with idempotency scope, request digest, first result refs, retry/reconciliation metadata, conflict attempt evidence, and compensation expectation; first and replayed Action Result IDs match; provider receipt IDs match; WorkflowRun count `1`; Action Result count `1`; provider receipt count `1`; connected outcome count `1`; outcome Artifact count `1`; conflict denial reason `CS_ACTION_IDEMPOTENCY_CONFLICT`; duplicate side effects `0`; hidden automatic compensation `0`; direct provider calls `0`; external HTTP calls `0`; provider mutations `0`; real provider calls `0`; credential leaks `0`; provider-internal findings `[]`. |
| Owner | AI |

## CS-CH-034 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-034` |
| Type | `MUST_PASS` |
| Goal | Prove every connector app, Delivery, Watch, evidence, and Action path is bound to trusted owner/namespace/workspace scope and denies cross-scope use without leaking objects or starting side effects. |
| Constraints | Connector app setup, Source Policy, Delivery receipt, Artifact, Evidence Bundle, Watch Result, Claim, Mission, ActionCard, preflight, approval, and audit records must carry complete scope metadata; every read/process/review/execute path must compare requested scope with durable object scope; denied paths must return stable `CS_SCOPE_DENIED` output and disclose no foreign object body; denied Action execution must not create WorkflowRuns, Action Results, provider receipts, external calls, or provider mutations. |
| Assumptions | Local fixture scope is `owner_id=local-user`, `namespace_id=personal`, `workspace_id=default`; cross-scope denial attempts use `owner_id=other-user`, `namespace_id=other`, and `workspace_id=default`; deterministic local state proves propagation and denial semantics but not production RLS/topology. |
| Out of Scope | Production Postgres RLS, OPA policy gateway deployment, real multi-tenant traffic, cross-namespace sharing/promote/copy UX, live provider calls, rendered UI/API proof, and production egress enforcement. |

| Field | Value |
|---|---|
| Trigger / Action | A scoped GitHub connector app is planned, a selected Projection Delivery is processed, an Evidence Bundle is assembled, a Watch Result is built, a claim/mission/action/preflight/approval chain is created, then setup, Delivery, Evidence Bundle, Watch Result review, and Action execution are repeated from a different owner/namespace scope. |
| Expected Result | In-scope setup, Delivery, Evidence Bundle, Watch Result, Claim, Mission, ActionCard, preflight, approval, and audit paths succeed with matching scope and evidence/audit refs. Cross-scope setup, Delivery, Evidence Bundle, Watch Result review, and Action execution all fail with `CS_SCOPE_DENIED` exit code `6`. Durable counts show one Delivery receipt, one Evidence Bundle, one Watch Result, one ActionCard, zero WorkflowRuns, zero Action Results, and zero provider receipts. Negative counters for cross-scope object returns, other-scope rows, ownerless rows, provider mutations, external calls, provider internals, and credential leaks all remain `0`. |
| Affected Layers | Native CLI, ConnectorHub contract setup, Source Policy snapshot, Delivery processing, Artifact archive, Evidence Bundle assembly, Watch Result review, Product claim/mission/action records, action preflight/approval, scope-denial error mapping, local durable state, audit, scenario verifier. |
| Verification Method | `connector contract validate`, `connector setup plan`, `connector delivery process`, `connector evidence bundle create`, `watch result build`, `claim create`, `claim approve`, `mission create`, `action propose`, `connector action-preflight run`, `action approve`, cross-scope setup/Delivery/evidence/Watch/action denial commands, focused CLI regression test, filtered scenario verifier, scenario gate, aggregate scenario verifier, durable state-count inspection, audit verification, provider-internal scan, secret scan, and negative evidence scan. |
| Evidence Required | Filtered `CS-CH-034` scenario report with `scope_isolation_checks` all `true`; denied command payloads with `CS_SCOPE_DENIED`; one scoped Delivery receipt, Evidence Bundle, Watch Result, and ActionCard; zero WorkflowRuns, Action Results, and provider receipts from denied execution; zero cross-scope object payload leaks; zero other-scope or ownerless records; zero provider-internal findings; zero secret findings; audit verification success. |
| Owner | AI |

## CS-CH-035 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-035` |
| Type | `REGRESSION_GUARD` |
| Goal | Prove Provider credentials stay exclusively inside ConnectorHub or an approved secret-manager boundary and Product-facing outputs expose only safe refs, fingerprints, status, evidence refs, and audit refs. |
| Constraints | Product/runtime commands must never persist or print raw credential values, reusable handles, auth headers, credential-bearing URLs, provider clients, or Product-owned secret writes. Rotation and revocation must update safe connection status metadata without moving the secret into Product state. Static Product/runtime scans must not find provider-auth imports. |
| Assumptions | The deterministic local fixture uses a canary ID and internally derived simulated secret; the raw canary is never passed as a CLI argument or written to reports. Production custody depends on the selected secret backend and operational access controls, which are outside local proof. |
| Out of Scope | Production secret-manager integration, live credential rotation/revocation, live provider audit logs, UI screenshots, production logs, and human operational access review. |

| Field | Value |
|---|---|
| Trigger / Action | ConnectorHub credential status, rotation, revocation, and boundary-check commands run for a GitHub connection fixture with a seeded canary ID. Product-facing stdout, durable state, audit, report fields, and Product/runtime source imports are inspected. |
| Expected Result | Status, rotate, and revoke commands persist `cs.connector_credential_lifecycle.v1` records with `credential_custody=connectorhub`, `secret_manager_boundary=ConnectorHub`, opaque `credential_ref`, short `credential_fingerprint`, safe connection status, evidence refs, and audit refs. Revocation records `status=revoked`; rotation records `last_rotated_at`. The raw canary is absent from stdout and state. Provider-auth import scan findings are empty. Credential boundary proof shows no direct provider access, no raw secret reads, no external calls, and no Product/agent credential exposure. |
| Affected Layers | Native CLI, LocalRuntimeStore credential lifecycle state, credential boundary command, audit, scenario verifier, static Product/runtime import scan, secret scan, generated scenario reports. |
| Verification Method | `connector credential status`, `connector credential rotate`, `connector credential revoke`, `connector credential-boundary-test`, `audit verify`, focused CLI regression test, filtered scenario verifier, scenario gate, aggregate scenario verifier, durable state-count inspection, seeded canary scan, provider-internal scan, secret scan, and static provider-auth import scan. |
| Evidence Required | Filtered `CS-CH-035` report with `credential_custody_checks` all `true`; three lifecycle records; one credential boundary record; zero raw canary stdout/state leaks; zero raw secret values, raw handles, auth headers, credential-bearing URLs, Product secret writes, provider-auth imports, provider internals, external calls, provider mutations, and secret findings; audit verification success. |
| Owner | AI |

## CS-CH-036 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-036` |
| Type | `MUST_PASS` |
| Goal | Prove ConnectorHub default-deny egress is enforced around Product/API, workers, tool runtime, and ConnectorHub-mediated external calls through the current reusable VS2 local network-boundary proof. |
| Constraints | Product, agents, workers, and tool runtime must not reach provider/network sinks directly. Only a governed egress proxy path with declared capability, policy, evidence, and audit context may reach the controlled provider fixture. The proof must not promote local Docker topology evidence into production network-control readiness. |
| Assumptions | The current AI-verifiable proof surface is the VS2 local Docker/internal-network harness plus egress proof reports. Production firewall/proxy/service-mesh topology, operator review, and independent network-control evidence remain human/external gates. |
| Out of Scope | Production network policy, Kubernetes/service-mesh/firewall rules, live external providers, production traffic logs, independent security review, and production operator approval. |

| Field | Value |
|---|---|
| Trigger / Action | Current VS2 local proof is regenerated or accepted as reusable, then ConnectorHub scenario verification inspects the VS2 egress rows, egress proof, local range report, Docker topology, direct API/worker/tool attempts, governed proxy attempt, provider counts, redirect/DNS/sandbox guards, audit records, and no-secret/no-payload checks. |
| Expected Result | Required VS2 egress rows pass; source-fingerprint reuse is current; API, worker, and tool-runtime containers fail direct HTTP and socket attempts to the provider; provider receives zero requests after direct attempts; governed egress proxy reaches the controlled provider once; redirect, DNS rebinding, protocol normalization, reserved destination, direct socket, sandbox/proxy/subprocess, untrusted-content, and fail-closed checks pass; egress audit records correlate attempts without storing raw payloads or credentials; production topology remains `NOT_VERIFIED`. |
| Affected Layers | Native scenario verifier, VS2 local security proof, Docker internal-network topology, egress proxy, controlled provider sink, policy/audit reports, ConnectorHub report lint boundaries, negative-evidence aggregation. |
| Verification Method | `cornerstone security vs2-local-proof --json`, `cornerstone scenario verify vs2-policy-tenancy-egress --reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json`, filtered `CS-CH-036` ConnectorHub scenario verifier, scenario gate, aggregate ConnectorHub verifier, aggregate gate, and focused CLI regression test. |
| Evidence Required | Filtered `CS-CH-036` report with `egress_topology_checks` all `true`; current reusable VS2 proof hash/source-fingerprint digests; required VS2 egress rows `PASS`; Docker topology with API/worker/tool separated from provider network; direct HTTP/socket attempts blocked; provider direct-attempt count `0`; governed proxy provider count `1`; zero direct bypass, denied-hop sink/trap calls, sensitive header forwarding, raw credential exposure, raw audit payloads, and production topology overclaim. |
| Owner | AI |

## CS-CH-037 Contract

Goal:

Prove ConnectorHub audit events are mirrored into the CornerStone audit ledger as safe correlation metadata, not as a replacement ledger and not as a copy of raw provider payloads or secrets.

Constraints:

- Product / Domain: connector events must remain attributable to affected ConnectorHub objects and CornerStone audit events.
- Architecture: ConnectorAuditBridge output is a local report over the tamper-evident CornerStone audit ledger; it does not bypass `LocalRuntimeStore.append_audit`.
- Security: correlation output must not copy raw payloads, secret values, credential handles, auth headers, provider clients, or direct provider internals.
- Observability: setup, policy, delivery, evidence, retry, quarantine, action, and credential event families must be independently visible with stable IDs.
- Reliability: audit verification must succeed before correlation is marked `PASS`, and tamper verification must fail closed on a modified audit copy.

Assumptions:

- The local fixture command sequence is sufficient to exercise all required ConnectorHub lifecycle event families.
- Connector event id is derived from the audited subject id, while CornerStone audit event id remains the ledger event id.
- Production audit retention, external SIEM export, and live provider audit logs are future proof surfaces.

Out of scope before coding:

- Production audit retention policy, external SIEM/log export, live provider audit-log comparison, UI audit explorer proof, and production security review.

Scenario Contract:

| Field | Value |
|---|---|
| ID | `CS-CH-037` |
| Type | `MUST_PASS` |
| Trigger / Action | A local ConnectorHub lifecycle sequence validates a contract, plans setup, processes delivery/evidence, schedules retry, quarantines and replays a poison delivery, confirms Source Policy, denies a direct write, records credential status/rotate/revoke, runs `cornerstone connector audit correlate --json`, verifies audit integrity, then verifies a tampered audit copy fails. |
| Expected Result | Every connector audit event correlates to a CornerStone audit event and affected object refs with a unique correlation ID. Required event families are present. Scope is consistent. Raw payloads and secrets are absent. Normal audit verification succeeds, and tampered audit verification fails closed. |
| Affected Layers | Native CLI, ConnectorRuntime audit-correlation report, LocalRuntimeStore audit hash chain, connector lifecycle commands, scenario verifier, negative-evidence aggregation, focused CLI regression test. |
| Verification Method | `connector audit correlate`, `audit verify`, tampered-state `audit verify`, filtered scenario verifier, scenario gate, aggregate scenario verifier, aggregate gate, focused CLI regression test, durable report inspection, provider-internal scan, and secret/raw-payload scan. |
| Evidence Required | Filtered `CS-CH-037` report with `audit_correlation_checks` all `true`; `cs.connector_audit_correlation.v1` report; setup/policy/delivery/evidence/retry/quarantine/action/credential family presence; connector event count equals correlated event count; sample correlation items include connector event id, CornerStone audit event id, affected refs, event hash, previous hash, `raw_payload_copied=false`, and `secret_copied=false`; zero audit-correlation negative counters; tamper error `AUDIT_EVENT_HASH_MISMATCH`. |
| Owner | AI |

## CS-CH-038 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-038` |
| Type | `MUST_PASS` |
| Trigger / Action | Owner plans a Provider Pack upgrade for an existing pinned connector capability contract. |
| Expected Result | Compatible target Provider Packs produce a migration plan while pinned versions remain active; incompatible target Provider Packs return a blocked upgrade plan with stable reason code, diff metadata, and rollback details. |
| Affected Layers | CLI, ConnectorPort adapter, upgrade planning state, Provider Pack registry, audit, scenario verifier. |
| Verification Method | Compatible and breaking Provider Pack fixtures, CLI upgrade-plan tests, scenario verifier, durable state inspection, audit verification, secret/provider-internal scan. |
| Evidence Required | Compatible and incompatible Upgrade Plan JSON, exit code `7` for incompatible upgrade, rollback provider ID, target Provider Pack IDs, audit refs, zero secret/internal findings. |
| Owner | AI |

## CS-CH-039 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-039` |
| Type | `REGRESSION_GUARD` |
| Trigger / Action | Connector-backed capabilities are exposed in product navigation, onboarding, CLI, and help surfaces. |
| Expected Result | Normal users see one CornerStone product with product concepts such as Connected Sources; connector internals stay progressively disclosed in admin/operator details and native commands begin with `cornerstone`. |
| Affected Layers | CLI, product walkthrough, Connected Sources surface contract, admin detail contract, scenario verifier. |
| Verification Method | Product walkthrough command, connector product-surface audit command, scenario verifier, forbidden normal-user term scan. |
| Evidence Required | Walkthrough JSON, product-surface audit JSON, zero forbidden normal-user term hits, zero sub-product counters, audit refs. |
| Owner | AI |

## CS-CH-040 Contract

| Field | Value |
|---|---|
| ID | `CS-CH-040` |
| Type | `REGRESSION_GUARD` |
| Trigger / Action | A connector scenario or release report is generated from local deterministic fixture proof. |
| Expected Result | The report labels local fixture evidence accurately, keeps live-provider, physical-device, human UX/privacy, publishing, and production security readiness separate, and never upgrades readiness by implication. |
| Affected Layers | CLI, scenario report schema, readiness dimensions, report linter, scenario verifier. |
| Verification Method | Generated candidate report plus `cornerstone connector report-lint --report ... --json`; scenario verifier checks readiness dimensions and overclaim counters. |
| Evidence Required | Candidate report JSON, report-lint JSON, readiness dimensions, negative overclaim counter `0`, audit refs. |
| Owner | AI |

Requirement enrichment note:

`CS-CH-040` includes `ER-07` in the current matrix because the report-lint guard is what keeps the ConnectorHub implementation trail engineer-executable: row statuses, human-required gates, exact evidence, negative overclaim counters, and weakest-verdict boundaries must remain machine-checkable instead of becoming a descriptive summary. This is an engineering-trail requirement enrichment only; it does not add live-provider, human-acceptance, UI, or production runtime proof.

## Implementation Approach

Senior developer perspectives folded into the first implementation:

- Product value: keep Connector Hub as an internal engine and expose a CornerStone connected-source setup path, not a second product.
- Product value: for `CS-CH-025`, make browser auto capture useful for recurring research and project monitoring without turning it into hidden surveillance or unreviewed evidence creation.
- Product value: for `CS-CH-026`, make browser capture trustworthy by refusing sensitive pages with clear owner guidance instead of silently capturing or silently failing.
- Product value: for `CS-CH-027`, give the owner a reversible control surface for continuous capture and retained local context without hiding retention/audit obligations.
- Product value: for `CS-CH-028`, make Watch Results useful as reviewable intelligence by showing what was observed, what CornerStone inferred, what caveats apply, and what non-executing next step is proposed.
- Domain correctness: Product code depends on a CornerStone ConnectorPort-style record and local adapter, not provider SDKs.
- Domain correctness: for `CS-CH-025`, treat browser auto capture as Watch/source consent plus site/source-pack/browser permission policy, not as extension-owned authority.
- Domain correctness: for `CS-CH-026`, treat browser page sensitivity as a backend policy decision over untrusted preflight signals, not as a browser-extension authority claim.
- Domain correctness: for `CS-CH-027`, treat pause/revoke/retention/export/delete as lifecycle decisions over captured state, not as data-file operations or implicit browser/agent behavior.
- Domain correctness: for `CS-CH-028`, treat observation, inference, and proposal as distinct truth states so source-backed evidence is not mixed with Product/Intelligence hypotheses or owner-review drafts.
- Architecture: add the adapter boundary additively; do not replace current Artifact/Search/Claim/Action flows.
- Architecture: for `CS-CH-025`, keep extension triggers as untrusted inputs and make backend policy the only component allowed to create Capture Inbox review state.
- Architecture: for `CS-CH-026`, keep sensitive-page policy decisions, degraded payloads, and owner-visible history items separate from capture summaries, Artifacts, Claims, Workflows, and Actions.
- Architecture: for `CS-CH-027`, keep lifecycle source state, decisions, exports, deletion receipts, and result reviews separate so deletion and export proof can be audited without mutating immutable evidence.
- Architecture: for `CS-CH-028`, keep Watch Observations, Watch Inferences, Watch Results, Corrections, and Reviews separate so correction history can change hypotheses without mutating immutable observed facts.
- Data contract: use versioned contract, setup result, and source policy schemas with explicit tenant, owner, namespace, and workspace.
- Reliability: persist contract, setup result, and policy before any future activation or delivery stream.
- Reliability: for `CS-CH-007`, persist the Artifact, exact original bytes, receipt, Projection snapshot, evidence link, and audit refs before any acknowledgement or Product interpretation.
- Reliability: for `CS-CH-008`, use a local inbox/outbox pattern so acknowledgement is sent only after durable commit and replay cannot create conflicting duplicate truth.
- Reliability: for `CS-CH-009`, keep failed Delivery retry/quarantine state separate from the healthy archive/ack path so a poison Delivery cannot block unrelated streams.
- Reliability: for `CS-CH-010`, keep transport redelivery idempotency and source-content versioning separate; duplicated events must not create duplicate active truth, and changed content must create a new linked version rather than overwriting the prior Artifact.
- Reliability: for `CS-CH-011`, make Source Policy evaluation a pre-archive gate so disallowed fields cannot create Artifact, receipt, current-version, or Product state.
- Reliability: for `CS-CH-012`, assemble Evidence Bundles only from committed connector delivery state so Product claims cannot depend on transient provider handles or EvidenceRef metadata alone.
- Reliability: for `CS-CH-013`, mediate every raw-access read through a durable grant record so expiry, read exhaustion, and revocation are checked on each access.
- Reliability: for `CS-CH-014`, record untrusted-content handling as a durable review linked to the Artifact, Delivery Receipt, Projection Snapshot, Evidence Link, and audit chain so safety checks survive replay and bundle assembly.
- Reliability: for `CS-CH-015`, make selected-repository scope part of Setup Result and Source Policy state, and reuse the pre-archive Source Policy gate so unselected events cannot create Artifact, receipt, ack, or current-version state.
- Reliability: for `CS-CH-016`, process repository, commit, change, issue, and file-snapshot families through the same pre-archive policy, immutable Artifact, content-version, and post-commit ack path.
- Reliability: for `CS-CH-018`, make content restriction a pre-archive decision so redacted/metadata-only inputs are sanitized before Artifact creation and skipped/quarantined inputs cannot create receipt, ack, content-version, or current-truth state.
- Reliability: for `CS-CH-019`, reject GitHub write declarations before contract persistence and record every runtime write attempt as a denied policy path with zero external calls and zero provider mutations.
- Reliability: for `CS-CH-021`, make capture readiness a guard decision over distinct consent and platform-permission records so permission-only or consent-only states cannot start collection.
- Reliability: for `CS-CH-022`, sessionize only sanitized permissioned samples, dedupe repeated events, split sessions on idle/project boundaries, filter low-information noise, and persist no Artifact or memory state from sessionization alone.
- Reliability: for `CS-CH-023`, keep lifecycle transitions scoped and durable, deny missing-source activation without setting an active version, preserve the active version across edits, and make delete reversible rather than physically destructive.
- Reliability: for `CS-CH-024`, deny no-consent and popup/browser-internal capture before summary creation, then create summary/inbox state only after backend policy revalidation succeeds.
- Reliability: for `CS-CH-025`, deny no-config, blocked, and duplicate triggers before summary creation, then create exactly one summary/inbox pair only after consent, config, allowlist, throttle, session, and idempotency checks pass.
- Reliability: for `CS-CH-026`, evaluate every sensitive-page case independently, preserve or increase the client restriction, and persist policy/history proof even when capture content is blocked.
- Reliability: for `CS-CH-027`, make sample-attempt checks consult lifecycle state before sample creation so paused/revoked sources produce durable denial decisions and zero sample files.
- Reliability: for `CS-CH-028`, deny low-confidence memory approval as a policy result, preserve observation hashes before and after correction, and persist review decisions without creating actions, claims, missions, or workflow runs.
- Data contract: for `CS-CH-010`, persist provider event ID, delivery idempotency key, source external ID, source revision, source content hash, content version ID, predecessor version ID, predecessor Artifact ID, and current-version pointer.
- Data contract: for `CS-CH-011`, persist a `cs.connector_projection_policy_decision.v1` record with allowed, excluded, forbidden, normalized, Source Policy, delivery, evidence, and audit references.
- Data contract: for `CS-CH-012`, persist a `cs.connector_evidence_bundle_link.v1` envelope inside the normal `cs.evidence_bundle.v0` item, plus a deterministic `cs.search_snapshot.v0` query snapshot.
- Data contract: for `CS-CH-013`, persist `cs.connector_raw_access_request.v1`, `cs.connector_raw_access_grant.v1`, and metadata-only `cs.connector_raw_access_export.v1` envelopes with purpose, classification, TTL, read limit, scope, evidence ref, Source Policy, and audit refs.
- Data contract: for `CS-CH-014`, persist `cs.connector_untrusted_content_review.v1` with source trust label, Artifact trust state, unsafe-instruction findings, evidence-only handling flags, required authority gates, and zero side-effect counters.
- Data contract: for `CS-CH-015`, persist `cs.connector_selected_resource_scope.v1` with provider kind, explicit selected-repository mode, available/selected/unselected counts, selected opaque source refs, namespace/version flags, no fallback flags, credential exposure flags, and write-permission flags.
- Data contract: for `CS-CH-016`, persist source ref, source external ID, source revision, content version, Artifact, receipt, evidence link, Evidence Bundle, and search snapshot IDs for each source-control Projection type.
- Data contract: for `CS-CH-018`, persist `cs.connector_content_restriction_decision.v1` with action, partial status, allowed path state, content size, declared size, marker scan metadata, safe normalized payload, source policy refs, evidence refs, and audit refs.
- Data contract: for `CS-CH-019`, emit `cs.connector_github_write_guard.v1` with Provider Pack scan, active contract scan, forbidden CLI/runtime scan, controlled egress matrix, and negative counters.
- Data contract: for `CS-CH-021`, persist `cs.connector_capture_permission_probe.v1`, `cs.connector_watch_source_consent.v1`, and `cs.connector_capture_guard_decision.v1` records with scope, source id, platform state, consent state, setup diagnostics, evidence refs, audit refs, and zero side-effect counters.
- Data contract: for `CS-CH-022`, persist `cs.connector_activity_sample_batch.v1`, `cs.connector_activity_sessionization.v1`, and `cs.activity_session_projection.v1` records with scope, source refs, algorithm metadata, input metrics, filtered-sample records, observed facts, confidence, caveats, privacy flags, evidence refs, audit refs, and inference separation.
- Data contract: for `CS-CH-023`, persist `cs.watch_rule.v1`, `cs.watch_rule_version.v1`, `cs.watch_rule_policy_decision.v1`, and `cs.watch_rule_evaluation_trace.v1` records with scope, source refs, connector contract refs, Source Policy refs, active/pending version ids, lifecycle status, allowed outputs, authority flags, evidence refs, and audit refs.
- Data contract: for `CS-CH-024`, persist `cs.connector_chrome_active_tab_permission_event.v1`, `cs.connector_chrome_active_tab_payload.v1`, `cs.connector_chrome_active_tab_policy_decision.v1`, `cs.connector_chrome_active_tab_capture_summary.v1`, and `cs.capture_inbox_item.v1` records with scope, source id, hashes, raw-data flags, evidence refs, audit refs, and negative counters.
- Data contract: for `CS-CH-025`, persist `cs.connector_chrome_auto_capture_config.v1`, `cs.connector_chrome_auto_capture_trigger.v1`, `cs.connector_chrome_auto_capture_policy_decision.v1`, `cs.connector_chrome_auto_capture_summary.v1`, and `cs.capture_inbox_item.v1` records with scope, source id, consent/config versions, source-pack refs, allowlist hashes, trigger idempotency, raw-data flags, evidence refs, audit refs, and negative counters.
- Data contract: for `CS-CH-026`, persist `cs.connector_chrome_sensitive_page_policy_decision.v1`, `cs.connector_chrome_sensitive_page_degraded_payload.v1`, and `cs.connector_chrome_sensitive_page_history_item.v1` records with scope, source id, case id, input hash, URL/origin/title hashes, reason codes, raw-data absence flags, evidence refs, audit refs, and negative counters.
- Data contract: for `CS-CH-027`, persist `cs.connector_capture_lifecycle_source_state.v1`, `cs.connector_capture_lifecycle_decision.v1`, `cs.connector_capture_lifecycle_export.v1`, `cs.connector_capture_lifecycle_deletion_receipt.v1`, and `cs.connector_capture_result_review.v1` records with scope, source id, target kind/id, lifecycle status, retention metadata, redaction flags, retained-audit explanation, evidence refs, audit refs, and negative counters.
- Data contract: for `CS-CH-028`, persist `cs.watch_observation.v1`, `cs.watch_inference.v1`, `cs.watch_result.v1`, `cs.watch_result_correction.v1`, and `cs.watch_result_review.v1` records with section ordering, trust state, confidence, caveats, alternatives, observation hashes, correction history, review decisions, evidence refs, audit refs, and negative counters.
- Data contract: for `CS-CH-037`, emit `cs.connector_audit_correlation.v1` with required event-family presence, one correlation item per connector audit event, connector event id, CornerStone audit event id, affected object refs, event hash, previous hash, scope, and explicit raw-payload/secret absence flags.
- Security: deny raw access in the fixtures, expose no credentials, keep provider calls at zero before activation, blocked setup, degraded setup, upgrade planning, or report linting, and deny Source Policy broadening.
- Security: keep Projection Delivery content untrusted, store no raw provider payload in Product state, and expose only EvidenceRef metadata plus immutable Artifact refs.
- Security: for `CS-CH-009`, quarantine stores only safe metadata, redacted reason codes, and Delivery references; it never persists raw provider payloads or credential-shaped diagnostics.
- Security: for `CS-CH-011`, reject forbidden full-body/raw fields and over-limit content before durable Product state, and prove by scanning local state for forbidden markers and raw-content leaks.
- Security: for `CS-CH-012`, deny EvidenceRef-only bundle requests, deny zero-evidence Claim approval, and keep raw access handles or provider payloads out of Evidence Bundle output.
- Security: for `CS-CH-013`, deny raw access unless contract and Source Policy declare `temporary_scoped`, human approval is present, purpose/classification/TTL/read limit fit policy, and outputs redact opaque handles while exposing only fingerprints.
- Security: for `CS-CH-014`, treat connector text as untrusted evidence regardless of content, block prompt-authority expansion through Agent policy, quarantine unsafe memory promotion, deny default egress, and scan for zero action/workflow/provider/shell/http side effects.
- Security: for `CS-CH-015`, deny unselected repository events with `CS_CONNECTOR_SOURCE_POLICY_RESOURCE_DENIED`, deny direct GitHub writes with `CS_DIRECT_WRITE_DENIED`, deny silent selected-resource expansion, and scan for zero unselected Artifacts/receipts/acks plus zero GitHub write permissions or calls.
- Security: for `CS-CH-016`, deny raw provider payload storage, keep Product logic provider-neutral, enforce the selected repository source ref for every Projection, and prove zero provider-specific Product-required fields or external provider calls.
- Security: for `CS-CH-018`, redact token-like excerpts, strip content fields for binary/oversized files, skip forbidden/generated paths, quarantine private-key material, and scan durable state plus CLI transcripts for zero raw markers or provider internals.
- Security: for `CS-CH-019`, reject GitHub/source-control write Actions with `CS_CONNECTOR_GITHUB_WRITE_ACTION_DENIED`, expose no write Provider Pack mappings or GitHub mutation CLI commands, and deny controlled write egress without provider access.
- Security: for `CS-CH-021`, keep permission probes metadata-only, exclude screenshots/raw window titles/keystrokes/clipboard/browser history, and require negative evidence for zero hidden startup capture, cross-namespace capture, external calls, or provider mutations.
- Security: for `CS-CH-022`, keep raw window titles, full URLs, keystrokes, clipboard values, screenshots, cookies, and browser history out of persisted session state, and require negative evidence for zero unsupported intent claims or inference-as-observed-fact storage.
- Security: for `CS-CH-023`, reject ownerless/global scope, deny cross-namespace reads and lifecycle mutations, ignore natural-language authority expansion, and require zero external action authority, provider mutation authority, external HTTP calls, and unconfirmed capture broadening.
- Security: for `CS-CH-024`, reject broad `<all_urls>` permission, no gesture, no confirmation, popup-only capture, browser-internal pages, non-active-tab capture, oversized payloads, raw browser data flags, external calls, and provider mutations.
- Security: for `CS-CH-025`, reject capture without owner rule, site allowlist, source-pack allowlist, specific browser permission, matching consent/config versions, allowed trigger type, active allowed page, throttle/session capacity, unique idempotency, and absent raw browser data.
- Security: for `CS-CH-026`, block password, payment, token-like false-safe, private-account, browser-internal, unsupported-scheme, and raw-browser-data cases; degrade compose/unknown-editable and oversized cases to hash-only metadata; and create no Artifact, Capture Inbox item, model-send, external call, or provider mutation.
- Security: for `CS-CH-027`, prevent paused/revoked sources from creating samples, keep exports scoped/redacted, require authorized local fixture deletion execution, retain audit/evidence obligations, and scan for zero raw exports, misleading erasure promises, external calls, or provider mutations.
- Security: for `CS-CH-028`, prevent inferred intent from being labeled as observed fact, deny low-confidence or unsupported inference promotion to Approved memory, and require zero direct proposal execution, ActionCard creation, Claim creation, Mission opening, workflow starts, raw content storage, external calls, or provider mutations.
- Security: for `CS-CH-037`, scan connector audit details and correlation output for forbidden raw payload, secret, auth-header, credential handle, provider-client, and direct-provider-access markers; a single leak fails the scenario.
- Observability: append audit events for contract validation, setup planning, Source Policy confirmation, upgrade planning, product-surface audit, and report linting.
- Observability: for `CS-CH-009`, append audit events for retry scheduling, quarantine creation, and replay request.
- Observability: for `CS-CH-010`, append audit events for delivery dedupe, content-version creation, current-version advancement, and lineage query.
- Observability: for `CS-CH-011`, append policy-enforcement and rejected-delivery audit events that link back to the policy decision record.
- Observability: for `CS-CH-012`, append search snapshot, Evidence Bundle creation, bundle assembly, Claim creation, approval, and denial audit events.
- Observability: for `CS-CH-013`, append raw-access denied, granted, read, read-denied, revoked, and metadata-exported audit events without storing handle or raw payload content.
- Observability: for `CS-CH-014`, append untrusted-content blocked/review-read events and include trust-boundary coverage plus denial audit refs in the scenario report.
- Observability: for `CS-CH-015`, include setup, allowed Delivery, denied unselected Delivery, direct-write denial, selection-broadening denial, and audit-integrity transcripts in the filtered scenario report.
- Observability: for `CS-CH-016`, include contract validation, setup plan, five delivery process transcripts, five Evidence Bundle transcripts, and audit-integrity verification in the filtered scenario report.
- Observability: for `CS-CH-018`, include six content restriction delivery transcripts, decision IDs/actions/partial statuses, quarantine ID, negative counters, provider-internal findings, and secret-scan counts in the filtered scenario report.
- Observability: for `CS-CH-019`, include rejected write-contract validation, static write-guard output, direct-write denial transcripts, negative counters, provider-internal findings, and audit-integrity verification in the filtered scenario report.
- Observability: for `CS-CH-021`, include permission probes, consent record, four guard decisions, setup diagnostics, physical-device `HUMAN_REQUIRED` marker, negative counters, provider-internal findings, and audit-integrity verification in the filtered scenario report.
- Observability: for `CS-CH-022`, include sessionization transcript, sample metrics, dedupe/idle/noise lists, session IDs, session source sample IDs, confidence caveats, retention flags, negative counters, provider-internal findings, and audit-integrity verification in the filtered scenario report.
- Observability: for `CS-CH-023`, include Watch Rule ids, version ids, policy decision ids, lifecycle statuses, evaluation trace id, trace-to-version link, durable counts, cross-scope denial transcript, negative counters, provider-internal findings, and audit-integrity verification in the filtered scenario report.
- Observability: for `CS-CH-024`, include no-consent, consent, popup-blocked, allowed-capture, and audit transcripts plus permission event ids, payload ids, policy ids, summary/inbox ids, durable counts, negative counters, provider-internal findings, and audit-integrity verification in the filtered scenario report.
- Observability: for `CS-CH-025`, include no-config, consent, config, blocked-trigger, allowed-trigger, duplicate-trigger, and audit transcripts plus config id, policy ids, trigger id, summary/inbox ids, durable counts, negative counters, provider-internal findings, and audit-integrity verification in the filtered scenario report.
- Observability: for `CS-CH-026`, include sensitive-policy and audit transcripts plus policy decision ids, degraded payload ids, history item ids, policy statuses, durable counts, negative counters, provider-internal findings, and audit-integrity verification in the filtered scenario report.
- Observability: for `CS-CH-027`, include seed, pause, sample-attempt, resume, Watch Rule pause, global pause, revoke, retention, export, review, delete dry-run, delete execute, and audit transcripts plus lifecycle ids, status summary, durable counts, negative counters, provider-internal findings, and audit-integrity verification in the filtered scenario report.
- Observability: for `CS-CH-028`, include build, correction, memory-approval denial, review, and audit transcripts plus observation ids, inference ids, Watch Result id, correction id, review ids, trust states, durable counts, negative counters, provider-internal findings, and audit-integrity verification in the filtered scenario report.
- Observability: for `CS-CH-037`, include audit-correlation command output, audit-integrity success, tampered-audit failure, required family presence, correlation counts, sample correlations, negative counters, and provider-internal findings in the filtered scenario report.
- Performance: for `CS-CH-010`, use deterministic keyed JSON records for local proof so duplicate detection is O(1) by idempotency/content key and future Postgres migration can map to unique indexes.
- Performance: for `CS-CH-011`, evaluate field membership and content-size limits before artifact writes so rejected deliveries avoid unnecessary durable writes.
- Performance: for `CS-CH-012`, resolve the evidence chain by durable IDs and existing Artifact provenance so bundle assembly does not scan unrelated connector state.
- Performance: for `CS-CH-013`, resolve grants by deterministic IDs and update read counters in one local record so fixture verification stays bounded and maps cleanly to future indexed rows.
- Performance: for `CS-CH-014`, derive review IDs deterministically from scope, contract version, delivery ID, and Artifact ID so verification reads one linked review record instead of scanning all connector content.
- Performance: for `CS-CH-015`, evaluate selected source refs before archive writes and count durable records by directory in local proof, mapping to future indexed `(scope, source_ref)` and policy-decision checks.
- Performance: for `CS-CH-016`, use deterministic keyed local records and count-based verification so future Postgres storage can index `(scope, source_ref, projection_type, source_external_id)`.
- Performance: for `CS-CH-018`, evaluate path/type/declared-size/sensitive markers before durable writes so skipped or quarantined content avoids unnecessary Artifact, receipt, ack, and content-version work.
- Performance: for `CS-CH-019`, keep zero-write scans bounded to Provider Pack manifests, active contract fixtures, parser declarations, and runtime command sources so local and CI proof stays fast.
- Performance: for `CS-CH-020`, keep failure-state simulation bounded to deterministic fixture records and state counts; no provider polling, live API retries, or background scheduler is part of local proof.
- Performance: for `CS-CH-021`, keep guard evaluation bounded to one latest consent record, one matching permission probe, and deterministic state counts; no OS polling or background collector is part of local proof.
- Performance: for `CS-CH-022`, keep local sessionization as one sorted pass over sanitized samples with deterministic IDs and count-based verification; no OS polling, browser scraping, or background collector is part of local proof.
- Performance: for `CS-CH-023`, keep lifecycle operations as deterministic scoped JSON writes and count-based verification; no polling loop, live source reads, notification fanout, or scheduler worker is part of local proof.
- Performance: for `CS-CH-024`, keep active-tab proof bounded to two fixture payloads, one consent record, deterministic IDs, hash-only storage, and count-based verification; no browser automation or page scraping is part of local proof.
- Performance: for `CS-CH-025`, keep auto-capture proof bounded to one config fixture, two trigger fixtures, deterministic IDs, hash-only storage, and count-based verification; no browser automation, background scraping, or scheduler loop is part of local proof.
- Performance: for `CS-CH-026`, keep sensitive-page proof bounded to one multi-case fixture, deterministic policy/history IDs, hash-only degraded metadata, and count-based verification; no browser automation or content extraction loop is part of local proof.
- Performance: for `CS-CH-027`, keep lifecycle proof bounded to one seed fixture, deterministic state/decision/export/receipt/review IDs, direct state lookup by scope/source/target, and count-based verification; no live collector, browser automation, or filesystem purge loop is part of local proof.
- Performance: for `CS-CH-028`, keep Watch Result proof bounded to one fixture, deterministic observation/inference/result/correction/review IDs, direct record reads, observation-hash comparison, and count-based verification; no model call or live capture loop is part of local proof.
- Performance: for `CS-CH-037`, scan the local audit JSONL once, build correlations directly from event subjects/details, and validate uniqueness/family presence with bounded sets so future storage can map to indexed audit-event joins.
- Testability: expose deterministic CLI commands and a scenario verifier that can be filtered to one scenario.
- Testability: for `CS-CH-011`, use isolated allowed, forbidden-body, and narrowed-limit fixtures plus negative state scans.
- Testability: for `CS-CH-012`, use a positive delivery-to-bundle path plus two negative paths: EvidenceRef-only bundle creation and zero-evidence Claim approval.
- Testability: for `CS-CH-013`, use independent positive and negative grants for default denial, TTL/read-limit boundary denial, successful read, read exhaustion, expiry, revocation, metadata export, and state leak scanning.
- Testability: for `CS-CH-014`, use a malicious GitHub issue Projection fixture plus deterministic CLI denial checks for Agent authority, memory promotion, and egress, and inspect state directories for zero side-effect records.
- Testability: for `CS-CH-015`, use a three-repository selected-resource contract fixture, one selected-repository Delivery, one unselected-repository Delivery, direct-write denial, Source Policy broadening denial, and deterministic state-count scans.
- Testability: for `CS-CH-016`, use five source-control Projection fixtures, one Evidence Bundle command per receipt, deterministic state counts, and negative scans for raw payload/provider-specific Product requirements.
- Testability: for `CS-CH-018`, use six isolated file-snapshot fixtures covering redact, metadata-only, skip, and quarantine paths, then verify counts for six decisions, three receipts, three Artifacts, three ack outboxes, and one quarantine record.
- Testability: for `CS-CH-019`, use one negative write-action contract, one static guard command, a controlled GitHub mutation matrix, direct-write denial commands per operation, and zero-counter assertions in both focused tests and filtered scenario verification.
- Testability: for `CS-CH-020`, use one baseline archived Delivery plus four provider failure-state fixture modes and assert recovery/freshness metadata, preserved evidence counts, warning metadata, and zero negative counters.
- Testability: for `CS-CH-021`, use metadata-only permission probes plus no-consent/no-permission, permission-only, consent-only, and both-gates-ready guard evaluations with zero sample and Artifact counts.
- Testability: for `CS-CH-022`, use one deterministic activity-sample batch with duplicate, app-switch, idle-gap, low-information, and sparse-data cases, then assert three bounded sessions, stable metrics, zero raw capture fields, zero unsupported intent, and zero Artifacts.
- Testability: for `CS-CH-023`, use one base Watch Rule fixture and one narrowed edit fixture, then assert draft create, denied missing-source activation, ready activation, evaluation trace pinning, pause/resume/delete, versioned edit, cross-scope denial, durable counts, and zero negative counters.
- Testability: for `CS-CH-024`, use one allowed active-tab fixture and one popup/browser-internal blocked fixture, then assert no-consent denial, consent recording, popup denial, allowed summary/inbox creation, durable counts, audit verification, and zero raw browser persistence.
- Testability: for `CS-CH-025`, use one auto-capture config fixture, one allowed trigger fixture, and one blocked trigger fixture, then assert no-config denial, consent/config recording, blocked diagnostics, allowed summary/inbox creation, duplicate denial, durable counts, audit verification, and zero raw browser persistence.
- Testability: for `CS-CH-026`, use one sensitive-page fixture with eight cases, then assert six block decisions, two degraded decisions, false-safe backend blocking, no client block downgrade, hash-only degraded payloads, history guidance, durable counts, audit verification, and zero raw/browser/model/artifact/inbox side effects.
- Testability: for `CS-CH-027`, use one capture lifecycle seed fixture, then assert source pause denial, resume allow check, Watch Rule/global pause persistence, Chrome revoke denial, retention update, scoped/redacted export, save/dismiss reviews, delete dry-run explanation, authorized delete receipt, durable counts, audit verification, and zero raw/export/provider side effects.
- Testability: for `CS-CH-028`, use one Watch Result fixture, then assert section ordering, source-backed observed facts, draft hypotheses with caveats/alternatives, correction hash preservation, low-confidence memory approval denial, draft review persistence, durable counts, audit verification, and zero proposal/action/workflow/provider side effects.
- Testability: for `CS-CH-037`, use one local lifecycle sequence that intentionally covers setup, policy, delivery, evidence, retry, quarantine, action, and credential audit families, then assert correlation counts, zero negative counters, no provider internals, and tampered audit failure.
- Maintainability: keep the local fixture adapter transport-replaceable and keep CH-1+ rows out of the local fixture claim until their own scenario units pass.
- Migration feasibility: use local JSON state for the first proof while keeping scope fields, activation blockers, feature availability, disabled surfaces, Source Policy diff hashes, provider-pack identity, upgrade plan IDs, readiness dimensions, and version IDs compatible with future Postgres/RLS storage.
- Migration feasibility: map `CS-CH-011` decisions to future policy-decision rows with unique delivery refs and no dependence on local filesystem paths.
- Migration feasibility: map `CS-CH-012` bundle links to future Evidence Bundle item tables and search snapshots without changing Claim approval semantics.
- Migration feasibility: map `CS-CH-013` request/grant/read state to future short-lived handle tables with unique scope/evidence/purpose keys, row-level tenant boundaries, TTL indexes, and audit joins.
- Migration feasibility: map `CS-CH-014` reviews to future connector safety-review rows with unique Artifact/Delivery refs, RLS scope columns, policy-decision joins, and indexed side-effect counters.
- Migration feasibility: map `CS-CH-037` correlation items to future audit-event join rows keyed by tenant, workspace, namespace, connector event id, CornerStone audit event id, and affected object refs without storing raw provider payloads.
- Migration feasibility: map `CS-CH-015` selected-resource scope to future GitHub installation/repository selection tables with opaque source refs, RLS scope columns, selection-version rows, policy-decision joins, and unique constraints preventing unselected ingestion.
- Migration feasibility: map `CS-CH-016` source-control Projection records to future repository object, source revision, content version, search snapshot, and Evidence Bundle tables without adding Product-owned GitHub schemas.
- Migration feasibility: map `CS-CH-018` content decisions to future content-policy rows with scope, contract version, source policy, delivery id, path, action, partial status, redaction fingerprints, and quarantine joins.
- Migration feasibility: map `CS-CH-019` validation rules to future Action declaration constraints, Provider Pack capability indexes, egress policy deny rows, CLI/API route audits, and provider mutation audit counters.
- Migration feasibility: map `CS-CH-020` failure states to future source availability, provider health, retry schedule, setup gap, warning, and audit tables keyed by scope, contract id, source ref, and failure mode.
- Migration feasibility: map `CS-CH-021` permission probes, Watch source consent records, and guard decisions to future source health, privacy consent, permission diagnostic, and audit tables keyed by scope and source id.
- Migration feasibility: map `CS-CH-022` sample batches, sessionization records, and ActivitySession projections to future activity-observation/session tables with scope, source id, source refs, algorithm version, source-sample joins, confidence/caveat fields, and indexed privacy-retention metadata.
- Migration feasibility: map `CS-CH-023` Watch Rules, versions, policy decisions, and evaluation traces to future `watch_rules`, `watch_rule_versions`, `watch_rule_policy_decisions`, and `watch_rule_evaluation_traces` tables keyed by tenant, owner, namespace, workspace, source refs, active version, and audit refs.
- Migration feasibility: map `CS-CH-024` permission events, active-tab payloads, policy decisions, summaries, and Capture Inbox items to future browser-capture tables keyed by scope, source id, payload hash, policy decision, summary id, inbox item id, and audit refs, with raw browser data retained only as absent flags and hashes.
- Migration feasibility: map `CS-CH-025` auto-capture configs, triggers, policy decisions, summaries, and Capture Inbox items to future browser-auto-capture tables keyed by scope, source id, config version, consent version, source pack, allowlist hash, trigger idempotency key, policy decision, summary id, inbox item id, and audit refs, with raw browser data retained only as absent flags and hashes.
- Migration feasibility: map `CS-CH-026` sensitive-page policy decisions, degraded payloads, and history items to future browser-sensitive-policy tables keyed by scope, source id, case id, policy decision, degraded payload id, history item id, URL/origin/title hashes, reason codes, and audit refs, with raw browser data retained only as absent flags and hashes.
- Migration feasibility: map `CS-CH-027` lifecycle source states, decisions, export bundles, deletion receipts, and result reviews to future capture lifecycle tables keyed by scope, source id, target kind/id, lifecycle decision id, retention class, deletion receipt id, review id, and audit refs, with raw capture payloads retained only as absent flags and hashes.
- Migration feasibility: map `CS-CH-028` observations, inferences, Watch Results, corrections, and reviews to future intelligence review tables keyed by scope, source refs, observation id, inference id, result id, correction id, review id, trust state, evidence refs, and audit refs, with raw source content retained only as absent flags and hashes.

## CLI Parity

- Command group: `cornerstone connector`
- Create/register command: `cornerstone connector contract validate --file fixtures/connectorhub/contracts/github_readonly_contract.json --json`
- Read/plan command: `cornerstone connector setup plan --contract-id ccon_project_alpha_github --json`
- Source Policy command: `cornerstone connector source-policy confirm --contract-id ccon_project_alpha_github --json`
- Delivery command: `cornerstone connector delivery ingest --file fixtures/connectorhub/deliveries/github_issue_projection_delivery.json --contract-id ccon_project_alpha_github --json`
- Durable ack command: `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_issue_projection_delivery.json --contract-id ccon_project_alpha_github --json`
- Policy enforcement command: `cornerstone connector delivery ingest --file fixtures/connectorhub/deliveries/github_issue_projection_delivery_forbidden_body.json --contract-id ccon_project_alpha_github --json`
- Selected-repository setup command: `cornerstone connector setup plan --contract-id ccon_selected_repos_github --json`
- Selected-repository Delivery command: `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_selected_repo_issue_projection_delivery.json --contract-id ccon_selected_repos_github --json`
- Unselected-repository denial command: `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_unselected_repo_issue_projection_delivery.json --contract-id ccon_selected_repos_github --json`
- GitHub direct-write denial command: `cornerstone connector direct-write-test --provider github --target github:repo:owner/project-alpha --json`
- GitHub write guard command: `cornerstone connector github-write-guard --json`
- GitHub write Action rejection command: `cornerstone connector contract validate --file fixtures/connectorhub/contracts/github_write_action_contract.json --json`
- GitHub failure-state command: `cornerstone connector github-failure simulate --failure-mode rate_limit --contract-id ccon_project_alpha_github --source-ref github:repo:owner/project-alpha --json`
- Capture permission probe command: `cornerstone connector capture permission probe --platform-permission-state not_granted --json`
- Capture consent command: `cornerstone connector capture consent granted --purpose <purpose> --json`
- Capture guard command: `cornerstone connector capture guard evaluate --platform-permission-state granted --json`
- Activity sessionization command: `cornerstone connector capture sessionize --file fixtures/connectorhub/activity_samples/macos_activity_samples_cs_ch_022.json --json`
- Chrome active-tab capture command: `cornerstone connector capture browser active-tab --file fixtures/connectorhub/chrome/active_tab_capture_allowed_cs_ch_024.json --json`
- Chrome auto-capture config command: `cornerstone connector capture browser auto-config --file fixtures/connectorhub/chrome/auto_capture_config_cs_ch_025.json --source-id chrome_auto_capture --json`
- Chrome auto-capture trigger command: `cornerstone connector capture browser auto-capture --file fixtures/connectorhub/chrome/auto_capture_allowed_cs_ch_025.json --source-id chrome_auto_capture --json`
- Chrome sensitive-page policy command: `cornerstone connector capture browser sensitive-policy --file fixtures/connectorhub/chrome/sensitive_pages_cs_ch_026.json --source-id chrome_sensitive_page --json`
- Capture lifecycle commands: `cornerstone connector capture lifecycle seed --file fixtures/connectorhub/capture/lifecycle_state_cs_ch_027.json --json`, `cornerstone connector capture lifecycle pause --source-id macos_activity --target-kind source --json`, `cornerstone connector capture lifecycle sample-attempt --source-id macos_activity --event-id sample-while-paused --json`, `cornerstone connector capture lifecycle resume --source-id macos_activity --target-kind source --json`, `cornerstone connector capture lifecycle revoke --source-id chrome_auto_capture --target-kind source --json`, `cornerstone connector capture lifecycle retention --source-id macos_activity --target-kind source --retention-days 7 --json`, `cornerstone connector capture lifecycle export --source-id macos_activity --include-history --json`, `cornerstone connector capture lifecycle review-result --result-id <capture_result_id> --decision save|dismiss --json`, and `cornerstone connector capture lifecycle delete --source-id macos_activity --dry-run|--execute --authorized --json`
- Watch Rule create command: `cornerstone watch rule create --file fixtures/connectorhub/watch_rules/project_alpha_watch_rule_cs_ch_023.json --json`
- Watch Rule activation command: `cornerstone watch rule activate --watch-rule-id <watch_rule_id> --source-readiness ready --json`
- Watch Rule lifecycle commands: `cornerstone watch rule evaluate --watch-rule-id <watch_rule_id> --source-evidence-ref <evidence_ref> --json`, `cornerstone watch rule pause --watch-rule-id <watch_rule_id> --json`, `cornerstone watch rule resume --watch-rule-id <watch_rule_id> --json`, `cornerstone watch rule edit --watch-rule-id <watch_rule_id> --file fixtures/connectorhub/watch_rules/project_alpha_watch_rule_edit_cs_ch_023.json --json`, `cornerstone watch rule show --watch-rule-id <watch_rule_id> --namespace-id other --json`, and `cornerstone watch rule delete --watch-rule-id <watch_rule_id> --json`
- Watch Result commands: `cornerstone watch result build --file fixtures/connectorhub/watch_results/project_alpha_watch_result_cs_ch_028.json --json`, `cornerstone watch result correct --watch-result-id <watch_result_id> --inference-id <watch_inference_id> --hypothesis <updated_hypothesis> --reason <reason> --json`, `cornerstone watch result approve-memory --watch-result-id <watch_result_id> --inference-id <watch_inference_id> --json`, and `cornerstone watch result review --watch-result-id <watch_result_id> --decision save_draft_memory --json`
- Selected-repository broadening denial command: `cornerstone connector source-policy confirm --contract-id ccon_selected_repos_github --selected-resource github:repo:owner/project-alpha --selected-resource github:repo:owner/project-beta --json`
- Source-control repository Delivery command: `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_repository_projection_delivery.json --contract-id ccon_selected_repos_github --json`
- Source-control commit Delivery command: `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_commit_projection_delivery.json --contract-id ccon_selected_repos_github --json`
- Source-control change Delivery command: `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_change_projection_delivery.json --contract-id ccon_selected_repos_github --json`
- Source-control issue Delivery command: `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_issue_projection_delivery.json --contract-id ccon_selected_repos_github --json`
- Source-control file-snapshot Delivery command: `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery.json --contract-id ccon_selected_repos_github --json`
- Content-restriction Delivery commands: `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery_secret_marker.json --contract-id ccon_project_alpha_github --json`, `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery_binary.json --contract-id ccon_project_alpha_github --json`, `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery_large.json --contract-id ccon_project_alpha_github --json`, `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery_forbidden_path.json --contract-id ccon_project_alpha_github --json`, `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery_generated.json --contract-id ccon_project_alpha_github --json`, and `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery_private_key.json --contract-id ccon_project_alpha_github --json`
- Incremental sync command: `cornerstone connector sync incremental --file fixtures/connectorhub/deliveries/github_issue_projection_delivery.json --contract-id ccon_project_alpha_github --signal webhook --cursor-id github:repo:owner/project-alpha:issues --json`
- Incremental sync reconciliation command: `cornerstone connector sync reconcile --cursor-id github:repo:owner/project-alpha:issues --json`
- Ack reconciliation command: `cornerstone connector delivery reconcile --json`
- Retry command: `cornerstone connector delivery process --file fixtures/connectorhub/deliveries/github_issue_projection_delivery.json --contract-id ccon_project_alpha_github --failure-mode transient --json`
- Quarantine list command: `cornerstone connector quarantine list --json`
- Quarantine replay command: `cornerstone connector quarantine replay --quarantine-id <id> --json`
- Lineage query command: `cornerstone connector lineage show --contract-id ccon_project_alpha_github --source-external-id github:repo:owner/project-alpha:issue:1001 --json`
- Evidence Bundle command: `cornerstone connector evidence bundle create --delivery-receipt-id <delivery_receipt_id> --query <query> --json`
- Raw-access request command: `cornerstone connector raw-access request --contract-id ccon_project_alpha_github_raw_access --evidence-ref-id <evidence_ref_id> --purpose diagnose_ingestion_gap --classification internal --ttl-seconds 60 --max-reads 1 --human-approved --json`
- Raw-access read command: `cornerstone connector raw-access read --grant-id <raw_access_grant_id> --json`
- Raw-access revoke command: `cornerstone connector raw-access revoke --grant-id <raw_access_grant_id> --reason <reason> --json`
- Raw-access export command: `cornerstone connector raw-access export --grant-id <raw_access_grant_id> --json`
- Untrusted-content review command: `cornerstone connector untrusted-content review --delivery-receipt-id <delivery_receipt_id> --json`
- Upgrade command: `cornerstone connector upgrade plan --contract-id ccon_project_alpha_github --target-provider-pack-id local_source_control_readonly_alt.v1 --json`
- Product-surface command: `cornerstone connector product-surface audit --json`
- Report-lint command: `cornerstone connector report-lint --report reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json --json`
- Audit-correlation command: `cornerstone connector audit correlate --json`
- Human-gate package command: `cornerstone connector human-gate package --scenario <CS-CH-H01..CS-CH-H07> --json`
- Human-gate package artifact command: `cornerstone connector human-gate package --scenario <CS-CH-H01..CS-CH-H07> --json --output reports/scenario/connectorhub-human-gate-package-<lower-scenario-id>-2026-06-24.json`
- Human-gate package artifacts: `reports/scenario/connectorhub-human-gate-package-cs-ch-h01-2026-06-24.json` through `reports/scenario/connectorhub-human-gate-package-cs-ch-h07-2026-06-24.json`
- Human-gate readiness report command: `cornerstone connector human-gate report --json --output reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json`
- Human-gate next selector: `cornerstone connector human-gate next --json`
- Human-gate validation handoff command: `cornerstone connector human-gate validation-handoff --json --output reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json`
- Human-gate record validation command: `cornerstone connector human-gate validate-record --scenario <CS-CH-H01..CS-CH-H07> --record-file <json> --json`
- Human-gate redacted validation-envelope command: `cornerstone connector human-gate validate-record --scenario <CS-CH-H01..CS-CH-H07> --record-file <json> --json --output <redacted-validation-envelope.json>`
- Human-gate record timestamp rule: `review_timestamp` must be an ISO-8601 timestamp with timezone, for example `2026-06-24T12:00:00Z`
- Engineering-trail manifest generation command: `make generate-connectorhub-engineering-trail-manifest`
- Engineering-trail verifier command: `make verify-connectorhub-engineering-trail`
- Scenario verification: `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-001 --json`
- Human-gate package schema: `cs.connector_human_gate_package.v1`
- Human-gate readiness report schema: `cs.connector_human_gate_readiness_report.v1`
- Human-gate validation handoff schema: `cs.connector_human_gate_validation_handoff.v1`
- Human-gate validation issue summary schema: `cs.connector_human_gate_validation_issue_summary.v1`
- Human-gate record validation schema: `cs.connector_human_gate_record_validation.v1`
- JSON schemas: `cs.connector_capability_contract.v1`, `cs.connector_setup_result.v1`, `cs.connector_source_policy.v1`, `cs.connector_selected_resource_scope.v1`, `cs.connector_projection_policy_decision.v1`, `cs.connector_content_restriction_decision.v1`, `cs.connector_github_write_guard.v1`, `cs.connector_provider_failure_state.v1`, `cs.connector_capture_permission_probe.v1`, `cs.connector_watch_source_consent.v1`, `cs.connector_capture_guard_decision.v1`, `cs.connector_activity_sample_batch.v1`, `cs.connector_activity_sessionization.v1`, `cs.activity_session_projection.v1`, `cs.connector_chrome_active_tab_permission_event.v1`, `cs.connector_chrome_active_tab_payload.v1`, `cs.connector_chrome_active_tab_policy_decision.v1`, `cs.connector_chrome_active_tab_capture_summary.v1`, `cs.connector_chrome_auto_capture_config.v1`, `cs.connector_chrome_auto_capture_trigger.v1`, `cs.connector_chrome_auto_capture_policy_decision.v1`, `cs.connector_chrome_auto_capture_summary.v1`, `cs.connector_chrome_sensitive_page_policy_decision.v1`, `cs.connector_chrome_sensitive_page_degraded_payload.v1`, `cs.connector_chrome_sensitive_page_history_item.v1`, `cs.capture_inbox_item.v1`, `cs.connector_capture_lifecycle_source_state.v1`, `cs.connector_capture_lifecycle_decision.v1`, `cs.connector_capture_lifecycle_export.v1`, `cs.connector_capture_lifecycle_deletion_receipt.v1`, `cs.connector_capture_result_review.v1`, `cs.watch_observation.v1`, `cs.watch_inference.v1`, `cs.watch_result.v1`, `cs.watch_result_correction.v1`, `cs.watch_result_review.v1`, `cs.watch_rule.v1`, `cs.watch_rule_version.v1`, `cs.watch_rule_policy_decision.v1`, `cs.watch_rule_evaluation_trace.v1`, `cs.connector_delivery_receipt.v1`, `cs.connector_projection_snapshot.v1`, `cs.connector_evidence_link.v1`, `cs.connector_evidence_bundle_link.v1`, `cs.connector_raw_access_request.v1`, `cs.connector_raw_access_grant.v1`, `cs.connector_raw_access_export.v1`, `cs.connector_untrusted_content_review.v1`, `cs.connector_ack_outbox.v1`, `cs.connector_ack_reconciliation.v1`, `cs.connector_delivery_retry_state.v1`, `cs.connector_delivery_quarantine.v1`, `cs.connector_quarantine_list.v1`, `cs.connector_delivery_dedupe_state.v1`, `cs.connector_content_version.v1`, `cs.connector_content_lineage.v1`, `cs.connector_sync_signal_receipt.v1`, `cs.connector_sync_cursor.v1`, `cs.connector_sync_reconciliation.v1`, `cs.connector_audit_correlation.v1`, `cs.connector_upgrade_plan.v1`, `cs.connector_product_surface_audit.v1`, `cs.connector_report_lint.v1`
- Exit codes covered: `0` success, `1` invalid input or Source Policy delivery denial, `3` not found, `4` evidence missing or report-lint failure, `5` runtime/retry scheduled failure, `6` scope denied, `7` connector unavailable for blocked required-capability setup, incompatible Provider Pack upgrade, or quarantined Delivery, `8` policy denied for raw-access requests/reads, prompt-authority expansion, default egress denial, direct GitHub write denial, selected-repository broadening denial, Watch Rule source-not-ready activation, Chrome active-tab capture policy denial, Chrome auto-capture policy denial, and Watch Result low-confidence memory approval denial
- Evidence refs emitted: connector contract, setup result, source policy, projection policy decision, content restriction decision, GitHub write guard, provider failure state, capture permission probe, Watch source consent, capture guard decision, activity sample batch, activity sessionization, ActivitySession projection, Chrome active-tab permission event, Chrome active-tab payload, Chrome active-tab policy decision, Chrome active-tab capture summary, Chrome auto-capture config, Chrome auto-capture trigger, Chrome auto-capture policy decision, Chrome auto-capture summary, Chrome sensitive-page policy decision, Chrome sensitive-page degraded payload, Chrome sensitive-page history item, Capture Inbox item, capture lifecycle source state, capture lifecycle decision, capture lifecycle export, capture lifecycle deletion receipt, capture result review, Watch Observation, Watch Inference, Watch Result, Watch Result correction, Watch Result review, Watch Rule, Watch Rule version, Watch Rule policy decision, Watch Rule evaluation trace, Artifact, storage ref, delivery receipt, Projection snapshot, evidence link, evidence bundle, search snapshot, Claim, raw-access request, raw-access grant, raw-access metadata export, untrusted-content review, memory quarantine, egress policy decision, ack outbox, ack reconciliation, retry state, quarantine item, dedupe state, content version, content lineage, sync signal receipt, sync cursor, sync reconciliation, audit correlation, human-gate package, human-gate readiness report, human-gate next selector, human-gate validation handoff, human-gate record validation, human-gate evidence-packet scaffold, upgrade plan, product surface, report lint, contract file hash
- Audit refs emitted: contract validated, setup planned, delivery archived, untrusted content blocked/reviewed/read, ack outbox created, delivery acknowledged, ack reconciled, delivery retry scheduled, delivery quarantined, content restriction quarantined, quarantine replay requested, delivery deduplicated, content version created, current version advanced, content lineage queried, sync webhook received, sync cursor advanced/observed, sync cursor reconciled, GitHub provider failure recorded, capture permission probed, capture consent recorded, capture guard evaluated, activity samples sessionized, Chrome active-tab policy decided, Chrome auto-capture config recorded, Chrome auto-capture policy decided, Chrome sensitive-page policy decided, capture lifecycle state seeded, capture lifecycle pause/resume/revoke/retention/sample attempt recorded, capture lifecycle exported, capture result reviewed, capture lifecycle delete dry-run/execution recorded, Watch Result built, Watch Result corrected, Watch Result reviewed, Watch Result memory approval denied, Watch Rule created, Watch Rule activation denied, Watch Rule activated, Watch Rule paused, Watch Rule resumed, Watch Rule edited, Watch Rule evaluated, Watch Rule deleted, Source Policy confirmed or denied, projection Source Policy enforced, rejected delivery recorded, GitHub write guard verified, direct write denied, credential status/rotate/revoke recorded, search snapshot created, Evidence Bundle created, connector Evidence Bundle assembled, Claim created, Claim approved, EvidenceRef-only bundle denied, zero-evidence Claim approval denied, raw access denied, raw access granted, raw access read, raw access read denied, raw access revoked, raw access metadata exported, agent prompt-authority denied, memory write quarantined, egress denied, audit correlation, human-gate package created, human-gate readiness reported, human-gate next selected, human-gate validation handoff created, human-gate record validated, human-gate evidence-packet scaffold reported, upgrade planned, product surface audited, report lint completed
- CLI status: `PASS` for CH-0 rows, `CS-CH-007`, `CS-CH-008`, `CS-CH-009`, `CS-CH-010`, `CS-CH-011`, `CS-CH-012`, `CS-CH-013`, `CS-CH-014`, `CS-CH-015`, `CS-CH-016`, `CS-CH-017`, `CS-CH-018`, `CS-CH-019`, `CS-CH-020`, `CS-CH-021`, `CS-CH-022`, `CS-CH-023`, `CS-CH-024`, `CS-CH-025`, `CS-CH-026`, `CS-CH-027`, `CS-CH-028`, `CS-CH-029`, `CS-CH-030`, `CS-CH-031`, `CS-CH-032`, `CS-CH-033`, `CS-CH-034`, `CS-CH-035`, `CS-CH-036`, and `CS-CH-037` only when filtered verifier evidence passes

## Proof Surface Matrix

| Surface | Verified Scope Status | Evidence |
|---|---|---|
| Document | FROZEN | This contract plus matrix CSV. |
| CLI | VERIFIED LOCALLY | `cornerstone connector contract validate`, `cornerstone connector setup plan`, `cornerstone connector delivery ingest`, `cornerstone connector delivery process`, `cornerstone connector delivery reconcile`, `cornerstone connector quarantine list`, `cornerstone connector quarantine replay`, `cornerstone connector lineage show`, `cornerstone connector sync incremental`, `cornerstone connector sync reconcile`, `cornerstone connector source-policy confirm`, `cornerstone connector direct-write-test`, `cornerstone connector credential status`, `cornerstone connector credential rotate`, `cornerstone connector credential revoke`, `cornerstone connector credential-boundary-test`, `cornerstone connector audit correlate`, `cornerstone connector github-write-guard`, `cornerstone connector github-failure simulate`, `cornerstone connector capture permission probe`, `cornerstone connector capture consent granted`, `cornerstone connector capture guard evaluate`, `cornerstone connector capture sessionize`, `cornerstone connector capture browser active-tab`, `cornerstone connector capture browser auto-config`, `cornerstone connector capture browser auto-capture`, `cornerstone connector capture browser sensitive-policy`, `cornerstone connector capture lifecycle seed`, `cornerstone connector capture lifecycle pause`, `cornerstone connector capture lifecycle sample-attempt`, `cornerstone connector capture lifecycle resume`, `cornerstone connector capture lifecycle revoke`, `cornerstone connector capture lifecycle retention`, `cornerstone connector capture lifecycle export`, `cornerstone connector capture lifecycle review-result`, `cornerstone connector capture lifecycle delete`, `cornerstone watch rule create`, `cornerstone watch rule activate`, `cornerstone watch rule evaluate`, `cornerstone watch rule pause`, `cornerstone watch rule resume`, `cornerstone watch rule edit`, `cornerstone watch rule show`, `cornerstone watch rule delete`, `cornerstone watch result build`, `cornerstone watch result correct`, `cornerstone watch result approve-memory`, `cornerstone watch result review`, `cornerstone connector evidence bundle create`, `cornerstone connector raw-access request`, `cornerstone connector raw-access read`, `cornerstone connector raw-access revoke`, `cornerstone connector raw-access export`, `cornerstone connector untrusted-content review`, `cornerstone agent prompt-authority-test`, `cornerstone memory quarantine-check`, `cornerstone egress test`, `cornerstone connector upgrade plan`, `cornerstone connector product-surface audit`, `cornerstone connector report-lint`, filtered `CS-CH-018` content-restriction delivery commands, filtered `CS-CH-019` write-guard commands, filtered `CS-CH-020` provider failure commands, filtered `CS-CH-021` capture guard commands, filtered `CS-CH-022` activity sessionization command, filtered `CS-CH-023` Watch Rule lifecycle commands, filtered `CS-CH-024` active-tab capture commands, filtered `CS-CH-025` auto-capture commands, filtered `CS-CH-026` sensitive-page policy commands, filtered `CS-CH-027` capture lifecycle commands, filtered `CS-CH-028` Watch Result commands, filtered `CS-CH-029` ActionCard/preflight commands, filtered `CS-CH-030` action safety commands, filtered `CS-CH-031` declared Action execution commands, filtered `CS-CH-032` bypass-denial commands, filtered `CS-CH-033` idempotent retry commands, filtered `CS-CH-034` scope-isolation commands, filtered `CS-CH-035` credential-custody commands, filtered `CS-CH-036` default-deny egress topology proof, and filtered `CS-CH-037` audit-correlation proof. |
| Durable state | VERIFIED LOCALLY | `tmp/scenario/connector-contract-adapter-*/connector/...` and `tmp/scenario/connector-contract-adapter-*/artifacts/...` generated by verifier. |
| Audit | VERIFIED LOCALLY | `cornerstone audit verify --state-dir ... --json`. |
| UI/API | NOT_VERIFIED | CS-CH-003 and CS-CH-006 verify data contracts only; CS-CH-028 verifies the Watch Result data contract only; CS-CH-039 verifies product-surface copy/navigation contract only. Rendered UI/API proof remains future work. |
| Human gates | HUMAN_REQUIRED | `CS-CH-H01` through `CS-CH-H07` each have a non-mutating preparation package via `cornerstone connector human-gate package --scenario <id> --json` with matching template readiness, Senior Review Perspectives, review order, dependencies, stop/reject condition, and blank proposed reviewer-record template including a typed evidence-packet manifest skeleton with required-evidence labels and allowed redaction statuses, embedded reviewer checklist for required fields, senior-review perspectives, evidence-packet manifest rows, dependency refs, and no-approval completion rule, scenario delivery-unit plan for senior-perspective research, implementation approach, smallest rehearsal, remediation/refactor, validation, documentation, and dependency-aware next-gate movement, optional blank reviewer-template output via `cornerstone connector human-gate package --scenario <id> --json --record-template-output <reviewer-record-template.json>`, proposed-record structural validation via `cornerstone connector human-gate validate-record --scenario <id> --record-file <json> --json` or `cornerstone connector human-gate validate-record --scenario <id> --record-file <json> --json --output <redacted-validation-envelope.json>` for required fields, redacted senior-review perspective findings, evidence-packet manifest coverage, required-evidence label matching, allowed redaction statuses, timestamp, dependency refs, decision values, and sensitive markers without persisting record body, raw path, decision value, finding text, or manifest values, package `record_template_output_command`, package `validation_output_command`, package `scenario_delivery_unit_plan`, rollup `record_template_output_command`, rollup `record_validation_output_command`, rollup `scenario_delivery_unit_plan`, next `next_record_template_output_command`, next `next_record_validation_output_command`, next `next_reviewer_checklist`, next `next_scenario_delivery_unit_plan`, and an optional redacted validation envelope with top-level `summary` and `final_verdict=HUMAN_REQUIRED`, H04-only `local_baseline_review_inputs` for current local VS2 and ConnectorHub dependency report refs/hashes/status summaries with `acceptance_sufficient=false`, `product_claim_allowed=false`, and `pass_claim_allowed=false`, H04 package/readiness `local_baseline_preflight_bundle` mirrors, a rollup via `cornerstone connector human-gate report --json` with execution queue, template-structure readiness counters, `senior_review_perspectives_ready_count`, `senior_review_perspective_findings_complete_count`, `evidence_packet_manifest_complete_count`, validation coverage summary fields, depends-on human-gate record validation readiness fields including `depends_on_human_gate_record_validation_status`, `dependency_unlock_allowed_by_validator`, and `depends_on_human_gates_missing_dependency_unlock_record_validation`, a next selector via `cornerstone connector human-gate next --json` for first uncompleted dependency-ready H gate plus blocked dependency reasons, H04-only next-gate local-baseline review inputs, first-class `next_local_baseline_preflight_bundle` mirror, required human delta, recommended preflight command counts, structured preflight command-plan rows with expected report paths, pinned next-selector artifact `reports/scenario/connectorhub-human-gate-next-2026-06-24.json`, a validation handoff via `cornerstone connector human-gate validation-handoff --json` with ordered `scenario_validation_handoff_rows`, reviewer commands, latest validation refs, redacted latest-validation issue summaries, H04 local-baseline summary counts, first-class handoff `local_baseline_preflight_bundle` mirror, H04 preflight command-plan rows, row-level `remaining_human_evidence_summary` objects with required fields/evidence/release-impact/stop-reject details, zero approval/PASS/product-claim counters, pinned validation-handoff artifact `reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json`, and a top-level CLI envelope `summary`/`final_verdict` mirror for operator scripting; structurally valid `REJECT` records remain validation evidence but do not unlock dependent H gates; templates from `docs/verification-reports/CONNECTOR_HUB_CS_CH_H01_HUMAN_REVIEW_TEMPLATE_2026-06-24.md` through `docs/verification-reports/CONNECTOR_HUB_CS_CH_H07_HUMAN_REVIEW_TEMPLATE_2026-06-24.md`; dated human evidence is still required before PASS. |
| Live provider | HUMAN_REQUIRED | `CS-CH-H01` for GitHub read-only and `CS-CH-H05` for a separately approved non-GitHub live Action. Human live evidence is still required before PASS. |
| Production | HUMAN_REQUIRED | `CS-CH-H04` and `CS-CH-H07`. |

H04 local-baseline comparison rows are report-row guarded in package, next-selector, and validation-handoff output: each individual baseline report row repeats `review_input_only=true`, `acceptance_sufficient=false`, `product_claim_allowed=false`, `pass_claim_allowed=false`, and `claim_boundary=h04_local_baseline_snapshot_is_review_input_not_human_acceptance`, so a successful local VS2 or ConnectorHub dependency report cannot be mistaken for human acceptance.

H04 preflight-bundle aliases are also script-friendly: `local_baseline_preflight_bundle` and `next_local_baseline_preflight_bundle` repeat `recommended_preflight_command_plan_schema_version`, `recommended_preflight_command_plan_count`, and `recommended_preflight_command_plan` while preserving `acceptance_sufficient=false`, `product_claim_allowed=false`, and `pass_claim_allowed=false`.

Human-gate package summaries are script-friendly: every package summary mirrors `package_id`, record-template command aliases, source-requirement metadata, remaining-evidence claim boundary, release impact, and stop/reject guidance without collecting approval, closing source requirements, or promoting any H row.

The H04 preflight-bundle top-level summary mirrors `preflight_bundle_report_id`, `operator_rule`, `local_baseline_review_inputs_schema_version`, `local_baseline_acceptance_sufficient`, `local_baseline_product_claim_allowed`, `local_baseline_pass_claim_allowed`, `required_human_delta`, `recommended_preflight_commands`, `recommended_preflight_command_count`, `recommended_preflight_command_plan_schema_version`, `recommended_preflight_command_plan_count`, and `recommended_preflight_command_plan` for script-friendly local comparison handoff without executing preflight commands, accepting production-like evidence, or changing H04 to PASS.

Each evidence-packet validation summary mirrors `evidence_packet_validation_report_id`, `operator_rule`, filename-only packet validation aliases, and schema-version refs without exposing packet contents, collecting approval, or changing any H row to PASS.

Each evidence-packet record-draft summary mirrors `evidence_packet_record_draft_report_id`, `operator_rule`, packet_validation_report_id, draft_record_output_written_by_runtime, draft_record_included_in_summary=false, raw/packet content persistence flags, and human_decision_recorded_by_evidence_packet_record_draft=false.

Each evidence-packet scaffold summary mirrors `evidence_packet_scaffold_report_id`, `operator_rule`, scaffold_template_file_names, scaffold_template_file_hashes, written_packet_file_names, written_packet_files, and template_contents_included_in_summary=false.

## Engineering Trail Maintenance

The ConnectorHub trail is maintained as a reproducible local evidence package, not an ad hoc collection of JSON files.

### Independent Scenario Delivery Loop

Every ConnectorHub `MUST_PASS` scenario is treated as an independent delivery unit. Do not start the next scenario's PASS claim until the current scenario has a closed local trail:

1. Freeze the scenario row: scenario id, type, phase, related requirements, proof surface, claim boundary, CLI command, expected JSON report path, and human-required exclusions.
2. Research from the five senior perspectives required by result docs: product/domain, architecture/data-contract, reliability/security, observability/performance/testability, and maintainability/migration.
3. Define the smallest implementation approach and explicitly name what will remain outside the PASS claim.
4. Implement the smallest complete behavior behind native `cornerstone ...` CLI paths, then refactor only inside the scenario's dependency boundary.
5. Verify the focused scenario with `cornerstone scenario verify connector-contract-adapter --scenario <CS-CH-###> --json --output <matrix evidence_required>` and immediately gate that report with `cornerstone scenario gate <report> --json`.
6. Document the result in `docs/verification-reports/CONNECTOR_HUB_<scenario>_RESULT_2026-06-23.md`, including the decision trail, lifecycle trail, evidence report path, proof boundary, negative evidence, and adoption contribution.
7. Refresh the aggregate report, scenario delivery-unit manifest, engineering-trail manifest, and engineering-trail verifier before treating the scenario as a completed delivery unit.

The per-scenario result document and JSON report are the acceptance record. The aggregate report is a rollup, not a substitute for the focused scenario proof.

`make verify-connector-contract-adapter` is the guarded local replay target for the 40 AI-owned scenarios: it must refresh the current VS2 local proof and VS2 scenario report before focused ConnectorHub replay, match the matrix-backed focused verify/gate pairs, then run the aggregate verify/gate pair and `python3 -m unittest tests.scenario.test_connectorhub_cli`.

### Package Refresh Sequence

1. Regenerate AI-owned per-scenario reports with filtered `cornerstone scenario verify connector-contract-adapter --scenario <CS-CH-###> --json --output <matrix evidence_required>` commands and gate each report with `cornerstone scenario gate <report> --json`.
2. Regenerate the aggregate report with `cornerstone scenario verify connector-contract-adapter --json --output reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json` and gate it.
3. Regenerate the pinned human-gate package/field-ref-contract/evidence-packet-contract/evidence-packet-file-contract/evidence-packet-scaffold/evidence-packet-validation/evidence-packet-record-draft/preflight-bundle/readiness/next/validation-handoff artifacts, scenario delivery-unit manifest, and engineering-trail manifest with `make generate-connectorhub-engineering-trail-manifest`.
4. Verify the trail with `make verify-connectorhub-engineering-trail`.

`make generate-connectorhub-engineering-trail-manifest` first refreshes the pinned human-gate package/field-ref-contract/evidence-packet-contract/evidence-packet-file-contract/evidence-packet-scaffold/evidence-packet-validation/evidence-packet-record-draft/preflight-bundle/readiness/next/validation-handoff artifacts through non-mutating `cornerstone connector human-gate ...` commands, then `scripts/generate_connectorhub_engineering_trail_manifest.py` imports the verifier's required-path list, writes `reports/scenario/connectorhub-scenario-delivery-unit-manifest-2026-06-24.json` for the 40 AI-owned delivery units, and writes hashes for the contract, matrix, index, verifier, generator, Connector runtime, CLI entrypoint, focused ConnectorHub CLI tests, README, result docs, per-scenario evidence reports, human templates, human preparation report, aggregate report, human-gate package/field-ref-contract/evidence-packet-contract/evidence-packet-file-contract/evidence-packet-scaffold/evidence-packet-validation/evidence-packet-record-draft/preflight-bundle/readiness/next/validation-handoff artifacts, and scenario delivery-unit manifest. The H04 field-ref-contract report exposes accepted evidence-ref shapes only, its summary mirrors `field_ref_contract_report_id` and `operator_rule`, and it records no submitted field values. The H04 evidence-packet-contract report exposes required evidence-packet manifest rows only, its summary mirrors `evidence_packet_contract_report_id` and `operator_rule`, and it records no submitted evidence refs. The H04 evidence-packet-file-contract report exposes required acceptance-packet filenames, content categories, and a non-executed scaffold plan with `packet_file_scaffold_command_count=9` only, its summary mirrors `evidence_packet_file_contract_report_id` and `operator_rule`, records no packet file contents, keeps `packet_file_scaffold_plan_executed_by_report=false`, and cannot accept H04. The H04 evidence-packet-scaffold dry-run report exposes eight blank-template hashes only, keeps `write_requested=false`, `write_executed=false`, `template_contents_included_in_report=false`, `packet_file_contents_read_by_scaffold=false`, `human_evidence_recorded_by_scaffold=false`, and `local_template_files_written_by_evidence_packet_scaffold=0`, and cannot accept H04. The H04 evidence-packet-validation placeholder report records `status=packet_not_submitted`, `missing_packet_file_count=8`, `packet_structurally_complete=false`, `raw_packet_file_contents_included_in_report=false`, `packet_file_contents_persisted_by_validator=false`, and `dependency_unlock_allowed_by_packet_validator=false`; even a structurally complete packet remains reviewer input only. The H04 evidence-packet-record-draft placeholder report records `status=packet_not_ready_for_record_draft`, `draft_record_available=false`, `draft_record_expected_validation_status_before_human_completion=record_structurally_invalid`, `raw_packet_file_contents_recorded_by_draft=false`, `packet_file_contents_persisted_by_draft=false`, and `dependency_unlock_allowed_by_record_draft=false`; a generated draft still cannot accept H04 until a dated human reviewer supplies decision, reviewer, timestamp, and senior-review findings. Duplicate evidence-ref validation stays fingerprint-only in issue summaries and never turns structural validation into H04 acceptance. This keeps manifest drift visible without changing any scenario PASS boundary.

### Done Criteria Before Moving To The Next Scenario

- Focused scenario JSON report: `status=success`, `scenario_count=1`, `pass=1`, `blocking=0`, and `scenario_filter` containing only the target scenario.
- Focused scenario gate: `status=success`, `blocking_count=0`.
- Result document: includes all eleven decision dimensions, the full lifecycle trail, proof boundary/out-of-scope, and no stale cross-scenario range wording.
- Scenario delivery-unit manifest: includes the row with `status=delivery_unit_closed`, exact closed AI and excluded human scenario ID sets, exact focused verify/gate commands, focused gate facts, machine-readable decision/research/lifecycle coverage facts, complete closure flags, focused report facts, and false product/live/human/production claim flags.
- Engineering trail: `make verify-connectorhub-engineering-trail` passes after manifest regeneration.
- Boundary discipline: live-provider, physical-device, rendered UI/API, human acceptance, and production readiness remain `HUMAN_REQUIRED` or `NOT_VERIFIED` unless their own proof surface has been collected.

## Scenario Result Policy

Unfiltered `connector-contract-adapter` verification is expected to pass the verified CH-0 local fixture rows plus `CS-CH-007`, `CS-CH-008`, `CS-CH-009`, `CS-CH-010`, `CS-CH-011`, `CS-CH-012`, `CS-CH-013`, `CS-CH-014`, `CS-CH-015`, `CS-CH-016`, `CS-CH-017`, `CS-CH-018`, `CS-CH-019`, `CS-CH-020`, `CS-CH-021`, `CS-CH-022`, `CS-CH-023`, `CS-CH-024`, `CS-CH-025`, `CS-CH-026`, `CS-CH-027`, `CS-CH-028`, `CS-CH-029`, `CS-CH-030`, `CS-CH-031`, `CS-CH-032`, `CS-CH-033`, `CS-CH-034`, `CS-CH-035`, `CS-CH-036`, `CS-CH-037`, and `CS-CH-040` report-lint proof. This does not claim live-provider readiness, physical-device readiness, rendered Watch Result UI readiness, human UX/privacy acceptance, destructive-action UX approval, production secret-backend custody, production RLS/topology, production network-control readiness, publishing approval, or production security readiness.

The independent delivery proof commands are:

```text
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-001 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-001.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-002 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-002.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-003 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-003.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-004 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-004.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-005 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-005.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-006 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-006.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-007 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-007.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-008 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-008.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-009 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-009.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-010 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-010.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-011 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-011.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-012 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-012.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-013 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-013.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-014 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-014.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-015 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-015.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-016 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-016.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-017 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-017.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-018 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-018.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-019 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-019.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-020 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-020.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-021 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-021.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-022 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-022.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-023 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-023.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-024 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-024.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-025 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-025.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-026 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-026.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-027 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-027.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-028 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-028.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-029 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-029.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-030 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-030.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-031 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-031.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-032 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-032.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-033 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-033.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-034 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-034.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-035 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-035.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-036.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-037 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-037.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-038 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-038.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-039 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-039.json
cornerstone scenario verify connector-contract-adapter --scenario CS-CH-040 --json --output reports/scenario/connector-contract-adapter/scenarios/CS-CH-040.json
cornerstone scenario verify connector-contract-adapter --json --output reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json
```

---
title: "Applying ConnectorHubKit to CornerStone"
subtitle: "Architecture, implementation mapping, and scenario guide for evidence-first connected sources"
author: "Prepared for JiYong / Tars"
date: "2026-06-22"
lang: en-US
---

> **Status:** Evidence-based investigation and implementation guide. This is not an implementation, live-provider, production-readiness, or release claim.
>
> **Core recommendation:** Adopt ConnectorHubKit as the internal Connector / Provider / Action protocol through a CornerStone-owned adapter. Preserve one product experience, archive every permitted input before interpretation, and require scenario-specific evidence before any live-provider claim.

# Contents

- [Executive summary](#executive-summary)
- [Scope, authority, and method](#scope-authority-and-method)
- [Part I — ConnectorHubKit deep dive](#part-i-connectorhubkit-deep-dive)
  - [Core concepts](#core-concepts)
  - [Contract lifecycle](#contract-lifecycle)
  - [WatchAgent and Chrome implementation lessons](#watchagent-and-chrome-implementation-lessons)
  - [Verification model](#verification-model)
  - [Strengths, limitations, and technical debt](#strengths-limitations-and-technical-debt)
- [Part II — Applying ConnectorHubKit to CornerStone](#part-ii-applying-connectorhubkit-to-cornerstone)
  - [Current state and gap analysis](#current-cornerstone-state-and-gap-analysis)
  - [Target integration architecture](#target-integration-architecture)
  - [GitHub read-only design](#github-read-only-design)
  - [WatchAgent adaptation](#watchagent-adaptation)
- [Part III — Scenario guide](#part-iii-scenario-guide)
  - [Scenario summary matrix](#scenario-summary-matrix)
  - [Detailed scenarios](#detailed-scenarios)
  - [Scenario implementation map](#scenario-implementation-map)
- [Proposed contract examples](#proposed-contract-examples)
- [Verification plane and evidence package](#verification-plane-and-evidence-package)
- [Human-required gates](#human-required-gates)
- [Decisions to freeze](#decisions-to-freeze-before-implementation)
- [Risks and mitigations](#risks-and-mitigations)
- [Recommended implementation sequence](#recommended-implementation-sequence)
- [Conclusion](#conclusion)
- [Source register](#source-register)
- [Glossary](#glossary)

# Executive summary

ConnectorHubKit is best understood as a **contract and safety substrate**, not merely a collection of API clients. An application declares what it needs, the Hub resolves provider access and Source Policy, and the application receives normalized Projections and declared Action results without receiving provider credentials, raw provider clients, or unbounded provider payloads. This model aligns strongly with CornerStone's one-product/three-engine architecture. [CH-README] [CH-CONTRACT] [CS-ARCH]

The practical integration strategy is **adapter-first and read-only-first**. CornerStone should not merge repositories or import provider SDKs into the Product/Mission engine. It should implement a versioned `connectorhub_client` port that persists Setup Results, validates Projection Envelopes, converts permitted inputs into immutable Artifacts, acknowledges deliveries only after durable commit, and correlates connector evidence and audit with CornerStone Evidence Bundles and Workflows.

The current maturity is useful but bounded. ConnectorHubKit has strong local deterministic contracts, fixtures, redaction tests, retry/quarantine, Action idempotency, and Chrome privacy enforcement. It remains a local/in-process release candidate with unverified live-provider E2E and no production multi-user control plane. The inspected Cornerstone source contains architecture documentation for a ConnectorHub client, but no implemented adapter; its current action path remains local/mock. [CH-RELEASE] [CH-LIMITS] [CS-CURRENT] [CS-ARCH]

## Key findings

1. **Adopt the protocol, not the repository.** A repository merge or provider SDK imports in Product code would defeat ConnectorHubKit's main portability and safety benefits.
2. **Keep one product experience.** Connected Sources, Capture Inbox, Watch Results, Evidence, and Actions belong inside CornerStone. Setup Result, Source Policy, permissions, retries, and quarantine are progressively disclosed operator surfaces.
3. **Archive before interpretation.** A delivered Projection should become a scoped immutable Artifact and ConnectorDeliveryReceipt before Watch logic, briefs, claims, memory, or missions use it.
4. **Separate provider feasibility from product authority.** ConnectorHub preflight answers whether a declared provider action is supported and feasible. CornerStone still owns evidence sufficiency, policy, workspace/mission authority, approval, and WorkflowRun state.
5. **Start read-only.** The first GitHub integration should add provider-neutral source-control read capabilities with no Action declarations and explicit zero-write regression tests.
6. **Treat WatchAgent as two concerns internally.** Collection adapters belong to ConnectorHub; Watch Rules, Observed/Inferred/Proposed results, approvals, memory candidates, and history belong to CornerStone Product/Mission.
7. **Do not overclaim maturity.** ConnectorHubKit is a strong local contract and verification prototype, not a hosted production connector service. CornerStone also has no real `connectorhub_client` implementation in the inspected source.
8. **Make VS2 a release gate.** Live connected-source release should wait for scenario-specific RequestContext, persistence/RLS, OPA, egress, backup/restore, and regression evidence. The latest VS2 report rejects readiness. [CS-VS2]

# Scope, authority, and method

## Frozen investigation contract

| Element | Frozen definition |
|---|---|
| Goal | Explain ConnectorHubKit concepts, architecture, implementation details, and practical CornerStone adaptation, with scenarios as the primary guide. |
| Constraints | Read-only repository investigation; no code or provider mutations; distinguish source-verified behavior from documented targets and proposals; GitHub remains read-only. |
| Out of scope | Implementing the adapter, running live provider tests, changing credentials, modifying repositories, approving production security, or publishing a release. |
| Verification | Source-path review at identified revisions, implementation/test triangulation, contract-to-CornerStone mapping, and rendered-document visual QA. |

## Authority and status discipline

The current CornerStone product SoT and MUST-PASS scenario standard in the Cornerstone repository are treated as product and acceptance authority. The user-provided operating constitution and scenario-first instruction reinforce the same evidence and boundary rules. The older uploaded `project-sot.md` is retained as historical context only because current repository authority supersedes its “only SoT” framing where conflicts exist. [CS-SOT] [CS-SCEN] [USER-CONSTITUTION] [USER-SCENARIO] [HIST-SOT]

This guide uses four status labels:

- **Source-verified local behavior:** visible in current source, tests, or local reports.
- **Documented target:** required by current CornerStone authority but not verified in implementation.
- **Proposed adaptation:** architecture or scenario recommended by this investigation.
- **Human required:** needs credentials, a physical device, external state, production environment, or subjective acceptance.

# Part I — ConnectorHubKit deep dive {#part-i-connectorhubkit-deep-dive}

# What ConnectorHubKit is

ConnectorHubKit implements the promise “build apps without building connectors” through a capability-oriented protocol. Applications own product purpose, handlers, approvals, memory, and domain workflows. The Hub owns provider access, credentials, provider mappings, Source Policy, normalized Projections, Delivery, declared Action execution, evidence metadata, raw-access control, connector audit, retry, quarantine, SDKs, generated integration workspaces, and a trust-focused Control UI. [CH-README] [CH-CONTRACT]

It is **not** a complete CornerStone product, evidence archive, Claim engine, memory system, mission orchestrator, or production security plane. Its internal contract explicitly says it does not replace CornerStone Artifact, Claim, Evidence Bundle, Policy, Workflow, or Audit Ledger systems.

![One product, three internal engines](/mnt/data/connectorhub_guide_assets/one_product_three_engines.png){width=95%}

# Core concepts

| ConnectorHubKit concept | Meaning | CornerStone interpretation |
|---|---|---|
| **App Requirements** | App-owned declaration of needed capabilities, accepted Projections, data limits, Delivery behavior, and allowed Actions. | Feature/mission connector contract, versioned and owner-scoped. |
| **Setup Result** | Hub-generated readiness and gap analysis with provider mappings, Source Policy, streams, action constraints, warnings, and verification. | `ConnectorSetupResult` used for activation, admin UX, policy input, and evidence. |
| **Common Capability** | Provider-neutral verb such as message read or activity observe. | Stable port preventing Product code from binding to provider APIs. |
| **Provider Pack** | Manifest, auth model, provider mappings, Projection schemas, Action support, risk defaults, and fixtures. | Connector implementation plug-in owned by the Connector / Provider engine. |
| **Source Policy** | Selected providers/resources, body and metadata restrictions, fallback, exclusions, and raw-access policy. | Effective policy intersecting organization rules, owner confirmation, product need, and provider support. |
| **Projection Envelope** | Versioned normalized payload plus app scope, capability, source summary, and EvidenceRef metadata. | Artifact/derived-representation input; never automatically product truth. |
| **Delivery** | App-scoped queue item with idempotency, ack deadline, attempts, and lifecycle state. | `ConnectorDeliveryReceipt` plus durable ingest outbox/ack protocol. |
| **Action Preflight** | Dry-run provider-boundary check for declaration, provider support, permission, Source Policy, risk, input, and idempotency. | One input to CornerStone ActionCard dry-run; not product approval. |
| **Action Result** | Provider execution result linked to request and idempotency. | Workflow execution evidence and mission-outcome input. |
| **EvidenceRef** | Safe provider-side provenance metadata and access policy without raw payload by default. | Evidence Bundle item linked to an immutable Artifact and connector receipt. |
| **Temporary Raw Access** | Exceptional, purpose-bound, time/read-limited Hub-controlled access. | Policy event; default deny; usually unnecessary in normal operation. |
| **Retry / Quarantine** | Operational handling for transient, poison, malformed, or permanently unavailable inputs. | Operator-visible failure evidence and replay workflow. |
| **Generated Workspace / SDK** | Generated connector glue plus preserved developer-owned handlers behind a Hub facade. | Adapter-generation pattern; Product handlers remain provider-free. |
| **Control UI** | Trust-focused local view of setup, permissions, Source Policy, deliveries, actions, evidence, and audit. | Admin/operator context inside CornerStone, not default daily navigation. |

# Contract lifecycle

![ConnectorHubKit contract lifecycle](/mnt/data/connectorhub_guide_assets/connectorhub_lifecycle.png){width=98%}

The lifecycle is deliberately split into setup-time and runtime contracts. Setup-time contracts answer whether the application can be safely activated; runtime contracts move normalized data and declared Actions. This prevents a product feature from discovering missing permissions only after it has produced incomplete or unsafe behavior.

## App Requirements

App Requirements are the app-owned declaration of truth about connector needs. They include common capabilities, whether each need is required, accepted Projection types, data handling, Delivery acknowledgement, source preferences, requested provider extensions, raw access, and declared Actions. The runtime validates unknown capabilities, unsupported Projections, undeclared extensions, unsafe Action declarations, and incompatible Delivery settings before setup. [CH-SCHEMAS] [CH-TEST02] [CH-TEST04]

## Setup Result

The Hub resolves requirements into `ready`, `ready_with_gaps`, or `blocked`. The result includes provider mappings, selected resources, missing permissions or credentials, Source Policy, Delivery streams, Projection schemas, allowed Actions, constraints, warnings, plain-language explanation, and verification commands. It is both operational state and reusable evidence of what was available, selected, missing, denied, or deferred. [CH-RUNTIME]

## Common capabilities and Provider Packs

The capability catalogue is the anti-coupling mechanism. Product code asks for a stable capability; Provider Packs map it to provider APIs, schemas, permissions, risks, and fixtures. The current catalogue covers activity, communication, knowledge documents, evidence metadata, and action results. It does **not** contain a source-control family, so GitHub support should add new provider-neutral capabilities rather than overload document or message semantics. [CH-CAPS] [CH-PACKS]

## Projection and Delivery

A Projection Envelope is a versioned normalized object with app scope, common capability, Projection type, source summary, permitted payload, and EvidenceRef metadata. A Delivery adds stream identity, state, idempotency, attempts, and acknowledgement timing. The local runtime enforces app-scoped polling, ack/fail/retry, duplicate handling, and quarantine. [CH-SCHEMAS] [CH-RUNTIME] [CH-TEST02]

## Actions and preflight

Action Preflight is a provider-boundary dry-run. It verifies that the Action is declared, the selected Provider Pack supports it, permissions are available, input is valid, and side-effecting requests have an idempotency key. It returns risk, predicted side effect, provider and permission summaries, Source Policy, denial reasons, and an app-owned approval requirement. Connector permission is explicitly not CornerStone approval. [CH-RUNTIME] [CH-CONTRACT]

![Action safety handoff](/mnt/data/connectorhub_guide_assets/action_safety_handoff.png){width=95%}

## Evidence and Temporary Raw Access

EvidenceRef is intentionally metadata-first. It identifies provider source, data class, restrictions, and access requirements without exposing raw payloads. Temporary Raw Access is denied unless declared and permitted; local tests exercise TTL, maximum reads, expiry, and redaction of access handles. The known-limitations document correctly warns that live-provider raw-access semantics remain unverified. [CH-TEST02] [CH-LIMITS]

## Retry, quarantine, and operational failures

The runtime distinguishes transient failure, permanent setup gaps, malformed provider payloads, and poison deliveries. It records retries and quarantines with audit events while applying redaction. This is useful for CornerStone because connector failure becomes visible evidence and freshness state rather than silent data loss. [CH-TEST04]

## Generated workspace and SDK boundary

Generated workspaces separate regenerated schemas, bindings, and verification from developer-owned handlers. The Python SDK exposes a Hub facade rather than provider clients. Tests also scan proof apps for direct provider/API dependencies. CornerStone can reuse this pattern for adapters and solution packs while keeping the native `cornerstone ...` CLI as the operator surface. [CH-WORKSPACE] [CH-SDK] [CH-TEST02] [CS-CLI]

## Control UI

The Control UI is a trust surface for setup gaps, provider permissions, Source Policy, deliveries, actions, evidence, retries, quarantine, and audit. In CornerStone it should become an admin/operator context, not the default landing screen and not a separately branded product. [CH-CONTROL] [CS-SOT]

# WatchAgent and Chrome implementation lessons

WatchAgent is the richest proof of how ConnectorHubKit can support continuous personal work context. The local runtime can collect configured macOS activity and selected message sources, build Projections, and render Watch/Review objects. The Chrome extension uses Manifest V3, `activeTab`, runtime optional host permissions, a local backend endpoint, bounded page clips and hashes, explicit consent, pause/resume, diagnostics, and a capture timeline. [CH-WATCH] [CH-CHROME] [OFF-CHROME]

The strongest implementation idea is **two-sided policy enforcement**. The extension performs a user-facing preflight and sends a strict versioned payload, but the backend independently validates schema, consent ID, configuration version, trigger, domain approval, pause state, idempotency, throttling, sensitive signals, and policy restrictions before creating evidence. Client checks improve usability; server checks remain authoritative.

![Watch capture flow](/mnt/data/connectorhub_guide_assets/watch_capture_flow.png){width=98%}

The main adaptation boundary is that capture and provider access stay in ConnectorHub, while WatchRule, WatchResult, approvals, history, memory candidates, and mission candidates become CornerStone Product/Mission objects. The proof app currently co-locates both concerns for demonstration; CornerStone should not preserve that repository-level coupling.

# Verification model

ConnectorHubKit has a strong local verification posture: strict schema fixtures, Provider Pack fixtures, app isolation, ack/fail/retry/quarantine, Action idempotency, undeclared-Action denial, raw-access denial and expiry, redaction scans, and proof-app bypass scans. `make verify-v1.0-rc1` aggregates contract, Python, and TypeScript tests. [CH-TEST02] [CH-TEST04] [CH-RELEASE]

The release report is appropriately conservative: provider safety and UI trust are only partially verified, fresh-developer success is not checked, and real provider E2E, remote CI, publishing, tags, and human sign-off remain external. The local fixture plane is credible evidence for contracts, not evidence for production readiness.

# Strengths, limitations, and technical debt

## Strengths

- Contract-first product boundary: requirements, setup, Projection, Delivery, Action, and evidence objects are explicit and schema-validated.
- Provider-neutral capability mapping reduces product/provider coupling and permits substitution or fallback.
- Deterministic fixtures test negative evidence such as secret leakage, undeclared Actions, duplicate Delivery, and raw-access denial.
- Delivery acknowledgement, retry, quarantine, idempotency, and provider failures are first-class behavior rather than hidden SDK concerns.
- The Chrome path demonstrates layered privacy: scoped extension permissions, bounded payloads, and authoritative backend revalidation.
- Generated workspaces preserve developer-owned handlers while regenerating connector glue.

## Limitations and debt

- The core `LocalHub` runtime is local and in-process; it is not a horizontally scalable, hosted, multi-user service.
- State is primarily local JSON/JSONL and fixture-oriented. Postgres, production tenancy, RLS, OPA, and tamper-evident cross-engine audit are not provided by the internal stabilization slice.
- Real provider end-to-end behavior is explicitly unverified without authorized credentials, permissions, and hardware.
- The Control UI is a static/local trust surface, not a production CornerStone admin application.
- The source-control/GitHub capability family is absent and must be designed rather than approximated through document/message capabilities.
- Later runtime behavior is attached through monkey-patching. That is workable for a prototype but should become explicit services/interfaces before CornerStone adoption.
- Proof-app Watch logic and connector runtime coexist in the Connector-Hub repository. CornerStone should split collection from product meaning.

> **Practical conclusion:** Treat ConnectorHubKit as a reference implementation and internal protocol source. Port the contracts, negative tests, generated-handler pattern, and Watch privacy model through a clean adapter. Do not inherit local JSON persistence, static UI, proof-app ownership boundaries, or runtime monkey-patching as production architecture.

# Part II — Applying ConnectorHubKit to CornerStone {#part-ii-applying-connectorhubkit-to-cornerstone}

# Current CornerStone state and gap analysis

The CornerStone SoT already defines ConnectorHub as the Connector / Provider / Action engine, and the implementation roadmap names a ConnectorHub boundary milestone. The architecture document proposes `connectorhub_client.adapters.connectorhubkit`. However, source search at the inspected revision found `connectorhub_client` only in architecture documentation, not as an implemented package or runtime adapter. Current Cornerstone action acceptance remains local/mock and reports zero real external HTTP calls. [CS-SOT] [CS-ROADMAP] [CS-ARCH] [CS-CURRENT]

The most important prerequisite gap is VS2. The latest final report rejects local readiness after finding that prior generated wrappers did not execute each scenario's Given/When/Then behavior. It records 86 AI-verifiable rows as not verified and lists missing trusted RequestContext, real Postgres/RLS paths, real OPA enforcement, default-deny network topology, backup/restore, and fresh regressions. Live connector release must not bypass that gate. [CS-VS2]

## SoT versus repository reality

| Area | Current target | Inspected implementation reality | Consequence |
|---|---|---|---|
| Connector boundary | ConnectorHub-mediated access only. | Architecture docs and local mock behavior; no `connectorhub_client` implementation found. | Build adapter before live providers. |
| Archive handoff | Original/evidence preserved before interpretation. | Local Artifact/Evidence primitives; no connector Delivery mapping. | Implement durable ingress plus ack protocol. |
| Policy/tenancy/egress | Trusted scope, RLS/OPA, default deny. | Latest VS2 report rejects readiness. | Treat as blocking release gate. |
| GitHub | Read-only by user direction. | No source-control capability or Provider Pack. | Design a provider-neutral family. |
| WatchAgent | Capture supports CornerStone value and memory. | Strong proof app/extension; product state remains local proof-app state. | Move meaning and durable records to CornerStone. |
| Actions | ActionCard → policy/approval → WorkflowRun → ConnectorHub. | Preflight and ActionCard exist separately. | Integrate contracts; keep GitHub writes absent. |

# Target integration architecture

The recommended architecture is a **hexagonal adapter boundary**. The Product and Archive engines depend on a CornerStone-owned `ConnectorPort`, not on provider SDKs or ConnectorHubKit implementation classes.

```text
CornerStone Product / Workflow / Archive
        |
        | ConnectorPort (CornerStone-owned contracts)
        v
connectorhub_client adapter
        |
        | App Requirements / Setup Result / Delivery / Preflight / Result
        v
ConnectorHubKit or future ConnectorHub service
        |
        v
Provider Packs and provider APIs / local platform permissions
```

## Recommended components

- **ConnectorPort:** CornerStone-owned interface for requirements, setup, Delivery polling/streaming, ack/fail, preflight, execution, evidence metadata, and health.
- **ConnectorContractRegistry:** Pins supported contract/Projection versions, capability catalogue, and migration rules.
- **ConnectorApplicationService:** Binds an application to tenant, owner, namespace, workspace, purpose, and Source Policy.
- **ConnectorIngressWorker:** Validates envelope, archives Artifact, writes receipt/outbox, then acknowledges.
- **ConnectorDeliveryReceipt:** Durable idempotency and lineage record linking Delivery, provider event, Artifact, and audit refs.
- **ConnectorPolicyBridge:** Combines Setup Result/Source Policy with RequestContext, classification, workspace mode, and OPA.
- **ConnectorActionBridge:** Attaches provider preflight/result to ActionCard and WorkflowRun without transferring approval.
- **ConnectorAuditBridge:** Correlates safe connector events with the CornerStone tamper-evident audit ledger.
- **ConnectedSourcesAdmin:** Admin context for setup, permissions, Source Policy, health, retry/quarantine, and raw-access decisions.

# Contract and object mapping

| ConnectorHubKit | CornerStone record | Mapping rule |
|---|---|---|
| App Requirements | `ConnectorCapabilityContract` | Product/Mission owns the declared need; Connector engine validates and resolves it. |
| Setup Result | `ConnectorSetupResult` plus policy/evidence refs | Persist as scoped product state and show in admin context. |
| Source Policy | `EffectiveSourcePolicy` plus `PolicyDecision` | Intersect with CornerStone workspace/organization policy. |
| Projection Envelope | `ConnectorIngressEnvelope` | Validate, archive, then create derived representation. |
| Delivery | `ConnectorDeliveryReceipt` plus Outbox | Ack only after durable Artifact commit. |
| EvidenceRef | `ConnectorEvidenceRef` linked to EvidenceItem | Metadata alone is not the immutable original. |
| Action Preflight | `ConnectorPreflight` attached to ActionCard | Provider feasibility; CornerStone still owns policy/approval. |
| Action Result | `ActionResultArtifact` plus WorkflowRun result | Re-ingest as evidence and mission experience. |
| Connector audit | `AuditCorrelation` / external audit ref | Mirror safe metadata into the tamper-evident product ledger. |
| Temporary Raw Access | `RawAccessRequest` / `RawAccessDecision` | Default deny; store decision metadata only. |
| Watch proof-app objects | `WatchRule`, `WatchResult`, `MemoryCandidate`, `Approval`, `MissionCandidate` | Move into Product/Mission and link to immutable evidence. |

## Durable acknowledgement rule

> **Acknowledge a Delivery only after** the permitted original or canonical snapshot, normalized representation, scope, provenance, ConnectorDeliveryReceipt, and audit intent are durably committed. Use an outbox or equivalent so post-commit/pre-ack replay is safe.

## Effective Source Policy

```text
Effective Source Policy =
  organization / tenant policy
  ∩ owner confirmation and selected resources
  ∩ workspace classification and retention
  ∩ App Requirements
  ∩ Provider Pack capability and permission
  ∩ runtime egress policy
```

## Trust states

A connector Projection is source evidence, not automatically a Claim, memory, ontology object, or rule. Artifacts and Evidence Bundles support Draft → Evidence-backed → Approved promotion. Watch inferences remain Draft with confidence and caveats until accepted.

# Native CLI, API, and UI application

CornerStone's no-CLI-no-feature-PASS rule means connector features need native commands that use the same adapter, policy, evidence, workflow, and audit paths as API and UI.

```text
cornerstone connector requirements validate/register
cornerstone connector setup show
cornerstone connector source list/status
cornerstone connector source-policy review/confirm/override
cornerstone connector sync run/reconcile
cornerstone connector delivery show/retry/quarantine/replay
cornerstone connector action preflight
cornerstone watch rule create/test/activate/pause
cornerstone watch source enable/disable/status
cornerstone capture inbox list/show/save/dismiss/delete
cornerstone capture browser approve-domain/pause/resume/timeline
```

Daily-user UI should expose **Connected Sources**, **Capture Inbox**, **Watch Results**, source-backed **Evidence**, and ordinary Brief/Claim/Mission/Action flows. Provider mappings, permissions, Source Policy, retry/quarantine, and raw-access controls belong in admin/operator context.

# GitHub read-only design

## Capability family

Add provider-neutral source-control read capabilities rather than a GitHub-specific Product interface. A future GitLab or on-prem Provider Pack can map the same family; GitHub-only fields belong in a declared, versioned extension.

```yaml
needs:
  - common_capability: source_control.repository.read
    required: true
    accepted_projection_types: [source_control.repository.v1]
  - common_capability: source_control.change.read
    required: true
    accepted_projection_types:
      - source_control.commit.v1
      - source_control.change.v1
  - common_capability: source_control.issue.read
    required: false
    accepted_projection_types: [source_control.issue.v1]
  - common_capability: source_control.file.read
    required: false
    accepted_projection_types: [source_control.file_snapshot.v1]
actions: []
raw_access:
  temporary_raw_access: denied
```

## Permission and resource posture

Use a GitHub App with the minimum read permissions needed for selected repositories. GitHub's current documentation states that apps have no permissions by default and recommends selecting the minimum required permissions. The Setup Result should enumerate requested read permissions and selected repositories; any write permission should block the read-only release profile. [OFF-GH]

## Projection types

- **`source_control.repository.v1`:** repository identity, default branch, visibility/classification, selected refs, and update time.
- **`source_control.commit.v1`:** commit ID, parents, author metadata, message restriction, and changed-path summary.
- **`source_control.change.v1`:** pull/merge request identity, state, refs, participants, review status, restriction, and links.
- **`source_control.issue.v1`:** issue identity, state, labels as metadata, participants, restriction, and links.
- **`source_control.file_snapshot.v1`:** repository/ref/path, content hash, allowed content or derived text, size/mime, and source URL hash.

## Absolute write prohibition

> **Read-only release invariant:** The initial GitHub Provider Pack declares no Actions. Verification must show zero comment, issue, label, merge, review, push, branch, release, workflow, or settings mutations; zero write-capable Git credentials; and zero HTTP mutation calls.

# WatchAgent adaptation

```text
explicit consent and Source Policy
→ ConnectorHub capture and Projection
→ immutable Artifact and connector receipt
→ Watch Rule evaluation
→ Watch Result: Observed / Inferred / Proposed
→ evidence-backed Brief, Claim, MemoryCandidate, or MissionCandidate
→ user correction / acceptance / dismissal
→ owner-scoped learning and audit
```

The Product/Mission engine should own WatchRule, WatchResult, ApprovalRequest, MemoryCandidate, MissionCandidate, result history, corrections, and review UX. The Connector engine should own macOS/Chrome APIs, consent/permission evidence, capture policy, Projection, Delivery, retries, and provider audit.

Privacy defaults should stay conservative: off by default, local-only in the first release, no screenshots, keystrokes, clipboard, cookies, browser history, debugger access, form values, or broad all-site permission. Chrome auto capture requires a domain allowlist, optional host permission, matching backend consent/config version, pause state, idempotency, and throttling.

# Phased implementation roadmap

| Phase | Scope | Exit evidence |
|---|---|---|
| **Phase 0 — Contract freeze and gap baseline** | Freeze the ConnectorHub-to-CornerStone task contract, capability catalogue extension, scope model, error/reason catalogue, and exact proof fixtures. Do not code live providers. | Scenario contract, architecture decision, source-control capability proposal, current-state report. |
| **Phase 1 — Adapter and deterministic ingress** | Implement `connectorhub_client` port, fixture transport, Setup Result persistence, Projection validation, durable Artifact conversion, receipt/outbox, and ack/retry/quarantine. | Read-only local fixture reaches Artifact, Search, Evidence Bundle, and audit through native CLI/API/UI. |
| **Phase 2 — GitHub read-only Provider Pack** | Add source-control capability family, selected-repository GitHub App permissions, Projections, polling/webhook reconciliation, rate limits, and zero-write guards. | Selected repo → immutable evidence → brief/claim; all writes denied with zero provider mutations. |
| **Phase 3 — WatchAgent collection boundary** | Integrate macOS activity and Chrome summary-only capture through the adapter; move WatchRule/WatchResult/history into CornerStone. | Consent → capture → Artifact → Watch Result → correction/save/mission, with pause/revoke and privacy controls. |
| **Phase 4 — Governed Action handoff** | Join ActionCard dry-run with ConnectorHub preflight, execution envelope, result ingestion, and audit correlation. Keep GitHub read-only. | Mock provider Action E2E with evidence/policy/approval; live provider remains human-gated. |
| **Phase 5 — Production/security readiness** | Complete scenario-specific RequestContext, RLS, OPA, egress topology, backup/restore, real provider tests, human UX, migration, and operations. | No production claim until every AI-verifiable security/regression row has scenario-specific evidence. |

> **Smallest safe first slice:** One deterministic read-only fixture must traverse App Requirements → Setup Result → Delivery → immutable Artifact → Search → Evidence Bundle → ack/audit, with retry/quarantine and zero secret leakage. Add live providers only after this slice passes scenario-specific verification.

# Part III — Scenario guide {#part-iii-scenario-guide}

# How to use this guide

These scenarios are acceptance criteria, not implementation tasks. Freeze the subset for each delivery batch, keep contract rows status-neutral, and record PASS, FAIL, NOT_VERIFIED, NOT_RUN, or HUMAN_REQUIRED only in generated verification reports. Every Product feature also needs native CLI parity. [USER-SCENARIO] [CS-SCEN]

- “ConnectorHubKit locally implements this behavior” does not mean CornerStone integrates or verifies it.
- Live GitHub, real macOS permissions, real Chrome UX, production tenancy/egress, and external Action execution remain environment or human gated until run.
- Negative evidence is mandatory for safety scenarios: zero unauthorized calls, writes, egress, secret leaks, and cross-namespace use.

# Scenario summary matrix

## A. Setup and capability contracts

| ID | Type | Scenario | Current posture |
|---|---|---|---|
| CS-CH-001 | MUST_PASS | Register a connector-backed CornerStone capability. | Locally implemented in ConnectorHubKit; CornerStone adapter absent. |
| CS-CH-002 | MUST_PASS | Required capability missing blocks activation. | Locally tested; CornerStone proposed. |
| CS-CH-003 | MUST_PASS | Optional capability degrades gracefully. | Locally tested; CornerStone proposed. |
| CS-CH-004 | MUST_PASS | Owner confirms or overrides Source Policy. | Local helpers exist; durable CornerStone policy proposed. |
| CS-CH-005 | REGRESSION_GUARD | Swap providers without changing Product logic. | Architecture/fixtures exist; CornerStone adapter absent. |
| CS-CH-006 | MUST_PASS | Explain credential and permission gaps without secrets. | Locally tested; CornerStone UI/adapter proposed. |

## B. Ingestion, Delivery, and evidence

| ID | Type | Scenario | Current posture |
|---|---|---|---|
| CS-CH-007 | MUST_PASS | Convert a Projection into an immutable Artifact. | Delivery exists; CornerStone durable adapter absent. |
| CS-CH-008 | MUST_PASS | Acknowledge only after durable archive commit. | Hub ack/retry exists; atomic product integration proposed. |
| CS-CH-009 | MUST_PASS | Retry transient failures and quarantine poison deliveries. | Locally tested; Product operations proposed. |
| CS-CH-010 | MUST_PASS | Deduplicate provider events and version changed content. | Hub dedupe exists; CornerStone lineage mapping proposed. |
| CS-CH-011 | MUST_PASS | Enforce field/body restrictions from Source Policy. | Locally tested; defense-in-depth Product validator proposed. |
| CS-CH-012 | MUST_PASS | Promote EvidenceRef metadata into an Evidence Bundle. | Contract documented; Product implementation proposed. |
| CS-CH-013 | MUST_PASS | Temporary raw access is denied and tightly bounded. | Local behavior tested; live semantics unverified. |
| CS-CH-014 | REGRESSION_GUARD | Untrusted connector content cannot direct agents/actions. | Required target; integrated path not verified. |

## C. GitHub read-only source

| ID | Type | Scenario | Current posture |
|---|---|---|---|
| CS-CH-015 | MUST_PASS | Connect only explicitly selected repositories. | New capability family and Provider Pack proposed. |
| CS-CH-016 | MUST_PASS | Ingest repo, commit, PR, issue, and file snapshot Projections. | Proposed; no source-control family exists. |
| CS-CH-017 | MUST_PASS | Incremental sync via webhooks/polling is idempotent. | Proposed using Hub Delivery semantics. |
| CS-CH-018 | MUST_PASS | Apply repository restrictions and secret hygiene. | Proposed. |
| CS-CH-019 | REGRESSION_GUARD | Deny every GitHub write path. | Required by user scope; not yet implemented. |
| CS-CH-020 | MUST_PASS | Handle rate limits, revoked permissions, and repo removal. | Generic failure simulation exists; GitHub proposed. |

## D. WatchAgent, macOS, and Chrome

| ID | Type | Scenario | Current posture |
|---|---|---|---|
| CS-CH-021 | MUST_PASS | macOS capture is off until explicit consent/permission. | Local paths exist; physical-device acceptance required. |
| CS-CH-022 | MUST_PASS | Turn samples into bounded activity sessions. | Local Projections exist; durable Product mapping proposed. |
| CS-CH-023 | MUST_PASS | Create an explicit owner-scoped Watch Rule. | Proof object exists; Product lifecycle proposed. |
| CS-CH-024 | MUST_PASS | Explicit Chrome active-tab capture. | Substantial local implementation exists; integration proposed. |
| CS-CH-025 | MUST_PASS | Allowlist-based Chrome auto capture with two-sided consent. | Locally implemented; human browser review required. |
| CS-CH-026 | MUST_PASS | Block or degrade sensitive page capture. | Backend/client policy exists; Product mapping proposed. |
| CS-CH-027 | MUST_PASS | Pause, revoke, retain, export, and delete eligible state. | Partial proof behavior; Product governance proposed. |
| CS-CH-028 | MUST_PASS | Separate observation, inference, and proposal. | Analogous proof objects exist; Product ownership proposed. |

## E. Governed Actions and outcomes

| ID | Type | Scenario | Current posture |
|---|---|---|---|
| CS-CH-029 | MUST_PASS | Combine ActionCard dry-run with ConnectorHub preflight. | Both exist separately; integration proposed. |
| CS-CH-030 | MUST_PASS | Require evidence, policy, and authorized approval. | Documented target; real VS2 paths unverified. |
| CS-CH-031 | MUST_PASS | Execute declared Action and re-ingest outcome. | Local fixture Action exists; live path human-gated. |
| CS-CH-032 | REGRESSION_GUARD | Deny undeclared Actions and direct provider bypass. | Locally tested in Hub; integrated guard proposed. |
| CS-CH-033 | MUST_PASS | Make retries idempotent and expose compensation. | Local duplicate-key behavior exists; production semantics proposed. |

## F. Namespace, security, operations, and evolution

| ID | Type | Scenario | Current posture |
|---|---|---|---|
| CS-CH-034 | MUST_PASS | Bind every connector app, Delivery, and Watch to owner/namespace. | Local app scoping exists; production tenancy blocked by VS2 gaps. |
| CS-CH-035 | REGRESSION_GUARD | Keep credentials exclusively inside ConnectorHub. | Local vault/redaction tested; production backend is a target. |
| CS-CH-036 | MUST_PASS | Enforce default-deny egress around ConnectorHub/tools. | Required but not verified in current CornerStone. |
| CS-CH-037 | MUST_PASS | Correlate connector audit with CornerStone audit. | Local concepts exist; production integration proposed. |
| CS-CH-038 | MUST_PASS | Version contracts, pin Provider Packs, and migrate safely. | Schema versions exist; migration automation limited. |
| CS-CH-039 | REGRESSION_GUARD | Present one CornerStone product, not a ConnectorHub sub-product. | Required by SoT; integrated UX absent. |
| CS-CH-040 | REGRESSION_GUARD | Separate fixture proof from live/production claims. | ConnectorHub models distinction; CornerStone history shows need. |

# Detailed scenarios

## CS-CH-001 — Register a connector-backed CornerStone capability

**Type:** MUST_PASS
**Current posture:** Source-verified in ConnectorHubKit; CornerStone adapter not found.
**Primary use case:** A CornerStone workspace enables a connected-source feature without importing a provider SDK into Product code.

**Preconditions**

- An authenticated owner is operating inside one tenant, namespace, and workspace.
- A versioned `ConnectorCapabilityContract` declares purpose, required and optional Common Capabilities, accepted Projection types, data restrictions, Delivery behavior, and any declared Actions.
- The ConnectorHub adapter supports the requested contract version.

**Workflow**

1. CornerStone stores the contract as Draft and submits equivalent App Requirements to ConnectorHub.
2. ConnectorHub validates capabilities, Projection types, Delivery semantics, raw-access posture, and Action declarations.
3. ConnectorHub generates a Setup Result containing readiness, provider mappings, Source Policy, streams, gaps, warnings, and verification commands.
4. CornerStone persists the Setup Result and renders a plain-language activation review.
5. Activation is allowed only when all required capabilities are available and owner/policy gates are satisfied.

**Expected records and boundaries**

- `ConnectorCapabilityContract`, `ConnectorSetupResult`, `ConnectorSourcePolicy`, policy decision, and audit event are owner- and namespace-scoped.
- Product code receives no provider token, provider client, raw local path, or direct API handle.
- Setup evidence is available to later Claims and Actions, but it is not itself a Claim or approval.

**Verification and evidence**

- Validate the contract against the pinned schema; run fixture Setup Result generation; inspect stored scope, mappings, warnings, evidence refs, and audit refs.
- Scan CLI/API/UI output and durable state for fixture secrets and raw paths.
- Required CLI path: `cornerstone connector contract validate ... --json` and `cornerstone connector setup plan ... --json`.

**Negative evidence:** zero provider calls before activation; zero secrets exposed; zero ownerless contracts.
**Implementation notes:** Adapt ConnectorHubKit App Requirements and Setup Result behind a CornerStone-owned port; do not expose the `connectorhub` CLI as a second product. [CH-SCHEMAS] [CH-RUNTIME] [CH-CLI] [CS-CLI]

## CS-CH-002 — Required capability missing blocks activation

**Type:** MUST_PASS
**Current posture:** Source-verified by ConnectorHubKit fixtures; CornerStone activation gate proposed.
**Primary use case:** A feature requires repository change metadata, but no allowed provider can supply it.

**Preconditions:** A contract marks one capability `required=true`, and Setup Result cannot map it to a permitted Provider Pack or valid connection.

**Workflow**

1. Setup resolves available and missing capabilities.
2. The missing item includes a stable reason code, suggested resolution, required/optional flag, and provider or permission gap when known.
3. CornerStone records activation state `blocked`, keeps unrelated connectors unchanged, and offers only safe resolutions such as selecting another provider, granting an approved permission, reducing scope, or changing the feature contract.

**Expected result**

- The feature cannot create a Watch Rule, start ingestion, or declare itself ready.
- No fallback silently broadens source scope, data fields, or permissions.
- The user sees the consequence in product language, while technical details remain available in admin context.

**Verification and evidence**

- Fixture Setup Result with one required missing capability returns `blocked`.
- Attempts to activate, poll, or execute against the missing capability return a stable non-zero exit code and audit a denial.
- Existing unrelated source remains operable.

**Negative evidence:** zero Delivery streams for the blocked capability; zero provider calls; zero silent fallback.
**Implementation notes:** Treat Setup Result as an activation gate, not merely documentation. [CH-TEST02] [CH-RUNTIME]

## CS-CH-003 — Optional capability degrades gracefully

**Type:** MUST_PASS
**Current posture:** Source-verified locally; CornerStone product degradation behavior proposed.
**Primary use case:** A daily brief can use GitHub and local activity, but GitHub issue access is optional.

**Preconditions:** At least one required capability is ready and one optional capability is unavailable or unapproved.

**Workflow**

1. ConnectorHub returns `ready_with_gaps` rather than blocking the entire application.
2. CornerStone activates only ready streams and records the missing optional coverage.
3. Briefs and Watch Results state the evidence gap and do not imply complete coverage.
4. Health and freshness surfaces continue to display the missing source until resolved or removed from the contract.

**Expected result**

- The core feature works with reduced evidence.
- Claims remain Draft or Evidence-backed according to actual evidence, not intended coverage.
- UI and API expose which conclusions may be incomplete because of the missing source.

**Verification and evidence**

- Fixture run produces `ready_with_gaps`; only available Delivery streams are created.
- A generated brief cites available Artifacts and lists the optional-source gap.
- No regression to first-value flow without any connectors.

**Negative evidence:** zero synthetic or fabricated data for the missing source; zero readiness overclaim.
**Implementation notes:** Carry Setup Result gaps into Evidence Bundle coverage and freshness, not only the admin screen. [CH-TEST02] [CS-SCEN]

## CS-CH-004 — Owner confirms or overrides Source Policy

**Type:** MUST_PASS
**Current posture:** Local confirm/override helpers exist; durable CornerStone governance proposed.
**Primary use case:** An owner chooses which selected repositories, channels, domains, or local activity modes may feed one workspace.

**Preconditions:** Setup Result contains a recommended Source Policy and the caller has policy authority for the target scope.

**Workflow**

1. CornerStone shows selected provider, resources, included fields, body/content restrictions, fallback, retention, and raw-access posture.
2. Owner confirms, narrows, or requests an override with rationale.
3. CornerStone policy evaluates the change against organization limits, classification, workspace mode, and connector capability.
4. The effective policy is versioned; the prior version remains auditable.
5. Delivery validation uses the effective version, and new data cannot exceed it.

**Expected records**

- `ConnectorSourcePolicyVersion`, owner decision, policy decision, effective constraints, and audit event.
- Cross-namespace promotion is a separate explicit operation; changing Source Policy cannot silently promote data.

**Verification and evidence**

- Confirm and narrow operations produce versioned diffs and stable refs.
- An attempted broadening beyond organization policy is denied with a safe resolution path.
- A later Delivery carrying forbidden fields is rejected or reduced and audited.

**Negative evidence:** no silent widening; no owner confirmation inferred from configuration defaults.
**Implementation notes:** Effective Source Policy should be the intersection of product need, owner choice, organization policy, provider permissions, and egress policy. [CH-RUNTIME] [CH-TEST04] [CS-VS2]

## CS-CH-005 — Swap providers without changing Product logic

**Type:** REGRESSION_GUARD
**Current posture:** Capability mapping and fixtures exist; integrated CornerStone proof absent.
**Primary use case:** Replace one document or source-control provider while preserving Watch, Evidence, and Claim behavior.

**Preconditions:** Two Provider Packs map the same Common Capability to compatible Projection contracts.

**Workflow**

1. Product declares only the Common Capability and accepted Projection type/version.
2. Source Policy selects Provider A, then later selects Provider B through a reviewed policy change.
3. ConnectorHub normalizes each provider into the same product-facing contract.
4. CornerStone ingress, Archive, Evidence, Watch, and Claim handlers remain unchanged.

**Expected result**

- Provider-specific fields appear only in declared, versioned extension namespaces and are optional to Product handlers.
- Existing Artifacts and Evidence remain explainable after a provider switch.
- Provider selection and switch rationale are visible in provenance and audit.

**Verification and evidence**

- Run the same scenario fixture against two Provider Packs; compare normalized object invariants.
- Static scan confirms Product/Mission packages contain no provider SDK imports or provider token handling.
- Existing scenario outputs remain semantically equivalent within declared differences.

**Negative evidence:** zero direct provider APIs in Product code; zero required dependence on provider extensions.
**Implementation notes:** This is the central value of Common Capabilities and generated handler boundaries. [CH-CAPS] [CH-WORKSPACE] [CH-SDK]

## CS-CH-006 — Explain credential and permission gaps without secrets

**Type:** MUST_PASS
**Current posture:** Source-verified locally; production credential backend unverified.
**Primary use case:** GitHub or macOS setup is incomplete, and the user needs a safe, actionable explanation.

**Workflow**

1. ConnectorHub checks connection state, required permissions, granted scopes, platform availability, and credential freshness.
2. Setup/health returns stable status and redacted gap metadata.
3. CornerStone presents cause, affected capabilities, impact, and an authorized resolution path.
4. Technical details are available only to permitted operators.

**Expected result**

- No token, credential-bearing URL, raw path, authorization header, or secret-shaped value appears in API, CLI, UI, logs, reports, screenshots, audit, or generated state.
- Reconnect or platform-permission steps do not claim success until independently verified.

**Verification and evidence**

- Fixture scans across outputs, errors, logs, Control UI model, and state files.
- Expired/revoked credential and missing platform permission scenarios return distinct reason codes.
- Redacted screenshots or DOM proof for operator messaging.

**Negative evidence:** zero raw credential leakage and zero false `connected` state.
**Implementation notes:** Store only an opaque `credential_ref` in CornerStone. [CH-TEST02] [CH-TEST04] [CS-SCEN]

## CS-CH-007 — Convert a Projection into an immutable Artifact

**Type:** MUST_PASS
**Current posture:** Projection Delivery exists; durable CornerStone handoff absent.
**Primary use case:** A GitHub issue, macOS activity session, or Chrome page summary becomes evidence in CornerStone.

**Preconditions:** A valid Delivery targets the active connector application and passes contract, scope, and Source Policy validation.

**Workflow**

1. Ingress validates Delivery and Projection Envelope versions, app identity, capability, source summary, included fields, and EvidenceRef metadata.
2. The Archive engine preserves the permitted canonical payload or source snapshot as an immutable Artifact before interpretation.
3. It writes provenance linking Delivery, provider event, Source Policy version, content hash, and EvidenceRef.
4. Derived representations may then be created; failures do not remove the original Artifact.
5. Product handlers receive only the committed Artifact/derived record, never an unarchived transient payload.

**Expected records**

- Artifact, `ConnectorDeliveryReceipt`, provenance links, derived status, and audit refs in one owner/namespace scope.
- Source classification and trust state remain explicit.

**Verification and evidence**

- Fixture Delivery produces a stable Artifact checksum and receipt.
- Force derived processing failure; original remains queryable with partial/failed derived state.
- Inspect evidence lineage from Artifact back to Projection and Setup/Source Policy.

**Negative evidence:** no Product interpretation before archive commit; no ownerless Artifact.
**Implementation notes:** EvidenceRef is connector provenance metadata, not a replacement for the immutable Artifact. [CH-CONTRACT] [CH-SCHEMAS] [CS-ARCH]

## CS-CH-008 — Acknowledge only after durable archive commit

**Type:** MUST_PASS
**Current posture:** Hub ack/fail/retry exists; cross-engine atomicity proposed.
**Primary use case:** Avoid losing provider data when CornerStone crashes during ingestion.

**Workflow**

1. Worker claims a pending Delivery using its idempotency key.
2. In one durable transaction, CornerStone writes Artifact, receipt, evidence metadata, and an ack-outbox event.
3. Only after commit does the adapter send `ack` to ConnectorHub.
4. If ack transmission fails, the outbox retries safely; duplicate polling resolves to the existing receipt.
5. If archive commit fails, CornerStone sends `fail` or leaves the Delivery pending according to retry policy.

**Expected result**

- No Delivery is acknowledged without a durable Artifact/receipt.
- Reprocessing cannot create conflicting duplicate truth.
- Ack state and archive state can be reconciled after restart.

**Verification and evidence**

- Inject crashes before commit, after commit/before ack, and after ack.
- Replay the same Delivery and verify one Artifact, one active receipt, and consistent audit lineage.
- Reconciliation command reports no orphan ack or orphan Artifact.

**Negative evidence:** zero acknowledged-but-missing Artifacts; zero duplicate downstream effects.
**Implementation notes:** Use an inbox/outbox pattern rather than a best-effort sequential handler. [CH-TEST02] [CS-ARCH]

## CS-CH-009 — Retry transient failures and quarantine poison deliveries

**Type:** MUST_PASS
**Current posture:** Source-verified locally; CornerStone operator integration proposed.
**Primary use case:** A malformed or repeatedly failing provider event must not block healthy ingestion.

**Workflow**

1. Transient failures use bounded exponential backoff and retain the original Delivery reference.
2. Validation failures record a safe reason and do not expose raw payloads.
3. After the configured threshold, poison input moves to quarantine.
4. CornerStone surfaces the affected source, capability, freshness impact, retry history, and safe resolution.
5. Authorized replay creates a new attempt linked to the quarantine record; it does not erase failure evidence.

**Expected records**

- Retry attempts, quarantine item, policy/audit events, source-health impact, and resolved/replayed state.

**Verification and evidence**

- Fixture transient failure retries and succeeds.
- Poison Delivery reaches quarantine at threshold while a healthy Delivery continues.
- Quarantine output and logs pass secret/raw-payload scans.

**Negative evidence:** zero infinite retry loop; zero queue-wide blockage; zero raw payload in operator output.
**Implementation notes:** Quarantine is operational evidence and should be linkable from Mission Control. [CH-TEST02] [CH-TEST04]

## CS-CH-010 — Deduplicate provider events and version changed content

**Type:** MUST_PASS
**Current posture:** Provider-event dedupe exists; CornerStone lineage adaptation proposed.
**Primary use case:** Webhook retries or polling overlap deliver the same GitHub change multiple times.

**Workflow**

1. ConnectorHub computes provider-event and Delivery idempotency identifiers.
2. CornerStone also computes/records source external ID, source revision, content hash, and Projection contract version.
3. Identical events resolve to the existing receipt and Artifact.
4. A changed source revision creates a new Artifact version or lineage-linked Artifact, never a silent overwrite.
5. Product state points to an explicit current version while prior evidence remains inspectable.

**Verification and evidence**

- Deliver the same provider event and payload twice: one logical Artifact/receipt.
- Deliver the same external ID with changed content: new version and lineage link.
- Search and Evidence Bundle snapshots remain reproducible against the version originally used.

**Negative evidence:** zero duplicate active truth; zero mutation of immutable historical evidence.
**Implementation notes:** Hub dedupe protects transport; Archive content addressing and lineage protect truth. Both are needed. [CH-RUNTIME] [CS-SCEN]

## CS-CH-011 — Enforce field and body restrictions from Source Policy

**Type:** MUST_PASS
**Current posture:** Source-verified for fixture providers; Product defense-in-depth proposed.
**Primary use case:** A message connector may deliver metadata and a preview but not full body content.

**Workflow**

1. Setup resolves selected resources and field/body restrictions.
2. Provider Pack applies restrictions while constructing the Projection.
3. CornerStone ingress independently checks `included_fields`, content restriction, classification, and current policy version.
4. Extra or prohibited fields are rejected or stripped according to a frozen policy rule; the event is audited.
5. Product summaries state the restriction so absence of body content is not misinterpreted.

**Verification and evidence**

- Fixture with `body_preview_only` contains no full body.
- Malicious fixture injects a forbidden field; ingress blocks it and records zero durable leak.
- Policy change to narrower scope applies to subsequent data and retention/remediation workflow handles existing data explicitly.

**Negative evidence:** zero forbidden field in Artifact, logs, model prompt, screenshot, or report.
**Implementation notes:** Never trust a provider adapter alone to enforce product data policy. [CH-TEST04] [CH-SCHEMAS]

## CS-CH-012 — Promote EvidenceRef metadata into an Evidence Bundle

**Type:** MUST_PASS
**Current posture:** Handoff documented; integrated implementation proposed.
**Primary use case:** A Claim or Action cites connected-source evidence without revealing provider payloads or credentials.

**Workflow**

1. Ingress stores EvidenceRef metadata and links it to the archived Artifact and Delivery receipt.
2. A search, brief, Claim, or Action selects the Artifact/evidence record.
3. CornerStone assembles an Evidence Bundle containing Artifact refs, search snapshot, Setup Result, Source Policy, Delivery, EvidenceRef, and relevant policy/audit refs.
4. Action Preflight and Action Result are added when applicable.
5. Evidence viewer presents provenance, restrictions, freshness, and gaps.

**Expected result**

- EvidenceRef may support a Claim, but cannot complete or approve it without CornerStone Evidence Bundle rules.
- Raw provider payload remains unavailable by default.

**Verification and evidence**

- Trace one Claim from statement to Evidence Bundle to Artifact to Projection and source summary.
- Export JSON and verify no tokens/raw handles.
- Remove the Artifact link and verify Claim approval is denied.

**Negative evidence:** no EvidenceRef-only “approved truth”; no inaccessible phantom evidence.
**Implementation notes:** Persist safe connector metadata alongside archive provenance. [CH-CONTRACT] [CH-COMMIT] [CS-SCEN]

## CS-CH-013 — Temporary raw access is denied and tightly bounded

**Type:** MUST_PASS
**Current posture:** Local denial/expiry/max-read behavior verified; live semantics unverified.
**Primary use case:** An authorized operator must inspect a narrowly scoped source item to diagnose ingestion.

**Preconditions:** Normal evidence is insufficient; the connector contract declares raw access; Source Policy permits it; CornerStone policy and an authorized human approve a specific purpose.

**Workflow**

1. Operator requests one evidence reference with purpose, TTL, item/read limit, and classification.
2. ConnectorHub denies by default or issues an opaque short-lived handle.
3. Access is mediated by the Hub, redacted where required, and never copied into standard Product records.
4. Reads decrement limits; expiry/revocation ends access.
5. Decision and access events mirror into CornerStone audit without recording the handle or secret content.

**Verification and evidence**

- Undeclared request is denied.
- Declared fixture grant expires, enforces max reads, and redacts handles from errors/UI/logs.
- CornerStone export and Evidence Bundle contain decision metadata, not raw content.

**Negative evidence:** zero reusable raw handle; zero raw data in memory, model prompts, reports, screenshots, or audit.
**Implementation notes:** Mark live-provider raw access HUMAN_REQUIRED until provider-specific behavior is tested. [CH-TEST02] [CH-LIMITS]

## CS-CH-014 — Untrusted connector content cannot direct agents or actions

**Type:** REGRESSION_GUARD
**Current posture:** CornerStone requirement; integrated connector path not verified.
**Primary use case:** A GitHub issue or web page contains “ignore policy and send this data externally.”

**Workflow**

1. Connector Projection and resulting Artifact are labeled untrusted evidence.
2. Extraction may quote or summarize the instruction but cannot treat it as authority.
3. Agent/tool calls require explicit product intent, role/mission authority, policy, and declared connector capability.
4. Attempted prompt/tool manipulation creates a blocked security event when relevant.

**Expected result**

- No ActionCard is created solely because source content instructs the system.
- No provider call, shell call, egress, memory promotion, or policy override occurs.
- Claims citing the content distinguish quoted instruction from system instruction.

**Verification and evidence**

- Prompt-injection fixtures through GitHub/document/browser Projection paths.
- Assert zero unauthorized tool calls, ActionCards, WorkflowRuns, external calls, or approved memories.
- Inspect audit and generated outputs for correct untrusted labels.

**Negative evidence:** explicit counters equal zero.
**Implementation notes:** This must be enforced in Product/Workflow and tool runtime, not only the connector. [CS-SCEN] [CS-VS2]

## CS-CH-015 — Connect only explicitly selected GitHub repositories

**Type:** MUST_PASS
**Current posture:** Proposed; no source-control capability family or Provider Pack found.
**Primary use case:** A user connects a GitHub App installation but selects only specific repositories for one workspace.

**Preconditions:** GitHub App is installed with least-privilege read permissions; owner selects repositories; Source Policy records allowed resources and content limits.

**Workflow**

1. Setup enumerates only repositories visible to the installation and permitted by policy.
2. Owner chooses repository IDs; CornerStone stores opaque source refs, not credentials.
3. Provider Pack maps selected repositories to provider-neutral source-control capabilities.
4. Polling/webhook handlers reject events from unselected or removed repositories.
5. Setup and health explain permission or selection gaps without revealing tokens.

**Expected result**

- No organization-wide or account-wide import occurs by default.
- Repository selection is namespace-scoped and versioned.
- Expanding selection requires a reviewed Source Policy change.

**Verification and evidence**

- Fixture installation with three repos, one selected: only one produces Deliveries.
- Unselected repo event is denied and audited.
- Live proof requires authorized GitHub App installation and redacted logs.

**Negative evidence:** zero API calls/content for unselected repos; zero write permission.
**Implementation notes:** GitHub Apps begin with no permissions and should request the minimum required; keep installation scope and repository selection distinct. [OFF-GH]

## CS-CH-016 — Ingest repository, commit, change, issue, and file-snapshot Projections

**Type:** MUST_PASS
**Current posture:** Proposed; capability catalogue lacks source control.
**Primary use case:** CornerStone builds evidence-backed project context from GitHub without binding Product logic to GitHub schemas.

**Proposed Common Capabilities**

- `source_control.repository.read`
- `source_control.change.read`
- `source_control.issue.read`
- `source_control.file.read`

**Proposed Projection types**

- `source_control.repository.v1`
- `source_control.commit.v1`
- `source_control.change.v1` for pull/merge requests
- `source_control.issue.v1`
- `source_control.file_snapshot.v1`

**Workflow**

1. Provider Pack maps GitHub REST/webhook objects into normalized types.
2. Each Projection carries source IDs, timestamps, actor references, state, safe links/hashes, included-field policy, and EvidenceRef metadata.
3. CornerStone archives each permitted original/snapshot and creates derived searchable representations.
4. Provider-specific fields remain in an optional `x-github-v1` extension and never become required Product semantics.

**Verification and evidence**

- Schema fixtures for all Projection types, including missing/extra-field failures.
- Same Product handler consumes a compatible non-GitHub fixture.
- Artifact Viewer shows provenance and source restrictions.

**Negative evidence:** zero raw access token; zero provider-specific field required by Product handler.
**Implementation notes:** Avoid misusing `knowledge.document.read` for source-control lifecycle objects; create a coherent provider-neutral family. [CH-CAPS]

## CS-CH-017 — Incremental GitHub sync is idempotent

**Type:** MUST_PASS
**Current posture:** Proposed using source-verified Delivery semantics.
**Primary use case:** Webhook redelivery overlaps with scheduled polling.

**Workflow**

1. Provider Pack forms a stable provider-event key from installation, repository, object ID, event/action, and source revision.
2. Webhook receiver verifies origin/signature inside the Connector boundary, then creates normalized Delivery.
3. Poller uses cursors/updated timestamps but emits the same idempotency identity for the same source revision.
4. CornerStone ingress deduplicates Delivery and Artifact as in CS-CH-008/010.
5. Cursor advancement occurs only after durable processing; gaps trigger reconciliation.

**Verification and evidence**

- Send the same fixture event through webhook and polling paths; one logical result.
- Crash before and after cursor update; replay recovers without missing or duplicate active state.
- Out-of-order updates preserve source revision lineage.

**Negative evidence:** zero duplicate Claim/Watch result from one source event; zero cursor advancement before durable commit.
**Implementation notes:** Transport idempotency, archive content identity, and product-event idempotency need separate keys. [CH-TEST02]

## CS-CH-018 — Apply GitHub content restrictions and secret hygiene

**Type:** MUST_PASS
**Current posture:** Proposed.
**Primary use case:** Import metadata and selected Markdown or diff excerpts without archiving secrets, private blobs, or excessive repository content.

**Workflow**

1. Source Policy specifies resource scope, file paths/globs, maximum bytes, content modes, binary handling, diff limits, and secret/redaction rules.
2. Provider Pack excludes disallowed content before Projection.
3. CornerStone ingress scans and redacts configured secret patterns before durable generated outputs while preserving the permitted original according to archive policy.
4. Large or unsupported content is preserved as metadata/partial status rather than silently truncated truth.
5. Generated briefs state limitations such as metadata-only or excerpt-only coverage.

**Verification and evidence**

- Fixtures include token-like strings, private-key material, large diffs, binary files, and forbidden paths.
- Generated outputs/logs/screenshots/reports contain redactions; policy decisions explain blocked fields.
- No raw GitHub authorization data reaches CornerStone.

**Negative evidence:** zero secret leak; zero import outside selected path/content policy.
**Implementation notes:** Secret scanning is defense in depth, not permission to request broad content. [CS-SCEN]

## CS-CH-019 — Deny every GitHub write path

**Type:** REGRESSION_GUARD
**Current posture:** Required by user scope; not implemented in the inspected repos.
**Primary use case:** Ensure the v3 GitHub source can never comment, label, merge, push, create issues, modify files, or change repository settings.

**Required controls**

- GitHub App manifest requests only approved read permissions.
- App Requirements contains no GitHub/source-control Actions.
- Provider Pack manifest declares no write mappings or Action types.
- ConnectorPort exposes no GitHub write helper.
- Egress policy blocks write methods/endpoints even if code attempts them.
- Product UI/CLI exposes no write command.

**Verification and evidence**

- Static scan for GitHub write endpoints, mutation SDK methods, and Action declarations.
- Contract tests reject attempts to declare or execute source-control writes.
- Controlled network test proves denied outbound write attempts and records zero provider mutations.
- GitHub installation permission review confirms read-only scopes.

**Negative evidence:** `github_write_calls=0`, `source_control_actions_declared=0`, `provider_mutations=0`.
**Implementation notes:** This is a release invariant, not a UI preference. A future write release requires a new frozen contract and explicit human approval. [CS-SCEN] [OFF-GH]

## CS-CH-020 — Handle GitHub rate limits, revoked permissions, and repository removal

**Type:** MUST_PASS
**Current posture:** Generic failure simulations exist; GitHub-specific behavior proposed.
**Primary use case:** Connected evidence becomes stale or unavailable without corrupting prior knowledge.

**Workflow**

1. Rate limits produce a retry schedule, visible freshness delay, and no tight retry loop.
2. Revoked credentials/permissions produce a permanent setup gap and suspend affected streams.
3. Repository removal stops future ingestion and records source unavailability.
4. Existing Artifacts remain evidence with source/freshness warnings; they are not silently deleted or treated as current.
5. Reconnection or repository re-selection requires owner action and new verification.

**Verification and evidence**

- Fixture responses for rate limit, auth revocation, permission reduction, repository rename/removal, and installation deletion.
- Health, Setup Result, audit, and Mission Control show distinct states and recovery paths.
- Search/Claim results display stale/unavailable source warnings.

**Negative evidence:** zero silent data deletion; zero claim of fresh sync while suspended.
**Implementation notes:** Reuse Hub failure taxonomy but add provider-specific reconciliation tests. [CH-TEST04]

## CS-CH-021 — macOS capture is off until explicit consent and permission

**Type:** MUST_PASS
**Current posture:** Local permission-gated paths exist; physical-device acceptance required.
**Primary use case:** A user enables personal activity capture without hidden surveillance.

**Workflow**

1. The source is disabled by default and reports required platform permissions without attempting capture.
2. CornerStone explains the data categories, sampling interval, privacy mode, retention, namespace, and how to pause/delete.
3. User explicitly enables the source; ConnectorHub verifies supported macOS environment and platform permission.
4. Collection begins only after both product consent and platform permission are active.
5. Revocation or pause immediately stops new samples and changes health state.

**Expected result**

- Consent and platform permission are distinct, inspectable records.
- No “permission confirmed” environment flag is treated as production proof by itself.
- Source remains personal/owner-scoped unless explicitly promoted.

**Verification and evidence**

- Fixture negative tests with no consent and no permission show zero capture.
- Real-device test records the macOS permission prompt/state, first sample, pause, and revocation.
- UI review confirms the user understands what is collected.

**Negative evidence:** zero samples before both gates; zero hidden startup capture; zero cross-namespace use.
**Implementation notes:** Physical-device and subjective privacy acceptance are HUMAN_REQUIRED. [CH-WATCH] [CH-TEST04]

## CS-CH-022 — Turn activity samples into bounded sessions

**Type:** MUST_PASS
**Current posture:** Activity session/event and usage-metric Projections exist; durable Product mapping proposed.
**Primary use case:** Convert noisy foreground-app observations into meaningful, privacy-safe work sessions.

**Workflow**

1. ConnectorHub samples permitted app/domain signals at the configured interval.
2. It applies privacy mode, retention, and redaction before emitting activity Projections.
3. CornerStone groups related observations by time gap, app/domain category, explicit project hints, and confidence into a `WorkSessionCandidate`.
4. The candidate records observed facts separately from inferred work mode and may be corrected or dismissed.
5. Only an accepted or evidence-backed candidate can become a durable memory or mission input.

**Expected result**

- Raw event noise is not the default user surface.
- Session boundaries, confidence, caveats, and source coverage are visible.
- Domain-only mode never reconstructs clear titles or full URLs.

**Verification and evidence**

- Deterministic sample corpus for focused work, app switching, idle periods, and sparse data.
- Assert stable sessionization, no fabricated intent, and preserved source refs.
- Browser/app usage privacy-mode checks.

**Negative evidence:** zero keystrokes, clipboard, screenshots, cookies, or browser history; zero inference stored as observed fact.
**Implementation notes:** Sessionization belongs to Product/Intelligence even when the Hub provides basic aggregation. [CH-ACTIVITY] [CH-WATCH]

## CS-CH-023 — Create an explicit owner-scoped Watch Rule

**Type:** MUST_PASS
**Current posture:** Proof object exists; CornerStone lifecycle proposed.
**Primary use case:** “Watch selected GitHub repositories and approved browser domains for evidence related to Project Alpha.”

**Workflow**

1. User defines goal, sources, match criteria, schedule/trigger, sensitivity, namespace, retention, and allowed outputs.
2. CornerStone compiles the rule into product logic plus a Connector Capability/Source Policy requirement.
3. Policy checks source permissions, classification, workspace mode, and requested autonomy.
4. Rule starts in Draft, becomes Active only after all required sources are ready, and supports pause/disable/supersede.
5. Every Watch Result links to the exact rule version and source evidence.

**Expected records**

- `WatchRule`, version, owner/scope, connector contract refs, policy decision, status, and audit.
- Rule cannot authorize external Action execution by itself.

**Verification and evidence**

- CLI/API/UI creation and activation with complete and missing source states.
- Cross-namespace activation attempt is denied.
- Rule update creates versioned diff; prior results retain original version link.

**Negative evidence:** zero ownerless/global rule; zero authority expansion from natural-language rule text.
**Implementation notes:** Watch Rules are Product/Mission objects; ConnectorHub only fulfills declared source capabilities. [CH-WATCH] [CS-SCEN]

## CS-CH-024 — Explicit Chrome active-tab capture

**Type:** MUST_PASS
**Current posture:** Substantial local implementation exists; CornerStone handoff proposed.
**Primary use case:** A user deliberately captures the current page as summary-only evidence.

**Workflow**

1. User invokes the extension, which gains temporary active-tab access through a gesture.
2. Extension runs local preflight, displays policy/sensitivity status, and sends only after confirmation.
3. Versioned bounded payload includes hashes, limited metadata/text clip, preflight decision, and `raw_*_stored=false`.
4. Backend independently validates schema and policy, creates summary-only evidence, and emits a Projection.
5. CornerStone archives the permitted evidence and shows it in Capture Inbox.

**Expected result**

- Opening the popup alone does not capture.
- Blocked pages carry no text clip.
- Raw HTML, cookies, local/session storage, screenshots, form values, and history are not collected.

**Verification and evidence**

- Static manifest/permission tests, mocked Chrome API tests, payload validation tests, backend policy tests, and manual unpacked-extension walkthrough.
- Verify source, privacy, confidence, correction, dismiss, and evidence links in CornerStone.

**Negative evidence:** zero broad `<all_urls>` permission; zero capture without gesture/confirmation.
**Implementation notes:** `activeTab` is a useful least-privilege pattern for explicit capture. [CH-CHROME] [OFF-CHROME]

## CS-CH-025 — Allowlist-based Chrome auto capture with two-sided consent

**Type:** MUST_PASS
**Current posture:** Source-verified locally; human browser acceptance required.
**Primary use case:** Capture approved documentation or project sites automatically, without all-site monitoring.

**Workflow**

1. User completes setup, chooses source packs/domains, grants optional host permission, and accepts summary-only policy.
2. Extension records consent/config IDs and approved domains.
3. On allowed triggers, it enforces active-tab, domain, pause, throttling, and bounded-content checks.
4. Payload includes consent/config version, trigger event, domain rule, idempotency key, and content hash.
5. Backend independently verifies matching consent/config, trigger, domain, pause, duplicate, and rate limits before processing.

**Expected result**

- Unknown domains are skipped, not opportunistically added.
- Permission revocation, config mismatch, or pause blocks backend ingestion even if a client is compromised or stale.
- Capture timeline records captured, degraded, skipped, duplicate, and blocked outcomes without raw text.

**Verification and evidence**

- Automated extension/backend tests plus manual Chrome review for grant, first capture, unknown-site skip, pause, resume, revoke, and timeline.
- Inspect backend history and audit for consent/config refs.

**Negative evidence:** zero auto capture before setup; zero unapproved-domain content sent/stored.
**Implementation notes:** Client and server consent must both pass; neither alone is authority. [CH-CHROME] [CH-PAGE]

## CS-CH-026 — Block or degrade sensitive page capture

**Type:** MUST_PASS
**Current posture:** Client/backend policy structures exist; CornerStone mapping proposed.
**Primary use case:** User visits a password, payment, token, compose, or other sensitive surface.

**Workflow**

1. Extension detects configured sensitive signals and assigns `block` or `degraded` preflight.
2. A blocked payload contains no page text; a degraded payload contains only policy-approved metadata/hashes or selected clip.
3. Backend may only maintain or increase restriction; it cannot downgrade a client block.
4. Blocked attempts produce policy/audit/history records but no searchable content Artifact.
5. Product UI explains why capture did not occur and how to use a safer manual alternative when appropriate.

**Verification and evidence**

- Fixtures for password, payment, secret/token, mail compose, private account, unsupported scheme, and oversized page.
- Assert no raw text/HTML in storage, logs, Projection, UI model, screenshots, or reports.
- Verify backend recheck blocks a malicious payload that falsely claims safe content.

**Negative evidence:** zero sensitive content persisted or sent to models.
**Implementation notes:** The backend policy decision should be linked to Capture Inbox as evidence of safe handling. [CH-PAGE]

## CS-CH-027 — Pause, revoke, retain, export, and delete eligible capture state

**Type:** MUST_PASS
**Current posture:** Partial local controls/history exist; full Product governance proposed.
**Primary use case:** A user regains control over continuous collection and retained personal context.

**Workflow**

1. User can pause one Watch Rule, one source, or all collection without losing configuration.
2. Revocation removes active consent/capability and stops future capture immediately.
3. Retention policy distinguishes immutable evidence/audit obligations from eligible derived samples, Watch candidates, and local capture history.
4. Export provides permitted configuration, sessions, results, evidence refs, and audit metadata with redaction.
5. Delete dry-run shows what will be deleted, disabled, retained, or anonymized; authorized execution is audited.

**Verification and evidence**

- State-machine tests for active → paused → resumed → revoked.
- Retention expiry and deletion fixture with before/after record counts and preserved audit integrity.
- Human review of privacy language and destructive-action approval.

**Negative evidence:** zero new samples while paused/revoked; zero misleading “delete everything” promise.
**Implementation notes:** Destructive deletion requires ActionCard/dry-run/approval; immutable evidence policy must be explicit. [CS-SCEN]

## CS-CH-028 — Separate observation, inference, and proposal

**Type:** MUST_PASS
**Current posture:** Analogous proof objects exist; CornerStone ownership proposed.
**Primary use case:** A Watch Result says what was seen, what CornerStone thinks it means, and what it recommends without collapsing them into truth.

**Required structure**

- **Observed:** source-backed facts, timestamps, Artifact/Evidence refs, and restrictions.
- **Inferred:** hypothesis, confidence, caveats, model/version, and alternatives.
- **Proposed:** optional next step, required authority, risk, and whether an ActionCard would be needed.

**Workflow**

1. ConnectorHub delivers evidence only.
2. Product/Intelligence builds an inference candidate.
3. User may correct, dismiss, save as Draft memory, create a Claim, or open a Mission.
4. No proposal executes directly.

**Verification and evidence**

- UI/API schema and fixture walkthrough with uncertain evidence.
- Correction changes inference/history but not the immutable observed Artifact.
- Low-confidence inference cannot become Approved memory without evidence/review.

**Negative evidence:** zero inferred intent labeled as observed fact; zero direct action from a Watch Result.
**Implementation notes:** This separation is central to a non-surveillance, evidence-first Watch experience. [CH-WATCH] [CS-SCEN]

## CS-CH-029 — Combine ActionCard dry-run with ConnectorHub preflight

**Type:** MUST_PASS
**Current posture:** Both mechanisms exist separately; integration proposed.
**Primary use case:** A non-GitHub connector action is proposed from an evidence-backed Claim.

**Workflow**

1. CornerStone creates ActionCard with goal, target, input, evidence, scope, risk, expected impact, and connector capability.
2. Product dry-run validates Claim/evidence, workspace/mission authority, and policy inputs.
3. Adapter calls ConnectorHub Action Preflight for declaration, provider support, Source Policy, permissions, input schema, risk, and idempotency.
4. CornerStone composes both results into one dry-run view and stores all refs.
5. Preflight remains non-side-effecting and cannot count as approval.

**Expected result**

- User sees product impact and provider feasibility together, with distinct ownership.
- Any deny blocks execution and includes a safe resolution path.
- GitHub read-only features have no Action path and cannot reach this scenario.

**Verification and evidence**

- Fixture allowed, undeclared, unsupported, missing permission, invalid input, and missing idempotency cases.
- Assert preflight performs zero provider mutations.
- ActionCard stores `preflight_ref`, policy refs, evidence refs, and audit refs.

**Negative evidence:** zero execution during dry-run.
**Implementation notes:** ConnectorHub preflight is an input to the Action Safety Envelope, not the full envelope. [CH-CONTRACT] [CH-RUNTIME]

## CS-CH-030 — Require evidence, policy, and authorized approval

**Type:** MUST_PASS
**Current posture:** Documented target; real VS2 enforcement currently unverified.
**Primary use case:** A side-effecting external Action is technically supported but must not run without product authority.

**Workflow**

1. ActionCard requires an Evidence Bundle or is labeled unsupported Draft.
2. Trusted RequestContext identifies tenant, owner, namespace, workspace, principal, role, classification, and mission authority.
3. OPA/product policy evaluates data scope, egress, workspace mode, mission contract, risk, cost, and approver.
4. Authorized approval is recorded when required.
5. WorkflowRun may call the connector only when preflight, evidence, policy, and approval refs are all valid and current.

**Verification and evidence**

- Negative matrix for missing evidence, stale preflight, wrong namespace, locked workspace, unapproved high risk, and invalid approver.
- Real policy path and audit correlation, not generated wrapper assertions.
- Successful fixture only after every gate.

**Negative evidence:** zero connector execution on any denied path.
**Implementation notes:** Live release is blocked until the corrected VS2 scenarios execute actual RequestContext/RLS/OPA/egress behavior. [CS-VS2]

## CS-CH-031 — Execute a declared Action and re-ingest its outcome

**Type:** MUST_PASS
**Current posture:** Local fixture Action exists; live path human-gated.
**Primary use case:** An approved message or ticket action produces an external outcome that becomes mission evidence.

**Workflow**

1. WorkflowRun sends declared capability, Action type, bounded input, idempotency key, and required refs through the adapter.
2. ConnectorHub executes only through the selected Provider Pack and returns a versioned Action Result.
3. CornerStone records execution status, provider-safe result metadata, audit correlation, and compensation/rollback expectation.
4. Outcome event is re-ingested as Artifact/Evidence and linked to Action, Claim, Mission, and trajectory.
5. Failures create retry/escalation or compensation candidates rather than disappearing.

**Verification and evidence**

- Fixture success and failure with Action Result schema and idempotency replay.
- Evidence Bundle after execution contains preflight, policy, approval, result, and audits.
- Live-provider execution requires explicit human approval and redacted provider proof.

**Negative evidence:** zero direct provider call from agent/Product code; zero duplicate side effect.
**Implementation notes:** Keep source systems read-only for v3 GitHub; apply this scenario only to separately approved connectors. [CH-CONTRACT] [CS-SCEN]

## CS-CH-032 — Deny undeclared Actions and direct provider bypass

**Type:** REGRESSION_GUARD
**Current posture:** Hub denial locally tested; integrated Product/tool guard proposed.
**Primary use case:** An agent, Agent Pack, or code path tries an Action not declared in the active connector contract.

**Workflow**

1. ConnectorPort exposes only declared capabilities and typed request methods.
2. Product policy rejects direct provider SDK/network access from agents, tools, and Product handlers.
3. ConnectorHub preflight rejects undeclared or unsupported Actions.
4. Pack validation rejects embedded credentials/provider clients or direct writeback logic.
5. Denials are audited with zero external call.

**Verification and evidence**

- Static dependency/import scan and runtime egress test.
- Undeclared Action fixture returns stable denial.
- Malicious Agent Pack/provider-bypass fixture fails activation.
- GitHub write attempt is covered by CS-CH-019.

**Negative evidence:** `direct_provider_calls=0`, `undeclared_actions_executed=0`, `credentials_exposed=0`.
**Implementation notes:** Backend/network enforcement is authoritative; UI omission alone is insufficient. [CH-TEST02] [CS-SCEN]

## CS-CH-033 — Make retries idempotent and expose compensation

**Type:** MUST_PASS
**Current posture:** Duplicate-key behavior exists locally; production semantics proposed.
**Primary use case:** Network timeout leaves the caller uncertain whether an external side effect occurred.

**Workflow**

1. CornerStone generates a stable idempotency key from WorkflowRun/action attempt semantics.
2. ConnectorHub stores request/result association before returning.
3. Retry with the same key returns the existing result or a safely reconciled state.
4. If provider cannot guarantee idempotency, ActionCard and policy expose risk and may require manual reconciliation.
5. Compensation or rollback expectation is recorded; executing compensation is a separate governed Action.

**Verification and evidence**

- Simulate timeout before response, duplicate retry, process restart, and provider duplicate response.
- Assert one external fixture mutation and one canonical Action Result.
- UI/API shows reconciliation and compensation state.

**Negative evidence:** zero hidden automatic compensation; zero duplicate external side effect.
**Implementation notes:** Idempotency is a cross-engine contract, not only a provider concern. [CH-TEST02] [CH-RUNTIME]

## CS-CH-034 — Bind every connector application, Delivery, and Watch to owner and namespace

**Type:** MUST_PASS
**Current posture:** Local app scoping exists; production tenancy blocked by VS2 gaps.
**Primary use case:** Personal GitHub and activity context must not influence an organization workspace without explicit promotion.

**Workflow**

1. Connector application stores tenant, owner, namespace, workspace, purpose, and classification.
2. Every Setup Result, Source Policy, Delivery receipt, Artifact, Watch Rule/Result, Action, and audit event carries or derives the same trusted scope.
3. Polling and read APIs filter by server-derived RequestContext, not client-supplied identifiers alone.
4. Cross-namespace movement uses explicit promote/share/copy/reference with provenance and policy.
5. ConnectorHub app identity is mapped one-to-one or through an explicit scoped mapping to CornerStone context.

**Verification and evidence**

- Tenant/owner/workspace matrix tests for setup, poll, evidence, Watch, Action, and audit.
- Personal-to-org leakage and reverse leakage tests.
- Database/RLS proof in production profile.

**Negative evidence:** zero cross-scope Delivery, search result, inference, Action, or audit disclosure.
**Implementation notes:** Do not rely on ConnectorHub local `app_id` isolation as a substitute for CornerStone tenancy. [CS-VS2] [CS-SCEN]

## CS-CH-035 — Keep credentials exclusively inside ConnectorHub

**Type:** REGRESSION_GUARD
**Current posture:** Hub-only metadata/redaction locally tested; production custody target.
**Primary use case:** Product, agents, generated handlers, and archives never receive provider secrets.

**Required behavior**

- CornerStone stores an opaque connection/credential reference and safe status metadata only.
- Provider authentication, refresh, revocation, and secret rotation occur inside ConnectorHub or an approved secret manager boundary.
- SDK, Projections, EvidenceRefs, errors, logs, audit, Control UI, and reports omit credentials.
- Temporary Raw Access does not become credential access.

**Verification and evidence**

- Fixture secret canary scans across process outputs, durable files, DB rows, logs, screenshots, exports, and generated workspaces.
- Static scan confirms no provider auth library in Product/Mission packages.
- Rotation/revocation test changes connection state without updating Product secrets.

**Negative evidence:** zero raw tokens, private keys, credential-bearing URLs, or auth headers outside ConnectorHub.
**Implementation notes:** Production proof requires the selected secret backend and operational access controls. [CH-TEST04] [CS-SCEN]

## CS-CH-036 — Enforce default-deny egress around ConnectorHub and tools

**Type:** MUST_PASS
**Current posture:** Required target; current CornerStone VS2 report rejects readiness.
**Primary use case:** Only ConnectorHub may reach explicitly approved provider endpoints, while agents/tools remain denied.

**Workflow**

1. Network topology separates API, workers, tool runtime, ConnectorHub, egress gateway, and controlled sink.
2. Default policy denies outbound traffic.
3. ConnectorHub requests egress with provider/capability/action context and allowlisted destination/protocol.
4. Egress gateway and policy enforce DNS/redirect/IP/proxy/protocol constraints and log the decision.
5. Product/tool attempts to bypass ConnectorHub are blocked.

**Verification and evidence**

- Real network tests for blocked sink, allowed provider fixture, redirects, DNS rebinding, IPv4/IPv6, proxy, subprocess, and alternate protocol.
- Correlated policy and network logs.
- Zero unauthorized calls in prompt-injection and undeclared-Action scenarios.

**Negative evidence:** zero unapproved outbound connection.
**Implementation notes:** Application-level mock counters are insufficient; use an actual default-deny topology. [CS-VS2]

## CS-CH-037 — Correlate connector audit with CornerStone audit

**Type:** MUST_PASS
**Current posture:** Both concepts exist locally; cross-engine durable integration proposed.
**Primary use case:** Reconstruct who connected a source, what data arrived, how it was used, and whether an Action ran.

**Workflow**

1. ConnectorHub emits safe audit events for setup, source policy, provider access, Delivery, retry/quarantine, raw access, preflight, and Action result.
2. Adapter mirrors or references events in CornerStone audit with stable correlation IDs.
3. CornerStone adds Artifact, Evidence Bundle, Claim, policy, approval, WorkflowRun, memory, and mission events.
4. Audit viewer reconstructs an end-to-end timeline without raw secrets/payloads.
5. Integrity verification detects missing/reordered/tampered CornerStone events; connector event retention policy is explicit.

**Verification and evidence**

- One ingestion and one Action fixture with complete correlation graph.
- Audit contract tests for required event families and tamper detection.
- Export is permission-aware and redacted.

**Negative evidence:** zero critical lifecycle gap; zero raw secret in audit.
**Implementation notes:** Connector audit is an input to, not a replacement for, the CornerStone tamper-evident ledger. [CH-CONTRACT] [CS-SCEN]

## CS-CH-038 — Version contracts, pin Provider Packs, and migrate safely

**Type:** MUST_PASS
**Current posture:** Schema versions exist; historical diff/migration automation limited.
**Primary use case:** Upgrade ConnectorHubKit or a Provider Pack without silently changing Product behavior.

**Workflow**

1. CornerStone pins supported App Requirements, Setup Result, Projection, Delivery, EvidenceRef, Preflight, Result, and Provider Pack versions.
2. Update analysis displays schema/field/capability/risk/permission changes and migration requirements.
3. New version runs fixture, compatibility, security, and scenario tests in a canary workspace.
4. Breaking change requires explicit migration and rollback plan.
5. Existing evidence remains readable with its original contract version.

**Verification and evidence**

- Golden contract fixtures, forward/backward compatibility tests, unsupported-version denial, migration replay, and rollback.
- Provider Pack version is present in provenance and audit.
- No automatic activation of behavior-changing update.

**Negative evidence:** zero silent contract coercion; zero loss of historical evidence.
**Implementation notes:** Replace prototype monkey-patching with explicit versioned services/interfaces before production adoption. [CH-LIMITS] [CH-RUNTIME]

## CS-CH-039 — Present one CornerStone product, not a ConnectorHub sub-product

**Type:** REGRESSION_GUARD
**Current posture:** Required by SoT; integrated UX absent.
**Primary use case:** A normal user enables GitHub or Watch without learning repository names or connector implementation internals.

**Expected UX**

- Daily navigation remains Home, Search, Artifacts, Claims, Actions, Watch/Capture as appropriate, and learning surfaces.
- “Connected Sources” is a CornerStone capability surface.
- Setup gaps, Source Policy, permissions, retries, quarantine, and audit appear progressively in admin/operator context.
- Provider names are shown when useful for provenance, not as product architecture.
- Native CLI commands begin with `cornerstone`.

**Verification and evidence**

- First-run and connected-source walkthrough with a non-technical user task.
- Navigation/copy review contains no requirement to understand Cornerstone/KnowledgeBase/Connector-Hub repo split.
- Existing first-value upload flow still works without configuring connectors.

**Negative evidence:** zero connector-admin-first landing page; zero separately branded ConnectorHub workflow required for normal use.
**Implementation notes:** Preserve “Calm Surface. Deep Evidence. Safe Action.” [CS-ARCH] [CS-SOT]

## CS-CH-040 — Separate fixture proof from live and production claims

**Type:** REGRESSION_GUARD
**Current posture:** ConnectorHub documents the distinction; CornerStone history demonstrates overclaim risk.
**Primary use case:** A release report summarizes connector readiness without turning deterministic fixtures into live-provider or production claims.

**Required status dimensions**

- Contract/schema verified.
- Local fixture behavior verified.
- Local physical-device behavior verified.
- Live provider read verified.
- Live provider write verified, when in scope.
- Production tenancy/policy/egress verified.
- Human UX/privacy accepted.
- Release/publishing approved.

**Verification and evidence**

- Scenario report lists every applicable row, method, evidence, status, human item, gaps, and verdict.
- Fixture reports include negative counters and exact source/runtime fingerprint.
- Live claims require authorized account/device evidence and redacted transcripts.
- Production claims require real security topology and operations evidence.

**Negative evidence:** zero “production ready” or “live provider ready” statement derived only from fixtures, static docs, wrappers, or mocks.
**Implementation notes:** This guard should be applied to every connector milestone and release package. [CH-RELEASE] [CS-VS2] [USER-SCENARIO]

# Scenario implementation map

| Batch | Recommended scenarios | Smallest valuable deliverable | Release posture |
|---|---|---|---|
| CH-0 Contract adapter | 001-006, 038-040 | CornerStone `ConnectorPort`, contract registry, Setup Result persistence, native CLI, fixture provider | Local fixture only |
| CH-1 Durable read ingestion | 007-014, 034-037 | Delivery inbox/outbox, Artifact/Evidence handoff, retries/quarantine, scope/audit | Local fixture; no live provider claim |
| CH-2 GitHub read-only | 015-020 plus regressions | Source-control capability family and read-only Provider Pack | Live read requires human/external proof; writes forbidden |
| CH-3 Watch local context | 021-028 plus regressions | macOS/Chrome capture through Hub, Capture Inbox and Watch Results in CornerStone | Physical-device/privacy human gates |
| CH-4 Governed Actions | 029-033 plus VS2 | Non-GitHub declared Action through ActionCard and WorkflowRun | Blocked until security and live-action gates pass |

# Proposed contract examples

## CornerStone connector capability contract for GitHub read-only

```yaml
schema_version: cornerstone.connector_contract.v1
contract_id: ccon_project_alpha_github
scope:
  tenant_id: local-dev
  owner_id: user-001
  namespace_id: personal
  workspace_id: project-alpha
purpose: Build evidence-backed project context from selected repositories.
needs:
  - common_capability: source_control.repository.read
    required: true
    accepted_projection_types: [source_control.repository.v1]
  - common_capability: source_control.change.read
    required: true
    accepted_projection_types: [source_control.commit.v1, source_control.change.v1]
  - common_capability: source_control.issue.read
    required: false
    accepted_projection_types: [source_control.issue.v1]
  - common_capability: source_control.file.read
    required: false
    accepted_projection_types: [source_control.file_snapshot.v1]
source_policy_request:
  selected_resources: [github:repo:owner/project-alpha]
  content_mode: metadata_and_markdown_excerpt
  max_content_bytes: 200000
  allowed_paths: ["docs/**", "README.md"]
  raw_access: denied
  retention_days: 90
actions: []
delivery:
  ack_required: true
  retry_policy: bounded_exponential
  quarantine_after_attempts: 3
```

The empty `actions` list is deliberate and should be enforced through schema, Provider Pack manifest, CLI surface, egress policy, and regression tests.

## Projection-to-Artifact mapping

```json
{
  "delivery_id": "del_000123",
  "projection_type": "source_control.change.v1",
  "source": {
    "provider": "github",
    "repository_ref": "github:repo:owner/project-alpha",
    "external_id": "pull_request:42",
    "source_revision": "sha256:..."
  },
  "artifact": {
    "artifact_id": "artifact_content_hash",
    "owner_id": "user-001",
    "namespace_id": "personal",
    "workspace_id": "project-alpha",
    "trust_state": "evidence_source",
    "security_classification": "internal",
    "connector_delivery_ref": "del_000123",
    "source_policy_ref": "cspol_0007",
    "evidence_refs": ["evref_0042"]
  }
}
```

## Action handoff envelope

```json
{
  "action_card_id": "action_001",
  "workflow_run_id": "wfrun_001",
  "connector_preflight_ref": "preflight_001",
  "evidence_bundle_ref": "eb_001",
  "policy_decision_ref": "policy_001",
  "approval_ref": "approval_001",
  "idempotency_key": "idem_wfrun_001_attempt_1",
  "execution_allowed": true
}
```

The connector preflight cannot set `execution_allowed`; that value is derived only after CornerStone validates the full safety envelope.

# Verification plane and evidence package

Each implementation batch should expose a native verifier such as:

```text
cornerstone scenario verify connector-contract-adapter --json
cornerstone scenario verify connector-read-ingestion --json
cornerstone scenario verify github-read-only --json
cornerstone scenario verify watch-connected-evidence --json
cornerstone scenario verify connector-governed-action --json
```

A release-facing evidence package should contain:

- frozen scenario contract and matrix;
- exact source commit/tree and dirty-state metadata;
- App Requirements/CornerStone contract fixtures and Setup Results;
- Projection, Delivery, Artifact, Evidence Bundle, Action, and audit records;
- CLI transcripts with stable exit codes and refs;
- browser/UI proof where claimed;
- negative-evidence counters;
- redaction and secret scans;
- provider/network call ledger;
- human-required table and human evidence when available;
- manifest hashes generated after final report bytes exist.

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

# Human-required gates

| ID | Why AI cannot verify | Required human action | Expected evidence | Release impact |
|---|---|---|---|---|
| CH-H01 | Real GitHub permissions and account state are unavailable to deterministic fixtures. | Approve and run selected-repository GitHub App read-only rehearsal. | Redacted installation permissions, selected repos, API/call ledger, Deliveries, audit, zero-write proof. | Blocks live GitHub readiness. |
| CH-H02 | macOS permission prompts and host behavior require a physical supported Mac. | Enable, pause, revoke, and observe local activity capture. | Screen recording/screenshots, status transcript, first sample, pause/revoke evidence. | Blocks physical-device Watch readiness. |
| CH-H03 | Chrome extension permission and privacy experience is subjective and browser-bound. | Review setup, explicit capture, allowlist auto capture, sensitive block, pause/revoke. | Recording/screenshots, timeline, issue list, accept/reject note. | Blocks human Chrome/privacy acceptance. |
| CH-H04 | Production RequestContext, RLS, OPA, and network isolation require a production-like environment. | Run corrected VS2 security scenarios against the integrated adapter. | Scenario-specific DB/policy/network transcripts and reports. | Blocks production connected-source release. |
| CH-H05 | Live side effects mutate third-party systems and require authorization. | For a separately scoped non-GitHub Action, approve and run live preflight/execution. | Approval, redacted request/result, provider state, audit, idempotency evidence. | Blocks live Action readiness; not required for GitHub read-only. |
| CH-H06 | Usability and trust are subjective. | Complete first-use Connected Sources and Capture Inbox walkthrough. | Timed task, notes, screenshots/recording, acceptance decision. | Blocks human UX acceptance. |
| CH-H07 | Backup/restore and operational recovery need production-like infrastructure. | Execute backup, restore, connector reconciliation, and audit verification. | Backup/restore logs, source cursor reconciliation, Artifact/search/audit replay. | Blocks production operations readiness. |

# Decisions to freeze before implementation

1. **Product release vs vertical-slice naming.** Use product `v3` separately from VS scenario identifiers to avoid collision with the historical VS-3 Tool SDK milestone.
2. **CornerStone-owned ConnectorPort.** Product and Archive code never depend directly on ConnectorHubKit classes or provider SDKs.
3. **First live source is read-only.** GitHub v3 has no Actions and requests only approved read permissions.
4. **Ack boundary.** Delivery acknowledgement occurs only after durable Artifact/receipt commit.
5. **Evidence model.** EvidenceRef is metadata; immutable Artifact plus Evidence Bundle remain truth foundations.
6. **Watch ownership.** ConnectorHub captures; CornerStone owns Watch rules, interpretation, approvals, memory, missions, and history.
7. **Effective Source Policy.** It is an intersection of organization policy, owner confirmation, workspace classification/retention, product need, provider permissions, and egress.
8. **No live readiness before VS2.** Production tenancy/policy/egress claims require scenario-specific execution evidence.
9. **Native CLI.** Every product capability has a `cornerstone ...` command with JSON output, scope, evidence refs, audit refs, and stable exits.
10. **Contract versioning.** Every Setup Result, Projection, Delivery, Preflight, Result, and Provider Pack is pinned and migration-aware.

# Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Repository merge or SDK leakage | Product becomes provider-coupled and credentials spread. | Hexagonal ConnectorPort; import/dependency scans; generated-handler boundary. |
| Acknowledging before archive commit | Silent evidence loss after crash. | Transactional inbox/outbox and reconciliation scenarios. |
| Treating EvidenceRef as truth | Claims cite metadata without durable source. | Require Artifact and Evidence Bundle links. |
| Watch feels like surveillance | Loss of trust and unsafe personal data collection. | Default off, explicit rule/consent, least privilege, Observed/Inferred/Proposed separation, pause/delete controls. |
| Chrome client checks are bypassed | Sensitive content reaches backend. | Strict versioned payload plus independent backend policy validation. |
| GitHub scope expands into writeback | Unauthorized source-system mutation. | No Actions, read-only permissions, egress enforcement, zero-write release counter. |
| Fixture success becomes production claim | Unsafe release and false confidence. | Multi-dimensional readiness statuses and CS-CH-040. |
| Prototype runtime copied unchanged | Local JSON/monkey-patch architecture becomes production debt. | Port contracts/tests; implement explicit services and durable storage. |
| VS2 bypass | Cross-tenant leaks or ungoverned egress. | Treat corrected VS2 as a hard prerequisite. |
| Contract/schema drift | Historical evidence or handlers break. | Pin versions, compatibility fixtures, migration and rollback scenarios. |

# Recommended implementation sequence

## Phase 0 — Contract and adapter foundation

Implement `ConnectorPort`, contract registry, scoped connector application records, Setup Result persistence, native CLI, and deterministic local Provider Pack fixture. Verify CS-CH-001 through 006, 038 through 040.

## Phase 1 — Durable connected evidence

Implement Delivery inbox/outbox, Artifact/receipt handoff, Source Policy validation, EvidenceRef mapping, retry/quarantine, and audit correlation. Verify CS-CH-007 through 014 and 034 through 037 using fixtures only.

## Phase 2 — GitHub read-only

Add the source-control capability family and GitHub Provider Pack with selected repositories, incremental sync, content restrictions, failure handling, and hard zero-write controls. Verify CS-CH-015 through 020. Only after deterministic proof should CH-H01 be attempted.

## Phase 3 — Watch connected evidence

Adapt macOS activity and Chrome capture into CornerStone Capture Inbox, Watch Rules/Results, corrections, privacy controls, and evidence-backed briefs. Verify CS-CH-021 through 028, then complete CH-H02, H03, and H06.

## Phase 4 — Governed Actions for separately approved connectors

After corrected VS2 and durable audit pass, integrate ActionCard, ConnectorHub preflight, WorkflowRun execution, results, and outcomes. GitHub remains read-only unless a future separately approved contract changes scope. Verify CS-CH-029 through 033 and complete applicable human gates.

# Conclusion

ConnectorHubKit provides CornerStone with a mature **conceptual and local-verification foundation** for connected sources: declare needs, resolve setup and policy, deliver normalized evidence, isolate credentials, constrain raw access, preflight declared Actions, and make failures inspectable. Its value is not the number of current providers; it is the disciplined boundary that prevents provider mechanics from contaminating product truth and authority.

The correct CornerStone adaptation is therefore neither a repo merge nor a connector-first product. It is a CornerStone-owned adapter and scenario set that turns safe connector outputs into immutable, owner-scoped evidence and then lets the normal CornerStone loop perform understanding, decisions, governed actions, and learning.

For v3, the strongest practical scope is **Connected Evidence**: selected GitHub read-only ingestion plus opt-in WatchAgent macOS/Chrome capture, all flowing into Artifact, Evidence, Brief, Claim, and Memory/Mission candidates. Governed external Actions should follow only after the corrected VS2 security plane and action scenarios have concrete evidence.

# Source register

## CornerStone authority and implementation evidence

- **[CS-SOT]** `Tocktock/Cornerstone`, `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md`, `docs/sot/README.md`, and root `README.md`; inspected revision `cb682e9060778148f5303021e450abc962bf9381`.
- **[CS-SCEN]** `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` and Scenario-First rules.
- **[CS-ARCH]** `docs/architecture/ONE_PRODUCT_THREE_ENGINES.md`.
- **[CS-ROADMAP]** `docs/implementation/ZERO_BASE_IMPLEMENTATION_ROADMAP.md`.
- **[CS-CLI]** `packages/cornerstone_cli/main.py` and CLI-native-first contract.
- **[CS-CURRENT]** `packages/cornerstone_cli/product_runtime.py`, current local/mock runtime evidence and source searches.
- **[CS-VS2]** `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_FINAL_REPORT_2026-06-20.md`, updated 2026-06-21.

## ConnectorHubKit implementation evidence

- **[CH-README]** `Tocktock/Connector-Hub`, root `README.md`; inspected revision `4c54fa6e7024396d0e15c2584dd146ec13d0871c`.
- **[CH-CONTRACT]** `docs/internal-cornerstone-connectorhub-contract.md`.
- **[CH-STABILIZE]** `docs/internal-stabilization-goal-and-scenarios.md`.
- **[CH-COMMIT]** Commit `4c54fa6e7024396d0e15c2584dd146ec13d0871c`, “Fix internal Claim mapping docs.”
- **[CH-RUNTIME]** `connectorhubkit/runtime.py`.
- **[CH-PACKS]** `connectorhubkit/provider_packs.py`.
- **[CH-CAPS]** `catalog/common-capabilities.json`.
- **[CH-WORKSPACE]** `connectorhubkit/workspace.py`.
- **[CH-SDK]** `connectorhubkit/sdk.py`.
- **[CH-SCHEMAS]** JSON schemas for App Requirements, Setup Result, Projection Envelope, Delivery, Action Preflight, and EvidenceRef.
- **[CH-CLI]** `connectorhubkit/cli.py` and `Makefile`.
- **[CH-CONTROL]** `connectorhubkit/control_ui.py`.
- **[CH-WATCH]** Watch Agent product/runtime docs, `connectorhubkit/daily_review.py`, and `examples/character_watch_agent/`.
- **[CH-ACTIVITY]** `docs/watch-agent-activity-usage-metrics.md`.
- **[CH-CHROME]** `browser_extension/character-watch-page-capture/` and first-run implementation report.
- **[CH-PAGE]** `character_watch_agent/page_summary/models.py`, `verification.py`, and API routes.
- **[CH-TEST02]** `tests/test_v0_2_local_vertical_slice.py`.
- **[CH-TEST04]** `tests/test_v0_4_provider_packs_runtime_safety.py`.
- **[CH-RELEASE]** `docs/master-acceptance-gate-report-v1.0-rc1.md`.
- **[CH-LIMITS]** `docs/known-limitations-v1.0.md`.

## User-provided project instructions

- **[USER-CONSTITUTION]** CornerStone Project Operating Constitution.
- **[USER-SCENARIO]** Scenario-First Agent Instruction — Final.
- **[HIST-SOT]** Uploaded `project-sot.md`, treated as historical where superseded by current repo authority.

## External primary references

- **[OFF-GH]** GitHub documentation on GitHub App permissions and least-privilege selection.
- **[OFF-CHROME]** Chrome Extensions documentation for `activeTab` and runtime/optional permissions.

# Glossary

| Term | Meaning in this guide |
|---|---|
| ConnectorHub | The Connector / Provider / Action engine concept used inside CornerStone. |
| ConnectorHubKit | The current `Tocktock/Connector-Hub` local reference implementation/package. |
| ConnectorPort | Proposed CornerStone-owned interface hiding ConnectorHubKit/provider implementation. |
| Connected Source | User-facing CornerStone capability backed by one or more ConnectorHub providers. |
| Projection | Normalized provider-derived product input; evidence candidate, not product truth. |
| Delivery | App-scoped transport record carrying a Projection with ack/retry lifecycle. |
| EvidenceRef | Safe connector-side provenance metadata; not an immutable Artifact by itself. |
| Setup Result | Readiness, mapping, Source Policy, gaps, streams, actions, warnings, and verification result. |
| Source Policy | Effective constraints on provider, resources, fields, content, fallback, and raw access. |
| Watch Rule | CornerStone-owned owner-scoped rule defining what to observe and what outputs are allowed. |
| Watch Result | Product object separating Observed facts, Inferred meaning, and Proposed next step. |

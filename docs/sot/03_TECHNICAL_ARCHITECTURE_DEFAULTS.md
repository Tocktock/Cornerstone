# CornerStone Technical Architecture Defaults

**Date:** 2026-06-08  
**Owner:** JiYong / Tars  
**Status:** Active technical-default document, subordinate to the product SoT  
**Source basis:** Compatible engineering defaults extracted from the older `project-sot.md`, updated to match the new product goal.

## 1. Authority and scope

This document is not the product goal. It is a technical-default companion to:

1. `01_PRODUCT_GOAL_AND_DIRECTION.md`
2. `02_MUST_PASS_SCENARIO_STANDARD.md`

If this file conflicts with the product goal or scenario standard, the product goal/scenario standard wins.

## 2. Architecture thesis

Build CornerStone as one product with modular internal engines:

```text
CornerStone product
├── Product / Mission / Intelligence Engine
├── Archive / Evidence / KnowledgeBase Engine
└── Connector / Provider / Action Engine
```

The user should not need to understand the engine split. The split exists to keep evidence, product meaning, and external-provider safety governable.

## 3. Zero-base default stack

For a new implementation, use these defaults unless a scenario contract or owner decision changes them:

| Layer | Default |
|---|---|
| Product shell | One web app + one API gateway/control plane |
| Backend API | FastAPI or equivalent typed HTTP service |
| Frontend | React/Next.js or equivalent single web app |
| Durable state | PostgreSQL-first |
| Tenant isolation | Postgres RLS plus service-layer policy checks |
| Keyword search | PostgreSQL full-text search |
| Vector search | pgvector |
| Object/original storage | Content-addressed local filesystem first; S3/MinIO adapter later |
| Policy | OPA/Rego-compatible policy engine |
| Audit | Append-only tamper-evident audit events with checkpoint/hash-chain/Merkle direction |
| Workers | Background job worker for extraction, redaction, chunking, indexing, embeddings, briefs |
| Model access | Provider-neutral model router with deterministic local/test provider for CI |
| Tool execution | Sandbox/capability-based execution; egress denied by default |
| Connector access | ConnectorHub-mediated capabilities only |

## 4. Non-negotiable data principles

### 4.1 Archive original first

Every input must be preserved as an immutable original artifact before extraction, summarization, embedding, normalization, or memory synthesis.

Derived processing may fail. Original preservation may not.

### 4.2 Evidence before authority

Drafts can exist without evidence. Completed/approved claims, memory entries, rules, playbooks, mission decisions, and actions require evidence or explicit risk-aware policy.

### 4.3 Owner-scoped context

Every truth-bearing object must have an owner and namespace. No ownerless global context is allowed.

### 4.4 Conversation starts, structure persists

Conversation is the workbench. Durable product value is captured in artifacts, evidence bundles, briefs, claims, knowledge capsules, memory entries, mission cards, action cards, workflows, audit events, and experience records.

## 5. Core domain objects

Minimum durable objects for VS-0+:

| Object | Purpose |
|---|---|
| `Tenant` | Isolation boundary |
| `Principal` | User/service account |
| `Namespace` | Owner-scoped context boundary |
| `Workspace` | User-facing active context |
| `Artifact` | Immutable original input |
| `DerivedRepresentation` | Extracted text/OCR/ASR/structured data/chunks/embeddings |
| `SearchSnapshot` | Saved query, filters, result refs, timestamp |
| `EvidenceBundle` | Reproducible evidence for claim/brief/action |
| `Brief` | Evidence-backed synthesis from messy input/search |
| `Claim` | Draft/Evidence-backed/Approved conclusion or recommendation |
| `ActionCard` | Proposed/executed action with diff, risk, policy, approval, audit |
| `WorkflowRun` | Governed execution path for actions |
| `PolicyDecision` | Allow/deny/escalate result with cause/resolution |
| `AuditEvent` | Append-only event record |
| `ModelRun` | Provider-neutral inference record |

Add after VS-0+:

| Object | Purpose |
|---|---|
| `KnowledgeCapsule` | Reusable source-aware knowledge unit |
| `MemoryEntry` | Permanent wiki/living memory entry |
| `MissionCard` | Durable mission/decision object |
| `MissionContract` | Scope, allowed actions, stop rules, authority |
| `AgentRoleContract` | Agent role/tool/memory/evidence/policy contract |
| `TrajectoryEvent` | Mission learning event |
| `ExperienceRecord` | After-action review and lesson source |
| `ConnectorCapability` | ConnectorHub-mediated declared capability |
| `AgentPack` | Extension unit with role contracts, capabilities, playbooks |

Every object that can influence answers, actions, memory, or learning should include:

```text
tenant_id
namespace_id
workspace_id where relevant
owner_id
trust_state where relevant
security_classification
provenance
created_at
updated_at
audit references
```

## 6. Trust ladder

Use this trust state consistently:

1. `Draft` — exploratory, unsupported, useful for thinking.
2. `Evidence-backed` — supported by artifacts, search snapshots, source refs, tool outputs, policy decisions, or action results.
3. `Approved` — accepted by owner, reviewer, team, organization, or policy-defined authority.

Promotion from one state to another must be explicit, auditable, and evidence-aware.

## 7. VS-0+ API surface

Minimum APIs for the first implementation slice:

```text
GET    /health
GET    /ready
GET    /openapi.json

GET    /workspaces
POST   /workspaces

POST   /artifacts
GET    /artifacts/{artifact_id}
GET    /artifacts/{artifact_id}/derived

POST   /search
GET    /search-snapshots/{snapshot_id}

POST   /evidence-bundles
GET    /evidence-bundles/{bundle_id}

POST   /briefs
GET    /briefs/{brief_id}

POST   /claims
GET    /claims/{claim_id}
POST   /claims/{claim_id}/attach-evidence
POST   /claims/{claim_id}/approve

POST   /actions
GET    /actions/{action_id}
POST   /actions/{action_id}/dry-run
POST   /actions/{action_id}/approve
POST   /actions/{action_id}/execute

POST   /policy/evaluate
GET    /audit
GET    /audit/{event_id}
```

Public HTTP APIs should be documented through OpenAPI.

## 8. VS-0+ database tables

Start with:

```text
tenants
principals
memberships
namespaces
workspaces

artifacts
artifact_blobs
artifact_versions
derived_representations
chunks
embeddings

search_snapshots
evidence_bundles
evidence_items

conversations
conversation_messages
briefs
claims
claim_events

action_cards
action_dry_runs
action_approvals
action_executions
workflow_runs

policy_decisions
audit_events
audit_checkpoints

model_runs
```

Do not add broad mission/autonomy/memory complexity until the VS-0+ loop passes.

## 9. Workflow and action safety

Agents, tools, and model outputs may propose actions. They may not directly mutate product state, source systems, or external systems.

External or risky actions must go through:

```text
ActionCard → dry-run → policy decision → approval when required → WorkflowRun → execution result → audit
```

Dry-run must show:

- diff;
- expected impact;
- data scope;
- external calls if any;
- policy decision;
- risk level;
- approval requirement;
- rollback/compensation expectation where possible.

## 10. Connector boundary

Connector/provider access belongs behind ConnectorHub-mediated capability.

CornerStone owns:

- product meaning;
- missions;
- approvals;
- claims;
- memory synthesis;
- workflow orchestration;
- user/operator UX;
- product state.

ConnectorHub owns:

- provider access;
- credentials;
- provider clients;
- source policy;
- projections;
- declared actions;
- delivery;
- connector audit;
- retry/quarantine;
- temporary raw access;
- verification.

No Agent Pack, model prompt, workflow handler, or product UI may directly receive raw provider credentials by default.

## 11. Prompt-injection and untrusted content boundary

Uploaded files, connected records, web pages, logs, code comments, emails, Slack/Notion text, tool outputs, and generated drafts are untrusted evidence.

Required behavior:

- label external/archived content as untrusted;
- never follow instructions embedded in artifacts;
- policy-check every tool/action call;
- redact secrets before durable generated knowledge;
- test prompt injection fixtures in CI;
- audit blocked tool/action attempts where relevant.

## 12. Local-first / on-prem requirement

VS-0+ must run locally with minimal setup. The first useful flow should not require:

- live external connector credentials;
- paid LLM provider credentials;
- organization SSO;
- custom ontology setup;
- production infrastructure.

Use a deterministic local/test model provider in CI. External model providers can be enabled later by workspace policy.

## 13. Definition of Done for VS-0+

VS-0+ is not done until these are verified with concrete evidence:

1. One product shell starts locally.
2. Personal workspace exists by default.
3. Artifact original is preserved before derived processing.
4. Identical content deduplicates or links to same content identity.
5. Unknown input is still archived.
6. Failed extraction does not lose original.
7. Redaction protects generated outputs/logs.
8. Prompt-injection fixture cannot trigger tools/actions.
9. Search finds uploaded content.
10. Search snapshot can become evidence.
11. Evidence opens original and derived representation.
12. Brief cites evidence and labels uncertainty/gaps.
13. Claim without evidence cannot become approved.
14. Claim with evidence can become evidence-backed/approved through policy.
15. Action card requires dry-run.
16. Risky/external action requires approval.
17. Direct write outside Workflow/Action path is denied.
18. Audit records ingestion, search, evidence, claim, policy, action, approval, execution.
19. Audit tampering is detectable.
20. Personal and organization namespaces do not mix silently.

## 14. Explicitly deferred from VS-0+

- Full permanent wiki and Memory Sovereignty Center.
- Live ConnectorHub provider E2E.
- External writeback to real systems.
- Full Agent Pack registry.
- Full multi-brain routing optimization.
- Full Product Learning/self-improvement.
- Production SSO, K8s, Helm, advanced backup/restore.

These are long-term requirements, but they should not block the first identity slice.

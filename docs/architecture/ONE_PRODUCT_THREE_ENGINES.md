# One Product, Three Internal Engines

**Date:** 2026-06-08  
**Owner:** JiYong / Tars  
**Status:** Architecture boundary document

## 1. Decision

CornerStone should be implemented as one coherent product experience with three internal engines:

```text
CornerStone
├── Product / Mission / Intelligence Engine
├── Archive / Evidence / KnowledgeBase Engine
└── Connector / Provider / Action Engine
```

This is a product and safety boundary, not a user-facing split.

## 2. User-facing rule

Users should see:

- Mission Control / Ops Inbox
- Conversation Workbench
- Permanent Wiki
- Evidence
- Briefs
- Claims
- Missions
- Actions
- Approvals
- Learn / Experience
- Admin / Operator controls

Users should not need to understand:

- which repository originally supplied a capability;
- whether a function came from `Cornerstone`, `KnowledgeBase`, or `Connector-Hub`;
- connector credentials or provider clients;
- archive internals;
- model-provider implementation details.

## 3. Engine ownership

### Product / Mission / Intelligence Engine

Owns:

- user-facing product shell;
- Mission Control / Ops Inbox;
- conversation workbench;
- evidence-backed briefs;
- claims and trust ladder;
- mission and decision cards;
- mission contracts;
- orchestrator/specialist agent UX;
- permanent wiki synthesis;
- memory sovereignty UX;
- action proposal UX;
- approvals and review flows;
- after-action reviews;
- experience library;
- product learning proposals.

Does not own:

- raw provider credentials;
- direct provider clients;
- arbitrary external writeback;
- immutable archive truth without Archive Engine.

### Archive / Evidence / KnowledgeBase Engine

Owns:

- immutable original artifacts;
- content-addressed storage;
- stable IDs/URIs;
- hashes/checksums;
- derived representations;
- chunks and embeddings;
- redaction;
- provenance;
- search indexes;
- evidence bundles;
- source-aware memory material.

Does not own:

- product UX as separate product;
- external provider credentials;
- final human/organization approval authority;
- autonomous action execution.

### Connector / Provider / Action Engine

Owns:

- provider access;
- credentials/custody references;
- provider clients;
- source policy;
- projections;
- declared connector actions;
- delivery;
- external action execution;
- connector audit;
- retry/quarantine;
- temporary raw access;
- verification;
- SDK/control bridge.

Does not own:

- CornerStone product meaning;
- permanent wiki truth;
- mission approval policy;
- direct user-facing product identity.

## 4. Boundary flow

### Ingestion

```text
Connector or upload
→ Archive Engine preserves original artifact
→ Archive Engine creates derived representations
→ Product Engine creates brief/claim/memory candidates
→ Evidence Bundle links everything
```

### Action

```text
Claim or Mission proposes action
→ Product Engine creates ActionCard
→ Workflow/Policy evaluates dry-run
→ Connector Engine executes external capability only if allowed
→ Audit records lifecycle
→ Outcome re-ingested as evidence/experience
```

## 5. Zero-base repo strategy

Do not start by merging the three existing repos.

Create a new canonical `CornerStone` repo and port capabilities by scenario:

```text
CornerStone/
  apps/
    web/
    api/
  packages/
    domain/
    archive_engine/
    evidence_engine/
    intelligence_engine/
    workflow_engine/
    connectorhub_client/
    policy_engine/
    audit_engine/
    model_router/
    scenario_runner/
  docs/
  tests/
  migrations/
  deploy/
```

Existing repos become evidence/adapters:

| Existing repo | Import role |
|---|---|
| `Tocktock/Cornerstone` | Product/RAG patterns, FastAPI/search/chat references |
| `Tocktock/KnowledgeBase` | Archive/evidence/redaction/stable URI concepts |
| `Tocktock/Connector-Hub` | Connector/provider/action boundary and capability model |

## 6. Adapter rule

Port by capability, not by repository.

Adapters must pass the same scenario contract before becoming default:

```text
archive_engine.adapters.knowledgebase_importer
intelligence_engine.adapters.cornerstone_rag_importer
connectorhub_client.adapters.connectorhubkit
```

Do not let adapter convenience break product boundaries.

## 7. Anti-drift rules

- Do not expose old repo names as product navigation.
- Do not duplicate connector credential handling in Product Engine.
- Do not let generated memory become archive truth.
- Do not let actions bypass Workflow/Action/Policy/Audit.
- Do not claim provider E2E without real verification.
- Do not claim scenario PASS without evidence.

## 8. First slice

Implement only this first:

`Artifact → Search → Evidence-backed Brief → Claim → ActionCard dry-run/approval/execution → Audit`

Memory, full autonomy, Agent Packs, and Experience Library depend on this foundation.

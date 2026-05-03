# 00 — Backend Architecture

## Goal

Build Cornerstone's backend trust foundation before expanding UI, graph exploration, or AI features. The backend must prove that official organizational context is grounded in source data, provenance, freshness, review state, and auditability.

## Product boundary

Cornerstone is not a wiki, search engine, or ungrounded AI answer bot. The backend is a context integrity system with six bounded areas:

1. **Source Studio** — source connection state, OAuth state, sync state, operational metadata.
2. **Evidence Layer** — Artifacts and EvidenceFragments with provenance and freshness.
3. **Review Layer** — EvidenceFragment review/reject transitions and reviewer authorization.
4. **Officialization Layer** — Concepts and DecisionRecords with quality gates.
5. **Serving Layer** — grounded context responses for humans, AI, MCP, and Codex clients.
6. **Observability Layer** — structured JSON logs, request IDs, and audit events.

## v0.3 backend trust loop

```text
1. Register production or non-production source
2. Complete OAuth or equivalent connection flow
3. Sync source objects into Artifacts
4. Extract EvidenceFragments with required provenance
5. Review EvidenceFragments
6. Create Concept candidates and/or DecisionRecords
7. Block officialization unless support is reviewed, fresh/aging, production-eligible, and reviewer-authorized
8. Serve grounded context with trust labels, citations, freshness, and limitations
9. Emit audit events and structured logs for every trust transition
```

## Runtime shape

```text
FastAPI app
  ├── /healthz
  └── /v1
      ├── /sources
      ├── /artifacts
      ├── /evidence
      │   └── /{evidenceFragmentId}/review
      ├── /decision-records
      ├── /concepts
      │   └── /{conceptId}/officialize
      ├── /context/query
      └── /audit-events

Services
  ├── FreshnessPolicy
  ├── EvidenceExtractor
  ├── OfficializationService
  └── GroundedContextService

Middleware
  └── RequestLoggingMiddleware

Repository boundary
  ├── InMemoryStore         # fast contract tests only
  └── SQLAlchemyStore       # PostgreSQL persistence adapter
```

## Data ownership

| Entity | Owner | Purpose |
| --- | --- | --- |
| DataSource | Source Studio | Operational state for connected systems. |
| Artifact | Evidence Layer | Captured source object snapshot. |
| EvidenceFragment | Evidence + Review Layer | Extracted evidence with provenance, freshness, and review state. |
| DecisionRecord | Officialization Layer | Reviewed decision and rationale backing official context. |
| Concept | Officialization Layer | Reviewable semantic unit that may become official. |
| AuditEvent | Observability Layer | Durable record of review and officialization transitions. |
| GroundedContextResponse | Serving Layer | Shared human/AI response contract. |

## Source status state machine

```text
disconnected
  → connecting
  → pending_auth
  → connected
  → sync_pending
  → syncing
  → connected | degraded | failed | stale
```

Current enforcement:

- `pending_auth` sources cannot sync.
- `pending_auth` sources do not count as real connected production sources.
- `productionEnabled=false` sources may be synced for development/demo, but their evidence cannot officialize production context.
- Failed sync after prior success marks the source `degraded`.
- Sync failure rolls back new Artifact/Evidence writes in both the in-memory and SQLAlchemy repositories.

## Officialization quality gate

A Concept can become `official` only when all required checks pass:

```text
reviewer is authorized
AND Concept has reviewed EvidenceFragments or valid DecisionRecords
AND referenced EvidenceFragments exist
AND referenced DecisionRecords exist
AND EvidenceFragments are reviewed
AND EvidenceFragments are fresh or aging
AND EvidenceFragments have valid Artifact and DataSource references
AND production mode excludes non-production source evidence
AND source status is connected
```

## Serving contract

Grounded context must never invent official meaning.

```text
No matching Concept                         → unsupported
Concept with no eligible evidence           → unsupported
Candidate with reviewed evidence            → evidence_supported
Candidate with unreviewed evidence          → partially_supported
Official with reviewed fresh/aging evidence → official
Official with stale evidence                → stale
Official with unknown/mixed freshness       → partially_supported
Conflicted Concept                          → conflicted
```

## Current persistence slice

The backend now has a PostgreSQL persistence foundation:

```text
SQLAlchemy models
Alembic migration
PostgreSQL extensions: pgcrypto, citext, vector
Normalized Concept/DecisionRecord evidence join tables
Artifact idempotency constraint
SQLAlchemy transaction rollback tests
Offline Alembic SQL rendering
```

The next backend step is live PostgreSQL CI plus real connector persistence, not UI.

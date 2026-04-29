# 01 — API Contract

All public MVP endpoints use JSON and lower camel case fields.

## Health

### `GET /healthz`

```json
{
  "status": "ok",
  "service": "cornerstone-backend",
  "version": "0.9.0"
}
```

## Source Studio

### `GET /v1/sources`

Returns source runtime state and honest production onboarding state.

```json
{
  "productionEnabled": true,
  "hasRealSources": false,
  "onboardingRequired": true,
  "message": "No production data sources are connected yet. Connect a real source to create artifacts and evidence.",
  "sources": []
}
```

Rules:

- `pending_auth` does not count as a real connected source.
- `productionEnabled=false` never counts as a production source.
- Degraded sources may be visible but are not treated as healthy officialization sources.

### `POST /v1/sources`

Creates **manual sources only**. Provider-backed sources such as Notion must be created through the connector connection-intent and OAuth callback flow. This avoids fake authorized/connected provider sources.

```json
{
  "type": "manual",
  "name": "Manual Pilot",
  "productionEnabled": true
}
```

Behavior:

- Manual source becomes `connected` and may use `/v1/manual-sources/{sourceId}/sync`.
- Provider-backed source creation returns `409 Conflict`.
- `/v1/sources/{sourceId}/oauth/complete` does not exist. OAuth completion is handled only by `/v1/oauth/{provider}/callback`.

### `POST /v1/manual-sources/{sourceId}/sync`

Syncs explicitly supplied manual source objects and creates Artifacts/EvidenceFragments. Provider-backed sources such as Notion must use connector discovery, selection, and sync-job APIs.

```json
{
  "objects": [
    {
      "sourceExternalId": "page-123",
      "title": "Product Principles",
      "content": "Official means reviewed. EvidenceFragments must preserve provenance.",
      "sourceUrl": "https://notion.example/page-123",
      "sourceUpdatedAt": "2026-04-24T00:00:00Z"
    }
  ]
}
```

Response includes `dataSource`, `artifacts`, and `evidenceFragments`.

Sync rules:

- Source must be type `manual`.
- Source must be syncable: `connected`, `sync_pending`, `degraded`, or `stale`.
- Same `dataSourceId + sourceExternalId + rawContentHash` reuses the existing Artifact and EvidenceFragments.
- Failed extraction rolls back newly written Artifacts/EvidenceFragments.
- Failed sync after previous success marks the source `degraded` and freshness `unknown`.

## Connector discovery and selection

### `GET /v1/connectors`

Returns connector definitions. Provider definitions distinguish objects that can be discovered from objects that can be ingested. For Notion in v0.8.2:

```json
{
  "provider": "notion",
  "supportedObjects": ["page"],
  "discoverableObjects": ["page", "database", "data_source"],
  "ingestibleObjects": ["page"]
}
```

### `GET /v1/sources/{sourceId}/objects`

Returns discovered provider objects and selection counts. Each object exposes whether it can be selected for ingestion.

```json
{
  "datasourceId": "source-id",
  "totalCount": 2,
  "accessibleCount": 2,
  "syncableCount": 1,
  "selectedCount": 0,
  "objects": [
    {
      "externalId": "notion-page-1",
      "objectType": "page",
      "accessState": "accessible",
      "ingestionSupported": true,
      "ingestionUnsupportedReason": null
    },
    {
      "externalId": "notion-database-1",
      "objectType": "database",
      "accessState": "accessible",
      "ingestionSupported": false,
      "ingestionUnsupportedReason": "Notion database ingestion is not implemented in this backend slice. Select Notion pages instead."
    }
  ]
}
```

### `PUT /v1/sources/{sourceId}/selections`

Saves the source selection used by connector sync jobs. Selection is allowed only for objects that are both accessible and ingestion-supported.

Rules:

- `selected_only` requires at least one selected ID.
- `all_accessible` expands only to accessible and ingestion-supported objects.
- `workspace_limited` returns `409` until workspace selection semantics are implemented.
- Discoverable-but-unsupported objects, such as Notion databases/data_sources in v0.8.2, return `409 unsupported_object_type`.


## Sync jobs and worker leases

### `POST /v1/sources/{sourceId}/sync-jobs`

Queues a connector sync job after source selection. In v0.8.5 and later, scheduled jobs may also carry `scheduleId` and `enqueueKey` for due-window idempotency.

Important `SyncJob` fields:

```json
{
  "status": "queued",
  "attemptCount": 0,
  "maxAttempts": 3,
  "leaseOwner": null,
  "leaseAcquiredAt": null,
  "leaseExpiresAt": null,
  "scheduleId": null,
  "enqueueKey": null
}
```



### `POST /v1/sync-jobs/{sync_job_id}/heartbeat`

Refreshes an active sync-job lease.

Only the current `leaseOwner` can refresh a lease.

```json
{
  "workerId": "worker-a",
  "leaseSeconds": 300
}
```

Response:

```json
{
  "job": {
    "status": "running",
    "leaseOwner": "worker-a",
    "leaseHeartbeatAt": "2026-04-26T00:00:00Z",
    "leaseExpiresAt": "2026-04-26T00:05:00Z"
  },
  "events": [
    {
      "eventType": "sync.lease_heartbeat"
    }
  ]
}
```

A wrong worker receives `409 Conflict`.


### `POST /v1/sync-worker/run`

Runs queued or due retry-waiting jobs through an explicit worker claim.

```json
{
  "maxJobs": 10,
  "includeNotReady": false,
  "workerId": "worker-a",
  "leaseSeconds": 300
}
```

Claim rules:

- Claimable jobs are `queued` or due `retry_waiting`.
- Claim changes the job to `running`, increments `attemptCount`, and stores lease metadata.
- The worker records `sync.job_claimed` before provider work.
- `succeeded`, `failed`, `retry_waiting`, and `cancelled` clear the lease fields.
- Another worker cannot process a job already claimed in the same store.

### `POST /v1/sync-jobs/{syncJobId}/run`

Runs one job directly. Query parameters:

```text
includeNotReady=false
workerId=api-job-runner
leaseSeconds=300
```

### `POST /v1/sync-scheduler/run`

Enqueues due schedule jobs. Scheduled jobs receive an `enqueueKey` in this form:

```text
sync-schedule:{scheduleId}:{nextRunAt}
```

The PostgreSQL schema has a unique index on `sync_jobs.enqueue_key`. v0.9.0 must validate that this prevents duplicate scheduled enqueue under real concurrent PostgreSQL transactions.

## Artifacts

### `GET /v1/artifacts?dataSourceId={id}`

Returns captured source snapshots.

Artifact fields include:

```json
{
  "id": "artifact-id",
  "datasourceId": "source-id",
  "sourceType": "manual",
  "sourceExternalId": "doc-1",
  "sourceUrl": "https://example.internal/doc-1",
  "title": "Cornerstone Overview",
  "rawContentHash": "sha256...",
  "capturedAt": "2026-04-25T00:00:00Z",
  "sourceUpdatedAt": null,
  "freshnessState": "fresh",
  "extractionStatus": "complete"
}
```

## Evidence

### `GET /v1/evidence?artifactId={id}`

Returns EvidenceFragments.

Required provenance fields:

```json
{
  "dataSourceId": "source-id",
  "sourceType": "manual",
  "sourceExternalId": "doc-1",
  "sourceUrl": "https://example.internal/doc-1",
  "artifactTitle": "Cornerstone Overview",
  "capturedAt": "2026-04-25T00:00:00Z",
  "sourceUpdatedAt": null,
  "quoteRange": {
    "startOffset": 0,
    "endOffset": 48
  }
}
```

### `POST /v1/evidence/{evidenceFragmentId}/review`

Marks evidence as reviewed or rejected.

```json
{
  "trustState": "reviewed",
  "reviewedBy": "reviewer@example.com"
}
```

Rules:

- `trustState` must be `reviewed` or `rejected`.
- Reviewer must be in the configured allow-list.
- Review action creates an audit event.

## DecisionRecords

### `GET /v1/decision-records`

Lists DecisionRecords.

### `GET /v1/decision-records/{decisionRecordId}`

Reads a DecisionRecord.

### `POST /v1/decision-records`

Creates a DecisionRecord backed by reviewed production-eligible evidence.

```json
{
  "title": "Define Cornerstone",
  "decision": "Cornerstone is the shared organizational context layer.",
  "reason": "This definition appears in reviewed source evidence.",
  "alternativesConsidered": [],
  "decidedBy": "reviewer@example.com",
  "evidenceFragmentIds": ["evidence-id"],
  "affectedConceptIds": []
}
```

Rules:

- Decider must be authorized.
- At least one EvidenceFragment is required.
- Evidence must be reviewed and fresh/aging.
- Evidence must come from a production-enabled connected source in production mode.

## Concepts

### `GET /v1/concepts`

Lists Concepts.

### `GET /v1/concepts/{conceptId}`

Reads a Concept.

### `POST /v1/concepts`

Creates a Concept candidate.

```json
{
  "name": "Cornerstone",
  "shortDefinition": "Shared organizational context layer.",
  "evidenceFragmentIds": ["evidence-id"],
  "decisionRecordIds": [],
  "createdBy": "reviewer@example.com"
}
```

Rules:

- Referenced EvidenceFragment IDs must exist.
- Referenced DecisionRecord IDs must exist.

### `POST /v1/concepts/{conceptId}/officialize`

Attempts to mark a Concept official.

```json
{
  "reviewedBy": "reviewer@example.com"
}
```

Blocks with `409` when trust requirements are not met. Blocks with `403` when reviewer is unauthorized.

## Grounded context

### `GET /v1/context/query?q=What%20is%20Cornerstone%3F`

Returns the shared human/AI grounded context contract.

```json
{
  "responseId": "response-id",
  "query": "What is Cornerstone?",
  "answer": "Cornerstone: Shared organizational context layer.",
  "trustLabel": "official",
  "concepts": [
    {"id": "concept-id", "name": "Cornerstone", "status": "official"}
  ],
  "relations": [],
  "decisions": [],
  "evidence": [
    {
      "evidenceFragmentId": "evidence-id",
      "artifactId": "artifact-id",
      "text": "Cornerstone is a shared organizational context layer.",
      "sourceUrl": "https://example.internal/doc-1",
      "artifactTitle": "Cornerstone Overview",
      "capturedAt": "2026-04-26T00:00:00Z",
      "sourceUpdatedAt": null,
      "freshnessState": "fresh",
      "trustState": "reviewed",
      "supports": [
        {
          "entityType": "concept",
          "entityId": "concept-id",
          "relationship": "supports_concept_definition"
        }
      ],
      "isValid": true,
      "validityErrors": []
    }
  ],
  "freshness": {
    "state": "fresh",
    "staleEvidenceCount": 0,
    "unknownEvidenceCount": 0
  },
  "limitations": [],
  "generatedAt": "2026-04-26T00:00:00Z",
  "officialAnswerAvailable": true
}
```

Trust-label rules:

- `official` requires an official Concept plus valid reviewed fresh/aging supporting evidence.
- `evidence_supported` means valid reviewed evidence exists, but the answer is not official.
- `partially_supported` means evidence is unreviewed, freshness is unknown/mixed, or support is incomplete.
- `stale` means at least one cited EvidenceFragment is stale.
- `conflicted` means the matched Concept or one cited EvidenceFragment is conflicted.
- `unsupported` means no valid eligible evidence can support the response.

Citation guardrails:

- Rejected EvidenceFragments are excluded.
- Missing Artifact/DataSource references are excluded.
- Invalid provenance is excluded.
- Non-production or unhealthy source evidence is excluded in production mode.
- Evidence citations expose exactly which Concept, ConceptRelation, DecisionRecord, or evidence-only path they support.

Rules:

- Unsupported is a valid response.
- Evidence-only responses are allowed, but they are never official.
- Non-production evidence is excluded in production mode.
- Unknown/mixed freshness cannot produce an `official` trust label.
- OpenAPI snapshot tests protect this response contract.

## Audit events

### `GET /v1/audit-events`

Returns audit events for review and officialization transitions.

Current audit event types:

```text
evidence.reviewed
decision_record.created
concept.officialization_blocked
concept.officialized
```

## Runtime Configuration Contract

The API process fails closed when `PRODUCTION_MODE=true` and unsafe runtime defaults are present.

Startup is blocked if production mode uses:

```text
PERSISTENCE_BACKEND=memory
DATABASE_URL pointing at localhost/default cornerstone credentials
NOTION_MOCK_EXTERNAL_API=true
CONNECTOR_ENCRYPTION_SECRET=local-dev-only-change-me-secret
CONNECTOR_OAUTH_CALLBACK_URL using localhost or non-HTTPS
placeholder reviewers such as system or reviewer@example.com
missing required Postgres extensions: pgcrypto,citext,vector
```

Contract tests may still call `create_app(store=...)` with an explicit in-memory or SQLite store. That bypass is not used by normal `uvicorn cornerstone.main:app` startup or the external sync worker CLI.

## v0.10.0 Evidence Review and Officialization Additions

### `GET /v1/evidence/review-queue`

Returns reviewer-ready EvidenceFragments with linked Artifact/DataSource context, linked Concept/Decision IDs, and suggested actions.

Query parameters:

```text
trustState=unreviewed|reviewed|rejected|conflicted
dataSourceId=<source id>
freshnessState=fresh|aging|stale|unknown
fragmentType=definition|decision|policy|requirement|example|claim|open_question
limit=1..200
```

### `POST /v1/evidence/{evidenceFragmentId}/concept-candidates`

Creates a Concept candidate/reviewing item from an EvidenceFragment. Requires an authorized reviewer.

### `GET /v1/concept-relations`

Lists ConceptRelations. Optional query parameter:

```text
conceptId=<source or target Concept ID>
```

### `POST /v1/concept-relations`

Creates a candidate ConceptRelation. Requires source and target Concepts to exist and the actor to be an authorized reviewer.

### `POST /v1/concept-relations/{relationId}/officialize`

Marks a ConceptRelation official only when:

```text
- Source Concept is official.
- Target Concept is official.
- Relation has reviewed eligible evidence or a valid DecisionRecord.
- Reviewer is authorized.
```

Blocked attempts are audit logged.


## v0.12.0 Evaluation API

```http
POST /v1/evaluations/tasks
GET  /v1/evaluations/tasks
GET  /v1/evaluations/tasks/{task_id}
POST /v1/evaluations/tasks/{task_id}/run
POST /v1/evaluations/run
GET  /v1/evaluations/results
GET  /v1/evaluations/results/{result_id}
GET  /v1/evaluations/summary
```

Evaluation tasks assert expected trust labels, answer fragments, freshness, required Concepts/DecisionRecords/EvidenceFragments, and whether evidence/official answers are required. Results preserve the grounded response snapshot and each PRD quality-gate boolean.

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

### `POST /v1/manual-sources/{sourceId}/uploads`

`v1.3.1` adds multipart manual file upload ingestion. The endpoint accepts one or more UTF-8 text-like files using the form field name `files`. Uploaded files are converted into provider-neutral `SourceObject` records and then passed through the same Artifact/EvidenceFragment pipeline as `/sync`.

Supported file intent:

```text
- UTF-8 text-like files only
- common suffixes such as .txt, .md, .csv, .json, .jsonl, .yaml, .yml, .xml, .html, .log, .rst
- `text/*` MIME types and a small set of structured text MIME types
```

Rejected in `v1.3.1`:

```text
- PDFs
- Office documents
- images
- binary files
- non-UTF-8 text
- empty files
```

Example:

```bash
curl -X POST "http://localhost:8000/v1/manual-sources/$SOURCE_ID/uploads" \
  -F "files=@settlement.md;type=text/markdown"
```

Response is the same `SyncSourceResponse` used by `/sync`. For an uploaded file, the Artifact uses:

```json
{
  "sourceExternalId": "manual-upload:settlement.md",
  "sourceObjectType": "uploaded_file",
  "providerMetadata": {
    "uploadKind": "manual_file",
    "fileName": "settlement.md",
    "contentType": "text/markdown",
    "sizeBytes": 1234,
    "encoding": "utf-8"
  }
}
```

Configured limits:

```text
MANUAL_UPLOAD_MAX_FILE_COUNT=10
MANUAL_UPLOAD_MAX_FILE_BYTES=5242880
```

### `POST /v1/manual-sources/{sourceId}/uploads/text`

`v1.3.1` also adds a JSON endpoint for pasted/manual text uploads. This is useful for UI clients that collect text directly instead of uploading a local file.

```json
{
  "objects": [
    {
      "title": "Settlement Notes",
      "content": "Settlement is the process of finalizing obligations.",
      "sourceExternalId": "optional-stable-id",
      "sourceUrl": "https://example.internal/optional",
      "providerMetadata": {"importBatch": "pilot"}
    }
  ]
}
```

If `sourceExternalId` is omitted, the backend uses `manual-upload:text:{title}`. The Artifact uses `sourceObjectType=uploaded_text` and `providerMetadata.uploadKind=manual_text`.

Manual upload rules:

- Upload endpoints are manual-source-only. Provider-backed sources still use connector discovery, selection, and sync jobs.
- Upload endpoints do not create Concepts, Relations, ConceptCandidates, or RelationCandidates.
- Uploaded text only creates Artifacts and EvidenceFragments. Review and officialization remain separate.
- Same `dataSourceId + sourceExternalId + rawContentHash` reuses existing Artifacts and EvidenceFragments.

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

## v1.3.0 Ontology Graph API

`v1.3.0` adds ontology graph serving over existing Concepts and ConceptRelations. It does not run LLM extraction or create candidates.

### `GET /v1/ontology/search`

Searches Concepts by primary name or alias.

Query parameters:

```text
q=<required search term>
mode=official|candidate|mixed    # default: official
limit=1..50                      # default: 10
```

Default `official` mode returns only official Concepts. Results include `matchedBy`, `matchedValue`, and a lexical score so clients can explain why a Concept matched.

### `GET /v1/ontology/graph`

Returns a UI-ready depth-limited graph for a Concept name or alias.

Query parameters:

```text
concept=<required concept name or alias>
depth=0|1                        # default: 1; depth above 1 is rejected in v1.3.0
mode=official|candidate|mixed    # default: official
```

Official mode returns only official Concepts and official ConceptRelations. The response includes:

```text
focusConcept
nodes
edges
evidence citations
freshness summary
trustLabel
limitations
officialGraphAvailable
```

If no matching Concept exists in the requested mode, the endpoint returns `trustLabel=unsupported` with an empty graph and limitations.


## v1.3.1 Manual Upload API

`v1.3.1` adds manual uploaded data ingestion as a source-ingestion feature, not an ontology-construction feature. It lets a user provide settlement notes or other domain text directly, then produces Artifacts and EvidenceFragments that future versions can use for candidate ontology extraction.

The release keeps these boundaries explicit:

```text
Manual upload → Artifact → EvidenceFragment
Manual upload ↛ ConceptCandidate
Manual upload ↛ RelationCandidate
Manual upload ↛ official Concept/Relation
```

This preserves the reviewed Single Source of Truth boundary defined in `v1.2.1`.


## v1.4.0 Ontology Extraction API

`v1.4.0` adds candidate-only ontology extraction over existing EvidenceFragments.

The release keeps the trust boundary explicit:

```text
Extraction → ConceptCandidate
Extraction → RelationCandidate
Extraction ↛ official Concept
Extraction ↛ official ConceptRelation
Extraction ↛ official graph mutation
```

### `POST /v1/ontology/extraction-runs`

Creates an extraction run from selected evidence or artifacts.

Request:

```json
{
  "evidenceFragmentIds": [],
  "artifactIds": ["artifact-id"],
  "focusConcept": "settlement",
  "provider": "local_rule_based",
  "modelName": "local-rule-based-ontology-extractor-v1.4.0",
  "promptVersion": "ontology-extraction-v1.4.0",
  "requestedBy": "reviewer@example.com",
  "maxEvidenceFragments": 50
}
```

Rules:

```text
- request must include evidenceFragmentIds or artifactIds
- each candidate must reference evidenceFragmentIds
- candidates are created with status=pending
- official Concepts and Relations are not created
```

Response:

```text
run
conceptCandidates
relationCandidates
```

### `GET /v1/ontology/extraction-runs`

Lists extraction runs.

### `GET /v1/ontology/extraction-runs/{runId}`

Returns a single extraction run and its ConceptCandidates and RelationCandidates.

### `GET /v1/ontology/concept-candidates`

Lists ConceptCandidates.

Query parameters:

```text
runId=<optional extraction run id>
status=pending|approved|rejected|merged
```

### `GET /v1/ontology/relation-candidates`

Lists RelationCandidates.

Query parameters:

```text
runId=<optional extraction run id>
status=pending|approved|rejected|merged
```

The bundled `local_rule_based` provider is deterministic and offline. Live external LLM providers are deferred until the candidate contract and review workflow are stable.


## v1.5.0 Ontology Candidate Review API

`v1.5.0` adds the review workflow for ontology candidates created by `v1.4.0` extraction runs.

The release keeps the trust boundary explicit:

```text
ConceptCandidate + reviewer approval + reviewed evidence → official Concept
RelationCandidate + reviewer approval + reviewed evidence → official ConceptRelation
Candidate rejection → no official object
Candidate merge → existing official object is updated only through the same evidence gates
Candidate edit → pending candidate only
Candidate review ↛ live external LLM call
Candidate review ↛ unsupported official graph mutation
```

All review endpoints require an authorized reviewer. In production mode, reviewer identity is checked against the configured reviewer allow-list.

### `PATCH /v1/ontology/concept-candidates/{candidateId}`

Edits a pending ConceptCandidate before review.

Request:

```json
{
  "editedBy": "reviewer@example.com",
  "name": "Settlement",
  "aliases": ["payment settlement"],
  "proposedDefinition": "The process of finalizing transaction obligations.",
  "conceptType": "process",
  "evidenceFragmentIds": ["evidence-id"],
  "confidence": 0.89,
  "rationale": "Reviewed wording from source evidence."
}
```

Rules:

```text
- candidate must be status=pending
- editedBy must be authorized
- replacement evidenceFragmentIds must exist
- the endpoint does not create a Concept
```

### `POST /v1/ontology/concept-candidates/{candidateId}/approve`

Promotes a pending ConceptCandidate into an official Concept through the existing officialization gate.

Request:

```json
{
  "reviewedBy": "reviewer@example.com",
  "name": "Settlement",
  "aliases": ["payment settlement"],
  "shortDefinition": "Settlement is the process of finalizing transaction obligations.",
  "owner": "Finance Operations",
  "reviewNote": "Approved for settlement ontology pilot."
}
```

Rules:

```text
- candidate must be status=pending
- reviewedBy must be authorized
- candidate evidence must exist and satisfy officialization requirements
- concept name and aliases must not overlap another Concept
- response includes the promoted official Concept and audit event ids
```

### `POST /v1/ontology/concept-candidates/{candidateId}/reject`

Rejects a pending ConceptCandidate.

Request:

```json
{
  "reviewedBy": "reviewer@example.com",
  "reviewNote": "Too ambiguous for the official graph."
}
```

Rules:

```text
- candidate must be status=pending
- reviewedBy must be authorized
- no Concept is created
- rejected candidates cannot later be approved without a new extraction/edit cycle
```

### `POST /v1/ontology/concept-candidates/{candidateId}/merge`

Merges a pending ConceptCandidate into an existing Concept.

Request:

```json
{
  "reviewedBy": "reviewer@example.com",
  "targetConceptId": "concept-id",
  "aliases": ["settlements"],
  "appendEvidence": true,
  "reviewNote": "Merged duplicate settlement wording."
}
```

Rules:

```text
- candidate must be status=pending
- target Concept must exist
- aliases are normalized and deduplicated
- appended evidence must still satisfy officialization if the target Concept is official
- response includes the updated target Concept and audit event ids
```

### `PATCH /v1/ontology/relation-candidates/{candidateId}`

Edits a pending RelationCandidate before review.

Request:

```json
{
  "editedBy": "reviewer@example.com",
  "sourceName": "Clearing",
  "targetName": "Settlement",
  "sourceConceptId": "clearing-concept-id",
  "targetConceptId": "settlement-concept-id",
  "relationType": "precedes",
  "evidenceFragmentIds": ["evidence-id"],
  "confidence": 0.84,
  "rationale": "Evidence says clearing happens before settlement."
}
```

Rules:

```text
- candidate must be status=pending
- editedBy must be authorized
- replacement evidenceFragmentIds must exist
- explicit source/target Concept ids must exist when provided
- the endpoint does not create a ConceptRelation
```

### `POST /v1/ontology/relation-candidates/{candidateId}/approve`

Promotes a pending RelationCandidate into an official ConceptRelation through the existing officialization gate.

Request:

```json
{
  "reviewedBy": "reviewer@example.com",
  "sourceConceptId": "clearing-concept-id",
  "targetConceptId": "settlement-concept-id",
  "relationType": "precedes",
  "reviewNote": "Approved settlement graph edge."
}
```

Rules:

```text
- candidate must be status=pending
- reviewedBy must be authorized
- source and target Concepts must resolve to existing Concepts
- candidate evidence must exist and satisfy officialization requirements
- duplicate source/target/relationType edges are rejected; use merge instead
- response includes the promoted official ConceptRelation and audit event ids
```

### `POST /v1/ontology/relation-candidates/{candidateId}/reject`

Rejects a pending RelationCandidate.

Request:

```json
{
  "reviewedBy": "reviewer@example.com",
  "reviewNote": "The cited evidence does not support this edge."
}
```

Rules:

```text
- candidate must be status=pending
- reviewedBy must be authorized
- no ConceptRelation is created
```

### `POST /v1/ontology/relation-candidates/{candidateId}/merge`

Merges a pending RelationCandidate into an existing ConceptRelation.

Request:

```json
{
  "reviewedBy": "reviewer@example.com",
  "targetRelationId": "relation-id",
  "appendEvidence": true,
  "reviewNote": "Merged duplicate relation candidate."
}
```

Rules:

```text
- candidate must be status=pending
- target ConceptRelation must exist
- appended evidence must still satisfy officialization if the target Relation is official
- response includes the updated target Relation and audit event ids
```

### Review API error behavior

```text
403 unauthorized reviewer
404 candidate, Concept, Relation, or EvidenceFragment not found
409 candidate is not pending, officialization fails, duplicate edge exists, or merge/approve would violate ontology rules
422 request shape is invalid
```

## v1.6.0 Explainable Graph Serving API

`v1.6.0` enhances ontology graph serving without changing the Single Source of Truth boundary. The graph still comes from existing Concepts and ConceptRelations. The response now explains why the graph is or is not official.

### `GET /v1/ontology/graph`

Existing endpoint, additive response fields.

Query parameters remain:

```text
concept=<required concept name or alias>
depth=0|1                        # default: 1; depth above 1 is rejected
mode=official|candidate|mixed    # default: official
```

New top-level response fields:

```text
supportSummary
candidateSummary
explanation
```

New node fields:

```text
reviewProvenance
supportSummary
explanation
```

New edge fields:

```text
sourceConceptName
targetConceptName
focusDirection
reviewProvenance
supportSummary
explanation
```

New citation fields:

```text
dataSourceId
sourceType
sourceExternalId
reviewedBy
reviewedAt
```

Boundary rules:

```text
- Official mode still returns only official Concepts and official ConceptRelations.
- Pending ConceptCandidates and RelationCandidates are summarized in candidateSummary only.
- CandidateSummary does not promote or officialize anything.
- Graph serving does not call an LLM.
- Graph serving does not infer missing Relations.
```

### `GET /v1/ontology/explain`

Returns the same response model as `/v1/ontology/graph`, intended for clients that want explanation-first graph serving.

Query parameters:

```text
concept=<required concept name or alias>
depth=0|1                        # default: 1
mode=official|candidate|mixed    # default: official
```

Response model:

```text
OntologyGraphResponse
```

The response includes:

```text
- graph nodes and edges
- evidence citations
- support summary
- candidate summary
- explanation object
- limitations
```

### Explanation object

```json
{
  "summary": "Settlement graph served 2 node(s), 1 direct relation(s), and 1 serving citation(s) at depth 1.",
  "ssotStatus": "official_ssot",
  "trustReason": "The focus Concept, visible neighbor Concepts, visible Relations, and serving citations are reviewed and official.",
  "graphScope": "Depth 1 includes the focus Concept and directly connected Concepts only.",
  "evidencePolicy": "Citations include serving-eligible EvidenceFragments for the focus Concept, visible neighbor Concepts, visible ConceptRelations, and linked DecisionRecords.",
  "candidateBoundary": "Candidate objects are excluded from official graph mode.",
  "reviewSummary": "2/2 node(s) official; 1/1 edge(s) official; 1/1 serving citation(s) reviewed; freshness=fresh.",
  "recommendedNextActions": ["Use the cited evidence and review provenance to audit the graph before downstream reuse."]
}
```

## v1.7.0 Ontology Evaluation API

`v1.7.0` adds read-only evaluation endpoints for ontology graph serving. The evaluator calls the existing ontology graph service, scores the returned `OntologyGraphResponse`, persists the result, and reports graph quality metrics.

The evaluator does not mutate Concepts, ConceptRelations, candidates, Artifacts, EvidenceFragments, or source data.

```http
POST /v1/evaluations/ontology/tasks
GET  /v1/evaluations/ontology/tasks
GET  /v1/evaluations/ontology/tasks/{taskId}
POST /v1/evaluations/ontology/tasks/{taskId}/run
POST /v1/evaluations/ontology/run
GET  /v1/evaluations/ontology/results
GET  /v1/evaluations/ontology/results/{resultId}
GET  /v1/evaluations/ontology/summary
```

### Create ontology graph evaluation task

```http
POST /v1/evaluations/ontology/tasks
```

Request body fields:

```json
{
  "name": "settlement official graph smoke test",
  "conceptQuery": "settlement",
  "mode": "official",
  "depth": 1,
  "expectedTrustLabel": "official",
  "expectedFreshnessState": "fresh",
  "requiredConceptIds": [],
  "requiredRelationIds": [],
  "requiredEvidenceFragmentIds": [],
  "requireOfficialGraph": true,
  "requireEvidence": true,
  "minEvidenceCount": 1,
  "minNodeCount": 1,
  "minEdgeCount": 0,
  "requireReviewProvenance": true,
  "maxPendingCandidateCount": null,
  "tags": ["smoke"],
  "createdBy": "operator@example.com",
  "metadata": {}
}
```

The task must include at least one explicit success condition. Unsupported graph tasks must explicitly opt out of official/evidence requirements.

### Run ontology graph evaluation task

```http
POST /v1/evaluations/ontology/tasks/{taskId}/run
```

Request body:

```json
{
  "evaluatedBy": "operator@example.com"
}
```

Response body includes:

```json
{
  "graphFound": true,
  "graphDepthRespected": true,
  "nodeRequirementsMet": true,
  "edgeRequirementsMet": true,
  "evidenceValid": true,
  "provenancePresent": true,
  "trustLabelCorrect": true,
  "freshnessPolicyRespected": true,
  "officialGraphSafe": true,
  "candidateBoundaryRespected": true,
  "relationIntegrityValid": true,
  "citationValidityRate": 1.0,
  "success": true,
  "failureReasons": []
}
```

### Run multiple ontology graph evaluation tasks

```http
POST /v1/evaluations/ontology/run
```

When `taskIds` is omitted or empty, all ontology graph evaluation tasks are run.

### List and read ontology graph evaluation results

```http
GET /v1/evaluations/ontology/results
GET /v1/evaluations/ontology/results?taskId={taskId}
GET /v1/evaluations/ontology/results/{resultId}
```

### Ontology graph evaluation summary

```http
GET /v1/evaluations/ontology/summary
```

The summary reports:

```text
ontologyGraphTaskSuccessRate
evidenceValidityRate
provenanceCoverageRate
citationValidityRate
officialGraphSafetyRate
candidateBoundaryRate
relationIntegrityRate
trustLabelCorrectnessRate
freshnessComplianceRate
```

## v1.8.0 Connector-driven Re-extraction API

`v1.8.0` adds API support for queuing and running ontology re-extraction after manual or connector source changes.

The re-extraction layer is candidate-only. It creates `OntologyExtractionRun`, `ConceptCandidate`, and `RelationCandidate` records. It does not create official Concepts, official ConceptRelations, or official graph edges.

```http
POST /v1/ontology/re-extraction-runs
GET  /v1/ontology/re-extraction-runs
GET  /v1/ontology/re-extraction-runs/{runId}
POST /v1/ontology/re-extraction-runs/{runId}/run
```

### Create ontology re-extraction run

```http
POST /v1/ontology/re-extraction-runs
```

Request body:

```json
{
  "datasourceId": "source-id",
  "syncJobId": null,
  "artifactIds": [],
  "focusConcept": "settlement",
  "trigger": "manual_request",
  "createdBy": "operator@example.com",
  "reason": "Refresh ontology candidates after source change.",
  "runInline": false
}
```

Exactly one explicit scope is recommended; the request must include at least one of:

```text
datasourceId
syncJobId
artifactIds
```

If `runInline` is true, the backend immediately creates the downstream ontology extraction run and candidate objects. Inline execution still does not promote candidates or mutate the official graph.

### List ontology re-extraction runs

```http
GET /v1/ontology/re-extraction-runs
GET /v1/ontology/re-extraction-runs?datasourceId={sourceId}
GET /v1/ontology/re-extraction-runs?status=queued
```

### Read ontology re-extraction run

```http
GET /v1/ontology/re-extraction-runs/{runId}
```

The response includes:

```text
run
extractionRuns
conceptCandidates
relationCandidates
```

A queued run may not have extraction output yet. A completed run includes linked extraction runs and generated candidates.

### Run ontology re-extraction

```http
POST /v1/ontology/re-extraction-runs/{runId}/run
```

Request body:

```json
{
  "requestedBy": "operator@example.com"
}
```

Running a re-extraction run is allowed for queued or failed runs. It uses the run's Artifact/Evidence scope and creates a downstream `OntologyExtractionRun`.

### Extended manual source sync and upload fields

`POST /v1/manual-sources/{sourceId}/sync` and `POST /v1/manual-sources/{sourceId}/uploads/text` now accept:

```text
queueOntologyReExtraction       # default true
runOntologyReExtractionInline   # default false
ontologyFocusConcept            # optional
```

`POST /v1/manual-sources/{sourceId}/uploads` accepts equivalent query parameters.

### Extended sync job fields

`POST /v1/sources/{sourceId}/sync-jobs` now accepts:

```text
queueOntologyReExtraction
runOntologyReExtractionInline
ontologyFocusConcept
```

Successful connector sync worker runs queue re-extraction when new or changed Artifacts are created.

### Extended sync response fields

`SyncSourceResponse` now includes:

```text
artifactChangedCount
createdArtifactIds
reusedArtifactIds
changedArtifactIds
ontologyReextractionRunId
ontologyReextractionStatus
```

Boundary rules:

```text
- New or changed Artifacts may queue re-extraction.
- Exact reused Artifacts do not queue re-extraction by default.
- Re-extraction output remains candidate-only.
- officialGraphMutated must remain false.
- Candidate approval still requires v1.5.0 review endpoints.
```

## Ontology Proof Runs v1.9.0

`v1.9.0` adds an operator-facing proof/checklist endpoint for the ontology Single Source of Truth loop.

```http
POST /v1/ontology/proof-runs
```

The endpoint can be used in two modes:

```text
1. dryRun=true: return the planned checklist without mutating the store.
2. confirmMutation=true: execute the proof loop and create explicit proof data.
```

### Request

```json
{
  "focusConcept": "Settlement",
  "reviewer": "reviewer@example.com",
  "createdBy": "operator@example.com",
  "sourceName": "v1.9.0 ontology proof - Settlement",
  "seedContent": null,
  "dryRun": false,
  "confirmMutation": true,
  "runEvaluation": true
}
```

### Response

```json
{
  "status": "passed",
  "focusConcept": "Settlement",
  "reviewer": "reviewer@example.com",
  "sourceId": "source-id",
  "artifactIds": ["artifact-id"],
  "evidenceFragmentIds": ["evidence-id"],
  "reextractionRunId": "reextraction-id",
  "extractionRunIds": ["extraction-id"],
  "conceptCandidateIds": ["candidate-id"],
  "relationCandidateIds": ["relation-candidate-id"],
  "approvedConceptIds": ["concept-id"],
  "approvedRelationIds": ["relation-id"],
  "graphResponseId": "graph-response-id",
  "evaluationTaskId": "eval-task-id",
  "evaluationResultId": "eval-result-id",
  "summary": {
    "status": "passed",
    "requiredTotal": 8,
    "requiredPassed": 8,
    "requiredFailed": 0,
    "officialGraphAvailable": true,
    "officialGraphMutated": false,
    "evaluationSuccess": true
  },
  "checklist": [
    {
      "key": "create_manual_source",
      "status": "passed",
      "required": true
    }
  ]
}
```

### Checklist contract

The proof checklist includes these keys:

```text
create_manual_source
sync_manual_seed
run_reextraction
review_evidence
approve_concepts
approve_relations
serve_explainable_graph
run_ontology_evaluation
```

### Safety contract

```text
- dryRun=true creates no objects.
- confirmMutation=true is required for non-dry execution.
- reviewer must be authorized.
- evidence review is required before candidate approval.
- re-extraction must keep officialGraphMutated=false.
- official graph mutation happens only through candidate review/officialization gates.
- v1.9.0 proof runs are operator proof data, not hidden UI behavior.
```


## Ontology SSOT Readiness API (v2.0.0)

### Get ontology SSOT readiness

```http
GET /v1/ontology/ssot/readiness?focusConcept=settlement&depth=1&mode=official&includeGraph=false
```

Purpose: return a read-only release/operator checklist for whether the current backend state is ready to serve an official explainable ontology Single Source of Truth graph for a focus concept.

Query parameters:

```text
focusConcept: concept name or alias, default settlement
depth: 0 or 1, default 1
mode: official / candidate / mixed, default official
includeGraph: boolean, default false
```

Response includes:

```text
responseId
releaseVersion
focusConcept
mode
depth
status
officialGraphAvailable
officialGraphSafe
trustLabel
graphResponseId
nodeCount
edgeCount
evidenceCount
pendingConceptCandidateCount
pendingRelationCandidateCount
latestEvaluationResultId
latestEvaluationSuccess
evaluationSummary
checks
recommendedActions
graph
generatedAt
```

Boundary rules:

```text
- The endpoint is read-only.
- The endpoint does not create source data.
- The endpoint does not run extraction or re-extraction.
- The endpoint does not approve candidates.
- The endpoint does not mutate the official graph.
- The endpoint does not create evaluation tasks or results.
- The endpoint reports `passed` only when all required readiness checks pass.
```

## v2.0.2 Product Documentation API Contract Note

`v2.0.2` introduces no new public API endpoint, no request/response shape change, and no database migration.

The only API-visible metadata change is that `OntologySsotReadinessResponse.releaseVersion` now reports:

```text
2.0.2
```

The existing readiness endpoint remains:

```http
GET /v1/ontology/ssot/readiness?focusConcept=settlement&depth=1&mode=official&includeGraph=false
```

The endpoint remains read-only and must not run extraction, review candidates, or mutate the official graph.

## v2.0.3 Dependency-Complete Verification API Contract Note

`v2.0.3` introduces no new public API endpoint, no request/response shape change, and no database migration.

The only runtime-facing metadata change is that SSOT readiness responses report:

```text
2.0.3
```

The release adds dependency-complete verification tooling and CI hardening:

```text
scripts/run_dependency_complete_verification.py
scripts/run_dependency_complete_verification.sh
.github/workflows/dependency-complete-verification.yml
```

These tools verify the existing API contract; they do not extend it.

## v2.0.4 Forward Roadmap API Contract Note

`v2.0.4` is a documentation-first roadmap release. It adds planning documents for `v2.1.0` through `v2.5.0` and does not add, remove, or rename any public API endpoint.

Freeze result:

```text
No public API endpoint change.
No request schema change.
No response schema change except version metadata reporting 2.0.4.
No database migration.
No graph behavior change.
No extraction/review behavior change.
No connector behavior change.
No live LLM behavior change.
No frontend or integration package behavior change.
```

The existing SSOT readiness response continues to expose release metadata through:

```json
{
  "releaseVersion": "2.0.4"
}
```

## v2.5.0 External Integration Package API Contract Note

`v2.5.0` implements the forward roadmap through the external integration package path. It adds public API surface for live extraction selection, review-operator previews, graph visualization metadata, connector support visibility, and external integration consumption.

Added and changed endpoints:

```text
POST /v1/ontology/extraction-runs
GET  /v1/ontology/review-queue/summary
GET  /v1/ontology/concept-candidates/{candidateId}/preview
GET  /v1/ontology/relation-candidates/{candidateId}/preview
GET  /v1/connectors/support-matrix
GET  /v1/integration/package/manifest
GET  /v1/integration/ontology/{concept}
```

Key contract additions:

```text
CreateOntologyExtractionRunRequest.provider accepts live_llm.
OntologyGraphResponse includes visualization.
ConnectorSupportMatrixResponse lists support states and live-proof guard names.
IntegrationOntologyResponse wraps official graph, SSOT readiness, citations, trust state, unsupported state, and reviewGateBypassAllowed=false.
```

Freeze result:

```text
No database migration.
No graph depth increase above 1.
No automatic candidate approval.
No connector path mutates the official graph directly.
No frontend MVP.
No candidate bypass through integration endpoints.
```

The SSOT readiness response now exposes release metadata through:

```json
{
  "releaseVersion": "2.5.0"
}
```

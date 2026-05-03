# Cornerstone v1.8.0 — Connector-driven re-extraction

## Purpose

`v1.8.0` connects source changes to ontology extraction without weakening the Single Source of Truth boundary.

Before this release, an operator could ingest manual, Notion, or Google Drive source content and separately start ontology extraction. That was safe, but it left the operational loop incomplete: changed source artifacts did not create an explicit ontology re-extraction work item.

This release introduces a safe re-extraction queue:

```text
connector/manual source change
→ SourceObject
→ Artifact / EvidenceFragment
→ OntologyReExtractionRun queued
→ operator/worker runs re-extraction
→ OntologyExtractionRun
→ ConceptCandidates / RelationCandidates
→ human review remains required
→ official ontology graph changes only after review
```

The release is intentionally conservative. Re-extraction creates or refreshes **candidates only**. It does not update the official graph directly.

## Product goal

The product goal is to make the ontology graph maintainable as source data changes.

For the reference concept `settlement`, the desired operating behavior is:

```text
1. A user uploads or syncs new settlement source material.
2. Cornerstone captures new or changed Artifacts and EvidenceFragments.
3. Cornerstone records that ontology candidates should be refreshed for that changed source scope.
4. Re-extraction creates new ConceptCandidates and RelationCandidates.
5. Reviewers approve, reject, edit, or merge those candidates.
6. Only reviewed objects enter the official Single Source of Truth graph.
```

## Non-goals

`v1.8.0` does not do any of the following:

```text
- no automatic Concept approval
- no automatic Relation approval
- no automatic official graph mutation
- no bypass of evidence review
- no live external LLM provider
- no semantic/vector change detection
- no recursive graph expansion beyond depth 1
- no graph visualization UI
- no candidate auto-merge into existing official objects
- no deletion or rollback of previously official Concepts or Relations
```

## Trust boundary

The trust boundary remains unchanged:

```text
Raw source document ≠ Single Source of Truth
Connector sync result ≠ Single Source of Truth
LLM/extractor output ≠ Single Source of Truth
OntologyReExtractionRun ≠ Single Source of Truth
ConceptCandidate / RelationCandidate ≠ Single Source of Truth
Reviewed official ontology graph = Single Source of Truth
```

Re-extraction may create a better review queue, but it does not create truth.

## New domain object

### OntologyReExtractionRun

`OntologyReExtractionRun` records that one or more changed Artifacts should be reprocessed by the ontology extraction layer.

Important fields:

```text
id
sourceId / datasourceId
provider
trigger
status
createdBy
syncJobId
focusConcept
reason
sourceExternalIds
artifactIds
changedArtifactIds
evidenceFragmentIds
extractionRunIds
conceptCandidateCount
relationCandidateCount
warningCount
officialGraphMutated
error
createdAt
startedAt
completedAt
```

Statuses:

```text
queued
running
completed
skipped
failed
```

Triggers:

```text
connector_sync
scheduled_sync
manual_upload
manual_sync
webhook
manual_request
```

The `officialGraphMutated` flag must remain `false` for all v1.8.0 re-extraction runs.

## Source change behavior

The source sync pipeline now reports explicit artifact change metadata:

```text
artifactCreatedCount
artifactReusedCount
artifactChangedCount
createdArtifactIds
reusedArtifactIds
changedArtifactIds
```

Meaning:

```text
createdArtifactIds
  Artifact ids created by this sync/upload.

reusedArtifactIds
  Artifact ids reused because the same source object content hash already existed.

changedArtifactIds
  Created Artifact ids whose sourceExternalId previously existed with different content.
```

First-time artifacts are new and may queue re-extraction. `changedArtifactIds` specifically identifies updates to previously known source objects.

Exact content reuse does not queue re-extraction by default.

## Manual source behavior

Manual sync and manual upload now accept re-extraction controls.

Manual source sync request body:

```json
{
  "objects": [
    {
      "sourceExternalId": "settlement-doc",
      "title": "Settlement Notes",
      "content": "Settlement is the process of finalizing obligations. Clearing precedes settlement."
    }
  ],
  "queueOntologyReExtraction": true,
  "runOntologyReExtractionInline": false,
  "ontologyFocusConcept": "settlement"
}
```

Manual text upload request body:

```json
{
  "objects": [
    {
      "title": "Settlement Notes",
      "content": "Settlement is the process of finalizing obligations. Clearing precedes settlement."
    }
  ],
  "queueOntologyReExtraction": true,
  "runOntologyReExtractionInline": false,
  "ontologyFocusConcept": "settlement"
}
```

Manual file upload query parameters:

```text
queueOntologyReExtraction=true
runOntologyReExtractionInline=false
ontologyFocusConcept=settlement
```

The default is to queue re-extraction but not run it inline.

## Connector sync job behavior

Connector sync jobs now carry re-extraction flags:

```text
queueOntologyReExtraction
runOntologyReExtractionInline
ontologyFocusConcept
```

When a sync job succeeds, the worker queues an `OntologyReExtractionRun` if the sync created new or changed Artifacts.

The sync job event log records:

```text
ontology.reextraction_queued
ontology.reextraction_skipped
```

This gives operators an audit trail between connector sync and ontology candidate refresh.

## API contract

### Create re-extraction run

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

A request must provide one of:

```text
datasourceId
syncJobId
artifactIds
```

If `runInline=true`, the backend immediately runs extraction and returns the completed or failed re-extraction response.

### List re-extraction runs

```http
GET /v1/ontology/re-extraction-runs
GET /v1/ontology/re-extraction-runs?datasourceId={sourceId}
GET /v1/ontology/re-extraction-runs?status=queued
```

### Read re-extraction run

```http
GET /v1/ontology/re-extraction-runs/{runId}
```

### Run re-extraction

```http
POST /v1/ontology/re-extraction-runs/{runId}/run
```

Request body:

```json
{
  "requestedBy": "operator@example.com"
}
```

Running a queued re-extraction run creates an `OntologyExtractionRun`, which creates `ConceptCandidates` and `RelationCandidates` only.

## Response contract

`SyncSourceResponse` now includes:

```json
{
  "artifactCreatedCount": 1,
  "artifactReusedCount": 0,
  "artifactChangedCount": 0,
  "createdArtifactIds": ["artifact-id"],
  "reusedArtifactIds": [],
  "changedArtifactIds": [],
  "ontologyReextractionRunId": "reextraction-run-id",
  "ontologyReextractionStatus": "queued"
}
```

`OntologyReExtractionRunResponse` includes:

```text
run
extractionRuns
conceptCandidates
relationCandidates
```

A queued run may have no extraction runs yet. A completed run includes the extraction run and generated candidates.

## Settlement reference behavior

Given manual uploaded text:

```text
Settlement is the process of finalizing obligations.
Clearing precedes settlement.
Reconciliation validates settlement.
```

Expected v1.8.0 behavior:

```text
1. Upload creates Artifact and EvidenceFragments.
2. Upload response includes ontologyReextractionRunId with status=queued.
3. Official graph for settlement remains unavailable unless already reviewed official Concepts/Relations exist.
4. Running the re-extraction run creates pending ConceptCandidates:
   - Settlement
   - Clearing
   - Reconciliation
5. Running the re-extraction run creates pending RelationCandidates:
   - Clearing --precedes--> Settlement
   - Reconciliation --validates--> Settlement
6. officialGraphMutated remains false.
7. The official graph changes only after v1.5.0 candidate review approval.
```

## Implementation checklist

```text
[x] Add OntologyReExtractionRun status and trigger enums.
[x] Add OntologyReExtractionRun request/response schemas.
[x] Add re-extraction persistence to in-memory store.
[x] Add re-extraction persistence to SQLAlchemy store.
[x] Add Alembic migration for ontology_reextraction_runs.
[x] Add re-extraction controls to SyncSourceRequest.
[x] Add re-extraction controls to ManualTextUploadRequest.
[x] Add re-extraction controls to CreateSyncJobRequest and SyncJob.
[x] Add Artifact created/reused/changed metadata to SyncSourceResponse.
[x] Queue re-extraction from manual sync/upload when new or changed Artifacts are created.
[x] Queue re-extraction from connector sync worker success when new or changed Artifacts are created.
[x] Add manual request API for re-extraction runs.
[x] Add run/read/list API for re-extraction runs.
[x] Ensure running re-extraction creates candidates only.
[x] Ensure official graph mutation flag remains false.
[x] Add version documentation, readiness document, release notes, API contract updates, and known limitations.
```

## Verification checklist

```text
[ ] Compile source, tests, and scripts.
[ ] Run release-candidate static checker.
[ ] Test manual text upload queues re-extraction.
[ ] Test exact artifact reuse does not queue re-extraction.
[ ] Test changed artifact content queues a new re-extraction run.
[ ] Test running re-extraction creates pending candidates only.
[ ] Test official graph remains unchanged after re-extraction.
[ ] Test connector sync worker records re-extraction queue/skip events.
[ ] Verify persistent-store migration in a dependency-complete environment.
```

## Known limitations

```text
- Change detection is content-hash/idempotency based, not semantic.
- First-time Artifacts queue re-extraction because they introduce new evidence.
- Exact reused Artifacts do not queue re-extraction by default.
- Re-extraction is API/worker-driven; there is no UI queue yet.
- Inline execution is supported but default-off to keep sync latency predictable.
- Re-extraction uses the existing deterministic/local ontology extractor from v1.4.0.
- No live external LLM provider is introduced.
- No stale candidate cleanup or supersession workflow is included.
- No automatic candidate merge or approval is included.
```

## Exit criteria

`v1.8.0` is complete when:

```text
- source sync/manual upload responses expose artifact change metadata;
- new/changed source evidence can queue an OntologyReExtractionRun;
- queued runs can be listed, read, and executed;
- execution creates candidate-only ontology extraction output;
- exact reuse can skip re-extraction;
- official graph mutation remains impossible from re-extraction alone;
- version-specific documentation and release checks are updated.
```

## Next version handoff

The recommended next version is:

```text
v1.9.0 — End-to-end proof and operator UX
```

Suggested goal:

```text
manual upload / connector sync
→ re-extraction queue
→ extraction run
→ candidate review
→ official graph
→ ontology evaluation
→ one-command proof report
```

That release should prove the full SSOT loop without changing the trust boundary.

## Chronicle position and measurable release checklist

This section was added during the v2.0.0 documentation chronicle pass so the version goal and checklist are explicit, measurable, and easy to audit.

**Version goal:** Queue candidate-only ontology re-extraction when manual or connector source data creates or changes Artifacts.

**Confirmed non-goal:** No official graph mutation from connector sync or re-extraction.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V180-01 | OntologyReExtractionRun exists with queued/running/completed/skipped/failed states. | Domain object section and schemas. | complete |
| V180-02 | Manual sync/upload can queue re-extraction for new or changed Artifacts. | Manual source behavior and tests. | complete |
| V180-03 | Connector sync worker can queue re-extraction after successful sync. | Connector sync job behavior. | complete |
| V180-04 | Running re-extraction creates OntologyExtractionRun and candidates only. | API/service tests. | complete |
| V180-05 | SyncSourceResponse exposes created/reused/changed Artifact metadata and re-extraction status. | Response contract. | complete |

**Chronicle handoff:** This version is recorded in `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. Later versions may build on this release only within the confirmed non-goal boundary above.


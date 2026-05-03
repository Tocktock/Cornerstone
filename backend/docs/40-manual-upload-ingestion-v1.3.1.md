# v1.3.1 — Manual Uploaded Data Ingestion

## Purpose

`v1.3.1` adds the first user-provided manual upload ingestion path for Cornerstone.

This release lets users provide settlement notes, policy snippets, internal process descriptions, or other domain text without first connecting Notion or Google Drive.

The runtime flow is intentionally small:

```text
manual uploaded file / pasted text
→ SourceObject
→ Artifact
→ EvidenceFragment
→ review queue
```

The release strengthens the future ontology roadmap by making it easy to seed source evidence manually before LLM ontology extraction exists.

## Product goal

A user should be able to:

```text
1. Create a manual source.
2. Upload one or more UTF-8 text-like files or pasted text objects.
3. Receive Artifacts and EvidenceFragments with provenance.
4. Review those EvidenceFragments later.
5. Use the reviewed evidence in existing Concept, ConceptRelation, and ontology graph workflows.
```

For the reference `settlement` use case, the expected pilot loop is:

```text
Upload settlement.md
→ Artifact(title="settlement.md")
→ EvidenceFragments such as "Settlement is ..." and "Clearing precedes settlement."
→ reviewer can approve evidence
→ existing v1.3.0 graph API can serve Concepts/Relations once humans create and officialize them
```

## What this version is not

`v1.3.1` intentionally does **not** implement:

```text
- LLM ontology extraction
- ConceptCandidate runtime tables
- RelationCandidate runtime tables
- candidate review workflow
- automatic Concept creation
- automatic Relation creation
- graph edge inference
- PDF extraction
- Office document extraction
- image OCR
- binary upload ingestion
- vector search
- graph visualization UI
```

Manual upload only creates source evidence. It does not make ontology claims.

## Trust boundary

The `v1.2.1` ontology SSOT trust boundary remains unchanged:

```text
Uploaded content is source material.
EvidenceFragments are not official truth until reviewed.
LLM output is still candidate-only and is not introduced in this release.
Official Concepts and official ConceptRelations still require reviewed evidence.
```

The reviewed ontology graph remains the Single Source of Truth, not the uploaded file itself.

## New API endpoints

### Upload text-like files

```http
POST /v1/manual-sources/{sourceId}/uploads
```

Request type:

```text
multipart/form-data
field name: files
```

Example:

```bash
curl -X POST "http://localhost:8000/v1/manual-sources/$SOURCE_ID/uploads" \
  -F "files=@settlement.md;type=text/markdown" \
  -F "files=@ledger-notes.txt;type=text/plain"
```

Response model:

```text
SyncSourceResponse
```

This is the same response shape used by `/v1/manual-sources/{sourceId}/sync`:

```text
dataSource
artifacts
evidenceFragments
artifactCreatedCount
artifactReusedCount
evidenceCreatedCount
```

### Upload pasted/manual text

```http
POST /v1/manual-sources/{sourceId}/uploads/text
```

Request example:

```json
{
  "objects": [
    {
      "title": "Settlement Notes",
      "content": "Settlement is the process of finalizing obligations. Clearing precedes settlement.",
      "providerMetadata": {
        "importBatch": "pilot"
      }
    }
  ]
}
```

Optional fields:

```text
sourceExternalId
sourceUrl
sourceUpdatedAt
providerMetadata
```

If `sourceExternalId` is omitted, the backend creates a stable manual id:

```text
manual-upload:text:{title}
```

## Supported upload types

`v1.3.1` accepts UTF-8 text-like content only.

Supported intent:

```text
- text/plain
- text/markdown
- text/csv
- text/html
- text/* in general
- application/json
- application/x-ndjson
- application/xml
- application/yaml
- application/x-yaml
- common text suffixes such as .txt, .md, .csv, .json, .jsonl, .yaml, .xml, .html, .log, .rst
```

If a client sends `application/octet-stream` for a known text suffix, the backend accepts the file and validates UTF-8 decoding.

Rejected:

```text
- PDFs
- Word/Office documents
- images
- binary files
- non-UTF-8 text
- empty files
```

Rejected uploads do not create Artifacts or EvidenceFragments.

## Configurable limits

Defaults:

```text
MANUAL_UPLOAD_MAX_FILE_COUNT=10
MANUAL_UPLOAD_MAX_FILE_BYTES=5242880
```

Behavior:

```text
- too many files returns 413
- file too large returns 413
- unsupported media returns 415
- empty file/text returns 422
```

## Artifact contract

Uploaded files are normalized into `SourceObject` records with:

```json
{
  "sourceExternalId": "manual-upload:settlement.md",
  "title": "settlement.md",
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

Pasted text is normalized into:

```json
{
  "sourceExternalId": "manual-upload:text:Settlement Notes",
  "title": "Settlement Notes",
  "sourceObjectType": "uploaded_text",
  "providerMetadata": {
    "uploadKind": "manual_text",
    "title": "Settlement Notes"
  }
}
```

## Idempotency

Manual upload uses the existing sync identity rule:

```text
dataSourceId + sourceExternalId + rawContentHash
```

The same filename with the same content reuses the existing Artifact and EvidenceFragments.

The same filename with changed content creates a new Artifact because the content hash changed.

This matches the existing connector ingestion behavior and keeps uploaded revisions explainable.

## Implementation checklist

```text
[x] Add manual upload service boundary.
[x] Add multipart file upload endpoint.
[x] Add JSON pasted-text upload endpoint.
[x] Keep upload endpoints manual-source-only.
[x] Reuse existing sync_source_objects service.
[x] Preserve Artifact/EvidenceFragment provenance.
[x] Reject unsupported/binary files before Artifact creation.
[x] Add upload size/count settings.
[x] Add python-multipart runtime dependency.
[x] Add integration tests for file upload, text upload, idempotency, and rejection paths.
[x] Update API contract, README, known limitations, release notes, readiness docs, and release checker.
```

## Verification checklist

Recommended local checks:

```bash
python -m compileall -q src tests scripts
python -m pytest tests/integration/test_manual_uploads_api.py -q
python -m pytest tests/integration/test_artifacts_evidence_api.py tests/integration/test_ontology_api.py tests/unit/test_release_candidate_docs.py -q
python scripts/check_release_candidate.py
```

Optional full local gate:

```bash
python -m pytest -q
python -m ruff check src tests scripts
python -m mypy src --show-error-codes --no-color-output --no-incremental
```

## Settlement reference behavior

Given a file:

```text
Settlement is the process of finalizing obligations. Clearing precedes settlement.
```

The backend should create:

```text
Artifact: settlement.md
EvidenceFragment: Settlement is the process of finalizing obligations.
EvidenceFragment: Clearing precedes settlement.
```

It should not create:

```text
Concept: Settlement
Concept: Clearing
Relation: Clearing precedes Settlement
```

Those ontology objects remain manual/reviewed work until `v1.4.0` and later introduce LLM-assisted candidate generation.

## Next version handoff

`v1.4.0` should use reviewed or unreviewed EvidenceFragments from manual uploads and connected sources as input to LLM ontology extraction.

Expected next flow:

```text
EvidenceFragments
→ OntologyExtractionRun
→ ConceptCandidates
→ RelationCandidates
```

Candidate output must remain separate from the official ontology graph.

## Chronicle position and measurable release checklist

This section was added during the v2.0.0 documentation chronicle pass so the version goal and checklist are explicit, measurable, and easy to audit.

**Version goal:** Let users seed Cornerstone with manual UTF-8 text-like files or pasted text that becomes source-backed evidence.

**Confirmed non-goal:** No automatic Concept, Relation, or official graph creation.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V131-01 | `POST /v1/manual-sources/{sourceId}/uploads` accepts multipart text-like files. | API contract and integration tests. | complete |
| V131-02 | `POST /v1/manual-sources/{sourceId}/uploads/text` accepts pasted text objects. | API contract and integration tests. | complete |
| V131-03 | Accepted uploads create SourceObjects, Artifacts, and EvidenceFragments with provenance. | SyncSourceResponse contract. | complete |
| V131-04 | Unsupported/binary/non-UTF-8 files are rejected before Artifact creation. | Rejection path tests. | complete |
| V131-05 | Manual upload does not officialize evidence or mutate ontology graph state. | Trust boundary section and tests. | complete |

**Chronicle handoff:** This version is recorded in `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. Later versions may build on this release only within the confirmed non-goal boundary above.


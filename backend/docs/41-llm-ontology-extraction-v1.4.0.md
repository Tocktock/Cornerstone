# v1.4.0 — LLM Ontology Extraction

## Purpose

`v1.4.0` adds the first ontology extraction runtime.

The release turns existing EvidenceFragments into reviewable ontology candidates:

```text
EvidenceFragments
→ OntologyExtractionRun
→ ConceptCandidates
→ RelationCandidates
```

This is the first step where Cornerstone uses an extraction provider to construct a proposed ontology from source-backed evidence.

## Product goal

A user should be able to:

```text
1. Upload or sync source material.
2. Create Artifacts and EvidenceFragments.
3. Request ontology extraction over selected EvidenceFragments or Artifacts.
4. Receive candidate Concepts and candidate Relations.
5. Inspect candidate definitions, relation types, confidence, rationale, and source evidence.
6. Keep all generated ontology objects pending until a later human review workflow approves them.
```

For the reference `settlement` use case:

```text
Uploaded evidence:
"Settlement is the process of finalizing obligations. Clearing precedes settlement."

v1.4.0 output:
ConceptCandidate: Settlement
ConceptCandidate: Clearing
RelationCandidate: Clearing --precedes--> Settlement
```

## What this version is not

`v1.4.0` intentionally does **not** implement:

```text
- candidate approval
- candidate rejection
- candidate merge workflow
- automatic official Concept creation
- automatic official Relation creation
- graph edge officialization
- live external LLM credentials
- prompt management UI
- graph visualization UI
- depth greater than 1
- vector search
- PDF or Office extraction
```

The output of this release is candidate-only.

## Trust boundary

The `v1.2.1` ontology SSOT boundary remains mandatory:

```text
LLM/extractor output is candidate-only.
Candidate Concepts are not official Concepts.
Candidate Relations are not official ConceptRelations.
No candidate can appear in the official graph until reviewed and promoted in a later version.
No evidence → no candidate.
No evidence → no official ontology object.
```

The official ontology graph remains the Single Source of Truth.

## Provider boundary

`v1.4.0` introduces a provider boundary but ships with a deterministic local provider:

```text
provider=local_rule_based
modelName=local-rule-based-ontology-extractor-v1.4.0
promptVersion=ontology-extraction-v1.4.0
```

This is deliberate. The runtime contract, persistence tables, validation rules, and API behavior are implemented before live model credentials are introduced.

A future live LLM provider should produce the same structured candidate contract.

## New data model

### OntologyExtractionRun

Tracks an extraction attempt.

```text
id
provider
modelName
promptVersion
status: queued | running | completed | failed
requestedBy
focusConcept
evidenceFragmentIds
artifactIds
conceptCandidateCount
relationCandidateCount
warningCount
error
createdAt
startedAt
completedAt
```

### ConceptCandidate

Represents a proposed Concept before review.

```text
id
extractionRunId
name
normalizedName
aliases
proposedDefinition
conceptType
evidenceFragmentIds
confidence
status: pending | approved | rejected | merged
matchedExistingConceptId
rationale
validationErrors
createdAt
```

### RelationCandidate

Represents a proposed relation before review.

```text
id
extractionRunId
sourceName
targetName
normalizedSourceName
normalizedTargetName
sourceCandidateId
targetCandidateId
sourceConceptId
targetConceptId
relationType
evidenceFragmentIds
confidence
rationale
status: pending | approved | rejected | merged
validationErrors
createdAt
```

## New API endpoints

### `POST /v1/ontology/extraction-runs`

Creates an extraction run and returns candidate objects.

Request:

```json
{
  "artifactIds": ["artifact-id"],
  "evidenceFragmentIds": [],
  "focusConcept": "settlement",
  "provider": "local_rule_based",
  "requestedBy": "reviewer@example.com",
  "maxEvidenceFragments": 50
}
```

Response:

```text
OntologyExtractionRunResponse
- run
- conceptCandidates
- relationCandidates
```

### `GET /v1/ontology/extraction-runs`

Lists extraction runs.

### `GET /v1/ontology/extraction-runs/{runId}`

Returns a run with its candidate Concepts and candidate Relations.

### `GET /v1/ontology/concept-candidates`

Lists candidate Concepts.

Query parameters:

```text
runId=<optional extraction run id>
status=pending|approved|rejected|merged
```

### `GET /v1/ontology/relation-candidates`

Lists candidate Relations.

Query parameters:

```text
runId=<optional extraction run id>
status=pending|approved|rejected|merged
```

## Validation rules

The backend rejects or skips invalid extraction output:

```text
- extraction scope must include artifactIds or evidenceFragmentIds
- every ConceptCandidate requires at least one EvidenceFragment
- every RelationCandidate requires at least one EvidenceFragment
- relation type must be in the allowed taxonomy
- source and target relation terms must differ
- candidates are created with status=pending
- extraction cannot create official Concepts or official Relations
```

## Example cURL

```bash
curl -X POST "http://localhost:8000/v1/ontology/extraction-runs"   -H 'Content-Type: application/json'   -d '{
    "artifactIds": ["ARTIFACT_ID"],
    "focusConcept": "settlement",
    "requestedBy": "reviewer@example.com"
  }'
```

Expected candidate relation example:

```json
{
  "sourceName": "Clearing",
  "relationType": "precedes",
  "targetName": "Settlement",
  "status": "pending",
  "evidenceFragmentIds": ["evidence-id"]
}
```

## Implementation checklist

```text
[x] Add OntologyExtractionRun schema.
[x] Add ConceptCandidate schema.
[x] Add RelationCandidate schema.
[x] Add candidate status enum.
[x] Add extraction provider enum.
[x] Add local deterministic extraction provider.
[x] Add extraction service.
[x] Persist extraction runs in memory.
[x] Persist concept candidates in memory.
[x] Persist relation candidates in memory.
[x] Add PostgreSQL/SQLAlchemy rows.
[x] Add Alembic migration.
[x] Add extraction run API endpoints.
[x] Add candidate list API endpoints.
[x] Add integration tests.
[x] Keep output candidate-only.
[x] Document non-goals.
```

## Known limitations

```text
- The bundled provider is deterministic and rule-based, not a live external LLM.
- Candidate quality is limited by explicit wording in EvidenceFragments.
- Candidate review and promotion are deferred to v1.5.0.
- Concept deduplication is limited to exact normalized names and aliases.
- Relation extraction supports a controlled set of explicit relation phrases.
- Candidates do not appear in the official graph.
```

## Exit criteria

`v1.4.0` is complete when:

```text
- a manual uploaded settlement document can produce ConceptCandidates and RelationCandidates
- each candidate has evidence ids
- each run is persisted and readable
- official graph output remains unchanged unless humans create official Concepts/Relations separately
- version docs, release notes, readiness docs, and verification reports are present
```

## Next version handoff

`v1.5.0` should implement the ontology review workflow:

```text
ConceptCandidate / RelationCandidate
→ approve / reject / merge
→ official Concept / ConceptRelation creation only after human review and evidence validation
```

## Chronicle position and measurable release checklist

This section was added during the v2.0.0 documentation chronicle pass so the version goal and checklist are explicit, measurable, and easy to audit.

**Version goal:** Turn EvidenceFragments into reviewable ConceptCandidates and RelationCandidates through an extraction provider boundary.

**Confirmed non-goal:** No automatic promotion to official Concepts or ConceptRelations; no live external LLM requirement.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V140-01 | `POST /v1/ontology/extraction-runs` creates an extraction run. | API contract and tests. | complete |
| V140-02 | Extraction creates ConceptCandidates and RelationCandidates with evidence ids. | Candidate list endpoints and tests. | complete |
| V140-03 | All generated ontology objects have status `pending`. | Candidate schemas/tests. | complete |
| V140-04 | Invalid output without evidence or valid relation type is rejectable. | Validation rules section. | complete |
| V140-05 | The bundled provider is deterministic/local and does not require external credentials. | Provider boundary section. | complete |

**Chronicle handoff:** This version is recorded in `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. Later versions may build on this release only within the confirmed non-goal boundary above.


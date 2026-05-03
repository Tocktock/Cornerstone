# API Freeze Review

## Purpose

Before backend v1.0.0, freeze the MVP API surface that supports the proven loop.

## Frozen MVP surfaces

```text
GET  /healthz
GET  /v1/sources
POST /v1/sources                         # manual sources only
POST /v1/manual-sources/{sourceId}/sync
GET  /v1/connectors
GET  /v1/connectors/{provider}
POST /v1/connections/intents
GET  /v1/connections/intents/{intentId}
GET  /v1/oauth/{provider}/authorize
GET  /v1/oauth/{provider}/callback
POST /v1/sources/{sourceId}/test
POST /v1/sources/{sourceId}/discover
GET  /v1/sources/{sourceId}/objects
GET  /v1/sources/{sourceId}/selections
PUT  /v1/sources/{sourceId}/selections
POST /v1/sources/{sourceId}/sync-jobs
GET  /v1/sources/{sourceId}/sync-jobs
GET  /v1/sync-jobs/{syncJobId}
POST /v1/sync-jobs/{syncJobId}/run
POST /v1/sync-jobs/{syncJobId}/cancel
POST /v1/sync-jobs/{syncJobId}/heartbeat
POST /v1/sync-worker/run
GET  /v1/artifacts
GET  /v1/evidence
GET  /v1/evidence/review-queue
POST /v1/evidence/{evidenceFragmentId}/review
POST /v1/evidence/{evidenceFragmentId}/concept-candidates
GET  /v1/concepts
POST /v1/concepts
GET  /v1/concepts/{conceptId}
POST /v1/concepts/{conceptId}/officialize
GET  /v1/concept-relations
POST /v1/concept-relations
POST /v1/concept-relations/{relationId}/officialize
GET  /v1/decision-records
POST /v1/decision-records
GET  /v1/context/query
POST /v1/evaluations/tasks
POST /v1/evaluations/tasks/{taskId}/run
GET  /v1/evaluations/summary
GET  /v1/audit-events
```

## Removed or intentionally unavailable surfaces

```text
POST /v1/sources/{sourceId}/sync                 # removed; legacy bypass
POST /v1/sources/{sourceId}/oauth/complete       # removed; fake OAuth path
POST /v1/sources with type=notion                # rejected; provider source must use connector flow
POST /v1/manual-sources/{notionSourceId}/sync    # rejected; manual sync is manual-only
```

## Snapshot protection

OpenAPI snapshots currently protect:

```text
- grounded serving contract
- evaluation contract
```

Before v1.0.0, run:

```bash
./scripts/run_tests.sh
```

and confirm the OpenAPI contract snapshot tests pass.


## v1.3.0 Ontology Graph API Additions

The v1.3.0 API additions are additive and do not remove existing backend MVP routes.

```text
GET  /v1/ontology/search
GET  /v1/ontology/graph
```

Boundary review:

```text
- `/v1/ontology/search` searches existing Concepts by name or alias.
- `/v1/ontology/graph` serves a depth-0 or depth-1 graph from existing Concepts and ConceptRelations.
- Default `mode=official` excludes candidate objects.
- Depth above 1 is rejected in v1.3.0.
- These routes do not run LLM extraction or create ontology candidates.
```

## v1.3.1 Manual Upload API Additions

The v1.3.1 API additions are additive and do not remove existing backend MVP or ontology graph routes.

```text
POST /v1/manual-sources/{sourceId}/uploads
POST /v1/manual-sources/{sourceId}/uploads/text
```

Boundary review:

```text
- Upload routes are manual-source-only.
- Provider-backed sources must still use connector discovery, selection, and sync jobs.
- File upload accepts UTF-8 text-like files only.
- Pasted text upload accepts JSON text objects only.
- Upload routes create Artifacts and EvidenceFragments through the existing sync pipeline.
- Upload routes do not create Concepts, ConceptRelations, ConceptCandidates, or RelationCandidates.
- Upload routes do not run an LLM or infer ontology graph edges.
```


## v1.4.0 Ontology Extraction API Additions

The v1.4.0 API additions are additive and do not remove existing backend MVP, manual upload, or ontology graph routes.

```text
POST /v1/ontology/extraction-runs
GET  /v1/ontology/extraction-runs
GET  /v1/ontology/extraction-runs/{runId}
GET  /v1/ontology/concept-candidates
GET  /v1/ontology/relation-candidates
```

Boundary review:

```text
- Extraction routes create OntologyExtractionRun records.
- Extraction routes create ConceptCandidates and RelationCandidates only.
- Candidates are created with status=pending.
- Extraction routes do not approve, reject, merge, or officialize candidates.
- Extraction routes do not create official Concepts or official ConceptRelations.
- Extraction routes do not mutate the official ontology graph.
- The bundled provider is deterministic/local; live external LLM credentials are not introduced in v1.4.0.
```

## v1.5.0 Ontology Candidate Review API Additions

The v1.5.0 API additions are additive and do not remove existing backend MVP, manual upload, ontology graph, or ontology extraction routes.

```text
PATCH /v1/ontology/concept-candidates/{candidateId}
POST  /v1/ontology/concept-candidates/{candidateId}/approve
POST  /v1/ontology/concept-candidates/{candidateId}/reject
POST  /v1/ontology/concept-candidates/{candidateId}/merge
PATCH /v1/ontology/relation-candidates/{candidateId}
POST  /v1/ontology/relation-candidates/{candidateId}/approve
POST  /v1/ontology/relation-candidates/{candidateId}/reject
POST  /v1/ontology/relation-candidates/{candidateId}/merge
```

Boundary review:

```text
- Review routes operate on existing ConceptCandidates and RelationCandidates only.
- Review routes require authorized reviewers in production mode.
- Edit routes only update pending candidates.
- Reject routes mark pending candidates rejected and create no official ontology object.
- Approve routes promote candidates only through the existing officialization gates.
- Merge routes update existing Concepts or ConceptRelations only through the same evidence gates.
- Review routes do not run extraction or call a live external LLM.
- Review routes do not bypass evidence review or create unsupported official graph edges.
```

## v1.6.0 Explainable Graph API Additions

The v1.6.0 API changes are additive and do not remove existing backend MVP, manual upload, ontology extraction, or candidate review routes.

```text
GET  /v1/ontology/explain
```

Boundary review:

```text
- `/v1/ontology/graph` keeps its existing route and response fields, with additive explanation/support/provenance fields.
- `/v1/ontology/explain` returns the same OntologyGraphResponse shape as `/graph` for explanation-first clients.
- Official graph mode still excludes candidate Concepts, candidate ConceptRelations, ConceptCandidates, and RelationCandidates.
- Pending candidates are summarized only in candidateSummary.
- Explainable graph serving does not run extraction, call an LLM, review candidates, or infer missing graph edges.
- Depth above 1 remains rejected.
```

## v1.7.0 Ontology Evaluation API Additions

The v1.7.0 API changes are additive and read-only with respect to ontology graph state. These endpoints create and run evaluation tasks, but they do not mutate Concepts, ConceptRelations, candidates, source data, or evidence.

Accepted additions:

```text
POST /v1/evaluations/ontology/tasks
GET  /v1/evaluations/ontology/tasks
GET  /v1/evaluations/ontology/tasks/{taskId}
POST /v1/evaluations/ontology/tasks/{taskId}/run
POST /v1/evaluations/ontology/run
GET  /v1/evaluations/ontology/results
GET  /v1/evaluations/ontology/results/{resultId}
GET  /v1/evaluations/ontology/summary
```

Freeze review conclusion:

```text
accepted; ontology evaluation is read-only
accepted; official graph mutation remains out of scope
accepted; candidate promotion remains out of scope
accepted; unsupported graph tasks require explicit unsupported configuration
```

## v1.8.0 Connector-driven Re-extraction API Additions

The v1.8.0 API changes are additive and preserve the existing SSOT trust boundary. These endpoints queue and run ontology re-extraction, but the output remains candidate-only.

Accepted additions:

```text
POST /v1/ontology/re-extraction-runs
GET  /v1/ontology/re-extraction-runs
GET  /v1/ontology/re-extraction-runs/{runId}
POST /v1/ontology/re-extraction-runs/{runId}/run
```

Accepted additive fields:

```text
queueOntologyReExtraction
runOntologyReExtractionInline
ontologyFocusConcept
artifactChangedCount
createdArtifactIds
reusedArtifactIds
changedArtifactIds
ontologyReextractionRunId
ontologyReextractionStatus
```

Freeze review conclusion:

```text
accepted; re-extraction is candidate-only
accepted; official graph mutation remains out of scope
accepted; automatic candidate approval remains out of scope
accepted; connector sync can queue review work but cannot create official truth
accepted; exact artifact reuse can skip re-extraction
```

## v1.9.0 End-to-end Proof and Operator Checklist API Addition

The v1.9.0 API change is additive and operator-focused. It creates an explicit proof/checklist route for the ontology SSOT loop.

Accepted addition:

```text
POST /v1/ontology/proof-runs
```

Freeze review conclusion:

```text
accepted; proof runs are explicit operator actions
accepted; non-dry proof mutation requires confirmMutation=true
accepted; dryRun=true creates no objects
accepted; proof loop still uses evidence review and candidate review gates
accepted; no frontend UI, live LLM provider, or automatic normal-data approval is introduced
```


## v2.0.0 Ontology SSOT Readiness API Addition

The v2.0.0 API change is additive and read-only. It exposes release/operator readiness for the ontology Single Source of Truth contract.

Accepted addition:

```text
GET /v1/ontology/ssot/readiness
```

Freeze review conclusion:

```text
accepted; endpoint is read-only
accepted; endpoint does not run extraction or re-extraction
accepted; endpoint does not review evidence or approve candidates
accepted; endpoint does not mutate Concepts, Relations, candidates, evidence, or evaluation records
accepted; readiness requires official graph safety and successful evaluation
```


## v2.0.1 refactor API freeze note

`v2.0.1` is a behavior-preserving refactor release. It does not add, remove, or rename public API endpoints.

Measurable API-freeze checks:

- [x] `GET /v1/ontology/ssot/readiness` remains the SSOT readiness endpoint.
- [x] `POST /v1/ontology/proof-runs` remains the operator proof endpoint.
- [x] Existing ontology extraction, review, graph, evaluation, re-extraction, and proof endpoints remain in place.
- [x] The package version and readiness `releaseVersion` now report `2.0.1`.
- [x] No migration is required because no persisted schema changed.

## v2.0.2 Product Documentation API Freeze Note

`v2.0.2` is a documentation-first release. It adds product documentation and rewrites README for product clarity. It does not add, remove, or rename public API endpoints.

Measurable API-freeze checks:

- [x] `GET /v1/ontology/ssot/readiness` remains the SSOT readiness endpoint.
- [x] `POST /v1/ontology/proof-runs` remains the operator proof endpoint.
- [x] Existing ontology extraction, review, graph, evaluation, re-extraction, proof, and readiness endpoints remain in place.
- [x] No migration is required because no persisted schema changed.
- [x] The package/runtime version and readiness `releaseVersion` now report `2.0.2`.
- [x] Product docs are documentation-only and introduce no new API surface.

## v2.0.3 Dependency-Complete Verification API Freeze Note

`v2.0.3` is a verification-hardening release. It adds a dependency-complete verification command plan, strict runner, and CI workflow. It does not add, remove, or rename public API endpoints.

Freeze result:

```text
No public API endpoint change.
No request schema change.
No response schema change except version metadata reporting 2.0.3.
No database migration.
No graph behavior change.
No extraction/review behavior change.
```

Checklist:

- [x] Existing ontology SSOT endpoints remain unchanged.
- [x] `GET /v1/ontology/ssot/readiness` remains read-only.
- [x] The package/runtime version and readiness `releaseVersion` now report `2.0.3`.
- [x] Verification scripts are tooling-only and do not expose new backend API routes.

## v2.0.4 Forward Roadmap API Freeze Note

`v2.0.4` is a documentation-first planning release. It adds roadmap documents for `v2.1.0` through `v2.5.0`. It does not add, remove, or rename public API endpoints.

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

Checklist:

- [x] Existing ontology SSOT endpoints remain unchanged.
- [x] `GET /v1/ontology/ssot/readiness` remains read-only.
- [x] The package/runtime version and readiness `releaseVersion` now report `2.0.4`.
- [x] Roadmap docs are planning-only and do not expose new backend routes.
- [x] Future releases `v2.1.0` through `v2.5.0` must still complete their own API freeze review before implementation release.

## v2.5.0 External Integration Package API Freeze Note

`v2.5.0` implements the roadmap through the external integration package path.

Freeze result:

```text
Public API endpoints added for review queue summary, candidate previews, connector support matrix, and integration package consumption.
No database migration.
No automatic candidate approval.
No direct connector-to-official-graph mutation.
No frontend MVP.
No candidate bypass.
```

Added or changed API surface:

```text
POST /v1/ontology/extraction-runs accepts provider=live_llm
GET  /v1/ontology/review-queue/summary
GET  /v1/ontology/concept-candidates/{candidateId}/preview
GET  /v1/ontology/relation-candidates/{candidateId}/preview
GET  /v1/connectors/support-matrix
GET  /v1/integration/package/manifest
GET  /v1/integration/ontology/{concept}
```

Checklist:

- [x] `provider=live_llm` is gated by explicit configuration.
- [x] Review preview endpoints are read-only.
- [x] `OntologyGraphResponse.visualization` is additive response metadata.
- [x] Connector support matrix exposes `mutatesOfficialGraph=false`.
- [x] Integration package rejects `includeCandidates=true`.
- [x] The package/runtime version and readiness `releaseVersion` now report `2.5.0`.

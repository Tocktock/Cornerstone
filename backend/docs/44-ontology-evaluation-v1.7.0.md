# Cornerstone v1.7.0 — Ontology Evaluation

## Purpose

`v1.7.0` adds a read-only evaluation framework for ontology graph serving. The release measures whether Cornerstone can safely serve an explainable ontology graph as a Single Source of Truth without changing extraction, review, or graph-generation behavior.

The core product flow remains:

```text
source data
→ Artifact
→ EvidenceFragment
→ ConceptCandidate / RelationCandidate
→ human review
→ official Concept / official ConceptRelation
→ explainable ontology graph
→ ontology graph evaluation
```

The evaluation layer answers a separate question:

```text
When Cornerstone serves a graph for a concept like "settlement", did the served graph meet the expected SSOT quality gates?
```

## Version goal

`v1.7.0` must let operators define, run, persist, and summarize ontology graph evaluation tasks.

A task can assert that a graph response must include explicit Concepts, Relations, EvidenceFragments, trust labels, freshness state, review provenance, evidence count, node count, edge count, and candidate-boundary behavior.

The evaluation service then calls the existing ontology graph service and records a deterministic result.

```text
OntologyGraphEvalTask
→ OntologyGraphService.graph(...)
→ OntologyGraphResponse
→ OntologyGraphEvalResult
→ OntologyGraphEvalMetricSummary
```

## Non-goals

`v1.7.0` intentionally does **not** add:

```text
- no new LLM provider
- no ontology extraction changes
- no candidate review changes
- no automatic Concept promotion
- no automatic Relation promotion
- no official graph mutation
- no semantic/vector scoring
- no graph visualization UI
- no graph depth above 1
- no connector-driven re-extraction
```

Evaluation is read-only. It must not create, edit, approve, reject, merge, or delete Concepts, Relations, candidates, evidence, or source data.

## Trust boundary

The Single Source of Truth rule remains unchanged:

```text
Raw source document ≠ Single Source of Truth
LLM/extractor output ≠ Single Source of Truth
Candidate object ≠ Single Source of Truth
Reviewed official ontology graph = Single Source of Truth
```

`v1.7.0` verifies the served graph against that rule. It does not weaken the rule.

## API contract

All endpoints are under the existing evaluation router.

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

Example request:

```json
{
  "name": "settlement official graph must be explainable",
  "conceptQuery": "settlement",
  "mode": "official",
  "depth": 1,
  "expectedTrustLabel": "official",
  "requiredConceptIds": ["concept-settlement", "concept-clearing"],
  "requiredRelationIds": ["relation-clearing-precedes-settlement"],
  "requiredEvidenceFragmentIds": ["evidence-settlement-definition"],
  "requireOfficialGraph": true,
  "requireEvidence": true,
  "minEvidenceCount": 2,
  "minNodeCount": 2,
  "minEdgeCount": 1,
  "requireReviewProvenance": true,
  "maxPendingCandidateCount": 5,
  "tags": ["settlement", "smoke"],
  "createdBy": "reviewer@example.com"
}
```

### Run a single task

```http
POST /v1/evaluations/ontology/tasks/{taskId}/run
```

Example request:

```json
{
  "evaluatedBy": "operator@example.com"
}
```

### Run many tasks

```http
POST /v1/evaluations/ontology/run
```

Example request:

```json
{
  "taskIds": ["task-id-1", "task-id-2"],
  "evaluatedBy": "operator@example.com"
}
```

When `taskIds` is empty, the backend runs all ontology graph evaluation tasks.

## Evaluation task fields

`OntologyGraphEvalTask` records the graph contract the operator expects.

| Field | Purpose |
|---|---|
| `conceptQuery` | Concept name or alias to evaluate, for example `settlement`. |
| `mode` | Graph serving mode. Defaults to `official`. |
| `depth` | Graph depth. `v1.7.0` supports `0` or `1`; default is `1`. |
| `expectedTrustLabel` | Expected graph trust label, usually `official` for SSOT tasks. |
| `expectedFreshnessState` | Expected freshness state when freshness is part of the task. |
| `requiredConceptIds` | Concepts that must appear in the served graph. |
| `requiredRelationIds` | Relations that must appear in the served graph. |
| `requiredEvidenceFragmentIds` | Evidence fragments that must be cited. |
| `requireOfficialGraph` | Requires official graph mode, official trust, and official objects. |
| `requireEvidence` | Requires evidence citations. |
| `minEvidenceCount` | Minimum citation count. |
| `minNodeCount` | Minimum node count. |
| `minEdgeCount` | Minimum edge count. |
| `requireReviewProvenance` | Requires reviewer/officialization provenance in citations/nodes/edges. |
| `maxPendingCandidateCount` | Optional boundary check for pending candidate backlog. |
| `tags` | Operator-defined grouping labels. |

## Evaluation result fields

`OntologyGraphEvalResult` records what happened when the task was run.

| Field | Meaning |
|---|---|
| `graphFound` | Focus Concept was found and served. |
| `graphDepthRespected` | Served graph depth matched the task and stayed within v1 depth limits. |
| `nodeRequirementsMet` | Required Concepts and node counts were satisfied. |
| `edgeRequirementsMet` | Required Relations and edge counts were satisfied. |
| `evidenceValid` | Required evidence was present and citation references were valid. |
| `provenancePresent` | Evidence, source, reviewer, and officialization provenance was present as required. |
| `trustLabelCorrect` | Served trust label matched the task and official graph requirement. |
| `freshnessPolicyRespected` | Freshness state and trust label did not conflict. |
| `officialGraphSafe` | Official graph response had no non-official nodes/edges or invalid support. |
| `candidateBoundaryRespected` | Candidate objects were not included in official graph mode and pending candidate limits were respected. |
| `relationIntegrityValid` | Every served edge connects nodes present in the graph. |
| `citationValidityRate` | Percentage of valid citations in the graph response. |
| `success` | True only when all required gates pass. |
| `failureReasons` | Stable machine-readable failure reason strings. |

## Quality gates

The evaluation service checks these gates:

```text
graph_found
graph_depth_respected
node_requirements_met
edge_requirements_met
evidence_valid
provenance_present
trust_label_correct
freshness_policy_respected
official_graph_safe
candidate_boundary_respected
relation_integrity_valid
citation_validity_rate
```

A successful official SSOT task usually requires:

```text
- focus Concept exists
- graph is served in official mode
- trust label is official
- required nodes are present
- required edges are present
- required evidence is cited
- citations are valid
- evidence has source provenance
- reviewed evidence provenance is present
- node and edge review provenance is present
- no candidate nodes or edges leak into the official graph
- every edge source/target appears in the response nodes
```

## Unsupported graph task contract

Unsupported graph tasks are allowed when the expected outcome is that no official graph should be served.

For unsupported tasks, the request must explicitly set:

```json
{
  "expectedTrustLabel": "unsupported",
  "requireOfficialGraph": false,
  "requireEvidence": false,
  "minEvidenceCount": 0,
  "minNodeCount": 0,
  "minEdgeCount": 0,
  "requireReviewProvenance": false
}
```

Unsupported tasks cannot require Concepts, Relations, or EvidenceFragments. This prevents vague or contradictory evaluation tasks.

## Metrics

`GET /v1/evaluations/ontology/summary` returns:

```text
totalTaskCount
evaluatedResultCount
successfulResultCount
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

These metrics allow operators to track whether the ontology graph can safely serve as the organizational Single Source of Truth.

## Settlement reference behavior

For the reference concept `settlement`, a healthy official evaluation should prove that Cornerstone can serve a depth-1 graph like:

```text
Clearing --precedes--> Settlement
Reconciliation --validates--> Settlement
Settlement --updates--> Ledger
```

The evaluator should verify that:

```text
- Settlement appears as the focus Concept.
- Direct neighbor Concepts appear when official Relations connect them.
- Required Relations appear.
- Relation evidence is cited.
- Concept evidence is cited when available.
- The graph remains depth=1.
- Candidate Concepts and RelationCandidates are not included in official mode.
```

## Implementation checklist

```text
[x] Add ontology graph evaluation schemas.
[x] Add read-only OntologyGraphEvaluationService.
[x] Add in-memory task/result persistence.
[x] Add SQLAlchemy task/result persistence models.
[x] Add Alembic migration for ontology evaluation tables.
[x] Add ontology evaluation API routes.
[x] Add tests for success, unsupported graph, validation failure, bulk run, result filtering, and 404s.
[x] Update API contract docs.
[x] Update API freeze review docs.
[x] Update release readiness docs.
[x] Keep graph mutation out of scope.
```

## Known limitations

```text
- Evaluation is deterministic and rule-based, not semantic.
- Evaluation does not judge whether a human-approved definition is domain-perfect.
- Evaluation does not call an external LLM.
- Evaluation does not generate new Concepts or Relations.
- Evaluation does not repair failed graphs.
- Evaluation is API-driven; there is no review/evaluation UI yet.
- Graph depth remains limited to 0 or 1.
```

## Exit criteria

`v1.7.0` is ready when:

```text
- Ontology graph evaluation tasks can be created.
- Tasks can be listed and read.
- Single tasks can be run.
- Task batches can be run.
- Results can be listed, filtered by task, and read.
- Summary metrics are generated.
- Unsupported graph tasks are explicit and safe.
- The evaluator is read-only.
- The release checker passes.
- The v1.7.0 version document, readiness document, and release notes exist.
```

## Next version handoff

The recommended next version is:

```text
v1.8.0 — Connector-driven re-extraction
```

Suggested goal:

```text
When manual, Google Drive, or Notion source data changes, Cornerstone can identify affected evidence and safely queue ontology re-extraction without automatically mutating the official graph.
```

## Chronicle position and measurable release checklist

This section was added during the v2.0.0 documentation chronicle pass so the version goal and checklist are explicit, measurable, and easy to audit.

**Version goal:** Add read-only evaluation tasks/results/summary for ontology graph quality and SSOT safety.

**Confirmed non-goal:** No graph mutation, no semantic/vector scoring, no extraction or review change.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V170-01 | Ontology evaluation task creation, run, list, result, and summary endpoints exist. | API contract and integration tests. | complete |
| V170-02 | Evaluation checks evidence validity, provenance, trust label, freshness, official safety, and candidate boundary. | Quality gates and result fields. | complete |
| V170-03 | Evaluation results persist in memory and SQLAlchemy models. | Persistence model/migration docs and tests. | complete |
| V170-04 | Evaluation service calls graph service read-only. | Trust boundary section and tests. | complete |
| V170-05 | Summary reports measurable rates for graph/task success and safety. | Metrics section. | complete |

**Chronicle handoff:** This version is recorded in `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. Later versions may build on this release only within the confirmed non-goal boundary above.


# Cornerstone v2.0.0 — Ontology SSOT Release

## Purpose

Cornerstone v2.0.0 stabilizes the backend contract for an evidence-grounded ontology Single Source of Truth. The goal is not to add another ontology feature. The goal is to make the full loop operable, explainable, and safe enough to treat the reviewed official graph as the product boundary.

The v2.0.0 loop is:

```text
source / manual upload
→ Artifact
→ EvidenceFragment
→ ontology re-extraction
→ ConceptCandidate / RelationCandidate
→ evidence review
→ candidate review
→ official Concept / official ConceptRelation
→ explainable depth-1 graph
→ ontology evaluation
→ operator SSOT readiness checklist
```

## Product goal

A user or operator should be able to ask whether a focus concept such as `settlement` is ready to serve as an official ontology graph. Cornerstone should answer with a checklist, not a vague status.

A successful v2.0.0 readiness result means:

```text
- source-backed evidence exists
- the focus concept has an official graph
- the graph depth policy is respected
- official graph safety gates pass
- every served citation is valid
- review provenance is present
- candidate/non-official objects remain outside official mode
- a successful ontology evaluation result exists
```

## Non-goals

v2.0.0 intentionally does not add:

```text
- automatic Concept approval
- automatic Relation approval
- automatic official graph mutation from connector sync
- live external LLM provider integration
- graph depth above 1
- graph visualization UI
- semantic/vector graph expansion
- background asynchronous execution
- destructive cleanup of stale candidates
```

The reviewed official graph remains the Single Source of Truth. Raw source documents, connector sync output, LLM/extractor output, and pending candidates are not the SSOT.

## New API

```http
GET /v1/ontology/ssot/readiness?focusConcept=settlement&depth=1&mode=official&includeGraph=false
```

Query parameters:

```text
focusConcept: concept name or alias, default settlement
depth: 0 or 1, default 1
mode: official / candidate / mixed, default official
includeGraph: boolean, default false
```

The endpoint is read-only. It must not create source data, run extraction, run re-extraction, review evidence, approve candidates, merge candidates, create official graph objects, run evaluation, or delete anything.

## Response contract

The response includes:

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

Each readiness check includes:

```text
key
title
category
goal
required
status
detail
checks
objectIds
nextActions
```

## Required checks

The v2.0.0 readiness service verifies:

```text
source_ingestion_available
official_graph_available
graph_depth_policy_respected
official_graph_safe
evidence_citations_valid
review_provenance_present
candidate_boundary_respected
ontology_evaluation_available
operator_proof_path_available
```

## CLI

```bash
cornerstone proof run --ssot-readiness --ontology-focus-concept Settlement --base-url http://localhost:8000
```

This CLI scope calls the read-only readiness endpoint. It is safe to include in release proof checks because it does not mutate backend state.

## Operator sequence

A typical operator sequence is:

```bash
cornerstone proof run --ontology-loop \
  --confirm-ontology-mutation \
  --ontology-focus-concept Settlement \
  --base-url http://localhost:8000

cornerstone proof run --ssot-readiness \
  --ontology-focus-concept Settlement \
  --base-url http://localhost:8000
```

The first command can create explicit proof data and promote proof candidates through normal review gates. The second command inspects whether the resulting graph is safe to treat as official SSOT.

## Trust boundary

The v2.0.0 release keeps these rules:

```text
No reviewed evidence → no official Concept.
No reviewed evidence → no official Relation.
LLM/extractor output → candidate only.
Official graph mode → official Concepts and Relations only.
Depth=1 → default and maximum release graph depth.
Readiness endpoint → read-only.
```

## Implementation checklist

```text
[x] Package version updated to 2.0.0.
[x] Added OntologySsotReadinessResponse schema.
[x] Added OntologySsotReadinessCheck schema.
[x] Added OntologySsotReadinessService.
[x] Added GET /v1/ontology/ssot/readiness.
[x] Added CLI --ssot-readiness scope.
[x] Added integration tests for pre-proof and post-proof readiness.
[x] Updated API contract documentation.
[x] Updated release checker and release documentation tests.
[x] Added v2.0.0 release readiness document.
[x] Added v2.0.0 operator checklist.
[x] Added v2.0.0 release notes.
```

## Exit criteria

v2.0.0 is ready when:

```text
- release candidate check passes
- syntax compilation passes
- SSOT readiness integration tests pass
- release docs tests pass
- v2.0.0 docs are present and actionable
- operator proof followed by readiness returns status=passed for Settlement
```

## Known limitations

```text
- The readiness endpoint checks current state only; it does not repair missing setup.
- It depends on successful ontology evaluation having already been run.
- It does not visualize the graph.
- It does not perform semantic quality scoring.
- It does not verify live Notion/Google Drive connectors unless operators run those proof paths separately.
```

## Next version handoff

After v2.0.0, future work should focus on production hardening rather than expanding scope:

```text
- live external LLM provider behind the candidate-only boundary
- UI for evidence/candidate review
- UI graph visualization of the existing response contract
- stale candidate cleanup and re-evaluation policies
- persistent-store proof runs in dependency-complete environments
```

## Chronicle position and measurable release checklist

This section was added during the v2.0.0 documentation chronicle pass so the version goal and checklist are explicit, measurable, and easy to audit.

**Version goal:** Stabilize the backend SSOT contract and expose read-only readiness checks for a focus concept.

**Confirmed non-goal:** No new feature spike, no visual UI, no automatic approval, no graph depth above 1.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V200-01 | `GET /v1/ontology/ssot/readiness` returns a read-only readiness checklist. | API contract and tests. | complete |
| V200-02 | Readiness checks source ingestion, official graph availability, depth policy, graph safety, citations, provenance, candidate boundary, evaluation, and proof path. | Required checks section. | complete |
| V200-03 | CLI supports `cornerstone proof run --ssot-readiness`. | CLI section and tests. | complete |
| V200-04 | A successful readiness result requires official graph safe, trust label official, and successful latest evaluation. | Exit criteria and readiness document. | complete |
| V200-05 | The release documents the SSOT trust boundary and operator sequence. | Trust boundary and operator checklist docs. | complete |

**Chronicle handoff:** This version is recorded in `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. Later versions may build on this release only within the confirmed non-goal boundary above.


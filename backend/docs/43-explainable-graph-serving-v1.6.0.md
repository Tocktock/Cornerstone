# Cornerstone v1.6.0 — Explainable Graph Serving Enhancements

## Purpose

`v1.6.0` improves ontology graph serving so users can understand **why** a graph is official, supported, partial, stale, conflicted, or unsupported.

This version builds on:

```text
v1.3.0 → ontology search and depth-1 graph runtime
v1.4.0 → candidate-only ontology extraction
v1.5.0 → candidate review and promotion workflow
```

The goal is not to create more ontology content. The goal is to make existing graph content easier to trust, audit, and operate.

## Product goal

When a user asks for a concept such as `settlement`, Cornerstone should return more than nodes and edges. It should also explain:

```text
- what graph scope was served
- why the graph has its trust label
- which Concepts and Relations are official
- which evidence supports each node and edge
- who created/reviewed/officialized graph objects
- whether pending candidates exist outside the official graph
- what the user should review next if the graph is incomplete
```

## Non-goals

`v1.6.0` intentionally does not add:

```text
- no new LLM provider
- no new extraction prompts
- no automatic candidate approval
- no automatic graph edge inference
- no graph visualization UI
- no graph depth above 1
- no graph database
- no semantic/vector search
- no batch review workflow
```

## Single Source of Truth boundary

The trust boundary remains unchanged:

```text
Raw source document ≠ Single Source of Truth
LLM/extractor output ≠ Single Source of Truth
Candidate object ≠ Single Source of Truth
Reviewed official ontology graph = Single Source of Truth
```

In official mode, candidate objects are not promoted or served as official graph nodes/edges. Pending candidates are only summarized so reviewers know that additional review work may exist.

## Runtime behavior

### Existing endpoint enhanced

```http
GET /v1/ontology/graph?concept=settlement&depth=1&mode=official
```

The endpoint still returns a depth-limited graph, but now includes additional explainability fields:

```text
supportSummary
candidateSummary
explanation
node.reviewProvenance
node.supportSummary
node.explanation
edge.sourceConceptName
edge.targetConceptName
edge.focusDirection
edge.reviewProvenance
edge.supportSummary
edge.explanation
evidence.dataSourceId
evidence.sourceType
evidence.sourceExternalId
evidence.reviewedBy
evidence.reviewedAt
```

### New explanation endpoint

```http
GET /v1/ontology/explain?concept=settlement&depth=1&mode=official
```

This endpoint returns the same response shape as `/graph`. It exists for clients that want to explicitly request an explanation-first graph response.

## Graph response explanation

The `explanation` object contains:

```text
summary
ssotStatus
trustReason
graphScope
evidencePolicy
candidateBoundary
reviewSummary
recommendedNextActions
```

Example:

```json
{
  "summary": "Settlement graph served 4 node(s), 3 direct relation(s), and 5 serving citation(s) at depth 1.",
  "ssotStatus": "official_ssot",
  "trustReason": "The focus Concept, visible neighbor Concepts, visible Relations, and serving citations are reviewed and official.",
  "graphScope": "Depth 1 includes the focus Concept and directly connected Concepts only.",
  "evidencePolicy": "Citations include serving-eligible EvidenceFragments for the focus Concept, visible neighbor Concepts, visible ConceptRelations, and linked DecisionRecords.",
  "candidateBoundary": "Candidate objects are excluded from official graph mode.",
  "reviewSummary": "4/4 node(s) official; 3/3 edge(s) official; 5/5 serving citation(s) reviewed; freshness=fresh.",
  "recommendedNextActions": ["Use the cited evidence and review provenance to audit the graph before downstream reuse."]
}
```

## Support summary

The top-level `supportSummary` summarizes graph-level serving support:

```text
nodeCount
edgeCount
evidenceCount
reviewedEvidenceCount
unreviewedEvidenceCount
conflictedEvidenceCount
staleEvidenceCount
unknownFreshnessEvidenceCount
officialNodeCount
nonOfficialNodeCount
officialEdgeCount
nonOfficialEdgeCount
invalidEvidenceReferenceCount
invalidRelationCount
```

Each node and edge also has its own `supportSummary`. This lets the UI highlight unsupported or partially supported objects without forcing the user to manually inspect every citation.

## Review provenance

Each node and edge now exposes `reviewProvenance`:

```text
createdBy
officializedBy
reviewedBy
createdAt
updatedAt
lastReviewedAt
status
```

This makes the graph auditable:

```text
Settlement node → who created and officialized it
Clearing precedes Settlement edge → who officialized that relation
Evidence citation → who reviewed the supporting evidence and when
```

## Candidate summary

The graph response now includes `candidateSummary` for the graph focus:

```text
pendingConceptCandidateCount
pendingRelationCandidateCount
approvedConceptCandidateCount
approvedRelationCandidateCount
rejectedConceptCandidateCount
rejectedRelationCandidateCount
mergedConceptCandidateCount
mergedRelationCandidateCount
hasPendingCandidates
note
```

This does not include pending candidates as graph nodes/edges. It only tells the reviewer whether more candidate work exists.

Example official graph behavior:

```text
Official graph for Settlement has 2 official edges.
Candidate summary says 3 pending RelationCandidates also mention Settlement.
Those 3 candidates are not in the official graph until reviewed.
```

## Evidence policy change

`v1.3.0` cited the focus Concept and visible Relations. `v1.6.0` expands graph citations to include:

```text
- focus Concept evidence
- visible neighbor Concept evidence
- visible ConceptRelation evidence
- linked DecisionRecord evidence
```

This makes depth-1 graph responses more explainable because neighbor nodes are no longer displayed without their own supporting citations when that evidence is available.

## Settlement reference behavior

For an official graph:

```text
Settlement --updates--> Ledger
Settlement --follows--> Clearing
Settlement --validates--> Reconciliation
```

A v1.6.0 response should explain:

```text
- Settlement is official.
- Ledger/Clearing/Reconciliation are official visible neighbor Concepts.
- Each visible edge is official.
- Each visible node/edge has serving-eligible citations.
- Pending candidate edges, if any, are excluded but summarized.
- The graph is depth 1 by default.
```

## Implementation checklist

```text
[x] Add graph-level supportSummary.
[x] Add graph-level candidateSummary.
[x] Add graph-level explanation.
[x] Add node reviewProvenance.
[x] Add node supportSummary.
[x] Add node explanation.
[x] Add edge source/target display names.
[x] Add edge focusDirection.
[x] Add edge reviewProvenance.
[x] Add edge supportSummary.
[x] Add edge explanation.
[x] Add citation dataSource/source/review metadata.
[x] Include neighbor Concept evidence in graph citations.
[x] Add /v1/ontology/explain endpoint.
[x] Keep depth maximum at 1.
[x] Keep official mode candidate-free.
[x] Add v1.6.0 tests and release documentation.
```

## API compatibility

The response shape is additive. Existing fields remain:

```text
focusConcept
nodes
edges
evidence
freshness
trustLabel
limitations
generatedAt
officialGraphAvailable
```

New clients can use the explanation fields. Existing clients can continue using the old graph fields.

## Known limitations

```text
1. The backend still does not render a visual graph UI.
2. Depth above 1 remains rejected.
3. CandidateSummary is lexical/focus-based, not semantic clustering.
4. Candidate objects are summarized but not embedded as graph nodes/edges.
5. No live external LLM provider is added.
6. No graph database is introduced.
```

## Exit criteria

`v1.6.0` is complete when:

```text
1. The package version is 1.6.0.
2. /v1/ontology/graph includes explanation, support summary, candidate summary, and provenance metadata.
3. /v1/ontology/explain returns the same explainable graph response shape.
4. Official mode still excludes candidate graph objects.
5. Pending candidates are summarized separately from graph nodes/edges.
6. Depth above 1 remains rejected.
7. Compile and targeted explainability smoke tests pass.
8. Release docs and API freeze docs include v1.6.0.
```

## Next version handoff

Recommended next version:

```text
v1.7.0 — Ontology Evaluation
```

Suggested goal:

```text
Measure whether extracted/reviewed Concepts and Relations are supported, correctly promoted, citation-valid, and graph-consistent.
```

Candidate checks:

```text
- unsupported official edge count
- official graph citation validity
- relation evidence support quality
- duplicate Concept/alias detection
- extraction candidate precision/recall against fixture documents
- settlement graph golden test
```

## Chronicle position and measurable release checklist

This section was added during the v2.0.0 documentation chronicle pass so the version goal and checklist are explicit, measurable, and easy to audit.

**Version goal:** Make served ontology graphs explain why they are official, supported, partial, stale, conflicted, or unsupported.

**Confirmed non-goal:** No new extraction, approval, or graph mutation behavior.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V160-01 | Graph responses include graph-level supportSummary, candidateSummary, and explanation. | Schema and unit verification. | complete |
| V160-02 | Nodes include reviewProvenance, supportSummary, and explanation. | Schema/service tests. | complete |
| V160-03 | Edges include direction, reviewProvenance, supportSummary, and explanation. | Schema/service tests. | complete |
| V160-04 | Citations include source and review metadata. | Citation schema and tests. | complete |
| V160-05 | `GET /v1/ontology/explain` returns the explanation-first graph response. | API contract and tests. | complete |

**Chronicle handoff:** This version is recorded in `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. Later versions may build on this release only within the confirmed non-goal boundary above.


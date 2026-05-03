# How Cornerstone Works

## The system in one diagram

```text
1. Source ingestion
   Manual uploads, manual sync, Google Drive, Notion, future connectors
        ↓
2. Artifact creation
   A captured version of a source object
        ↓
3. Evidence extraction
   Source text becomes EvidenceFragments
        ↓
4. Ontology extraction
   Extractor proposes ConceptCandidates and RelationCandidates
        ↓
5. Human review
   Reviewer approves, rejects, edits, or merges candidates
        ↓
6. Official ontology graph
   Reviewed Concepts and ConceptRelations become the SSOT
        ↓
7. Explainable serving
   Users query Concepts and receive graph, citations, trust, and provenance
        ↓
8. Evaluation and readiness
   Operators measure graph safety and proof completeness
```

## 1. Source ingestion

Source ingestion gathers knowledge from supported inputs.

Current supported practical inputs include:

```text
- manual source sync
- manual text-like file upload
- Google Drive Docs and text files
- Notion pages
```

Source ingestion does not create truth. It creates source-backed material that can become evidence.

## 2. Artifact creation

An Artifact is the captured backend representation of a source object.

Examples:

```text
- uploaded settlement.md
- Google Doc exported as text
- selected Notion page
```

Artifact identity and content allow Cornerstone to detect whether source material is new, reused, or changed.

## 3. Evidence extraction

EvidenceFragments are the pieces of source material that can support claims.

A source sentence such as:

```text
Clearing precedes settlement.
```

can become an EvidenceFragment used to support:

```text
Clearing --precedes--> Settlement
```

## 4. Ontology extraction

The extractor proposes ontology structure.

It may propose:

```text
ConceptCandidate: Settlement
ConceptCandidate: Clearing
RelationCandidate: Clearing --precedes--> Settlement
```

Extractor output remains candidate-only.

## 5. Human review

Reviewers decide which proposed Concepts and Relations become official.

They can:

```text
- approve
- reject
- edit
- merge
```

Approval must still pass evidence and officialization gates.

## 6. Official ontology graph

The official graph is made of reviewed Concepts and reviewed ConceptRelations.

This graph is the product's Single Source of Truth.

## 7. Explainable serving

Graph serving answers questions such as:

```text
What is Settlement?
What is directly related to Settlement?
Why is this official?
What evidence supports this Relation?
```

Default graph depth is `1`, which means direct neighbors only.

## 8. Evaluation and readiness

Evaluation checks graph quality. Readiness checks whether the current state is safe to serve.

Examples:

```text
POST /v1/evaluations/ontology/run
GET /v1/ontology/ssot/readiness?focusConcept=Settlement
```

## How Cornerstone works acceptance checklist

| Check ID | Requirement | Measurement | Status |
|---|---|---|---|
| PROD-HOW-01 | Full product loop is described. | Eight-step sequence from source ingestion to readiness. | complete |
| PROD-HOW-02 | Source ingestion is not confused with truth. | Explicitly says source ingestion does not create truth. | complete |
| PROD-HOW-03 | Artifact and EvidenceFragment are defined. | Provides examples and role in the loop. | complete |
| PROD-HOW-04 | Candidate-only extraction boundary is explicit. | Extractor output remains candidate-only. | complete |
| PROD-HOW-05 | Evaluation/readiness are included. | Describes final safety and operator checks. | complete |

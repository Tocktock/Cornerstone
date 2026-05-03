# Settlement Walkthrough

This walkthrough is the primary product example for Cornerstone.

It shows how scattered source text becomes an official, explainable ontology graph.

## Starting point

A user or operator uploads settlement notes:

```text
Settlement is the process of finalizing financial obligations.
Clearing happens before settlement.
Reconciliation validates settlement results.
Settlement updates the ledger after obligations are finalized.
```

The user wants to know:

```text
What is settlement?
How is settlement related to other concepts?
Why should I trust the answer?
```

## Step 1 — Source material is captured

The uploaded file becomes source-backed data:

```text
SourceObject: settlement-notes.md
Artifact: captured text content
```

This is not yet official truth.

## Step 2 — EvidenceFragments are created

Cornerstone extracts source-backed evidence:

```text
EvidenceFragment A: Settlement is the process of finalizing financial obligations.
EvidenceFragment B: Clearing happens before settlement.
EvidenceFragment C: Reconciliation validates settlement results.
EvidenceFragment D: Settlement updates the ledger after obligations are finalized.
```

These fragments can later support Concepts and Relations.

## Step 3 — Ontology extraction proposes candidates

The extractor proposes Concepts:

```text
ConceptCandidate: Settlement
ConceptCandidate: Clearing
ConceptCandidate: Reconciliation
ConceptCandidate: Ledger
```

It also proposes Relations:

```text
RelationCandidate: Clearing --precedes--> Settlement
RelationCandidate: Reconciliation --validates--> Settlement
RelationCandidate: Settlement --updates--> Ledger
```

These candidates are proposals, not truth.

## Step 4 — Reviewer approves or rejects

A reviewer inspects evidence and candidates.

Possible actions:

```text
Approve Settlement as an official Concept.
Approve Clearing as an official Concept.
Approve Ledger as an official Concept.
Approve Clearing --precedes--> Settlement.
Reject a weak or unsupported candidate.
Merge duplicate aliases such as payment settlement and settlement.
```

Only approved candidates that pass evidence gates become official graph objects.

## Step 5 — Official graph is served

Now a user asks:

```text
What is Settlement?
```

Cornerstone can serve a depth-1 graph:

```text
Clearing --------precedes--------> Settlement
Reconciliation --validates-------> Settlement
Settlement ------updates---------> Ledger
```

It can also explain:

```text
- Settlement is official.
- The direct graph depth is 1.
- Each edge has supporting EvidenceFragments.
- Pending candidates are excluded from official mode.
- Review provenance is included.
```

## Example answer shape

```text
Settlement is the reviewed official Concept for finalizing financial obligations.

Direct Relations:
- Clearing precedes Settlement.
- Reconciliation validates Settlement.
- Settlement updates Ledger.

Why this is trusted:
- The Concept and Relations were reviewed.
- Each claim cites source-backed EvidenceFragments.
- The graph is served in official mode.
- Pending candidates are not included.
```

## What happens if evidence is missing

If a proposed Relation has no supporting EvidenceFragment, it must remain unsupported or rejected.

```text
No evidence → no official Relation.
```

## What happens if review is missing

If candidates exist but are not reviewed, the official graph should not include them.

```text
Candidate exists → useful for review.
Candidate is not official → excluded from official graph mode.
```

## What happens if documents disagree

If two evidence fragments conflict, the graph should expose a conflict or require review before officializing.

Cornerstone should not hide uncertainty behind fluent text.

## Settlement walkthrough acceptance checklist

| Check ID | Requirement | Measurement | Status |
|---|---|---|---|
| PROD-SETTLEMENT-01 | Walkthrough uses one concrete source example. | Settlement notes are shown as input text. | complete |
| PROD-SETTLEMENT-02 | Evidence extraction is visible. | Four EvidenceFragments are shown. | complete |
| PROD-SETTLEMENT-03 | Candidate-only boundary is visible. | Concepts and Relations are proposed as candidates first. | complete |
| PROD-SETTLEMENT-04 | Review workflow is visible. | Approve, reject, and merge examples are included. | complete |
| PROD-SETTLEMENT-05 | Final official graph is explainable. | Depth-1 graph, citations, and trust rationale are described. | complete |

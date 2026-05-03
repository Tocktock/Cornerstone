# Trust Model

## Core rule

Cornerstone is designed around one trust rule:

```text
Raw source data is not the Single Source of Truth.
Extractor output is not the Single Source of Truth.
Pending candidates are not the Single Source of Truth.
The reviewed official ontology graph is the Single Source of Truth.
```

## Why this rule exists

Documents can be wrong. Documents can be stale. Different documents can disagree.

Models can also be wrong. They can extract a plausible Concept or Relation that the evidence does not actually support.

Cornerstone uses the extractor to help humans discover structure, not to replace review.

## Trust states

Cornerstone responses should make state visible.

```text
official            reviewed and safe to serve as official graph truth
evidence_supported  supported by evidence but not necessarily official
candidate           proposed, awaiting review
unsupported         no adequate evidence or no official object
stale               evidence or source freshness is outside policy
conflicted          evidence or graph state contains unresolved conflict
partial             answer is incomplete or constrained by available evidence
```

## Evidence requirements

For official ontology graph claims:

```text
Concept must have supporting evidence.
Relation must have supporting evidence.
Evidence must be traceable to source material.
Review provenance must be available.
```

## Candidate boundary

Candidates help reviewers work faster.

They must not be presented as official truth.

```text
ConceptCandidate is not Concept.
RelationCandidate is not ConceptRelation.
Pending candidate is not official graph object.
```

## Official graph boundary

In official graph mode:

```text
- only official Concepts should appear as official nodes
- only official ConceptRelations should appear as official edges
- pending candidates may be summarized, but not included as official graph truth
- every official node/edge should be explainable through evidence and review provenance
```

## What users should see

Users should see not only an answer, but also why the answer is trustworthy.

```text
Answer: Settlement updates Ledger.
Evidence: source fragment from settlement-notes.md.
Reviewer: reviewer@example.com.
Trust: official.
Mode: official.
Depth: 1.
```

## Failure modes that must be visible

Cornerstone should expose limitations such as:

```text
- no official Concept found
- official Concept exists but has no official Relations
- pending candidates exist outside official graph
- evidence is stale
- evidence is conflicted
- latest evaluation failed
```

## Trust model acceptance checklist

| Check ID | Requirement | Measurement | Status |
|---|---|---|---|
| PROD-TRUST-01 | SSOT trust boundary is explicit. | Core rule appears at the top. | complete |
| PROD-TRUST-02 | LLM/extractor risk is addressed. | Extractor helps discovery, not review replacement. | complete |
| PROD-TRUST-03 | Trust states are listed. | Official, candidate, unsupported, stale, conflicted, partial are defined. | complete |
| PROD-TRUST-04 | Official graph mode is constrained. | Candidate exclusion and evidence/review requirements are stated. | complete |
| PROD-TRUST-05 | Failure modes are visible. | Limitations list is included. | complete |

# ADR 0001 — Ontology Official Trust Boundary

## Status

Accepted for `v1.2.1`.

## Context

Cornerstone is evolving toward an ontology graph that represents an organization's Single Source of Truth.

The system will eventually use an LLM to propose Concepts and Relations from source evidence. This creates a trust risk: users may treat LLM-generated graph objects as official organizational truth even when they have not been reviewed.

Cornerstone already has a trust model for EvidenceFragments, Concepts, ConceptRelations, officialization, freshness, and grounded responses. The ontology direction must preserve that trust model.

## Decision

LLM output must be candidate-only.

The official ontology graph must contain only reviewed, evidence-backed Concepts and ConceptRelations.

The application must not allow an LLM extraction run to directly create official Concepts or official ConceptRelations.

The official graph is defined as:

```text
official Concepts + official ConceptRelations + reviewed supporting evidence
```

Candidate Concepts and RelationCandidates may be stored, displayed, evaluated, and reviewed, but they are not the Single Source of Truth.

## Rules

```text
1. LLM output can create ConceptCandidates.
2. LLM output can create RelationCandidates.
3. LLM output cannot create official Concepts directly.
4. LLM output cannot create official ConceptRelations directly.
5. Every candidate must reference evidence.
6. Every official Concept must be supported by reviewed evidence or accepted DecisionRecord.
7. Every official Relation must be supported by reviewed evidence or accepted DecisionRecord.
8. Candidate graph mode must be visibly labeled.
9. Official graph mode must exclude candidates.
10. Mixed graph mode must label every object by status.
```

## Rationale

The product goal is explainable Single Source of Truth, not automated knowledge generation.

Users need to trust that:

```text
- source evidence exists
- evidence was reviewed
- Concepts are not invented
- Relations are not invented
- the graph can explain itself
```

An LLM can speed discovery, but it cannot be the authority.

## Consequences

Positive:

```text
- protects user trust
- reduces hallucination risk
- keeps official graph auditable
- supports reviewer accountability
- makes evidence coverage measurable
- preserves existing Cornerstone trust model
```

Negative:

```text
- review workflow adds product complexity
- candidate queues must be built
- useful LLM suggestions remain unofficial until reviewed
- users need merge and duplicate controls
```

## Rejected alternatives

### LLM writes official graph directly

Rejected. This would make the model the authority.

### Auto-officialize high-confidence candidates

Rejected for initial ontology releases. Model confidence can prioritize review but cannot replace review.

### Treat raw source documents as the Single Source of Truth

Rejected. Raw documents may conflict, age, or contain ambiguous language.

### Use a graph database immediately

Rejected for MVP. Existing Concepts and ConceptRelations plus PostgreSQL are enough for depth-1 and depth-2 graph serving.

## Implementation impact

Future implementation should add:

```text
OntologyExtractionRun
ConceptCandidate
RelationCandidate
ConceptAlias
OntologyGraphResponse
```

Existing `Concept` and `ConceptRelation` remain the official graph layer.

## Testing impact

Future tests must prove:

```text
- LLM extraction cannot create official Concepts.
- LLM extraction cannot create official Relations.
- Candidate approval requires authorized reviewer.
- Official graph mode excludes candidates.
- Official graph objects require evidence.
- Unsupported queries do not fabricate graph answers.
```

## Review

Revisit before `v2.0.0` if automatic promotion is proposed. Automatic promotion should only be considered with deterministic evidence validation, strong evaluation data, and explicit operator opt-in.

# Ontology Graph Explained

## Simple definition

An ontology graph is a map of organizational meaning.

It contains:

```text
Concepts  = things the organization needs to define
Relations = how those Concepts connect
Evidence  = why Cornerstone believes a Concept or Relation
```

Example:

```text
Settlement --updates--> Ledger
```

This means:

```text
The official Concept Settlement has an official Relation to the official Concept Ledger, and reviewed evidence says settlement updates the ledger.
```

## Concept

A Concept is a reviewed thing the organization wants to define.

Examples:

```text
Settlement
Clearing
Ledger
Reconciliation
Settlement Policy
```

A Concept should have:

```text
- name
- short definition
- aliases
- evidence support
- review state
- official status when approved
```

## Relation

A Relation explains how two Concepts connect.

Examples:

```text
Clearing --precedes--> Settlement
Reconciliation --validates--> Settlement
Settlement --updates--> Ledger
Settlement --governed_by--> Settlement Policy
```

A Relation should have:

```text
- source Concept
- target Concept
- relation type
- evidence support
- review provenance
- official status when approved
```

## Evidence

Evidence is source-backed support for a claim.

The graph should not ask users to trust a relation only because a model said it.

```text
No evidence → no official Concept.
No evidence → no official Relation.
No review → no official graph claim.
```

## Candidate vs official

A candidate is a proposal.

```text
ConceptCandidate: Settlement
RelationCandidate: Settlement --updates--> Ledger
```

An official graph object is reviewed and approved.

```text
Concept: Settlement, status official
ConceptRelation: Settlement --updates--> Ledger, status official
```

## Graph depth

Default depth is `1`.

Depth 1 means:

```text
Show the focus Concept.
Show directly connected Concepts.
Do not recursively expand the entire graph.
```

For `Settlement`, depth 1 might include `Clearing`, `Ledger`, and `Reconciliation`, but not every Concept connected to those neighbors.

## Why depth 1 matters

Depth 1 keeps graph answers:

```text
- readable
- auditable
- focused
- less likely to overstate indirect relationships
```

Higher depth can be added later, but it should not be the default.

## Ontology graph acceptance checklist

| Check ID | Requirement | Measurement | Status |
|---|---|---|---|
| PROD-GRAPH-01 | Ontology graph is explained without graph-database jargon. | Defines graph as map of organizational meaning. | complete |
| PROD-GRAPH-02 | Concept, Relation, and Evidence are distinct. | Each has a dedicated section. | complete |
| PROD-GRAPH-03 | Candidate vs official is clear. | Shows candidate proposal vs official reviewed object. | complete |
| PROD-GRAPH-04 | Depth 1 is clear. | Defines direct-neighbor-only behavior. | complete |
| PROD-GRAPH-05 | Evidence rule is clear. | States no evidence/review means no official claim. | complete |

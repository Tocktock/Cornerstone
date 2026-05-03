# Product Glossary

This glossary defines the core Cornerstone product terms used across product, technical, and release documentation.

## Source

A configured origin of knowledge, such as manual input, Google Drive, or Notion.

## SourceObject

A provider-specific object selected for ingestion, such as a Google Doc, Notion page, or manually supplied object.

## Artifact

The captured backend representation of a source object at a point in time.

## EvidenceFragment

A source-backed piece of content that can support a Concept, Relation, or answer.

## Concept

A reviewed ontology node that represents a thing the organization needs to define.

Example:

```text
Settlement
```

## ConceptCandidate

A proposed Concept created by extraction. It is not official until reviewed and approved.

## Relation

A connection between two Concepts.

Example:

```text
Settlement --updates--> Ledger
```

## RelationCandidate

A proposed Relation created by extraction. It is not official until reviewed and approved.

## OntologyGraph

A graph made of Concepts and Relations. In official mode, it should contain only official reviewed graph objects.

## Official Graph

The reviewed ontology graph that acts as the Single Source of Truth.

## Candidate Graph

An exploratory view of proposed or non-official ontology objects. It is useful for reviewers but is not the Single Source of Truth.

## Single Source of Truth

The reviewed official ontology graph. Raw documents and LLM output are inputs, not the SSOT.

## Review

Human action that approves, rejects, edits, or merges evidence or ontology candidates.

## Officialization

The process of making a reviewed Concept or Relation official after gates pass.

## Depth

The graph expansion distance from a focus Concept. Default depth is `1`, meaning direct neighbors only.

## Citation

A reference from a graph claim or answer back to supporting evidence.

## Provenance

Metadata about where knowledge came from and who reviewed it.

## Evaluation

A read-only quality check that measures graph safety, evidence validity, provenance, trust label correctness, freshness, and candidate boundary.

## Readiness

A read-only operator check that reports whether a focus Concept's official graph is safe to serve.

## Trust Label

A user-visible state such as `official`, `candidate`, `unsupported`, `stale`, `conflicted`, or `partial`.

## Glossary acceptance checklist

| Check ID | Requirement | Measurement | Status |
|---|---|---|---|
| PROD-GLOSSARY-01 | Core source/evidence terms are defined. | Source, SourceObject, Artifact, EvidenceFragment are included. | complete |
| PROD-GLOSSARY-02 | Core ontology terms are defined. | Concept, ConceptCandidate, Relation, RelationCandidate are included. | complete |
| PROD-GLOSSARY-03 | Graph and SSOT terms are defined. | OntologyGraph, Official Graph, Candidate Graph, SSOT are included. | complete |
| PROD-GLOSSARY-04 | Review and trust terms are defined. | Review, Officialization, Citation, Provenance, Trust Label are included. | complete |
| PROD-GLOSSARY-05 | Operator terms are defined. | Evaluation and Readiness are included. | complete |

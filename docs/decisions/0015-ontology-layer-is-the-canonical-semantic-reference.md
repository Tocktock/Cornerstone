# 0015 - Ontology layer is the canonical semantic reference

## Status
Accepted

## Decision

Cornerstone maintains a canonical ontology layer that owns:
- the abstract class hierarchy
- the concrete semantic object taxonomy
- the semantic graph predicate taxonomy
- the distinction between graph predicates and non-graph structural links
- the projection ontology for glossary, graph, answer, passport, and provenance views

Feature specs may refine behavior, but they must not redefine those semantic objects or relations inconsistently.

## Why

Cornerstone has reached the point where feature-level prose alone is not enough to keep object meanings and relation semantics stable.

Without an ontology layer:
- object kinds drift across specs
- semantic graph predicates and lineage links get mixed together
- promoted personal support is modeled inconsistently
- serving-contract payloads lose a single semantic owner

## Consequences

- New canonical object kinds require an ontology update.
- New semantic graph predicates require an ontology update.
- Domain model, serving contract, and feature specs must align to the ontology rather than redefining class meaning locally.

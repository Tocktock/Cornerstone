# 0012 - Serving contract is canonical across UI, API, and MCP surfaces

- **Status:** Accepted

## Context

Cornerstone serves the same truth layer to humans, services, and models. Without one contract, surfaces would drift in shape and trust semantics.

## Decision

UI, programmatic API, and MCP/model-facing surfaces must use one canonical serving contract covering response kinds, required fields, cardinality, support-visibility semantics, and provenance semantics.

## Consequences

- Transport details remain replaceable.
- Surface behavior may not redefine the meaning of concepts, answers, or trust labels.

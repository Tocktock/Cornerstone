# 0004 - Decision records are first-class

- **Status:** Accepted

## Context

Definitions and relationships are not enough. Users also need durable rationale for why something is true here.

## Decision

Decision context is first-class and is stored as `DecisionRecord` rather than hidden inside notes, threads, or implementation-specific metadata.

## Consequences

- Decision-facing surfaces are required.
- Decision records participate in retrieval, provenance, and official context.

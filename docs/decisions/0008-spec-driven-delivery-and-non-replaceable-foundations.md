# 0008 - Spec-driven delivery and non-replaceable foundations

- **Status:** Accepted

## Context

The product spans multiple layers and consumers. Without documented foundations, implementation details will become accidental product decisions.

## Decision

Cornerstone is docs-first and spec-driven. Non-replaceable foundations are part of the product contract and must be changed intentionally through docs.

## Consequences

- Meaningful feature work requires an owning spec.
- Durable boundaries require an updated decision record.
- Implementation may not redefine canonical behavior.

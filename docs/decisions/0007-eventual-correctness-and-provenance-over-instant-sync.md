# 0007 - Eventual correctness and provenance over instant sync

- **Status:** Accepted

## Context

Not every source can or should behave as a strict real-time feed, but trust still depends on recoverability and provenance.

## Decision

Cornerstone prioritizes eventual correctness, provenance, and recoverability over strict real-time sync guarantees.

## Consequences

- Source freshness must be surfaced honestly.
- Drift or failure reopens review when needed, rather than silently preserving stale trust.

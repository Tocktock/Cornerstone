# 0005 - Glossary and graph are projections over curated entities

- **Status:** Accepted

## Context

User-facing glossary and graph views are important, but they should not become separate storage roots with independent truth.

## Decision

Glossary, graph, and answer views are projections over curated entities, relations, decisions, and support summaries.

## Consequences

- The canonical domain model stays small and stable.
- Serving surfaces must not redefine what the underlying entities mean.

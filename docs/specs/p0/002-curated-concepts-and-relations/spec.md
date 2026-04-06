# P0-002 Curated Concepts and Relations

## Status

Draft

## Summary

Cornerstone must allow workspace context to be curated into canonical concepts and first-class concept relations.

## Goals

- create canonical concepts that represent official organizational meaning
- create first-class concept relations that represent how organizational meaning fits together
- preserve stable identities, owning domains, and review domains
- keep support and trust visible enough for later review and retrieval

## Functional requirements

- The product must allow draft creation and refinement of concepts and relations inside one workspace.
- Every shared concept must carry one `owning_domain` and one `review_domain`.
- Every shared relation must carry one `review_domain` derived deterministically from its endpoints.
- Concepts and relations must support one or more support items drawn from `EvidenceFragment`, `PromotedSupport`, accepted decision lineage, or combinations allowed by policy.
- Concepts and relations may become official only through review.
- Glossary and graph views must be projections over the same curated entities, not separate storage roots.

## Acceptance criteria

- Concepts can be created, refined, reviewed, and published as official workspace meaning.
- Relations can be created, refined, reviewed, and published as first-class assertions.
- Relation review behavior is deterministic when domains differ.
- Member-facing concept and relation views distinguish `source_backed`, `restricted_support`, and `insufficient_support` honestly.

## Linked canonical specs

- [../../concepts/spec.md](../../concepts/spec.md)
- [../../graph-and-relations/spec.md](../../graph-and-relations/spec.md)
- [../../domain-model/spec.md](../../domain-model/spec.md)
- [../../review-and-validation/spec.md](../../review-and-validation/spec.md)

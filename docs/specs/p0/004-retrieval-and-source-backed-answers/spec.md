# P0-004 Retrieval and Source-Backed Answers

## Status

Draft

## Summary

Cornerstone must let humans and AI retrieve official workspace context through canonical answer, concept, relation, decision, graph, provenance, and no-match surfaces.

## Goals

- let humans and AI consume the same official workspace truth
- return structured context, not only free-form text
- preserve honest support visibility and provenance disclosure

## Functional requirements

- Retrieval must remain workspace-scoped.
- Member-facing retrieval defaults to `member_visible` support.
- Review retrieval may include `evidence_only` support when policy and permissions allow.
- Grounded answers must return one canonical response kind and payload shape from the serving contract.
- A member-facing answer may be labeled `source_backed` only when at least one visible support item is inspectable by that member.
- If an answer is official but depends on hidden support, it must disclose `restricted_support` or remain unavailable when policy disallows restricted-support publication.
- When no official result exists, the system must return `no_match` with one canonical reason.

## Acceptance criteria

- Members and AI consumers can retrieve official workspace context through the canonical response kinds.
- Search results, graph slices, provenance follow-up, and no-match responses all use canonical shapes.
- Answers expose enough support and provenance to justify trust without over-claiming inspectability.

## Linked canonical specs

- [../../retrieval-and-answers/spec.md](../../retrieval-and-answers/spec.md)
- [../../serving-contract/spec.md](../../serving-contract/spec.md)
- [../../state-vocabulary/spec.md](../../state-vocabulary/spec.md)

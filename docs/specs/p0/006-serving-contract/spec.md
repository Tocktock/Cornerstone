# P0-006 Serving Contract

## Status

Draft

## Summary

P0 must implement one canonical consumer contract for grounded answers, concept detail, relation detail, decision detail, search results, graph slices, provenance follow-up, and no-match responses.

## Goals

- eliminate interface ambiguity across UI, API, and model-facing surfaces
- preserve trust labels and support-visibility semantics across all consumers
- make response kinds, required fields, enums, and cardinality implementation-ready without transport coupling

## Functional requirements

- Every consumer-facing surface must return one of the canonical `response_kind` values.
- Every consumer-facing response must include the common response envelope defined by the canonical serving contract.
- `concept`, `relation`, `decision`, `answer`, `search_results`, `graph_slice`, `provenance`, and `no_match` must each use their canonical payload shape.
- Canonical enums from the state-vocabulary spec must be used exactly where the serving contract requires them.
- Human-facing and model-facing surfaces may package the contract differently, but they may not change field meaning, required presence, or cardinality.

## Acceptance criteria

- The docs provide one authoritative answer to what each canonical response kind contains.
- UI, API, and MCP/model-facing implementations can be checked against the same field list and cardinality rules.
- No response kind listed in the canonical serving contract is left shape-undefined.

## Linked canonical specs

- [../../serving-contract/spec.md](../../serving-contract/spec.md)
- [../../state-vocabulary/spec.md](../../state-vocabulary/spec.md)
- [../../retrieval-and-answers/spec.md](../../retrieval-and-answers/spec.md)

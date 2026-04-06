# Foundations

## Summary

This spec defines the boundaries that keep Cornerstone coherent over time.

It separates **non-replaceable foundations** from **replaceable implementation choices** so the product can evolve without drifting away from its core purpose.

## Why this exists

Cornerstone is broad enough that implementation details can easily start steering the product. Without explicit foundations, the system would drift toward being:
- a source-system replacement
- an enterprise search tool
- a generic RAG stack
- a model-specific assistant
- an unreviewed content generator

This spec prevents that drift.

## Non-replaceable foundations

The following foundations are fixed unless changed through a formal decision record and corresponding spec updates.

| Foundation | Rule |
| --- | --- |
| Product identity | Cornerstone is the shared organizational context layer for humans and AI. |
| Truth layering | Source systems keep raw operational truth. Cornerstone owns the official layer of meaning, relationships, decision context, and trust disclosure on top. |
| Officiality | Official concepts, relations, and decisions must be grounded, reviewable, and honest about what support the current consumer can inspect. |
| AI role | AI may draft, summarize, and propose. AI is a consumer and worker, not the product identity or final truth authority. |
| Decision context | Decision context is first-class and is stored as `DecisionRecord`. |
| Projection model | Glossary, graph, and answer surfaces are projections over curated entities, not separate storage roots. |
| Sync priority | Eventual correctness, provenance, and recoverability are more important than strict real-time sync. |
| Workspace boundary | Shared context is workspace-scoped and must not be implicitly shared across workspaces. Personal context is secondary and remains separate until explicitly promoted. |
| Contract stability | Canonical cross-surface shapes, field meanings, and exposed enum values are part of the product contract and are not replaceable. |
| Delivery model | Development is spec-driven. Durable changes require docs updates before or with implementation. |

## Replaceable implementation choices

The following choices may evolve without changing Cornerstone’s identity as long as they keep honoring the non-replaceable foundations.

| Area | Replaceable choices |
| --- | --- |
| Connector coverage | Which providers are supported first, how provider pickers work, and how templates are grouped |
| Storage | Relational store, graph projection strategy, indexing pipeline, caching strategy |
| Sync mechanics | Polling intervals, webhook usage, worker topology, retry behavior |
| AI provider | Claude, Gemini, Codex, internal models, or multiple models |
| Serving transport | REST, GraphQL, MCP, internal service APIs, UI-specific composition |
| Ranking and retrieval | Search ranking algorithm, hybrid retrieval strategy, source-native fallback rules |
| UI shape | Navigation, review workspace layout, glossary/detail page composition |
| Auth provider | Google login, SSO, passwordless, service credentials |
| Deployment shape | Monolith, modular monolith, service split, managed infrastructure choices |

## Product-quality bar

No capability should be treated as complete unless it preserves all of the following:
- workspace-bounded context
- traceable provenance
- clear review ownership
- durable official outputs
- answer explainability for humans and AI
- stable contract semantics across surfaces

## Delivery model

Every feature change should follow this sequence:

1. confirm which existing spec owns the behavior
2. update the spec or create a new one
3. add or update a decision record if a durable boundary changes
4. add a memory note if the rationale will matter later
5. implement only after the documentation reflects the intended behavior

## Related docs

- [../system-overview/spec.md](../system-overview/spec.md)
- [../state-vocabulary/spec.md](../state-vocabulary/spec.md)
- [../serving-contract/spec.md](../serving-contract/spec.md)
- [../../decisions/0008-spec-driven-delivery-and-non-replaceable-foundations.md](../../decisions/0008-spec-driven-delivery-and-non-replaceable-foundations.md)

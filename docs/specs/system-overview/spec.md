# System Overview

## Summary

Cornerstone reconstructs organizational context from scattered documents, conversations, and systems, then serves that context back to humans and AI in a grounded, reviewable form.

It is not a source-system replacement. It is the official layer of meaning, relationships, decision context, and trust disclosure built on top of source memory.

## Real-world problem

Organizations often know more than either people or AI can reconstruct in the moment. Definitions, rationale, constraints, and supporting source material are scattered across tools and memory.

As a result:
- people repeat the same clarification work
- AI answers from fragments instead of organizational context
- decisions are made without stable knowledge of why something is true here

Cornerstone exists to reconstruct that context into a shared, durable layer.

## Product posture

Cornerstone is best understood in two ways:

- **Internally:** a semantic system of record for organizational meaning, relationships, decision context, and trust disclosure
- **Externally:** the shared organizational context layer for humans and AI

## Core product loop

1. An authorized workspace actor connects shared sources or promotes selected personal support into the workspace.
2. Source sync creates or updates source memory such as artifacts, evidence fragments, provenance, and freshness state.
3. Humans or AI create draft concepts, relations, and decision records from source memory or manual input.
4. Review and validation determine whether the draft is grounded, current enough, policy-compliant, and honestly disclosed.
5. Official outputs are published as glossary, graph, and answer projections.
6. Human UI, programmatic APIs, and MCP/model-facing tools consume those outputs through the same serving contract.
7. New sync, new support, or policy changes may reopen review without erasing official history.

## Core objects

Cornerstone’s canonical domain is built from:
- `ContextSpace`
- `SourceConnection`
- `Artifact`
- `EvidenceFragment`
- `PromotedSupport`
- `Concept`
- `ConceptRelation`
- `DecisionRecord`
- `Actor`

The product serves these objects through projections such as:
- glossary views
- graph slices
- context passports
- grounded answers
- provenance follow-up views

## Access model overview

Cornerstone uses one permission model:
- base roles: `owner`, `admin`, `member`
- scoped capabilities: `manage_connectors`, `operate`, `review`, `own_domain`

Human-facing labels are derived, not independent:
- connector manager = has `manage_connectors`
- knowledge operator = has `operate`
- authorized reviewer = has `review`
- domain owner = has `own_domain`

## Review scope overview

Review is deterministic, not interpretive.

Every reviewable shared object has exactly one canonical `review_domain`:
- a specific workspace domain slug, or
- the special workspace-wide domain `workspace`

A review action is allowed only when the acting principal has:
- workspace-wide `review`, or
- domain-scoped `review` matching the object’s `review_domain`

Cross-domain relations and any object explicitly marked `workspace` require workspace-wide review.

## Trust model overview

All consumer-facing surfaces must preserve the same trust vocabulary:
- `support_visibility`
- `lifecycle_state`
- `verification_state`
- `freshness_state`
- provenance summary
- review lineage where relevant

Member-facing outputs may be approved using hidden support only when policy allows, but they may be labeled `source_backed` only when the current member can inspect at least one visible support item.

## Personal context overview

Personal context is optional and secondary.

It exists for private source memory, private preparation, and selective sharing. Personal source content does not directly support shared official outputs. The only way personal material enters shared workspace review is through explicit creation of `PromotedSupport`.

## Constraints and non-goals

- Cornerstone is not a replacement for Notion, Slack, Google Docs, or other source systems.
- Cornerstone is not a generic enterprise search engine.
- Cornerstone is not a generic vector store or RAG database.
- Cornerstone is not a single-model AI assistant.
- Cornerstone does not require strict real-time sync from every provider.

## Related docs

- [../foundations/spec.md](../foundations/spec.md)
- [../state-vocabulary/spec.md](../state-vocabulary/spec.md)
- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)
- [../domain-model/spec.md](../domain-model/spec.md)
- [../review-and-validation/spec.md](../review-and-validation/spec.md)
- [../serving-contract/spec.md](../serving-contract/spec.md)
- [../../decisions/0001-cornerstone-is-the-shared-organizational-context-layer.md](../../decisions/0001-cornerstone-is-the-shared-organizational-context-layer.md)
- [../../decisions/0009-access-uses-base-roles-plus-scoped-capabilities.md](../../decisions/0009-access-uses-base-roles-plus-scoped-capabilities.md)
- [../../decisions/0014-review-authorization-scope-is-derived-deterministically.md](../../decisions/0014-review-authorization-scope-is-derived-deterministically.md)

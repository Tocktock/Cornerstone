# Domain Model

## Summary

Cornerstone’s domain model defines the **first-class concrete product objects** that persist inside the product boundary.

This spec works with, but does not replace, the ontology layer:
- [`../ontology/spec.md`](../ontology/spec.md) owns the abstract semantic model, predicate taxonomy, and projection ontology.
- this spec owns the concrete product objects, required field groups, and cross-object invariants needed for implementation.

## Why this exists

Cornerstone’s outputs—glossary, graph, grounded answers, provenance views, and review surfaces—should not be mistaken for storage roots.

The product needs a stable concrete model that can:
- preserve workspace boundaries
- keep source memory distinct from curated meaning
- preserve support and decision lineage
- allow deterministic review and officialization
- serve both humans and AI from one underlying truth layer

## Abstract ontology types used here

The following abstract classes are defined in [`../ontology/spec.md`](../ontology/spec.md):
- `ScopedObject`
- `SupportItem`
- `ReviewableObject`
- `CuratedObject`
- `Projection`

This spec focuses on the concrete first-class objects that instantiate those abstractions.

## First-class concrete objects

| Object | Ontology role | Purpose |
| --- | --- | --- |
| `ContextSpace` | concrete `ScopedObject` | Shared workspace or personal boundary for meaning, access, and source memory |
| `Actor` | concrete `ScopedObject` | Human, team, service, or AI principal |
| `VerificationPolicy` | concrete `ScopedObject` | Workspace policy defining what counts as official enough |
| `ReviewScopeGrant` | concrete `ScopedObject` | Explicit review authority over object kinds, actions, and review domains |
| `ConnectorScopeGrant` | concrete `ScopedObject` | Explicit shared-connector administration authority |
| `SourceConnection` | concrete `ScopedObject` | Durable connection to one upstream source boundary |
| `Artifact` | concrete `ScopedObject` | Persistent normalized representation of source content |
| `EvidenceFragment` | concrete `SupportItem` | Citation-ready support derived from source memory |
| `PromotedSupport` | concrete `SupportItem` | Workspace-scoped support created from explicit promotion of selected personal material |
| `PromotionLineage` | concrete `LineageObject` | Immutable lineage connecting promoted support to private personal origin |
| `Concept` | concrete `CuratedObject` | Canonical organizational term, policy, workflow, role, system, metric, or other meaning unit |
| `ConceptRelation` | concrete `CuratedObject` | First-class semantic relationship between two concepts |
| `DecisionRecord` | concrete `CuratedObject` | First-class stored decision context |

## Required field groups by object

### `ContextSpace`

Must preserve:
- stable `context_space_ref`
- `context_space_kind`
- canonical display name
- membership boundary
- visibility and verification defaults for the boundary

### `Actor`

Must preserve:
- stable `actor_ref`
- actor kind
- boundary membership
- any applicable base role and scoped capabilities

### `VerificationPolicy`

Must preserve:
- stable `verification_policy_ref`
- owning workspace
- policy label and version
- requirements for official grounding, freshness, visibility, and revalidation

### `ReviewScopeGrant`

Must preserve:
- stable `review_scope_grant_ref`
- target actor
- owning workspace
- one or more review domains
- one or more allowed review actions
- one or more target object kinds

### `ConnectorScopeGrant`

Must preserve:
- stable `connector_scope_grant_ref`
- target actor
- owning workspace
- one or more allowed connector actions

### `SourceConnection`

Must preserve:
- stable `source_connection_ref`
- owning context space
- source-system identity and source boundary locator
- canonical visibility class defaults
- source-connection state
- provenance and freshness metadata needed for downstream trust evaluation

### `Artifact`

Must preserve:
- stable `artifact_ref`
- owning context space
- parent source connection
- source locator or equivalent origin pointer
- refresh and freshness metadata
- enough normalized representation to support extraction and provenance

### `EvidenceFragment`

Must preserve:
- stable `support_ref`
- owning context space
- parent artifact
- visibility class
- support payload usable in review and provenance views
- provenance summary needed to return to source memory

### `PromotedSupport`

Must preserve:
- stable workspace `support_ref`
- target workspace ref
- promoter ref
- promoted content unit
- shared payload visible inside the workspace
- visibility class
- immutable `promotion_lineage_ref`
- disclosure fields needed for review and honest member-facing provenance

### `PromotionLineage`

Must preserve:
- stable `promotion_lineage_ref`
- source context kind, which is always `personal`
- personal source owner
- restricted private origin reference
- selection method
- selection scope summary
- workspace disclosure note

### `Concept`

Must preserve:
- stable `concept_ref`
- stable `public_slug`
- canonical name
- definition
- exactly one `concept_kind`
- exactly one `owning_domain`
- exactly one `review_domain`
- lifecycle state
- verification state
- support-visibility state

### `ConceptRelation`

Must preserve:
- stable `relation_ref`
- exactly one subject concept ref
- exactly one canonical predicate
- exactly one object concept ref
- exactly one `review_domain`
- lifecycle state
- verification state
- support-visibility state

### `DecisionRecord`

Must preserve:
- stable `decision_ref`
- stable `public_slug`
- title
- problem statement
- decision statement
- rationale
- zero or more constraints
- zero or one impact summary
- exactly one `owning_domain`
- exactly one `review_domain`
- lifecycle state
- verification state
- support-visibility state
- optional supersession lineage

## Relationship families implemented by the model

### Boundary ownership
- `ContextSpace` contains every scoped object in the boundary.

### Source memory
- `SourceConnection` produces `Artifact`.
- `Artifact` yields zero or more `EvidenceFragment` objects.

### Support and lineage
- `SupportItem` may support zero or more curated objects.
- `PromotedSupport` always has exactly one `PromotionLineage`.
- `PromotionLineage` never exposes the full private personal artifact to other workspace actors.

### Curated meaning
- `ConceptRelation` links exactly two concepts.
- `DecisionRecord` may affect zero or more concepts.
- `DecisionRecord` may affect zero or more concept relations.
- `DecisionRecord` may supersede one earlier decision record.

### Projection assembly
- glossary views are projections over `Concept` plus linked relations, decisions, and support summaries
- graph views are projections over `Concept` and `ConceptRelation`, with optional support or decision overlays
- answer views are projections over curated objects plus support and provenance summaries

## Review-domain rules

The canonical derivation rules are fixed:
- `Concept.review_domain = Concept.owning_domain`
- `DecisionRecord.review_domain = DecisionRecord.owning_domain`
- `ConceptRelation.review_domain =`
  - the shared endpoint domain when subject and object concepts have the same owning domain
  - `workspace` when the endpoints span different owning domains or the relation is explicitly workspace-wide

## Invariants

- Every first-class object belongs to exactly one `ContextSpace`.
- Only workspace context may contain official shared outputs.
- Personal-context support never directly grounds shared official meaning.
- Only `PromotedSupport` may carry selected personal material into shared review and retrieval.
- `ConceptRelation` must use a canonical semantic predicate owned by the ontology spec.
- Projections are not independent storage roots.

## Related docs

- [`../ontology/spec.md`](../ontology/spec.md)
- [`../workspace-and-access/spec.md`](../workspace-and-access/spec.md)
- [`../sync-and-provenance/spec.md`](../sync-and-provenance/spec.md)
- [`../review-and-validation/spec.md`](../review-and-validation/spec.md)
- [`../serving-contract/spec.md`](../serving-contract/spec.md)
- [`../../decisions/0003-official-knowledge-is-curated-over-source-memory.md`](../../decisions/0003-official-knowledge-is-curated-over-source-memory.md)
- [`../../decisions/0011-personal-sources-remain-separate-until-explicitly-promoted-into-shared-context.md`](../../decisions/0011-personal-sources-remain-separate-until-explicitly-promoted-into-shared-context.md)
- [`../../decisions/0015-ontology-layer-is-the-canonical-semantic-reference.md`](../../decisions/0015-ontology-layer-is-the-canonical-semantic-reference.md)

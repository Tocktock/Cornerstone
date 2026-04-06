# Ontology

## Summary

This spec defines the **canonical ontology layer** of Cornerstone.

It describes:
- which kinds of things exist in Cornerstone
- which of those things are abstract classes versus concrete first-class objects
- how those things may relate to one another
- which relation predicates are part of the shared semantic graph
- which links are system-level lineage or governance links rather than member-facing graph edges
- which invariants must hold before the same product behavior can be implemented safely across multiple systems

This spec is **implementation-neutral**. It does not choose storage engines, transports, or programming models.

## Why this exists

Cornerstone now has enough product surface area that feature-level prose alone is not sufficient.

Without a canonical ontology:
- the same object can be interpreted differently across features
- graph predicates can drift between surfaces
- support lineage and promoted personal material can be modeled inconsistently
- review and connector permissions can be attached to different conceptual objects in different implementations
- the serving contract can expose fields whose semantic ownership is unclear

The ontology layer solves that problem by defining the semantic model beneath the feature specs.

## Scope and owned behavior

This spec owns:
- the abstract class hierarchy used across Cornerstone
- the canonical concrete object taxonomy
- the distinction between **semantic relations** and **system links**
- domain and range rules for core object relationships
- canonical concept-kind taxonomy
- canonical graph predicate taxonomy
- support-item and promotion ontology
- reviewable-object ontology
- projection ontology for glossary, graph, answer, and provenance views

This spec does **not** own:
- visual presentation
- transport protocols
- endpoint layout
- storage schema design
- indexing strategy
- sync worker topology

Those remain replaceable as long as they preserve this ontology.

## Ownership boundaries with other specs

- [`../domain-model/spec.md`](../domain-model/spec.md) owns the **first-class concrete product objects** and their required field groups.
- [`../state-vocabulary/spec.md`](../state-vocabulary/spec.md) owns the **canonical exposed enum values** for states.
- [`../serving-contract/spec.md`](../serving-contract/spec.md) owns the **consumer-facing payload shapes**.
- This ontology spec owns the **semantic meaning** of those objects and fields.

When a feature spec needs to say what a concept, relation, support item, decision, or reviewable object *is*, it should defer to this spec instead of redefining the ontology locally.

## Modeling principles

1. **Context is bounded.**
   Every canonical object belongs to exactly one `ContextSpace`.
2. **Curated meaning is distinct from source memory.**
   Source-derived objects and curated objects are not the same kind of thing.
3. **Support is first-class.**
   Official meaning must be grounded in explicit support objects, not implied from raw source presence.
4. **Decision rationale is first-class.**
   Cornerstone preserves not only what is true, but why it is true here.
5. **Visibility must be honest.**
   A consumer-facing object may only be called `source_backed` when the current consumer can inspect visible support.
6. **Promotion is explicit.**
   Personal material can affect shared context only through explicit promotion into a workspace-scoped support object.
7. **Review is deterministic.**
   Reviewable objects expose a canonical review domain and can be approved only through explicit scoped authority.
8. **Projections are views, not storage roots.**
   Glossary, graph, answer, passport, and provenance views are assembled from canonical objects.

## Ontology strata

Cornerstone’s ontology is easiest to understand as five strata.

### 1. Boundary and governance stratum
Defines who acts, where they act, and how scope is controlled.

Objects:
- `ContextSpace`
- `Actor`
- `VerificationPolicy`
- `ReviewScopeGrant`
- `ConnectorScopeGrant`

### 2. Source-memory stratum
Defines how external systems and artifacts become durable source memory inside a boundary.

Objects:
- `SourceConnection`
- `Artifact`

### 3. Support and lineage stratum
Defines explicit support items that may ground curated meaning.

Objects:
- `SupportItem` *(abstract)*
- `EvidenceFragment`
- `PromotedSupport`
- `PromotionLineage`

### 4. Curated-meaning stratum
Defines the canonical objects that become official organizational context.

Objects:
- `ReviewableObject` *(abstract)*
- `CuratedObject` *(abstract)*
- `Concept`
- `ConceptRelation`
- `DecisionRecord`

### 5. Projection stratum
Defines consumer-facing assembled views over curated meaning and support.

Objects:
- `Projection` *(abstract)*
- `GlossaryProjection`
- `GraphSliceProjection`
- `AnswerProjection`
- `ContextPassport`
- `SearchResultsProjection`
- `ProvenanceProjection`

Projection objects are **not** separate truth roots.

## Canonical class hierarchy

```text
ScopedObject (abstract)
├── ContextSpace
├── Actor
├── VerificationPolicy
├── ReviewScopeGrant
├── ConnectorScopeGrant
├── SourceConnection
├── Artifact
├── SupportItem (abstract)
│   ├── EvidenceFragment
│   └── PromotedSupport
└── ReviewableObject (abstract)
    └── CuratedObject (abstract)
        ├── Concept
        ├── ConceptRelation
        └── DecisionRecord

LineageObject (abstract)
└── PromotionLineage

Projection (abstract)
├── GlossaryProjection
├── GraphSliceProjection
├── AnswerProjection
├── ContextPassport
├── SearchResultsProjection
└── ProvenanceProjection
```

### Abstract-class semantics

| Abstract class | Meaning | Concrete subclasses |
| --- | --- | --- |
| `ScopedObject` | Any canonical object owned by exactly one `ContextSpace` | Most durable objects in the model |
| `SupportItem` | Any workspace-usable support object that may ground curated meaning | `EvidenceFragment`, `PromotedSupport` |
| `ReviewableObject` | Any object that may enter review and officialization flows | `Concept`, `ConceptRelation`, `DecisionRecord` |
| `CuratedObject` | Any reviewable object representing official organizational meaning rather than source memory | `Concept`, `ConceptRelation`, `DecisionRecord` |
| `Projection` | Any consumer-facing assembled view over canonical objects | glossary, graph, answer, passport, provenance, search-result projections |
| `LineageObject` | Any immutable record that preserves origin lineage without becoming a support object itself | `PromotionLineage` |

## Concrete object catalog

### Boundary and governance objects

| Object | Identity | Boundary | Purpose |
| --- | --- | --- | --- |
| `ContextSpace` | `context_space_ref` | self-owned | Shared workspace or personal boundary for meaning, access, and source memory |
| `Actor` | `actor_ref` | exactly one `ContextSpace` | Human, team, service, or AI principal acting in the boundary |
| `VerificationPolicy` | `verification_policy_ref` | exactly one workspace `ContextSpace` | Rules for what counts as official enough |
| `ReviewScopeGrant` | `review_scope_grant_ref` | exactly one workspace `ContextSpace` | Explicit scope object for review authority |
| `ConnectorScopeGrant` | `connector_scope_grant_ref` | exactly one workspace `ContextSpace` | Explicit scope object for shared connector administration |

### Source-memory objects

| Object | Identity | Boundary | Purpose |
| --- | --- | --- | --- |
| `SourceConnection` | `source_connection_ref` | exactly one `ContextSpace` | Durable connection to one upstream source boundary |
| `Artifact` | `artifact_ref` | exactly one `ContextSpace` | Persistent normalized representation of source content |

### Support and lineage objects

| Object | Identity | Boundary | Purpose |
| --- | --- | --- | --- |
| `EvidenceFragment` | `support_ref` | exactly one workspace or personal `ContextSpace` | Citation-ready support derived from shared or personal source memory |
| `PromotedSupport` | `support_ref` | exactly one workspace `ContextSpace` | Workspace-scoped support created by explicitly promoting selected personal material |
| `PromotionLineage` | `promotion_lineage_ref` | logically attached to one workspace promotion | Immutable lineage from promoted support back to personal origin |

### Curated-meaning objects

| Object | Identity | Boundary | Purpose |
| --- | --- | --- | --- |
| `Concept` | `concept_ref` and stable `public_slug` | exactly one workspace `ContextSpace` | Canonical organizational meaning unit |
| `ConceptRelation` | `relation_ref` | exactly one workspace `ContextSpace` | First-class semantic relationship between two concepts |
| `DecisionRecord` | `decision_ref` | exactly one workspace `ContextSpace` | First-class stored decision context |

## Canonical concept kinds

A `Concept` may be classified using exactly one canonical `concept_kind`.

Allowed values:
- `term`
- `domain`
- `system`
- `policy`
- `workflow`
- `role`
- `metric`
- `event`
- `artifact_type`
- `status`

Rules:
- The `concept_kind` is part of the ontology even if a given surface chooses not to expose it.
- New canonical concept kinds require an ontology update.
- A concept kind helps constrain relation meaning, but it does not by itself grant permissions or lifecycle behavior.

## Required semantic fields by class

### `Concept`

A concept must preserve all of the following semantics:
- stable identity
- stable `public_slug`
- canonical name
- definition
- exactly one `concept_kind`
- exactly one `owning_domain`
- exactly one canonical `review_domain`
- canonical lifecycle state
- canonical verification state
- canonical support-visibility state
- zero or more linked support items
- zero or more linked relations
- zero or more linked decisions

### `ConceptRelation`

A relation must preserve all of the following semantics:
- stable identity
- exactly one subject concept
- exactly one predicate
- exactly one object concept
- zero or one human-readable description
- exactly one canonical `review_domain`
- canonical lifecycle state
- canonical verification state
- canonical support-visibility state
- zero or more linked support items
- zero or more linked decisions

### `DecisionRecord`

A decision record must preserve all of the following semantics:
- stable identity
- title
- problem statement
- decision statement
- rationale
- zero or more constraints
- zero or one impact summary
- exactly one `owning_domain`
- exactly one canonical `review_domain`
- canonical lifecycle state
- canonical verification state
- canonical support-visibility state
- zero or more linked support items
- zero or more linked concept refs
- zero or more linked relation refs
- zero or one `supersedes_ref`
- zero or one `superseded_by_ref`

### `SupportItem`

Any support item must preserve all of the following semantics:
- stable `support_ref`
- exactly one owning `ContextSpace`
- exactly one `visibility_class`
- support payload that can be inspected or cited by an authorized consumer
- provenance bundle or promotion lineage, depending on support subtype

## Structural links vs semantic graph predicates

Cornerstone makes a strict distinction between:

### Structural links
Links used for containment, lineage, support, governance, or review.
These are **not** member-facing semantic graph predicates.

### Semantic graph predicates
Predicates stored on `ConceptRelation` and used to express organizational meaning between concepts.
These **are** member-facing graph semantics.

This distinction is non-optional. A support link, review-scope link, or decision applicability link must not be represented as if it were a semantic graph predicate.

## Structural link families

The following link families are canonical.

| Link family | Domain | Range | Cardinality rule | Meaning |
| --- | --- | --- | --- | --- |
| `contains` | `ContextSpace` | any `ScopedObject` | one-to-many | Context boundary ownership |
| `has_policy` | `ContextSpace` | `VerificationPolicy` | zero-or-more | Policies that govern officialization |
| `grants_review_scope_to` | `ReviewScopeGrant` | `Actor` | exactly one actor per grant | Review authority assignment |
| `grants_connector_scope_to` | `ConnectorScopeGrant` | `Actor` | exactly one actor per grant | Shared connector administration authority |
| `produces` | `SourceConnection` | `Artifact` | one-to-many | Source memory creation |
| `yields_support` | `Artifact` | `EvidenceFragment` | zero-or-more | Support extraction from source memory |
| `supports` | `SupportItem` | `CuratedObject` | many-to-many | Grounding link from support to curated meaning |
| `affects` | `DecisionRecord` | `Concept` or `ConceptRelation` | many-to-many | Decision applicability or impact |
| `supersedes_decision` | `DecisionRecord` | `DecisionRecord` | zero-or-one forward link | Newer decision replaces or narrows older decision |
| `creates_workspace_snapshot` | `PromotionLineage` | `PromotedSupport` | exactly one | Promotion lineage produces one workspace support snapshot |
| `originates_from_personal_context` | `PromotionLineage` | personal `ContextSpace` | exactly one | Promotion lineage records its private source boundary |

## Canonical semantic graph predicates

The following predicates are the canonical allowed values for `ConceptRelation.predicate`.

### Ontological predicates

| Predicate | Direction | Meaning |
| --- | --- | --- |
| `is_a` | subject → object | Subject is a subtype or narrower kind of object |
| `part_of` | subject → object | Subject is a component or constituent part of object |
| `instance_of` | subject → object | Subject is a concrete instance of object |

### Operational predicates

| Predicate | Direction | Meaning |
| --- | --- | --- |
| `depends_on` | subject → object | Subject requires object to function or be valid |
| `used_in` | subject → object | Subject participates in or is used inside object |
| `input_to` | subject → object | Subject serves as input to object |
| `output_of` | subject → object | Subject is produced by object |

### Flow and temporal predicates

| Predicate | Direction | Meaning |
| --- | --- | --- |
| `precedes` | subject → object | Subject occurs or applies before object |
| `triggers` | subject → object | Subject causes object to begin or be evaluated |
| `results_in` | subject → object | Subject leads to object as an outcome |

### Governance and semantic predicates

| Predicate | Direction | Meaning |
| --- | --- | --- |
| `owned_by` | subject → object | Subject is operationally owned by object |
| `governed_by` | subject → object | Subject is constrained or controlled by object |
| `defined_by` | subject → object | Subject derives canonical meaning from object |
| `applies_to` | subject → object | Subject applies to object as a rule, policy, or classification |
| `conflicts_with` | subject → object | Subject is in semantic or operational conflict with object |
| `supersedes` | subject → object | Subject replaces object in current usage or official meaning |

### Predicate rules

- `ConceptRelation.predicate` must use one of the canonical predicate values in this spec.
- A new predicate requires an ontology update before it may be exposed in the canonical graph.
- A surface may compute inverse presentation labels, but it must not invent a new canonical predicate.
- The graph may render support links or decision applicability as overlays or linked context, but those are not replacements for semantic predicates.

## Review and governance ontology

### `ReviewableObject`

A `ReviewableObject` is any object that may be submitted, approved, rejected, superseded, or revalidated.

In Cornerstone, only the following concrete objects are reviewable:
- `Concept`
- `ConceptRelation`
- `DecisionRecord`

A reviewable object must expose:
- exactly one `review_domain`
- one canonical lifecycle state
- one canonical verification state
- one canonical support-visibility state

### `ReviewScopeGrant`

A review-scope grant must preserve:
- stable identity
- exactly one target actor
- exactly one workspace
- one or more `review_domain` values
- one or more allowed review actions
- one or more target object kinds

A review decision is valid only when:
- the actor acts inside the owning workspace
- the actor holds applicable `review` authority
- the grant covers the target object kind
- the grant covers the object’s exact `review_domain`
- the grant covers the requested review action

### `ConnectorScopeGrant`

A connector-scope grant must preserve:
- stable identity
- exactly one target actor
- exactly one workspace
- one or more allowed connector actions

`ConnectorScopeGrant` is used only for shared workspace connectors. Personal connectors are controlled by the personal context owner.

## Support and promotion ontology

### `EvidenceFragment`

`EvidenceFragment` is the standard support object derived from source memory.

It may originate in:
- a workspace context, when derived from shared source memory
- a personal context, when derived from personal source memory

Personal-context `EvidenceFragment` objects are **not** directly usable in shared official outputs.

### `PromotedSupport`

`PromotedSupport` is the only support subtype that may carry selected personal material into a workspace boundary.

A promoted support object must preserve all of the following semantics:
- stable workspace `support_ref`
- target workspace ref
- promoter identity
- promotion timestamp
- promoted content unit
- shared payload visible in the workspace
- visibility class
- immutable promotion-lineage ref
- disclosure note suitable for workspace reviewers and consumers

Allowed `promoted_content_unit` values are defined in [`../state-vocabulary/spec.md`](../state-vocabulary/spec.md).

### `PromotionLineage`

A promotion lineage record must preserve:
- stable `promotion_lineage_ref`
- source context kind, which is always `personal`
- personal source owner
- restricted pointer to the original private source location
- selection method
- selection scope summary
- workspace disclosure note

### Promotion rules

- Promotion creates a **workspace-scoped snapshot**, not a live shared mirror of the personal source.
- Promotion may select only `excerpt`, `structured_field`, or `summary_assertion` content units.
- Promotion does not implicitly share the full personal artifact.
- Only the personal context owner may navigate from the promotion back into the full personal source artifact.

## Projection ontology

Projection objects assemble curated meaning and support for consumption.

### `GlossaryProjection`
A member-facing or reviewer-facing view over `Concept` objects.

### `GraphSliceProjection`
A view over one or more root concepts, included nodes, included semantic relations, and optional support or decision overlays.

### `AnswerProjection`
A grounded answer built from concepts, relations, decisions, and support visibility semantics.

### `ContextPassport`
A compact detail projection for one `Concept`, `ConceptRelation`, or `DecisionRecord` that combines:
- canonical meaning
- current states
- linked support visible to the current consumer
- provenance summary
- linked decisions
- related concepts or relations

### `SearchResultsProjection`
An ordered projection of candidate resources and why they matched.

### `ProvenanceProjection`
A consumer-facing projection that explains support counts, visible support counts, freshness, and promotion presence.

Projection rules:
- A projection may never become the sole source of truth for the underlying object.
- A projection must preserve the owning workspace boundary of the underlying objects.
- A projection must use the canonical state vocabulary when it exposes states.

## Invariants

The following invariants are normative.

1. Every canonical object belongs to exactly one `ContextSpace`.
2. No curated object may become official without satisfying workspace verification policy.
3. A member-facing object may be labeled `source_backed` only when the current consumer can inspect at least one visible support item.
4. Personal-context support never directly grounds shared official outputs; only `PromotedSupport` may do so.
5. Cross-workspace semantic graph edges are not allowed.
6. `ConceptRelation` always links exactly two `Concept` objects in the same workspace.
7. `DecisionRecord` may affect many concepts or relations, but it is not itself a semantic graph predicate.
8. Projections do not create new ontology ownership; they reuse the ownership of underlying canonical objects.
9. A new concept kind or semantic predicate requires an ontology update before it is canonical.
10. State fields exposed to consumers must use the canonical values owned by the state-vocabulary spec.

## Worked example (non-normative)

A simple slice may look like this:

- `Concept(term): customer_health_score`
- `Concept(metric): account_expansion_score`
- `Concept(policy): renewal_risk_policy`
- `Concept(workflow): quarterly_customer_review`
- `ConceptRelation(customer_health_score, depends_on, account_expansion_score)`
- `ConceptRelation(renewal_risk_policy, applies_to, quarterly_customer_review)`
- `DecisionRecord`: “Use customer health score as an early renewal-risk signal for enterprise accounts.”
- `EvidenceFragment`: excerpt from a shared renewal playbook
- `PromotedSupport`: selected summary assertion from a personal analyst notebook explicitly promoted into the workspace

In this slice:
- the graph contains only the `ConceptRelation` predicates
- the support items ground the concept and decision
- the decision explains why the policy is current
- the promoted support contributes only because it was explicitly promoted across the privacy boundary

## Related docs

- [`../domain-model/spec.md`](../domain-model/spec.md)
- [`../concepts/spec.md`](../concepts/spec.md)
- [`../graph-and-relations/spec.md`](../graph-and-relations/spec.md)
- [`../decision-context/spec.md`](../decision-context/spec.md)
- [`../review-and-validation/spec.md`](../review-and-validation/spec.md)
- [`../serving-contract/spec.md`](../serving-contract/spec.md)
- [`../state-vocabulary/spec.md`](../state-vocabulary/spec.md)
- [`../../decisions/0015-ontology-layer-is-the-canonical-semantic-reference.md`](../../decisions/0015-ontology-layer-is-the-canonical-semantic-reference.md)

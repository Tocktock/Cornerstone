# Workspace and Access

## Summary

Cornerstone is workspace-bounded by default.

Every shared source connection, artifact, evidence fragment, promoted support object, concept, relation, decision record, review action, and answer is resolved inside one **workspace `ContextSpace`**. Personal context may exist as a secondary pattern, but it remains separate until explicitly promoted into a workspace.

## Why this exists

Cornerstone reconstructs organizational context. That context is meaningful only inside a real organizational boundary.

Without a strong workspace boundary and explicit access rules, the product would risk:
- leaking context across organizations
- weakening provenance and ownership
- confusing review responsibility
- allowing inconsistent implementations of source management, authoring, and approval

## Scope and owned behavior

This spec owns:
- context-space kinds and boundaries
- membership roles and scoped capabilities
- review-scope derivation and authorization rules
- access defaults
- visibility rules for member-facing and admin-facing surfaces
- production-mode empty-workspace and role-aware datasource guidance expectations
- personal-source boundary rules
- service access principles for programmatic and model-facing consumers

## Context-space kinds

Cornerstone recognizes two context-space kinds:

### Workspace context
- the primary and default mode
- shared by a real organization or team
- the only context kind that can hold shared official outputs

### Personal context
- optional and secondary
- owned by one member
- intended for private drafts, private source memory, or pre-sharing preparation
- cannot directly publish shared official outputs

Every canonical object belongs to exactly one `ContextSpace`.

## Canonical permission model

Cornerstone uses one permission model: **base role + scoped capability**.

### Base membership roles

| Base role | Default rights |
| --- | --- |
| `owner` | full workspace control, workspace-wide `manage_connectors`, workspace-wide `operate`, workspace-wide `review` |
| `admin` | workspace operational control, workspace-wide `manage_connectors`, workspace-wide `operate`, workspace-wide `review` |
| `member` | read official shared context, submit requests, no source administration or review by default |

### Scoped capabilities

| Capability | Meaning | Scope model | Human-facing label |
| --- | --- | --- | --- |
| `manage_connectors` | create, update, pause, resume, remove, and rebind shared workspace source connections | workspace-wide only | connector manager |
| `operate` | create and refine drafts, attach support, manage curation tasks | workspace-wide or domain-scoped | knowledge operator |
| `review` | approve, reject, officialize, supersede, and revalidate reviewable shared objects | workspace-wide or domain-scoped | authorized reviewer |
| `own_domain` | responsibility assignment over one or more domains used for routing and accountability | domain-scoped only | domain owner |

### Mapping rules

- `owner` and `admin` implicitly have workspace-wide `manage_connectors`, `operate`, and `review`.
- `member` may be granted `manage_connectors`, `operate`, `review`, and/or `own_domain` in scoped form.
- `operate` does not imply `manage_connectors`.
- `own_domain` does not imply `review`.
- Service and AI consumers inherit the same workspace boundary and visibility rules as the credential or token that authorizes them.

## Review scope model

### Canonical scope values

Review grants and reviewable objects use one of the following scope values:
- `workspace`
- a workspace domain slug such as `sales_ops`, `platform`, or `security`

### Review-domain derivation

Every reviewable shared object must expose exactly one canonical `review_domain`.

The derivation rules are fixed:
- A `Concept` uses its `owning_domain` as `review_domain`.
- A `DecisionRecord` uses its `owning_domain` as `review_domain`.
- A `ConceptRelation` uses:
  - the shared domain slug when both endpoints belong to the same owning domain
  - `workspace` when the endpoints span different owning domains or the relation is explicitly workspace-wide
- `PromotedSupport` is not officialized directly. It becomes reviewable only through the draft concept, relation, or decision that cites it.

### Review-authorization rule

A review action is allowed only when all of the following are true:
1. the principal acts inside the current workspace
2. the principal has `review`
3. the object has a canonical `review_domain`
4. the principal holds either:
   - workspace-wide `review`, or
   - domain-scoped `review` for the object’s exact `review_domain`

No other interpretation of “allowed scope” is valid.

## Personal-source boundary

Personal sources are optional and secondary.

The following rules are fixed:
- Personal sources attach to a **personal** context, not directly to a shared workspace context.
- Artifacts and evidence derived from personal sources are private to the personal context by default.
- Workspace owners, admins, connector managers, operators, and reviewers do **not** automatically gain access to personal connector contents or credentials.
- Shared official outputs may not depend directly on personal-context artifacts or personal-context evidence.
- A personal-source owner may explicitly create `PromotedSupport` for a workspace in which they are a member.
- `PromotedSupport` is the only shared object that may be created directly from personal context.
- Only `PromotedSupport` may participate in shared review, officialization, retrieval, and answers.

## Promoted-support disclosure rules

A `PromotedSupport` object must preserve all of the following:
- target workspace
- promoter identity
- promotion time
- shared selection kind
- workspace-visible shared payload
- origin disclosure level
- private origin lineage reference
- visibility class

Inspection rules are fixed:
- members may inspect a promoted-support item only when it is `member_visible` and linked to a member-visible official output
- reviewers and admins may inspect the workspace-visible payload and lineage summary needed for review
- no workspace actor other than the promoter automatically gains access to the underlying personal artifact, connector, or full personal origin reference

## Visibility rules

- `member_visible` sources and support items may appear in member-facing retrieval, concept views, graph views, and answers.
- `evidence_only` sources and support items may support curation and review but are hidden from normal member-facing retrieval.
- A member-facing output may be labeled `source_backed` only when at least one visible support item is available to that member.
- If an output is official but relies partly or wholly on hidden support, the output must disclose `restricted_support`.

## Production workspace onboarding behavior

When the runtime mode is `production`, shared workspaces may start without demo content.

The following rules are fixed:
- A production workspace with no linked shared datasource must not render demo concepts, decisions, answers, or browse results as filler.
- Managers may receive clear CTA guidance into `Source Studio` so they can connect the first datasource.
- Non-managers must receive explanatory read-only guidance that a connector manager needs to link a datasource.
- Once sources are linked but the first usable artifact set is not ready, the UI may describe first-sync progress, but must not invent member-facing knowledge.
- If source health becomes degraded, the UI may surface recovery messaging while preserving any already-visible member-facing content.

## Constraints and non-goals

- This spec does not define the exact identity provider or SSO vendor.
- This spec does not require anonymous access to workspace content.
- This spec does not make personal connectors the primary model.
- This spec does not define public publishing or cross-workspace sharing.

## Related docs

- [../ontology/spec.md](../ontology/spec.md)
- [../ai-operator-surfaces/spec.md](../ai-operator-surfaces/spec.md)
- [../connectors/spec.md](../connectors/spec.md)
- [../review-and-validation/spec.md](../review-and-validation/spec.md)
- [../sync-and-provenance/spec.md](../sync-and-provenance/spec.md)
- [../state-vocabulary/spec.md](../state-vocabulary/spec.md)
- [../../decisions/0002-context-is-workspace-scoped.md](../../decisions/0002-context-is-workspace-scoped.md)
- [../../decisions/0009-access-uses-base-roles-plus-scoped-capabilities.md](../../decisions/0009-access-uses-base-roles-plus-scoped-capabilities.md)
- [../../decisions/0011-personal-sources-remain-separate-until-explicitly-promoted-into-shared-context.md](../../decisions/0011-personal-sources-remain-separate-until-explicitly-promoted-into-shared-context.md)
- [../../decisions/0014-review-authorization-scope-is-derived-deterministically.md](../../decisions/0014-review-authorization-scope-is-derived-deterministically.md)

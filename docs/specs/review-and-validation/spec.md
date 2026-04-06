# Review and Validation

## Summary

Review and validation are the workflows that turn proposed context into trustworthy official meaning.

They govern how concepts, relations, and decision records become official, how support sufficiency is judged, and how already-published outputs are re-checked when source conditions change.

## Scope and owned behavior

This spec owns:
- review queue behavior
- validation state
- verification policy
- officialization rules
- support-visibility policy
- revalidation behavior
- role-limited review actions
- deterministic review-scope evaluation

## Verification policy

Each workspace owns a verification policy.

Canonical policy fields include:
- `minimum_support_items`
- `minimum_durable_support_items`
- `minimum_visible_support_items_for_source_backed`
- `allow_restricted_support_for_officialization`
- `allow_member_restricted_support_publication`
- `freshness_target`
- `continuous_revalidation_enabled`
- `allow_accepted_decision_lineage_as_support`

### Required interpretation

- If restricted support is allowed for officialization, reviewers may approve an output even when some decisive support is hidden from members.
- That approval does **not** allow member-facing surfaces to call the output `source_backed` unless the minimum visible support requirement is satisfied.
- If an official output is shown to members without enough visible support for a `source_backed` label, the output must disclose `restricted_support`.
- If workspace policy disallows member-facing restricted-support publication, the output remains reviewable but is not shown in ordinary member-facing surfaces.

## Review actions

Creating and refining drafts requires `operate`.

The following actions require matching `review` scope:
- approve
- reject
- officialize
- supersede
- mark for revalidation
- resolve `review_required`

## Deterministic scope evaluation

A review action is allowed only when all of the following are true:
1. the actor is in the current workspace
2. the actor has `review`
3. the target object exposes one canonical `review_domain`
4. the actor holds workspace-wide `review`, or domain-scoped `review` for that exact `review_domain`

Additional rules:
- `workspace` is never satisfied by a domain-scoped review grant.
- Cross-domain relations always resolve to `workspace`.
- Domain ownership routes accountability but does not replace review permission.
- Answers are not approved directly. They inherit officiality, visibility, and trust disclosure from the official objects and policy that support them.

## State model

Shared surfaces must use the canonical lifecycle and `verification_state` values defined in [../state-vocabulary/spec.md](../state-vocabulary/spec.md).

## Current behavior

- Officiality is separate from draft existence.
- Concepts, relations, and decision records may all exist in draft form before they become official.
- Review must account for support sufficiency, freshness, provenance clarity, linked decision lineage where relevant, and trust disclosure appropriate to the current consumer.
- Official outputs remain visible when possible even if later support drifts, but the system should reopen review and mark the item accordingly.
- Member requests and AI-generated suggestions enter review as draft or suggested content and are never auto-approved by default.
- Review surfaces may include `evidence_only` support that is hidden from member-facing consumption.
- Member-facing surfaces may not present restricted outputs as fully `source_backed`.

## Constraints and non-goals

- This spec does not define the visual review queue implementation.
- This spec does not require every workspace to use the same numeric policy values.
- This spec does not allow hidden support to masquerade as visible explanation.

## Related docs

- [`../ontology/spec.md`](../ontology/spec.md)

- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)
- [../state-vocabulary/spec.md](../state-vocabulary/spec.md)
- [../retrieval-and-answers/spec.md](../retrieval-and-answers/spec.md)
- [../serving-contract/spec.md](../serving-contract/spec.md)
- [../../decisions/0009-access-uses-base-roles-plus-scoped-capabilities.md](../../decisions/0009-access-uses-base-roles-plus-scoped-capabilities.md)
- [../../decisions/0010-member-facing-source-backed-claims-require-visible-support-or-restricted-support-disclosure.md](../../decisions/0010-member-facing-source-backed-claims-require-visible-support-or-restricted-support-disclosure.md)
- [../../decisions/0014-review-authorization-scope-is-derived-deterministically.md](../../decisions/0014-review-authorization-scope-is-derived-deterministically.md)

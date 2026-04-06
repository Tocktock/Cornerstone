# Authoring and Curation

## Summary

Authoring and curation are the draft-creation layers of Cornerstone.

They allow humans and AI to propose concepts, relations, and decision records from source memory or manual input, while preserving the rule that official meaning is created only through reviewable workflows.

## Scope and owned behavior

This spec owns:
- manual and AI-assisted draft creation
- refinement of concepts, relations, and decisions before approval
- member request intake as a draft source
- authoring constraints that prevent bypassing review

## Current behavior

- Authoring may create or update draft concepts, relations, decision records, and support links.
- Drafts may originate from synced artifacts and evidence, promoted support, member requests, operator manual input, or AI-assisted proposal.
- Authoring should preserve provenance wherever possible.
- Manual authoring is a fallback and complement, not the default substitute for connectors.
- AI-assisted authoring may help synthesize support and create candidate structures, but cannot bypass review and validation.
- Long-form narrative documents may exist as supporting context, but Cornerstone should bias toward structured meaning and rationale rather than unstructured parallel documentation.

## Permissions and visibility

- Creating or refining drafts requires `operate`.
- Owners and admins have workspace-wide `operate`.
- Members may create and refine drafts only when they have scoped `operate`.
- Review actions remain separate and require matching `review` scope.
- AI systems may create drafts only under explicit workspace policy and review constraints.

## Constraints and non-goals

- Authoring does not turn Cornerstone into a replacement wiki.
- Authoring does not allow direct publication without review.
- This spec does not define the visual editor implementation.

## Related docs

- [`../ontology/spec.md`](../ontology/spec.md)

- [../review-and-validation/spec.md](../review-and-validation/spec.md)
- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)
- [../../decisions/0009-access-uses-base-roles-plus-scoped-capabilities.md](../../decisions/0009-access-uses-base-roles-plus-scoped-capabilities.md)

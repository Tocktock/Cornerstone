# P0-005 Workspace Governance and Access

## Status

Draft

## Summary

Cornerstone must enforce the workspace as the boundary of meaning, access, review authority, and verification policy.

## Goals

- make the workspace the boundary of meaning, access, and governance
- define who can manage shared sources, curate meaning, review drafts, and consume official outputs
- make review authorization deterministic
- preserve the privacy boundary between personal context and shared workspace context

## Functional requirements

### Roles and capabilities

P0 must support the canonical permission model:
- base roles: `owner`, `admin`, `member`
- scoped capabilities: `manage_connectors`, `operate`, `review`, `own_domain`

### Shared-source management

P0 must ensure that:
- only owners, admins, or members with workspace-wide `manage_connectors` can manage shared connectors
- `operate` alone does not allow shared connector setup or credential changes

### Review authorization

P0 must ensure that:
- every reviewable shared object exposes exactly one `review_domain`
- review is allowed only with workspace-wide `review` or exact domain-scoped `review` matching that `review_domain`
- cross-domain relations require workspace-wide review

### Personal-source promotion

P0 must ensure that:
- personal sources remain private by default
- personal-source content cannot directly support shared official outputs
- explicit promotion creates `PromotedSupport`
- promoted support preserves disclosure level and private origin lineage
- workspace actors do not automatically gain access to the full private origin

### Visibility boundaries

P0 must distinguish between:
- normal member-facing official context
- review surfaces with deeper support visibility
- hidden or review-only support that informs curation without appearing in normal member retrieval

## Acceptance criteria

- Sources, curated objects, review state, and answers are all clearly workspace-scoped.
- Shared connector management is clearly separated from general authoring.
- Review allow-or-deny behavior is deterministic for the same object and grant set.
- Personal-source promotion produces shared support without leaking the private source by default.

## Linked canonical specs

- [../../workspace-and-access/spec.md](../../workspace-and-access/spec.md)
- [../../review-and-validation/spec.md](../../review-and-validation/spec.md)
- [../../sync-and-provenance/spec.md](../../sync-and-provenance/spec.md)

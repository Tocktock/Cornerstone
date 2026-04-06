# P0-003 Decision Context and Officialization

## Status

Draft

## Summary

Cornerstone must support decision context as a first-class reviewable object and allow it to shape official concepts and relations.

## Goals

- preserve why something is true here
- make decision records reviewable and linkable to concepts and relations
- allow officialization only through governed review

## Functional requirements

- The product must support `DecisionRecord` as a first-class shared object.
- Every shared decision record must expose the canonical required fields defined by the decision-context and serving-contract specs.
- Every shared decision record must carry one `owning_domain` and one `review_domain`.
- Decision records may be supported by evidence fragments, promoted support, accepted decision lineage, or combinations allowed by policy.
- Decision approval, rejection, supersession, and revalidation must follow deterministic review-scope rules.
- Accepted decisions must remain readable after supersession when the newer lineage is shown.

## Acceptance criteria

- A reviewer can approve, reject, supersede, and revalidate decision records within valid review scope.
- Concepts and relations can link to accepted decision records.
- Member-facing outputs can explain official meaning through linked decision context without exposing disallowed support.

## Linked canonical specs

- [../../decision-context/spec.md](../../decision-context/spec.md)
- [../../review-and-validation/spec.md](../../review-and-validation/spec.md)
- [../../serving-contract/spec.md](../../serving-contract/spec.md)

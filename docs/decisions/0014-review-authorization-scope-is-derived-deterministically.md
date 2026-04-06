# 0014 - Review authorization scope is derived deterministically

- **Status:** Accepted

## Context

Review authority cannot be implemented consistently if “allowed scope” is left to interpretation.

## Decision

Every reviewable shared object has exactly one canonical `review_domain`. Review is allowed only with workspace-wide review or exact domain-scoped review that matches that `review_domain`.

## Consequences

- Two compliant implementations should reach the same allow-or-deny decision for the same review action.
- Cross-domain relations require workspace-wide review.
- Answers inherit review posture from their supporting official objects and are not approved directly.

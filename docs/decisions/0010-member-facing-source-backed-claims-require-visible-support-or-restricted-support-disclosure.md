# 0010 - Member-facing source-backed claims require visible support or restricted-support disclosure

- **Status:** Accepted

## Context

An output may be correctly grounded for internal review yet still hide decisive support from ordinary members. Trust labels must remain honest for the current consumer.

## Decision

A member-facing output may be labeled `source_backed` only when the current member can inspect at least one visible support item. Otherwise it must disclose `restricted_support`, or remain hidden if policy disallows restricted-support publication.

## Consequences

- Hidden support may inform approval when policy allows.
- Hidden support may not masquerade as inspectable explanation.

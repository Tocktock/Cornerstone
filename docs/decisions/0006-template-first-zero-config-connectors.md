# 0006 - Template-first zero-config connectors

- **Status:** Accepted

## Context

Connector setup becomes fragile when users must think in raw provider configuration first.

## Decision

Connector setup should default to template-led flows with low operational friction and strong provenance preservation.

## Consequences

- Provider-specific details remain replaceable.
- Shared connector setup remains operational, but user-facing setup should feel guided rather than raw.

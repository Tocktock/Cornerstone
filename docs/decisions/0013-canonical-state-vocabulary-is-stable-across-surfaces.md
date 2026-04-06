# 0013 - Canonical state vocabulary is stable across surfaces

- **Status:** Accepted

## Context

Consumers cannot reliably integrate when lifecycle, trust, and freshness values vary between surfaces or implementations.

## Decision

Cornerstone defines one canonical exposed state vocabulary for support visibility, lifecycle, verification, visibility class, freshness, and source-connection state. Internal implementations may use richer values only if they map to the canonical values before crossing the contract boundary.

## Consequences

- Clients and model integrations can depend on stable enums.
- Specs must stop using “exact names may vary” for surfaced state values.

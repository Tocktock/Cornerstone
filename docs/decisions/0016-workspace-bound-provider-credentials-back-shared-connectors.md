# 0016 - Workspace-bound provider credentials back shared connectors

- **Status:** Accepted

## Context

Shared connectors need provider-specific credentials to bind and sync upstream sources, but exposing raw provider auth state through shared connector objects would blur trust boundaries and widen secret exposure.

## Decision

Cornerstone stores provider auth payloads in internal provider-credential records and lets shared `SourceConnection` objects reference those records only through workspace-bound credential references.

Shared binding, rebind, and credential inspection remain manager-only flows.

## Consequences

- Connector management can support OAuth-style provider binding without exposing tokens in ordinary source status APIs.
- `SourceConnection` remains the canonical shared connector object, while provider credentials stay operational support objects.
- Non-manager actors can inspect source status without gaining access to provider secrets.

# 0017 - MCP is the first model-facing transport for AI operator surfaces

- **Status:** Accepted

## Context

Cornerstone already treats the serving contract as canonical across human, API, and model-facing consumers. The repository also already uses `connectors` to mean upstream source-ingestion integrations.

As model-facing work expanded, the team needed a stable boundary that:
- preserves the serving contract as the non-replaceable semantic layer
- keeps model-host behavior outside product semantics
- exposes a first-class model-facing transport without renaming transport into contract
- avoids confusing model-facing tools with source-ingestion connectors

## Decision

- `AI operator surfaces` is the canonical workstream name for model-facing transports and tool surfaces.
- `connectors` continues to mean upstream source-ingestion integrations.
- Cornerstone’s serving contract remains canonical.
- MCP is the first public model-facing transport for Cornerstone.
- Model hosts such as Codex, Claude, and Gemini are replaceable clients over that contract.
- V1 exposes read-only MCP tools only.

## Consequences

- `/mcp` is the public model-facing endpoint for v1.
- `/api/v1/mcp/read` is removed instead of being preserved as a parallel compatibility surface.
- Host-specific instructions, agent skills, and orchestration packs remain outside Cornerstone core unless explicitly adopted later.
- Write, review, and officialization tools remain out of scope for the first MCP milestone.

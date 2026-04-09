# 2026-04-08 - rename from AI connectors to AI operator surfaces

## Summary

The team renamed the model-facing workstream from `AI connectors` to `AI operator surfaces`.

## Why this matters

`connectors` already has a stable meaning in Cornerstone: upstream source-ingestion integrations that bind external systems into source memory.

Using the same term for Codex-, Claude-, or Gemini-facing tool surfaces would have collapsed two different boundaries:
- source ingestion into Cornerstone
- model consumption of Cornerstone

The rename keeps source-ingestion semantics under the connectors specs and moves model-facing transport behavior under a dedicated AI operator surfaces spec.

## Outcome

- Cornerstone contract remains canonical.
- MCP is treated as the first model-facing transport.
- Host-specific agent customization stays outside product semantics for the v1 milestone.

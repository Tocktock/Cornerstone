# 2026-04-07 - implementation - personal connectors deferred for shared Notion P0

The first real connector implementation focuses on shared Notion sources before personal-source connectors.

Why:
- shared-source connector behavior is the main P0 acceptance surface
- manager-only binding, preview, sync history, and authorization already add substantial scope
- personal connectors carry additional privacy, promotion, and disclosure risk

This does not change the canonical personal-source boundary. It only sequences implementation so the shared path stabilizes first.

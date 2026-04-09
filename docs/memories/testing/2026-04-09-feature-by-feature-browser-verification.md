# 2026-04-09 - Feature-by-feature browser verification

## Context

Cornerstone already had Postgres-backed integration, contract, and MCP coverage, but the browser gate was still too monolithic and brittle:

- Playwright treated `/api/v1/health` as the backend boot gate instead of a post-boot smoke check.
- One large `app.spec.ts` file mixed many surfaces together, which made failures harder to localize.
- The default browser-backend port conflicted with another local listener on some developer machines.

That meant `make symptoms` could fail before the actual UI checks even started, and when it did run, the failure surface was noisier than the product surface being verified.

## Decision

The browser verification path now follows a feature-by-feature contract:

- Playwright boots the backend from a stable root endpoint and asserts `/api/v1/health` inside the suite.
- Browser tests are split by route and feature area:
  - bootstrap and access
  - sources and connector management
  - glossary and provenance disclosure
  - graph exploration
  - decisions and lineage
  - review and validation
  - dashboard retrieval and no-match behavior
- Shared helpers own actor switching, deterministic route waits, API-driven setup, and artifact capture.
- Browser artifacts now land under `output/playwright/`.
- The default browser-backend port moved to `8011` to avoid the observed local conflict on `8001`.

## Why it matters

- Browser failures now map directly to the feature area that regressed.
- The browser gate complements the existing backend integration, contract, and MCP checks instead of duplicating them.
- Local verification is more reliable because harness startup and product-health assertions are no longer conflated.

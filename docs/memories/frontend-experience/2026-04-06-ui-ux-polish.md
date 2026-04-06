# 2026-04-06 UI UX Polish

## Context

The current P0 frontend was functionally correct but exposed several high-friction symptoms in direct browser review:
- workspace and persona metadata collapsed into unreadable inline text
- mobile routing showed branding and navigation before task content
- the dashboard search form broke on narrow widths
- the graph route looked like a passive list rather than a relationship explorer
- review navigation led member actors into a raw permission dead end
- source status cards were technically correct but hard to scan quickly

## Intent

This pass keeps the existing visual direction and serving contract but raises usability for the existing route set.

The implementation should favor:
- permission-aware navigation over dead-end error states
- readable scan patterns before adding more information
- responsive layout fixes before deeper visual redesign
- route-specific clarity while reusing the current payloads and fixtures

## Traceability

- Spec anchor: `docs/specs/frontend-experience/ui-ux-polish.md`
- Implementation anchor: frontend shell, dashboard, glossary, decisions, graph, review, and sources surfaces
- Verification anchor: `frontend/tests/symptoms/app.spec.ts`, `npm run build`, and `npm run test:symptoms`

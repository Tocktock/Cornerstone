# 2026-04-06 Color System Refresh

## Context

The frontend became functionally stronger after the UI/UX polish pass, but it still looked like a default dark SaaS product:
- cyan and purple shared too much visual weight
- glass and glow softened operational pages more than necessary
- semantic states competed with the brand color
- route surfaces were readable but not distinctive

## Intent

This refresh keeps the current information architecture and route behavior while giving the product a stronger visual point of view.

The chosen direction is warm editorial:
- dark-first and trustworthy
- calmer surface hierarchy
- clearer semantic color roles
- more precise type and shape choices

## Traceability

- Spec anchor: `docs/specs/frontend-experience/color-system-refresh.md`
- Implementation anchor: shared CSS theme layer plus route-specific styling for dashboard, glossary, graph, decisions, review, and sources
- Verification anchor: `frontend/tests/symptoms/app.spec.ts`, `npm run build`, and `npm run test:symptoms`

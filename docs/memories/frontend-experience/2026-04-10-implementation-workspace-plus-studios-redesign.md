# 2026-04-10 Workspace Plus Studios Redesign

## Context

The 2026-04-09 Craft benchmark confirmed that Cornerstone's next frontend step could not be another incremental polish pass. The old shell and route model were still operationally correct, but they led with navigation and admin posture before the primary knowledge artifact.

The team wanted to translate Craft's strongest qualities into Cornerstone:
- content-first first view
- clearer artifact composition
- stronger mode-specific visual tone
- presentable detail surfaces

The team did not want to translate Craft into product scope expansion. Personal productivity behaviors such as journaling, tasks, and calendar planning remained out of scope.

## Intent

This implementation pass applies the benchmark by changing structure, not by copying consumer-note behavior.

The translation rules were:
- keep Cornerstone's trust and provenance vocabulary explicit
- make the default member experience artifact-first
- move review and source-management into secondary studios
- add only the contract changes needed to support the new IA and direct presentable routes

## Implementation notes

- Added `workspace_home` as an additive serving-contract surface for the new landing page.
- Added `DecisionPayload.public_slug` so concept and decision detail routes both use stable presentable URLs.
- Replaced the persistent sidebar shell with a compact header and workspace/profile tray.
- Consolidated glossary, decisions, and graph exploration under `Explore`.
- Split visual tone into reader routes and studio routes.
- Kept review authorization, support visibility, freshness semantics, and provenance summaries unchanged.

## Traceability

- Benchmark anchor: `docs/specs/frontend-experience/craft-ui-ux-benchmark.md`
- Owning spec: `docs/specs/frontend-experience/workspace-plus-studios-redesign.md`
- Durable rule: `docs/decisions/0018-workspace-plus-studios-navigation-separates-reader-and-studio-surfaces.md`
- Contract anchor: `docs/specs/serving-contract/spec.md`
- Frontend anchors: `frontend/src/App.tsx`, `frontend/src/components/Layout.tsx`, `frontend/src/styles.css`
- Verification anchors: `backend/tests/contract/test_contract_snapshots.py` and `frontend/tests/symptoms/`

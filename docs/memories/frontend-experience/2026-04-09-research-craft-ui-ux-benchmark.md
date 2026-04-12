# 2026-04-09 Research Craft UI UX Benchmark

## Context

The immediate user goal was to benchmark Craft Docs for design and UI UX before moving into further frontend direction work for Cornerstone.

This investigation was completed on 2026-04-09 and focused only on public product, help-center, App Store, and live Cornerstone frontend evidence.

## Source trail

External sources reviewed on 2026-04-09:
- [Craft homepage](https://www.craft.do/)
- [Craft Write](https://www.craft.do/write)
- [Craft Organize](https://www.craft.do/organize)
- [Craft Customize](https://www.craft.do/customize)
- [Craft Help Center: Write and Edit](https://support.craft.do/en/write-and-edit)
- [Craft Help Center: Journaling](https://support.craft.do/hc/en-us/articles/15335736436380-Journaling)
- [Craft Help Center: Share and Publish](https://support.craft.do/hc/en-us/categories/15274247430684-Share-Publish-and-Collaborate)
- [Craft Help Center: Switching from Notion to Craft](https://support.craft.do/hc/en-us/articles/20070315420956-Switching-from-Notion-to-Craft)
- [App Store listing: Craft: Notes, Documents, AI](https://apps.apple.com/us/app/craft-notes-documents-ai/id1487937127)

Cornerstone evidence reviewed:
- `frontend/src/components/Layout.tsx`
- `frontend/src/styles.css`
- `frontend/src/pages/DashboardPage.tsx`
- `frontend/src/pages/GlossaryPage.tsx`
- `frontend/src/pages/DecisionsPage.tsx`
- `frontend/src/pages/GraphPage.tsx`
- `frontend/src/pages/SourcesPage.tsx`
- screenshot artifacts under `output/playwright/`

Additional visual captures created during this investigation:
- `output/craft-organize.png`
- `output/craft-customize.png`

## Main observations

### Craft is emotionally calm and editorial

Craft presents itself less like a productivity control center and more like a designed home for thought. Even when it introduces structure, tasks, or AI, the presentation stays grounded in writing, mood, and personal clarity.

### Craft sells one continuous workspace, not a set of disconnected tools

The public product story joins writing, planning, organizing, whiteboards, publishing, and AI into one mental model. That continuity is a major reason the UI feels coherent.

### Craft makes styling and presentation part of the product promise

Craft does not hide visual expression behind secondary settings. Templates, styling, page appearance, block composition, and polished publishing outputs are part of the headline experience.

### Craft keeps organization flexible

Spaces, folders, tags, and collections coexist. The UX message is clear: capture first, structure when useful, and choose the organizational system that matches the user's thinking.

### Craft treats AI as a helper inside a stable interface

The current App Store listing and website position AI as an assistant that helps people find notes, edit documents, summarize, and work in the background. The interface story remains document-first.

## Interpretation for Cornerstone

Craft is a strong benchmark for:
- product calm
- typographic confidence
- integrated knowledge workflows
- modular document composition
- user-visible polish

Craft is a weak benchmark if interpreted as a reason to:
- hide provenance
- downplay state and permission
- remove route clarity
- turn Cornerstone into a consumer notes clone

The right translation for Cornerstone is:
- editorial calm for knowledge surfaces
- operational rigor for review and source-management surfaces
- better content hierarchy without losing trust semantics
- stronger presentation of artifacts, not just records

## Immediate implication

The benchmark should inform the next frontend direction in two layers:

1. A product-experience layer for how knowledge should feel to read, inspect, and present.
2. A route-architecture layer for how operator controls should remain available without visually overwhelming the knowledge artifact.

## Traceability

- Spec anchor: `docs/specs/frontend-experience/craft-ui-ux-benchmark.md`
- Existing frontend-experience anchors:
  - `docs/specs/frontend-experience/ui-ux-polish.md`
  - `docs/specs/frontend-experience/color-system-refresh.md`

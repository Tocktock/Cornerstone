# Page Clarity, Readability, and Prod Isolation

## Summary

This pass tightened Cornerstone’s frontend reading posture and local operator setup without changing canonical trust, provenance, lineage, access, or serving-contract semantics.

The implementation focused on three problems:
- reader and studio pages were over-explaining themselves before showing the primary artifact or workflow
- typography and spacing made dense pages harder to read than necessary
- the local production launcher could inherit demo-seeded data through the shared Compose persistence boundary

## Implemented decisions

- Shared reader/studio primitives now prefer sparse headers and title-first section intros.
- Reader routes drop repeated explanatory paragraphs where the artifact or workflow title already communicates intent.
- Concept and decision detail routes no longer repeat the hero definition or statement in the main narrative column.
- Shared typography now uses looser display tracking, looser label tracking, and slightly more generous paragraph rhythm.
- The shared shell header is shallower and the first-content offset is tighter so the first viewport reaches the primary route artifact sooner.
- The workspace/profile tray now closes on route and actor changes so it does not leave a stale overlay sitting on top of the page content.
- Compact and standard artifact summaries now clamp to two lines, while lead artifact summaries clamp to three lines.
- `run-dev.sh` and `run-prod.sh` now target separate Compose project names so production-like local startup does not reuse mock/demo database state.

## Why this translation

The problem was not just “too much copy.” The more important issue was that page structure treated explanation, artifact, and support rails as equal-weight content.

The fix therefore prioritized:
- one dominant first-viewport message per route
- explicit but subordinate trust cues
- lighter shell chrome
- operator-safe production isolation at the launcher level instead of trying to hide demo data after the fact

## Route outcomes

- `Workspace` now leads with the ask surface and one primary answer or artifact before secondary operational summaries.
- `Explore Topics` and `Explore Decisions` keep the browse family structure but remove most orientation prose, relying on the lead artifact, counts, and trust cues to carry the page.
- `Explore Map` now leans on selected-object context and relation grouping instead of explanatory “how to use this page” copy.
- `Concept Detail` and `Decision Detail` now treat the hero artifact as the primary reading block and reserve the rest of the page for genuinely secondary material.
- `Review Studio` and `Source Studio` keep one compact guidance block at most and otherwise let queue, intervention, and composer content lead.

## Verification

- `npm --prefix frontend run build`
- `cd frontend && CORNERSTONE_BROWSER_TEST_DATABASE_URL=postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone_test npm run test:symptoms`

Those checks passed after the copy-budget, layout, and route composition changes.

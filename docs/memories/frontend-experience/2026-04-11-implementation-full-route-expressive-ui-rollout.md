# 2026-04-11 Full-Route Expressive UI Rollout

## Context

The workspace-plus-studios redesign established the correct information architecture, but the first implementation was still conservative in route composition and visual identity.

The next step was to apply the benchmark more aggressively across all current routes without reopening canonical product rules. The team wanted a visibly stronger frontend language while keeping:
- serving-contract shapes unchanged
- current route map and redirects unchanged
- review authorization and connector permissions unchanged
- trust, provenance, freshness, and lineage vocabulary unchanged

## Intent

This pass translated the benchmark into a frontend-only rollout with one shared family language:
- reader routes became more editorial, layered, and artifact-led
- studio routes stayed operational, but gained stronger grouping and hierarchy
- shared experience primitives carried more of the visual identity so route pages did not fork their own ad hoc layouts

Any treatment that would have required contract, domain, access, or provenance semantics to change stayed out of scope.

## Implementation notes

- Reworked the shared shell into a lighter top-frame with clearer reader-versus-studio identity.
- Expanded shared experience primitives so artifact cards, provenance strips, lineage rails, alert banners, and section intros could support lead, compact, rail, and timeline presentation patterns.
- Rebuilt `Workspace` around a lead ask lane, featured answer panel, recent-change river, and a quieter support rail.
- Rebuilt `Explore Topics` and `Explore Decisions` into a shared browse family with lead artifacts and compact supporting cards.
- Reworked `Explore Map` around selected-object continuity, relation lanes, and stronger root-versus-linked concept separation.
- Restyled `Concept Detail` and `Decision Detail` into narrative reader artifacts with grouped support/provenance rails.
- Reorganized `Review Studio` into a lead queue item plus supporting queue stack with explicit action safety.
- Reorganized `Source Studio` into summary, composer, intervention, and healthy-monitoring zones while preserving operational density.

## Protected invariants

The rollout intentionally preserved:
- `support_visibility`
- `verification_state`
- `freshness_state`
- provenance summary semantics
- lineage semantics
- review authorization rules
- connector-management permissions
- existing route paths and request flows

## Verification

- Build: `cd frontend && npm run build`
- Synthetic browser suite: `cd frontend && npm run test:symptoms`
- Focused regression rerun: `cd frontend && npx playwright test tests/symptoms/04-decisions.spec.ts`

The final symptom run passed all 11 frontend browser tests after restoring stable DOM identities for map focus, decision lineage cards, workspace answer targeting, and source locator rendering.

## Traceability

- Owning spec: `docs/specs/frontend-experience/workspace-plus-studios-redesign.md`
- Frontend anchors:
  - `frontend/src/components/Layout.tsx`
  - `frontend/src/components/experience.tsx`
  - `frontend/src/styles.css`
  - `frontend/src/pages/WorkspacePage.tsx`
  - `frontend/src/pages/ExploreTopicsPage.tsx`
  - `frontend/src/pages/ExploreDecisionsPage.tsx`
  - `frontend/src/pages/ExploreMapPage.tsx`
  - `frontend/src/pages/ConceptDetailPage.tsx`
  - `frontend/src/pages/DecisionDetailPage.tsx`
  - `frontend/src/pages/ReviewStudioPage.tsx`
  - `frontend/src/pages/SourceStudioPage.tsx`

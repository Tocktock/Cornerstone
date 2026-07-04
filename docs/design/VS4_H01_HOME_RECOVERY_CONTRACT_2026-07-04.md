# VS4-H01 Home Recovery Contract

**Date:** 2026-07-04 KST  
**Owner:** JiYong / Tars  
**Status:** Frozen recovery contract for the rejected VS4-H01 Product Alpha Home UI  
**Applies to:** `VS4-H01` retry preparation and the VS5 external-user entry gate  
**Primary evidence:** `docs/verification-reports/VS4_H01_UI_UX_REJECTION_REPORT_2026-07-04.md` and `reports/human-gates/vs4/filled-records/VS4-H01.review-record.json`

## Feature / Task

Recover the rejected Product Alpha Home / landing experience by separating normal-user first value from internal verifier/review surfaces.

## Goal

A first-time reviewer should understand CornerStone's first value in five seconds:

```text
Drop / Ask -> Evidence-backed Brief -> Decision -> Audit
```

The Home screen must feel like a calm product workspace, not a scenario verifier, human-gate packet, admin console, or workflow debugger.

## Success Criteria

1. The first viewport visually prioritizes Drop and Ask, with a product promise focused on briefs with receipts.
2. The normal-user Home includes the reference-required regions: small standard nav, prominent global search, hero, large Drop Zone, Ask Box, suggested prompts, recent items, knowledge states, suggested next steps, and recent activity.
3. Internal review/proof details remain reachable but are not visually dominant on the default Home surface.
4. The Home surface does not claim production, on-prem, live-provider, final-security, human UX acceptance, or VS5 product-value readiness.
5. The next `VS4-H01` retry is supported by desktop and mobile screenshots plus a dated owner review record.

## Constraints

### Product / UX

- Use **CornerStone** for product/project language.
- The user-facing product story is one product, not visible Cornerstone + KnowledgeBase + ConnectorHub products.
- Default Home language uses Drop, Ask, Brief, Source, Decision, Review, Inbox, History.
- Avoid normal-user first-viewport jargon: scenario, human gate, package, verifier, raw command, ontology, connector, policy internals, dense audit IDs.
- Safety copy is allowed only as calm, human-facing boundary text such as `Local preview` and `No external send`.

### Data / State

- Do not weaken artifact preservation, evidence links, trust states, auditability, or owner/workspace scope.
- Do not mark template/extractive/model-unavailable output as `evidence_backed` in user-facing copy.
- Evidence, policy, approval, and audit detail must remain one action away through progressive disclosure.

### Permission / Security

- No external writeback.
- No destructive action.
- No live provider claim.
- No human acceptance claim without a dated human record.
- Do not hide safety boundaries; relocate them to the right layer.

### Compatibility / Format

- Preserve the existing standard navigation: Home, Search, Artifacts, Claims, Actions.
- Preserve existing VS4 structural surface markers unless a verifier update explicitly moves them to a proof/review surface.
- Do not introduce a new CLI command family.

### Operational / Environment

- Runtime evidence remains local-first.
- Automated checks prepare evidence only; subjective design acceptance remains `HUMAN_REQUIRED`.
- The PR may add a recovery prototype and static checks before the runtime swap, but it must not claim `VS4-H01` PASS.

## Assumptions

- The current `VS4-H01` owner decision is `REJECT`.
- VS5 external-user sessions remain blocked until a renewed owner decision is recorded.
- The reference-image direction is active design input but not PASS evidence by itself.
- A narrow Home recovery slice is allowed because it directly serves the VS5 entry gate and active product spine.

## Out of Scope Before Coding

- Full VS5 model-backed Brief / Ask implementation.
- New ingest connectors or live provider writeback.
- Memory promotion machinery, missions, autopilot, agent packs, ontology expansion, VS2/VS3 production hardening.
- New scenario families or report-family expansion beyond this focused recovery contract.
- Claiming `VS4-H01` accepted without a new owner review.

## Scenario Contract

| ID | Type | Trigger / Action | Expected Result | Affected Layers | Verification Method | Evidence Required | Owner |
|---|---|---|---|---|---|---|---|
| HREC-S01 | MUST_PASS | Open the recovery Home target. | First viewport exposes standard nav, global search, hero, large Drop Zone, Ask Box, suggested prompts, recent items, knowledge states, suggested next steps, and recent activity. | UI / design / source | Static source/DOM marker check. | All required region markers present. | AI |
| HREC-S02 | MUST_PASS | Inspect normal-user Home first viewport copy. | The page explains `Drop / Ask -> Brief with receipts` before internal proof details. | UX copy / UI | Static absence/presence check. | Product promise and Drop/Ask labels present; internal proof terms absent from first-viewport markup. | AI |
| HREC-S03 | MUST_PASS | Inspect normal-user Home first viewport for internal leakage. | Scenario verifier, human-gate package, package paths, raw command lists, dense audit IDs, and implementation readiness counters are absent from the default Home first viewport. | UI / verification boundary | Static forbidden-term check scoped to default Home prototype. | Forbidden first-viewport term count is zero. | AI |
| HREC-S04 | MUST_PASS | Inspect safety boundary copy. | Safety remains visible as calm chips (`Local preview`, `No external send`, `Sources preserved`, `Review before action`) without dominating the page. | UX copy / security boundary | Static marker check. | Required safety chips present; no production/live-provider/final-security readiness claim. | AI |
| HREC-S05 | MUST_PASS | Inspect progressive disclosure. | Evidence, policy, approval, audit, and proof details are reachable from a secondary review/proof drawer or section, not the main first task. | UI / safety / audit | Static marker check. | Progressive proof/review region present outside primary first-value region. | AI |
| HREC-S06 | MUST_PASS | Inspect mobile/responsive contract. | Recovery target defines a mobile-first collapse preserving Drop and Ask before secondary rails. | UI / responsive | Static CSS/source check. | Mobile media rule / responsive marker present. | AI |
| HREC-R01 | REGRESSION | Inspect claim boundary copy. | The recovery target does not claim production, on-prem, live-provider, final-security, human UX acceptance, or VS5 completion. | Docs / UI copy | Static forbidden-claim check. | Forbidden readiness claim count is zero. | AI |
| HREC-R02 | REGRESSION | Inspect active acceptance boundary. | `VS4-H01` remains human-owned and blocked until owner retry. | Docs / process | Source review. | Contract states no PASS without new owner record. | AI |
| HREC-H01 | HUMAN_REQUIRED | JiYong / Tars reviews the rebuilt runtime Home with screenshots. | Owner records `APPROVE`, `APPROVE_WITH_EXCEPTIONS`, or `REJECT`. | Human gate / product UX | Human walkthrough. | Filled `VS4-H01` retry record with screenshots/recording refs. | Human |

## Target Home Composition

```text
AppShell
  SidebarNav: Home, Search, Artifacts, Claims, Actions
  TopSearch: Search sources, briefs, claims, and actions...
  HomeMain
    Hero: Drop anything, or ask what we know.
    DropZone: Paste, upload, or drag a source.
    AskBox: Ask across saved sources.
    SuggestedPrompts
    RecentItems
    KnowledgeStateCard
    SuggestedNextSteps
  RecentActivityRail
  ProgressiveProofDrawer / ReviewProof section
```

## Must Not Show in Normal-User First Viewport

- scenario verifier language;
- human-gate package language;
- package paths;
- raw command lists;
- dense audit IDs;
- implementation readiness counters;
- admin / connector / ontology / policy-first content;
- production, on-prem, final-security, live-provider, human-acceptance, or VS5 completion claims.

## Runtime Recovery Notes

The current hardcoded runtime Home mixes normal-user product work with review packets, proof details, and structural scenario readiness. The runtime swap should move review/proof details out of the first-value region while preserving inspectability through progressive disclosure.

This contract intentionally does **not** mark `VS4-H01` as accepted. It defines the AI-verifiable target for a retry package and the human-required gate that remains after implementation.

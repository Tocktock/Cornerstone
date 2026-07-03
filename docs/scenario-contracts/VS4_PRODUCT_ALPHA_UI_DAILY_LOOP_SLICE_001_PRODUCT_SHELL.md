# CornerStone VS4 Slice 001 Product Alpha UI Shell

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice can prove only local Product Alpha shell behavior, not full VS4 completion.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`

## Goal

Implement the smallest high-value VS4 Product Alpha UI slice:

```text
Home / Ops Inbox
with Drop, Ask, Continue, pending work, workspace context, local-mode boundary,
and no production/on-prem/security/live-provider/human-UX overclaim.
```

The default user experience must start from product work, not scenario verification, admin connectors, ontology setup, or policy/security tooling.

## Scope

In this slice:

- render the Product Alpha Home / Ops Inbox as the first visible runtime surface;
- keep normal navigation small: `Home`, `Search`, `Artifacts`, `Claims`, `Actions`;
- show Drop and Ask entry points;
- show Continue work, pending briefs, evidence gaps, memory candidates, action cards, and recent activity;
- show active workspace context and local/mock execution boundary;
- preserve the existing VS0 and VS1 local runtime flows in the same server;
- add a native VS4 scenario verifier path that can verify the selected slice rows and report later rows as `NOT_RUN`;
- capture browser/DOM/screenshot evidence for the shell.

## Non-Scope

This slice does not implement the full VS4 loop:

- no durable first-class Brief storage model;
- no full Brief detail object lifecycle;
- no full Memory/Wiki approval lifecycle;
- no new live-provider writeback;
- no production/on-prem/final security readiness claim;
- no human product-alpha UX acceptance claim.

## Assumptions

- Existing VS0 runtime APIs for Artifact, Search, Evidence Bundle, Claim, Action, and Audit remain the underlying local proof substrate.
- The first VS4 UI shell may expose work items as local fixture-backed shell content before the full Brief/Memory data model is implemented.
- Reference images under `docs/design/reference-images/` guide layout and state language only; they are not PASS evidence.

## Selected Scenarios

This slice selects these AI-verifiable rows:

| ID | Classification | Why |
|---|---|---|
| VS4-GATE-001 | in_this_slice | Parent contract and matrix are already frozen and registered. |
| VS4-UI-001 | in_this_slice | Product Alpha Home is the central first screen. |
| VS4-UI-012 | in_this_slice | Ops Inbox returning-work shell is part of the first screen. |
| VS4-UI-015 | in_this_slice | Workspace context must be visible in the shell. |
| VS4-UI-016 | in_this_slice | Product language must lead normal-user UI copy. |
| VS4-REG-003 | in_this_slice | No forbidden readiness overclaim in VS4 UI/report/copy. |
| VS4-REG-006 | in_this_slice | Home must remain product-first, not admin/connector/ontology-first. |

## Full Scenario Classification

| ID | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001 | in_this_slice | Verify docs and matrix freeze. |
| VS4-UI-001 | in_this_slice | Verify Product Alpha Home shell. |
| VS4-UI-002 | later_slice | Requires functional Drop/Paste artifact preservation proof from VS4 shell. |
| VS4-UI-003 | later_slice | Requires Brief creation/preparation object proof. |
| VS4-UI-004 | later_slice | Requires Brief detail content and evidence linkage. |
| VS4-UI-005 | later_slice | Requires shared Evidence Drawer implementation. |
| VS4-UI-006 | later_slice | Requires Claim candidate creation from Brief. |
| VS4-UI-007 | later_slice | Requires VS4 Claim approval negative test. |
| VS4-UI-008 | later_slice | Requires Memory/Wiki candidate flow. |
| VS4-UI-009 | later_slice | Requires no-hidden-durable-memory state inspection. |
| VS4-UI-010 | later_slice | Requires Action Card detail from Brief/Claim. |
| VS4-UI-011 | later_slice | Requires Action Card local/mock execution-mode proof. |
| VS4-UI-012 | in_this_slice | Verify returning-work shell rows and Continue links. |
| VS4-UI-013 | later_slice | Requires Ask-to-work-item flow and state proof. |
| VS4-UI-014 | later_slice | Requires three general-purpose packs through the VS4 loop. |
| VS4-UI-015 | in_this_slice | Verify visible active workspace context. |
| VS4-UI-016 | in_this_slice | Verify product language dominates normal UI. |
| VS4-STATE-001 | later_slice | Requires state coverage across major VS4 pages. |
| VS4-REF-001 | later_slice | Requires source/browser design mapping for Home/Search/Artifact. |
| VS4-REF-002 | later_slice | Requires source/browser design mapping for Claim/Action. |
| VS4-REG-001 | later_slice | Requires fresh VS0 operator acceptance rerun after VS4 implementation. |
| VS4-REG-002 | later_slice | Requires fresh VS1 ontology rerun after VS4 implementation. |
| VS4-REG-003 | in_this_slice | Verify no overclaim markers in shell and VS4 report. |
| VS4-REG-004 | later_slice | Requires adversarial prompt-injection fixture against VS4 shell flows. |
| VS4-REG-005 | later_slice | Requires report lint that reference images are not PASS evidence. |
| VS4-REG-006 | in_this_slice | Verify default Home remains product-first. |
| VS4-REG-007 | later_slice | Requires full CLI parity gate for VS4 feature rows. |
| VS4-H01 | human_required | JiYong/Tars product-alpha UX acceptance remains human-only. |

## Proof Needed

The selected slice can be marked `PASS` only with:

- source references for the Product Alpha shell in `packages/cornerstone_cli/product_runtime.py`;
- native verifier output from `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario ... --json`;
- browser proof JSON, DOM, and screenshot for the VS4 shell;
- negative evidence that `production`, `on-prem`, `live-provider`, `final security`, and `human UX acceptance` are not claimed;
- `git diff --check`;
- relevant unit/scenario tests.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED`.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.
- These gates do not block this local Product Alpha shell slice.

## Done Criteria

This slice is done when:

1. the Product Alpha shell is first in the local runtime UI;
2. default normal-user navigation is `Home`, `Search`, `Artifacts`, `Claims`, `Actions`;
3. Drop, Ask, Continue, pending work, evidence gaps, memory candidates, action cards, recent activity, workspace context, and local-mode boundary are visible;
4. the VS4 verifier reports selected slice rows as `PASS` when filtered to this slice;
5. all non-selected VS4 rows remain explicitly `NOT_RUN` or `HUMAN_REQUIRED`;
6. existing VS0/VS1 runtime surfaces remain reachable;
7. no production/on-prem/security/live-provider/human-UX readiness claim is introduced.

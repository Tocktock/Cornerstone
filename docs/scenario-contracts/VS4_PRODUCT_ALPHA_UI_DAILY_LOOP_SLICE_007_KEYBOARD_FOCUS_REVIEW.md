# CornerStone VS4 Slice 007 Keyboard Focus Review

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice adds deterministic keyboard/focus proof for the local Product Alpha UI; it does not add canonical VS4 rows or provide human UX acceptance.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Previous slices:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_005_UX_POLISH_LEARN.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_006_RESPONSIVE_MOBILE_PROOF.md`

## Goal

Make the local VS4 Product Alpha daily loop more reviewable without a mouse by proving that the main surfaces expose clear keyboard entry points, visible focus treatment, valid Continue links, labeled controls, and reachable evidence/detail disclosures.

This proof is AI-verifiable browser/runtime evidence. It is not subjective accessibility certification and does not replace `VS4-H01`.

## Scope

In this slice:

- add a product-language skip link into the Product Alpha daily work surface;
- add visible focus treatment for links, buttons, inputs, textareas, selects, and disclosure summaries;
- ensure Home/Ops Inbox Continue links target real product sections;
- ensure Drop, Ask, Evidence Drawer, Brief detail, Claim candidate, Memory/Wiki candidate, Action Card, Learn review, Search, and Artifact surfaces expose keyboard-reachable anchors or controls;
- ensure primary form controls and buttons have accessible names;
- record keyboard/focus markers in the VS4 browser proof;
- register keyboard proof in the VS4 scenario report and human review package;
- leave the VS4 matrix at 20 `MUST_PASS`, 7 `REGRESSION_GUARD`, and 1 `HUMAN_REQUIRED` row.

## Non-Scope

This slice does not implement:

- WCAG certification or full accessibility audit;
- screen-reader manual acceptance;
- human Product Alpha UX acceptance;
- new canonical scenario rows or changes to the 206-scenario matrix;
- production persistence, production deployment, on-prem packaging, live-provider writeback, real IdP, or real network proof.

## Assumptions

- Deterministic browser/runtime checks can prove basic keyboard/focus structure: focusable controls, visible focus styling, valid targets, and labeled controls.
- Full assistive-technology review remains human/manual and outside AI PASS.
- Keyboard/focus proof strengthens existing VS4 rows rather than creating a new product behavior row.
- `VS4-H01` remains the only acceptance path for subjective product-alpha UX approval.

## Selected Scenarios

This slice strengthens existing rows rather than adding new ones:

| ID | Classification | Why |
|---|---|---|
| VS4-GATE-001 | in_this_slice | Slice 007 contract and registration must be frozen without changing the parent matrix. |
| VS4-UI-001 | in_this_slice | Home must be reachable from keyboard via skip link, primary nav, Drop, Ask, and Continue controls. |
| VS4-UI-005 | in_this_slice | Shared Evidence Drawer must be keyboard-reachable through a disclosure summary. |
| VS4-UI-010 | in_this_slice | Action Card review surface must be keyboard-reachable and labeled. |
| VS4-UI-012 | in_this_slice | Ops Inbox Continue links must target real product surfaces. |
| VS4-UI-013 | in_this_slice | Ask input and Ask-with-evidence control must be labeled and keyboard-focusable. |
| VS4-UI-016 | in_this_slice | Keyboard/focus markers must use product language and keep internal proof detail progressive. |
| VS4-STATE-001 | in_this_slice | Required page states must remain reachable without creating a keyboard trap or hiding review/recovery states. |
| VS4-REF-001 | in_this_slice | Home, Search, and Artifact reference-aligned surfaces must expose keyboard-reachable structure. |
| VS4-REF-002 | in_this_slice | Claim and Action reference-aligned surfaces must expose keyboard-reachable trust/action detail. |
| VS4-REG-003 | in_this_slice | Keyboard proof must preserve no production/on-prem/final-security/live-provider/human-UX overclaim. |
| VS4-REG-006 | in_this_slice | Product-first small nav must remain keyboard-reachable without admin-first default. |
| VS4-H01 | human_required | JiYong/Tars UX/accessibility acceptance remains human-only. |

All other AI-verifiable rows are `previous_slice` and must still pass through the unfiltered VS4 verifier.

## Full Scenario Classification

| ID Range | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001, VS4-UI-001, VS4-UI-005, VS4-UI-010, VS4-UI-012, VS4-UI-013, VS4-UI-016, VS4-STATE-001, VS4-REF-001, VS4-REF-002, VS4-REG-003, VS4-REG-006 | in_this_slice | Strengthened by browser-collected keyboard/focus markers. |
| VS4-UI-002 through VS4-UI-004, VS4-UI-006 through VS4-UI-009, VS4-UI-011, VS4-UI-014 through VS4-UI-015, VS4-REG-001 through VS4-REG-002, VS4-REG-004 through VS4-REG-005, VS4-REG-007 | previous_slice | Must remain locally `PASS`; this slice must not weaken existing evidence. |
| VS4-H01 | human_required | Human acceptance remains unclaimed until JiYong/Tars provides dated review evidence. |
| VS3-H01 through VS3-H07 | conditional_deferred | Still block production/on-prem/security/live-provider/human acceptance claims only. |

## Proof Needed

The selected slice can be considered prepared only with:

- a frozen Slice 007 contract registered in the repo README and SoT README;
- desktop and mobile browser proof still reporting `PASS`;
- keyboard/focus markers proving skip link target, landmarks, primary nav reachability, valid Continue targets, labeled controls, evidence drawer disclosure reachability, Claim/Action detail reachability, visible focus style, and no generic or overclaiming focus copy;
- negative evidence counters showing no hidden durable memory, live external writeback, production/on-prem/final-security/live-provider claims, or human UX acceptance claim;
- the full VS4 verifier still reporting 27 AI-verifiable rows `PASS`, 1 `HUMAN_REQUIRED`, and 0 blocking rows;
- targeted unit tests plus docs, CLI-native-first, design-system, canonical matrix, py-compile, and `git diff --check` checks.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED`.
- Slice 007 can improve human review readiness, but it cannot create, infer, or validate JiYong/Tars acceptance.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.

## Done Criteria

This slice is done when:

1. the VS4 verifier emits a `keyboard_focus_proof` or equivalent keyboard/focus marker set in browser proof;
2. desktop and mobile browser proofs record all keyboard/focus markers as true;
3. selected scenario rows are classified as `in_this_slice` without changing the parent VS4 matrix;
4. the human review package includes keyboard/focus proof as review input only;
5. all existing VS4 AI-verifiable rows still pass and `VS4-H01` remains `HUMAN_REQUIRED`;
6. no report, README, UI, package, or validation output claims production, on-prem, final security, live-provider, accessibility certification, or human UX readiness.

## CLI Parity

- Verification: `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json`.
- Gate: `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json`.
- Slice target: `make verify-vs4-product-alpha-keyboard-focus`.
- CLI status: the existing VS4 CLI/API paths remain the product behavior proof. Slice 007 adds browser-state keyboard/focus proof only; it does not create a UI-only feature PASS.

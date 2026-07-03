# CornerStone VS4 Slice 006 Responsive Mobile Proof

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice adds deterministic narrow-viewport proof for the local Product Alpha UI; it does not add canonical VS4 rows or provide human UX acceptance.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Previous slices:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_005_UX_POLISH_LEARN.md`

## Goal

Prove that the current local VS4 Product Alpha daily loop remains usable and structurally contained at a narrow mobile viewport while preserving the same product-first shell:

```text
Drop / Ask
-> Evidence-backed Brief
-> Claim candidate
-> Memory/Wiki candidate
-> Action Card draft
-> Ops Inbox follow-up
-> Evidence/Audit detail
-> Learn
```

The proof must be browser/runtime evidence, not reference-image interpretation, docs-only intent, or subjective human acceptance.

## Scope

In this slice:

- capture a second VS4 browser proof at a narrow viewport (`390,844`);
- record responsive layout markers from the live DOM/runtime state;
- prove the Product Alpha shell, Drop, Ask, Ops Inbox, workspace context, product-language status, Learn review, and safe-action boundary remain visible at mobile width;
- prove major grids collapse to one column at mobile width;
- prove the document body does not horizontally overflow the viewport;
- contain the page-state matrix inside its own horizontal scroll region instead of expanding the body;
- register the mobile proof in the VS4 scenario report and human review package;
- leave the VS4 matrix at 20 `MUST_PASS`, 7 `REGRESSION_GUARD`, and 1 `HUMAN_REQUIRED` row.

## Non-Scope

This slice does not implement:

- human Product Alpha UX acceptance;
- new canonical scenario rows or changes to the 206-scenario matrix;
- production persistence, production deployment, on-prem packaging, live-provider writeback, real IdP, or real network proof;
- final security acceptance or VS3-P release unlock;
- native mobile app behavior;
- accessibility certification beyond the responsive browser markers described here.

## Assumptions

- The local browser proof can use a deterministic headless Chrome viewport as AI-verifiable evidence for responsive layout containment.
- The mobile proof strengthens existing VS4 rows rather than creating new product behavior.
- A horizontally scrollable state table is acceptable if the body itself does not overflow and the table is explicitly contained.
- `VS4-H01` remains the only acceptance path for subjective product-alpha UX approval.

## Selected Scenarios

This slice strengthens existing rows rather than adding new ones:

| ID | Classification | Why |
|---|---|---|
| VS4-GATE-001 | in_this_slice | Slice 006 contract and registration must be frozen without changing the parent matrix. |
| VS4-UI-001 | in_this_slice | Home must render the Product Alpha shell with Drop and Ask at mobile width. |
| VS4-UI-004 | in_this_slice | Brief contents and Learn review output must remain visible in the mobile detail loop. |
| VS4-UI-012 | in_this_slice | Ops Inbox returning work must remain visible and stacked at mobile width. |
| VS4-UI-015 | in_this_slice | Workspace context must remain visible at mobile width. |
| VS4-UI-016 | in_this_slice | Product language must remain first at mobile width. |
| VS4-STATE-001 | in_this_slice | Required page states must be contained without body horizontal overflow. |
| VS4-REG-003 | in_this_slice | Mobile proof must preserve no production/on-prem/final-security/live-provider/human-UX overclaim. |
| VS4-REG-006 | in_this_slice | Mobile first screen must remain product-first, small-nav, and non-admin-first. |
| VS4-H01 | human_required | JiYong/Tars UX acceptance remains human-only; responsive proof cannot replace it. |

All other AI-verifiable rows are `previous_slice` and must still pass through the unfiltered VS4 verifier.

## Full Scenario Classification

| ID Range | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001, VS4-UI-001, VS4-UI-004, VS4-UI-012, VS4-UI-015, VS4-UI-016, VS4-STATE-001, VS4-REG-003, VS4-REG-006 | in_this_slice | Strengthened by narrow-viewport browser proof and responsive markers. |
| VS4-UI-002 through VS4-UI-011 except VS4-UI-004 and VS4-UI-012, VS4-UI-013 through VS4-UI-014, VS4-REF-001 through VS4-REF-002, VS4-REG-001 through VS4-REG-002, VS4-REG-004 through VS4-REG-005, VS4-REG-007 | previous_slice | Must remain locally `PASS`; this slice must not weaken existing evidence. |
| VS4-H01 | human_required | Human acceptance remains unclaimed until JiYong/Tars provides dated review evidence. |
| VS3-H01 through VS3-H07 | conditional_deferred | Still block production/on-prem/security/live-provider/human acceptance claims only. |

## Proof Needed

The selected slice can be considered prepared only with:

- a frozen Slice 006 contract registered in the repo README and SoT README;
- desktop browser proof still reporting `PASS`;
- mobile browser proof JSON, DOM, and screenshot at `390,844`;
- mobile responsive markers proving no body horizontal overflow, one-column main/topbar/drop-ask/ops/work-row layout, visible global search, visible product shell, visible Ops Inbox, visible workspace context, visible Learn review, and contained state matrix;
- negative evidence counters showing no hidden durable memory, live external writeback, production/on-prem/final-security/live-provider claims, or human UX acceptance claim;
- the full VS4 verifier still reporting 27 AI-verifiable rows `PASS`, 1 `HUMAN_REQUIRED`, and 0 blocking rows;
- targeted unit tests plus docs, CLI-native-first, design-system, canonical matrix, and `git diff --check` checks.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED`.
- Slice 006 can make human review safer on mobile-sized screens, but it cannot create, infer, or validate JiYong/Tars acceptance.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.

## Done Criteria

This slice is done when:

1. the VS4 verifier emits a dedicated `mobile_browser_proof` object;
2. mobile proof records `browser.window_size="390,844"`;
3. mobile proof records all responsive markers as true;
4. the state matrix is inside a contained scroll region and does not expand the body;
5. selected scenario rows are classified as `in_this_slice` without changing the parent VS4 matrix;
6. all existing VS4 AI-verifiable rows still pass and `VS4-H01` remains `HUMAN_REQUIRED`;
7. no report, README, UI, package, or validation output claims production, on-prem, final security, live-provider, or human UX readiness.

## CLI Parity

- Verification: `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json`.
- Gate: `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json`.
- Slice target: `make verify-vs4-product-alpha-responsive-mobile`.
- CLI status: the existing VS4 CLI/API paths remain the product behavior proof. Slice 006 adds browser-state responsive proof only; it does not create a UI-only feature PASS.

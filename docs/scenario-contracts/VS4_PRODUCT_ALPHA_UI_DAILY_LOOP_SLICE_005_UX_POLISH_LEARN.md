# CornerStone VS4 Slice 005 UX Polish and Learn Review

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice improves local Product Alpha human-review readiness; it does not add canonical VS4 rows or provide human UX acceptance.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Previous slices:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md`

## Goal

Make the already-passing local VS4 Product Alpha loop more reviewable as a daily product surface:

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

This slice must reduce normal-user exposure to internal proof/readiness flags, make Learn a first-class review surface, and preserve every existing AI-verifiable VS4 `PASS` boundary without claiming production, on-prem, final security, live-provider, or human UX readiness.

## Scope

In this slice:

- move raw proof/readiness terms such as `local_scenario_ready`, `production_release_ready`, and `real_external_http_calls` out of the normal-user status row into a progressive proof-details surface;
- keep normal-user copy product-first: Local mode, workspace scope, review required, no live writeback;
- add a first-class Learn review candidate to Home/Ops Inbox and the Brief detail loop;
- show that outcomes, corrections, approvals, rejections, and failures are learning inputs only after review;
- keep learning owner-scoped and unable to change durable memory, claims, or action behavior before review;
- extend browser/runtime proof markers and tests to cover proof-jargon progressive disclosure and Learn review visibility;
- leave the VS4 matrix at 20 `MUST_PASS`, 7 `REGRESSION_GUARD`, and 1 `HUMAN_REQUIRED` row.

## Non-Scope

This slice does not implement:

- human Product Alpha UX acceptance;
- new canonical scenario rows or changes to the 206-scenario matrix;
- production persistence, production deployment, on-prem packaging, live-provider writeback, real IdP, or real network proof;
- final security acceptance or VS3-P release unlock;
- autonomous learning, hidden durable memory, hidden model adaptation, or background action execution.

## Assumptions

- Existing VS4 local verifier rows already cover the AI-verifiable loop; this slice strengthens the human-review surface rather than changing the canonical ledger.
- Product language can be checked through DOM/browser markers while internal proof details remain available for operators and verifiers.
- Learn can be represented as a local review candidate without deciding a final backend storage model.
- `VS4-H01` remains the only acceptance path for subjective product-alpha UX approval.

## Selected Scenarios

This slice strengthens existing rows rather than adding new ones:

| ID | Classification | Why |
|---|---|---|
| VS4-UI-001 | in_this_slice | Home must remain the daily product shell while hiding raw proof jargon from the first normal-user status row. |
| VS4-UI-004 | in_this_slice | Brief contents must include the Learn review output as part of the daily loop. |
| VS4-UI-012 | in_this_slice | Ops Inbox must include a returning-work Learn candidate, not only brief/claim/memory/action follow-up. |
| VS4-UI-016 | in_this_slice | Product language must come before internal proof terms. |
| VS4-REG-003 | in_this_slice | Moving proof flags must preserve the no production/security/live-provider overclaim boundary. |
| VS4-REG-006 | in_this_slice | Product-first default must remain small-nav and non-admin-first. |
| VS4-H01 | human_required | JiYong/Tars UX acceptance remains human-only; this slice only improves reviewability. |

All other AI-verifiable rows are `previous_slice` and must still pass through the unfiltered VS4 verifier.

## Full Scenario Classification

| ID Range | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001 | previous_slice | Contract/matrix freeze remains verified; Slice 005 contract is additional local implementation guidance only. |
| VS4-UI-001, VS4-UI-004, VS4-UI-012, VS4-UI-016, VS4-REG-003, VS4-REG-006 | in_this_slice | Strengthened by product-language, Learn-review, and progressive proof-detail markers. |
| VS4-UI-002, VS4-UI-003, VS4-UI-005 through VS4-UI-011, VS4-UI-013 through VS4-UI-015, VS4-STATE-001, VS4-REF-001 through VS4-REF-002, VS4-REG-001 through VS4-REG-002, VS4-REG-004 through VS4-REG-005, VS4-REG-007 | previous_slice | Must remain locally `PASS`; this slice must not weaken existing evidence. |
| VS4-H01 | human_required | Human acceptance remains unclaimed until JiYong/Tars provides dated review evidence. |
| VS3-H01 through VS3-H07 | conditional_deferred | Still block production/on-prem/security/live-provider/human acceptance claims only. |

## Proof Needed

The selected slice can be considered prepared only with:

- a frozen Slice 005 contract registered in the repo SoT/README index;
- browser proof JSON, DOM, and screenshot showing the Product Alpha shell still passes;
- browser markers proving the normal-user status row uses product language and raw proof terms are progressively disclosed;
- browser/runtime markers proving Learn review is visible from Home/Ops Inbox and included in the Brief loop state;
- negative evidence counters showing no approved durable memory, hidden learning, live external writeback, production/on-prem/final-security/live-provider claims, or human UX acceptance claim;
- the full VS4 verifier still reporting 27 AI-verifiable rows `PASS`, 1 `HUMAN_REQUIRED`, and 0 blocking rows;
- targeted unit tests plus docs, CLI-native-first, design-system, canonical matrix, and `git diff --check` checks.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED`.
- Slice 005 can make the human review easier, but it cannot create, infer, or validate JiYong/Tars acceptance.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.

## Done Criteria

This slice is done when:

1. the first normal-user status row no longer exposes raw proof/readiness flag strings;
2. internal proof flags remain reachable in a collapsed proof-details surface for verifier/operator use;
3. Home/Ops Inbox includes a Learn review candidate with evidence, outcome, correction/failure, and owner-scope language;
4. the Brief detail loop emits Learn candidate state that cannot change durable memory, answers, claims, or actions before review;
5. browser proof records `normal_user_status_product_language`, `proof_details_progressively_disclosed`, `learn_review_visible`, and `learn_candidate_detail_visible` as true;
6. all existing VS4 AI-verifiable rows still pass and `VS4-H01` remains `HUMAN_REQUIRED`;
7. no report, README, UI, package, or validation output claims production, on-prem, final security, live-provider, or human UX readiness.

## CLI Parity

- Verification: `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json`.
- Gate: `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json`.
- Slice target: `make verify-vs4-product-alpha-ux-polish-learn`.
- CLI status: the existing VS4 CLI/API paths remain the product behavior proof. Slice 005 adds presentation and browser-state assertions only; it does not create a UI-only feature PASS.

# CornerStone VS4 Slice 009 Claim and Action Nav Detail

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice turns the normal-user `Claims` and `Actions` nav destinations into product-ready review pages; it does not add canonical VS4 rows or provide human UX acceptance.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Previous slices:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_005_UX_POLISH_LEARN.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_006_RESPONSIVE_MOBILE_PROOF.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_007_KEYBOARD_FOCUS_REVIEW.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_008_ASK_RESULT_READABILITY.md`

## Goal

Make the `Claims` and `Actions` normal-user nav targets useful Product Alpha review pages.

After Drop / Ask creates reviewable work, a user who follows `Continue` or the primary nav should land on pages that show the Claim candidate trust ladder and Action Card draft details in product language, not placeholder copy.

## Scope

In this slice:

- replace the placeholder `Claim Builder` section with a Claim review page that shows Draft / Evidence-backed / Approved trust state, evidence, zero-evidence denial, owner/workspace scope, and audit/activity readiness;
- replace the placeholder `Action Card` section with an Action review page that shows goal, why, evidence, proposed change, expected impact, risk, approval, execution mode, dry-run/local mock boundary, and activity readiness;
- update the VS4 browser runtime so these pages are populated by the same Drop / Ask / Evidence-backed Brief flow, not a separate UI-only path;
- add deterministic desktop and mobile browser markers proving the claim/action pages are visible, product-language, evidence-aware, local/mock bounded, and free of production/live/human overclaims;
- preserve the small normal nav: `Home`, `Search`, `Artifacts`, `Claims`, `Actions`;
- leave the VS4 parent matrix at 20 `MUST_PASS`, 7 `REGRESSION_GUARD`, and 1 `HUMAN_REQUIRED` row.

## Non-Scope

This slice does not implement:

- final human Product Alpha UX acceptance;
- approved durable claims without human/owner approval evidence;
- durable memory approval;
- live external writeback;
- new Claim or Action backend storage models;
- new canonical scenario rows or changes to the 206-scenario matrix;
- production persistence, production deployment, on-prem packaging, live-provider writeback, real IdP, or real network proof.

## Assumptions

- Claim and Action detail already exist in the Brief rail; this slice makes the primary nav destinations first-class review pages using the same state.
- Product-language page readiness can be verified through deterministic DOM/browser markers without judging subjective UX acceptance.
- CLI parity remains anchored in the existing `cornerstone claim ...`, `cornerstone action ...`, `cornerstone scenario verify ...`, and audit paths; this slice improves browser presentation of those behaviors.
- `VS4-H01` remains the only acceptance path for subjective product-alpha UX approval.

## Selected Scenarios

This slice strengthens existing rows rather than adding new ones:

| ID | Classification | Why |
|---|---|---|
| VS4-GATE-001 | in_this_slice | Slice 009 contract and registration must be frozen without changing the parent matrix. |
| VS4-UI-006 | in_this_slice | Claim candidate must be reviewable from the normal `Claims` destination, with trust ladder and evidence. |
| VS4-UI-007 | in_this_slice | Evidence-free approval must remain visibly blocked from the Claim page. |
| VS4-UI-010 | in_this_slice | Action Card draft must be reviewable from the normal `Actions` destination with goal, why, evidence, impact, risk, approval, and execution mode. |
| VS4-UI-011 | in_this_slice | Action page must preserve local/mock execution mode and no live writeback. |
| VS4-REF-002 | in_this_slice | Claim and Action reference alignment must include normal-nav destinations, not only Brief rail content. |
| VS4-REG-003 | in_this_slice | Claim/Action pages must not claim production, on-prem, final security, live-provider, or human UX readiness. |
| VS4-REG-006 | in_this_slice | Normal-user `Claims` and `Actions` destinations must stay product-first, not admin/connector/ontology-first. |
| VS4-H01 | human_required | JiYong/Tars product-alpha UX acceptance remains human-only. |

All other AI-verifiable rows are `previous_slice` and must still pass through the unfiltered VS4 verifier.

## Full Scenario Classification

| ID Range | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001, VS4-UI-006, VS4-UI-007, VS4-UI-010, VS4-UI-011, VS4-REF-002, VS4-REG-003, VS4-REG-006 | in_this_slice | Strengthened by Claim/Action nav-detail page markers and shared VS4 state proof. |
| VS4-UI-001 through VS4-UI-005, VS4-UI-008 through VS4-UI-009, VS4-UI-012 through VS4-UI-016, VS4-STATE-001, VS4-REF-001, VS4-REG-001 through VS4-REG-002, VS4-REG-004 through VS4-REG-005, VS4-REG-007 | previous_slice | Must remain locally `PASS`; this slice must not weaken existing evidence. |
| VS4-H01 | human_required | Human acceptance remains unclaimed until JiYong/Tars provides dated review evidence. |
| VS3-H01 through VS3-H07 | conditional_deferred | Still block production/on-prem/security/live-provider/human acceptance claims only. |

## Proof Needed

The selected slice can be considered prepared only with:

- a frozen Slice 009 contract registered in the repo README and SoT README;
- desktop and mobile browser proof still reporting `PASS`;
- Claim page markers proving the primary-nav `Claims` target is visible, uses product language, shows Draft / Evidence-backed / Approved trust state, preserves evidence, and shows zero-evidence approval as blocked;
- Action page markers proving the primary-nav `Actions` target is visible, shows goal, why, evidence, impact, risk, approval, execution mode, local/mock preview, and no live writeback;
- normal nav still exactly `Home`, `Search`, `Artifacts`, `Claims`, `Actions`;
- keyboard/focus markers still true after replacing the placeholder pages;
- negative evidence counters showing no hidden durable memory, live external writeback, production/on-prem/final-security/live-provider claims, accessibility certification claim, or human UX acceptance claim;
- the full VS4 verifier still reporting 27 AI-verifiable rows `PASS`, 1 `HUMAN_REQUIRED`, and 0 blocking rows;
- targeted unit tests plus docs, CLI-native-first, design-system, canonical matrix, py-compile, and `git diff --check` checks.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED`.
- Slice 009 can improve human review readiness, but it cannot create, infer, or validate JiYong/Tars acceptance.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.

## Done Criteria

This slice is done when:

1. the VS4 browser proof emits Claim/Action nav-detail markers;
2. desktop and mobile browser proofs record all Claim/Action nav-detail markers as true;
3. the `Claims` primary-nav target is a product Claim review page, not placeholder copy;
4. the `Actions` primary-nav target is a product Action Card review page, not placeholder copy;
5. selected scenario rows are classified as `in_this_slice` without changing the parent VS4 matrix;
6. all existing VS4 AI-verifiable rows still pass and `VS4-H01` remains `HUMAN_REQUIRED`;
7. no report, README, UI, package, or validation output claims production, on-prem, final security, live-provider, accessibility certification, or human UX readiness.

## CLI Parity

- Verification: `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json`.
- Gate: `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json`.
- Slice target: `make verify-vs4-product-alpha-decision-pages`.
- CLI status: the existing VS4 CLI/API paths remain the product behavior proof. Slice 009 improves browser presentation of already-created Claim and Action work and does not create a UI-only feature PASS.

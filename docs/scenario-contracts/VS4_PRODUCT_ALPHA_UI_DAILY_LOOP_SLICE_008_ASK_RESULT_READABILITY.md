# CornerStone VS4 Slice 008 Ask Result Readability

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice strengthens the local Product Alpha Ask result so created work is readable before raw refs; it does not add canonical VS4 rows or provide human UX acceptance.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Previous slices:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_005_UX_POLISH_LEARN.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_006_RESPONSIVE_MOBILE_PROOF.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_007_KEYBOARD_FOCUS_REVIEW.md`

## Goal

Make the Ask result feel like a Product Alpha work handoff rather than a raw object-ID transcript. After a user asks a question, the first visible answer should name the created work in product language: Evidence-backed Brief, Claim candidate, Memory/Wiki candidate, and Action Card draft.

Raw refs must remain available for evidence, auditability, and CLI parity, but they should be progressively disclosed instead of dominating the normal answer surface.

## Scope

In this slice:

- replace the normal Ask result paragraph with product-language created-work copy;
- render created work as labeled chips or rows for Evidence-backed Brief, Claim candidate, Memory/Wiki candidate, and Action Card draft;
- keep raw refs in a closed details/disclosure surface tied to the Ask result;
- record deterministic browser markers that prove readable labels are visible, raw refs are preserved, and raw refs are not first-order answer copy;
- preserve desktop and mobile browser proof, keyboard/focus proof, CLI parity, and existing VS4 AI-verifiable rows;
- leave the VS4 parent matrix at 20 `MUST_PASS`, 7 `REGRESSION_GUARD`, and 1 `HUMAN_REQUIRED` row.

## Non-Scope

This slice does not implement:

- human Product Alpha UX acceptance;
- a new conversational AI engine or semantic answer quality scoring;
- durable memory approval;
- claim approval without evidence;
- live external writeback;
- new canonical scenario rows or changes to the 206-scenario matrix;
- production persistence, production deployment, on-prem packaging, live-provider writeback, real IdP, or real network proof.

## Assumptions

- Ask readability can be verified through deterministic DOM/browser markers without judging subjective answer quality.
- Created object refs are still useful implementation evidence, but they belong in progressive detail rather than normal user copy.
- Improving the Ask result strengthens existing VS4 rows; it does not create a new product behavior row.
- `VS4-H01` remains the only acceptance path for subjective product-alpha UX approval.

## Selected Scenarios

This slice strengthens existing rows rather than adding new ones:

| ID | Classification | Why |
|---|---|---|
| VS4-GATE-001 | in_this_slice | Slice 008 contract and registration must be frozen without changing the parent matrix. |
| VS4-UI-001 | in_this_slice | Home Ask result must remain product-first as part of the first daily-work surface. |
| VS4-UI-012 | in_this_slice | Ops Inbox and Continue work must remain aligned with the created work surfaced from Ask. |
| VS4-UI-013 | in_this_slice | Ask must produce reviewable work items without degrading into raw-ref or chatbot-only output. |
| VS4-UI-016 | in_this_slice | Product language must appear before internal IDs and proof details in the normal Ask result. |
| VS4-REG-003 | in_this_slice | Ask readability proof must preserve no production/on-prem/final-security/live-provider/human-UX overclaim. |
| VS4-REG-006 | in_this_slice | Product-first Home remains the default; raw implementation/proof detail stays secondary. |
| VS4-H01 | human_required | JiYong/Tars product-alpha UX acceptance remains human-only. |

All other AI-verifiable rows are `previous_slice` and must still pass through the unfiltered VS4 verifier.

## Full Scenario Classification

| ID Range | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001, VS4-UI-001, VS4-UI-012, VS4-UI-013, VS4-UI-016, VS4-REG-003, VS4-REG-006 | in_this_slice | Strengthened by Ask readability markers and product-language answer proof. |
| VS4-UI-002 through VS4-UI-011, VS4-UI-014 through VS4-UI-015, VS4-STATE-001, VS4-REF-001 through VS4-REF-002, VS4-REG-001 through VS4-REG-002, VS4-REG-004 through VS4-REG-005, VS4-REG-007 | previous_slice | Must remain locally `PASS`; this slice must not weaken existing evidence. |
| VS4-H01 | human_required | Human acceptance remains unclaimed until JiYong/Tars provides dated review evidence. |
| VS3-H01 through VS3-H07 | conditional_deferred | Still block production/on-prem/security/live-provider/human acceptance claims only. |

## Proof Needed

The selected slice can be considered prepared only with:

- a frozen Slice 008 contract registered in the repo README and SoT README;
- desktop and mobile browser proof still reporting `PASS`;
- Ask readability markers proving product labels are visible for Brief, Claim, Memory/Wiki, and Action Card;
- raw refs preserved in a closed progressive detail surface;
- normal Ask answer copy free of raw object IDs such as `brief_`, `claim_`, `evb_`, and `action_`;
- keyboard/focus markers still true after the Ask result structure changes;
- negative evidence counters showing no hidden durable memory, live external writeback, production/on-prem/final-security/live-provider claims, accessibility certification claim, or human UX acceptance claim;
- the full VS4 verifier still reporting 27 AI-verifiable rows `PASS`, 1 `HUMAN_REQUIRED`, and 0 blocking rows;
- targeted unit tests plus docs, CLI-native-first, design-system, canonical matrix, py-compile, and `git diff --check` checks.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED`.
- Slice 008 can improve human review readiness, but it cannot create, infer, or validate JiYong/Tars acceptance.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.

## Done Criteria

This slice is done when:

1. the VS4 browser proof emits Ask readability markers;
2. desktop and mobile browser proofs record all Ask readability markers as true;
3. the normal Ask result displays product-language created-work labels before raw refs;
4. raw refs remain available in progressive detail for evidence/audit/CLI parity;
5. selected scenario rows are classified as `in_this_slice` without changing the parent VS4 matrix;
6. all existing VS4 AI-verifiable rows still pass and `VS4-H01` remains `HUMAN_REQUIRED`;
7. no report, README, UI, package, or validation output claims production, on-prem, final security, live-provider, accessibility certification, or human UX readiness.

## CLI Parity

- Verification: `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json`.
- Gate: `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json`.
- Slice target: `make verify-vs4-product-alpha-ask-readability`.
- CLI status: the existing VS4 CLI/API paths remain the product behavior proof. Slice 008 improves browser presentation of already-created Ask work and does not create a UI-only feature PASS.

# CornerStone VS4 Slice 013 Desktop Overflow Containment

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice fixes a desktop Product Alpha layout defect; it does not add canonical VS4 rows or provide production, live-provider, final security, or human UX acceptance.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Previous slices:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md` through `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_012_ACTION_EXECUTION_BOUNDARY.md`

## Goal

Keep the Product Alpha desktop workspace contained within the viewport even when policy reason codes, safety-envelope labels, evidence refs, or audit refs are long.

The user should be able to review the Home/Ops Inbox, Brief detail, Claim, Action, Evidence Drawer, and Audit/Learn surfaces without body-level horizontal scrolling or clipped safety text.

## Scope

In this slice:

- fix desktop body-level horizontal overflow in the VS4 Product Alpha UI;
- keep mobile no-overflow proof passing;
- preserve visible action-denial reason codes, policy, safety-envelope, and audit language without letting long tokens resize the layout;
- expose desktop overflow containment in browser proof and selected scenario evidence;
- keep the normal user nav small: `Home`, `Search`, `Artifacts`, `Claims`, `Actions`;
- keep the parent VS4 matrix at 20 `MUST_PASS`, 7 `REGRESSION_GUARD`, and 1 `HUMAN_REQUIRED`.

## Non-Scope

This slice does not implement or claim:

- a broad visual redesign;
- new canonical VS4 scenario rows or changes to the 206-scenario matrix;
- live external writeback, production, on-prem, final security, real IdP, real network, or live-provider readiness;
- final human Product Alpha UX acceptance;
- accessibility certification;
- changes to the action denial policy model.

## Assumptions

- The current desktop overflow is caused by long policy/action boundary tokens inside the Brief detail and Action review surfaces.
- Wrapping long safety/evidence/audit tokens is a Product Alpha usability requirement because users must see policy and denial detail without layout breakage.
- Reference images guide density and containment, but they are not PASS evidence.
- `VS4-H01` remains the only path for subjective Product Alpha UX acceptance.

## Selected Scenarios

This slice strengthens existing rows rather than adding new rows:

| ID | Classification | Why |
|---|---|---|
| VS4-GATE-001 | in_this_slice | Slice 013 contract and registration must be frozen without changing the parent matrix. |
| VS4-UI-001 | in_this_slice | Home/Ops Inbox must remain usable and contained on desktop. |
| VS4-UI-004 | in_this_slice | Brief detail must contain long evidence/action boundary text without body-level overflow. |
| VS4-UI-010 | in_this_slice | Action Card review must preserve long policy/safety-envelope detail without layout breakage. |
| VS4-UI-011 | in_this_slice | Local/mock and no-live-writeback boundary text must remain visible and contained. |
| VS4-REF-002 | in_this_slice | Claim/Action review pages must align with dry-run/approval reference direction through runtime proof, not screenshots alone. |
| VS4-REG-003 | in_this_slice | The containment fix must not introduce production, on-prem, final security, live-provider, or human UX readiness claims. |
| VS4-REG-006 | in_this_slice | The default UI must remain product-first and not become admin, connector, ontology, or verifier first. |
| VS4-REG-007 | in_this_slice | The same CLI parity evidence must remain valid; this slice cannot create a UI-only PASS. |
| VS4-H01 | human_required | JiYong/Tars Product Alpha UX acceptance remains human-only. |

## Full Scenario Classification

| ID Range | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001, VS4-UI-001, VS4-UI-004, VS4-UI-010, VS4-UI-011, VS4-REF-002, VS4-REG-003, VS4-REG-006, VS4-REG-007 | in_this_slice | Strengthened by desktop overflow containment, browser proof markers, and unchanged CLI/report boundaries. |
| All other AI-verifiable VS4 rows | previous_slice | Must remain locally `PASS`; this slice must not weaken Drop, Ask, Brief, Claim, Memory/Wiki, Action, Ops Inbox, Learn, reference, or regression evidence. |
| VS4-H01 | human_required | Human acceptance remains unclaimed until JiYong/Tars provides dated review evidence. |
| VS3-H01 through VS3-H07 | conditional_deferred | Still block production/on-prem/security/live-provider/human acceptance claims only. |

## Proof Needed

The selected slice can be considered prepared only with:

- a frozen Slice 013 contract registered in the repo README and SoT README;
- desktop browser proof showing `horizontal_overflow=false` and `document_scroll_width <= inner_width`;
- mobile browser proof still showing `horizontal_overflow=false`;
- browser proof that the Action Card denial boundary, policy reason codes, safety envelope, no-live-writeback text, and direct-provider absence remain visible;
- negative evidence counters still zero for live external writeback, provider side effects, hidden durable memory, evidence-free claim approval, production/on-prem/final-security/live-provider claims, and human UX acceptance claims;
- the full VS4 verifier still reporting 27 AI-verifiable rows `PASS`, 1 `HUMAN_REQUIRED`, and 0 blocking rows;
- targeted unit test plus docs, CLI-native-first, design-system, canonical matrix, py-compile, and `git diff --check` checks.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED`.
- Slice 013 can improve human review readiness, but it cannot create, infer, or validate JiYong/Tars acceptance.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.

## Done Criteria

This slice is done when:

1. desktop VS4 browser proof records no body-level horizontal overflow;
2. mobile VS4 browser proof still records no body-level horizontal overflow;
3. long policy reason codes and safety-envelope labels wrap inside their cards instead of expanding the page;
4. Action Card and Actions page denial evidence remains visible and AI-verifiable;
5. selected scenario rows are classified as `in_this_slice` without changing the parent VS4 matrix;
6. all existing VS4 AI-verifiable rows still pass and `VS4-H01` remains `HUMAN_REQUIRED`;
7. no report, README, UI, package, or validation output claims production, on-prem, final security, live-provider, accessibility certification, or human UX readiness.

## CLI Parity

- Verification: `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json`.
- Gate: `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json`.
- Slice target: `make verify-vs4-product-alpha-desktop-overflow`.
- CLI status: this slice improves deterministic UI containment; existing VS4 CLI/API paths remain the product behavior proof and must continue to pass.

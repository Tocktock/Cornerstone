# CornerStone VS4 Slice 011 Ops Inbox Triage Detail

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice deepens Home / Ops Inbox returning-work behavior; it does not add canonical VS4 rows or provide human UX acceptance.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Previous slices:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_005_UX_POLISH_LEARN.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_006_RESPONSIVE_MOBILE_PROOF.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_007_KEYBOARD_FOCUS_REVIEW.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_008_ASK_RESULT_READABILITY.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_009_CLAIM_ACTION_NAV_DETAIL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_010_ASK_INJECTION_BOUNDARY.md`

## Goal

Make Home / Ops Inbox feel like a usable daily-return surface, not only a static Continue list.

The user should see triage lanes for needs-review work, approval requests, policy-blocked work, and failed-with-recovery work; opening the selected item should show enough evidence, scope, risk, and next action detail to continue safely.

## Scope

In this slice:

- add Ops Inbox triage lanes and counts to the VS4 Home UI;
- add a selected-item detail rail with evidence, scope, risk, status, and next action;
- keep the default normal-user nav small: `Home`, `Search`, `Artifacts`, `Claims`, `Actions`;
- strengthen browser proof markers for triage lanes, selected detail, policy-blocked work, failed-with-recovery work, and product-language copy;
- use existing `cornerstone product mission-control --json` CLI parity for the Ops Inbox projection;
- keep the parent VS4 matrix at 20 `MUST_PASS`, 7 `REGRESSION_GUARD`, and 1 `HUMAN_REQUIRED` row.

## Non-Scope

This slice does not implement:

- new canonical VS4 scenario rows or changes to the 206-scenario matrix;
- live external work queues, production notification delivery, or provider writeback;
- final human Product Alpha UX acceptance;
- production, on-prem, final security, real IdP, real network, or live-provider proof;
- admin connector, policy, or ontology-first navigation;
- durable memory approval or automatic learning promotion.

## Assumptions

- The reference Operations Inbox image guides layout and content hierarchy only; it is not PASS evidence.
- `cornerstone product mission-control --json` is the existing CLI-native product surface for the local Ops Inbox projection.
- Triage rows can be deterministic local fixtures for this Product Alpha proof.
- Failed or blocked work must remain recoverable review work and must not imply live external execution.
- `VS4-H01` remains the only path for subjective Product Alpha UX acceptance.

## Selected Scenarios

This slice strengthens existing rows rather than adding new ones:

| ID | Classification | Why |
|---|---|---|
| VS4-GATE-001 | in_this_slice | Slice 011 contract and registration must be frozen without changing the parent matrix. |
| VS4-UI-001 | in_this_slice | Home must keep Drop / Ask / Continue first while adding daily-return triage. |
| VS4-UI-012 | in_this_slice | Ops Inbox must show pending work, evidence gaps, approvals, policy-blocked work, failed work, and Continue/detail paths. |
| VS4-UI-015 | in_this_slice | Selected work detail must keep active workspace and owner context visible. |
| VS4-UI-016 | in_this_slice | Normal-user inbox copy must stay product-first and not expose internal verifier jargon by default. |
| VS4-STATE-001 | in_this_slice | Home / Ops Inbox must expose needs-review, policy-blocked, failed-with-recovery, and audit/log states as observable UI states. |
| VS4-REF-001 | in_this_slice | Home/Ops Inbox should align with reference direction through runtime UI evidence, not reference-image existence. |
| VS4-REG-003 | in_this_slice | New inbox detail must not claim production, on-prem, final security, live-provider, or human UX readiness. |
| VS4-REG-006 | in_this_slice | The default user experience must remain product-first, not admin, connector, ontology, or verifier first. |
| VS4-REG-007 | in_this_slice | Ops Inbox projection must have native CLI JSON evidence through `cornerstone product mission-control --json`. |
| VS4-H01 | human_required | JiYong/Tars Product Alpha UX acceptance remains human-only. |

## Full Scenario Classification

| ID Range | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001, VS4-UI-001, VS4-UI-012, VS4-UI-015, VS4-UI-016, VS4-STATE-001, VS4-REF-001, VS4-REG-003, VS4-REG-006, VS4-REG-007 | in_this_slice | Strengthened by Ops Inbox triage/detail UI, browser proof markers, CLI product mission-control transcript, and overclaim/product-first guards. |
| VS4-UI-002 through VS4-UI-011, VS4-UI-013 through VS4-UI-014, VS4-REF-002, VS4-REG-001 through VS4-REG-002, VS4-REG-004 through VS4-REG-005 | previous_slice | Must remain locally `PASS`; this slice must not weaken source, Brief, Claim, Memory/Wiki, Action, Ask, pack, security, or regression evidence. |
| VS4-H01 | human_required | Human acceptance remains unclaimed until JiYong/Tars provides dated review evidence. |
| VS3-H01 through VS3-H07 | conditional_deferred | Still block production/on-prem/security/live-provider/human acceptance claims only. |

## Proof Needed

The selected slice can be considered prepared only with:

- a frozen Slice 011 contract registered in the repo README and SoT README;
- browser and mobile DOM proof that Home / Ops Inbox includes lanes for needs review, approval requests, policy blocked, and failed with recovery;
- browser proof that a selected item detail shows source/evidence refs, workspace/owner scope, risk/status, next action, and audit/activity detail;
- proof that normal user nav remains `Home`, `Search`, `Artifacts`, `Claims`, `Actions`;
- proof that reference images are not used as scenario PASS evidence;
- CLI transcript evidence from `cornerstone product mission-control --json` with an Ops Inbox projection, sections, evidence refs, audit refs, and workspace scope;
- negative evidence counters remaining zero for production, on-prem, final security, live-provider, human UX, reference-image PASS, and live external writeback claims;
- the full VS4 verifier still reporting 27 AI-verifiable rows `PASS`, 1 `HUMAN_REQUIRED`, and 0 blocking rows;
- targeted unit tests plus docs, CLI-native-first, design-system, canonical matrix, py-compile, and `git diff --check` checks.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED`.
- Slice 011 can improve human review readiness, but it cannot create, infer, or validate JiYong/Tars acceptance.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.

## Done Criteria

This slice is done when:

1. Home / Ops Inbox exposes deterministic triage lanes and selected-item detail for returning work;
2. needs-review, approval request, policy-blocked, and failed-with-recovery work are visible without admin-first navigation;
3. selected detail includes evidence, scope, risk/status, next action, and audit/activity proof;
4. `cornerstone product mission-control --json` remains the CLI parity path for the inbox projection;
5. selected scenario rows are classified as `in_this_slice` without changing the parent VS4 matrix;
6. all existing VS4 AI-verifiable rows still pass and `VS4-H01` remains `HUMAN_REQUIRED`;
7. no report, README, UI, package, or validation output claims production, on-prem, final security, live-provider, accessibility certification, or human UX readiness.

## CLI Parity

- Ops Inbox projection: `cornerstone product mission-control --json`.
- Verification: `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json`.
- Gate: `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json`.
- Slice target: `make verify-vs4-product-alpha-ops-inbox-triage`.

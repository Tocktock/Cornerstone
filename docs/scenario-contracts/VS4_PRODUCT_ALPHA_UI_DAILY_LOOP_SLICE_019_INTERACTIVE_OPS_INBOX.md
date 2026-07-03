# CornerStone VS4 Product Alpha UI Daily Loop Slice 019 Interactive Ops Inbox Contract

**Date:** 2026-07-04 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`

## Goal

Turn the VS4 Ops Inbox from a static returning-work projection into a selectable daily-work surface.

A returning user must be able to choose a lane or work item, see the selected work detail update, keep evidence/audit/scope/risk visible, and continue into the existing Brief, Claim, Memory/Wiki, Action, Evidence/Audit, or Learn surfaces without creating hidden authority or live writeback.

## Scope

- Add deterministic lane and row selection to the local Product Alpha Ops Inbox UI.
- Prove click and keyboard selection across needs-review, approval-request, policy-blocked, and failed-with-recovery work.
- Update selected detail from the selected row: title, kind, lane, status, owner/workspace, evidence refs, audit refs, risk, next action, and Continue target.
- Add a visible daily journey cue: `Inbox -> Brief -> Claim -> Memory/Wiki -> Action -> Learn`.
- Add CLI parity for selecting the same work via `cornerstone product mission-control --lane ... --selected-item ... --json`.
- Use existing `cornerstone product loop-view ... --json` to prove the selected item is bound to the daily loop.
- Add browser and mobile proof markers that exercise real interaction, not only static DOM presence.
- Add a Slice 019 Make target that refreshes the VS4 scenario report and gate output.

## Non-Scope

- No new canonical VS4 scenario rows.
- No change to the canonical 206-scenario matrix.
- No live provider, external writeback, production, on-prem, final security, or human UX readiness claim.
- No approval shortcut from inbox selection.
- No hidden durable memory write or automatic learning promotion.
- No broad backend storage migration for Brief, Inbox, or Memory/Wiki.

## Assumptions

- The existing Slice 011 Ops Inbox triage/detail behavior remains the base surface.
- Interactive selection can be deterministic local Product Alpha proof using fixture work items.
- `cornerstone product mission-control` is the CLI-native Ops Inbox projection.
- `cornerstone product loop-view` is the CLI-native path for showing the selected work item's daily loop orientation.
- Reference image `cornerstone-reference-02-operations-inbox.png` guides layout only and is not scenario PASS evidence.
- `VS4-H01` remains human-only Product Alpha UX acceptance.

## Scenario Inventory

| Scenario ID | Priority | Slice classification | Gate meaning |
|---|---|---|---|
| `VS4-GATE-001` | MUST_PASS | `in_this_slice` | Slice 019 contract, registration, verifier wiring, Make target, report, and gate output are structurally frozen and verified. |
| `VS4-UI-001` | MUST_PASS | `in_this_slice` | Home remains Drop / Ask / Continue first while Ops Inbox becomes selectable. |
| `VS4-UI-012` | MUST_PASS | `in_this_slice` | Returning work supports lane filtering, selected detail updates, Continue targets, and daily loop continuity. |
| `VS4-UI-013` | MUST_PASS | `in_this_slice` | Ask-created work remains part of the same durable loop and can be continued through the inbox/journey surface. |
| `VS4-UI-015` | MUST_PASS | `in_this_slice` | Selected work preserves visible owner/workspace scope after interaction. |
| `VS4-UI-016` | MUST_PASS | `in_this_slice` | Selected detail keeps product language before internal refs and progressively discloses evidence/audit detail. |
| `VS4-STATE-001` | MUST_PASS | `in_this_slice` | Needs-review, approval-request, policy-blocked, failed-with-recovery, and audit/log states remain observable after selection. |
| `VS4-REF-001` | MUST_PASS | `in_this_slice` | Home/Ops Inbox follows the reference direction through runtime UI behavior, not image existence. |
| `VS4-REG-003` | REGRESSION_GUARD | `in_this_slice` | Selection does not introduce production, on-prem, final security, live-provider, or human UX readiness claims. |
| `VS4-REG-006` | REGRESSION_GUARD | `in_this_slice` | The default surface remains product-first, not admin, connector, ontology, or verifier first. |
| `VS4-REG-007` | REGRESSION_GUARD | `in_this_slice` | Ops Inbox selection has native CLI JSON transcript evidence and missing CLI parity still blocks PASS. |
| `VS4-H01` | HUMAN_REQUIRED | `human_required` | JiYong/Tars Product Alpha UX acceptance remains unclaimed. |

All other AI-verifiable VS4 rows remain `previous_slice` in the active verifier ledger. `VS3-H01` through `VS3-H07` remain conditional deferred blockers only for production/on-prem/security/live-provider/human-acceptance claims.

## Required Verification

- `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-001 --scenario VS4-UI-012 --scenario VS4-UI-013 --scenario VS4-UI-015 --scenario VS4-UI-016 --scenario VS4-STATE-001 --scenario VS4-REF-001 --scenario VS4-REG-003 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json`
- CLI smoke:
  - `cornerstone product mission-control --lane policy-blocked --selected-item evidence-free-approval-attempt --json`
  - `cornerstone product mission-control --lane approval-requests --selected-item action-card-draft --json`
  - `cornerstone product loop-view --brief-id ... --claim-id ... --mission-id ... --action-id ... --json`
- Focused unittest asserting desktop/mobile interaction markers, CLI selected-item parity, loop-view parity, zero inbox-selection side effects, and the Slice 019 proof boundary.
- Existing documentation checks: SoT, CLI-native-first, design-system, canonical scenario matrix, and `git diff --check`.

## Done Means

- Lane selection changes `aria-selected`, filters visible rows, and selects a matching row.
- Row click and keyboard selection update selected detail from row data.
- Selected detail shows title, kind, lane/status, owner/workspace, evidence refs, audit refs, risk, next action, Continue target, and the daily journey cue.
- Continue target matches the selected row and resolves to an existing local product surface.
- CLI selection returns the same selected item id, lane, evidence refs, audit refs, scope, and Continue target.
- Loop-view CLI transcript shows the selected item belongs to `Inbox -> Brief -> Claim -> Action -> Learn`.
- Zero inbox selection side effects: no hidden memory write, no claim approval, no action execution, no provider call, no live external writeback, no policy change, and no authority expansion.
- `VS4-H01` remains `HUMAN_REQUIRED`.
- No production/on-prem/security/live-provider/human UX readiness claim is introduced.

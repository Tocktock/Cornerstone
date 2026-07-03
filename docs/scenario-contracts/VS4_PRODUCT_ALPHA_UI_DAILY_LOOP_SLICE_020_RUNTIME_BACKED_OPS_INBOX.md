# CornerStone VS4 Product Alpha UI Daily Loop Slice 020 Runtime-backed Ops Inbox Contract

**Date:** 2026-07-04 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`

## Goal

Turn the VS4 Ops Inbox from an interactive fixture projection into a runtime-backed daily-work projection.

After a local Drop / Ask flow prepares work, the Home / Ops Inbox must refresh from the same local runtime records used by the native CLI: Brief, Claim, Memory/Wiki candidate, Action Card, policy-blocked claim, selected work detail, evidence refs, activity refs, and loop-view refs. The result must remain Product Alpha local proof only.

## Scope

- Add local Product Alpha API parity for `product mission-control` and `product loop-view`.
- Create a real draft Memory/Wiki candidate during the browser Drop flow instead of leaving the browser with only pseudo memory state.
- Refresh the visible Ops Inbox rows and selected detail from runtime Mission Control after the browser creates Brief, Claim, Memory, and Action records.
- Preserve lane and row selection behavior from Slice 019 after runtime refresh.
- Prove the selected runtime rows expose real record refs such as `brief:<id>`, `claim:<id>`, `memory:<id>`, and `action:<id>`.
- Prove runtime selected detail keeps evidence refs, activity/audit refs, owner/workspace scope, Continue targets, and `Inbox -> Brief -> Claim -> Memory/Wiki -> Action -> Learn`.
- Add a Slice 020 Make target that refreshes the VS4 scenario report and gate output.

## Non-Scope

- No new canonical VS4 scenario rows.
- No change to the canonical 206-scenario matrix.
- No production, on-prem, final security, live-provider, or human UX readiness claim.
- No live provider writeback, real external HTTP call, or provider mutation.
- No approved memory write from the browser Drop/Ask flow; the Memory/Wiki object remains `draft`.
- No approval shortcut from runtime Inbox refresh.
- No broad backend storage migration.

## Assumptions

- `cornerstone product mission-control --json` is the native CLI source for the runtime Ops Inbox projection.
- `cornerstone product loop-view --json` is the native CLI source for the selected work item's daily-loop orientation.
- Local Product Alpha API routes may expose those same runtime projections for browser proof.
- Reference image `cornerstone-reference-02-operations-inbox.png` guides selected-row/detail layout only and is not scenario PASS evidence.
- `VS4-H01` remains human-only Product Alpha UX acceptance.

## Scenario Inventory

| Scenario ID | Priority | Slice classification | Gate meaning |
|---|---|---|---|
| `VS4-GATE-001` | MUST_PASS | `in_this_slice` | Slice 020 contract, registration, API/verifier wiring, Make target, report, and gate output are structurally frozen and verified. |
| `VS4-UI-001` | MUST_PASS | `in_this_slice` | Home remains Drop / Ask / Continue first while the Inbox refreshes from runtime records. |
| `VS4-UI-008` | MUST_PASS | `in_this_slice` | Memory/Wiki candidate is a real draft runtime object with source/evidence refs, not a hidden approved memory. |
| `VS4-UI-009` | MUST_PASS | `in_this_slice` | Runtime-backed Memory/Wiki remains draft and cannot influence answers/actions before review. |
| `VS4-UI-012` | MUST_PASS | `in_this_slice` | Returning work uses runtime Mission Control records and selected detail, not only static rows. |
| `VS4-UI-013` | MUST_PASS | `in_this_slice` | Ask-created work remains on the same durable loop and can be represented through runtime Mission Control / loop-view parity. |
| `VS4-UI-015` | MUST_PASS | `in_this_slice` | Runtime selected detail preserves visible owner/workspace scope. |
| `VS4-UI-016` | MUST_PASS | `in_this_slice` | Runtime-backed Inbox keeps product labels before internal refs and progressively discloses evidence/audit detail. |
| `VS4-STATE-001` | MUST_PASS | `in_this_slice` | Needs-review, approval-request, policy-blocked, failed-with-recovery, and audit/log states remain observable after runtime refresh. |
| `VS4-REF-001` | MUST_PASS | `in_this_slice` | Home/Ops Inbox follows reference direction through runtime UI behavior, not image existence. |
| `VS4-REG-003` | REGRESSION_GUARD | `in_this_slice` | Runtime projection does not introduce production, on-prem, final security, live-provider, or human UX readiness claims. |
| `VS4-REG-004` | REGRESSION_GUARD | `in_this_slice` | Runtime refresh cannot approve claims, approve memory, execute actions, change policy, call providers, or expand authority. |
| `VS4-REG-006` | REGRESSION_GUARD | `in_this_slice` | The default surface remains product-first, not admin, connector, ontology, or verifier first. |
| `VS4-REG-007` | REGRESSION_GUARD | `in_this_slice` | Runtime Inbox refresh has native CLI JSON transcript evidence and missing CLI parity still blocks PASS. |
| `VS4-H01` | HUMAN_REQUIRED | `human_required` | JiYong/Tars Product Alpha UX acceptance remains unclaimed. |

All other AI-verifiable VS4 rows remain `previous_slice` in the active verifier ledger. `VS3-H01` through `VS3-H07` remain conditional deferred blockers only for production/on-prem/security/live-provider/human-acceptance claims.

## Required Verification

- `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-001 --scenario VS4-UI-008 --scenario VS4-UI-009 --scenario VS4-UI-012 --scenario VS4-UI-013 --scenario VS4-UI-015 --scenario VS4-UI-016 --scenario VS4-STATE-001 --scenario VS4-REF-001 --scenario VS4-REG-003 --scenario VS4-REG-004 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json`
- Browser proof showing runtime-backed Ops Inbox markers on desktop and mobile.
- CLI smoke:
  - `cornerstone product mission-control --json`
  - `cornerstone product mission-control --lane needs-review --selected-item memory-candidate --json`
  - `cornerstone product mission-control --lane approval-requests --selected-item action-card-draft --json`
  - `cornerstone product loop-view --brief-id ... --claim-id ... --memory-id ... --mission-id ... --action-id ... --json`
- Focused unittest asserting active Slice 020 metadata, runtime-backed browser markers, CLI selected-item parity, draft memory boundary, loop-view parity, and zero runtime-refresh side effects.
- Existing documentation checks: SoT, CLI-native-first, design-system, canonical scenario matrix, and `git diff --check`.

## Done Means

- Browser Drop flow creates a real draft Memory/Wiki candidate with `status=draft`, `trust_state=draft`, `owner_approved=false`, `can_influence_answers=false`, and `can_influence_actions=false`.
- Product API exposes Mission Control and Loop View from `LocalRuntimeStore`.
- Browser Ops Inbox rows and selected detail refresh from runtime Mission Control after local work creation.
- Runtime-backed rows include real `brief:`, `claim:`, `memory:`, and `action:` refs where records exist.
- Runtime selected detail keeps evidence refs, activity/audit refs, owner/workspace scope, Continue target, risk, next action, and daily journey.
- CLI Mission Control selected item returns the same runtime-backed ids and boundaries.
- Loop View CLI transcript shows the selected work belongs to `Inbox -> Brief -> Claim -> Memory/Wiki -> Action -> Learn`.
- Zero runtime-refresh side effects: no hidden memory write, no approved memory before review, no claim approval, no action execution, no provider call, no live external writeback, no policy change, and no authority expansion.
- `VS4-H01` remains `HUMAN_REQUIRED`.
- No production/on-prem/security/live-provider/human UX readiness claim is introduced.

# CornerStone VS4 Product Alpha UI Daily Loop Slice 021 Runtime Loop Coherence Contract

**Date:** 2026-07-04 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`

## Goal

Make the full Product Alpha daily loop coherent at the runtime, browser, CLI, and gate levels.

Slice 020 made Ops Inbox rows runtime-backed for Brief, Claim, Memory/Wiki, and Action. Slice 021 closes the remaining product-loop gap by turning Learn into a review-only runtime candidate with native `trajectory`, `lesson`, evidence, and audit refs, by removing contradictory Action approval copy, and by binding active browser proof paths to the active slice. The result remains local Product Alpha evidence only.

## Scope

- Add local Product Alpha API parity for creating an experience trajectory and lesson candidate from the browser flow.
- Create a runtime-backed Learn candidate after the local Action Card review path, with `trajectory:`, `learn:`, `lesson:`, evidence refs, audit refs, owner/workspace scope, and review-only boundary.
- Refresh Ops Inbox failed/recovery Learn row from runtime lesson records instead of static `learning:local_vs4_recovery` fallback when a lesson exists.
- Extend `product loop-view` to carry a Learn-stage lesson ref.
- Normalize Action Card approval copy so Product Alpha never shows `Approval not_required` beside denied-until-authorized execution copy.
- Bind desktop and mobile VS4 browser proof defaults to Slice 021 output directories.
- Add gate negative evidence for stale proof paths, contradictory Action approval copy, static Learn fallback, missing Learn refs, missing Learn evidence/audit refs, and hidden durable Learn authority.
- Add a Slice 021 Make target and focused tests.

## Non-Scope

- No new canonical VS4 scenario rows.
- No change to the canonical 206-scenario matrix.
- No production, on-prem, final security, live-provider, or human UX readiness claim.
- No live provider writeback, real external HTTP call, provider mutation, or production connector path.
- No approved memory or durable behavior change from Learn. Lesson candidates remain review-only.
- No final backend storage model decision beyond using existing local runtime records.

## Assumptions

- Existing `experience trajectory record` and `experience lesson propose` CLI commands are the native CLI parity surface for Learn candidates.
- Product Alpha API routes may expose those same local runtime operations for browser proof.
- A Learn candidate can be represented as `learn:<lesson_id>` plus native `lesson:<lesson_id>` and `trajectory:<trajectory_id>` refs.
- Reference image `cornerstone-reference-02-operations-inbox.png` guides failed/recovery and detail layout only and is not scenario PASS evidence.
- `VS4-H01` remains human-only Product Alpha UX acceptance.

## Scenario Inventory

| Scenario ID | Priority | Slice classification | Gate meaning |
|---|---|---|---|
| `VS4-GATE-001` | MUST_PASS | `in_this_slice` | Slice 021 contract, registration, active proof paths, verifier wiring, Make target, report, and gate output are structurally frozen and verified. |
| `VS4-UI-001` | MUST_PASS | `in_this_slice` | Home keeps Drop / Ask / Continue first while the active proof and runtime loop stay product-first. |
| `VS4-UI-008` | MUST_PASS | `in_this_slice` | Memory/Wiki remains draft while Learn creates only a review-only lesson candidate. |
| `VS4-UI-009` | MUST_PASS | `in_this_slice` | Learn candidate cannot become hidden durable memory, approved memory, answer authority, or action authority before review. |
| `VS4-UI-010` | MUST_PASS | `in_this_slice` | Action Card copy and state are coherent: review is required before execution, with goal, evidence, risk, policy, approval, and local/mock mode visible. |
| `VS4-UI-011` | MUST_PASS | `in_this_slice` | Action execution mode remains Draft / Local / Mock and denial-safe; no live writeback is implied. |
| `VS4-UI-012` | MUST_PASS | `in_this_slice` | Ops Inbox failed/recovery lane shows a runtime-backed Learn candidate with record refs, evidence refs, audit refs, scope, and Continue link. |
| `VS4-UI-013` | MUST_PASS | `in_this_slice` | Ask-created work can progress through runtime Loop View through Learn without becoming chatbot-only output. |
| `VS4-UI-016` | MUST_PASS | `in_this_slice` | Learn and Action copy use product language before internal refs and disclose raw ids progressively. |
| `VS4-STATE-001` | MUST_PASS | `in_this_slice` | Failed-with-recovery and needs-review states are observable for Learn without implying durable behavior change. |
| `VS4-REF-002` | MUST_PASS | `in_this_slice` | Claim/Action reference alignment includes denial-safe Action approval copy and Learn-linked audit/detail refs. |
| `VS4-REG-003` | REGRESSION_GUARD | `in_this_slice` | Runtime Learn and active proof updates do not introduce production, on-prem, final security, live-provider, or human UX readiness claims. |
| `VS4-REG-004` | REGRESSION_GUARD | `in_this_slice` | Prompt injection and untrusted content cannot approve memory/action, promote Learn, call providers, execute actions, or expand authority. |
| `VS4-REG-005` | REGRESSION_GUARD | `in_this_slice` | Reference images remain visual guidance only; active proof paths and runtime evidence prove behavior. |
| `VS4-REG-006` | REGRESSION_GUARD | `in_this_slice` | The default surface remains product-first, not admin, connector, ontology, or verifier first. |
| `VS4-REG-007` | REGRESSION_GUARD | `in_this_slice` | Learn and loop-view updates have native CLI JSON transcript evidence; missing parity blocks PASS. |
| `VS4-H01` | HUMAN_REQUIRED | `human_required` | JiYong/Tars Product Alpha UX acceptance remains unclaimed. |

All other AI-verifiable VS4 rows remain `previous_slice` in the active verifier ledger. `VS3-H01` through `VS3-H07` remain conditional deferred blockers only for production/on-prem/security/live-provider/human-acceptance claims.

## Required Verification

- `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-001 --scenario VS4-UI-008 --scenario VS4-UI-009 --scenario VS4-UI-010 --scenario VS4-UI-011 --scenario VS4-UI-012 --scenario VS4-UI-013 --scenario VS4-UI-016 --scenario VS4-STATE-001 --scenario VS4-REF-002 --scenario VS4-REG-003 --scenario VS4-REG-004 --scenario VS4-REG-005 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json`
- Browser proof showing active Slice 021 desktop and mobile paths, runtime-backed Learn markers, coherent Action approval copy, and zero live/provider side effects.
- CLI smoke:
  - `cornerstone experience trajectory record --json`
  - `cornerstone experience lesson propose --json`
  - `cornerstone product mission-control --lane failed-recovery --selected-item learning-recovery-candidate --json`
  - `cornerstone product loop-view --brief-id ... --claim-id ... --memory-id ... --mission-id ... --action-id ... --lesson-id ... --json`
- Focused unittest asserting Slice 021 metadata, active proof paths, runtime-backed Learn refs, action copy coherence, CLI Learn parity, loop-view Learn ref, zero negative evidence, and human/deferred boundaries.
- Existing documentation checks: SoT, CLI-native-first, design-system, canonical scenario matrix, and `git diff --check`.

## Done Means

- The browser Drop / Ask flow creates a runtime trajectory and lesson candidate after local Action review.
- Learn candidate state contains `learn:<lesson_id>`, `lesson:<lesson_id>`, `trajectory:<trajectory_id>`, evidence refs, audit refs, owner/workspace scope, review state, and `can_change_* = false`.
- Ops Inbox failed/recovery Learn row is runtime-backed when a lesson exists.
- Product Loop View exposes Learn as `learn:<lesson_id>` and native `lesson:<lesson_id>` detail when a lesson exists.
- Action Card and Action detail do not render `Approval not_required`; they show review required before execution unless a real authorized approval exists.
- Desktop and mobile active proof directories include `slice-021-runtime-loop-coherence`.
- Gate fails if active proof paths are stale, Action approval copy is contradictory, Learn falls back to static-only proof, Learn refs/evidence/audit are missing, or Learn implies hidden durable authority.
- `VS4-H01` remains `HUMAN_REQUIRED`.
- No production/on-prem/security/live-provider/human UX readiness claim is introduced.

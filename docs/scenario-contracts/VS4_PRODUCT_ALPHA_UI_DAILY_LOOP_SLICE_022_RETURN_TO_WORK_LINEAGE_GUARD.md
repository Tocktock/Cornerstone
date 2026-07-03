# CornerStone VS4 Product Alpha UI Daily Loop Slice 022 Return-to-Work Lineage Guard Contract

**Date:** 2026-07-04 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`

## Goal

Make the Product Alpha return-to-work loop safe against forged, missing, cross-scope, or mismatched references.

Slice 021 made the happy path coherent through runtime-backed Learn and active proof binding. Slice 022 closes the next trust gap: `product loop-view` must not render an apparently coherent `Inbox -> Brief -> Claim -> Memory/Wiki -> Action -> Learn` journey from arbitrary IDs. The CLI and API must validate that referenced records exist, belong to the requested owner/workspace scope, and share the same evidence/mission lineage before showing a loop.

## Scope

- Validate `product loop-view` refs for Brief, Claim, Memory/Wiki, Mission, Action, and Learn lesson.
- Reject missing refs with deterministic `not_found` errors.
- Reject cross-scope refs with deterministic `scope_denied` errors.
- Reject mismatched lineage when:
  - Claim evidence does not match the Brief evidence bundle.
  - Memory source does not match the Brief/Claim evidence bundle.
  - Mission source claim or evidence does not match the Claim/Brief evidence.
  - Action mission/claim/evidence does not match the Mission/Claim/Brief evidence.
  - Lesson trajectory does not match the Mission and evidence lineage.
- Return product-language failure details that say the loop could not be shown because the work items do not belong to the same evidence-backed journey.
- Add CLI/API negative proof for missing, cross-scope, and mismatched refs.
- Keep the existing happy path and Slice 021 browser proof behavior intact.

## Non-Scope

- No new canonical VS4 scenario rows.
- No change to the canonical 206-scenario matrix.
- No production, on-prem, final security, live-provider, or final human UX readiness claim.
- No live provider writeback, external HTTP call, provider mutation, or production connector path.
- No memory approval, lesson promotion, or durable behavior change.
- No broad UI redesign; browser happy path remains the current Product Alpha shell.

## Assumptions

- The existing local runtime records carry enough lineage for this slice: evidence bundle id, artifact refs, source claim id, mission id, trajectory id, owner, namespace, workspace, and tenant.
- A loop with only one or two refs may render if each provided ref exists and is in scope; lineage checks apply only across refs that are provided.
- `product loop-view` is the native CLI parity surface for return-to-work journey validation.
- `/product/loop-view` must enforce the same runtime validation as the CLI.
- Reference images remain visual guidance only and are not PASS evidence.

## Scenario Inventory

| Scenario ID | Priority | Slice classification | Gate meaning |
|---|---|---|---|
| `VS4-GATE-001` | MUST_PASS | `in_this_slice` | Slice 022 contract, registration, verifier wiring, Make target, report, and gate output are structurally frozen and verified. |
| `VS4-UI-001` | MUST_PASS | `previous_slice` | Home remains Product Alpha first value from earlier slices. |
| `VS4-UI-002` | MUST_PASS | `previous_slice` | Drop/Paste source preservation remains covered by prior slices. |
| `VS4-UI-003` | MUST_PASS | `previous_slice` | Evidence-backed Brief creation remains covered by prior slices. |
| `VS4-UI-004` | MUST_PASS | `previous_slice` | Brief detail contents remain covered by prior slices. |
| `VS4-UI-005` | MUST_PASS | `previous_slice` | Evidence Drawer remains covered by prior slices. |
| `VS4-UI-006` | MUST_PASS | `previous_slice` | Claim candidate creation remains covered by prior slices. |
| `VS4-UI-007` | MUST_PASS | `previous_slice` | Evidence-free approval denial remains covered by prior slices. |
| `VS4-UI-008` | MUST_PASS | `in_this_slice` | Memory/Wiki candidate lineage must match the same evidence-backed journey before Loop View renders it as part of return-to-work. |
| `VS4-UI-009` | MUST_PASS | `in_this_slice` | Invalid loop refs cannot create hidden memory, approved memory, lesson promotion, or authority changes. |
| `VS4-UI-010` | MUST_PASS | `in_this_slice` | Action Card refs must match the same Mission/Claim/Brief evidence lineage before Loop View renders them. |
| `VS4-UI-011` | MUST_PASS | `in_this_slice` | Loop validation cannot imply live writeback or local/mock execution success. |
| `VS4-UI-012` | MUST_PASS | `in_this_slice` | Returning Ops Inbox work must continue through validated same-scope, same-lineage refs. |
| `VS4-UI-013` | MUST_PASS | `in_this_slice` | Ask-created work can become a loop only when refs validate against real records and shared evidence lineage. |
| `VS4-UI-014` | MUST_PASS | `previous_slice` | Three general-purpose packs remain covered by prior slices. |
| `VS4-UI-015` | MUST_PASS | `in_this_slice` | Workspace/owner scope must be enforced on every Loop View ref. |
| `VS4-UI-016` | MUST_PASS | `in_this_slice` | Invalid-loop copy must use product language before internal error details. |
| `VS4-STATE-001` | MUST_PASS | `in_this_slice` | Missing, permission denied, policy/lineage blocked, and failed-with-recovery loop states are observable through CLI/API proof. |
| `VS4-REF-001` | MUST_PASS | `previous_slice` | Home/Search/Artifact reference alignment remains covered by prior slices. |
| `VS4-REF-002` | MUST_PASS | `in_this_slice` | Claim/Action reference alignment now includes negative lineage validation, not only happy-path refs. |
| `VS4-REG-001` | REGRESSION_GUARD | `previous_slice` | VS0 regression remains covered by prior slices and is not rerun in this narrow slice. |
| `VS4-REG-002` | REGRESSION_GUARD | `previous_slice` | VS1 regression remains covered by prior slices and is not rerun in this narrow slice. |
| `VS4-REG-003` | REGRESSION_GUARD | `in_this_slice` | Loop validation introduces no production, on-prem, final security, live-provider, or human UX readiness claims. |
| `VS4-REG-004` | REGRESSION_GUARD | `in_this_slice` | Forged or untrusted refs cannot approve memory/action, execute actions, promote lessons, call providers, or expand authority. |
| `VS4-REG-005` | REGRESSION_GUARD | `in_this_slice` | Reference images remain visual guidance only; loop validation proof comes from CLI/API/report evidence. |
| `VS4-REG-006` | REGRESSION_GUARD | `in_this_slice` | The default surface remains product-first and does not become verifier/admin/ontology-first. |
| `VS4-REG-007` | REGRESSION_GUARD | `in_this_slice` | Missing CLI/API loop validation parity blocks PASS. |
| `VS4-H01` | HUMAN_REQUIRED | `human_required` | JiYong/Tars Product Alpha UX acceptance remains unclaimed. |

`VS3-H01` through `VS3-H07` remain `conditional_deferred` blockers only for production/on-prem/security/live-provider/human-acceptance claims. They do not block local VS4 Product Alpha Slice 022 work.

## Required Verification

- `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-008 --scenario VS4-UI-009 --scenario VS4-UI-010 --scenario VS4-UI-011 --scenario VS4-UI-012 --scenario VS4-UI-013 --scenario VS4-UI-015 --scenario VS4-UI-016 --scenario VS4-STATE-001 --scenario VS4-REF-002 --scenario VS4-REG-003 --scenario VS4-REG-004 --scenario VS4-REG-005 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-slice-022-return-to-work-lineage.json`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-slice-022-return-to-work-lineage.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-slice-022-return-to-work-lineage-gate.json`
- CLI transcripts for:
  - valid same-lineage `cornerstone product loop-view --json`;
  - missing Brief/Claim/Memory/Mission/Action/Lesson ref rejection;
  - cross-scope ref rejection;
  - mismatched Claim/Memory/Action/Lesson lineage rejection.
- API proof for `/product/loop-view` enforcing the same valid and invalid boundaries.
- Focused unittest for Slice 022 metadata, positive loop-view parity, negative validation counters, product-language errors, zero authority side effects, and human/deferred boundaries.
- Existing documentation checks: SoT, CLI-native-first, design-system, canonical scenario matrix, and `git diff --check`.

## Done Means

- `product loop-view` still renders the valid Product Alpha journey when provided real same-scope, same-lineage refs.
- `product loop-view` returns non-zero CLI exit and structured JSON errors for missing, cross-scope, or mismatched refs.
- `/product/loop-view` returns matching HTTP error statuses and structured JSON errors for the same invalid cases.
- Invalid refs do not create product loop records, memory approvals, lesson promotions, action executions, provider calls, external HTTP calls, policy changes, or authority expansion.
- Scenario report records zero negative counters for invalid-loop acceptance, cross-scope acceptance, missing-ref acceptance, authority side effects, live writeback, and overclaims.
- `VS4-H01` remains `HUMAN_REQUIRED`.
- No production/on-prem/final-security/live-provider/human UX readiness claim is introduced.

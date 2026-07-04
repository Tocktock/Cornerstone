# CornerStone VS4 Product Alpha UI Daily Loop Slice 025 Ops Inbox Journey Timeline Contract

**Date:** 2026-07-04 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`

## Goal

Turn the Ops Inbox selected-work detail from a flat journey label into a runtime-backed journey timeline.

When a user returns to work from the Product Alpha Home / Ops Inbox, the selected item must show the daily loop as staged, product-readable progress:

```text
Inbox -> Brief -> Claim -> Memory/Wiki -> Action -> Learn
```

Each stage must be backed by the same `product loop-view` data used by CLI/API verification, show product language first, retain refs for progressive Evidence/Audit detail, and surface safe recovery states for missing, cross-scope, or lineage-mismatched refs without creating new authority.

## Scope

- Add a visible Journey Timeline inside the Ops Inbox selected-work detail.
- Populate timeline stages from existing `/product/loop-view` / `cornerstone product loop-view --json` `product_loop.stages` data.
- Keep the existing selected item detail, lane filtering, keyboard selection, runtime-backed Ops Inbox, and loop-lineage guards intact.
- Show stage label, status, product-readable description, primary ref, evidence/audit refs, and next review cue where available.
- Add product-language recovery states for missing ref, cross-scope ref, and lineage mismatch loop-view failures.
- Require desktop and mobile browser proof markers for timeline visibility, stage count, refs, progressive evidence/audit detail, recovery copy, and no authority expansion.
- Add a focused `verify-vs4-product-alpha-ops-inbox-journey-timeline` target with its own report and gate paths.
- Make Slice 025 the active VS4 implementation slice while preserving the canonical full VS4 report path used by `VS4-H01`.

## Non-Scope

- No new canonical VS4 matrix rows.
- No change to the canonical 206-scenario matrix.
- No broad UI redesign, new top-level navigation, admin-first surface, ontology-first default, or connector setup default.
- No new backend storage model, new durable memory approval path, lesson promotion, live provider writeback, or production connector path.
- No production, on-prem, final security, live-provider, accessibility certification, or human UX readiness claim.
- No attempt to collect or simulate JiYong/Tars human acceptance.
- Reference images guide layout only; they are not PASS or human-acceptance evidence.

## Assumptions

- Slice 022 already validates `product loop-view` missing, cross-scope, and lineage-mismatch refs.
- Slice 024 already requires active report/package metadata coherence; Slice 025 must update that active metadata instead of bypassing it.
- The existing runtime selected-detail and loop-view payloads carry enough stage refs, evidence refs, audit refs, owner/workspace scope, and status values for a UI timeline.
- The reference image pattern for Operations Inbox supports a table/list plus right-detail panel; Journey Timeline belongs in selected detail, not normal nav.

## Scenario Inventory

| Scenario ID | Priority | Slice classification | Gate meaning |
|---|---|---|---|
| `VS4-GATE-001` | MUST_PASS | `in_this_slice` | Slice 025 contract, active metadata, focused target paths, report, and gate output are structurally frozen and verified. |
| `VS4-UI-001` | MUST_PASS | `in_this_slice` | Home / Ops Inbox remains the product-first entry point and now exposes the selected-work timeline without expanding default nav. |
| `VS4-UI-002` | MUST_PASS | `previous_slice` | Drop/Paste source preservation remains covered by prior slices. |
| `VS4-UI-003` | MUST_PASS | `previous_slice` | Evidence-backed Brief creation remains covered by prior slices. |
| `VS4-UI-004` | MUST_PASS | `previous_slice` | Brief contents remain covered by prior slices. |
| `VS4-UI-005` | MUST_PASS | `in_this_slice` | Timeline stages keep evidence/audit refs one detail step away through progressive selected-work detail. |
| `VS4-UI-006` | MUST_PASS | `in_this_slice` | Claim stage appears as a reviewable candidate in the timeline with supporting refs, not as approved truth. |
| `VS4-UI-007` | MUST_PASS | `in_this_slice` | Recovery states and timeline copy do not permit evidence-free approval. |
| `VS4-UI-008` | MUST_PASS | `in_this_slice` | Memory/Wiki stage remains draft/needs review and never becomes hidden durable memory from timeline display. |
| `VS4-UI-009` | MUST_PASS | `in_this_slice` | Timeline rendering and invalid-loop recovery states do not create hidden durable memory or behavior changes. |
| `VS4-UI-010` | MUST_PASS | `in_this_slice` | Action stage remains an Action Card review with local/mock execution mode and approval boundary visible. |
| `VS4-UI-011` | MUST_PASS | `in_this_slice` | Timeline rendering does not imply live writeback or provider execution. |
| `VS4-UI-012` | MUST_PASS | `in_this_slice` | Returning work in Ops Inbox can be continued through one staged daily-loop journey. |
| `VS4-UI-013` | MUST_PASS | `in_this_slice` | Ask-created work remains represented by the same loop-view refs and does not become a separate chat-only path. |
| `VS4-UI-014` | MUST_PASS | `previous_slice` | Three general-purpose packs remain covered by prior slices. |
| `VS4-UI-015` | MUST_PASS | `in_this_slice` | Timeline and recovery states preserve active owner/workspace scope. |
| `VS4-UI-016` | MUST_PASS | `in_this_slice` | Product language appears before internal refs, error codes, or scenario proof jargon. |
| `VS4-STATE-001` | MUST_PASS | `in_this_slice` | Timeline covers ready, needs-review, blocked/recovery, and audit/log-available states for selected work. |
| `VS4-REF-001` | MUST_PASS | `in_this_slice` | Home/Ops Inbox reference alignment includes the selected-work timeline without adding default nav items. |
| `VS4-REF-002` | MUST_PASS | `in_this_slice` | Claim/Action refs stay aligned in timeline stages and invalid-loop recovery states. |
| `VS4-REG-001` | REGRESSION_GUARD | `later_slice` | VS0 regression is not rerun in this narrow UI slice; existing full-report regression remains separate evidence. |
| `VS4-REG-002` | REGRESSION_GUARD | `later_slice` | VS1 regression is not rerun in this narrow UI slice; existing full-report regression remains separate evidence. |
| `VS4-REG-003` | REGRESSION_GUARD | `in_this_slice` | Timeline introduces no production, on-prem, final security, live-provider, or human UX readiness claim. |
| `VS4-REG-004` | REGRESSION_GUARD | `in_this_slice` | Prompt-injection or invalid-loop recovery cannot approve memory/action, execute actions, call providers, or expand authority. |
| `VS4-REG-005` | REGRESSION_GUARD | `in_this_slice` | Reference images remain design inputs only and are not PASS evidence. |
| `VS4-REG-006` | REGRESSION_GUARD | `in_this_slice` | The default surface remains product-first and not admin/connector/ontology/verifier-first. |
| `VS4-REG-007` | REGRESSION_GUARD | `in_this_slice` | Missing CLI loop-view parity, UI markers, active metadata, or focused gate output blocks PASS. |
| `VS4-H01` | HUMAN_REQUIRED | `human_required` | JiYong/Tars Product Alpha UX acceptance remains unclaimed and requires dated human review evidence. |

`VS3-H01` through `VS3-H07` remain `conditional_deferred` blockers only for production/on-prem/security/live-provider/human-acceptance claims. They do not block local VS4 Product Alpha Slice 025 work.

## Required Verification

- `make verify-vs4-product-alpha-ops-inbox-journey-timeline`
- `make verify-vs4-product-alpha-report-package-integrity`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-slice-025-ops-inbox-journey-timeline.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-slice-025-ops-inbox-journey-timeline-gate.json`
- Focused unittest proving:
  - the active report uses `slice-025-ops-inbox-journey-timeline`;
  - `slice_contracts.slice_025` points to this contract;
  - the proof boundary includes `vs4_slice_025_ops_inbox_journey_timeline`;
  - desktop and mobile browser proof include Journey Timeline markers;
  - timeline stages are populated from runtime loop-view refs;
  - missing/cross-scope/lineage-mismatch recovery states are product-readable and do not create loops, audit refs, authority expansion, or live writeback;
  - `VS4-H01` remains `HUMAN_REQUIRED`.
- Existing documentation checks: SoT, CLI-native-first, design-system, canonical scenario matrix, and `git diff --check`.

## Done Means

- Ops Inbox selected detail shows a staged Journey Timeline with at least six stages: Inbox, Brief, Claim, Memory/Wiki, Action, and Learn.
- Timeline stage refs come from `product_loop.stages`, not a new UI-only source of truth.
- Timeline copy shows product language before raw refs and keeps evidence/audit detail progressively visible.
- Missing, cross-scope, and lineage-mismatch loop refs show safe recovery states and preserve zero negative side effects.
- Focused Slice 025 report and gate paths are distinct from the canonical full VS4 report and full gate paths.
- Full VS4 report metadata, human-review package conditions, and scenario gate active-slice checks recognize Slice 025 as current.
- No production/on-prem/final-security/live-provider/human UX readiness claim is introduced.

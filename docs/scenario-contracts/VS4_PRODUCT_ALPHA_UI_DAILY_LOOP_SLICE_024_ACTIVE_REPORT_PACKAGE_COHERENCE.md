# CornerStone VS4 Product Alpha UI Daily Loop Slice 024 Active Report Package Coherence Contract

**Date:** 2026-07-04 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`

## Goal

Keep the active VS4 scenario report metadata, focused slice verification target, and `VS4-H01` human-review package evidence aligned after Slice 023 split filtered slice reports from the canonical full VS4 report.

Slice 023 prevented filtered return-to-work evidence from overwriting the full 28-row report used by human-package generation. Slice 024 makes that separation self-describing: the full report must identify the latest active slice, include the Slice 024 contract in `slice_contracts`, expose a Slice 024 proof-boundary marker, and keep package/review inputs on the full report plus current slice contracts.

## Scope

- Add Slice 024 as the current active VS4 implementation-slice contract.
- Add a focused `verify-vs4-product-alpha-active-report-package-coherence` target with its own report and gate paths.
- Keep the focused Slice 022 return-to-work report paths distinct from the canonical full VS4 report paths.
- Require the VS4 scenario gate to reject stale active slice metadata or missing Slice 024 contract/proof-boundary markers.
- Add Slice 024 to human-review package input artifacts, evidence refs, and review commands.
- Keep `VS4-H01` `HUMAN_REQUIRED` and package/template validation as review input only.

## Non-Scope

- No new canonical VS4 matrix rows.
- No change to the canonical 206-scenario matrix.
- No product UI redesign, backend storage decision, production provider integration, or new live external writeback.
- No production, on-prem, final security, live-provider, accessibility certification, or human UX readiness claim.
- No attempt to collect or simulate JiYong/Tars human acceptance.

## Assumptions

- Full VS4 reports remain the only valid input for the `VS4-H01` package because the package must observe 27 AI-verifiable PASS rows plus one `HUMAN_REQUIRED` row.
- Focused reports remain valid slice evidence only when their paths, gate output, and `scenario_filter` are explicit.
- Slice 021 browser proof, Slice 022 lineage guard, and Slice 023 report/package split remain previous-slice evidence.
- Reference images remain design guidance only and are not PASS or human-acceptance evidence.

## Scenario Inventory

| Scenario ID | Priority | Slice classification | Gate meaning |
|---|---|---|---|
| `VS4-GATE-001` | MUST_PASS | `in_this_slice` | Slice 024 contract, active report metadata, focused target paths, and gate condition are structurally frozen and verified. |
| `VS4-UI-012` | MUST_PASS | `in_this_slice` | Ops Inbox returning-work proof remains reachable while filtered reports stay separate from the full H01 package report. |
| `VS4-UI-015` | MUST_PASS | `in_this_slice` | Workspace/owner context remains visible in full-report and H01 package evidence. |
| `VS4-REG-003` | REGRESSION_GUARD | `in_this_slice` | Active/package metadata must not introduce production, on-prem, final security, live-provider, or human UX readiness claims. |
| `VS4-REG-005` | REGRESSION_GUARD | `in_this_slice` | Reference images remain visual guidance only and are not package PASS or human-acceptance evidence. |
| `VS4-REG-007` | REGRESSION_GUARD | `in_this_slice` | Missing CLI/report/package parity, stale active-slice metadata, or path overlap blocks PASS. |
| `VS4-H01` | HUMAN_REQUIRED | `human_required` | JiYong/Tars Product Alpha UX acceptance remains unclaimed and requires a dated human review record. |

All other VS4 AI-verifiable rows remain covered by previous slices. `VS3-H01` through `VS3-H07` remain conditional deferred blockers only for production/on-prem/security/live-provider/human-acceptance claims. They do not block local VS4 Product Alpha Slice 024 work.

## Required Verification

- `make verify-vs4-product-alpha-active-report-package-coherence`
- `make verify-vs4-product-alpha-report-package-integrity`
- `make verify-vs4-product-alpha-human-package`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-slice-024-active-report-package-coherence.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-slice-024-active-report-package-coherence-gate.json`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json`
- Focused unittest proving:
  - the active report uses `slice-024-active-report-package-coherence`;
  - `slice_contracts.slice_024` points to this contract;
  - the proof boundary includes `vs4_slice_024_active_report_package_coherence`;
  - the scenario gate rejects stale active-slice metadata;
  - the focused Slice 024 target writes to a slice-specific path and does not overwrite the full report;
  - the human package lists Slice 024 as review input without accepting `VS4-H01`.
- Existing documentation checks: SoT, CLI-native-first, design-system, canonical scenario matrix, and `git diff --check`.

## Done Means

- Full VS4 verification writes active metadata for `slice-024-active-report-package-coherence`.
- Slice 024 focused verification writes to `reports/scenario/vs4-product-alpha-ui-daily-loop-slice-024-active-report-package-coherence.json`.
- Slice 024 focused gate output writes to `reports/scenario/vs4-product-alpha-ui-daily-loop-slice-024-active-report-package-coherence-gate.json`.
- Canonical full VS4 verification and gate paths remain unchanged.
- `cornerstone human-gate package --scope vs4` uses the full report, lists Slice 024 review input, and keeps `VS4-H01` `HUMAN_REQUIRED`.
- The scenario gate fails stale active-slice metadata, missing Slice 024 proof boundary, or missing Slice 024 contract refs.
- No production/on-prem/final-security/live-provider/human UX readiness claim is introduced.

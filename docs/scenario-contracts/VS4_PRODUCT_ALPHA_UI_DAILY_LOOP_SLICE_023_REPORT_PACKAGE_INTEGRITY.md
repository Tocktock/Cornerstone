# CornerStone VS4 Product Alpha UI Daily Loop Slice 023 Report Package Integrity Contract

**Date:** 2026-07-04 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`

## Goal

Keep filtered slice verification evidence separate from the canonical full VS4 report that feeds the `VS4-H01` human-review package.

Slice 022 made `product loop-view` reject missing, cross-scope, and mismatched refs. Its focused make target also wrote the filtered 16-row report to `reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json`, the same path used by the full VS4 human-package flow. Slice 023 closes that evidence hygiene gap: focused slice targets may produce filtered reports, but they must not overwrite the canonical full-report path required by `cornerstone human-gate package --scope vs4`.

## Scope

- Give the return-to-work lineage slice its own report and gate output paths.
- Preserve `reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json` as the full 28-row VS4 report path.
- Preserve `reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json` as the full-report gate output path.
- Keep the `VS4-H01` package command bound to the full report path, not to a filtered slice report.
- Add static regression coverage that prevents the Slice 022 make target from writing to the full-report path again.
- Add the Slice 023 contract to VS4 package/review input evidence without claiming human acceptance.

## Non-Scope

- No new canonical VS4 matrix rows.
- No change to the canonical 206-scenario matrix.
- No product UI redesign or new runtime behavior.
- No new PASS claim from reference images, screenshots alone, or docs alone.
- No production, on-prem, final security, live-provider, or final human UX readiness claim.
- No attempt to collect or simulate JiYong/Tars human acceptance.

## Assumptions

- Filtered scenario reports remain valid local evidence for focused slices.
- Full VS4 reports remain the only valid input for `VS4-H01` package generation because the package must see 27 AI-verifiable PASS rows plus one `HUMAN_REQUIRED` row.
- The existing scenario gate already accepts filtered reports when their `scenario_filter` matches the rows present.
- Reference images remain design guidance only and are not PASS evidence.

## Scenario Inventory

| Scenario ID | Priority | Slice classification | Gate meaning |
|---|---|---|---|
| `VS4-GATE-001` | MUST_PASS | `in_this_slice` | Slice 023 contract, Makefile target paths, full-report package path, and gate output are structurally frozen and verified. |
| `VS4-UI-001` | MUST_PASS | `previous_slice` | Home remains Product Alpha first value from earlier slices. |
| `VS4-UI-002` | MUST_PASS | `previous_slice` | Drop/Paste source preservation remains covered by prior slices. |
| `VS4-UI-003` | MUST_PASS | `previous_slice` | Evidence-backed Brief creation remains covered by prior slices. |
| `VS4-UI-004` | MUST_PASS | `previous_slice` | Brief detail contents remain covered by prior slices. |
| `VS4-UI-005` | MUST_PASS | `previous_slice` | Evidence Drawer remains covered by prior slices. |
| `VS4-UI-006` | MUST_PASS | `previous_slice` | Claim candidate creation remains covered by prior slices. |
| `VS4-UI-007` | MUST_PASS | `previous_slice` | Evidence-free approval denial remains covered by prior slices. |
| `VS4-UI-008` | MUST_PASS | `previous_slice` | Memory/Wiki candidate lineage remains covered by Slice 022. |
| `VS4-UI-009` | MUST_PASS | `previous_slice` | No hidden memory remains covered by prior slices. |
| `VS4-UI-010` | MUST_PASS | `previous_slice` | Action Card review remains covered by prior slices. |
| `VS4-UI-011` | MUST_PASS | `previous_slice` | Local/mock execution boundary remains covered by prior slices. |
| `VS4-UI-012` | MUST_PASS | `in_this_slice` | Ops Inbox returning-work proof must remain available in focused slice evidence without replacing full VS4 evidence. |
| `VS4-UI-013` | MUST_PASS | `previous_slice` | Ask-to-work-item remains covered by prior slices. |
| `VS4-UI-014` | MUST_PASS | `previous_slice` | Three general-purpose packs remain covered by prior slices. |
| `VS4-UI-015` | MUST_PASS | `in_this_slice` | Workspace/owner context must remain visible in full-report and H01 package evidence. |
| `VS4-UI-016` | MUST_PASS | `previous_slice` | Product language remains covered by prior slices. |
| `VS4-STATE-001` | MUST_PASS | `previous_slice` | Required page states remain covered by prior slices. |
| `VS4-REF-001` | MUST_PASS | `previous_slice` | Home/Search/Artifact reference alignment remains covered by prior slices. |
| `VS4-REF-002` | MUST_PASS | `previous_slice` | Claim/Action reference alignment remains covered by prior slices. |
| `VS4-REG-001` | REGRESSION_GUARD | `previous_slice` | VS0 regression remains covered by prior slices. |
| `VS4-REG-002` | REGRESSION_GUARD | `previous_slice` | VS1 regression remains covered by prior slices. |
| `VS4-REG-003` | REGRESSION_GUARD | `in_this_slice` | Report/package separation must not introduce production, on-prem, final security, live-provider, or human UX readiness claims. |
| `VS4-REG-004` | REGRESSION_GUARD | `previous_slice` | Prompt-injection authority denial remains covered by prior slices. |
| `VS4-REG-005` | REGRESSION_GUARD | `in_this_slice` | Reference images remain visual guidance only; package evidence comes from full report, browser proof, CLI, and human template inputs. |
| `VS4-REG-006` | REGRESSION_GUARD | `previous_slice` | Product-first UI remains covered by prior slices. |
| `VS4-REG-007` | REGRESSION_GUARD | `in_this_slice` | Missing CLI/report/package parity blocks PASS; slice and full report paths must be distinct. |
| `VS4-H01` | HUMAN_REQUIRED | `human_required` | JiYong/Tars Product Alpha UX acceptance remains unclaimed and requires a full 28-row report package. |

`VS3-H01` through `VS3-H07` remain `conditional_deferred` blockers only for production/on-prem/security/live-provider/human-acceptance claims. They do not block local VS4 Product Alpha Slice 023 work.

## Required Verification

- `make verify-vs4-product-alpha-return-to-work-lineage`
- `make verify-vs4-product-alpha-human-package`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-slice-022-return-to-work-lineage.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-slice-022-return-to-work-lineage-gate.json`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json`
- Focused unittest proving:
  - the Slice 022 make target writes to a slice-specific report path;
  - the Slice 022 make target does not write to the canonical full-report path;
  - the human-package target uses the canonical full-report path;
  - Slice 023 is listed as review/package evidence.
- Existing documentation checks: SoT, CLI-native-first, design-system, canonical scenario matrix, and `git diff --check`.

## Done Means

- Filtered Slice 022 verification writes to `reports/scenario/vs4-product-alpha-ui-daily-loop-slice-022-return-to-work-lineage.json`.
- Filtered Slice 022 gate output writes to `reports/scenario/vs4-product-alpha-ui-daily-loop-slice-022-return-to-work-lineage-gate.json`.
- Canonical full VS4 verification writes to `reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json`.
- Canonical full VS4 gate output writes to `reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json`.
- `cornerstone human-gate package --scope vs4` succeeds only with the full report and keeps `VS4-H01` as `HUMAN_REQUIRED`.
- Slice 023 appears in README/SoT/package evidence and does not add canonical matrix rows.
- No production/on-prem/final-security/live-provider/human UX readiness claim is introduced.

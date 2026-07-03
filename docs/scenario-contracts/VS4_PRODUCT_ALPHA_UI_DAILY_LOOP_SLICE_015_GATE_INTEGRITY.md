# CornerStone VS4 Product Alpha UI Daily Loop Slice 015 Gate Integrity Contract

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`

## Goal

Make `cornerstone scenario gate` enforce the VS4 Product Alpha proof boundary instead of accepting a report only because AI rows have no generic blocking status.

The gate must reject VS4 reports that overclaim production, on-prem, final security, live-provider, or human UX readiness; mark `VS4-H01` as `PASS`; use reference images as PASS or acceptance evidence; omit CLI parity transcripts; omit source-tree freshness metadata; or carry nonzero negative evidence.

## Scope

- Add VS4-specific scenario-gate validation for `vs4-product-alpha-ui-daily-loop` reports.
- Add `cornerstone scenario gate ... --output <path>` so the gate can produce a durable JSON evidence artifact.
- Keep existing filtered VS4 slice reports gateable when their filtered rows, proof boundary, negative evidence, CLI parity, and source-tree metadata are valid.
- Require full VS4 reports to preserve 28 rows: 27 AI-verifiable `PASS`, 1 `HUMAN_REQUIRED`.
- Emit gate validation details in the gate payload: conditions, summary, negative evidence, source-report metadata, and failures.
- Update the VS4 report generator so Slice 015 is the active slice and source-tree metadata is included.

## Non-Scope

- No new canonical VS4 matrix rows.
- No change to the canonical 206-scenario matrix.
- No UI redesign or new Product Alpha page behavior.
- No production, on-prem, final security, live-provider, or human UX readiness claim.
- No promotion of `VS4-H01` beyond `HUMAN_REQUIRED`.
- No strict current-HEAD-only gate that makes committed reports impossible to inspect after a later commit.

## Assumptions

- VS4 report generation can include source-tree metadata from `git_verification_metadata(root)`.
- Generated reports are evidence snapshots; a later commit may make them historical, so the gate checks that freshness metadata exists and is structured rather than requiring simple HEAD equality.
- Existing Slice 001 through Slice 014 runtime/UI/CLI proof remains the product behavior surface; Slice 015 hardens the acceptance gate around that proof.

## Selected Scenarios

| Scenario ID | Priority | Slice classification | Gate meaning |
|---|---|---|---|
| `VS4-GATE-001` | MUST_PASS | `in_this_slice` | Contract, matrix, slice registration, and gate output are structurally frozen and verified. |
| `VS4-REG-003` | REGRESSION_GUARD | `in_this_slice` | Production/on-prem/final-security/live-provider/human UX overclaims are rejected. |
| `VS4-REG-004` | REGRESSION_GUARD | `in_this_slice` | Prompt-injection and unsafe authority side-effect counters stay zero. |
| `VS4-REG-005` | REGRESSION_GUARD | `in_this_slice` | Reference images remain design guidance only, not PASS or human-acceptance evidence. |
| `VS4-REG-007` | REGRESSION_GUARD | `in_this_slice` | CLI parity transcripts and JSON/native command evidence remain required. |
| `VS4-H01` | HUMAN_REQUIRED | `human_required` | Human product-alpha UX acceptance remains unclaimed. |

All other AI-verifiable VS4 rows remain `previous_slice` when the full report is generated. `VS3-H01` through `VS3-H07` remain conditional deferred blockers only for production/on-prem/security/live-provider/human-acceptance claims.

## Required Gate Checks

The VS4 scenario gate must validate:

- report identity: `scenario_set=vs4-product-alpha-ui-daily-loop` and `status=success`;
- row coverage: exact full or filtered scenario IDs, no duplicates, no unexpected rows, matching `id` and `scenario_id`;
- row status: AI-owned rows are `PASS`; `VS4-H01` is `HUMAN_REQUIRED` and Human-owned when present;
- summary counts match the gated rows, and full reports keep `scenario_count=28`, `pass=27`, `human_required=1`, `blocking=0`, `fail=0`, `not_run=0`;
- proof boundary keeps `production`, `production_onprem`, `final_security_acceptance`, and `live_provider` as `NOT_CLAIMED`, and `human_ux_acceptance` as `HUMAN_REQUIRED`;
- `vs4_slice_015_gate_integrity` is present in the proof boundary;
- negative evidence is present, numeric, and zero for required safety counters;
- reference-image markers prove reference images are not PASS or human-acceptance evidence;
- self command transcript records native `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json --output ...`;
- CLI workflow, Ask/pack workflow, and regression workflow transcripts are present;
- source-tree metadata is present with base commit, tree hash, source snapshot, generated-dirty snapshot, and dirty-path fields.

## Proof Needed

- `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json`
- Positive and tampered-report unit tests for VS4 gate acceptance/rejection.
- Existing documentation checks: SoT, CLI-native-first, design-system, canonical scenario matrix, and `git diff --check`.

## Done Means

- Full VS4 report gates successfully with VS4-specific validation details.
- Tampered VS4 reports fail for human-row PASS, overclaim, reference-image evidence misuse, missing CLI transcript, missing source-tree metadata, and nonzero negative evidence.
- `VS4-H01` remains `HUMAN_REQUIRED`.
- No production/on-prem/security/live-provider/human UX readiness claim is introduced.

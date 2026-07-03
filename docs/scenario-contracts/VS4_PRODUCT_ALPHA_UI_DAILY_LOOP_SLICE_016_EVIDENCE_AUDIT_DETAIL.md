# CornerStone VS4 Product Alpha UI Daily Loop Slice 016 Evidence/Audit Detail Contract

**Date:** 2026-07-04 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`

## Goal

Make `Evidence / Audit` a product-readable detail surface in the VS4 Product Alpha UI, not a one-line verification note.

The user must be able to move from Evidence Drawer or Action detail into a detail view that explains source, provenance, supporting evidence, safety check, activity timeline, audit verification, and Learn review linkage without adding `Audit` to the small normal-user navigation.

## Scope

- Replace the minimal `Audit Detail` section with a product-ready `Evidence / Audit` detail surface.
- Keep `Audit` reachable contextually from the shared Evidence Drawer and Action evidence detail.
- Show source/provenance, Evidence-backed Brief, Claim, Action Card, safety check, local boundary, activity timeline, audit verification, external-call boundary, and Learn review candidate linkage.
- Add deterministic browser-proof markers for desktop and mobile Evidence/Audit detail.
- Update the VS4 scenario verifier so Slice 016 is the active slice and selected rows require the new Evidence/Audit markers.
- Add a focused make target and unittest for this slice.

## Non-Scope

- No new canonical VS4 matrix rows.
- No change to the canonical 206-scenario matrix.
- No default-nav expansion beyond `Home`, `Search`, `Artifacts`, `Claims`, `Actions`.
- No new backend storage model or audit-ledger migration.
- No production, on-prem, final security, live-provider, or human UX readiness claim.
- No promotion of `VS4-H01` beyond `HUMAN_REQUIRED`.

## Assumptions

- The existing local audit verifier (`cornerstone audit verify --json`) remains the deterministic CLI proof for local audit integrity.
- The Evidence/Audit detail can be a product surface over existing local state and browser proof for this slice; no final storage model is required.
- Reference images guide layout and information hierarchy only. They are not PASS evidence.

## Selected Scenarios

| Scenario ID | Priority | Slice classification | Gate meaning |
|---|---|---|---|
| `VS4-GATE-001` | MUST_PASS | `in_this_slice` | Slice 016 contract, registration, verifier wiring, and gate output are structurally frozen and verified. |
| `VS4-UI-005` | MUST_PASS | `in_this_slice` | Evidence Drawer leads to product-readable Evidence/Audit detail with source, provenance, support, safety, activity, and audit refs. |
| `VS4-UI-012` | MUST_PASS | `in_this_slice` | Returning daily work exposes an Evidence/Audit path from Ops/Action detail without making the product one-shot. |
| `VS4-UI-016` | MUST_PASS | `in_this_slice` | Evidence/Audit copy uses product language first and keeps raw refs progressively disclosed. |
| `VS4-STATE-001` | MUST_PASS | `in_this_slice` | Audit/log available state is observable on the Evidence/Audit detail, with local boundary and review-required state. |
| `VS4-REG-003` | REGRESSION_GUARD | `in_this_slice` | Evidence/Audit detail does not overclaim production, on-prem, final security, live-provider, accessibility, or human UX readiness. |
| `VS4-REG-006` | REGRESSION_GUARD | `in_this_slice` | Evidence/Audit remains contextual and product-first; default normal-user nav is still small and not admin/audit-heavy. |
| `VS4-REG-007` | REGRESSION_GUARD | `in_this_slice` | Audit detail has native CLI parity evidence through local audit verification and VS4 JSON report transcripts. |
| `VS4-H01` | HUMAN_REQUIRED | `human_required` | Human product-alpha UX acceptance remains unclaimed. |

All other AI-verifiable VS4 rows remain `previous_slice` when the full report is generated. `VS3-H01` through `VS3-H07` remain conditional deferred blockers only for production/on-prem/security/live-provider/human-acceptance claims.

## Required Verification

- `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-005 --scenario VS4-UI-012 --scenario VS4-UI-016 --scenario VS4-STATE-001 --scenario VS4-REG-003 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json`
- Focused unittest asserting desktop and mobile `evidence_audit_detail_markers`.
- Existing documentation checks: SoT, CLI-native-first, design-system, canonical scenario matrix, and `git diff --check`.

## Done Means

- Evidence/Audit detail is visible in browser proof and reachable from the shared Evidence Drawer and Action detail.
- The detail surface shows source, provenance, evidence, claim, action, safety check, activity timeline, audit verification, and Learn review linkage.
- Desktop and mobile browser proofs expose `evidence_audit_detail_markers` and all selected marker values are true.
- Full and filtered VS4 reports gate successfully with `vs4_slice_016_evidence_audit_detail` in `proof_boundary`.
- `VS4-H01` remains `HUMAN_REQUIRED`.
- No production/on-prem/security/live-provider/human UX readiness claim is introduced.

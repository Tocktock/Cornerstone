# CornerStone VS4 Slice 004 Human Acceptance Package

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice prepares `VS4-H01` review evidence only; it does not provide human UX acceptance.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Previous slices:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md`

## Goal

Prepare the remaining VS4 human gate so JiYong/Tars can accept or reject the Product Alpha daily loop with explicit, dated evidence:

```text
Drop / Ask
-> Evidence-backed Brief
-> Claim candidate
-> Memory/Wiki candidate
-> Action Card draft
-> Ops Inbox follow-up
-> Evidence/Audit detail
-> Learn
```

This slice must make `VS4-H01` reviewable without marking it `PASS`, collecting approval, unlocking production/on-prem/security/live-provider claims, or treating structural validation as human acceptance.

## Scope

In this slice:

- generate a `VS4-H01` human review package bound to the current full VS4 scenario report;
- generate a blank reviewer-record template for JiYong/Tars;
- expose a native `cornerstone human-gate ... --scope vs4 --json` path for package creation and record validation;
- validate filled reviewer records for structure, redaction safety, required evidence fields, and overclaim markers;
- keep all package, template, and validation outputs as preparation evidence only;
- preserve the full VS4 proof boundary: 27 AI-verifiable rows can pass locally, `VS4-H01` remains `HUMAN_REQUIRED`, and VS3-H01 through VS3-H07 remain conditional deferred.

## Non-Scope

This slice does not implement:

- human Product Alpha UX acceptance;
- a signed JiYong/Tars review decision;
- production, on-prem, final security, real IdP, real network, live-provider, or migration readiness;
- any live provider writeback or external mutation;
- any state transition that marks `VS4-H01` as `PASS`.

## Assumptions

- The unfiltered `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json` report is the authoritative AI-verifiable input for this human package.
- The package may include local screenshot, DOM, scenario-report, and CLI/browser proof references as review inputs, but those inputs are not human acceptance.
- The reviewer may approve, approve with exceptions, or reject the UX after using the local Product Alpha flow.
- Redacted evidence references are acceptable if the reviewer records what was redacted and why no secrets or private tokens are included.

## Selected Scenarios

This slice selects the remaining human row as review preparation:

| ID | Classification | Why |
|---|---|---|
| VS4-H01 | human_required | JiYong/Tars product-alpha UX acceptance requires a dated human walkthrough record. This slice prepares the review package and validator only. |

All AI-verifiable rows remain `previous_slice` and must still be rerunnable through the unfiltered VS4 verifier.

## Full Scenario Classification

| ID Range | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001; VS4-UI-001 through VS4-UI-016; VS4-STATE-001; VS4-REF-001 through VS4-REF-002; VS4-REG-001 through VS4-REG-007 | previous_slice | Must remain locally `PASS` in the unfiltered VS4 verifier before the human package is considered ready. |
| VS4-H01 | human_required | Package and blank template can be generated; acceptance remains human-owned. |
| VS3-H01 through VS3-H07 | conditional_deferred | Still block production/on-prem/security/live-provider/human acceptance claims only. |

## Proof Needed

The selected slice can be considered prepared only with:

- a full VS4 scenario report showing 28 rows, 27 local AI-verifiable `PASS`, 1 `HUMAN_REQUIRED`, 0 blocking, and no production/on-prem/final-security/live-provider claims;
- `reports/human-gates/vs4/VS4-H01.json` containing review checklist, reject conditions, required evidence fields, validation command, claim-boundary flags, and package digest;
- `reports/human-gates/vs4/record-templates/VS4-H01.review-record.template.json` as a blank template with no decision recorded;
- `cornerstone human-gate package --scope vs4 --json` output showing `final_verdict=HUMAN_REQUIRED`;
- `cornerstone human-gate validate-record --scope vs4 --scenario VS4-H01 --record-file <record> --json` output proving structural validation keeps `matrix_status_after_validation=HUMAN_REQUIRED`;
- negative evidence counters for package-created approval, human row marked PASS, acceptance claim, production/on-prem/security/live-provider claim, live calls, external mutations, persisted raw reviewer body, and unredacted sensitive marker leakage;
- docs, CLI-native-first, design-system, scenario-matrix, unit, and diff checks.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED` until JiYong/Tars supplies a dated review record.
- A structurally valid `APPROVE` or `APPROVE_WITH_EXCEPTIONS` record may be validation evidence, but it does not by itself mark `VS4-H01` as `PASS`.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.

## Done Criteria

This slice is done when:

1. the full VS4 verifier still reports 27 AI-owned rows `PASS`, 1 `HUMAN_REQUIRED`, and 0 blocking rows;
2. the `VS4-H01` package and blank template are generated under `reports/human-gates/vs4/`;
3. the package includes review tasks for Drop/Ask, Brief, Claim, Memory/Wiki, Action Card, Ops Inbox, Evidence/Audit, Learn, local-mode boundary, product language, and misleading-state review;
4. the package and template state that they are not acceptance evidence and cannot promote `VS4-H01`;
5. the validator accepts a structurally complete redacted sample record while still returning `final_verdict=HUMAN_REQUIRED`;
6. the validator rejects blank, malformed, sensitive, or overclaiming records;
7. no package, validation output, report, README, or doc claims production, on-prem, final security, live-provider, or human UX readiness.

## CLI Parity

- Package: `cornerstone human-gate package --scope vs4 --scenario-report reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json`.
- Template output: `cornerstone human-gate package --scope vs4 --record-template-output reports/human-gates/vs4/record-templates/VS4-H01.review-record.template.json --json`.
- Validation: `cornerstone human-gate validate-record --scope vs4 --scenario VS4-H01 --record-file <filled-review-record.json> --json --output <redacted-validation-envelope.json>`.
- Verification: `make verify-vs4-product-alpha-human-package`.
- CLI status: `PASS` applies only to package/validator structure. `VS4-H01` remains `HUMAN_REQUIRED` until explicit human acceptance evidence exists.

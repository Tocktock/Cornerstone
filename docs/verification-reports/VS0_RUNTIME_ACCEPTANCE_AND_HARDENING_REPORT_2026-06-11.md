# VS0 Runtime Acceptance And Hardening Report - 2026-06-11

## Summary

- Verdict: local VS0 runtime acceptance and hardening is AI-verified.
- Scope: local release-candidate evidence only.
- Date: 2026-06-11
- Owner: JiYong / Tars
- Commit: `90c15b5`

This report does not mark production release, live connector readiness, or human usability acceptance complete.

## Frozen Goal

Turn the local VS0 runtime from AI-verifiable scenario pass into an operator-acceptable local release candidate.

Close:

1. real browser UI proof;
2. human usability walkthrough path;
3. clearer readiness semantics;
4. release-facing evidence package;
5. no production/live-provider overclaim.

Live ConnectorHub/provider execution remains `HUMAN_REQUIRED` and out of scope.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| VS0-ACC-001 | MUST_PASS | PASS | Chrome headless browser proof at `reports/browser/vs0-runtime-acceptance-2026-06-11/browser-proof.json`; screenshot `home.png`; DOM snapshot `home.dom.html`. |
| VS0-ACC-002 | MUST_PASS | PASS | `cornerstone ready --json` includes last successful `vs0-product-runtime` report path, timestamp, commit, scenario status, and gate status. |
| VS0-ACC-003 | MUST_PASS | PASS | Action dry-run impact uses `expected_connector_calls=1`, `mock_connector_calls=1`, and `real_external_http_calls=0`; no ambiguous dry-run `external_calls` field. |
| VS0-ACC-004 | MUST_PASS | PASS | README contains `VS0 Runtime Acceptance Quickstart` from runtime start through audit verification. |
| VS0-ACC-005 | MUST_PASS | PASS | Release package manifest at `reports/release/vs0-runtime-acceptance-2026-06-11/manifest.json`. |
| VS0-ACC-R01 | REGRESSION_GUARD | PASS | `production_release_ready=false`; live provider and human usability remain `HUMAN_REQUIRED`. |
| VS0-ACC-R02 | REGRESSION_GUARD | PASS | Existing `vs0-product-runtime` report remains success with 12 PASS, 2 HUMAN_REQUIRED, 0 blocking. |
| VS0-ACC-H01 | HUMAN_REQUIRED | HUMAN_REQUIRED | Live ConnectorHub/provider verification requires approval, credentials, and redacted transcript/audit refs. |
| VS0-ACC-H02 | HUMAN_REQUIRED | HUMAN_REQUIRED | JiYong/Tars usability walkthrough and acceptance/rejection note required. |

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS0-ACC-H01 | Live provider execution needs credentials and may mutate third-party state. | Approve and perform live ConnectorHub/provider dry-run/execution later. | Redacted transcript, provider/action result, audit refs, written approval. | Blocks live-provider production release, not local runtime acceptance. |
| VS0-ACC-H02 | Usability acceptance is subjective. | JiYong/Tars walks through the VS0 runtime and records accept/reject. | Acceptance note plus screenshots/recording or issue list. | Blocks human product acceptance claim, not deterministic local acceptance checks. |

## Command Evidence

Passed:

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-runtime-acceptance --json --output reports/scenario/vs0-runtime-acceptance-2026-06-11.json
make verify-vs0-acceptance
make verify-vs0-runtime
make verify-local-fast
PATH="$PWD:$PATH" cornerstone release report-check --scenario-report reports/scenario/vs0-runtime-acceptance-2026-06-11.json --verification-report docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_REPORT_2026-06-11.md --json
```

Observed summary:

```json
{
  "status": "success",
  "summary": {
    "scenario_count": 9,
    "pass": 7,
    "human_required": 2,
    "blocking": 0,
    "product_feature_claims": "LOCAL_VS0_RUNTIME_ACCEPTANCE_READY_PRODUCTION_NOT_READY"
  }
}
```

Release package:

```json
{
  "status": "success",
  "manifest_path": "reports/release/vs0-runtime-acceptance-2026-06-11/manifest.json",
  "artifact_count": 12,
  "missing_required": []
}
```

Release report-check:

```json
{
  "status": "passed",
  "scenario_count": 9,
  "pass_count": 7,
  "human_required_count": 2,
  "blocking": 0,
  "errors": []
}
```

Browser proof:

```json
{
  "status": "passed",
  "screenshot_path": "reports/browser/vs0-runtime-acceptance-2026-06-11/home.png",
  "screenshot_bytes": 101421,
  "production_overclaim_absent": true
}
```

Negative evidence:

```json
{
  "real_external_http_calls": 0,
  "unqualified_external_calls_in_release_report": 0,
  "production_release_overclaim": 0,
  "live_connector_claim_without_human_evidence": 0,
  "human_usability_claim_without_human_evidence": 0
}
```

## Files Changed

- `packages/cornerstone_cli/acceptance.py`
- `packages/cornerstone_cli/product_runtime.py`
- `packages/cornerstone_cli/runtime.py`
- `packages/cornerstone_cli/scenarios.py`
- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`
- `Makefile`
- `README.md`
- `docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_CONTRACT.md`
- `docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_MATRIX.csv`
- `docs/scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md`
- `docs/sot/README.md`
- `docs/sot/sot_manifest.yaml`
- `reports/scenario/vs0-product-runtime-2026-06-11.json`
- `reports/scenario/vs0-runtime-acceptance-2026-06-11.json`
- `reports/browser/vs0-runtime-acceptance-2026-06-11/*`
- `reports/release/vs0-runtime-acceptance-2026-06-11/*`

## Gaps And Risks

- Chrome headless on this macOS environment writes the screenshot but times out instead of exiting normally; the proof records `chrome_timeout_after_screenshot=true` and validates the screenshot file, DOM snapshot, required UI labels, and no production overclaim.
- Live ConnectorHub/provider execution remains `HUMAN_REQUIRED`.
- Human usability acceptance remains `HUMAN_REQUIRED`; the release package includes `reports/release/vs0-runtime-acceptance-2026-06-11/human-usability-walkthrough.md` for JiYong/Tars to record the human decision.
- Production release readiness remains false by design.

## Verdict

- AI-verifiable scope: done.
- Human/release gate: needs-human-verification for live-provider and usability acceptance.
- Release boundary: local VS0 runtime acceptance ready; production release not ready.

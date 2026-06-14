# VS0 Evidence Cleanup And Interactive UI Loop Report - 2026-06-13

## Summary

- Verdict: local VS0 evidence cleanup and interactive UI loop is AI-verified.
- Scope: local/mock EVUX milestone evidence only.
- Date: 2026-06-13
- Owner: JiYong / Tars
- Verified base commit: `43d7cc5`
- Verified base tree hash: `037592d2b90e959b18c9ba1bb11d60ef9d5edd78`
- Verified source worktree hash: `f31f0ed75cf134015411fa7d02572f619465f2d5ad6396a783b76fb5da9be61c`
- Final commit: pending while this report is generated from a dirty worktree; see the governance post-commit rollup after commit.

This report does not mark production release, live connector readiness, or human usability acceptance complete.

## Frozen Goal

Cleanly finish the local VS-0 product milestone by fixing acceptance evidence quality and making the VS-0 loop operable from the UI:

```text
upload/select Artifact
-> Search
-> Evidence Bundle
-> Claim
-> Action Card dry-run
-> approval
-> local/mock execution
-> Audit timeline
```

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| VS0-EVUX-001 | MUST_PASS | PASS | API transcript `POST /artifacts`, `GET /artifacts/{artifact_id}`; browser proof `reports/browser/vs0-evux-2026-06-13/browser-proof.json`; generated Artifact `art_2735313e0cb92563`. |
| VS0-EVUX-002 | MUST_PASS | PASS | API transcript `POST /search`, `GET /search-snapshots/{snapshot_id}`; browser marker `search_snapshot_id`; generated Search Snapshot `search_a1e48168ccb61ece`. |
| VS0-EVUX-003 | MUST_PASS | PASS | API transcript `POST /evidence-bundles`, `POST /claims`, `POST /claims/{claim_id}/approve`; generated Evidence Bundle `evb_ce31988b959efa38`; Claim `claim_ffca2b21989599d3`. |
| VS0-EVUX-004 | MUST_PASS | PASS | Zero-evidence Claim approval returns `CS_CLAIM_EVIDENCE_REQUIRED`; browser proof includes denial marker. |
| VS0-EVUX-005 | MUST_PASS | PASS | Action Card `action_837b4356b6ee5eab` API transcript and dry-run JSON expose diff, expected impact, policy decision, risk, approval state, ConnectorHub boundary, evidence refs, and audit refs. |
| VS0-EVUX-006 | MUST_PASS | PASS | Action approval/execution transcript records `mock_connector_calls=1`, `real_external_http_calls=0`, and no credential exposure. |
| VS0-EVUX-007 | MUST_PASS | PASS | Audit timeline includes artifact, search, evidence bundle, claim approval, action proposal, approval, execution, and `audit verify` success. |
| VS0-EVUX-008 | MUST_PASS | PASS | Release evidence manifest `reports/release/vs0-evux-2026-06-13/manifest.json` includes scenario report, browser proof, screenshot, DOM, workflow trace, quickstart report, command transcript, command evidence, human-required checklist, verification matrix, and hashes. |
| VS0-EVUX-R01 | REGRESSION_GUARD | PASS | Browser proof uses Chrome DevTools Protocol, clicks `#run-evux`, captures workflow trace, and requires generated IDs; static labels alone cannot pass. |
| VS0-EVUX-R02 | REGRESSION_GUARD | PASS | `cornerstone quickstart verify vs0-evux --json --output reports/quickstart/vs0-evux-quickstart.json` exits 0 and records generated IDs plus audit success. |
| VS0-EVUX-R03 | REGRESSION_GUARD | PASS | Regression transcript records exit 0 for `make verify-local-fast`, `make verify-vs0-runtime`, `make verify-vs0-acceptance`, and the EVUX candidate gate. |
| VS0-EVUX-R04 | REGRESSION_GUARD | PASS | Browser proof status is `PASS` only when Chrome exits cleanly; current proof has `clean_browser_exit=true`, `chrome_exit_code=0`, `chrome_timeout=false`. |
| VS0-EVUX-H01 | HUMAN_REQUIRED | HUMAN_REQUIRED | JiYong/Tars must complete UI walkthrough and record accept/reject with screenshots/recording or issue list. |
| VS0-EVUX-H02 | HUMAN_REQUIRED | HUMAN_REQUIRED | Human-approved live ConnectorHub/provider proof requires credentials/external state and redacted transcript/audit refs. |

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS0-EVUX-H01 | Human usability is subjective. | JiYong/Tars completes the local UI walkthrough and records accept or reject. | Acceptance note plus screenshots/recording or issue list. | Blocks operator-accepted product claim, not AI-verifiable local EVUX gate. |
| VS0-EVUX-H02 | Live provider verification requires credentials and may mutate third-party state. | Human approves and performs live ConnectorHub/provider dry-run or execution later. | Redacted provider transcript, written approval, execution result, audit refs. | Blocks live-provider production release, not local EVUX proof. |

## Command Evidence

Passed:

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-evux --json --output reports/scenario/vs0-evux-2026-06-13.json
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs0-evux-2026-06-13.json --json
PATH="$PWD:$PATH" cornerstone quickstart verify vs0-evux --json --output reports/quickstart/vs0-evux-quickstart.json
PATH="$PWD:$PATH" cornerstone release evidence collect --scope vs0-evux --json
make verify-local-fast
make verify-vs0-runtime
make verify-vs0-acceptance
make verify-vs0-evux
```

Observed scenario summary:

```json
{
  "status": "success",
  "summary": {
    "scenario_count": 14,
    "pass": 12,
    "human_required": 2,
    "blocking": 0,
    "product_feature_claims": "LOCAL_VS0_EVUX_READY_PRODUCTION_NOT_READY"
  }
}
```

Scenario gate:

```json
{
  "status": "success",
  "scenario_count": 14,
  "blocking_count": 0
}
```

Browser proof:

```json
{
  "status": "PASS",
  "clean_browser_exit": true,
  "chrome_exit_code": 0,
  "chrome_timeout": false,
  "screenshot_path": "reports/browser/vs0-evux-2026-06-13/workflow.png",
  "screenshot_bytes": 101350,
  "trace_path": "reports/browser/vs0-evux-2026-06-13/workflow-trace.json"
}
```

Quickstart:

```json
{
  "status": "success",
  "generated_ids": {
    "artifact_id": "art_2735313e0cb92563",
    "search_snapshot_id": "search_e35d06543277c827",
    "evidence_bundle_id": "evb_bb6553638535df6f",
    "claim_id": "claim_ccea3ecc34408418",
    "action_id": "action_358903d8448a0da0"
  },
  "final_audit_verification": {
    "status": "success",
    "event_count": 16
  }
}
```

Release package:

```json
{
  "status": "success",
  "manifest_path": "reports/release/vs0-evux-2026-06-13/manifest.json",
  "artifact_count": 16,
  "command_transcript_path": "reports/release/vs0-evux-2026-06-13/command-transcript.json",
  "missing_required": []
}
```

Command transcript:

```json
{
  "schema_version": "cs.release_command_transcript.v0",
  "command_count": 8,
  "pass": 8,
  "blocking": 0
}
```

Negative evidence:

```json
{
  "real_external_http_calls": 0,
  "zero_evidence_claim_approved": 0,
  "production_release_overclaim": 0,
  "live_connector_claim_without_human_evidence": 0,
  "human_usability_claim_without_human_evidence": 0,
  "browser_timeout_marked_pass": 0,
  "tool_calls_from_untrusted_artifact": 0,
  "action_cards_from_prompt_injection": 0,
  "cross_namespace_reads": 0
}
```

## Changed Files

- `Makefile`
- `README.md`
- `docs/sot/README.md`
- `docs/sot/sot_manifest.yaml`
- `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_VERIFICATION_MATRIX.csv`
- `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_CONTRACT.md`
- `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_MATRIX.csv`
- `docs/verification-reports/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_REPORT_2026-06-13.md`
- `packages/cornerstone_cli/acceptance.py`
- `packages/cornerstone_cli/main.py`
- `packages/cornerstone_cli/product_runtime.py`
- `packages/cornerstone_cli/scenarios.py`
- `tests/scenario/test_scaffold_cli.py`
- `reports/browser/vs0-evux-2026-06-13/*`
- `reports/quickstart/vs0-evux-quickstart.json`
- `reports/scenario/vs0-evux-2026-06-13.json`
- `reports/release/vs0-evux-2026-06-13/*`
- `reports/release/vs0-runtime-acceptance-2026-06-11/command-transcript.json`

## Gaps And Risks

- Human usability acceptance remains `HUMAN_REQUIRED`.
- Live ConnectorHub/provider execution remains `HUMAN_REQUIRED`.
- Production release readiness remains false by design.
- The report was generated before the implementation/evidence commit, so scenario metadata records `worktree_dirty_at_verification=true`, `report_generated_before_commit=true`, `final_commit=null`, and `final_commit_pending_reason=worktree_dirty_at_verification`.
- Commit/push sign-off must use the `VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE` post-commit rollup before claiming clean governance closure.

## Verdict

- AI-verifiable scope: done.
- Human/release gate: needs-human-verification for live-provider and usability acceptance.
- Release boundary: local VS0 EVUX ready; production release not ready.

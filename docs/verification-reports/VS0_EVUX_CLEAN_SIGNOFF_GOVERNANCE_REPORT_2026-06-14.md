# VS0 EVUX Clean Sign-off Governance Report - 2026-06-14

## Summary

- Goal: make the existing local VS0 EVUX evidence package cleanly sign-offable without adding product features.
- Verdict: AI-verifiable governance scope PASS.
- Scenario set: `vs0-evux-governance`
- Governance report: `reports/scenario/vs0-evux-governance-2026-06-14.json`
- EVUX source report: `reports/scenario/vs0-evux-2026-06-13.json`
- Release manifest: `reports/release/vs0-evux-2026-06-13/manifest.json`
- Command transcript: `reports/release/vs0-evux-2026-06-13/command-transcript.json`
- Post-commit rollup: `reports/release/vs0-evux-2026-06-13/post_commit_rollup.json`
- Finalized implementation/evidence commit: `349dea7`
- Finalized tree hash: `f12b9393f957a72f1eb26af252b92f19a4f126bb`

This report does not claim production release, live provider readiness, or human usability acceptance.

## Scenario Summary

| Scenario Set | Status | Scenario Count | PASS | HUMAN_REQUIRED | Blocking |
|---|---:|---:|---:|---:|---:|
| `vs0-evux` | success | 14 | 12 | 2 | 0 |
| `vs0-evux-governance` | success | 16 | 14 | 2 | 0 |

## Governance Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| VS0-GOV-001 | MUST_PASS | PASS | EVUX frozen matrix is split from `VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_VERIFICATION_MATRIX.csv`; EVUX AI rows are PASS and human rows remain HUMAN_REQUIRED. |
| VS0-GOV-002 | MUST_PASS | PASS | `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_CONTRACT.md` defines criteria and routes status to reports/matrices. |
| VS0-GOV-003 | MUST_PASS | PASS | `reports/scenario/vs0-evux-2026-06-13.json` metadata has `verified_base_tree_hash`, `verified_source_worktree_hash`, `dirty_paths`, `final_commit`, and no ambiguous `verified_tree_hash`. |
| VS0-GOV-004 | MUST_PASS | PASS | Source snapshot hash excludes generated evidence paths and lists source/doc dirty paths. |
| VS0-GOV-005 | MUST_PASS | PASS | `command-transcript.json` has 8 commands, exit codes, timeout flags, elapsed seconds, and stdout/stderr tails. |
| VS0-GOV-006 | MUST_PASS | PASS | Release manifest includes `command_transcript` with `present=true`, bytes, and matching sha256. |
| VS0-GOV-007 | MUST_PASS | PASS | Release manifest scenario report hash matches `reports/scenario/vs0-evux-2026-06-13.json`. |
| VS0-GOV-008 | MUST_PASS | PASS | EVUX report and release manifest claim local/mock evidence only; production/live/human claims remain false or HUMAN_REQUIRED. |
| VS0-GOV-009 | MUST_PASS | PASS | Post-commit rollup records commit `349dea7`, tree hash `f12b9393f957a72f1eb26af252b92f19a4f126bb`, artifact hashes, and clean pre-rollup worktree. |
| VS0-GOV-R01 | REGRESSION_GUARD | PASS | Existing EVUX behavior report remains `12 PASS / 2 HUMAN_REQUIRED / 0 blocking`. |
| VS0-GOV-R02 | REGRESSION_GUARD | PASS | Release command transcript records successful `verify-local-fast`, `verify-vs0-runtime`, `verify-vs0-acceptance`, and EVUX candidate gate. |
| VS0-GOV-R03 | REGRESSION_GUARD | PASS | Browser proof has `status=PASS`, `clean_browser_exit=true`, `chrome_exit_code=0`, and `chrome_timeout=false`. |
| VS0-GOV-R04 | REGRESSION_GUARD | PASS | `production_release_ready=false`, `live_connector_ready=false`, `human_usability_accepted=false`; human rows preserved. |
| VS0-GOV-R05 | REGRESSION_GUARD | PASS | No dependency lockfile or production dependency manifest changed. |
| VS0-GOV-H01 | HUMAN_REQUIRED | HUMAN_REQUIRED | JiYong/Tars must complete operator UI walkthrough and record accept/reject evidence. |
| VS0-GOV-H02 | HUMAN_REQUIRED | HUMAN_REQUIRED | Human-approved live ConnectorHub/provider proof requires credentials and redacted transcript/audit refs. |

## Command Evidence

Passed:

```sh
python3 -m py_compile packages/cornerstone_cli/acceptance.py packages/cornerstone_cli/main.py packages/cornerstone_cli/scenarios.py
scripts/verify_sot_docs.sh
make verify-vs0-evux
PATH="$PWD:$PATH" cornerstone release evidence collect --scope vs0-evux --json
PATH="$PWD:$PATH" cornerstone release evidence finalize --scope vs0-evux --json
PATH="$PWD:$PATH" cornerstone scenario verify vs0-evux-governance --json --output reports/scenario/vs0-evux-governance-2026-06-14.json
```

Observed release finalization:

```json
{
  "status": "success",
  "artifact_count": 17,
  "final_commit": "349dea7",
  "final_tree_hash": "f12b9393f957a72f1eb26af252b92f19a4f126bb",
  "missing_required": [],
  "worktree_dirty_before_rollup": false
}
```

Observed governance summary:

```json
{
  "status": "success",
  "summary": {
    "scenario_count": 16,
    "pass": 14,
    "human_required": 2,
    "blocking": 0,
    "product_feature_claims": "LOCAL_VS0_EVUX_GOVERNANCE_READY_PRODUCTION_NOT_READY"
  }
}
```

Release command transcript:

```json
{
  "command_count": 8,
  "pass": 8,
  "blocking": 0,
  "blocking_commands": []
}
```

Negative evidence:

```json
{
  "real_external_http_calls": 0,
  "production_release_overclaim": 0,
  "live_connector_claim_without_human_evidence": 0,
  "human_usability_claim_without_human_evidence": 0,
  "browser_timeout_marked_pass": 0
}
```

## Model And LLM Boundary

The governance PASS decision is deterministic. It uses fixture artifacts, mocked connector/action evidence, CLI transcripts, browser proof, hashes, and audit/policy records. It does not use an LLM as the PASS judge.

Local Ollama availability was checked after the governance pass:

```text
ollama list
nemotron3:33b
qwen3-embedding:0.6b
qwen3.6:27b
```

This confirms a local semantic-smoke backend is available, but no `vs0-llm` scenario is part of this governance gate.

## Changed Files

- `packages/cornerstone_cli/acceptance.py`
- `packages/cornerstone_cli/main.py`
- `packages/cornerstone_cli/scenarios.py`
- `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_CONTRACT.md`
- `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_MATRIX.csv`
- `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_VERIFICATION_MATRIX.csv`
- `docs/verification-reports/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_REPORT_2026-06-13.md`
- `docs/verification-reports/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_REPORT_2026-06-14.md`
- `README.md`, `AGENTS.md`, `docs/sot/README.md`, `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`, `docs/sot/sot_manifest.yaml`
- Refreshed evidence under `reports/scenario/`, `reports/browser/`, `reports/quickstart/`, and `reports/release/`

## Gaps And Risks

- Human usability acceptance remains `HUMAN_REQUIRED`.
- Live ConnectorHub/provider proof remains `HUMAN_REQUIRED`.
- Production release readiness remains false by design.
- The post-commit rollup records the finalized implementation/evidence commit `349dea7`; this report and governance matrix are follow-up documentation/evidence closure for that finalized base.

## Verdict

- AI-verifiable governance scope: PASS.
- Human/release gate: needs-human-verification for usability and live-provider proof.
- Release boundary: local VS0 EVUX evidence governance clean; production release not ready.

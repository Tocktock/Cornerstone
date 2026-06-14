# CornerStone VS0 EVUX Clean Sign-off Governance Contract

**Date:** 2026-06-14
**Owner:** JiYong / Tars
**Status:** Frozen task-scoped scenario contract. Criteria are status-neutral; current verification status belongs in the governance matrix, scenario report, release manifest, and verification report.

## Feature / Task

`VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE`

## Goal

Make the existing local VS0 EVUX milestone cleanly sign-offable by fixing evidence-governance inconsistencies without expanding product scope.

The future implementation must preserve the already verified local EVUX behavior:

```text
Artifact
-> Search
-> Evidence Bundle
-> Claim
-> Action Card dry-run
-> approval
-> local/mock execution
-> Audit timeline
```

The work is complete only when the repository has consistent, reviewable evidence showing:

```text
local VS0 EVUX: PASS
production release: false / not claimed
live provider: HUMAN_REQUIRED
human usability acceptance: HUMAN_REQUIRED
```

## Purpose

This is not a new feature milestone. It is a clean sign-off and handoff hardening task for the existing local VS0 EVUX milestone.

It freezes the scenario rows required to close these review blockers:

1. EVUX matrix/report status mismatch.
2. Ambiguous dirty-worktree and tree-hash verification metadata.
3. Release package command evidence that lists commands without exit-code transcripts.
4. Optional post-commit rollup evidence linking final commit to generated evidence artifacts when commit/push is in scope.

## Relationship To Existing Scenario Authority

This contract is subordinate to the product SoT, the full 206-scenario standard, the CLI-native-first contract, and the local verification plane.

It extends the current `VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP` evidence surface with governance and sign-off criteria. It does not replace the EVUX product-loop scenarios, and it must not be used to claim production release, live-provider readiness, or human usability acceptance.

Scenario contracts define criteria. Current `PASS`, `FAIL`, `NOT_VERIFIED`, `NOT_RUN`, and `HUMAN_REQUIRED` status belongs in scenario reports, verification reports, release manifests, and machine-readable matrices with clear semantics.

## Success Criteria

The implementation report may claim `VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE` only when:

- every AI-verifiable `VS0-GOV-*` MUST_PASS and REGRESSION_GUARD row is `PASS` with concrete evidence;
- the EVUX matrix, scenario report, implementation report, release manifest, and command evidence do not contradict each other;
- verification metadata distinguishes verified base commit/tree state, dirty source snapshot, final commit when available, and pre/post-commit report generation;
- dirty-worktree verification includes a deterministic hash over the relevant source/doc snapshot and the path list used for that hash;
- release evidence includes compact command transcripts with exit codes, timeout flags, and stdout/stderr tails;
- release evidence is generated from final scenario report bytes, not placeholder or provisional report bytes;
- the final report wording is no stronger than the evidence;
- `real_external_http_calls=0` remains true for the local/mock path;
- no production, live-provider, or human-usability acceptance is overclaimed;
- `VS0-GOV-H01` and `VS0-GOV-H02` remain `HUMAN_REQUIRED` until named human evidence exists.

## Constraints

### Product And UX

- Do not redesign the UI.
- Do not add a new product flow.
- Preserve the EVUX browser workflow and one-product CornerStone language.
- Do not claim human usability acceptance.

### Data And State

- Local file-backed runtime is acceptable for this cleanup.
- Do not migrate to Postgres, RLS, or OPA in this task.
- Generated reports may be updated only when they are part of the evidence package.
- Do not store secrets or credentials in reports.

### Permission And Safety

- No live provider credentials.
- No real external connector writeback.
- `real_external_http_calls` must remain `0`.
- No destructive action, release tag, production deployment, or irreversible migration.

### Compatibility And Format

- Preserve existing CLI commands unless a small additive command is needed.
- Preserve `--json` machine-readable output.
- Keep scenario contracts status-neutral where possible.
- Reports and verification matrices must not contradict each other.

### Operational And Environment

- Local Chrome dependency may remain for EVUX browser proof.
- If Chrome is unavailable, browser proof must be `NOT_VERIFIED`, `PARTIAL`, or `FAIL`, not clean `PASS`.
- No new production dependency without explicit approval.

## Assumptions

- The current EVUX behavioral implementation is mostly correct and should not be rewritten.
- This task is about evidence correctness, not feature expansion.
- Future implementation may update docs, reports, matrix files, release evidence generation, and verification helpers.
- If the implementation agent cannot commit/push in its environment, it must still produce accurate pre-commit metadata and mark post-commit proof as `NOT_RUN` rather than pretending final commit verification exists.

## Out Of Scope

- VS-1 ontology auto-suggest/promote.
- Production DB/storage migration.
- Postgres/RLS/OPA implementation.
- Full ConnectorHub live provider verification.
- Real external writeback.
- New UX product surface beyond evidence cleanup.
- Human usability acceptance.
- Security deep-dive unless a P0 issue is found.
- Broad refactor of runtime, scenario verifier, or UI.

## CLI Parity

- Command group: `cornerstone scenario|quickstart|release`
- Scenario verification command: `cornerstone scenario verify vs0-evux --json --output reports/scenario/vs0-evux-YYYY-MM-DD.json`
- Scenario gate command: `cornerstone scenario gate reports/scenario/vs0-evux-YYYY-MM-DD.json --json`
- Quickstart command: `cornerstone quickstart verify vs0-evux --json --output reports/quickstart/vs0-evux-quickstart.json`
- Release evidence command: `cornerstone release evidence collect --scope vs0-evux --json`
- Optional post-commit finalization command: `cornerstone release evidence finalize --scope vs0-evux --json`
- JSON schema path: scenario report, quickstart report, release manifest, command transcript, and optional post-commit rollup schemas are implementation-owned and must be referenced from the final verification report.
- Exit codes covered: success, verification gap, browser timeout/partial proof, policy/safety denial, missing artifact, and runtime failure.
- Workspace/namespace scope: local VS0 fixture workspace only unless the final report explicitly documents another local namespace.
- Dry-run behavior: external or risky action remains local/mock only; real writeback remains out of scope.
- Evidence refs emitted: scenario report, browser proof, quickstart transcript, command transcript, release manifest, implementation report, and audit refs where applicable.
- Audit refs emitted: artifact/search/evidence/claim/action/policy/approval/execution/audit refs from the existing EVUX proof path.
- Same backend path evidence: future implementation must show CLI/API/UI evidence use the same Artifact, Evidence, Claim, Action, Policy, and Audit boundaries as the EVUX milestone.
- Governance scenario verifier: `cornerstone scenario verify vs0-evux-governance --json --output reports/scenario/vs0-evux-governance-YYYY-MM-DD.json`
- CLI status: governed by scenario reports and verification reports, not by this contract prose.

## Scenario Table

Total task-scoped scenarios: **16**.

| ID | Type | Expected Result | Verification Method | Evidence Required | Owner |
|---|---|---|---|---|---|
| VS0-GOV-001 | MUST_PASS | EVUX matrix no longer contradicts EVUX report. | Inspect matrix and scenario report. | Either current matrix AI rows show `PASS`, or files are split into `FREEZE_MATRIX` and `VERIFICATION_MATRIX` with clear semantics. | AI |
| VS0-GOV-002 | MUST_PASS | Scenario contract remains status-neutral. | Source review. | Contract defines criteria; current `PASS`/`FAIL` status lives in scenario and verification reports only. | AI |
| VS0-GOV-003 | MUST_PASS | Verification metadata accurately represents dirty-worktree vs committed tree state. | Run EVUX verifier and inspect `verification_metadata`. | Clear fields such as `verified_base_tree_hash`, `verified_source_worktree_hash`, `dirty_paths`, `final_commit`, and `report_generated_before_commit`; no misleading `verified_tree_hash`. | AI |
| VS0-GOV-004 | MUST_PASS | If worktree is dirty during verification, the verified source snapshot is hashable and reproducible. | Run metadata helper or EVUX verifier. | Deterministic hash over relevant source/doc dirty paths, excluding self-referential generated output, plus path list used for the hash. | AI |
| VS0-GOV-005 | MUST_PASS | Final release evidence includes a compact command transcript with exit codes. | Inspect release package. | `reports/release/vs0-evux-YYYY-MM-DD/command-transcript.json` or equivalent includes command, start/end or elapsed time, exit code, timed_out, and stdout/stderr tail. | AI |
| VS0-GOV-006 | MUST_PASS | Release manifest includes and hashes command transcript evidence. | Inspect manifest. | Manifest has required artifact entry for command transcript, with path, bytes, sha256, and `present=true`. | AI |
| VS0-GOV-007 | MUST_PASS | Release evidence package is generated from final scenario report bytes, not placeholder or provisional report bytes. | Run scenario verify with output, then release collect/check. | Manifest hash for scenario report matches the committed/generated `reports/scenario/vs0-evux-YYYY-MM-DD.json`. | AI |
| VS0-GOV-008 | MUST_PASS | Final report wording is no stronger than evidence. | Source/report review. | Report says local VS0 EVUX evidence is clean; production release, live provider, and human usability remain unclaimed. | AI |
| VS0-GOV-009 | MUST_PASS | Post-commit rollup is present if commit/push is in scope. | Inspect post-commit artifact. | `post_commit_rollup.json` or report section records final commit, final tree hash, evidence artifact hashes, and relationship to verified base/worktree snapshot. If commit is not in scope, this row may be `NOT_RUN` and final verdict cannot be clean sign-off. | AI |
| VS0-GOV-R01 | REGRESSION_GUARD | Existing EVUX behavior still passes. | `make verify-vs0-evux`. | Exit code 0 and scenario summary `blocking=0`, `pass=12`, `human_required=2`. | AI |
| VS0-GOV-R02 | REGRESSION_GUARD | Existing local gates still pass. | Run regression commands. | Exit-code transcript for `make verify-local-fast`, `make verify-vs0-runtime`, and `make verify-vs0-acceptance`. | AI |
| VS0-GOV-R03 | REGRESSION_GUARD | Browser timeout cannot be marked clean PASS. | Inspect browser proof and/or targeted test. | Clean PASS requires `clean_browser_exit=true`, `chrome_exit_code=0`, and `chrome_timeout=false`; timeout path must be `PARTIAL`, `FAIL`, or `NOT_VERIFIED`. | AI |
| VS0-GOV-R04 | REGRESSION_GUARD | No production/live-provider/human UX overclaim. | Inspect scenario report and release manifest. | `production_release_ready=false`, `live_connector_ready=false`, `human_usability_accepted=false`, and human-required rows preserved. | AI |
| VS0-GOV-R05 | REGRESSION_GUARD | No new production dependency or broad architecture migration. | `git diff` and manifest review. | No new dependency lockfiles or production service migration unless explicitly approved. | AI |
| VS0-GOV-H01 | HUMAN_REQUIRED | Human usability acceptance. | Human walkthrough. | JiYong/Tars records accept/reject with screenshots/recording or issue list. | Human |
| VS0-GOV-H02 | HUMAN_REQUIRED | Live ConnectorHub/provider proof. | Human-approved live provider test. | Redacted provider transcript, approval record, action result, and audit refs. | Human |

## Required Local Verification

Future implementation should run these commands, or report exactly why any command could not run:

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-evux --json --output reports/scenario/vs0-evux-YYYY-MM-DD.json
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs0-evux-YYYY-MM-DD.json --json
PATH="$PWD:$PATH" cornerstone quickstart verify vs0-evux --json --output reports/quickstart/vs0-evux-quickstart.json
PATH="$PWD:$PATH" cornerstone release evidence collect --scope vs0-evux --json
make verify-local-fast
make verify-vs0-runtime
make verify-vs0-acceptance
make verify-vs0-evux
```

If implementation adds a finalization command, use:

```sh
PATH="$PWD:$PATH" cornerstone release evidence finalize --scope vs0-evux --json
```

or an equivalent script. The final report must name the exact command used.

## Definition Of All Scenarios Passing

All scenarios pass when:

1. Every AI-owned `VS0-GOV-*` MUST_PASS row is `PASS`.
2. Every AI-owned `VS0-GOV-*` REGRESSION_GUARD row is `PASS`.
3. No AI-owned row is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.
4. Human-required rows remain `HUMAN_REQUIRED` with clear required action, expected evidence, and release impact.
5. Scenario report, matrix, release manifest, and verification report do not contradict each other.
6. Evidence package contains concrete command/browser/API/report artifacts with hashes.
7. Final verdict does not claim production release, live-provider readiness, or human usability acceptance.

If any AI-owned row is not `PASS`, the future implementation agent must not say "done." It must provide root cause, failed layer, fix/blocker, and re-verification plan.

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS0-GOV-H01 | Human usability is subjective and requires real operator judgment. | JiYong/Tars completes the local UI walkthrough and records accept or reject. | Acceptance note plus screenshots/recording or issue list. | Blocks operator-accepted product claim, not AI-verifiable local governance gate. |
| VS0-GOV-H02 | Live provider verification requires credentials and may mutate third-party state. | Human approves and performs live ConnectorHub/provider dry-run or execution later. | Redacted provider transcript, written approval, execution result, and audit refs. | Blocks live-provider production release, not local governance proof. |

## When To Stop Future Implementation

Stop immediately when one of these is true:

```text
A. All AI-verifiable scenarios above are PASS with concrete evidence.
B. A scenario fails and the root cause is not fixed.
C. A required check cannot be run safely or locally.
D. The task would require new production dependency, production migration, live connector credentials, destructive action, or human approval.
E. The implementation starts drifting into UI redesign, VS-1 ontology, live provider work, or production architecture.
```

If stopped under B through E, the final verdict must be:

```text
AI-verifiable scope: needs-follow-up or blocked
Human/release gate: needs-human-verification or blocked
```

## Done Means

This scenario contract is frozen when this document and its machine-readable matrix are discoverable from the SoT/README.

Implementation is done only when every AI-verifiable `VS0-GOV-*` MUST_PASS and REGRESSION_GUARD scenario is `PASS` with concrete evidence, no `FAIL`, `NOT_VERIFIED`, or `NOT_RUN` remains for AI rows, and human-only rows list required human action, expected evidence, and release impact.

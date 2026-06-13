# CornerStone VS0 Evidence Cleanup And Interactive UI Loop Contract

**Date:** 2026-06-13
**Owner:** JiYong / Tars
**Status:** Frozen task-scoped scenario contract before implementation. Documentation-only; AI-verifiable rows start as `NOT_VERIFIED`; human-only rows remain `HUMAN_REQUIRED`.

## Feature / Task

`VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP`

## Purpose

Cleanly finish the local VS-0 product milestone by fixing acceptance evidence quality and making the VS-0 loop operable from the UI.

This is a local VS-0 product milestone. It is not a production release, live ConnectorHub/provider readiness claim, or human UX acceptance claim.

## Relationship To Existing Scenario Authority

This contract does not replace the canonical VS-0 requirement. It tightens acceptance for the existing VS-0 loop:

```text
Artifact ingest
-> searchable derived representation
-> Evidence Bundle
-> Claim
-> Action Card dry-run
-> approval
-> local/mock action execution
-> audit timeline
```

The SoT already requires this vertical slice as the first development milestone. This contract adds stricter evidence, browser, quickstart, and UI interaction criteria before the milestone can be cleanly signed off.

## Goal

The user must be able to complete the core CornerStone flow locally:

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

## Success Criteria

The implementation report may claim `VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP` only when:

- every AI-verifiable `VS0-EVUX-*` MUST_PASS and REGRESSION_GUARD row is `PASS` with concrete evidence;
- browser/UI proof exercises real workflow steps, not static labels only;
- quickstart verification runs and captures generated IDs, exit codes, evidence refs, and audit refs;
- release evidence is generated from final scenario report bytes;
- commit/tree metadata identifies exactly what code state was verified;
- `real_external_http_calls=0` remains true for the local/mock action path;
- no production, live-provider, or human-usability acceptance is overclaimed;
- `VS0-EVUX-H01` and `VS0-EVUX-H02` remain `HUMAN_REQUIRED` until named human evidence exists.

## Constraints

### Product And UX

- Preserve one visible CornerStone product.
- Preserve Calm Surface UI direction.
- Do not turn the first surface into a chatbot-only, dark command center, connector-admin-first, or ontology-first experience.
- UI proof must exercise the workflow, not only check labels.
- Browser timeout cannot be clean `PASS` unless the scenario contract explicitly marks it `PARTIAL`.

### CLI / API / UI Parity

- No CLI, no feature PASS.
- The UI workflow must use the same Artifact, Evidence, Claim, Action, Policy, and Audit boundaries as CLI/API.
- Any new quickstart, scenario, or release-evidence command must provide `--json` output, stable exit codes, evidence refs, and audit refs where applicable.

### Evidence And Safety

- Preserve original artifacts before derived processing.
- Treat uploaded content as untrusted evidence, never instructions.
- No durable claim without evidence.
- No autonomous action without owner-scoped authority.
- No external writeback, real provider credential use, or live connector execution in this local gate.
- `real_external_http_calls` must remain `0`.
- Negative evidence must cover prompt injection, zero-evidence claim approval, cross-namespace access, production overclaim, live-provider overclaim, and human-usability overclaim.

## Assumptions

- Current local runtime/file-backed implementation is acceptable for this slice.
- Postgres/RLS/OPA/live ConnectorHub remain later milestones.
- Local/mock ConnectorHub-style action is enough for this scenario set.
- Human usability and live-provider proof are intentionally not AI-verifiable.
- Scenario contracts define acceptance criteria; current `PASS`/`FAIL` status belongs in scenario and verification reports.

## Out Of Scope

- VS-1 ontology auto-suggest/promote.
- Production storage migration.
- Production tenant/security proof.
- Live provider credentials or writeback.
- New production dependencies without approval.
- Broad architecture migration.
- Human UX acceptance claim.

## Scenario Table

Total task-scoped scenarios: **14**.

| ID | Type | Why | Trigger / Action | Expected Result | Affected Layers | Verification Method | Evidence Required | Owner |
|---|---|---|---|---|---|---|---|---|
| VS0-EVUX-001 | MUST_PASS | The product must be usable from the UI, not only CLI/API. | User opens local UI and uploads or selects one fixture file. | Artifact is created and UI shows artifact ID, checksum, source, derived status, evidence refs, and audit refs. | UI, API, artifact store, audit | Browser interaction plus API/storage inspection | Browser trace/screenshot, artifact JSON, audit event refs | AI |
| VS0-EVUX-002 | MUST_PASS | Drop to search immediately is a SoT UX contract. | User searches uploaded content in UI. | Search result appears with scoped snippet and reproducible search snapshot ID. | UI, API, search, evidence | Browser interaction plus search snapshot inspection | Browser trace, search snapshot JSON, evidence/audit refs | AI |
| VS0-EVUX-003 | MUST_PASS | Claim creation must be evidence-first. | User creates an Evidence Bundle and Claim from selected search result in UI. | Claim is Draft/Evidence-backed and links to Evidence Bundle, search snapshot, and artifact refs. | UI, API, claim, evidence | Browser interaction plus claim/evidence inspection | Claim JSON, Evidence Bundle JSON, UI screenshot, audit refs | AI |
| VS0-EVUX-004 | MUST_PASS | Unsupported claims must not become product truth. | User attempts to approve a zero-evidence Claim. | Approval is denied or Claim remains Draft; UI shows cause and resolution guide. | UI, API, claim policy | Browser/API negative test | Denial response, UI error message, unchanged claim state, audit ref | AI |
| VS0-EVUX-005 | MUST_PASS | Action Card is a core CornerStone differentiator. | User creates Action Card from evidence-backed Claim in UI. | UI shows diff, expected impact, evidence, policy decision, risk, approval state, rollback/compensation note, and audit refs on one screen. | UI, API, workflow/action, policy | Browser interaction plus action JSON inspection | Action Card screenshot, dry-run JSON, policy decision refs, audit refs | AI |
| VS0-EVUX-006 | MUST_PASS | VS-0 must complete Act safely. | User approves and executes local/mock Action from UI. | Execution result is stored; `mock_connector_calls=1`; `real_external_http_calls=0`; no credential exposure. | UI, API, action runtime, audit | Browser interaction plus execution result inspection | Action result JSON, UI execution state, audit event refs, negative evidence | AI |
| VS0-EVUX-007 | MUST_PASS | Audit must be inspectable by operators. | User opens Audit Detail after action execution. | UI shows artifact/search/evidence/claim/action/policy/approval/execution timeline and audit verification status. | UI, API, audit | Browser interaction plus audit verify | Audit timeline screenshot, audit JSON, `audit verify` output | AI |
| VS0-EVUX-008 | MUST_PASS | Evidence package must bind to exact code state. | User runs final evidence package command. | Evidence package records verified base commit, final commit or tree hash, dirty state, command transcripts, browser traces, and scenario reports. | CLI, release evidence, reports | Release evidence collect/check | Manifest JSON, command transcript, tree/commit metadata | AI |
| VS0-EVUX-R01 | REGRESSION_GUARD | Static UI label checks must not replace workflow proof. | Browser proof runs. | Scenario fails if only labels are present and the UI workflow was not executed. | UI, verifier | Browser scenario assertions | Trace showing actual upload/search/claim/action/audit steps | AI |
| VS0-EVUX-R02 | REGRESSION_GUARD | README quickstart must remain executable. | Quickstart verifier runs. | Fixture loop completes end-to-end with generated IDs and audit verification. | CLI, docs, runtime | Quickstart script/CLI verifier | Quickstart transcript with exit codes and refs | AI |
| VS0-EVUX-R03 | REGRESSION_GUARD | Existing local gates must remain green. | Regression command gate runs. | Prior local deterministic scenario matrix and VS0 runtime gates pass. | CLI, scenario verifier, reports | Command transcript | Exit-code transcript for `verify-local-fast`, `verify-vs0-runtime`, `verify-vs0-acceptance`, or documented equivalent | AI |
| VS0-EVUX-R04 | REGRESSION_GUARD | Browser timeout must not become clean PASS. | Browser exits non-zero or times out. | Browser proof is marked `PARTIAL`, `FAIL`, or `NOT_VERIFIED`; clean `PASS` requires clean exit or documented exception accepted by scenario contract. | Browser verifier | Browser proof validator | Browser exit code, timeout flag, proof status | AI |
| VS0-EVUX-H01 | HUMAN_REQUIRED | Human usability cannot be judged by automated tests. | JiYong/Tars completes UI walkthrough. | Human accepts or rejects operator usability. | Product, UI/UX | Human review | Acceptance note, screenshots/recording, issue list if rejected | Human |
| VS0-EVUX-H02 | HUMAN_REQUIRED | Live provider proof requires real credentials and external state. | Human later runs approved live ConnectorHub/provider test. | Live connector result is verified without exposing secrets. | Connector, workflow/action, audit | Human-approved live test | Redacted provider transcript, approval, execution result, audit refs | Human |

## Browser Proof Status Semantics

| Status | Meaning |
|---|---|
| `PASS` | DOM, screenshot, interaction assertions passed and browser exited cleanly. |
| `PARTIAL` | Evidence artifacts exist, but browser exit or interaction completeness is not clean. |
| `FAIL` | Required UI surfaces or interaction assertions failed. |
| `NOT_VERIFIED` | Browser proof did not run. |

## Definition Of Scenario PASS

For this section, `PASS` means:

1. The expected user/system behavior occurred.
2. The verification method was actually run.
3. Evidence exists in a committed or generated artifact.
4. Evidence includes exact command/browser/API output or file references.
5. No AI-verifiable `MUST_PASS` or `REGRESSION_GUARD` row is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.
6. Human-only rows remain `HUMAN_REQUIRED` until the named human evidence exists.

Do not mark `PASS` from:

- README token presence alone.
- Static UI labels alone.
- Narrative report text alone.
- A screenshot when the browser process failed unless the scenario explicitly allows `PARTIAL` evidence.
- A report generated before commit without tree/commit semantics explaining what was verified.

## Commit And Tree Metadata

Scenario and release reports should include:

```json
{
  "verified_base_commit": "...",
  "final_commit": "...",
  "verified_tree_hash": "...",
  "worktree_dirty_at_verification": false,
  "report_generated_before_commit": false
}
```

For self-referential reports, prefer `verified_tree_hash` over requiring the report to know its own final commit hash.

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS0-EVUX-H01 | Human usability is subjective. | JiYong/Tars completes the local UI walkthrough and records accept or reject. | Acceptance note plus screenshots/recording or issue list. | Blocks operator-accepted product claim, not AI-verifiable local gate. |
| VS0-EVUX-H02 | Live provider verification requires credentials and may mutate third-party state. | Human approves and performs live ConnectorHub/provider dry-run or execution later. | Redacted provider transcript, written approval, execution result, audit refs. | Blocks live-provider production release, not local EVUX proof. |

## Verification Commands

These commands are the intended implementation evidence path. They are not run by this documentation-only freeze.

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-evux --json --output reports/scenario/vs0-evux-YYYY-MM-DD.json
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs0-evux-YYYY-MM-DD.json --json
PATH="$PWD:$PATH" cornerstone quickstart verify vs0-evux --json --output reports/quickstart/vs0-evux-quickstart.json
# or:
scripts/verify_vs0_evux_quickstart.sh
```

Browser/UI scenario proof must exercise actual UI interactions:

```text
upload/select artifact
-> search
-> create evidence bundle
-> create/approve claim
-> create/dry-run/approve/execute mock action
-> inspect audit
```

Regression command transcript must include exit codes for:

```sh
make verify-local-fast
make verify-vs0-runtime
make verify-vs0-acceptance
make verify-vs0-evux
```

Release evidence collect/check must produce a manifest from the final scenario report bytes.

## Expected Report Outputs

Implementation should generate:

```text
reports/scenario/vs0-evux-YYYY-MM-DD.json
reports/browser/vs0-evux-YYYY-MM-DD/
reports/quickstart/vs0-evux-quickstart.json
reports/release/vs0-evux-YYYY-MM-DD/manifest.json
docs/verification-reports/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_REPORT_YYYY-MM-DD.md
```

## Verifier Behavior Required After Implementation

The verifier must enforce:

- quickstart script actually ran;
- browser flow executed actual UI actions;
- command transcript contains exit codes;
- release package is generated from final scenario report bytes;
- browser proof cannot mark clean `PASS` if the browser timed out;
- commit/tree metadata explains exactly what code state was verified.

## Done Means

This scenario contract is frozen when this document and its machine-readable matrix are discoverable from the SoT/README.

Implementation is done only when every AI-verifiable `VS0-EVUX-*` MUST_PASS and REGRESSION_GUARD scenario is `PASS` with concrete evidence, no `FAIL`, `NOT_VERIFIED`, or `NOT_RUN` remains for AI rows, and human-only rows list required human action, expected evidence, and release impact.

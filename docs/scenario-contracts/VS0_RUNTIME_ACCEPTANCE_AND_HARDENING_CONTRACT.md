# CornerStone VS0 Runtime Acceptance And Hardening Contract

**Date:** 2026-06-11
**Owner:** JiYong / Tars
**Status:** Frozen task-scoped scenario contract; status-neutral acceptance criteria. Current `PASS`, `FAIL`, `NOT_VERIFIED`, and `HUMAN_REQUIRED` results live only in scenario reports and verification reports.

## Purpose

This contract freezes the next scenario set after the accepted local VS0 Product Runtime Readiness slice.

The previous runtime slice proves that a local deterministic VS0 loop can run across CLI, API, and a minimal UI surface. This contract does not add product code. It defines the acceptance gate for turning that local runtime proof into an operator-reviewable local release candidate without overclaiming production release, live connector readiness, or human usability acceptance.

Scenario contracts define acceptance criteria. Current `PASS`, `FAIL`, `NOT_VERIFIED`, and `HUMAN_REQUIRED` status belongs only in scenario reports and verification reports.

Operator-reviewable local release candidate means automated local evidence is sufficient for a human operator to review. Operator-accepted local release candidate remains blocked until the human walkthrough is completed and recorded.

## Relationship To Existing Scenario Authority

- Canonical long-term product standard remains `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` with 206 `CS-*` scenarios.
- VS-0 implementation subset remains `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` with 58 scenarios.
- The accepted local runtime readiness contract remains `docs/scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md`.
- This contract is a task-scoped acceptance and hardening overlay for the local VS0 runtime.
- Scenario `PASS` still requires deterministic evidence and native CLI parity.
- Human-only, credentialed, external-provider, production, and subjective checks must remain `HUMAN_REQUIRED`.

## Goal

Turn local VS0 runtime readiness into an operator-reviewable local release candidate.

The acceptance gate must close these gaps:

1. real browser UI proof;
2. human usability walkthrough path;
3. clearer readiness and connector-call semantics;
4. release-facing evidence package;
5. no production, live-provider, or human-acceptance overclaim.

## Success Criteria

An implementation report may claim `VS0_RUNTIME_ACCEPTANCE_AND_HARDENING` only when:

- every AI-verifiable `VS0-ACC-*` MUST_PASS and REGRESSION_GUARD row is `PASS` with concrete evidence;
- `cornerstone ready --json` exposes last successful local runtime scenario evidence, commit, timestamp, and gate status;
- browser UI proof exists for the required Calm Surface runtime screens;
- the release-facing evidence package is reviewable without rerunning the full scenario suite;
- ambiguous connector-call wording is split so mock/expected connector calls cannot be mistaken for real external egress;
- production release readiness remains false until separate production/live-provider gates pass;
- `VS0-ACC-H01` and `VS0-ACC-H02` remain `HUMAN_REQUIRED` until a human supplies the required evidence.

## Acceptance Status Source

This contract is not a status ledger. The current acceptance result is determined by:

- `reports/scenario/vs0-runtime-acceptance-YYYY-MM-DD.json`;
- `docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_REPORT_YYYY-MM-DD.md`;
- release evidence manifests generated from the final scenario report bytes.

For self-referential release evidence, prefer `verified_tree_hash` over requiring a report to know its own final commit hash. Reports should identify `verified_base_commit`, `final_commit` or `verified_tree_hash`, `worktree_dirty_at_verification`, and `report_generated_before_commit`.

## Constraints

### Documentation-Only Freeze

The scenario-freeze step did not change runtime code. The implementation step for this contract must keep runtime changes narrowly scoped to acceptance/hardening evidence and must not widen into VS-1 features or live providers.

### Product And UX

- Preserve one visible CornerStone product.
- Preserve Calm Surface UI direction.
- Do not turn the first surface into a chatbot-only, dark command center, connector-admin-first, or ontology-first experience.
- Human usability acceptance is not AI-verifiable and must not be marked `PASS` by an agent.

### CLI / API / UI Parity

- No CLI, no feature PASS.
- Any new readiness, evidence-package, or acceptance command must have `--json` output, stable exit codes, evidence refs, and audit refs where applicable.
- Browser/UI proof can supplement the CLI gate, but cannot replace CLI-native verification.

### Evidence And Safety

- Preserve Artifact, Evidence, Audit, Action, Policy, and Namespace safety boundaries.
- Do not use real provider credentials for this local acceptance gate.
- Do not perform live external writeback.
- Do not claim production release readiness from local runtime evidence.
- Negative evidence must distinguish real external egress from local/mock connector simulation.

## Assumptions

- Commit `90c15b51627d9ea53b46fc0e891b7499058adecf` was accepted as local VS0 Product Runtime Readiness.
- The existing runtime report records 12 PASS, 2 HUMAN_REQUIRED, and 0 blocking AI-verifiable rows for `VS0-RT-*`.
- `production_release_ready=false` remains the correct release boundary.
- The next task should harden local acceptance evidence before moving to VS-1 ontology auto-suggest/promote.

## Out Of Scope

- Installing browser binaries or running external downloads.
- Real Gmail, Slack, Notion, GitHub, or other live provider execution.
- Live ConnectorHub/provider proof.
- Production tenant/security proof.
- Production release claim.
- VS-1 ontology auto-suggest/promote implementation.
- Broad full-suite feature expansion beyond local VS0 runtime acceptance and hardening.

## Checklist For Future Implementation

- [ ] Real browser UI proof captured for required screens.
- [ ] Readiness JSON includes last successful runtime scenario report path, timestamp, commit, and gate status.
- [ ] Ambiguous connector-call fields are split or clearly qualified.
- [ ] One-command quickstart verifies a local fixture loop from runtime start through audit.
- [ ] Release evidence bundle is generated and reviewable.
- [ ] Production/live-provider/human-acceptance overclaim remains blocked.
- [ ] Existing local deterministic scenario checks remain green.
- [ ] Human-required rows list required action and expected evidence.

## Scenario Table

Total task-scoped scenarios: **9**.

| ID | Type | Why | Required Behavior | Verification Method / Evidence Required |
|---|---|---|---|---|
| VS0-ACC-001 | MUST_PASS | Browser proof must show more than static HTML presence. | Browser proof covers Home/Ops Inbox, Artifact Viewer, Search, Claim Builder, Action Card, and Audit Detail. It separately reports DOM surface check, screenshot generation, production-overclaim absence, and browser clean exit. | Browser run against local runtime. Evidence: `browser-proof.json`, screenshot, DOM snapshot, clean-exit status. If browser times out, evidence status is `PARTIAL` or `NOT_VERIFIED`, not clean `PASS`. |
| VS0-ACC-002 | MUST_PASS | Readiness evidence must be tied to the exact verified code state. | `cornerstone ready --json` includes last successful runtime report, scenario status, gate status, timestamp, verified base commit, final committed revision or tree hash, and whether the report was generated pre-commit. | CLI readiness check and report inspection. Evidence: readiness JSON plus metadata fields `verified_base_commit`, `final_commit` or `verified_tree_hash`, `worktree_dirty_at_verification`, and `report_generated_before_commit`. |
| VS0-ACC-003 | MUST_PASS | Mock connector behavior must not look like real external egress. | Dry-run and execution evidence distinguish `expected_connector_calls`, `mock_connector_calls`, and `real_external_http_calls=0`. | CLI/API action dry-run and execute. Evidence: action dry-run JSON, action result JSON, negative evidence counters. |
| VS0-ACC-004 | MUST_PASS | Local acceptance must be repeatable, not only documented. | A quickstart verifier runs the local VS0 loop end-to-end from fixture ingest through audit verify. | Executable script or CLI command. Evidence: transcript with command list, generated IDs, exit codes, evidence refs, audit refs, elapsed time, final audit verification. |
| VS0-ACC-005 | MUST_PASS | Human review needs one coherent evidence package. | Release package is generated after the final scenario report bytes exist and includes scenario report, browser proof, quickstart transcript, command transcript, negative evidence, human-required rows, and manifest hashes. | Release evidence collection. Evidence: manifest with hashes/stable refs, no missing required artifacts, package generated from final report rather than placeholder/provisional report. |
| VS0-ACC-R01 | REGRESSION_GUARD | Local acceptance must not imply production readiness. | `production_release_ready=false`; live connector and human usability remain `HUMAN_REQUIRED`; local acceptance cannot unlock production release. | Readiness JSON, report, manifest, and negative evidence counters for production overclaim, live-provider overclaim, and human-usability overclaim. |
| VS0-ACC-R02 | REGRESSION_GUARD | Acceptance hardening must not regress earlier local deterministic gates. | `make verify-local-fast`, `make verify-vs0-runtime`, and `make verify-vs0-acceptance` or equivalent targeted commands are actually run and captured. | Command transcript artifact with command names, start/end time, exit codes, relevant stdout/stderr tail, and report refs. |
| VS0-ACC-H01 | HUMAN_REQUIRED | Live provider execution requires credentials and may mutate third-party systems. | Later approved live ConnectorHub/provider test with redacted transcript and audit refs. | Human approval, redacted live-provider transcript, provider/action result, audit refs. |
| VS0-ACC-H02 | HUMAN_REQUIRED | Usability acceptance is subjective and requires the owner/operator. | JiYong/Tars performs the VS0 runtime walkthrough and records accept/reject with notes. | Human acceptance note plus screenshots/recording or issue list. |

## Mapping To Existing Product Scenarios

| Acceptance Scenario | Existing Scenario Coverage |
|---|---|
| VS0-ACC-001 | `CS-PROD-001`, `CS-PROD-005`, `CS-PROD-006`, `CS-PROD-007`, `DS-S01`, `DS-S03`, `DS-S04`, `DS-S05`, `DS-S07` |
| VS0-ACC-002 | `CS-SEC-019`, `CS-SEC-020`, `CS-REG-020`, `CS-CLI-009` |
| VS0-ACC-003 | `CS-AUTO-007`, `CS-AUTO-008`, `CS-AUTO-011`, `CS-SEC-014`, `CS-REG-013` |
| VS0-ACC-004 | `CS-PROD-004`, `CS-PROD-005`, `CS-CLI-001`, `CS-CLI-004`, `CS-CLI-010` |
| VS0-ACC-005 | `CS-SEC-006`, `CS-SEC-019`, `CS-SEC-020`, `CS-REG-017`, `CS-REG-020` |
| VS0-ACC-R01 | `CS-SEC-019`, `CS-SEC-020`, `CS-REG-020` |
| VS0-ACC-R02 | `CS-REG-001`, `CS-REG-013`, `CS-REG-016`, `CS-REG-017`, `CS-REG-020` |

## CLI Parity

Implementation must satisfy CLI parity before any AI-verifiable `VS0-ACC-*` row can be marked `PASS` in a scenario or verification report.

| Scenario | Required CLI Commands | Required JSON / Evidence | Acceptance Status Source |
|---|---|---|---|
| VS0-ACC-001 | `cornerstone scenario verify vs0-runtime-acceptance --scenario VS0-ACC-001 --json` | browser trace refs, screenshot refs, DOM assertions, clean-exit or explicit partial status, production-overclaim assertion | Scenario report and verification report |
| VS0-ACC-002 | `cornerstone ready --json`; `cornerstone scenario verify vs0-runtime-acceptance --json` | last report path, timestamp, commit/tree metadata, gate status, scenario status | Scenario report and verification report |
| VS0-ACC-003 | `cornerstone action dry-run <action_id> --json`; `cornerstone action execute <action_id> --json` | unambiguous mock/expected/real external call counters | Scenario report and verification report |
| VS0-ACC-004 | `cornerstone quickstart verify vs0-runtime-acceptance --json` or executable quickstart script | transcript with command list, generated IDs, exit codes, evidence refs, audit refs, elapsed time | Scenario report and verification report |
| VS0-ACC-005 | `cornerstone release evidence collect --scope vs0-runtime-acceptance --json` | evidence bundle manifest, artifact refs, hashes, final report byte binding | Scenario report and verification report |
| VS0-ACC-R01 | `cornerstone ready --json`; `cornerstone scenario verify vs0-runtime-acceptance --scenario VS0-ACC-R01 --json` | production false, human-required rows preserved, negative evidence counters | Scenario report and verification report |
| VS0-ACC-R02 | `make verify-local-fast`; `make verify-vs0-runtime`; `make verify-vs0-acceptance` | command transcript with exit codes, stdout/stderr tail, scenario reports | Scenario report and verification report |

## Negative Evidence Required

Future reports must include zero-valued negative evidence counters for:

- `real_external_http_calls`;
- `unqualified_external_calls_in_release_report`;
- `production_release_overclaim`;
- `live_connector_claim_without_human_evidence`;
- `human_usability_claim_without_human_evidence`;
- `tool_calls_from_untrusted_artifact`;
- `action_cards_from_prompt_injection`;
- `cross_namespace_reads`;
- `zero_evidence_claim_approvals`;
- `audit_tamper_verify_failures`.

## Verification Commands For Future Implementation

These commands are required evidence inputs for implementation reports. A contract row is not `PASS` unless the relevant command actually ran and produced concrete evidence.

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-runtime-acceptance --json --output reports/scenario/vs0-runtime-acceptance-2026-06-11.json
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs0-runtime-acceptance-2026-06-11.json --json
PATH="$PWD:$PATH" cornerstone release evidence collect --scope vs0-runtime-acceptance --json
make verify-vs0-runtime
make verify-vs0-acceptance
make verify-local-fast
```

## Done Means

VS0 Runtime Acceptance And Hardening is locally done in a scenario or verification report when:

- every AI-verifiable `VS0-ACC-*` MUST_PASS and REGRESSION_GUARD scenario is `PASS`;
- every `PASS` has concrete evidence;
- no `FAIL`, `NOT_VERIFIED`, or `NOT_RUN` remains for AI-verifiable rows;
- human-required rows remain explicitly listed with required human action and expected evidence;
- production release readiness remains false unless a separate production/live-provider gate has real approval and evidence.

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS0-ACC-H01 | Live connector/provider verification needs credentials and may mutate third-party state. | Approve and perform live ConnectorHub/provider dry-run/execution later. | Redacted transcript, provider/action result, audit refs, written approval. | Blocks live-provider production release, not local runtime acceptance. |
| VS0-ACC-H02 | Usability acceptance is subjective. | Human operator walks through VS0 runtime and confirms whether the loop is understandable/useful. | Acceptance note plus screenshots/recording or issue list. | Blocks human product acceptance claim, not deterministic local acceptance checks. |

## Verdict Rule

This contract is accepted as a scenario contract when the criteria exist and are discoverable from the SoT/README. The implementation gate cannot be marked done in a report if any AI-verifiable `VS0-ACC-*` MUST_PASS or REGRESSION_GUARD scenario is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.

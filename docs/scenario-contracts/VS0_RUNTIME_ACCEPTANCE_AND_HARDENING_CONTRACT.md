# CornerStone VS0 Runtime Acceptance And Hardening Contract

**Date:** 2026-06-11
**Owner:** JiYong / Tars
**Status:** Frozen task-scoped scenario contract; local deterministic implementation target. Human-required rows remain human-required.

## Purpose

This contract freezes the next scenario set after the accepted local VS0 Product Runtime Readiness slice.

The previous runtime slice proves that a local deterministic VS0 loop can run across CLI, API, and a minimal UI surface. This contract does not add product code. It defines the next implementation gate: turn that local runtime proof into an operator-acceptable local release candidate without overclaiming production release, live connector readiness, or human usability acceptance.

## Relationship To Existing Scenario Authority

- Canonical long-term product standard remains `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` with 206 `CS-*` scenarios.
- VS-0 implementation subset remains `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` with 58 scenarios.
- The accepted local runtime readiness contract remains `docs/scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md`.
- This contract is a task-scoped acceptance and hardening overlay for the local VS0 runtime.
- Scenario `PASS` still requires deterministic evidence and native CLI parity.
- Human-only, credentialed, external-provider, production, and subjective checks must remain `HUMAN_REQUIRED`.

## Goal

Turn local VS0 runtime readiness into an operator-acceptable local release candidate.

The future implementation task must close these gaps:

1. real browser UI proof;
2. human usability walkthrough path;
3. clearer readiness and connector-call semantics;
4. release-facing evidence package;
5. no production, live-provider, or human-acceptance overclaim.

## Success Criteria

The next implementation task may claim `VS0_RUNTIME_ACCEPTANCE_AND_HARDENING` only when:

- every AI-verifiable `VS0-ACC-*` MUST_PASS and REGRESSION_GUARD row is `PASS` with concrete evidence;
- `cornerstone ready --json` exposes last successful local runtime scenario evidence, commit, timestamp, and gate status;
- browser UI proof exists for the required Calm Surface runtime screens;
- the release-facing evidence package is reviewable without rerunning the full scenario suite;
- ambiguous connector-call wording is split so mock/expected connector calls cannot be mistaken for real external egress;
- production release readiness remains false until separate production/live-provider gates pass;
- `VS0-ACC-H01` and `VS0-ACC-H02` remain `HUMAN_REQUIRED` until a human supplies the required evidence.

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

| ID | Type | Why | Required Behavior | Verification Method / Evidence Required | Verified Status |
|---|---|---|---|---|---|
| VS0-ACC-001 | MUST_PASS | Current UI proof is deterministic HTTP/assertion based; release-facing acceptance needs real browser proof. | Browser UI proof covers Home/Ops Inbox, Artifact Viewer, Search, Claim Builder, Action Card, and Audit Detail. | Browser screenshots or trace, route/viewport metadata, required text/surface assertions, and no production overclaim text. | PASS |
| VS0-ACC-002 | MUST_PASS | Readiness must not be only file-presence or static availability. | `cornerstone ready --json` includes last successful `vs0-product-runtime` scenario report path, timestamp, commit, scenario status, and gate status. | CLI transcript, JSON output, referenced scenario report, commit match, exit-code evidence. | PASS |
| VS0-ACC-003 | MUST_PASS | Mock connector behavior must not look like real external egress. | Release-facing action/dry-run/execution evidence separates expected/mock connector calls from real external HTTP calls. | JSON/schema transcript showing explicit fields such as `expected_connector_calls`, `mock_connector_calls`, and `real_external_http_calls=0`, or equivalent unambiguous names. | PASS |
| VS0-ACC-004 | MUST_PASS | Local acceptance must be repeatable by an operator. | README quickstart starts runtime, ingests one fixture, searches, creates evidence/claim/action, dry-runs or executes only local/mock action, and verifies audit. | README quickstart section, command transcript, exit codes, evidence refs, audit refs, and no live-provider requirement. | PASS |
| VS0-ACC-005 | MUST_PASS | Humans need a reviewable evidence package, not only scattered command output. | Generate a release evidence bundle containing scenario report, screenshots/trace, command outputs, readiness output, negative evidence, changed/gap list, and human-required rows. | Evidence package path, manifest, included artifacts, checksums or stable refs, and review summary. | PASS |
| VS0-ACC-R01 | REGRESSION_GUARD | Production and live-provider readiness must not be overclaimed. | `production_release_ready=false`; live connector proof and human usability proof remain `HUMAN_REQUIRED`; local acceptance cannot unlock production release. | Readiness JSON, release report, scenario matrix row statuses, negative evidence counters. | PASS |
| VS0-ACC-R02 | REGRESSION_GUARD | Acceptance hardening must not regress the accepted runtime or frozen scenario matrix. | Existing local deterministic checks still pass, including `make verify-local-fast` and `make verify-vs0-runtime`. | Command output, scenario reports, and unchanged canonical 206-scenario count. | PASS |
| VS0-ACC-H01 | HUMAN_REQUIRED | Live provider execution requires credentials and may mutate third-party systems. | Later approved live ConnectorHub/provider test with redacted transcript and audit refs. | Human approval, redacted live-provider transcript, provider/action result, audit refs. | HUMAN_REQUIRED |
| VS0-ACC-H02 | HUMAN_REQUIRED | Usability acceptance is subjective and requires the owner/operator. | JiYong/Tars performs the VS0 runtime walkthrough and records accept/reject with notes. | Human acceptance note plus screenshots/recording or issue list. | HUMAN_REQUIRED |

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

Future implementation must satisfy CLI parity before any AI-verifiable `VS0-ACC-*` row can be marked `PASS`.

| Scenario | Required CLI Commands | Required JSON / Evidence | Initial CLI Status |
|---|---|---|---|
| VS0-ACC-001 | `cornerstone scenario verify vs0-runtime-acceptance --scenario VS0-ACC-001 --json` | browser trace refs, screenshot refs, route assertions, production-overclaim assertion | PASS |
| VS0-ACC-002 | `cornerstone ready --json`; `cornerstone scenario verify vs0-runtime-acceptance --json` | last report path, timestamp, commit, gate status, scenario status | PASS |
| VS0-ACC-003 | `cornerstone action dry-run <action_id> --json`; `cornerstone action execute <action_id> --json` | unambiguous mock/expected/real external call counters | PASS |
| VS0-ACC-004 | README quickstart commands using native `cornerstone ... --json` paths | transcript with exit codes, evidence refs, audit refs | PASS |
| VS0-ACC-005 | `cornerstone release evidence collect --scope vs0-runtime-acceptance --json` | evidence bundle manifest and artifact refs | PASS |
| VS0-ACC-R01 | `cornerstone ready --json`; `cornerstone scenario verify vs0-runtime-acceptance --scenario VS0-ACC-R01 --json` | production false, human-required rows preserved | PASS |
| VS0-ACC-R02 | `make verify-local-fast`; `make verify-vs0-runtime`; `make verify-vs0-acceptance` | command output and scenario reports | PASS |

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

These commands are proposed for the future implementation task. They are not run by this documentation-only freeze.

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-runtime-acceptance --json --output reports/scenario/vs0-runtime-acceptance-2026-06-11.json
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs0-runtime-acceptance-2026-06-11.json --json
PATH="$PWD:$PATH" cornerstone release evidence collect --scope vs0-runtime-acceptance --json
make verify-vs0-runtime
make verify-vs0-acceptance
make verify-local-fast
```

## Done Means

VS0 Runtime Acceptance And Hardening is locally done when:

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

This contract is locally accepted when the documentation exists, is discoverable from the SoT/README, the acceptance verifier passes, the release evidence package is generated, and the existing local runtime/scenario gates remain green.

The future implementation gate cannot be marked done if any AI-verifiable `VS0-ACC-*` MUST_PASS or REGRESSION_GUARD scenario is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.

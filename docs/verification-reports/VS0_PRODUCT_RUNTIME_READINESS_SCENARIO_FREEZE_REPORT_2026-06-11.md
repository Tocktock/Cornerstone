# VS0 Product Runtime Readiness Scenario Freeze Report - 2026-06-11

## Summary

- Verdict: scenarios frozen; implementation not started.
- Scope: documentation-only scenario contract for VS0 Product Runtime Readiness.
- Date: 2026-06-11
- Owner: JiYong / Tars
- Commit: NOT_COMMITTED

## Goal

Freeze the next VS0 Product Runtime Readiness scenarios before implementation.

This report records the scenario contract for the next task:

```text
Artifact ingest
-> searchable derived representation
-> reproducible search snapshot
-> Evidence Bundle
-> Draft/Evidence-backed Claim
-> Action Card dry-run
-> approval
-> local/mock ConnectorHub-style execution
-> audit timeline and tamper verification
```

No runtime code, CLI code, API code, UI code, tests, fixtures, or verification scripts were changed by this scenario-freeze step.

## Current Behavior Snapshot

`cornerstone ready --json` currently exits 4 and reports:

```text
status: not_ready
missing: api_runtime, web_runtime
```

This is expected. The new scenarios are the next implementation gate and must stay `NOT_RUN` until implementation exists.

## Scenario Verification

| ID | Type | Expected Result | Verification Method | Evidence | Status |
|---|---|---|---|---|---|
| VS0-RT-001 | MUST_PASS | Runtime readiness is truthful; API health works; minimal UI loads. | CLI/API/UI runtime verification. | Future `vs0-product-runtime` report. | NOT_RUN |
| VS0-RT-002 | MUST_PASS | Uploaded text artifact is immutable, scoped, checksummed, derived, evidence-linked, and audited. | Upload through CLI/API/UI. | Future artifact/audit transcript. | NOT_RUN |
| VS0-RT-003 | MUST_PASS | Search uploaded content through CLI/API/UI and create reproducible search snapshot. | Search and snapshot transcript. | Future search snapshot report. | NOT_RUN |
| VS0-RT-004 | MUST_PASS | Claim from evidence has Evidence Bundle; zero-evidence approval is denied. | Claim create/approve tests. | Future claim/evidence transcript. | NOT_RUN |
| VS0-RT-005 | MUST_PASS | Action Card dry-run shows diff, impact, policy, risk, approval need, and reversibility note. | Action dry-run transcript. | Future action/policy report. | NOT_RUN |
| VS0-RT-006 | MUST_PASS | Local/mock action execution records WorkflowRun/action result with `external_calls = 0`. | Approval/execution transcript. | Future workflow/action/audit report. | NOT_RUN |
| VS0-RT-007 | MUST_PASS | Audit timeline covers artifact/search/claim/action/policy events and tamper verify passes. | Audit query/export/verify transcript. | Future audit verification report. | NOT_RUN |
| VS0-RT-008 | MUST_PASS | Minimal Calm Surface UI exposes Home/Ops Inbox, Artifact Viewer, Search, Claim Builder, Action Card, and Audit Detail. | Browser trace/screenshots. | Future UI evidence. | NOT_RUN |
| VS0-RT-R01 | REGRESSION_GUARD | Prompt-injection fixture creates no tool call, action card, egress, or authority expansion. | Negative evidence counters. | Future safety report. | NOT_RUN |
| VS0-RT-R02 | REGRESSION_GUARD | Cross-namespace read is denied with cause, resolution guide, policy ref, and audit ref. | Denial transcript. | Future policy/audit report. | NOT_RUN |
| VS0-RT-R03 | REGRESSION_GUARD | Zero-evidence Claim cannot be approved or published. | Claim approval denial transcript. | Future claim trust-state report. | NOT_RUN |
| VS0-RT-R04 | REGRESSION_GUARD | Readiness output separates local, VS0 runtime, and production readiness without overclaim. | Readiness/release transcript. | Future readiness report. | NOT_RUN |
| VS0-RT-H01 | HUMAN_REQUIRED | Live ConnectorHub/provider dry-run/execution is verified later with redacted evidence. | Human live-provider verification. | Future redacted transcript and approval. | HUMAN_REQUIRED |
| VS0-RT-H02 | HUMAN_REQUIRED | Human operator confirms VS0 runtime flow is understandable and useful. | Human usability walkthrough. | Future acceptance note/screenshots. | HUMAN_REQUIRED |

## CLI Parity Summary

| Feature / Scenario | CLI Command(s) | JSON Schema | Exit-Code Tests | Evidence/Audit Refs | Same Backend Path | Status |
|---|---|---|---|---|---|---|
| VS0 runtime readiness | `cornerstone ready --json`; `cornerstone health --json`; `cornerstone scenario verify vs0-product-runtime --json` | Future `cs.cli.v0`/runtime readiness schema. | 0 ready, 4 not ready, 5 runtime failure. | readiness report refs. | CLI/API/UI readiness service. | NOT_RUN |
| Artifact ingest | `cornerstone artifact ingest <path> --json`; `cornerstone artifact show <id> --json` | Future artifact schema. | 0 success, 1 validation, 2 policy, 8 unsafe output. | artifact/evidence/audit refs. | Shared archive service. | NOT_RUN |
| Search snapshot | `cornerstone search query "<query>" --json`; `cornerstone search snapshot show <id> --json` | Future search snapshot schema. | 0 success, 2 policy, 3 not found. | search snapshot/evidence/audit refs. | Shared search service. | NOT_RUN |
| Claim lifecycle | `cornerstone claim create ... --json`; `cornerstone claim approve <id> --json` | Future claim schema. | 0 success, 4 evidence missing. | claim/evidence/audit refs. | Shared claim service. | NOT_RUN |
| Action lifecycle | `cornerstone action propose ... --json`; `cornerstone action dry-run <id> --json`; `cornerstone action approve <id> --json`; `cornerstone action execute <id> --json` | Future action/workflow schema. | 0 success, 2 policy denial, 6 approval required, 7 connector unavailable. | action/policy/audit refs. | Shared action/workflow path. | NOT_RUN |
| Audit | `cornerstone audit list --json`; `cornerstone audit verify --json`; `cornerstone audit export --json` | Future audit schema. | 0 success, 5 integrity failure. | audit event/checkpoint refs. | Shared audit ledger. | NOT_RUN |
| UI runtime | `cornerstone scenario verify vs0-product-runtime --ui --json` or equivalent verifier | Future UI assertion schema. | 0 success, 4 missing evidence, 5 runtime failure. | screenshot/trace/audit refs. | UI uses same API/backend. | NOT_RUN |

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS0-RT-H01 | Live connector/provider verification needs credentials and may mutate third-party state. | Approve and perform live ConnectorHub/provider dry-run/execution later. | Redacted transcript, provider/action result, audit refs, written approval. | Blocks live-provider production release, not local VS0 runtime PASS. |
| VS0-RT-H02 | Usability acceptance is subjective. | Human operator walks through VS0 runtime and confirms the loop is understandable/useful. | Acceptance note plus screenshots/recording or issue list. | Blocks product acceptance claim, not deterministic local runtime checks. |

## Tool / Process Evidence

- Inputs inspected:
  - attachment `pasted-text.txt`;
  - `README.md`;
  - `AGENTS.md`;
  - `docs/sot/README.md`;
  - `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`;
  - `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md`;
  - `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md`;
  - `docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md`;
  - `docs/scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv`;
  - `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md`;
  - `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`;
  - `docs/sot/sot_manifest.yaml`.
- Current behavior reverse-engineered:
  - full scenario list remains 206;
  - VS-0 implementation subset remains 58;
  - `cornerstone ready --json` reports API/web runtime missing;
  - scenario coverage command reports matrix OK.
- Files or artifacts changed:
  - `docs/scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md`;
  - `docs/scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_MATRIX.csv`;
  - `docs/verification-reports/VS0_PRODUCT_RUNTIME_READINESS_SCENARIO_FREEZE_REPORT_2026-06-11.md`;
  - README/SoT discoverability docs.
- Commands/checks run:
  - `git status --short --branch`;
  - `cornerstone ready --json`;
  - `cornerstone scenario list --set full --json`;
  - `cornerstone scenario coverage --json`;
  - documentation verification scripts after edits.
- Failed checks and fixes:
  - none expected for documentation-only scenario freeze.
- Checks not run:
  - `cornerstone scenario verify vs0-product-runtime --json`;
  - `make verify-vs0-runtime`;
  - API health endpoint;
  - browser/UI runtime walkthrough;
  - live ConnectorHub/provider verification.

## Failure Reverse Engineering

| Scenario | Expected | Actual / Missing Evidence | First Failing Layer | Root Cause | Fix or Blocker | Re-verification Plan |
|---|---|---|---|---|---|---|
| VS0-RT-001 | Ready reports VS0 runtime with CLI/API/UI evidence. | `cornerstone ready --json` reports missing `api_runtime` and `web_runtime`. | Runtime surface. | API and web runtime are not implemented yet. | Implement runtime in a future task. | Run `cornerstone scenario verify vs0-product-runtime --json`. |
| VS0-RT-002..VS0-RT-007 | Artifact/search/claim/action/audit loop works across runtime. | Runtime loop not implemented. | Product runtime. | Current repo only has deterministic local scaffold/scenario proof. | Future implementation. | Capture CLI/API/UI transcripts and scenario report. |
| VS0-RT-008 | Minimal UI loads and exposes loop. | Web runtime missing. | UI runtime. | Current repo has no minimal product UI runtime. | Future UI implementation. | Browser trace/screenshots plus UI assertions. |
| VS0-RT-R01..VS0-RT-R04 | Runtime regressions are guarded in the product runtime. | Runtime verifier does not exist yet. | Scenario verifier. | Contract only freezes scenarios. | Future verifier implementation. | Run targeted `--scenario` checks after implementation. |

## Verification Gaps

- No `vs0-product-runtime` verifier exists yet.
- No API runtime exists yet.
- No web runtime exists yet.
- No minimal runtime UI evidence exists yet.
- No local/mock WorkflowRun execution evidence exists yet for this task-scoped runtime contract.
- No human live-provider or usability evidence exists yet.

## Risks

- Treating current scaffold scenario PASS as runtime PASS would overclaim readiness.
- Adding API/UI implementation without CLI parity would violate no-CLI-no-feature-PASS.
- Live connector work could require credentials or mutate third-party systems, so it must stay human-required.
- UI implementation must preserve Calm Surface and not regress into chatbot-only, connector-admin-first, or dark command-center UX.

## Verdict

- AI-verifiable scope: needs-follow-up.
- Human/release gate: needs-human-verification for live-provider and usability acceptance.
- Implementation gate: scenarios are frozen; runtime implementation may start only in a future task that explicitly accepts this contract.

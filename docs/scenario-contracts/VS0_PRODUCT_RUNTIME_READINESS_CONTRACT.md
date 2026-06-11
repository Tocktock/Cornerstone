# CornerStone VS0 Product Runtime Readiness Contract

**Date:** 2026-06-11
**Owner:** JiYong / Tars
**Status:** Frozen task-scoped scenario contract; local deterministic implementation verified on 2026-06-11. Production/live-provider acceptance remains human-required.

## Purpose

This contract freezes the VS0 Product Runtime Readiness implementation scenarios and records the current local deterministic implementation boundary.

It is for **VS0 Product Runtime Readiness**, not live connector production release. It does not expand or replace the canonical 206 `CS-*` scenario standard. It defines the next task-specific `VS0-RT-*` gate that converts the current verified local deterministic CLI/scenario proof into a runnable VS-0 product runtime with CLI, API, and minimal UI parity.

## Relationship To Existing Scenario Authority

- Canonical long-term product standard remains `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` with 206 scenarios.
- VS-0 implementation subset remains `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` with 58 scenarios.
- This contract is a task-scoped runtime readiness overlay for the first runnable product loop.
- Scenario `PASS` still requires deterministic evidence and native CLI parity.
- Local LLMs may support semantic smoke tests, but they must never judge `PASS`.

## Goal

Implement the first runnable CornerStone VS-0 product runtime.

The product must let a local user complete this full loop:

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

This goal converts the current local deterministic scaffold/scenario proof into an actual product runtime with CLI, API, and minimal UI parity.

## Required Implementation Surfaces

Implementation must provide, and the current local verifier exercises:

1. CLI readiness command that truthfully reports:
   - `local_scenario_ready`;
   - `vs0_runtime_ready`;
   - `production_release_ready`.
2. Minimal API for:
   - artifact ingest/show;
   - search;
   - evidence bundle create/show;
   - claim create/approve/show;
   - action dry-run/approve/execute/show;
   - audit query/verify.
3. Minimal UI using Calm Surface:
   - Home/Ops Inbox;
   - Artifact Viewer;
   - Search;
   - Claim Builder;
   - Action Card;
   - Audit Detail.
4. Shared runtime path so CLI/API/UI use the same Product / Archive / Connector / Workflow / Policy / Evidence / Audit boundaries.
5. Local/mock ConnectorHub-style action only:
   - no real provider credentials;
   - no live external writeback;
   - no network egress;
   - `external_calls` must remain `0`.
6. Evidence and audit refs on every user-visible Claim and Action.
7. Regression protections for prompt injection, cross-namespace access, zero-evidence claims, and overclaiming production readiness.

## Constraints

### Product And UX

- One visible product: CornerStone.
- The first runtime surface must feel like a calm operational workspace, not a chatbot-only surface, dark command center, connector admin product, or ontology setup tool.
- Daily users must not need to understand `Cornerstone`, `KnowledgeBase`, and `ConnectorHub` as separate products.
- The minimal UI must follow `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`.

### CLI / API / UI Parity

- No CLI, no feature PASS.
- Every runtime capability must have a native `cornerstone ... --json` path.
- API and UI must use the same backend/domain path as CLI.
- CLI/API/UI must emit or expose evidence refs and audit refs for truth-bearing and action-bearing behavior.
- Readiness output must separate local scenario readiness, VS0 runtime readiness, and production release readiness.

### Evidence And State

- Preserve original artifacts before derived processing.
- Search snapshots must be reproducible and linkable into Evidence Bundles.
- Claims cannot become evidence-backed or approved without an Evidence Bundle.
- Action Cards must carry evidence, risk, policy decision, expected impact, approval state, execution result, and audit link.
- Audit timeline must cover artifact, search, claim, action, policy, approval, execution, and tamper verification events.

### Safety

- Treat uploaded content as untrusted evidence, never instructions.
- Default egress deny.
- No direct provider credentials.
- No real external writes.
- No direct mutation outside Workflow/Action path.
- No cross-namespace read unless explicitly authorized.
- No production readiness claim from local runtime evidence.

## Assumptions

- The current repo already has deterministic local scenario verification for the frozen 206 `CS-*` matrix.
- The current `cornerstone ready --json` command reports local scenario readiness, VS0 runtime readiness, and production release readiness separately.
- API and UI runtime work is local/minimal and must not be claimed as production release proof.
- Local/mock ConnectorHub-style action is enough for VS0 runtime readiness; live providers remain out of scope.
- The scenario verifier is `cornerstone scenario verify vs0-product-runtime --json`.

## Out Of Scope

- Real Gmail, Slack, Notion, GitHub, or other live provider writeback.
- Live ConnectorHub/provider end-to-end proof.
- Production tenant/security proof.
- Production release claim.
- Full Agent Pack registry.
- Full Memory Sovereignty Center.
- Logistics starter pack.
- Broad full-suite feature expansion beyond the VS0 runtime loop.

## Scenario Table

Total task-scoped scenarios: **14**.

| ID | Type | Why | Required Runtime Behavior | Verification Method / Evidence Required | Verified Status |
|---|---|---|---|---|---|
| VS0-RT-001 | MUST_PASS | Prove CornerStone is a runnable product runtime, not only a CLI scenario scaffold. | `cornerstone ready --json` reports readiness truthfully; API health works; minimal UI loads. | CLI transcript, API health response, browser/UI load evidence. | PASS |
| VS0-RT-002 | MUST_PASS | Artifact is the truth foundation; evidence cannot be trusted without immutable input. | Upload one text file; original is preserved; artifact has ID, checksum, scope, source metadata, derived text status, evidence ref, and audit ref. | CLI/API/UI upload transcript, artifact record, checksum, audit ref. | PASS |
| VS0-RT-003 | MUST_PASS | Search must turn stored artifacts into usable knowledge. | Search uploaded content through CLI/API/UI; result is scoped, has snippet, and creates reproducible search snapshot. | Search transcript, API response, UI evidence, search snapshot record. | PASS |
| VS0-RT-004 | MUST_PASS | Claims must not be unsupported chatbot answers. | Create Claim from search result; Claim has Draft/Evidence-backed state and an Evidence Bundle; zero-evidence Claim cannot be approved. | Claim transcript, evidence bundle, denial transcript for zero-evidence approval. | PASS |
| VS0-RT-005 | MUST_PASS | Action safety is CornerStone's difference from normal RAG. | Create Action Card from Claim; dry-run shows diff, impact, policy decision, risk, approval need, and rollback/compensation note. | Action dry-run transcript, policy decision, audit ref, UI/API action detail. | PASS |
| VS0-RT-006 | MUST_PASS | VS-0 must complete Act safely. | Approve and execute a local/mock ConnectorHub-style action; WorkflowRun/action result is recorded; `external_calls = 0`. | Approval transcript, execution transcript, workflow/action result, audit ref, negative evidence. | PASS |
| VS0-RT-007 | MUST_PASS | Audit proves operational intelligence, not loose automation. | Audit timeline shows artifact/search/claim/action/policy/approval/execution events; `cornerstone audit verify` passes tamper check. | Audit list/export, tamper verification transcript, timeline evidence. | PASS |
| VS0-RT-008 | MUST_PASS | UI must expose the loop in a usable way. | Minimal Calm Surface UI supports Home/Ops Inbox, Artifact Viewer, Search, Claim Builder, Action Card, and Audit Detail. | Browser trace/screenshots plus UI assertions for required surfaces. | PASS |
| VS0-RT-R01 | REGRESSION_GUARD | Prompt-injection documents must never become instructions. | Ingest prompt-injection fixture; no tool call, no action card, no egress, no authority expansion. | Negative evidence counters and audit/policy records. | PASS |
| VS0-RT-R02 | REGRESSION_GUARD | Namespace/owner boundary is a core trust rule. | Attempt cross-namespace read; request is denied with cause, resolution guide, policy ref, and audit ref. | Denial transcript with exit code/policy/audit refs. | PASS |
| VS0-RT-R03 | REGRESSION_GUARD | Evidence-first rule must remain enforced. | Try approving/publishing Claim with zero evidence; it remains Draft or is denied. | Denial transcript and unchanged claim trust state. | PASS |
| VS0-RT-R04 | REGRESSION_GUARD | Do not accidentally claim production readiness. | Readiness output separates `local_scenario_ready`, `vs0_runtime_ready`, and `production_release_ready=false/human_required`. | Readiness JSON transcript and release report. | PASS |
| VS0-RT-H01 | HUMAN_REQUIRED | Real connector verification needs credentials and may mutate third-party state. | Later human verifies live ConnectorHub/provider dry-run/execution with redacted evidence and audit refs. | Redacted live-provider transcript, audit refs, written approval. | HUMAN_REQUIRED |
| VS0-RT-H02 | HUMAN_REQUIRED | Usability acceptance is subjective. | Human operator confirms VS-0 flow is understandable and useful. | Human acceptance note, screenshots or recording, issue list if rejected. | HUMAN_REQUIRED |

## Mapping To Existing Product Scenarios

| Runtime Scenario | Existing Scenario Coverage |
|---|---|
| VS0-RT-001 | `CS-SEC-001`, `CS-PROD-001`, `CS-PROD-002`, `CS-SEC-019`, `CS-REG-020` |
| VS0-RT-002 | `CS-ARCH-001`, `CS-ARCH-004`, `CS-ARCH-009` |
| VS0-RT-003 | `CS-UND-001`, `CS-UND-002`, `CS-UND-003`, `CS-ARCH-008` |
| VS0-RT-004 | `CS-CLAIM-005`, `CS-CLAIM-006`, `CS-CLAIM-007`, `CS-CLAIM-008` |
| VS0-RT-005 | `CS-AUTO-007`, `CS-AUTO-008`, `CS-AUTO-009`, `CS-AUTO-011` |
| VS0-RT-006 | `CS-AUTO-010`, `CS-AUTO-011`, `CS-AUTO-012`, `CS-SEC-014` |
| VS0-RT-007 | `CS-SEC-006`, `CS-AUTO-017`, `CS-REG-017` |
| VS0-RT-008 | `CS-PROD-001`, `CS-PROD-005`, `CS-PROD-006`, `CS-PROD-007`, `DS-S01`, `DS-S03`, `DS-S04`, `DS-S05`, `DS-S07` |
| VS0-RT-R01 | `CS-ARCH-007`, `CS-SEC-007`, `CS-REG-013` |
| VS0-RT-R02 | `CS-NS-001`, `CS-NS-003`, `CS-SEC-004`, `CS-SEC-005`, `CS-REG-006` |
| VS0-RT-R03 | `CS-CLAIM-006`, `CS-CLAIM-007`, `CS-REG-016` |
| VS0-RT-R04 | `CS-SEC-019`, `CS-SEC-020`, `CS-REG-020` |

## CLI Parity

Implementation must satisfy CLI parity before any runtime scenario can be marked `PASS`.

| Scenario | Required CLI Commands | Required JSON / Evidence | Initial CLI Status |
|---|---|---|---|
| VS0-RT-001 | `cornerstone ready --json`; `cornerstone health --json`; `cornerstone scenario verify vs0-product-runtime --json` | readiness booleans, runtime surface status, scenario report refs | PASS |
| VS0-RT-002 | `cornerstone artifact ingest <path> --json`; `cornerstone artifact show <artifact_id> --json` | artifact ID, checksum, scope, source metadata, evidence refs, audit refs | PASS |
| VS0-RT-003 | `cornerstone search query "<query>" --json`; `cornerstone search snapshot show <snapshot_id> --json` | scoped results, snippet, reproducible search snapshot | PASS |
| VS0-RT-004 | `cornerstone evidence bundle create --search-snapshot-id <id> --json`; `cornerstone claim create --evidence-bundle-id <id> --json`; `cornerstone claim approve <claim_id> --json` | trust state, evidence bundle, approval denial for zero evidence | PASS |
| VS0-RT-005 | `cornerstone action propose --mission-id <id> --claim-id <id> --json`; `cornerstone action dry-run <action_id> --json` | diff, impact, policy decision, risk, approval requirement, audit refs | PASS |
| VS0-RT-006 | `cornerstone action approve <action_id> --json`; `cornerstone action execute <action_id> --json` | local/mock execution result, `external_calls = 0`, audit refs | PASS |
| VS0-RT-007 | `cornerstone audit list --json`; `cornerstone audit verify --json`; `cornerstone audit export --json` | timeline and tamper verification | PASS |
| VS0-RT-008 | `cornerstone scenario verify vs0-product-runtime --json` | UI trace summary, assertions, CLI transcript | PASS |
| VS0-RT-R01 | `cornerstone scenario verify vs0-product-runtime --scenario VS0-RT-R01 --json` | zero tool calls, zero action cards, zero egress, audit record | PASS |
| VS0-RT-R02 | `cornerstone access evaluate ... --json`; `cornerstone scenario verify vs0-product-runtime --scenario VS0-RT-R02 --json` | denial with cause, resolution guide, policy ref, audit ref | PASS |
| VS0-RT-R03 | `cornerstone claim approve <claim_id> --json` | exit code 4 or policy denial; claim remains Draft | PASS |
| VS0-RT-R04 | `cornerstone ready --json` | separate local/runtime/production readiness fields | PASS |

CLI status: `PASS` for local deterministic VS0 runtime verification. Production release remains out of scope.

## API Parity

The local API runtime exposes the same underlying runtime path as the CLI. Minimum API surfaces:

- `GET /health`
- `GET /ready`
- `POST /artifacts`
- `GET /artifacts/{artifact_id}`
- `POST /search`
- `GET /search-snapshots/{snapshot_id}`
- `POST /evidence-bundles`
- `POST /claims`
- `POST /claims/{claim_id}/approve`
- `POST /actions`
- `POST /actions/{action_id}/dry-run`
- `POST /actions/{action_id}/approve`
- `POST /actions/{action_id}/execute`
- `GET /audit-events`
- `POST /audit/verify`

API status: `PASS` for local deterministic VS0 runtime verification. Live provider API verification remains `HUMAN_REQUIRED`.

## UI Parity

The local UI implementation must satisfy the Calm Surface design contract and expose:

- Home/Ops Inbox;
- Artifact Viewer;
- Search;
- Claim Builder;
- Action Card;
- Audit Detail.

UI proof requires browser/UI assertions that text does not overclaim production readiness.

UI status: `PASS` for local deterministic VS0 runtime verification. Human usability acceptance remains `HUMAN_REQUIRED`.

## Negative Evidence Required

Scenario reports must include zero-valued negative evidence counters for:

- `external_calls`;
- `tool_calls_from_untrusted_artifact`;
- `action_cards_from_prompt_injection`;
- `authority_expansion_from_prompt`;
- `cross_namespace_reads`;
- `zero_evidence_claim_approvals`;
- `production_release_overclaim`;
- `direct_provider_credentials_used`;
- `direct_mutations_outside_action_path`;
- `audit_tamper_verify_failures`.

## Verification Commands

These commands verify the current local deterministic VS0 runtime implementation. They do not prove production release readiness or live-provider safety.

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-product-runtime --json --output reports/scenario/vs0-product-runtime-2026-06-11.json
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs0-product-runtime-2026-06-11.json --json
make verify-vs0-runtime
make verify-local-fast
```

## Done Means

VS0 Product Runtime Readiness is locally done when:

- `cornerstone scenario verify vs0-product-runtime --json` exits 0;
- every AI-verifiable `VS0-RT-*` MUST_PASS and REGRESSION_GUARD row is `PASS` with concrete evidence;
- `make verify-vs0-runtime` passes;
- existing local deterministic scenario verification remains green;
- API health/ready endpoints work;
- minimal UI flow is verified by browser trace or screenshots;
- final report exists at `docs/verification-reports/VS0_PRODUCT_RUNTIME_READINESS_REPORT_2026-06-11.md`;
- human-required rows list required human action/evidence and are not falsely marked PASS.

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS0-RT-H01 | Live connector/provider verification needs credentials and may mutate third-party state. | Approve and perform live ConnectorHub/provider dry-run/execution later. | Redacted transcript, provider/action result, audit refs, written approval. | Blocks live-provider production release, not local VS0 runtime PASS. |
| VS0-RT-H02 | Usability acceptance is subjective. | Human operator walks through VS0 runtime and confirms the loop is understandable/useful. | Acceptance note plus screenshots/recording or issue list. | Blocks product acceptance claim, not deterministic local runtime checks. |

## Verdict Rule

VS0 Product Runtime Readiness cannot be marked done if any AI-verifiable `VS0-RT-*` MUST_PASS or REGRESSION_GUARD scenario is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.

This contract freezes the scenarios and records local deterministic runtime verification. It does not verify live providers, production tenant/security posture, or human usability acceptance.

# CornerStone VS4 Slice 012 Action Execution Boundary

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice strengthens Action Card execution/approval safety in the Product Alpha loop; it does not add canonical VS4 rows or provide production, live-provider, final security, or human UX acceptance.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Previous slices:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_005_UX_POLISH_LEARN.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_006_RESPONSIVE_MOBILE_PROOF.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_007_KEYBOARD_FOCUS_REVIEW.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_008_ASK_RESULT_READABILITY.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_009_CLAIM_ACTION_NAV_DETAIL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_010_ASK_INJECTION_BOUNDARY.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_011_OPS_INBOX_TRIAGE_DETAIL.md`

## Goal

Make the VS4 Action Card feel safe to review because the UI and CLI both prove what happens when execution is not authorized.

The user should see that Action Cards remain reviewable product objects with goal, why, evidence, impact, risk, approval, execution mode, policy, and activity, while unauthorized approval or execution attempts produce a denial, safety envelope, audit record, and zero provider/writeback side effects.

## Scope

In this slice:

- add visible Action Card execution-boundary detail to the Product Alpha UI and Actions page;
- prove an Action Card execution attempt before authorized approval is denied with policy and audit refs;
- prove an unauthorized approval attempt is denied without changing the Action Card to approved/executed;
- prove denial creates no action result, workflow run, provider receipt, real provider call, direct provider access, provider mutation, or external HTTP call;
- expose the same boundary through native `cornerstone action ... --json` transcripts;
- keep the normal user nav small: `Home`, `Search`, `Artifacts`, `Claims`, `Actions`;
- keep the parent VS4 matrix at 20 `MUST_PASS`, 7 `REGRESSION_GUARD`, and 1 `HUMAN_REQUIRED`.

## Non-Scope

This slice does not implement or claim:

- live external writeback;
- production, on-prem, final security, real IdP, real network, migration, backup, restore, or live-provider readiness;
- final human Product Alpha UX acceptance;
- automatic durable memory approval;
- evidence-free claim approval;
- real provider credentials, network calls, or ConnectorHub production execution;
- new canonical VS4 scenario rows or changes to the 206-scenario matrix.

## Assumptions

- `cornerstone action propose`, `cornerstone action dry-run`, `cornerstone action approve`, `cornerstone action execute`, and `cornerstone action show` are the native CLI parity path for this slice.
- A high-risk or external-writeback Action Card in local VS4 can be used as a deterministic denial fixture without performing live writeback.
- A denial safety envelope is product evidence only when it has policy refs, audit refs, scope, reason code, resolution path, and zero side-effect counters.
- Reference image `cornerstone-reference-08-action-dry-run-approval.png` guides the action-review surface but is not PASS evidence.
- `VS4-H01` remains the only path for subjective Product Alpha UX acceptance.

## Selected Scenarios

This slice strengthens existing rows rather than adding new rows:

| ID | Classification | Why |
|---|---|---|
| VS4-GATE-001 | in_this_slice | Slice 012 contract and registration must be frozen without changing the parent matrix. |
| VS4-UI-007 | in_this_slice | The same unsafe-action boundary must preserve evidence-required approval behavior for Claims. |
| VS4-UI-009 | in_this_slice | The same unsafe-action boundary must preserve no-hidden-durable-memory behavior before review. |
| VS4-UI-010 | in_this_slice | Action Card review must show goal, evidence, impact, risk, policy, approval, execution mode, and activity, including denial detail. |
| VS4-UI-011 | in_this_slice | Local/dev execution mode must remain visible and must prove unauthorized execution does not imply live writeback. |
| VS4-REF-002 | in_this_slice | Claim/Action surfaces must align with dry-run, policy/risk, approval, execution mode, and activity patterns using runtime proof, not reference-image PASS. |
| VS4-REG-003 | in_this_slice | UI, report, and help copy must not claim production, on-prem, final security, live-provider, or human UX readiness. |
| VS4-REG-004 | in_this_slice | Prompt-injection and unauthorized action paths must not approve memory/claim/action, execute actions, call providers, mutate policy, or expand authority. |
| VS4-REG-007 | in_this_slice | Action execution boundary must have native CLI JSON transcripts; missing CLI parity blocks PASS. |
| VS4-H01 | human_required | JiYong/Tars Product Alpha UX acceptance remains human-only. |

## Full Scenario Classification

| ID Range | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001, VS4-UI-007, VS4-UI-009, VS4-UI-010, VS4-UI-011, VS4-REF-002, VS4-REG-003, VS4-REG-004, VS4-REG-007 | in_this_slice | Strengthened by Action Card execution-boundary UI, CLI denial transcripts, safety envelope evidence, audit refs, and negative side-effect counters. |
| VS4-UI-001 through VS4-UI-006, VS4-UI-008, VS4-UI-012 through VS4-UI-016, VS4-STATE-001, VS4-REF-001, VS4-REG-001 through VS4-REG-002, VS4-REG-005 through VS4-REG-006 | previous_slice | Must remain locally `PASS`; this slice must not weaken source, Brief, Claim, Memory/Wiki, Ask, Ops Inbox, reference, regression, or product-first evidence. |
| VS4-H01 | human_required | Human acceptance remains unclaimed until JiYong/Tars provides dated review evidence. |
| VS3-H01 through VS3-H07 | conditional_deferred | Still block production/on-prem/security/live-provider/human acceptance claims only. |

## Proof Needed

The selected slice can be considered prepared only with:

- a frozen Slice 012 contract registered in the repo README and SoT README;
- desktop and mobile browser proof that the Action Card surface shows execution-boundary, policy, approval, denied-until-authorized, no-live-writeback, and no-provider-result-on-denial detail;
- CLI transcripts for an evidence-backed high-risk Action Card covering `action propose`, `action dry-run`, `action execute` before approval, unauthorized `action approve`, and final `action show`;
- denial evidence with `CS_ACTION_AUTHORIZED_APPROVAL_REQUIRED` or equivalent policy reason, policy refs, audit refs, and action safety envelope refs;
- negative counters showing zero action result, workflow run, provider receipt, external HTTP call, provider mutation, real provider call, direct provider access, unauthorized approval, or executed state on denial;
- continued proof that prompt-injection paths create zero action execution side effects;
- the full VS4 verifier still reporting 27 AI-verifiable rows `PASS`, 1 `HUMAN_REQUIRED`, and 0 blocking rows;
- targeted unit tests plus docs, CLI-native-first, design-system, canonical matrix, py-compile, and `git diff --check` checks.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED`.
- Slice 012 can improve human review readiness, but it cannot create, infer, or validate JiYong/Tars acceptance.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.

## Done Criteria

This slice is done when:

1. Action Card UI and Actions page expose execution-boundary detail in product language;
2. unauthorized execution before approval is denied with policy, safety envelope, audit refs, and resolution path;
3. unauthorized approval is denied and does not change the Action Card to approved or executed;
4. denial creates no action result, workflow run, provider receipt, external HTTP call, provider mutation, real provider call, direct provider access, or live writeback;
5. `cornerstone action ... --json` transcripts are included as CLI parity evidence;
6. selected scenario rows are classified as `in_this_slice` without changing the parent VS4 matrix;
7. all existing VS4 AI-verifiable rows still pass and `VS4-H01` remains `HUMAN_REQUIRED`;
8. no report, README, UI, package, or validation output claims production, on-prem, final security, live-provider, accessibility certification, or human UX readiness.

## CLI Parity

- Propose Action Card: `cornerstone action propose --mission-id <mission_id> --claim-id <claim_id> --goal <goal> --action-kind external_writeback --risk high --connector mock_connector --target mock://... --json`.
- Read dry-run: `cornerstone action dry-run <action_id> --json`.
- Denied execution: `cornerstone action execute <action_id> --json`.
- Denied unauthorized approval: `cornerstone action approve <action_id> --approver unauthorized_delegate --json`.
- Read final Action Card state: `cornerstone action show <action_id> --json`.
- Verification: `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json`.
- Gate: `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json`.
- Slice target: `make verify-vs4-product-alpha-action-execution-boundary`.

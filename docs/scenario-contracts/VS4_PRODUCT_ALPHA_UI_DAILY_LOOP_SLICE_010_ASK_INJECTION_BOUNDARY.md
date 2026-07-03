# CornerStone VS4 Slice 010 Ask Injection Boundary

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice strengthens Ask prompt-injection and authority-boundary proof; it does not add canonical VS4 rows or provide human UX acceptance.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Previous slices:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_003_ASK_PACKS_STATES_REGRESSION.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_004_HUMAN_ACCEPTANCE_PACKAGE.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_005_UX_POLISH_LEARN.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_006_RESPONSIVE_MOBILE_PROOF.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_007_KEYBOARD_FOCUS_REVIEW.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_008_ASK_RESULT_READABILITY.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_009_CLAIM_ACTION_NAV_DETAIL.md`

## Goal

Make the Ask path safe as a daily product entry point when the user message itself contains prompt-injection or authority-granting text.

Unsafe Ask text must remain preserved as untrusted evidence, but it must not become an authority source for claim approval, memory approval, action execution, policy change, provider call, or authority expansion.

## Scope

In this slice:

- record unsafe-instruction detection on conversation-start evidence;
- keep benign Ask-to-work-item behavior unchanged;
- deny `cornerstone conversation promote` when the source conversation contains detected unsafe instructions;
- emit a structured JSON error and audit event for unsafe promotion denial;
- add Ask-specific negative evidence counters to the VS4 verifier;
- keep the parent VS4 matrix at 20 `MUST_PASS`, 7 `REGRESSION_GUARD`, and 1 `HUMAN_REQUIRED` row.

## Non-Scope

This slice does not implement:

- final human Product Alpha UX acceptance;
- natural-language LLM safety judgment;
- live provider or external writeback;
- durable memory approval;
- new canonical scenario rows or changes to the 206-scenario matrix;
- production, on-prem, final security, real IdP, or real network proof;
- a broader Ops Inbox redesign.

## Assumptions

- Ask messages are treated as untrusted evidence even when they come from the local user.
- Deterministic pattern detection is sufficient for this local Product Alpha proof; it is not a final security acceptance claim.
- A benign Ask message may still be promoted into a reviewable evidence-backed Claim candidate when evidence exists.
- A detected unsafe Ask message may still be answered or inspected as evidence, but cannot be promoted into product authority.
- `VS4-H01` remains the only path for subjective product-alpha UX acceptance.

## Selected Scenarios

This slice strengthens existing rows rather than adding new ones:

| ID | Classification | Why |
|---|---|---|
| VS4-GATE-001 | in_this_slice | Slice 010 contract and registration must be frozen without changing the parent matrix. |
| VS4-UI-007 | in_this_slice | Unsafe Ask promotion must be blocked like zero-evidence approval; prompt text cannot approve a Claim. |
| VS4-UI-009 | in_this_slice | Unsafe Ask text must create zero hidden durable memory or memory-approval side effects. |
| VS4-UI-011 | in_this_slice | Unsafe Ask text must create zero action execution, provider-call, or live-writeback side effects. |
| VS4-UI-013 | in_this_slice | Ask must still create normal reviewable work for benign input while unsafe input stays evidence-only. |
| VS4-REG-004 | in_this_slice | Prompt-injection content cannot approve memory/action/claim or create authority. |
| VS4-REG-007 | in_this_slice | Ask injection-boundary behavior must have native CLI JSON transcripts and expected denial exit code. |
| VS4-H01 | human_required | JiYong/Tars product-alpha UX acceptance remains human-only. |

All other AI-verifiable rows are `previous_slice` and must still pass through the unfiltered VS4 verifier.

## Full Scenario Classification

| ID Range | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001, VS4-UI-007, VS4-UI-009, VS4-UI-011, VS4-UI-013, VS4-REG-004, VS4-REG-007 | in_this_slice | Strengthened by unsafe Ask conversation safety, promotion denial, CLI transcript, and negative evidence counters. |
| VS4-UI-001 through VS4-UI-006, VS4-UI-008, VS4-UI-010, VS4-UI-012, VS4-UI-014 through VS4-UI-016, VS4-STATE-001, VS4-REF-001 through VS4-REF-002, VS4-REG-001 through VS4-REG-003, VS4-REG-005 through VS4-REG-006 | previous_slice | Must remain locally `PASS`; this slice must not weaken existing evidence. |
| VS4-H01 | human_required | Human acceptance remains unclaimed until JiYong/Tars provides dated review evidence. |
| VS3-H01 through VS3-H07 | conditional_deferred | Still block production/on-prem/security/live-provider/human acceptance claims only. |

## Proof Needed

The selected slice can be considered prepared only with:

- a frozen Slice 010 contract registered in the repo README and SoT README;
- a CLI transcript for benign Ask still producing conversation answer, Brief, Claim candidate, Memory/Wiki candidate, Action Card, local dry-run, and audit refs;
- a CLI transcript for unsafe Ask start showing `unsafe_instruction_detected=true` and `blocked_attempt_count>=1`;
- a CLI transcript for unsafe Ask promotion returning `CS_CONVERSATION_UNSAFE_SOURCE` with policy-denied exit code;
- negative evidence counters proving `unsafe_ask_tool_calls_created=0`, `unsafe_ask_action_cards_created=0`, `unsafe_ask_claim_promotions_created=0`, `unsafe_ask_claim_approvals_created=0`, `unsafe_ask_memory_approvals_created=0`, `unsafe_ask_hidden_memory_writes_created=0`, `unsafe_ask_action_executions_created=0`, `unsafe_ask_policy_changes_created=0`, `unsafe_ask_external_http_calls=0`, `unsafe_ask_direct_provider_access=0`, `unsafe_ask_authority_expanded=0`, `unsafe_ask_cross_workspace_reads=0`, and `unsafe_ask_unredacted_secret_leaks=0`;
- existing artifact prompt-injection fixture proof still passing;
- the full VS4 verifier still reporting 27 AI-verifiable rows `PASS`, 1 `HUMAN_REQUIRED`, and 0 blocking rows;
- targeted unit tests plus docs, CLI-native-first, design-system, canonical matrix, py-compile, and `git diff --check` checks.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED`.
- Slice 010 can improve human review readiness, but it cannot create, infer, or validate JiYong/Tars acceptance.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.

## Done Criteria

This slice is done when:

1. unsafe Ask text is preserved as untrusted evidence with deterministic unsafe-instruction metadata;
2. unsafe Ask promotion to Claim is denied with `CS_CONVERSATION_UNSAFE_SOURCE`;
3. benign Ask-to-work-item behavior still passes;
4. selected scenario rows are classified as `in_this_slice` without changing the parent VS4 matrix;
5. all existing VS4 AI-verifiable rows still pass and `VS4-H01` remains `HUMAN_REQUIRED`;
6. no report, README, UI, package, or validation output claims production, on-prem, final security, live-provider, accessibility certification, or human UX readiness.

## CLI Parity

- Unsafe Ask start: `cornerstone conversation start --message <unsafe text> --json`.
- Unsafe Ask answer inspection: `cornerstone conversation answer <conversation_id> --question <question> --json`.
- Unsafe promotion denial: `cornerstone conversation promote <conversation_id> --kind claim --statement <statement> --evidence-bundle-id <id> --json`.
- Verification: `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json`.
- Gate: `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json`.
- Slice target: `make verify-vs4-product-alpha-ask-injection-boundary`.

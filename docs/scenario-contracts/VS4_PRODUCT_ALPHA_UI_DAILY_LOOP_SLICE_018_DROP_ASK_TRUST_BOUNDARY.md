# CornerStone VS4 Product Alpha UI Daily Loop Slice 018 Drop/Ask Trust Boundary Contract

**Date:** 2026-07-04 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`

## Goal

Harden the Product Alpha `Drop / Ask` trust boundary so user-pasted text and conversation text cannot become trusted authority through caller-provided trust flags, HTTP/API intake, or checksum deduplication.

The local VS4 UI and CLI must keep pasted and Ask text as untrusted evidence, detect unsafe instruction attempts, deny unsafe claim promotion with structured policy/audit evidence, and prove zero hidden memory, claim approval, action execution, live provider, policy-change, or authority-expansion side effects.

## Scope

- Force inline text sources with `source_type=user_paste` or `source_type=conversation_turn` to `trust_state=untrusted`.
- Preserve same-checksum deduplication while downgrading any prior trusted artifact record when the current source is user-pasted or conversation text.
- Record trust forcing and downgrade evidence in artifact output and audit details.
- Force Product Alpha HTTP text intake through `user_paste` and `untrusted`, regardless of caller-supplied `source` or `trust`.
- Return structured HTTP/API denial for unsafe conversation promotion, including evidence refs, policy decision refs, audit refs, and `CS_CONVERSATION_UNSAFE_SOURCE`.
- Add browser proof markers for unsafe HTTP/API Drop/Ask trust-boundary behavior on desktop and mobile.
- Add native CLI proof that a same-checksum trusted text seed is downgraded when re-ingested as user paste.
- Add a Slice 018 Make target that refreshes both the scenario report and VS4 gate output.

## Non-Scope

- No new canonical VS4 matrix rows.
- No change to the canonical 206-scenario matrix.
- No final backend storage model decision for Brief, artifact dedupe, or memory.
- No visual redesign, Ops Inbox interactivity slice, or product-polish claim.
- No live provider, production, on-prem, final security, or human UX readiness claim.
- No promotion of `VS4-H01` beyond `HUMAN_REQUIRED`.

## Assumptions

- User-pasted and conversation-turn text are untrusted evidence even when an internal caller provides `trust=trusted`.
- Trust downgrade on dedupe is safer than preserving a stale trusted view of the same raw bytes when the current intake path is user-controlled.
- CLI parity requires proof of both ordinary text intake and adversarial same-checksum trust downgrade.
- HTTP/API proof is local Product Alpha proof only; it does not claim production API hardening or final security acceptance.

## Scenario Inventory

| Scenario ID | Priority | Slice classification | Gate meaning |
|---|---|---|---|
| `VS4-GATE-001` | MUST_PASS | `in_this_slice` | Slice 018 contract, verifier wiring, Make target, report, and gate output are structurally frozen and verified. |
| `VS4-UI-002` | MUST_PASS | `in_this_slice` | User-pasted Drop text remains an untrusted Source/Artifact even when callers or dedupe try to mark it trusted. |
| `VS4-UI-007` | MUST_PASS | `in_this_slice` | Unsafe conversation promotion is denied with structured policy/audit/evidence refs in CLI and HTTP/API paths. |
| `VS4-UI-009` | MUST_PASS | `in_this_slice` | Unsafe pasted or Ask text creates no hidden durable memory, approved memory, or authority side effects. |
| `VS4-UI-011` | MUST_PASS | `in_this_slice` | Local/mock boundary remains intact: no live provider, direct provider, external HTTP, workflow, or writeback side effects. |
| `VS4-UI-013` | MUST_PASS | `in_this_slice` | Unsafe Ask text stays evidence-only and cannot become a work-authority source through HTTP/API promotion. |
| `VS4-REG-003` | REGRESSION_GUARD | `in_this_slice` | The new trust-boundary proof does not overclaim production, on-prem, final security, live-provider, or human UX readiness. |
| `VS4-REG-004` | REGRESSION_GUARD | `in_this_slice` | Prompt-injection and trust-spoofing attempts cannot approve memory/claims, execute actions, call providers, change policy, or expand authority. |
| `VS4-REG-007` | REGRESSION_GUARD | `in_this_slice` | Trust-boundary behavior has native CLI parity and JSON/audit/evidence proof. |
| `VS4-H01` | HUMAN_REQUIRED | `human_required` | Human product-alpha UX acceptance remains unclaimed. |

All other AI-verifiable VS4 rows remain `previous_slice` or `later_slice` according to the active verifier ledger. `VS3-H01` through `VS3-H07` remain conditional deferred blockers only for production/on-prem/security/live-provider/human-acceptance claims.

## Required Verification

- `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-002 --scenario VS4-UI-007 --scenario VS4-UI-009 --scenario VS4-UI-011 --scenario VS4-UI-013 --scenario VS4-REG-003 --scenario VS4-REG-004 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json`
- Focused unittest asserting desktop/mobile `unsafe_http_boundary_markers`, CLI `text_trust_downgrade`, and the Slice 018 proof boundary.
- CLI smoke check for `cornerstone artifact ingest --text ... --source user_paste --trust trusted --json` after same-checksum trusted seed.
- Existing documentation checks: SoT, CLI-native-first, design-system, canonical scenario matrix, and `git diff --check`.

## Done Means

- `user_paste` and `conversation_turn` text artifacts are forced to `trust_state=untrusted`.
- Same-checksum re-ingest from user paste downgrades any prior trusted text artifact view and records source history.
- Product Alpha HTTP text intake cannot honor caller-supplied trusted source labels for inline text.
- Unsafe HTTP/API conversation promotion returns structured denial instead of falling through to claim creation.
- Browser proof exposes `unsafe_http_boundary` detail and all required desktop/mobile markers are true.
- CLI workflow exposes `text_trust_downgrade=true` and zero trust-boundary negative evidence.
- The Slice 018 Make target refreshes both the scenario report and gate output.
- `VS4-H01` remains `HUMAN_REQUIRED`.
- No production/on-prem/security/live-provider/human UX readiness claim is introduced.

# CornerStone VS4 Product Alpha UI Daily Loop Slice 017 User Drop/Ask Source Contract

**Date:** 2026-07-04 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`

## Goal

Make the primary Product Alpha `Drop / Ask` path fixture-free for normal users.

When a user pastes source text and asks a question, the local VS4 UI must preserve that exact text as an untrusted Source/Artifact, create an evidence-backed Brief from the resulting Evidence Bundle, and continue into Claim, Memory/Wiki, Action Card, Ops Inbox, Evidence/Audit, and Learn review without relying on the fixed `fixtures/vs0/packs/01_artifact_basic/input.txt` source.

## Scope

- Give the Home Drop textarea a deterministic user-paste source value and stable proof markers.
- Route the browser `Prepare Brief work item` and `Ask with evidence` flow through the pasted source text instead of the fixed fixture artifact.
- Add local HTTP runtime support for text artifact ingestion using the existing `LocalRuntimeStore.ingest_text_artifact` path.
- Add CLI parity for text source intake through `cornerstone artifact ingest --text ... --json`.
- Prove the resulting Brief is created from the user-pasted source checksum, source type, artifact ref, evidence bundle, and audit refs.
- Add deterministic desktop and mobile browser-proof markers for user Drop/Ask source preservation.
- Keep prompt-injection and local/no-live-writeback boundaries intact.

## Non-Scope

- No new canonical VS4 matrix rows.
- No change to the canonical 206-scenario matrix.
- No final backend storage model decision for Brief.
- No upload/file-picker implementation.
- No live provider, production, on-prem, final security, or human UX readiness claim.
- No promotion of `VS4-H01` beyond `HUMAN_REQUIRED`.

## Assumptions

- User-pasted text is untrusted evidence and must be preserved before any generated Brief output.
- `LocalRuntimeStore.ingest_text_artifact` is the correct local storage path for inline text because it already preserves original bytes, checksum, derived text, redaction, safety markers, provenance, and audit.
- CLI parity can be satisfied by adding `cornerstone artifact ingest --text` while keeping existing path-based artifact ingestion compatible.
- Reference images guide layout and hierarchy only. They are not PASS evidence.

## Scenario Inventory

| Scenario ID | Priority | Slice classification | Gate meaning |
|---|---|---|---|
| `VS4-GATE-001` | MUST_PASS | `in_this_slice` | Slice 017 contract, registration, verifier wiring, and gate output are structurally frozen and verified. |
| `VS4-UI-001` | MUST_PASS | `in_this_slice` | Home remains product-first with Drop/Ask/Continue, now using pasted source text for the primary flow. |
| `VS4-UI-002` | MUST_PASS | `in_this_slice` | Pasted input is preserved as an untrusted Source/Artifact with checksum, original storage ref, provenance, derived text, and audit refs. |
| `VS4-UI-003` | MUST_PASS | `in_this_slice` | The Evidence-backed Brief is created from the pasted-source Evidence Bundle, not from a fixed fixture. |
| `VS4-UI-004` | MUST_PASS | `in_this_slice` | Brief content shows supported findings and gaps derived from the pasted source. |
| `VS4-UI-013` | MUST_PASS | `in_this_slice` | Ask creates reviewable work from the user source and question, not chatbot-only output. |
| `VS4-UI-016` | MUST_PASS | `in_this_slice` | Normal user copy says Source, Evidence-backed Brief, and Ask before raw refs or fixture jargon. |
| `VS4-REG-003` | REGRESSION_GUARD | `in_this_slice` | The new source path does not overclaim production, on-prem, final security, live-provider, or human UX readiness. |
| `VS4-REG-004` | REGRESSION_GUARD | `in_this_slice` | Untrusted pasted or Ask content cannot approve memory/claims, execute actions, call providers, change policy, or expand authority. |
| `VS4-REG-006` | REGRESSION_GUARD | `in_this_slice` | Default UI remains product-first and not fixture/admin/connector/ontology/scenario-first. |
| `VS4-REG-007` | REGRESSION_GUARD | `in_this_slice` | Text source intake has native CLI parity through `cornerstone artifact ingest --text ... --json`. |
| `VS4-H01` | HUMAN_REQUIRED | `human_required` | Human product-alpha UX acceptance remains unclaimed. |

All other AI-verifiable VS4 rows remain `previous_slice` or `later_slice` according to the active verifier ledger. `VS3-H01` through `VS3-H07` remain conditional deferred blockers only for production/on-prem/security/live-provider/human-acceptance claims.

## Required Verification

- `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-001 --scenario VS4-UI-002 --scenario VS4-UI-003 --scenario VS4-UI-004 --scenario VS4-UI-013 --scenario VS4-UI-016 --scenario VS4-REG-003 --scenario VS4-REG-004 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json`
- `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json`
- Focused unittest asserting desktop and mobile `user_drop_ask_source_markers`.
- CLI parity check for `cornerstone artifact ingest --text ... --json`.
- Existing documentation checks: SoT, CLI-native-first, design-system, canonical scenario matrix, and `git diff --check`.

## Done Means

- Primary Home Drop/Ask flow uses pasted source text, not the fixed artifact fixture.
- Browser proof exposes `user_drop_ask_source` detail and all required desktop/mobile markers are true.
- Artifact source type/ref, checksum, original storage ref, derived text status, safety state, evidence bundle, brief, and audit refs are visible in proof.
- Ask output remains reviewable work with Brief, Claim, Memory/Wiki, Action Card, Evidence/Audit, and Learn refs.
- `cornerstone artifact ingest --text ... --json` returns a preserved artifact with storage/evidence/audit refs.
- `VS4-H01` remains `HUMAN_REQUIRED`.
- No production/on-prem/security/live-provider/human UX readiness claim is introduced.

# CornerStone VS4 Slice 003 Ask, Packs, States, and Regression

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice can prove the remaining AI-verifiable local Product Alpha rows; it does not provide human UX acceptance.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Previous slices:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`, `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_002_BRIEF_DETAIL.md`

## Goal

Complete the local AI-verifiable VS4 Product Alpha loop around the surfaces that Slice 001 and Slice 002 deliberately left open:

```text
Drop / Ask
-> Evidence-backed Brief
-> Claim candidate
-> Memory/Wiki candidate
-> Action Card draft
-> Ops Inbox follow-up
-> Evidence/Audit detail
-> Learn
```

The slice must prove that Ask, Search, Artifact, page-state coverage, three general-purpose packs, and fresh VS0/VS1 regressions all support the same product loop. It must not claim production, on-prem, final security, live-provider, or final human UX readiness.

## Scope

In this slice:

- make Ask visible as an evidence-backed workbench path that can become Brief, Claim, Memory/Wiki candidate, or Action Card draft;
- prove Ask through native `cornerstone conversation ...` CLI transcripts plus browser/runtime state;
- run Personal Research, Company Policy Review, and Operations Issue through the same local Brief / Claim / Memory / Action workflow;
- enrich Home, Search, and Artifact surfaces so they align with the reference direction using implementation/browser evidence, not reference images as PASS evidence;
- define and expose required page states for major Product Alpha pages: empty, loading, ready, partial/degraded, needs-review, permission denied, policy blocked, failed-with-recovery, and audit/log available;
- rerun fresh local VS0 and VS1 regression verifiers from the current tree and record their outputs as VS4 regression evidence;
- leave `VS4-H01` human-only acceptance unresolved.

## Non-Scope

This slice does not implement:

- production persistence, production deployment, on-prem packaging, live-provider writeback, real IdP, or real network proof;
- final security acceptance or VS3-P release unlock;
- human Product Alpha UX acceptance;
- admin, connector, or ontology-first default navigation;
- final backend storage-model changes beyond existing local deterministic runtime state.

## Assumptions

- Existing conversation, artifact, search, evidence, brief, claim, memory, mission, action, and audit CLI commands are the authoritative local CLI parity surface.
- General-purpose pack proof can use deterministic local fixtures as long as the verifier confirms each pack creates Brief, Claim candidate, Memory/Wiki candidate, Action Card draft, and Ops Inbox follow-up.
- Search and Artifact reference alignment can be proven by DOM/browser/source markers and runtime state; reference images remain visual guidance only.
- Page-state coverage can be represented by an explicit local UI state coverage matrix when backed by browser/DOM inspection.
- Fresh VS0/VS1 regression evidence may be nested inside the VS4 report and does not by itself claim production readiness.

## Selected Scenarios

This slice selects these AI-verifiable rows:

| ID | Classification | Why |
|---|---|---|
| VS4-UI-013 | in_this_slice | Ask must promote durable, evidence-backed work instead of ending as chatbot-only output. |
| VS4-UI-014 | in_this_slice | Three non-logistics packs must pass through the same loop. |
| VS4-STATE-001 | in_this_slice | Required page states must be defined and observable for major pages. |
| VS4-REF-001 | in_this_slice | Home, Search, and Artifact surfaces must align with the design reference direction using implementation evidence. |
| VS4-REG-001 | in_this_slice | Fresh VS0 local loop regression must pass on this tree. |
| VS4-REG-002 | in_this_slice | Fresh VS1 ontology suggest/review/promote regression must pass on this tree. |

Slice 001 and Slice 002 rows remain `previous_slice` and must still be rerunnable. `VS4-H01` remains `HUMAN_REQUIRED`.

## Full Scenario Classification

| ID | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001 | previous_slice | Parent contract, matrix, and slice contracts remain frozen and verified. |
| VS4-UI-001 | previous_slice | Product Alpha Home shell remains first screen. |
| VS4-UI-002 | previous_slice | Source / Artifact preservation remains verified through Slice 002. |
| VS4-UI-003 | previous_slice | Evidence-backed Brief creation remains verified through Slice 002. |
| VS4-UI-004 | previous_slice | Brief contents remain verified through Slice 002. |
| VS4-UI-005 | previous_slice | Shared Evidence Drawer remains verified through Slice 002. |
| VS4-UI-006 | previous_slice | Claim candidate remains verified through Slice 002. |
| VS4-UI-007 | previous_slice | Zero-evidence Claim approval denial remains verified through Slice 002. |
| VS4-UI-008 | previous_slice | Memory/Wiki candidate remains verified through Slice 002. |
| VS4-UI-009 | previous_slice | No hidden durable memory remains verified through Slice 002. |
| VS4-UI-010 | previous_slice | Action Card review remains verified through Slice 002. |
| VS4-UI-011 | previous_slice | Local/mock execution mode remains verified through Slice 002. |
| VS4-UI-012 | previous_slice | Ops Inbox returning-work shell remains verified through Slice 001. |
| VS4-UI-013 | in_this_slice | Verify Ask-to-work-item browser state and conversation CLI parity. |
| VS4-UI-014 | in_this_slice | Verify Personal Research, Company Policy Review, and Operations Issue pack loops. |
| VS4-UI-015 | previous_slice | Workspace context remains visible. |
| VS4-UI-016 | previous_slice | Product language remains first. |
| VS4-STATE-001 | in_this_slice | Verify explicit required page-state matrix. |
| VS4-REF-001 | in_this_slice | Verify Home/Search/Artifact reference alignment from runtime evidence. |
| VS4-REF-002 | previous_slice | Claim/Action reference alignment remains verified through Slice 002. |
| VS4-REG-001 | in_this_slice | Verify fresh VS0 operator acceptance UI regression. |
| VS4-REG-002 | in_this_slice | Verify fresh VS1 ontology suggest/review/promote regression. |
| VS4-REG-003 | previous_slice | Forbidden production/security/live-provider readiness overclaim remains denied. |
| VS4-REG-004 | previous_slice | Prompt-injection guard remains verified through Slice 002. |
| VS4-REG-005 | previous_slice | Reference images remain visual guidance only. |
| VS4-REG-006 | previous_slice | No admin/connector/ontology-first default remains verified. |
| VS4-REG-007 | previous_slice | CLI parity remains verified for selected feature rows. |
| VS4-H01 | human_required | JiYong/Tars Product Alpha UX acceptance remains human-only. |

## Proof Needed

The selected slice can be marked `PASS` only with:

- browser proof JSON, DOM, and screenshot showing Ask, packs, required states, Search, Artifact, and Home markers;
- browser-collected `window.__cornerstoneVs4BriefEvidence()` state showing Ask created evidence-backed work-item refs and every pack produced Brief, Claim, Memory/Wiki candidate, Action Card draft, and follow-up refs;
- native CLI transcripts using `cornerstone conversation start`, `cornerstone conversation answer`, `cornerstone conversation promote`, and the existing Artifact/Search/Evidence/Brief/Claim/Memory/Mission/Action/Audit commands;
- nested fresh `cornerstone scenario verify vs0-operator-acceptance-ui --json` evidence for `VS4-REG-001`;
- nested fresh `cornerstone scenario verify vs1-ontology-suggest-promote --json` evidence for `VS4-REG-002`;
- negative evidence counters for chatbot-only Ask output, missing pack outputs, missing page states, reference-image PASS misuse, live writeback, hidden memory, and stale regression reuse;
- `scripts/verify_sot_docs.sh`, `scripts/verify_cli_native_first_docs.sh`, `scripts/verify_design_system_docs.sh`, canonical matrix verification, `git diff --check`, and targeted unit tests.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED`.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.
- These gates do not block this local Product Alpha Slice 003 proof.

## Done Criteria

This slice is done when:

1. Ask can create an evidence-backed work item and expose Brief, Claim, Memory/Wiki candidate, Action Card, evidence, and audit refs;
2. Ask proof shows evidence/memory refs and is not chatbot-only output;
3. Personal Research, Company Policy Review, and Operations Issue each produce the full local loop outputs;
4. Home, Search, and Artifact surfaces align with light calm shell, small nav, prominent search, original/source primacy, trust/evidence chips, and progressive evidence;
5. required page states are visible or explicitly mapped for major Product Alpha pages;
6. VS0 operator acceptance UI regression passes freshly on this tree;
7. VS1 ontology suggest/review/promote regression passes freshly on this tree;
8. reference images remain design guidance only and are not used as PASS evidence;
9. all AI-verifiable VS4 rows can pass when the verifier is run without a scenario filter;
10. full VS4 remains blocked only by `VS4-H01` human acceptance and conditional deferred production/on-prem/security/live-provider gates.

## CLI Parity

- Ask: `cornerstone conversation start --message <question-or-input> --json`; `cornerstone conversation answer <conversation_id> --question <question> --json`.
- Ask promotion: `cornerstone conversation promote <conversation_id> --kind claim --statement <statement> --evidence-bundle-id <id> --json`.
- Source / Artifact: `cornerstone artifact ingest <path> --json`; `cornerstone artifact show <artifact_id> --json`; `cornerstone artifact download <artifact_id> --output <path> [--force] --json`.
- Search: `cornerstone search query <query> --json`.
- Evidence: `cornerstone evidence bundle create --search-snapshot-id <id> --json`.
- Brief: `cornerstone brief create --evidence-bundle-id <id> --json`; `cornerstone brief show <brief_id> --json`.
- Memory/Wiki candidate: `cornerstone memory create --evidence-bundle-id <id> --statement <text> --status draft --trust-state draft --json`; `cornerstone wiki show --json`.
- Action Card: `cornerstone mission create --goal <goal> --claim-id <id> --evidence-bundle-id <id> --json`; `cornerstone action propose --mission-id <id> --claim-id <id> --goal <goal> --action-kind draft_task --risk medium --json`; `cornerstone action dry-run <action_id> --json`.
- Regression: `cornerstone scenario verify vs0-operator-acceptance-ui --json`; `cornerstone scenario verify vs1-ontology-suggest-promote --json`.
- Verification: `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario ... --json`.
- CLI status: `PASS` only when the VS4 scenario report includes successful transcripts for selected feature families and denial/negative-evidence counters remain zero.

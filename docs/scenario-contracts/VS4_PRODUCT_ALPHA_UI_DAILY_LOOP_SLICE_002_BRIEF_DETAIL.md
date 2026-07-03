# CornerStone VS4 Slice 002 Evidence-Backed Brief Detail

**Date:** 2026-07-03 KST
**Owner:** JiYong / Tars
**Status:** Frozen implementation-slice contract. This slice can prove local Product Alpha Brief detail behavior only; it does not complete full VS4.
**Parent contract:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md`
**Parent matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Previous slice:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_SLICE_001_PRODUCT_SHELL.md`

## Goal

Turn the Slice 001 Product Alpha shell into a concrete local Brief workbench:

```text
Drop / Ask
-> preserved Source / Artifact
-> Evidence-backed Brief
-> Claim candidate
-> Memory/Wiki candidate
-> Action Card draft
-> Evidence/Audit detail
```

The Brief must be the central product object. Evidence, gaps, claim trust state, memory review state, and action preview must be visible without claiming production, on-prem, final security, live-provider, or final human UX readiness.

## Scope

In this slice:

- expose local runtime API routes for creating and reading Briefs from existing Evidence Bundles;
- add a VS4 Brief detail/workbench surface to the local UI;
- make the existing Drop/Ask shell able to run a deterministic local Brief workflow for browser proof;
- show source preservation, evidence-backed summary, supported finding, gaps/uncertainty, Claim candidate, Memory/Wiki candidate, Action Card draft, Shared Evidence Drawer, and activity/audit detail;
- verify state through native CLI transcripts for Artifact, Search, Evidence Bundle, Brief, Claim, Memory, Mission, Action, dry-run, and Audit;
- verify zero-evidence Claim approval denial;
- verify draft Memory/Wiki candidate cannot act as approved durable truth before review;
- verify Action Card defaults to local/mock and real external calls stay at zero;
- verify prompt-injection fixture cannot approve memory/action, create authority, execute action, or cause egress;
- keep reference images as visual direction only, never PASS evidence.

## Non-Scope

This slice does not implement:

- final multi-pack VS4 E2E coverage for all three general-purpose packs;
- full Ask-to-work-item conversation promotion;
- final state coverage for every major page;
- production storage migration, on-prem deployment, live provider writeback, real IdP, or real network proof;
- final human Product Alpha UX acceptance.

## Assumptions

- Existing `LocalRuntimeStore` methods for `create_brief_from_evidence_bundle`, Claim, Memory, Mission, Action, and Audit are authoritative local product state for this slice.
- A draft memory created with `cornerstone memory create --status draft --trust-state draft` is a reviewable Memory/Wiki candidate, not approved durable truth.
- The UI may use deterministic local fixture content for the first VS4 Brief proof, as long as the verifier also inspects CLI/runtime state and audit evidence.
- Reference images under `docs/design/reference-images/` guide layout and interaction language only.

## Selected Scenarios

This slice selects these AI-verifiable rows:

| ID | Classification | Why |
|---|---|---|
| VS4-UI-002 | in_this_slice | Source / Artifact preservation is required before Brief generation. |
| VS4-UI-003 | in_this_slice | Brief creation from evidence is the central Slice 002 behavior. |
| VS4-UI-004 | in_this_slice | Brief detail must show summary, findings, gaps, candidates, evidence, and activity. |
| VS4-UI-005 | in_this_slice | Shared Evidence Drawer is required for evidence-linked findings. |
| VS4-UI-006 | in_this_slice | Claim candidate must remain evidence-aware and scoped. |
| VS4-UI-007 | in_this_slice | Zero-evidence Claim approval must be denied. |
| VS4-UI-008 | in_this_slice | Memory/Wiki candidate must be visible as draft / needs review. |
| VS4-UI-009 | in_this_slice | Draft memory cannot become hidden approved durable truth. |
| VS4-UI-010 | in_this_slice | Action Card draft must show goal, why, evidence, impact, risk, approval, execution mode, and activity. |
| VS4-UI-011 | in_this_slice | Action execution mode must be local/mock by default with no live writeback. |
| VS4-REF-002 | in_this_slice | Claim and Action surfaces are reviewed against reference direction using implementation/browser evidence only. |
| VS4-REG-004 | in_this_slice | Prompt-injection content must not create authority, memory approval, action execution, or egress. |
| VS4-REG-005 | in_this_slice | Reference images remain visual guidance only and are not PASS evidence. |
| VS4-REG-007 | in_this_slice | New VS4 feature PASS requires native CLI transcripts. |

Slice 001 rows remain `PASS` only when the verifier reruns and confirms their existing evidence. Non-selected rows remain `NOT_RUN` or `HUMAN_REQUIRED`.

## Full Scenario Classification

| ID | Classification | Slice Status |
|---|---|---|
| VS4-GATE-001 | previous_slice | Contract and matrix freeze verified by Slice 001 and rerun by the VS4 verifier. |
| VS4-UI-001 | previous_slice | Product Alpha Home shell remains first screen. |
| VS4-UI-002 | in_this_slice | Verify source preservation through CLI/API/browser state. |
| VS4-UI-003 | in_this_slice | Verify Brief creation from Evidence Bundle. |
| VS4-UI-004 | in_this_slice | Verify Brief detail content and evidence linkage. |
| VS4-UI-005 | in_this_slice | Verify Shared Evidence Drawer source/snippet/provenance/activity/audit. |
| VS4-UI-006 | in_this_slice | Verify Claim candidate statement, trust state, rationale, evidence, gaps, related Brief, and activity. |
| VS4-UI-007 | in_this_slice | Verify zero-evidence Claim approval denial and unchanged unsupported Claim authority. |
| VS4-UI-008 | in_this_slice | Verify draft Memory/Wiki candidate with source, trust, freshness, owner/workspace, and controls. |
| VS4-UI-009 | in_this_slice | Verify no approved durable memory is created before review. |
| VS4-UI-010 | in_this_slice | Verify Action Card draft detail. |
| VS4-UI-011 | in_this_slice | Verify local/mock execution mode and zero real external calls. |
| VS4-UI-012 | previous_slice | Ops Inbox returning-work shell remains visible. |
| VS4-UI-013 | later_slice | Requires Ask-to-work-item flow and state proof. |
| VS4-UI-014 | later_slice | Requires three general-purpose packs through the full VS4 loop. |
| VS4-UI-015 | previous_slice | Workspace context remains visible. |
| VS4-UI-016 | previous_slice | Product language remains first. |
| VS4-STATE-001 | later_slice | Requires state coverage across all major pages. |
| VS4-REF-001 | later_slice | Requires Home/Search/Artifact reference mapping beyond Slice 001 shell proof. |
| VS4-REF-002 | in_this_slice | Verify Claim/Action trust ladder, evidence, dry-run, policy/risk, approval, and local execution mode. |
| VS4-REG-001 | later_slice | Requires fresh VS0 operator acceptance rerun after this slice. |
| VS4-REG-002 | later_slice | Requires fresh VS1 ontology rerun after this slice. |
| VS4-REG-003 | previous_slice | No forbidden readiness overclaim remains enforced by the verifier. |
| VS4-REG-004 | in_this_slice | Verify prompt-injection negative evidence. |
| VS4-REG-005 | in_this_slice | Verify reference images are not used as PASS evidence. |
| VS4-REG-006 | previous_slice | Product-first default Home remains enforced by the verifier. |
| VS4-REG-007 | in_this_slice | Verify CLI transcript coverage for selected feature rows. |
| VS4-H01 | human_required | JiYong/Tars Product Alpha UX acceptance remains human-only. |

## Proof Needed

The selected slice can be marked `PASS` only with:

- native CLI transcripts covering Artifact, Search, Evidence Bundle, Brief, Claim, Memory, Mission, Action, dry-run, and Audit paths;
- browser proof JSON, DOM, and screenshot for the VS4 Brief workbench;
- browser-collected `window.__cornerstoneVs4BriefEvidence()` state from the deterministic local UI flow;
- source references in `packages/cornerstone_cli/product_runtime.py`, `packages/cornerstone_cli/runtime.py`, `packages/cornerstone_cli/main.py`, and `packages/cornerstone_cli/scenarios.py`;
- negative evidence counters for prompt injection, hidden memory approval, forbidden readiness claims, live external writeback, and reference-image PASS misuse;
- `scripts/verify_sot_docs.sh`, `scripts/verify_cli_native_first_docs.sh`, `scripts/verify_design_system_docs.sh`, canonical matrix verification, `git diff --check`, and targeted unit tests.

## Human / Conditional Deferred Gates

- `VS4-H01` remains `HUMAN_REQUIRED`.
- `VS3-H01` through `VS3-H07` remain Conditional Deferred Gates for production/on-prem/security/live-provider/human acceptance claims.
- These gates do not block this local Product Alpha Brief detail slice.

## Done Criteria

This slice is done when:

1. the local UI exposes a Brief detail/workbench surface reachable from Home / Ops Inbox;
2. source preservation and searchable state are visible before generated Brief content;
3. the Brief shows supported findings, gaps, Claim candidate, Memory/Wiki candidate, Action Card draft, evidence, and activity;
4. Shared Evidence Drawer resolves source, snippet, provenance, related objects, activity, and audit detail;
5. Claim candidate uses Draft -> Evidence-backed -> Approved language and zero-evidence approval is denied;
6. Memory/Wiki candidate is draft / needs review and not approved durable memory before review;
7. Action Card shows goal, why, evidence, impacted objects, proposed change, expected impact, risk, safety check, approval state, local/mock execution mode, and activity;
8. prompt-injection fixture creates zero approved memory, zero approved claim, zero executed action, zero egress, and zero authority expansion;
9. the VS4 verifier reports selected Slice 002 rows as `PASS` when filtered to this slice;
10. full VS4 remains `NOT_COMPLETE` until every AI row passes and `VS4-H01` has human acceptance evidence.

## CLI Parity

- Source / Artifact: `cornerstone artifact ingest <path> --json`.
- Search: `cornerstone search query <query> --json`.
- Evidence: `cornerstone evidence bundle create --search-snapshot-id <id> --json`.
- Brief: `cornerstone brief create --evidence-bundle-id <id> --json`; `cornerstone brief show <brief_id> --json`.
- Claim: `cornerstone claim create --evidence-bundle-id <id> --statement <text> --json`; `cornerstone claim approve <claim_id> --json`.
- Memory/Wiki candidate: `cornerstone memory create --evidence-bundle-id <id> --statement <text> --status draft --trust-state draft --json`; `cornerstone memory show <memory_id> --json`; `cornerstone wiki show --json`.
- Action Card: `cornerstone mission create --goal <goal> --claim-id <id> --evidence-bundle-id <id> --json`; `cornerstone action propose --mission-id <id> --claim-id <id> --goal <goal> --action-kind draft_task --risk medium --json`; `cornerstone action dry-run <action_id> --json`; `cornerstone action show <action_id> --json`.
- Audit: `cornerstone audit verify --json`.
- Verification: `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario ... --json`.
- Exit codes covered: success, missing evidence/trust-state violation, and safe local denial paths.
- CLI status: `PASS` only when the VS4 scenario report includes successful transcripts for every selected feature family and denial transcript for zero-evidence approval.

# CornerStone VS4 Product Alpha UI Daily Loop Contract

**Date:** 2026-07-02 KST
**Owner:** JiYong / Tars
**Status:** Frozen task-scoped scenario contract for VS4 documentation and future implementation planning. This is not implementation evidence.
**Matrix:** `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv`
**Canonical milestone name:** `VS4_PRODUCT_ALPHA_UI_DAILY_LOOP`

## Summary

VS4 is CornerStone's Product Alpha UI milestone.

It turns the verified local evidence/action engine into a daily-use product shell centered on:

```text
Drop / Ask
-> Evidence-backed Brief
-> Claim
-> Memory / Wiki
-> Action Card
-> Ops Inbox
-> Evidence / Audit
-> Learn
```

This document is a scenario contract, not an implementation report. AI-verifiable rows begin as `NOT_RUN`. The human UX acceptance row remains `HUMAN_REQUIRED` until JiYong/Tars records dated product-alpha acceptance evidence.

## Source Basis

- User-provided VS4 Product Alpha UI Daily Loop draft: `/Users/jiyong/.codex/attachments/503d342a-7ae6-400d-b914-3d07ba2c8330/pasted-text.txt`.
- `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md`: CornerStone is an Evidence-first Operational Intelligence Platform and one coherent product experience.
- `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`: canonical long-term scenarios, especially `CS-PROD-*`, `CS-CLAIM-*`, `CS-MEM-*`, `CS-AUTO-*`, `CS-SEC-*`, and `CS-REG-*`.
- `docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md`: no CLI, no feature PASS.
- `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md`: deterministic local scenario verification, CLI-native evidence, UI evidence, and negative evidence requirements.
- `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`: Calm Surface. Deep Evidence. Safe Action.
- `docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md` and `docs/design/tokens/cornerstone_design_tokens_v0_3.json`: visual concept, navigation, component, state, and token guidance.
- `docs/design/reference-images/README.md`: reference images are design inputs only, not PASS evidence.
- `docs/scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_CONTRACT.md`: accepted local UI proof boundary for the VS0 Artifact/Search/Evidence/Claim/Action/Audit loop.
- `docs/scenario-contracts/VS1_ONTOLOGY_AUTO_SUGGEST_PROMOTE_CONTRACT.md`: accepted product direction that understanding must start from Drop / Ask / Search, not forced ontology modeling.
- `docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md` and `docs/verification-reports/VS3_LOCAL_DEV_ASSURANCE_CHECKPOINT_2026-06-29.md`: local/dev assurance and production/on-prem readiness remain separate claims.

## Current Evidence Boundary

The strongest current claim available to VS4 planning is:

```text
VS3-L local/dev assurance: verified in current reports.
VS3-P production/on-prem readiness: NOT_CLAIMED.
VS3-H01 through VS3-H07: HUMAN_REQUIRED.
```

VS4 may proceed as local Product Alpha UI planning and future local implementation work. It must not claim production readiness, on-prem readiness, final security acceptance, real IdP readiness, real network readiness, live-provider readiness, migration/restore readiness, or completed human UX acceptance.

## Feature / Task

`VS4_PRODUCT_ALPHA_UI_DAILY_LOOP`

## Goal

Create a product-alpha UI contract where a user can start daily work, preserve messy input, get an evidence-backed brief, review claims and memories, prepare safe action cards, return through Ops Inbox, inspect evidence/audit detail, and feed outcomes back into learning.

The user should experience CornerStone as one product, not as scenario verification tooling, an admin console, an ontology editor, a connector product, or a generic chatbot.

## Success Criteria

VS4 Product Alpha local implementation may claim AI-verifiable PASS only when:

- every AI-owned `VS4-*` `MUST_PASS` and `REGRESSION_GUARD` row is `PASS` with concrete evidence;
- Home starts with Drop / Ask / Continue and does not require connector, ontology, model, or policy setup before first value;
- Evidence-backed Brief is a P0 product behavior and shows findings, evidence, uncertainty/gaps, next steps, claim candidates, memory candidates, and suggested actions;
- Claims, Memory/Wiki, and Action Cards preserve evidence, trust state, review, approval, local/mock execution boundaries, and auditability;
- normal user navigation remains small: `Home`, `Search`, `Artifacts`, `Claims`, `Actions`;
- `Briefs`, `Memory/Wiki`, and `Audit` are reachable through Home/Ops Inbox, detail pages, evidence drawers, secondary views, or contextual routes without making the default navigation admin-heavy;
- three general-purpose fixture packs use the same UI loop without logistics-specific assumptions;
- VS0 and VS1 accepted local loops remain passing on the same source tree;
- reference images guide layout and state patterns but are not cited as implementation PASS evidence;
- no VS4 UI, report, README, or help copy claims production, on-prem, security, live-provider, or human UX readiness.

## Constraints

### Product and UX

- Preserve one CornerStone product experience.
- Product Alpha starts from daily user value: Drop / Ask / Continue.
- Use calm, precise, evidence-aware language.
- Do not make the first screen a connector setup page, scenario verifier, ontology editor, policy console, or admin dashboard.
- Keep admin connector, policy, audit, registry, and security controls in admin/operator context or progressive disclosure.
- Trust, risk, approval, and policy states must use text plus color, never color alone.

### Data and State

- Preserve original input as Source / Artifact before derived processing or generated output.
- Generated briefs, claims, memories, action cards, and learning candidates must carry evidence, provenance, trust state, and owner/workspace scope where applicable.
- Drafts may exist without full evidence; approved claims, durable memory, and action-driving outputs require evidence or explicit policy.
- This contract does not decide whether Brief is a new first-class table/object or a derived view over existing state. Future implementation must choose the smallest design that satisfies the scenarios.

### Permission and Safety

- Treat uploaded, connected, retrieved, and generated content as untrusted evidence.
- No hidden durable memory write.
- No evidence-free approved claim.
- No live external writeback by default.
- Action Cards default to draft/local/mock unless a later scenario explicitly verifies live-provider readiness.
- External or risky action requires dry-run, policy decision, approval when required, execution result, and audit.
- Prompt-injection content cannot grant authority, approve memory, approve claims, create action execution, or bypass policy.
- Secrets must not appear in logs, screenshots, reports, generated docs, durable memory, or CLI/browser transcripts.

### CLI and Verification

- No UI/API-only feature PASS. Product state and product behavior exposed through VS4 UI must have a native `cornerstone ... --json` path, unless the row is explicitly presentation-only or `HUMAN_REQUIRED`.
- Local scenario PASS must be validated by deterministic checks over durable evidence, UI traces, CLI transcripts, state records, policy decisions, action records, and audit records. LLM output alone never proves PASS.
- Browser/UI evidence is required for UI-facing scenarios, but subjective product-alpha acceptance remains `HUMAN_REQUIRED`.

### Operational

- Local/CI verification must not require production access, real IdP accounts, live provider credentials, broad network access, paid model providers, or real customer data.
- No production dependency, broad architecture migration, destructive command, release tag, or source-system mutation is part of this documentation setup.

## Assumptions

- VS4 can remain local/dev mode by default.
- Existing VS0, VS1, and VS3 local proof surfaces are inputs for planning, not proof that VS4 behavior is already implemented.
- Reference images are available under `docs/design/reference-images/`.
- Browser-based verification is feasible in the target development environment for future implementation.
- No Notion mirror is required unless separately requested.

## Out of Scope

- Final VS3-H01 through VS3-H07 evaluation.
- Production/on-prem release readiness.
- Final security acceptance.
- Real IdP / SSO readiness.
- Real network, DNS, proxy, firewall, service mesh, or sandbox readiness.
- Live ConnectorHub/provider writeback or live-provider readiness.
- Human product-alpha UX acceptance without an explicit JiYong/Tars review record.
- Full Tool SDK marketplace, signed registry ecosystem, or Agent Pack production rollout.
- Logistics-only product identity.
- Broad code implementation or storage migration as part of this documentation setup.

## Product Loop Contract

VS4 uses this daily loop:

| Stage | User-facing meaning | Required guard |
|---|---|---|
| Drop / Ask | User adds messy input or asks what CornerStone knows. | Preserve original Source / Artifact; active workspace visible. |
| Evidence-backed Brief | CornerStone summarizes findings, gaps, evidence, and next steps. | Every supported finding has evidence or an uncertainty label. |
| Claim | User can turn a supported finding into a Claim candidate. | Evidence-free approval is blocked. |
| Memory / Wiki | User reviews what should become durable memory or wiki knowledge. | Draft / Needs Review before durable truth; correction and rollback path visible. |
| Action Card | User reviews proposed action, impact, risk, dry-run, approval state, and execution mode. | No live writeback by default; policy, evidence, and audit remain attached. |
| Ops Inbox | Returning user sees pending briefs, evidence gaps, claims, memories, actions, approvals, and recent activity. | Work remains scoped to active workspace and trust state. |
| Evidence / Audit | User can inspect source, provenance, policy decisions, activity, and audit trail. | Evidence and audit are progressive but one clear action away. |
| Learn | Outcomes, corrections, approvals, rejections, and failures become future evidence or learning candidates. | Learning remains owner-scoped and proposal/review based where it can change durable behavior. |

## UI Surface Contract

### Default normal-user navigation

```text
Home
Search
Artifacts
Claims
Actions
```

### Reachable product surfaces

- `Evidence-backed Brief`: reachable from Drop / Ask, search results, artifact detail, Ops Inbox, and recent work.
- `Memory / Wiki`: reachable from Brief, Claim, Ops Inbox, and workspace context; may be a secondary nav or contextual surface, not required in default left nav.
- `Evidence / Audit`: reachable through Evidence Drawer, Action Card, activity timeline, and operator/admin detail.
- `Admin Connectors / Policies`: admin context only; not the default user experience.

### Required P0 pages or components

- Home / Universal Workspace.
- DropZone and AskBox.
- Continue Work / Ops Inbox sections.
- Evidence-backed Brief detail.
- Evidence Drawer.
- Claim candidate/detail with trust ladder.
- Memory candidate and basic Wiki/Memory review surface.
- Action Card with dry-run/preview, impact, risk, approval, execution mode, and activity.
- Workspace context indicator.
- Recent Activity / Audit detail entry point.

### Required page states

Every major VS4 page must define these states:

- empty;
- loading;
- ready;
- partial / degraded;
- needs review;
- permission denied;
- policy blocked;
- failed with recovery;
- audit/log available.

Failure states must explain what happened, why it happened, what stayed safe, and what the user can do next.

## Product Language Rules

VS4 UI should show product language first and internal language second.

| Internal / proof language | VS4 product language |
|---|---|
| Artifact | Source / Evidence item |
| Evidence Bundle | Supporting evidence |
| Search Snapshot | Saved search / Evidence snapshot |
| Policy Decision | Safety check |
| Audit Ref | Activity record |
| Mock Connector Call | Local-only action preview |
| Namespace | Workspace |
| Ontology Suggestion | Suggested evidence map / Found object |
| Promotion | Save to knowledge map |
| Derived Representation | Extracted text / Parsed content |
| Production release false | Local mode / No live writeback yet |

Internal IDs, audit refs, policy refs, evidence bundle IDs, scenario IDs, and raw connector details remain available in detail drawers, developer/operator mode, CLI output, or verification reports.

## CLI Parity

- Command group: `cornerstone scenario|workspace|artifact|search|evidence|brief|claim|memory|wiki|action|approval|audit|release`.
- Future scenario verification command: `cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-YYYY-MM-DD.json`.
- Future scenario gate command: `cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-YYYY-MM-DD.json --json`.
- Read commands: `cornerstone workspace current --json`, `cornerstone artifact show <artifact_id> --json`, `cornerstone artifact download <artifact_id> --output <path> [--force] --json`, `cornerstone brief show <brief_id> --json`, `cornerstone memory list --workspace <workspace_id> --json`, `cornerstone action show <action_id> --json`, `cornerstone audit list --workspace <workspace_id> --json`.
- Create/update commands: `cornerstone artifact ingest <path> --json`, `cornerstone brief create --from-artifact <artifact_id> --json`, `cornerstone claim create --brief <brief_id> --json`, `cornerstone memory propose --from-brief <brief_id> --json`, `cornerstone action create --from-claim <claim_id> --json`.
- Artifact-original download is a scoped local read/export: it writes only the requested local output and a read-audit event, does not mutate the Artifact or call an external system, and therefore has no dry-run. JSON emits scope, Artifact/checksum IDs, size/media type, evidence refs, and audit refs; exit codes are `0` success, `1` invalid/existing output, `3` unavailable (missing or outside the requested scope, intentionally non-disclosing), and `5` integrity/write failure. Deterministic transcript: `tests/scenario/test_scaffold_cli.py::ScaffoldCliTests.test_artifact_ingest_show_and_audit_verify`.
- Dry-run command, if mutating: `cornerstone action dry-run <action_id> --json`.
- Approval/execution command, if applicable: `cornerstone action approve <action_id> --json`; `cornerstone action execute <action_id> --json`.
- JSON schema path: implementation-owned scenario report, CLI transcript, browser proof, UI assertions, and product object schemas must be referenced from the future verification report.
- Exit codes covered: success, invalid input, not found/conflict, permission/policy denial, missing evidence/trust-state violation, human approval required, connector/live-provider unavailable, prompt-injection/egress/secret block.
- Workspace/namespace scope: tenant, owner, namespace, and workspace must be visible or inspectable in `--json` output where product state is read or changed.
- Evidence refs emitted: artifact/source refs, search snapshot refs, evidence refs, brief refs, claim refs, memory refs, action refs, policy decision refs, and audit refs where applicable.
- Audit refs emitted: every state-changing, policy-relevant, memory-relevant, trust-state, approval, action, and execution operation.
- Same backend path evidence: future implementation must show UI/API/CLI use the same Product, Archive/Evidence, Workflow/Action, Policy, and Audit boundaries.
- CLI status: `NOT_RUN` until future implementation adds and verifies these paths.

## Scenario Inventory

Counts:

| Type | Count |
|---|---:|
| MUST_PASS | 20 |
| REGRESSION_GUARD | 7 |
| HUMAN_REQUIRED | 1 |
| Total | 28 |

All AI-owned rows begin as `NOT_RUN`. Human-owned rows begin as `HUMAN_REQUIRED`.

| ID | Type | Trigger / Action | Expected Result | Verification Method | Evidence Required | Owner | Initial Status |
|---|---|---|---|---|---|---|---|
| VS4-GATE-001 | MUST_PASS | VS4 contract and matrix are frozen. | Markdown and CSV exist, counts match 20/7/1, every row has observable behavior, verification method, evidence, pass/fail criteria, owner, and initial status. | Source review, duplicate-ID check, row-count check, docs verifiers. | Contract path, matrix path, row-count output, `scripts/verify_sot_docs.sh`, `git diff --check`. | AI | NOT_RUN |
| VS4-UI-001 | MUST_PASS | User opens Product Alpha Home in local/dev mode. | Home shows Drop, Ask, Continue work, pending work, knowledge states, and recent activity without forcing admin/proof setup. | Browser observation plus source review. | Browser screenshot/DOM, route/source refs, scenario report row. | AI | NOT_RUN |
| VS4-UI-002 | MUST_PASS | User drops, uploads, or pastes messy input. | Original input is preserved as Source / Artifact and the UI confirms saved/searchable or partial/degraded status. | Browser flow plus API/state inspection. | Artifact/source record, checksum/provenance, UI proof, audit refs. | AI | NOT_RUN |
| VS4-UI-003 | MUST_PASS | Dropped or pasted input completes first processing. | An Evidence-backed Brief is created or prepared as the central work item. | Browser flow plus state inspection. | Brief record/state, source refs, evidence refs, audit refs. | AI | NOT_RUN |
| VS4-UI-004 | MUST_PASS | User opens the Brief. | Brief shows summary, supported findings, gaps/uncertainty, claim candidates, memory candidates, suggested actions, evidence, and activity. | Browser observation plus product object inspection. | Screenshot/DOM, Brief JSON/state, evidence linkage. | AI | NOT_RUN |
| VS4-UI-005 | MUST_PASS | User opens evidence from the Brief. | Evidence Drawer shows source, snippet, why it supports the finding, provenance, related objects, activity, and audit detail. | Browser observation plus evidence record inspection. | Evidence Drawer screenshot/DOM, evidence refs, provenance refs. | AI | NOT_RUN |
| VS4-UI-006 | MUST_PASS | User creates a Claim candidate from a Brief. | Claim candidate carries statement, trust state, rationale, supporting evidence, gaps, related Brief, and activity. | Browser flow plus state inspection. | Claim record, evidence refs, trust state, audit refs. | AI | NOT_RUN |
| VS4-UI-007 | MUST_PASS | User attempts to approve a Claim without supporting evidence. | Approval is blocked or Claim remains Draft / Insufficient evidence with a cause and resolution path. | Automated negative test plus UI check. | Denial response, unchanged claim state, UI cause/resolution text, audit ref. | AI | NOT_RUN |
| VS4-UI-008 | MUST_PASS | Brief or Claim suggests durable knowledge. | Memory candidate is created as Draft / Needs Review with source, trust state, freshness, owner/workspace, and controls. | Browser flow plus state inspection. | Memory candidate record, UI proof, source/evidence refs, audit refs. | AI | NOT_RUN |
| VS4-UI-009 | MUST_PASS | Memory candidate exists before approval. | It is not hidden durable truth and cannot be used as Approved memory until reviewed or approved. | State inspection plus browser check. | Approved-memory absence before approval, approved-memory presence only after approval, audit refs. | AI | NOT_RUN |
| VS4-UI-010 | MUST_PASS | User opens an Action Card from Brief or Claim. | Action Card shows goal, why, supporting evidence, impacted objects, proposed change, expected impact, risk, safety check, approval state, execution mode, and activity. | Browser observation plus action state inspection. | Action Card screenshot/DOM, action JSON/state, policy decision refs, audit refs. | AI | NOT_RUN |
| VS4-UI-011 | MUST_PASS | User reviews Action Card execution mode in VS4 local/dev mode. | Execution mode is visible as Draft / Local / Mock by default and no live external writeback is implied. | Browser observation plus state/negative-evidence inspection. | UI proof, action state, `real_external_http_calls=0`, audit refs. | AI | NOT_RUN |
| VS4-UI-012 | MUST_PASS | Returning user opens Home / Ops Inbox with pending work. | Ops Inbox shows pending briefs, evidence gaps, claims, memory candidates, action cards, approvals, and recent activity with Continue links. | Browser observation with fixture state. | Fixture state, screenshot/DOM, route refs. | AI | NOT_RUN |
| VS4-UI-013 | MUST_PASS | User asks a question in AskBox. | Answer shows evidence/memory used and can become a Brief, Claim, Memory candidate, or Action Card without becoming generic chatbot-only output. | Browser flow plus state inspection. | Ask transcript, evidence/memory refs, created work item refs, audit refs. | AI | NOT_RUN |
| VS4-UI-014 | MUST_PASS | Three general-purpose fixture packs run through VS4 UI loop. | Personal Research, Company Policy Review, and Operations Issue packs each produce Brief, Claim candidate, Memory candidate, Action Card draft, and Ops Inbox follow-up. | Browser/E2E fixture runs. | Fixture outputs, browser traces, scenario report rows. | AI | NOT_RUN |
| VS4-UI-015 | MUST_PASS | User switches or views active workspace context. | Active workspace/owner context is visible enough to understand which context CornerStone is using. | Browser observation plus state inspection. | Workspace UI proof, JSON/state scope, zero cross-workspace ambiguity. | AI | NOT_RUN |
| VS4-UI-016 | MUST_PASS | User scans VS4 normal-user UI. | Product language appears before internal proof language; internal IDs/details are progressively disclosed. | Text scan plus browser review. | Copy scan output, browser screenshots/DOM, allowed internal-detail locations. | AI | NOT_RUN |
| VS4-STATE-001 | MUST_PASS | Major VS4 pages are exercised across common states. | Empty, loading, ready, partial/degraded, needs-review, permission denied, policy blocked, failed-with-recovery, and audit/log available states are defined and observable or documented per page. | Source review plus browser/story/state assertions. | State coverage matrix, screenshots/DOM or component/source refs. | AI | NOT_RUN |
| VS4-REF-001 | MUST_PASS | Home, Search, and Artifact surfaces are implemented or reviewed. | They follow reference direction for light calm shell, small nav, prominent search, original/source primacy, trust/evidence chips, and progressive evidence. | Design/source/browser review. | Screenshot/DOM, source refs, reference mapping notes; no PASS from images alone. | AI | NOT_RUN |
| VS4-REF-002 | MUST_PASS | Claim and Action surfaces are implemented or reviewed. | They follow reference evidence/action patterns: trust ladder, evidence picker/drawer, dry-run, diff/impact, policy/risk, approval, execution mode, and activity/audit. | Design/source/browser review. | Screenshot/DOM, source refs, reference mapping notes; no PASS from images alone. | AI | NOT_RUN |
| VS4-REG-001 | REGRESSION_GUARD | VS4 changes are applied. | Existing VS0 Artifact -> Search -> Evidence -> Claim -> Action -> Audit loop still passes on the same tree. | Existing VS0 verifier and browser/API proof rerun. | Scenario report, CLI transcript, browser/API evidence. | AI | NOT_RUN |
| VS4-REG-002 | REGRESSION_GUARD | VS4 changes are applied over accepted VS1 ontology slice. | VS1 ontology suggest/review/promote still starts from evidence and keeps suggestions draft until explicit promotion. | Existing VS1 verifier and browser/API proof rerun. | VS1 scenario report, zero auto-promotion evidence, audit refs. | AI | NOT_RUN |
| VS4-REG-003 | REGRESSION_GUARD | VS4 UI, reports, help, and README text are scanned. | No production, on-prem, final security, real IdP, real network, live-provider, migration/restore, or human UX readiness claim appears unless explicitly historical/deferred. | Static overclaim scan plus browser review. | Text scan output, browser observations, negative evidence counters. | AI | NOT_RUN |
| VS4-REG-004 | REGRESSION_GUARD | Prompt-injection artifact or Ask input attempts to approve memory/action or create authority. | No memory approval, claim approval, action execution, egress, policy change, or authority expansion occurs from untrusted content. | Security/adversarial fixture plus audit/counter checks. | Zero unauthorized counters, blocked-attempt audit, untrusted label, evidence refs. | AI | NOT_RUN |
| VS4-REG-005 | REGRESSION_GUARD | VS4 verification report cites visual references. | Reference images are treated as visual inputs only, not implementation PASS, scenario PASS, or human UX acceptance evidence. | Report lint/source review. | Lint output or source refs showing images are design references only. | AI | NOT_RUN |
| VS4-REG-006 | REGRESSION_GUARD | Normal user opens VS4 after admin, connector, policy, or ontology work exists. | Home remains product-first and not admin/connector/ontology/scenario-verifier-first. | Browser/nav review plus text scan. | Screenshots/DOM, nav map, absence of admin-first route. | AI | NOT_RUN |
| VS4-REG-007 | REGRESSION_GUARD | A VS4 feature appears in UI/API without CLI parity. | Scenario gate blocks PASS or marks feature `NOT_VERIFIED` until native CLI path and JSON transcript exist. | CLI parity matrix/report review. | CLI transcript refs, missing-command failure row, scenario report status. | AI | NOT_RUN |
| VS4-H01 | HUMAN_REQUIRED | JiYong/Tars uses the VS4 Product Alpha UI flow. | Human accepts or rejects whether the daily loop is usable, understandable, product-first, evidence-aware, and not misleading. | Human walkthrough. | Acceptance note with screenshots/recording, or rejection note with issue list. | Human | HUMAN_REQUIRED |

## Required Local Verification

Future VS4 implementation should run these commands or report exactly why any command could not run:

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-YYYY-MM-DD.json
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-YYYY-MM-DD.json --json
PATH="$PWD:$PATH" cornerstone scenario verify vs0-operator-acceptance-ui --json --output reports/scenario/vs0-operator-acceptance-ui-YYYY-MM-DD.json
PATH="$PWD:$PATH" cornerstone scenario verify vs1-ontology-suggest-promote --json --output reports/scenario/vs1-ontology-suggest-promote-YYYY-MM-DD.json
scripts/verify_sot_docs.sh
scripts/verify_cli_native_first_docs.sh
scripts/verify_design_system_docs.sh
python3 scripts/verify_scenario_matrix.py docs/scenario-contracts/SCENARIO_MATRIX_FULL.csv docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md
git diff --check
```

For this documentation setup, only documentation verification is required. Product implementation checks remain `NOT_RUN` until future code implements VS4 behavior.

## Human Required / Conditional Deferred Gates

### VS4 human acceptance

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS4-H01 | Product-alpha UX acceptance is subjective and requires JiYong/Tars to judge whether the daily loop is usable, understandable, product-first, evidence-aware, and not misleading. | JiYong/Tars completes the local VS4 walkthrough and records accept or reject. | Acceptance note with screenshots/recording, or rejection note with issue list. | Blocks product-alpha human UX acceptance claim; does not block local documentation freeze or AI-verifiable implementation planning. |

### VS3 deferred gates

| ID | Classification for VS4 | Required Later Evidence | Blocks |
|---|---|---|---|
| VS3-H01 | Conditional Deferred | Owner architecture/security approval record. | Production/security acceptance claims. |
| VS3-H02 | Conditional Deferred | Independent security review and retest evidence. | Production/security acceptance claims. |
| VS3-H03 | Conditional Deferred | Real IdP mapping and revocation transcript. | Real IdP readiness claim. |
| VS3-H04 | Conditional Deferred | Real network/security topology evidence. | On-prem network readiness claim. |
| VS3-H05 | Conditional Deferred | Redacted live ConnectorHub/provider rehearsal transcript and audit refs. | Live-provider readiness claim. |
| VS3-H06 | Conditional Deferred | Human operator UX/trust accept or reject record. | Final operator trust acceptance claim for VS3-P surfaces. |
| VS3-H07 | Conditional Deferred | Human-supervised migration/backup/restore/rollback drill evidence. | Migration/restore readiness claim. |

These rows do not block local VS4 Product Alpha documentation or implementation planning. They continue to block production/on-prem/security/live-provider claims.

## Proof Surface Matrix

| Proof Surface | Proves | Does Not Prove |
|---|---|---|
| Contract and matrix review | VS4 scenario scope is frozen and internally consistent. | UI implementation exists. |
| Source review | Components/routes/state paths exist. | Runtime behavior or user comprehension. |
| Browser/UI evidence | User can complete visible UI flows. | Backend correctness beyond inspected state. |
| CLI transcript | Native reproducible product path exists. | Subjective UX quality. |
| API/state inspection | UI action created expected product records. | Human trust acceptance. |
| Scenario report | Per-row status and evidence summary. | Production/on-prem readiness unless specifically verified. |
| Overclaim scan | Forbidden readiness claims are absent. | Semantic product quality. |
| Human review | Product-alpha UX acceptance or rejection. | Automated correctness. |

## Done Means

This documentation setup is complete when:

1. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_CONTRACT.md` exists.
2. `docs/scenario-contracts/VS4_PRODUCT_ALPHA_UI_DAILY_LOOP_MATRIX.csv` exists with 28 rows: 20 `MUST_PASS`, 7 `REGRESSION_GUARD`, and 1 `HUMAN_REQUIRED`.
3. `docs/sot/README.md` registers the VS4 contract and matrix.
4. The canonical 206-scenario matrix remains unchanged.
5. Documentation verification checks pass.

Future VS4 implementation is complete only when every AI-verifiable `VS4-*` `MUST_PASS` and `REGRESSION_GUARD` row is `PASS` with concrete evidence, no AI-owned row remains `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`, and `VS4-H01` is accepted by JiYong/Tars with explicit human evidence.

## Notion Mirror

No Notion mirror is part of this task. Local Markdown is the canonical contract.

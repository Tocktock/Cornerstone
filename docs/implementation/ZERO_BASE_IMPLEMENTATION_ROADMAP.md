# Zero-Base CornerStone Implementation Roadmap

**Date:** 2026-06-08  
**Owner:** JiYong / Tars  
**Status:** Implementation planning document, subordinate to the SoT and scenario standard

## 1. Implementation Stance

Build from zero base as one CornerStone product.

Do not begin with a big-bang repo merge. Build a clean core with scenario-first verification, then import proven behavior from existing projects through adapters.

## 2. Milestone 0 - Documentation and Scenario Foundation

Goal: prevent drift before coding.

Deliverables:

- SoT docs installed under `docs/sot/`.
- Full V2 handoff preserved under `docs/handoff/`.
- Full scenario matrix installed under `docs/scenario-contracts/`.
- Old conflicting docs archived, replaced, or explicitly marked superseded.
- `VS0_IMPLEMENTATION_CONTRACT.md` frozen.
- Scenario report template added.
- Local docs verification script added.

Exit criteria:

- No active doc claims old `project-sot.md` is the only SoT.
- Product goal points to living evidence-first autonomous operational intelligence.
- Full scenario standard contains 206 scenarios.
- First scenario contract contains 58 VS-0 scenarios.
- `scripts/verify_sot_docs.sh` passes.

## 3. Milestone 1 - One-Command Product Shell

Goal: one CornerStone app starts locally.

Deliverables:

- API gateway.
- Web shell.
- Postgres.
- Policy placeholder.
- Dev auth.
- Tenant/namespace/workspace bootstrap.
- Health/ready/OpenAPI endpoints.

Exit criteria:

- User sees one CornerStone product shell.
- Personal workspace exists.
- Audit records bootstrap events.

## 4. Milestone 2 - Archive Engine

Goal: preserve original input first.

Deliverables:

- Artifact upload API.
- SHA-256 content addressing.
- Original storage.
- Derived status model.
- Unknown input preservation.
- Basic redaction.
- Prompt-injection fixture handling.

Exit criteria:

- Original survives failed extraction.
- Duplicate content dedupes/links.
- Redacted generated output/log tests pass.

## 5. Milestone 3 - Search and Evidence Snapshots

Goal: search becomes reproducible evidence.

Deliverables:

- Postgres FTS.
- pgvector/deterministic test embeddings.
- Active workspace filters.
- SearchSnapshot.
- EvidenceBundle.

Exit criteria:

- Uploaded text is searchable.
- Search result can be saved as evidence.
- Namespace isolation tests pass.

## 6. Milestone 4 - Briefs and Claims

Goal: messy input becomes evidence-backed understanding.

Deliverables:

- Evidence-backed brief generator.
- Claim state machine: Draft, Evidence-backed, Approved, Rejected, Stale.
- Evidence viewer.
- Unsupported assertion labeling.

Exit criteria:

- Brief cites evidence.
- Claim without evidence cannot be approved.
- Claim with evidence can be promoted.

## 7. Milestone 5 - ActionCard, Workflow, Policy, Audit

Goal: claims can lead to safe internal action.

Deliverables:

- ActionCard API/UI.
- Dry-run.
- Policy decision record.
- Approval flow.
- Internal safe action execution.
- Append-only audit events.
- Tamper-detection test.

Exit criteria:

- Action cannot execute before dry-run.
- Risky/external action requires approval.
- Direct write outside Workflow/Action is denied.
- Audit lifecycle is complete.

## 8. Milestone 6 - First-Value Onboarding and Mission Control

Goal: make VS-0 usable.

Deliverables:

- Onboarding guide.
- Inbox -> Brief -> Claim -> Action -> Learn visible path.
- Mission Control with artifacts, briefs, claims, evidence gaps, actions, approvals, audit links.

Exit criteria:

- New user reaches first value without connectors/models/ontology setup.
- Product does not expose repo split.

## 9. Milestone 7 - Permanent Wiki and Memory Sovereignty v0

Goal: start living memory safely.

Deliverables:

- KnowledgeCapsule.
- MemoryEntry.
- Permanent Wiki view.
- Memory states: Draft, Evidence-backed, Approved.
- Memory source/freshness/correction history.
- Inspect/correct/demote/disable controls.

Exit criteria:

- Memory is source-aware synthesis, not raw truth.
- Personal memory does not influence org workspace unless promoted.

## 10. Milestone 8 - ConnectorHub Boundary Integration

Goal: connect source systems safely.

Deliverables:

- ConnectorHub client interface.
- Capability registry.
- Read-only ingestion fixture.
- Declared action capability path.
- Credential reference boundary.

Exit criteria:

- Read-only connector ingestion creates artifacts/evidence.
- External writeback path requires ActionCard, dry-run, policy, approval/autopilot authority, and audit.

## 11. Milestone 9 - Mission Contracts and Bounded Autonomy

Goal: introduce safe autonomy.

Deliverables:

- Workspace modes: Manual, Assist, Autopilot, Locked.
- Mission Goal Contract.
- Orchestrator agent role card.
- Specialist Agent Role Contracts.
- Pause/stop/revoke controls.

Exit criteria:

- Autopilot cannot act outside mission contract.
- Locked mode blocks autonomous actions.
- High-risk action escalates.

## 12. Milestone 10 - Experience Library and Learning

Goal: close the loop.

Deliverables:

- Mission trajectory ledger.
- After-action review.
- Autonomy scorecard.
- Experience Library.
- Candidate lessons and promotion ladder.
- Product Learning namespace separation.

Exit criteria:

- Completed/failed missions generate learning material.
- Lessons are proposed, not auto-globalized.
- Product learning cannot silently rewrite user/org truth.

## 13. Release Rule

Every milestone must produce a scenario verification report.

No milestone may claim complete without:

- applicable MUST_PASS scenarios;
- applicable REGRESSION_GUARD scenarios;
- PASS evidence for AI-verifiable items;
- human-required items clearly listed;
- failures/gaps/root causes;
- final verdict no stronger than evidence.

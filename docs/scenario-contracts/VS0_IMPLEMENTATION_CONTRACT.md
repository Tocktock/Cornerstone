# CornerStone VS-0 Implementation Contract - Strict Scenario Subset

**Date:** 2026-06-08  
**Owner:** JiYong / Tars  
**Status:** Frozen first implementation contract for zero-base CornerStone work.

## Feature / Task

Build the first zero-base CornerStone vertical slice.

## Goal

Implement the first complete product identity loop:

```text
Personal messy input
→ immutable artifact
→ searchable derived representation
→ evidence-backed brief
→ draft/evidence-backed claim
→ action card dry-run
→ approval/execution
→ audit trail
```

## Success Criteria

A new user can start CornerStone locally, create a personal workspace, drop messy input, search it, generate an evidence-backed brief, create a claim, create an action card, run dry-run, approve/execute a safe internal action, and inspect audit/evidence.

## Constraints

### Product / UX

- One visible product: CornerStone.
- Do not expose `Cornerstone`, `KnowledgeBase`, or `ConnectorHub` as required user mental models.
- First value must not require connector setup, paid model provider, organization admin setup, or ontology modeling.
- Primary path: `Inbox -> Brief -> Claim -> Action -> Learn`.

### Data / State

- Preserve original artifact before derived processing.
- Every truth-bearing object has tenant, namespace, owner, provenance, and trust state where relevant.
- Search snapshots must be reproducible evidence.
- Claims cannot become approved without evidence.

### Permission / Security

- Owner-scoped namespace by default.
- No cross-namespace context mixing.
- Default egress deny.
- Untrusted content cannot instruct the system.
- Secrets are redacted before durable generated knowledge.
- Actions require Workflow/Action path; no direct mutation by agents.

### Compatibility / Format

- Postgres-first durable data model.
- OpenAPI for public APIs.
- Deterministic local/test model provider for CI.
- ConnectorHub integration deferred behind capability interface.

### Operational / Environment

- One-command local run target.
- Tests must not require live external credentials.
- Verification report must list every scenario and evidence.

## Assumptions

- This is a zero-base implementation, not a direct merge of the existing three repos.
- Existing `Cornerstone`, `KnowledgeBase`, and `Connector-Hub` projects are references/adapters until imported through scenario-verified boundaries.
- Live external writeback is out of scope for VS-0.
- Full permanent wiki/memory sovereignty is started later after artifact/evidence/claim/action/audit foundation passes.

## Non-negotiable scenario rule

The full scenario source is `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`. This VS-0 contract is only the first implementation subset. The full scenario standard remains the product acceptance standard and must not be deleted, summarized away, or replaced by this subset.

## Out of scope for VS-0

- Live external connector writeback.
- Full Agent Pack registry.
- Full Memory Sovereignty Center.
- Full mission trajectory / Experience Library implementation.
- Full provider-neutral multi-brain routing.
- Production deployment hardening beyond local/on-prem quickstart.

## Applicable VS-0 scenario IDs

Total VS-0 scenario IDs: **58**.

| ID | Type | Expected Result | Verification Method / Evidence Required |
|---|---|---|---|
| CS-PROD-001 | MUST_PASS | The product presents a single CornerStone experience. Archive/evidence, connector/action, and intelligence/mission capabilities may have clear internal boundaries, but daily users do not need to understand repository or subsystem boundaries to get value. | First-run UI or product walkthrough showing one service, one navigation model, and clear capability language. |
| CS-PROD-002 | MUST_PASS | The product behaves like an evidence-first operational intelligence platform: it preserves information, produces evidence-backed understanding, supports claims, and can lead toward action and learning. It must not feel like only a chatbot, file search app, connector framework, or automation script runner. | End-to-end demo transcript or test walkthrough showing evidence, claim, action, and learning surfaces. |
| CS-PROD-003 | MUST_PASS | The same core loop works without requiring a logistics-specific model. Solution packs may specialize workflows, but the universal core remains domain-agnostic. | Fixture runs or demos across at least three domains using the same artifact, search, claim, mission, and evidence concepts. |
| CS-PROD-004 | MUST_PASS | Within the first-use flow, CornerStone produces an evidence-backed brief, shows uncertainty or evidence gaps, and offers a next step such as draft claim, save capsule, or open mission. | Timed local/onboarding scenario or recorded UI flow using fixture input. |
| CS-PROD-005 | MUST_PASS | The product guides the user through the first useful path: personal messy input → evidence-backed brief → draft claim or saved capsule. The user is not forced to configure connectors, model providers, ontology, or organization policies before seeing value. | Onboarding flow screenshot, browser walkthrough, or E2E test. |
| CS-ARCH-001 | MUST_PASS | CornerStone preserves the original input as an immutable artifact before attempting extraction, summarization, embedding, normalization, or ontology mapping. | Stored artifact record with stable ID, checksum/hash, source, timestamp, and original storage reference. |
| CS-ARCH-002 | MUST_PASS | The original artifact remains stored and discoverable with a partial/failed derived-processing status. The user sees a helpful explanation and can retry or reprocess later. | Fixture with intentionally failing derived processor; original remains accessible. |
| CS-ARCH-003 | MUST_PASS | CornerStone recognizes unchanged content through stable identifiers or hashes and avoids creating conflicting duplicate truth. Changed content is versioned or recorded as a distinct artifact with lineage. | Re-ingestion test showing same ID for identical content or explicit version/link for changed content. |
| CS-ARCH-004 | MUST_PASS | The user can inspect where it came from, when it was created or ingested, what transformed it, and which source or connector produced it. | Provenance panel, API response, or exported evidence bundle. |
| CS-ARCH-005 | MUST_PASS | CornerStone still stores the immutable original, marks the format/derived state accurately, and allows later parser support without requiring re-upload from the user. | Unknown-format fixture with preserved original and deferred processing status. |
| CS-ARCH-006 | MUST_PASS | Secrets are redacted before generated drafts, memory entries, evidence summaries, logs, screenshots, or reports persist or display them unnecessarily. Raw original access remains controlled by policy. | Secret-containing fixture; generated outputs and logs show redacted values. |
| CS-ARCH-007 | MUST_PASS | CornerStone treats the document as untrusted evidence. It may summarize or cite the content, but it does not follow embedded instructions or trigger tools/actions from the document. | Prompt-injection fixture showing blocked tool/action execution and correct untrusted labeling. |
| CS-ARCH-008 | MUST_PASS | The query, filters, result snapshot, and relevant artifacts can be attached to an Evidence Bundle so the claim is reproducible. | Claim evidence bundle containing stored search query and result references. |
| CS-ARCH-009 | MUST_PASS | The user can open both the derived representation and the original source that supports the evidence. | Evidence viewer showing original plus derived text/metadata. |
| CS-UND-001 | MUST_PASS | CornerStone returns the artifact or derived representation quickly enough for the first-use experience, with relevant snippets and evidence-ready references. | Timed fixture upload and search output. |
| CS-UND-002 | MUST_PASS | CornerStone can retrieve relevant content through keyword and semantic retrieval, while showing enough evidence for the user to inspect why a result was returned. | Search test with exact-match and semantic-match queries. |
| CS-UND-003 | MUST_PASS | Results come from the active workspace and allowed inherited/referenced context only. CornerStone does not silently mix unrelated namespaces. | Search result comparison across workspaces with controlled fixture data. |
| CS-UND-004 | MUST_PASS | The user can inspect the original artifact, derived text or structured extraction, metadata, source, evidence references, and related claims/missions. | UI walkthrough or API response for artifact detail. |
| CS-UND-005 | MUST_PASS | CornerStone still produces search results and evidence-backed briefs. Ontology suggestions appear later as optional promotions, not as a prerequisite. | First-use scenario with no preconfigured ontology beyond universal defaults. |
| CS-CLAIM-001 | MUST_PASS | The user can work naturally in conversation without manually creating a case, mission, ontology, or document first. | Conversation flow where the user reaches a brief and claim without pre-modeling. |
| CS-CLAIM-002 | MUST_PASS | CornerStone produces a concise brief with key points, evidence links, uncertainty, contradictions, and recommended next steps. | Brief output linked to source evidence. |
| CS-CLAIM-003 | MUST_PASS | CornerStone suggests durable outputs such as Mission Card, Knowledge Capsule, Claim, Action Card, Memory, or Playbook Candidate. It does not force conversion. | Conversation transcript with suggested promoted outputs. |
| CS-CLAIM-004 | MUST_PASS | CornerStone creates the chosen durable object with source conversation reference, evidence, owner namespace, trust state, and provenance. | Promoted object detail showing source conversation and evidence bundle. |
| CS-CLAIM-005 | MUST_PASS | CornerStone clearly labels the trust state as Draft, Evidence-backed, or Approved. The user can tell what the item can and cannot be used for. | UI/API examples for all three trust states. |
| CS-CLAIM-006 | MUST_PASS | The idea may exist as Draft, but it cannot be approved, published as shared truth, or used for autonomous action without sufficient evidence or explicit risk-aware policy. | Attempt to approve unsupported draft is blocked with helpful reason. |
| CS-CLAIM-007 | MUST_PASS | CornerStone requires an Evidence Bundle, or blocks the operation and explains what evidence is missing. | Claim approval test with zero evidence denied; claim with evidence allowed. |
| CS-CLAIM-008 | MUST_PASS | The user can open supporting evidence in one click or one clear action, including source artifact and relevant excerpt/query/policy/tool result. | UI walkthrough from claim to evidence source. |
| CS-CLAIM-009 | MUST_PASS | It labels the statement as assumption, hypothesis, or insufficient evidence instead of presenting it as fact. | Fixture question with insufficient evidence producing correct labeling. |
| CS-CLAIM-010 | MUST_PASS | The claim can become a Mission Goal Contract or action proposal, carrying evidence, risk state, owner namespace, and approval requirements forward. | Claim-to-mission or claim-to-action flow. |
| CS-NS-001 | MUST_PASS | The item has an explicit owner and namespace. It is never ownerless global context. | API/database/UI records showing owner and namespace for each context item. |
| CS-NS-002 | MUST_PASS | The active workspace is always clear enough that the user understands which context CornerStone is using. | UI walkthrough showing active workspace and context boundary. |
| CS-NS-003 | MUST_PASS | CornerStone does not use personal memory in the organization workspace unless explicitly promoted, referenced, copied with provenance, or allowed by policy. | Cross-namespace isolation test with known personal-only memory. |
| CS-NS-004 | MUST_PASS | The promoted item receives organization ownership/scope, keeps provenance to the original personal item if allowed, carries evidence, and records an audit trail. | Promotion event with source, target, owner, evidence, and audit record. |
| CS-AUTO-001 | MUST_PASS | The workspace supports clear modes such as Manual, Assist, Autopilot, and Locked, with understandable behavior differences. | Mode-switch UI/API and behavior tests. |
| CS-AUTO-002 | MUST_PASS | CornerStone begins conservatively, demonstrates reliability through briefs, suggestions, internal tasks, and successful playbooks, then recommends Autopilot with a clear reason and mission contract. | Readiness recommendation based on fixture history. |
| CS-AUTO-003 | MUST_PASS | CornerStone converts the request into an editable Mission Goal Contract containing goal, scope, allowed actions, forbidden actions, success criteria, stop conditions, review cadence, escalation rules, and evidence expectations. | Generated contract with user-editable structured fields. |
| CS-AUTO-004 | MUST_PASS | The user can see what CornerStone may do, where it may act, what it may not do, what requires escalation, and how to pause or revoke authority. | Mission activation UI/API showing granted scope and restrictions. |
| CS-AUTO-005 | MUST_PASS | CornerStone executes only allowed actions inside the active owner-scoped workspace and Mission Goal Contract. It escalates high-risk, destructive, cross-namespace, sensitive, or out-of-scope actions. | Mission run with allowed auto-actions and blocked/escalated disallowed actions. |
| CS-AUTO-006 | MUST_PASS | The user sees simple governance, while hidden policy evaluates risk categories, action scope, data sensitivity, egress, approvals, and escalation. | Simple UX plus policy decision logs. |
| CS-AUTO-007 | MUST_PASS | The action appears as an Action Card showing goal, evidence, diff, expected impact, policy decision, risk, approval/execution status, and audit link. | Action Card UI/API example. |
| CS-AUTO-008 | MUST_PASS | CornerStone performs a dry-run with diff, impact, policy result, and expected external calls before execution unless the mission policy explicitly authorizes a safe low-risk auto-run. | Dry-run record linked to action execution. |
| CS-AUTO-009 | MUST_PASS | CornerStone requires owner or authorized approver review before execution. Autopilot cannot bypass this requirement. | High-risk action blocked until approval. |
| CS-AUTO-010 | MUST_PASS | CornerStone can execute these actions automatically and record the result in the mission trajectory and audit log. | Low-risk Autopilot action run with audit record. |
| CS-AUTO-011 | MUST_PASS | The agent cannot directly write to the system. It must call a Workflow/Action path mediated by policy, dry-run, connector capability, approval rules, execution result, and audit. | Direct write attempt denied; workflow-mediated write path recorded. |
| CS-SEC-001 | MUST_PASS | The system starts with minimal commands and reaches first successful upload/search/brief within the expected onboarding path. | Fresh environment quickstart log. |
| CS-SEC-002 | MUST_PASS | Egress is denied by default and the denial is recorded with helpful explanation. | External-call denial test and audit/policy log. |
| CS-SEC-003 | MUST_PASS | Access is denied unless explicitly granted by policy in a safe runtime boundary. | Sandbox escape/unauthorized access test. |
| CS-SEC-004 | MUST_PASS | CornerStone enforces access based on role, attributes, namespace, classification, mission authority, and policy. | Access-control matrix tests. |
| CS-SEC-005 | MUST_PASS | CornerStone explains the cause and provides a resolution path when safe: request access, change workspace, reduce scope, add evidence, ask approver, or change policy. | Policy denial UI/API examples. |
| CS-SEC-006 | MUST_PASS | Events are logged in a tamper-evident audit ledger with enough detail for review and verification. | Audit log query and tamper-detection test. |
| CS-SEC-007 | MUST_PASS | CornerStone treats them as evidence only, prevents tool/action hijacking, and records blocked attempts where relevant. | Prompt-injection regression suite. |
| CS-SEC-008 | MUST_PASS | CornerStone redacts or controls them according to policy and never exposes secrets in generated outputs, logs, screenshots, or reports unnecessarily. | Secret fixture redaction tests. |
| CS-REG-001 | REGRESSION_GUARD | Evidence, durable memory, claims, missions, actions, audit, and learning remain part of the product loop. | Release scenario walkthrough includes non-chat durable outputs. |
| CS-REG-002 | REGRESSION_GUARD | Search results can become evidence for claims, actions, memory, missions, or learning. | Search-to-claim/action scenario. |
| CS-REG-003 | REGRESSION_GUARD | Connector capabilities are framed as supporting user/org intelligence, evidence, missions, actions, and audit; not as the product identity. | Connector scenario mediated through mission/action/evidence flow. |
| CS-REG-004 | REGRESSION_GUARD | Universal core remains usable without logistics concepts, and at least one non-logistics scenario still passes. | General-purpose fixture scenario. |
| CS-REG-005 | REGRESSION_GUARD | Durable archive/evidence, owner-approved memory, and audit remain truth foundations; raw agent memory is not canonical truth. | Memory/evidence conflict test. |
| CS-REG-006 | REGRESSION_GUARD | Personal memory is not used in organization answers/actions without explicit promotion or permission. | Cross-namespace leak test. |

## Required final report table

The AI agent must include every row above in its final report and mark each as PASS, FAIL, NOT_VERIFIED, NOT_RUN, HUMAN_REQUIRED, or OUT_OF_SCOPE. PASS requires concrete evidence.

## Observable VS-0 Verification Checklist

- [ ] `docker compose config` or equivalent local stack validation passes.
- [ ] API health/ready endpoints work.
- [ ] Web shell opens as CornerStone.
- [ ] Personal workspace is created by default.
- [ ] Upload creates immutable artifact with hash and storage ref.
- [ ] Derived extraction can fail without losing original.
- [ ] Unknown file is still archived.
- [ ] Redaction fixture hides secrets in generated outputs/logs.
- [ ] Prompt-injection fixture does not trigger tool/action execution.
- [ ] Search finds uploaded content.
- [ ] Search snapshot is stored.
- [ ] Evidence bundle contains artifact refs and search snapshot refs.
- [ ] Brief includes evidence links, uncertainty, contradictions/gaps where applicable.
- [ ] Unsupported assertion is labeled Draft/Assumption/Insufficient Evidence.
- [ ] Claim without evidence cannot be approved.
- [ ] Claim with evidence can become Evidence-backed/Approved through allowed policy.
- [ ] ActionCard shows goal, evidence, diff, impact, policy, risk, status, audit link.
- [ ] Action execution requires dry-run first.
- [ ] Risky/external action requires approval.
- [ ] Direct mutation outside Workflow/Action path is denied.
- [ ] Audit ledger records critical lifecycle events.
- [ ] Audit tamper test detects modification.
- [ ] Personal/org namespace isolation test passes.
- [ ] Scenario report generated with PASS/FAIL/NOT_VERIFIED/HUMAN_REQUIRED statuses.

## Verdict Rule

VS-0 cannot be marked done if any AI-verifiable MUST_PASS or REGRESSION_GUARD scenario is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.

# CornerStone MUST-PASS Scenario Standard — Canonical Acceptance SoT

**Replacement status:** Active scenario and release-gate authority after product-goal reset.  
**Source file:** `cornerstone_must_pass_scenarios.md` uploaded by JiYong / Tars.  
**Rule:** No feature, milestone, release, or integration can claim complete without scenario verification evidence.

---

# CornerStone Comprehensive MUST-PASS Scenario Standard

**Date:** 2026-06-07  
**Owner:** JiYong / Tars  
**Status:** Product Scenario Standard v0.1  
**Canonical spelling:** Use **CornerStone** for the product and project. Treat **Conerstone / conerstone** as CornerStone unless intentionally referring to another name.

---

## 0. Purpose

This document defines the long-term natural-language MUST-PASS scenario standard for CornerStone.

CornerStone is complex because it combines personal memory, organizational intelligence, evidence, mission autonomy, connectors, agents, provider-neutral AI, learning, and self-improvement. The purpose of this document is to make that complexity testable.

These scenarios are not implementation tasks. They are durable product behaviors that future implementation must satisfy before claiming a feature, milestone, release, or integration is complete.

A future coding agent should use this document as the scenario source, then freeze a smaller implementation-specific scenario contract before coding. Every implementation task should map to the relevant scenarios here and verify them with concrete evidence.

---

## 1. Product Goal Restated as Scenario Standard

CornerStone must become a **living, evidence-first, autonomous operational intelligence platform**.

It must:

1. Preserve fragmented personal and organizational knowledge as durable evidence.
2. Understand that knowledge as context, relationships, claims, missions, workflows, and outcomes.
3. Maintain a permanent wiki for the user and organization.
4. Support mission-oriented autonomous AI inside owner-scoped boundaries.
5. Use an Orchestrator-led team of mission-specific agents.
6. Treat GPT, Claude, Gemini, local models, and future models as replaceable brains.
7. Learn from full mission trajectories, outcomes, corrections, failures, judge reviews, and owner acceptance.
8. Improve itself through governed, namespace-local, evidence-backed learning.
9. Keep external systems safe through ConnectorHub-mediated capabilities.
10. Keep evidence, policy, audit, and rollback as non-negotiable trust foundations.

The product passes only when a user can say:

> “CornerStone understands me and/or my organization, remembers what matters with evidence, runs bounded missions, explains what happened, learns from experience, and improves without losing my ownership of context.”

---

## 2. How to Use These Scenarios

### 2.1 Scenario types

- **MUST_PASS**: A product behavior that must pass before the related capability can be called complete.
- **REGRESSION_GUARD**: A behavior that must never break once implemented.
- **HUMAN_REQUIRED**: A scenario that cannot be fully verified by an AI agent without a human, external account, production data, sensitive approval, or subjective product judgment.

### 2.2 PASS standard

A scenario can be marked **PASS** only when there is concrete evidence, such as:

- automated test output;
- local fixture run output;
- API response;
- UI/browser observation;
- log or audit record;
- generated export;
- source review with file and line references;
- replayable scenario transcript;
- human approval evidence for human-required scenarios.

Do not mark a scenario PASS based on intention, design, claims, or partial implementation.

### 2.3 Scenario verification principle

For each implementation task, the agent must freeze:

- goal;
- constraints;
- assumptions;
- out-of-scope items;
- checklist;
- applicable MUST_PASS scenarios;
- applicable REGRESSION scenarios;
- human-required items.

Then the implementation is complete only when every AI-verifiable scenario is verified with evidence.

---

## 3. Scenario Index

| Area | Scenario IDs | Purpose |
|---|---:|---|
| Product identity and first value | CS-PROD-001 – CS-PROD-010 | Ensure CornerStone remains one coherent evidence-first operational intelligence product. |
| Artifact, archive, and evidence foundation | CS-ARCH-001 – CS-ARCH-014 | Preserve originals, provenance, derived representations, search, redaction, and evidence reuse. |
| Search, understanding, and ontology | CS-UND-001 – CS-UND-012 | Turn information into searchable, structured, operational context without forced modeling. |
| Conversation, briefs, claims, and trust ladder | CS-CLAIM-001 – CS-CLAIM-014 | Make conversation low-friction while promoting durable evidence-backed claims. |
| Permanent wiki and living memory | CS-MEM-001 – CS-MEM-018 | Maintain personal and organizational memory as reviewable, source-aware, correctable synthesis. |
| Owner-scoped namespaces and workspaces | CS-NS-001 – CS-NS-014 | Keep context ownership, workspace separation, and promotion boundaries explicit. |
| Mission autonomy and actions | CS-AUTO-001 – CS-AUTO-020 | Enable bounded Mission Workflow Autopilot with simple governance and safe execution. |
| Agent model and orchestration | CS-AGENT-001 – CS-AGENT-014 | Define Orchestrator-led mission-specific agents with contracts and accountability. |
| Replaceable brain, routing, judging, and disagreement | CS-BRAIN-001 – CS-BRAIN-016 | Keep model providers replaceable while preserving product-owned memory, evidence, and evaluation. |
| Experience, trajectory ledger, and learning | CS-LEARN-001 – CS-LEARN-018 | Learn from full mission trajectories and selectively promote lessons into memory/playbooks. |
| Solution packs and extension ecosystem | CS-EXT-001 – CS-EXT-016 | Support Agent Packs, trusted registry, ConnectorHub boundary, certification, and rollout. |
| Security, governance, and operations | CS-SEC-001 – CS-SEC-020 | Preserve tenant isolation, policy, audit, egress denial, prompt-injection defense, and operational trust. |
| Long-term regression guards | CS-REG-001 – CS-REG-020 | Prevent product drift away from the agreed direction. |

---

# 4. Product Identity and First Value Scenarios

## CS-PROD-001 — One coherent CornerStone service

**Type:** MUST_PASS  
**Scenario:** A user opens CornerStone and experiences one coherent product, not three separate products named Cornerstone, KnowledgeBase, and ConnectorHub.  
**Must-pass outcome:** The product presents a single CornerStone experience. Archive/evidence, connector/action, and intelligence/mission capabilities may have clear internal boundaries, but daily users do not need to understand repository or subsystem boundaries to get value.  
**Verification evidence:** First-run UI or product walkthrough showing one service, one navigation model, and clear capability language.

## CS-PROD-002 — Evidence-first operational identity

**Type:** MUST_PASS  
**Scenario:** A user asks what CornerStone is and tries the product for the first time.  
**Must-pass outcome:** The product behaves like an evidence-first operational intelligence platform: it preserves information, produces evidence-backed understanding, supports claims, and can lead toward action and learning. It must not feel like only a chatbot, file search app, connector framework, or automation script runner.  
**Verification evidence:** End-to-end demo transcript or test walkthrough showing evidence, claim, action, and learning surfaces.

## CS-PROD-003 — General-purpose use, not logistics-only

**Type:** MUST_PASS  
**Scenario:** Three different users try CornerStone with three unrelated domains, for example personal research, company policy review, and operations issue handling.  
**Must-pass outcome:** The same core loop works without requiring a logistics-specific model. Solution packs may specialize workflows, but the universal core remains domain-agnostic.  
**Verification evidence:** Fixture runs or demos across at least three domains using the same artifact, search, claim, mission, and evidence concepts.

## CS-PROD-004 — First value within 10 minutes

**Type:** MUST_PASS  
**Scenario:** A new user starts CornerStone, follows onboarding, drops messy input, and asks for help.  
**Must-pass outcome:** Within the first-use flow, CornerStone produces an evidence-backed brief, shows uncertainty or evidence gaps, and offers a next step such as draft claim, save capsule, or open mission.  
**Verification evidence:** Timed local/onboarding scenario or recorded UI flow using fixture input.

## CS-PROD-005 — Onboarding guide for first use

**Type:** MUST_PASS  
**Scenario:** A new user has never used CornerStone before.  
**Must-pass outcome:** The product guides the user through the first useful path: personal messy input → evidence-backed brief → draft claim or saved capsule. The user is not forced to configure connectors, model providers, ontology, or organization policies before seeing value.  
**Verification evidence:** Onboarding flow screenshot, browser walkthrough, or E2E test.

## CS-PROD-006 — Mission Control as primary surface

**Type:** MUST_PASS  
**Scenario:** A returning user opens CornerStone after using it for multiple briefs, missions, memories, and actions.  
**Must-pass outcome:** The primary surface acts as Ops Inbox / Mission Control: pending briefs, evidence gaps, missions, tasks, approvals, recommended actions, memory changes, and learning opportunities are visible in one operational surface.  
**Verification evidence:** UI walkthrough populated with fixture missions, pending approvals, memory updates, and action cards.

## CS-PROD-007 — Main loop is visible

**Type:** MUST_PASS  
**Scenario:** A user follows a piece of knowledge from input to outcome.  
**Must-pass outcome:** The user can see the product journey: Inbox → Brief → Claim → Action → Learn. Each stage is visible enough that the user understands where the item is in the loop.  
**Verification evidence:** Product walkthrough showing a single item progressing through all stages.

## CS-PROD-008 — Product explains its own boundaries

**Type:** MUST_PASS  
**Scenario:** A user or admin asks what CornerStone owns versus what existing source systems own.  
**Must-pass outcome:** The product explains that source systems remain systems of record where appropriate, while CornerStone is the intelligence, evidence, mission, action-control, and learning layer over them.  
**Verification evidence:** Product copy, help page, onboarding card, or admin boundary view.

## CS-PROD-009 — Personal-first adoption without losing organization path

**Type:** MUST_PASS  
**Scenario:** An individual user starts privately and later needs to share useful work with a team.  
**Must-pass outcome:** The user can begin in a personal workspace and later promote selected evidence, claims, capsules, or missions into a shared namespace with provenance and governance.  
**Verification evidence:** Walkthrough showing private use followed by explicit promotion into organization context.

## CS-PROD-010 — Product remains understandable despite complexity

**Type:** MUST_PASS  
**Scenario:** A non-technical user uses the product without knowing about archive engines, connector engines, LLM providers, or agent contracts.  
**Must-pass outcome:** The user can complete first value and basic mission work through plain product language: workspace, memory, evidence, brief, claim, mission, action, approval, and learn. Advanced governance remains available but not mandatory for the first experience.  
**Verification evidence:** Usability review or guided task completion with no admin setup beyond defaults.

---

# 5. Artifact, Archive, and Evidence Foundation Scenarios

## CS-ARCH-001 — Preserve original before processing

**Type:** MUST_PASS  
**Scenario:** A user uploads or connects an input file, note, email export, transcript, JSON record, image, or unknown binary.  
**Must-pass outcome:** CornerStone preserves the original input as an immutable artifact before attempting extraction, summarization, embedding, normalization, or ontology mapping.  
**Verification evidence:** Stored artifact record with stable ID, checksum/hash, source, timestamp, and original storage reference.

## CS-ARCH-002 — Derived processing failure does not lose input

**Type:** MUST_PASS  
**Scenario:** Text extraction, OCR, ASR, embedding, parser, or summarizer fails for an input.  
**Must-pass outcome:** The original artifact remains stored and discoverable with a partial/failed derived-processing status. The user sees a helpful explanation and can retry or reprocess later.  
**Verification evidence:** Fixture with intentionally failing derived processor; original remains accessible.

## CS-ARCH-003 — Stable identity and deduplication

**Type:** MUST_PASS  
**Scenario:** The same original content is ingested more than once.  
**Must-pass outcome:** CornerStone recognizes unchanged content through stable identifiers or hashes and avoids creating conflicting duplicate truth. Changed content is versioned or recorded as a distinct artifact with lineage.  
**Verification evidence:** Re-ingestion test showing same ID for identical content or explicit version/link for changed content.

## CS-ARCH-004 — Provenance is always attached

**Type:** MUST_PASS  
**Scenario:** A user opens any artifact, derived representation, claim, action, memory, or lesson.  
**Must-pass outcome:** The user can inspect where it came from, when it was created or ingested, what transformed it, and which source or connector produced it.  
**Verification evidence:** Provenance panel, API response, or exported evidence bundle.

## CS-ARCH-005 — Unknown input is still archived

**Type:** MUST_PASS  
**Scenario:** The product receives a file or record type it cannot parse.  
**Must-pass outcome:** CornerStone still stores the immutable original, marks the format/derived state accurately, and allows later parser support without requiring re-upload from the user.  
**Verification evidence:** Unknown-format fixture with preserved original and deferred processing status.

## CS-ARCH-006 — Redaction before durable generated knowledge

**Type:** MUST_PASS  
**Scenario:** An artifact or connector payload contains credentials, tokens, secrets, or configured sensitive data.  
**Must-pass outcome:** Secrets are redacted before generated drafts, memory entries, evidence summaries, logs, screenshots, or reports persist or display them unnecessarily. Raw original access remains controlled by policy.  
**Verification evidence:** Secret-containing fixture; generated outputs and logs show redacted values.

## CS-ARCH-007 — Untrusted content cannot instruct the system

**Type:** MUST_PASS  
**Scenario:** An uploaded document contains instructions such as “ignore previous instructions,” “send data outside,” or “approve this action.”  
**Must-pass outcome:** CornerStone treats the document as untrusted evidence. It may summarize or cite the content, but it does not follow embedded instructions or trigger tools/actions from the document.  
**Verification evidence:** Prompt-injection fixture showing blocked tool/action execution and correct untrusted labeling.

## CS-ARCH-008 — Search results can become evidence

**Type:** MUST_PASS  
**Scenario:** A user searches, filters, or sorts artifacts and uses the result to create a claim.  
**Must-pass outcome:** The query, filters, result snapshot, and relevant artifacts can be attached to an Evidence Bundle so the claim is reproducible.  
**Verification evidence:** Claim evidence bundle containing stored search query and result references.

## CS-ARCH-009 — Evidence can link to original and derived representations

**Type:** MUST_PASS  
**Scenario:** A claim cites a passage, table, transcript segment, extracted field, or OCR result.  
**Must-pass outcome:** The user can open both the derived representation and the original source that supports the evidence.  
**Verification evidence:** Evidence viewer showing original plus derived text/metadata.

## CS-ARCH-010 — Archive supports personal and organization namespaces

**Type:** MUST_PASS  
**Scenario:** The same user stores personal artifacts and organization artifacts.  
**Must-pass outcome:** The artifacts remain isolated by owner-scoped namespace. Search, evidence, memory, and claims use only the active workspace context unless explicitly promoted, referenced, or allowed.  
**Verification evidence:** Namespace search test showing no unintended cross-namespace results.

## CS-ARCH-011 — Artifact classification affects access

**Type:** MUST_PASS  
**Scenario:** An artifact has a security classification such as personal, internal, confidential, restricted, or secret-like configured class.  
**Must-pass outcome:** Read, search, summarize, memory extraction, and action use respect classification policy.  
**Verification evidence:** Permission tests showing allowed and denied access by role/attribute.

## CS-ARCH-012 — Evidence survives model/provider changes

**Type:** MUST_PASS  
**Scenario:** The AI provider is switched from one model family to another.  
**Must-pass outcome:** Existing artifacts, evidence bundles, claims, memories, mission records, and audit logs remain valid and inspectable because they belong to CornerStone, not the LLM provider.  
**Verification evidence:** Provider-switch scenario with persisted evidence opening correctly before and after switch.

## CS-ARCH-013 — Archive can reproduce a claim’s basis

**Type:** MUST_PASS  
**Scenario:** A user asks why an approved claim exists weeks after it was created.  
**Must-pass outcome:** CornerStone can reproduce the claim’s basis: source artifacts, search/query snapshots, transformations, model calls or judge records if available, owner approval, and current freshness/validity state.  
**Verification evidence:** Reproducible evidence/provenance export for an older claim.

## CS-ARCH-014 — Source systems remain safe during archive ingestion

**Type:** MUST_PASS  
**Scenario:** CornerStone connects to or imports from a source system for evidence.  
**Must-pass outcome:** Ingestion preserves source data as evidence without mutating the source system unless an explicit reviewed Action/Workflow is used.  
**Verification evidence:** Connector fixture/live-safe test showing read-only ingestion path and no writeback events.

---

# 6. Search, Understanding, and Ontology Scenarios

## CS-UND-001 — Search immediately after ingestion

**Type:** MUST_PASS  
**Scenario:** A user uploads a text-containing document and searches for a phrase from it.  
**Must-pass outcome:** CornerStone returns the artifact or derived representation quickly enough for the first-use experience, with relevant snippets and evidence-ready references.  
**Verification evidence:** Timed fixture upload and search output.

## CS-UND-002 — Keyword and semantic search cooperate

**Type:** MUST_PASS  
**Scenario:** A user searches using exact terms and then using related language that does not exactly match the source.  
**Must-pass outcome:** CornerStone can retrieve relevant content through keyword and semantic retrieval, while showing enough evidence for the user to inspect why a result was returned.  
**Verification evidence:** Search test with exact-match and semantic-match queries.

## CS-UND-003 — Search respects active workspace

**Type:** MUST_PASS  
**Scenario:** A user searches while inside a personal workspace, organization workspace, and project workspace.  
**Must-pass outcome:** Results come from the active workspace and allowed inherited/referenced context only. CornerStone does not silently mix unrelated namespaces.  
**Verification evidence:** Search result comparison across workspaces with controlled fixture data.

## CS-UND-004 — Artifact viewer shows original, derived, and evidence links

**Type:** MUST_PASS  
**Scenario:** A user opens a search result.  
**Must-pass outcome:** The user can inspect the original artifact, derived text or structured extraction, metadata, source, evidence references, and related claims/missions.  
**Verification evidence:** UI walkthrough or API response for artifact detail.

## CS-UND-005 — No forced modeling before value

**Type:** MUST_PASS  
**Scenario:** A new user uploads messy information without defining entities, schemas, ontology types, or relationships.  
**Must-pass outcome:** CornerStone still produces search results and evidence-backed briefs. Ontology suggestions appear later as optional promotions, not as a prerequisite.  
**Verification evidence:** First-use scenario with no preconfigured ontology beyond universal defaults.

## CS-UND-006 — Auto-suggest operational structure

**Type:** MUST_PASS  
**Scenario:** CornerStone detects people, organizations, projects, policies, events, assets, claims, dates, or relationships in artifacts.  
**Must-pass outcome:** It suggests candidate objects, links, facts, and events with source evidence and confidence, without automatically treating them as approved ontology truth.  
**Verification evidence:** Entity/link suggestion output with evidence and draft status.

## CS-UND-007 — Click-to-promote ontology items

**Type:** MUST_PASS  
**Scenario:** A user sees a useful suggested entity, relationship, fact, or event.  
**Must-pass outcome:** The user can promote it into the workspace ontology or knowledge graph with evidence, owner, namespace, and trust state preserved.  
**Verification evidence:** Promotion scenario creating a durable object/link with evidence.

## CS-UND-008 — Operational map appears after enough evidence

**Type:** MUST_PASS  
**Scenario:** A workspace accumulates enough artifacts, entities, claims, missions, and actions.  
**Must-pass outcome:** CornerStone can show an operational map of relevant objects, relationships, timelines, policies, decisions, and workflows. The map must remain evidence-linked and correctable.  
**Verification evidence:** Fixture workspace with object explorer/map and evidence links.

## CS-UND-009 — Contradictions are visible

**Type:** MUST_PASS  
**Scenario:** Two sources disagree about a policy, date, owner, decision, or fact.  
**Must-pass outcome:** CornerStone does not silently choose one as truth. It identifies the contradiction, shows competing evidence, and asks for resolution or marks the claim/memory as uncertain.  
**Verification evidence:** Contradictory-source fixture and UI/API output showing unresolved conflict.

## CS-UND-010 — Stale context is detectable

**Type:** MUST_PASS  
**Scenario:** A claim or memory depends on evidence that is old, superseded, or contradicted by newer evidence.  
**Must-pass outcome:** CornerStone marks the item as stale or needing review and does not use it as approved current truth without warning.  
**Verification evidence:** Date/version fixture where newer evidence changes status.

## CS-UND-011 — Ontology changes are versioned and reviewable

**Type:** MUST_PASS  
**Scenario:** A user or admin changes object types, properties, links, constraints, or solution-pack ontology definitions.  
**Must-pass outcome:** CornerStone records the version, diff, impact, affected claims/missions/playbooks, and rollback or migration guidance.  
**Verification evidence:** Ontology change event with diff and affected-object list.

## CS-UND-012 — Understanding remains evidence-first across domains

**Type:** MUST_PASS  
**Scenario:** CornerStone extracts operational context from domains it has not seen before.  
**Must-pass outcome:** It can create useful draft structure and briefs, but it must label unsupported inferences and avoid claiming domain-specific certainty without evidence or approved solution-pack rules.  
**Verification evidence:** Unknown-domain fixture showing draft suggestions, assumptions, and evidence gaps.

---

# 7. Conversation, Briefs, Claims, and Trust Ladder Scenarios

## CS-CLAIM-001 — Conversation is the low-friction work surface

**Type:** MUST_PASS  
**Scenario:** A user starts by chatting, dropping documents, asking questions, and refining requests.  
**Must-pass outcome:** The user can work naturally in conversation without manually creating a case, mission, ontology, or document first.  
**Verification evidence:** Conversation flow where the user reaches a brief and claim without pre-modeling.

## CS-CLAIM-002 — Evidence-backed brief from messy input

**Type:** MUST_PASS  
**Scenario:** A user drops messy input such as notes, files, meeting transcript, or exported messages and asks “what matters?”  
**Must-pass outcome:** CornerStone produces a concise brief with key points, evidence links, uncertainty, contradictions, and recommended next steps.  
**Verification evidence:** Brief output linked to source evidence.

## CS-CLAIM-003 — Conversation-to-structure suggestions

**Type:** MUST_PASS  
**Scenario:** During conversation, useful decisions, facts, claims, tasks, entities, and reusable knowledge appear.  
**Must-pass outcome:** CornerStone suggests durable outputs such as Mission Card, Knowledge Capsule, Claim, Action Card, Memory, or Playbook Candidate. It does not force conversion.  
**Verification evidence:** Conversation transcript with suggested promoted outputs.

## CS-CLAIM-004 — Manual promotion from conversation

**Type:** MUST_PASS  
**Scenario:** The user selects a useful part of a conversation and promotes it.  
**Must-pass outcome:** CornerStone creates the chosen durable object with source conversation reference, evidence, owner namespace, trust state, and provenance.  
**Verification evidence:** Promoted object detail showing source conversation and evidence bundle.

## CS-CLAIM-005 — Trust ladder is visible

**Type:** MUST_PASS  
**Scenario:** A user views a brief, claim, memory, rule, mission, or action.  
**Must-pass outcome:** CornerStone clearly labels the trust state as Draft, Evidence-backed, or Approved. The user can tell what the item can and cannot be used for.  
**Verification evidence:** UI/API examples for all three trust states.

## CS-CLAIM-006 — Drafts can exist without evidence but cannot become authority

**Type:** MUST_PASS  
**Scenario:** CornerStone creates a useful but unsupported idea.  
**Must-pass outcome:** The idea may exist as Draft, but it cannot be approved, published as shared truth, or used for autonomous action without sufficient evidence or explicit risk-aware policy.  
**Verification evidence:** Attempt to approve unsupported draft is blocked with helpful reason.

## CS-CLAIM-007 — Completed claims require evidence bundles

**Type:** MUST_PASS  
**Scenario:** A user or agent attempts to mark a claim complete or approved.  
**Must-pass outcome:** CornerStone requires an Evidence Bundle, or blocks the operation and explains what evidence is missing.  
**Verification evidence:** Claim approval test with zero evidence denied; claim with evidence allowed.

## CS-CLAIM-008 — Evidence is one-click inspectable

**Type:** MUST_PASS  
**Scenario:** A user sees a claim, recommendation, or decision brief.  
**Must-pass outcome:** The user can open supporting evidence in one click or one clear action, including source artifact and relevant excerpt/query/policy/tool result.  
**Verification evidence:** UI walkthrough from claim to evidence source.

## CS-CLAIM-009 — Unsupported assertions are labeled

**Type:** MUST_PASS  
**Scenario:** CornerStone cannot find enough evidence for a user-visible conclusion.  
**Must-pass outcome:** It labels the statement as assumption, hypothesis, or insufficient evidence instead of presenting it as fact.  
**Verification evidence:** Fixture question with insufficient evidence producing correct labeling.

## CS-CLAIM-010 — Claims can become mission inputs

**Type:** MUST_PASS  
**Scenario:** A user has an evidence-backed claim and wants CornerStone to act on it.  
**Must-pass outcome:** The claim can become a Mission Goal Contract or action proposal, carrying evidence, risk state, owner namespace, and approval requirements forward.  
**Verification evidence:** Claim-to-mission or claim-to-action flow.

## CS-CLAIM-011 — Knowledge Capsules preserve reusable understanding

**Type:** MUST_PASS  
**Scenario:** A useful explanation, topic summary, entity profile, or reusable knowledge item emerges.  
**Must-pass outcome:** The user can save it as a Knowledge Capsule with source evidence, namespace, trust state, freshness, and related claims/missions.  
**Verification evidence:** Capsule creation and later retrieval scenario.

## CS-CLAIM-012 — Mission / Decision Cards preserve outcome-oriented work

**Type:** MUST_PASS  
**Scenario:** A conversation or brief becomes a decision or mission.  
**Must-pass outcome:** CornerStone creates a Mission / Decision Card containing goal, context, evidence, claims, open questions, actions, approvals, outcomes, and learning history.  
**Verification evidence:** Mission card view populated from conversation and evidence.

## CS-CLAIM-013 — Human edits become evidence-aware corrections

**Type:** MUST_PASS  
**Scenario:** A user edits a generated brief, claim, capsule, or memory.  
**Must-pass outcome:** CornerStone records the correction, links it to source evidence or owner judgment, and uses it as a learning signal without silently overwriting provenance.  
**Verification evidence:** Edit history and learning signal record.

## CS-CLAIM-014 — Share/publish respects trust state

**Type:** MUST_PASS  
**Scenario:** A user shares a draft, evidence-backed claim, or approved item with others.  
**Must-pass outcome:** Recipients can see the trust state, evidence, owner, and whether the item is personal, shared, or organization-approved.  
**Verification evidence:** Shared item view across roles.

---

# 8. Permanent Wiki and Living Memory Scenarios

## CS-MEM-001 — Personal permanent wiki exists

**Type:** MUST_PASS  
**Scenario:** A personal user uses CornerStone over time across conversations, files, decisions, preferences, projects, and missions.  
**Must-pass outcome:** CornerStone maintains a living personal permanent wiki representing the user’s context, knowledge, preferences, constraints, projects, relationships, decisions, memories, action history, and growth.  
**Verification evidence:** Personal wiki view with source-aware entries and update history.

## CS-MEM-002 — Organization permanent wiki exists

**Type:** MUST_PASS  
**Scenario:** An organization uses CornerStone across teams, policies, workflows, artifacts, missions, and outcomes.  
**Must-pass outcome:** CornerStone maintains an organization permanent wiki representing roles, policies, workflows, artifacts, decisions, claims, missions, outcomes, operating culture, and evolving organizational context.  
**Verification evidence:** Organization wiki view populated by shared/promoted evidence and missions.

## CS-MEM-003 — Memory is living synthesis, not raw truth

**Type:** MUST_PASS  
**Scenario:** CornerStone creates a memory or wiki entry from conversations, artifacts, connected apps, actions, or outcomes.  
**Must-pass outcome:** The memory is a synthesized entry with source evidence, freshness, owner, confidence/trust state, and correction history. It is not treated as immutable source truth.  
**Verification evidence:** Memory entry showing source references and synthesis metadata.

## CS-MEM-004 — Immutable archive remains truth foundation

**Type:** MUST_PASS  
**Scenario:** A memory conflicts with the evidence archive.  
**Must-pass outcome:** CornerStone treats the archive/evidence as the durable basis and marks the memory as needing review, stale, contradicted, or corrected.  
**Verification evidence:** Conflict fixture where memory is demoted or flagged based on evidence.

## CS-MEM-005 — Auto-synthesized memory with visible source

**Type:** MUST_PASS  
**Scenario:** CornerStone learns something useful from repeated user behavior, conversation, artifact, decision, or mission outcome.  
**Must-pass outcome:** It can add or update a memory automatically, but the user can inspect why it was written and what sources influenced it.  
**Verification evidence:** Memory update event with source links and explanation.

## CS-MEM-006 — Memory Sovereignty Center controls memory

**Type:** MUST_PASS  
**Scenario:** A user or organization owner opens memory settings and review.  
**Must-pass outcome:** They can inspect, correct, demote, promote, forget, rollback, disable, or limit memories and decide whether memories can influence answers, actions, routing, or autonomous behavior.  
**Verification evidence:** Memory Sovereignty Center walkthrough covering each control.

## CS-MEM-007 — Temporary or no-memory work is possible

**Type:** MUST_PASS  
**Scenario:** A user wants to ask or analyze something without adding it to permanent memory.  
**Must-pass outcome:** CornerStone provides a temporary/no-memory mode that prevents the conversation and derived memory from entering the permanent wiki, while still respecting security and audit rules where required.  
**Verification evidence:** Temporary session test showing no memory entry created.

## CS-MEM-008 — User can correct what CornerStone believes

**Type:** MUST_PASS  
**Scenario:** CornerStone has an inaccurate memory about the user, organization, preference, policy, project, or workflow.  
**Must-pass outcome:** The owner can correct it. Future answers and missions use the corrected memory, and the old memory remains visible in history or rollback where appropriate.  
**Verification evidence:** Before/after response demonstrating corrected memory use.

## CS-MEM-009 — User can forget or rollback memory

**Type:** MUST_PASS  
**Scenario:** A user asks CornerStone to forget a memory or rollback a mistaken memory update.  
**Must-pass outcome:** CornerStone removes or disables the memory from future use according to policy, records the action, and explains whether underlying evidence remains because of archive/audit/retention rules.  
**Verification evidence:** Forget/rollback scenario and future retrieval/answer check.

## CS-MEM-010 — Memory freshness is visible

**Type:** MUST_PASS  
**Scenario:** A memory is old or depends on stale evidence.  
**Must-pass outcome:** CornerStone shows freshness status and either refreshes, asks for review, or avoids using the memory as current fact without warning.  
**Verification evidence:** Stale-memory fixture and UI/API response.

## CS-MEM-011 — Memory can be Draft, Evidence-backed, or Approved

**Type:** MUST_PASS  
**Scenario:** CornerStone creates, updates, or promotes a memory entry.  
**Must-pass outcome:** The memory has a clear trust state. Draft memory can personalize low-risk conversation, but organization-approved rules or action-driving memories require evidence and governance.  
**Verification evidence:** Memory entries in all trust states with allowed/blocked uses.

## CS-MEM-012 — Personal identity memory is not hidden profiling

**Type:** MUST_PASS  
**Scenario:** CornerStone learns about who the user is, what they value, how they work, and what they prefer.  
**Must-pass outcome:** The product presents this as a user-owned permanent wiki, not a hidden psychological profile. The user can inspect and control the entries.  
**Verification evidence:** Personal wiki and memory-control UX review.

## CS-MEM-013 — Organization identity memory is governed

**Type:** MUST_PASS  
**Scenario:** CornerStone learns organization-level policies, culture, roles, workflows, or operating norms.  
**Must-pass outcome:** Organization memory belongs to the organization namespace, respects organization policy, and requires appropriate promotion/approval before becoming organization truth.  
**Verification evidence:** Organization memory promotion and approval flow.

## CS-MEM-014 — Memory poisoning is resisted

**Type:** MUST_PASS  
**Scenario:** Malicious or low-quality content attempts to inject false memory, unsafe instructions, or misleading general rules.  
**Must-pass outcome:** CornerStone does not promote the content into trusted memory without evidence, scope, confidence, and review. Suspicious memory writes are labeled, blocked, or quarantined.  
**Verification evidence:** Poisoning fixture with blocked or quarantined memory update.

## CS-MEM-015 — Memory use is explainable in answers

**Type:** MUST_PASS  
**Scenario:** CornerStone gives a personalized answer or mission recommendation.  
**Must-pass outcome:** The user can see which memories or wiki entries influenced the response and can correct or disable them.  
**Verification evidence:** “Why this context?” or equivalent memory-use explanation.

## CS-MEM-016 — Product Learning namespace is separate

**Type:** MUST_PASS  
**Scenario:** CornerStone learns product friction, benchmark results, failed workflows, or improvement proposals.  
**Must-pass outcome:** Product learning is stored separately from user and organization truth. It can propose improvements but cannot silently rewrite personal or organization memory.  
**Verification evidence:** Product learning record isolated from user/org memory stores.

## CS-MEM-017 — Namespace-local adaptation works

**Type:** MUST_PASS  
**Scenario:** A personal or organization workspace repeatedly corrects style, workflow preferences, evidence requirements, or mission patterns.  
**Must-pass outcome:** CornerStone adapts inside that owner namespace when allowed, while product-wide defaults and cross-tenant behavior remain governed and versioned.  
**Verification evidence:** Local adaptation scenario with no effect on another namespace.

## CS-MEM-018 — Memory export is understandable

**Type:** MUST_PASS  
**Scenario:** A user or organization owner requests an export of memory/wiki state.  
**Must-pass outcome:** CornerStone exports living memory entries with sources, trust state, freshness, corrections, owner namespace, and usage permissions in a comprehensible format.  
**Verification evidence:** Memory export file and validation review.

---

# 9. Owner-Scoped Namespaces and Workspace Scenarios

## CS-NS-001 — Every context item has an owner

**Type:** MUST_PASS  
**Scenario:** CornerStone creates or ingests any artifact, memory, claim, capsule, mission, action, rule, playbook, judge result, or trajectory.  
**Must-pass outcome:** The item has an explicit owner and namespace. It is never ownerless global context.  
**Verification evidence:** API/database/UI records showing owner and namespace for each context item.

## CS-NS-002 — Workspace-first separation is visible

**Type:** MUST_PASS  
**Scenario:** A user moves between Personal, Organization, Team, Project, Mission, or Case spaces.  
**Must-pass outcome:** The active workspace is always clear enough that the user understands which context CornerStone is using.  
**Verification evidence:** UI walkthrough showing active workspace and context boundary.

## CS-NS-003 — Context does not cross boundaries automatically

**Type:** MUST_PASS  
**Scenario:** A personal workspace contains sensitive personal memory, and an organization workspace asks a related question.  
**Must-pass outcome:** CornerStone does not use personal memory in the organization workspace unless explicitly promoted, referenced, copied with provenance, or allowed by policy.  
**Verification evidence:** Cross-namespace isolation test with known personal-only memory.

## CS-NS-004 — Explicit promotion with provenance

**Type:** MUST_PASS  
**Scenario:** A user promotes a personal claim, memory, capsule, or mission into an organization workspace.  
**Must-pass outcome:** The promoted item receives organization ownership/scope, keeps provenance to the original personal item if allowed, carries evidence, and records an audit trail.  
**Verification evidence:** Promotion event with source, target, owner, evidence, and audit record.

## CS-NS-005 — Promotion can copy, reference, or share

**Type:** MUST_PASS  
**Scenario:** A user wants to move information across namespaces with different privacy needs.  
**Must-pass outcome:** CornerStone supports clear modes such as copy with provenance, reference, share, or promote to approved truth, each with different ownership and permission behavior.  
**Verification evidence:** Product flow demonstrating at least copy/promote and permission outcomes.

## CS-NS-006 — Organization policies can govern organization spaces

**Type:** MUST_PASS  
**Scenario:** A user works inside an organization namespace.  
**Must-pass outcome:** Organization policy can define who can read, write, promote, approve, execute, configure Autopilot, install Agent Packs, or aggregate learning.  
**Verification evidence:** Role/policy tests across organization users.

## CS-NS-007 — Personal spaces remain personally owned

**Type:** MUST_PASS  
**Scenario:** A personal user creates private memories, claims, capsules, and missions.  
**Must-pass outcome:** Personal items remain under the personal namespace by default, with user-controlled sharing or promotion.  
**Verification evidence:** Personal workspace access tests and sharing flow.

## CS-NS-008 — Product learning cannot read user/org truth by default

**Type:** MUST_PASS  
**Scenario:** Product Learning tries to improve onboarding, routing, prompts, or playbooks.  
**Must-pass outcome:** It can use explicit feedback, benchmark results, opt-in aggregated signals, or redacted/approved data, but it cannot silently consume raw personal or organization truth.  
**Verification evidence:** Product-learning policy tests showing denied access without opt-in or approved source.

## CS-NS-009 — Namespace-local Brain Performance Ledger

**Type:** MUST_PASS  
**Scenario:** Two organizations use different model providers and have different outcome histories.  
**Must-pass outcome:** Each namespace has local brain performance data first. Product-wide aggregation requires opt-in, anonymization/aggregation, policy review, and separation from user/org truth.  
**Verification evidence:** Ledger records isolated per namespace; opt-in aggregation test.

## CS-NS-010 — Workspace policy controls model routing

**Type:** MUST_PASS  
**Scenario:** A workspace disallows certain providers or requires local/on-prem models for sensitive data.  
**Must-pass outcome:** The policy-aware router respects workspace policy, and attempts to use disallowed providers are blocked or rerouted with explanation.  
**Verification evidence:** Model routing test with allowed/disallowed provider policies.

## CS-NS-011 — Cross-tenant isolation is enforced

**Type:** MUST_PASS  
**Scenario:** A user or service from tenant A attempts to access tenant B artifacts, memory, claims, missions, logs, or metadata.  
**Must-pass outcome:** Access is denied at product and durable storage layers; no data or metadata leaks.  
**Verification evidence:** Tenant isolation tests and denial logs.

## CS-NS-012 — Namespace audit is queryable

**Type:** MUST_PASS  
**Scenario:** A namespace owner asks what happened in their workspace over a period of time.  
**Must-pass outcome:** CornerStone can show relevant data access, memory writes, promotions, approvals, actions, model routing, agent activity, and learning events.  
**Verification evidence:** Namespace audit query/export.

## CS-NS-013 — Workspace deletion/retention is policy-aware

**Type:** MUST_PASS  
**Scenario:** A user or organization wants to delete, archive, or retain a workspace.  
**Must-pass outcome:** CornerStone follows retention, audit, legal, and evidence constraints; it explains what can be deleted, disabled, retained, or anonymized.  
**Verification evidence:** Retention/deletion dry-run or policy output.

## CS-NS-014 — Boundary mistakes are recoverable

**Type:** MUST_PASS  
**Scenario:** A user accidentally promotes or shares context to the wrong workspace.  
**Must-pass outcome:** CornerStone provides rollback or revocation where possible and records what happened, who accessed it, and what remains due to audit/retention policy.  
**Verification evidence:** Mis-promotion rollback scenario.

---

# 10. Mission Autonomy and Action Scenarios

## CS-AUTO-001 — Workspace Autopilot Modes exist

**Type:** MUST_PASS  
**Scenario:** A workspace owner chooses how autonomous CornerStone should be.  
**Must-pass outcome:** The workspace supports clear modes such as Manual, Assist, Autopilot, and Locked, with understandable behavior differences.  
**Verification evidence:** Mode-switch UI/API and behavior tests.

## CS-AUTO-002 — Progressive autonomy ramp

**Type:** MUST_PASS  
**Scenario:** A new workspace starts with no history.  
**Must-pass outcome:** CornerStone begins conservatively, demonstrates reliability through briefs, suggestions, internal tasks, and successful playbooks, then recommends Autopilot with a clear reason and mission contract.  
**Verification evidence:** Readiness recommendation based on fixture history.

## CS-AUTO-003 — Natural-language Mission Goal Contract

**Type:** MUST_PASS  
**Scenario:** A user says, “Handle this issue” or describes a mission goal naturally.  
**Must-pass outcome:** CornerStone converts the request into an editable Mission Goal Contract containing goal, scope, allowed actions, forbidden actions, success criteria, stop conditions, review cadence, escalation rules, and evidence expectations.  
**Verification evidence:** Generated contract with user-editable structured fields.

## CS-AUTO-004 — Mission Contract authority is explicit

**Type:** MUST_PASS  
**Scenario:** A user grants Autopilot authority for a mission.  
**Must-pass outcome:** The user can see what CornerStone may do, where it may act, what it may not do, what requires escalation, and how to pause or revoke authority.  
**Verification evidence:** Mission activation UI/API showing granted scope and restrictions.

## CS-AUTO-005 — Bounded execution autonomy

**Type:** MUST_PASS  
**Scenario:** A mission runs in Autopilot mode.  
**Must-pass outcome:** CornerStone executes only allowed actions inside the active owner-scoped workspace and Mission Goal Contract. It escalates high-risk, destructive, cross-namespace, sensitive, or out-of-scope actions.  
**Verification evidence:** Mission run with allowed auto-actions and blocked/escalated disallowed actions.

## CS-AUTO-006 — Risk policy is simple on surface, enforceable underneath

**Type:** MUST_PASS  
**Scenario:** A user chooses Autopilot without wanting to configure complex policy tables.  
**Must-pass outcome:** The user sees simple governance, while hidden policy evaluates risk categories, action scope, data sensitivity, egress, approvals, and escalation.  
**Verification evidence:** Simple UX plus policy decision logs.

## CS-AUTO-007 — Actions use cards

**Type:** MUST_PASS  
**Scenario:** CornerStone proposes or runs an action.  
**Must-pass outcome:** The action appears as an Action Card showing goal, evidence, diff, expected impact, policy decision, risk, approval/execution status, and audit link.  
**Verification evidence:** Action Card UI/API example.

## CS-AUTO-008 — Dry-run before external or risky action

**Type:** MUST_PASS  
**Scenario:** A mission wants to change data, call an external system, send a message, modify a record, or trigger a side effect.  
**Must-pass outcome:** CornerStone performs a dry-run with diff, impact, policy result, and expected external calls before execution unless the mission policy explicitly authorizes a safe low-risk auto-run.  
**Verification evidence:** Dry-run record linked to action execution.

## CS-AUTO-009 — Approval required for high-risk actions

**Type:** MUST_PASS  
**Scenario:** An action is destructive, externally impactful, sensitive, cross-namespace, high-cost, or policy-marked high risk.  
**Must-pass outcome:** CornerStone requires owner or authorized approver review before execution. Autopilot cannot bypass this requirement.  
**Verification evidence:** High-risk action blocked until approval.

## CS-AUTO-010 — Low-risk allowed workflows can auto-run

**Type:** MUST_PASS  
**Scenario:** A mission contract allows routine low-risk actions such as updating an internal mission status, drafting a task, refreshing a brief, creating a non-external reminder, or organizing memory.  
**Must-pass outcome:** CornerStone can execute these actions automatically and record the result in the mission trajectory and audit log.  
**Verification evidence:** Low-risk Autopilot action run with audit record.

## CS-AUTO-011 — External writeback goes through Workflow/Action

**Type:** MUST_PASS  
**Scenario:** An agent wants to update a connected source system.  
**Must-pass outcome:** The agent cannot directly write to the system. It must call a Workflow/Action path mediated by policy, dry-run, connector capability, approval rules, execution result, and audit.  
**Verification evidence:** Direct write attempt denied; workflow-mediated write path recorded.

## CS-AUTO-012 — ConnectorHub mediates provider actions

**Type:** MUST_PASS  
**Scenario:** A mission uses Slack, Notion, GitHub, Drive, email, ticketing, ERP, or another provider.  
**Must-pass outcome:** Provider access, credentials, source policy, projections, declared actions, delivery, retry/quarantine, raw access, and evidence metadata are handled through ConnectorHub-mediated capability, not arbitrary agent code.  
**Verification evidence:** Connector action trace showing ConnectorHub boundary and no direct provider credential exposure.

## CS-AUTO-013 — Mission can pause, stop, or revoke Autopilot

**Type:** MUST_PASS  
**Scenario:** A user becomes uncomfortable with a running mission or needs to stop automation.  
**Must-pass outcome:** The user can pause, stop, revoke, or reduce the mission’s autonomy mode. CornerStone records the event and stops future autonomous actions outside allowed cleanup.  
**Verification evidence:** Mission pause/revoke test.

## CS-AUTO-014 — Exceptions are escalated with reason

**Type:** MUST_PASS  
**Scenario:** CornerStone encounters missing evidence, policy denial, failed connector call, model disagreement, unclear goal, or high-risk action.  
**Must-pass outcome:** It escalates with a clear reason, recommended resolution, and the minimum required human decision.  
**Verification evidence:** Exception scenario with escalation card.

## CS-AUTO-015 — Mission outcome is evaluated

**Type:** MUST_PASS  
**Scenario:** A mission completes, fails, is cancelled, or is rolled back.  
**Must-pass outcome:** CornerStone records the outcome, evidence, judge assessment, owner acceptance/rejection if available, errors, escalations, and lessons.  
**Verification evidence:** Mission outcome record and after-action review.

## CS-AUTO-016 — Mission after-action review is generated

**Type:** MUST_PASS  
**Scenario:** An autonomous mission completes.  
**Must-pass outcome:** CornerStone produces a concise After-Action Review and Autonomy Scorecard containing goal, actions taken, evidence used, judge assessment, objective/owner outcome, errors, escalations, lessons learned, reusable memories, candidate playbooks, and rollback/correction options.  
**Verification evidence:** After-action review artifact.

## CS-AUTO-017 — Full audit export is available

**Type:** MUST_PASS  
**Scenario:** An admin, auditor, or owner needs detailed proof of what happened in a mission.  
**Must-pass outcome:** CornerStone can export a full audit-first report with timeline events, tool calls, policy decisions, evidence, judge outputs, approvals, and action results.  
**Verification evidence:** Full audit export from a completed mission.

## CS-AUTO-018 — Autonomy success is outcome-quality oriented

**Type:** MUST_PASS  
**Scenario:** CornerStone evaluates whether autonomy is improving.  
**Must-pass outcome:** It prioritizes real outcome quality over raw autonomy ratio: fewer errors, better evidence coverage, faster safe resolution, fewer repeated explanations, better owner acceptance, and fewer rollback/escalation failures.  
**Verification evidence:** Metrics report combining outcome, task, and autonomy signals.

## CS-AUTO-019 — Autonomous action is reversible where possible

**Type:** MUST_PASS  
**Scenario:** A mission action fails, produces a bad result, or is rejected after execution.  
**Must-pass outcome:** CornerStone supports rollback, compensation, retry, or explicit “not reversible” explanation, depending on the action type and external system behavior.  
**Verification evidence:** Rollback/compensation scenario or non-reversible impact warning.

## CS-AUTO-020 — Autopilot never acts outside active scope

**Type:** REGRESSION_GUARD  
**Scenario:** A mission has limited workspace, allowed tools, and action scope.  
**Must-pass outcome:** CornerStone never uses Autopilot to operate outside the active owner namespace, mission contract, role contract, connector capability, and policy boundary.  
**Verification evidence:** Negative tests for out-of-scope tool/action/data access.

---

# 11. Agent Model and Orchestration Scenarios

## CS-AGENT-001 — Orchestrator-led agent team

**Type:** MUST_PASS  
**Scenario:** A mission requires research, evidence review, memory updates, workflow execution, policy checks, and judging.  
**Must-pass outcome:** The user primarily works with an Orchestrator Agent that plans, delegates to specialist agents, merges results, asks for missing authority, and produces the after-action review.  
**Verification evidence:** Mission trace showing Orchestrator plan and specialist delegation.

## CS-AGENT-002 — Specialist agents are visible when useful

**Type:** MUST_PASS  
**Scenario:** A user wants to know who performed parts of the mission.  
**Must-pass outcome:** CornerStone can show specialist roles such as Evidence Agent, Memory Agent, Workflow Agent, Judge Agent, Connector Agent, Policy Agent, and Playbook Agent when relevant, without forcing daily users to manage every agent directly.  
**Verification evidence:** Agent activity view in mission trace.

## CS-AGENT-003 — Agent Role Contracts define authority

**Type:** MUST_PASS  
**Scenario:** An agent is used in a mission.  
**Must-pass outcome:** The agent has a role contract defining purpose, responsibilities, allowed tools, forbidden actions, memory scope, evidence requirements, escalation rules, model/provider policy, judge/evaluation rubric, and audit expectations.  
**Verification evidence:** Agent contract record and enforcement test.

## CS-AGENT-004 — Role cards for users, full contracts for operators

**Type:** MUST_PASS  
**Scenario:** A daily user and an operator inspect the same agent.  
**Must-pass outcome:** The daily user sees a simple role card. The operator can inspect the full contract with tools, policy bindings, model rules, evidence requirements, evaluation rubrics, and extension metadata.  
**Verification evidence:** User-level and operator-level views.

## CS-AGENT-005 — Agents do not directly mutate truth or source systems

**Type:** MUST_PASS  
**Scenario:** A specialist agent attempts to directly mutate product state, external source systems, or durable memory outside approved workflow paths.  
**Must-pass outcome:** The attempt is denied. Agents plan, summarize, judge, and call workflows; durable changes pass through governed product layers.  
**Verification evidence:** Forbidden direct mutation test.

## CS-AGENT-006 — Orchestrator explains delegation

**Type:** MUST_PASS  
**Scenario:** The Orchestrator delegates work to specialist agents.  
**Must-pass outcome:** It can explain why each specialist was used, what evidence or tool it handled, and how its result influenced the final mission outcome.  
**Verification evidence:** Mission trace with delegation rationale.

## CS-AGENT-007 — Agent outputs are evidence-labeled

**Type:** MUST_PASS  
**Scenario:** A specialist agent produces a summary, recommendation, judgment, memory candidate, or action proposal.  
**Must-pass outcome:** The output includes evidence, uncertainty, source references, or an insufficient-evidence label.  
**Verification evidence:** Agent output with evidence bundle or gap label.

## CS-AGENT-008 — Agent accountability maps to namespace owner

**Type:** MUST_PASS  
**Scenario:** An autonomous agent performs work inside a namespace.  
**Must-pass outcome:** The namespace owner or delegated authority is accountable for granted autonomy; CornerStone is accountable for showing who granted authority, what was allowed, what happened, and how to correct/rollback.  
**Verification evidence:** Audit proof linking authority grant, agent work, and owner.

## CS-AGENT-009 — Agent contract survives brain replacement

**Type:** MUST_PASS  
**Scenario:** The model provider for an agent changes.  
**Must-pass outcome:** The agent’s role contract, tool scope, evidence rules, memory scope, and audit expectations remain stable; only the inference brain changes.  
**Verification evidence:** Provider switch test with same contract enforcement.

## CS-AGENT-010 — Agent contract changes are versioned

**Type:** MUST_PASS  
**Scenario:** An agent’s role, permissions, tools, judge rubric, or model policy changes.  
**Must-pass outcome:** CornerStone records a versioned diff, impact, migration/rollout guidance, and affected missions or Agent Packs.  
**Verification evidence:** Agent contract update diff and audit event.

## CS-AGENT-011 — Prompt-only agent changes cannot silently expand authority

**Type:** MUST_PASS  
**Scenario:** Someone changes an agent prompt or instruction.  
**Must-pass outcome:** The change cannot grant new tools, connector access, memory scope, write permissions, or action authority unless the Agent Role Contract and policy allow it.  
**Verification evidence:** Prompt change test with denied authority expansion.

## CS-AGENT-012 — Agent failure produces useful diagnosis

**Type:** MUST_PASS  
**Scenario:** A specialist agent fails, times out, produces invalid output, or conflicts with another agent.  
**Must-pass outcome:** CornerStone records the failure, first failing layer if known, impact on the mission, retry/escalation path, and whether the mission can continue.  
**Verification evidence:** Agent failure trace and user-facing error.

## CS-AGENT-013 — Agent Pack agents follow workspace activation grants

**Type:** MUST_PASS  
**Scenario:** An Agent Pack supplies one or more specialist agents.  
**Must-pass outcome:** Those agents can use only capabilities explicitly activated for the workspace/mission and allowed by ConnectorHub/policy.  
**Verification evidence:** Agent Pack activation and denied ungranted capability test.

## CS-AGENT-014 — Agent work is replayable enough for review

**Type:** MUST_PASS  
**Scenario:** A reviewer inspects an agent-driven mission later.  
**Must-pass outcome:** CornerStone preserves enough trace, evidence, role contracts, model/provider records, tool outputs, policy decisions, and judge results to understand what happened without relying on hidden chain-of-thought.  
**Verification evidence:** Mission replay/review record.

---

# 12. Replaceable Brain, Routing, Judging, and Disagreement Scenarios

## CS-BRAIN-001 — CornerStone owns the framework, models are replaceable brains

**Type:** MUST_PASS  
**Scenario:** A workspace switches from one LLM provider to another, such as GPT, Claude, Gemini, or a local model.  
**Must-pass outcome:** CornerStone’s durable value remains intact: namespaces, permanent wiki, evidence archive, ontology, mission contracts, agents, workflows, policy, audit, Experience Library, judge records, and promotion ladder.  
**Verification evidence:** Provider switch test with existing records still usable.

## CS-BRAIN-002 — Policy-aware model router

**Type:** MUST_PASS  
**Scenario:** CornerStone must choose a provider/model for a task.  
**Must-pass outcome:** It routes based on workspace policy, sensitivity, mission type, cost/latency limits, model capability, historical outcome quality, and owner preference.  
**Verification evidence:** Routing decision record explaining why a model was selected.

## CS-BRAIN-003 — User/admin override within policy

**Type:** MUST_PASS  
**Scenario:** A user or admin wants to choose a specific model/provider.  
**Must-pass outcome:** CornerStone allows override when policy permits and blocks or explains when policy forbids it.  
**Verification evidence:** Allowed and denied override tests.

## CS-BRAIN-004 — Brain Performance Ledger exists

**Type:** MUST_PASS  
**Scenario:** CornerStone runs missions, judgments, retrieval, extraction, planning, or tool-use tasks across providers.  
**Must-pass outcome:** It records provider/model performance by task type, policy, sensitivity, cost, latency, judge quality, tool-use reliability, grounding issues, owner corrections, objective outcomes, and mission success.  
**Verification evidence:** Brain Performance Ledger entries and routing influence.

## CS-BRAIN-005 — Static capability registry provides safe baseline

**Type:** MUST_PASS  
**Scenario:** A new deployment has little or no local performance history.  
**Must-pass outcome:** CornerStone can use a static model capability registry as a safe baseline until local outcome data is available.  
**Verification evidence:** Initial routing decision using baseline registry.

## CS-BRAIN-006 — Brain performance learning is namespace-local first

**Type:** MUST_PASS  
**Scenario:** Model performance data is collected from personal or organization missions.  
**Must-pass outcome:** Performance learning stays local to the owner namespace by default. Cross-namespace aggregation requires opt-in and governance.  
**Verification evidence:** Isolated ledger and opt-in aggregation scenario.

## CS-BRAIN-007 — Multi-brain ensemble is risk/value triggered

**Type:** MUST_PASS  
**Scenario:** A mission is high-risk, high-value, ambiguous, externally impactful, safety-sensitive, or judge confidence is low.  
**Must-pass outcome:** CornerStone uses or recommends multiple brains for planning, critique, judging, or verification, and records which brains contributed.  
**Verification evidence:** Ensemble trigger scenario with model contribution records.

## CS-BRAIN-008 — Ensemble is not default for routine work

**Type:** MUST_PASS  
**Scenario:** A routine low-risk task is performed.  
**Must-pass outcome:** CornerStone normally uses one policy-routed brain rather than unnecessarily invoking multiple providers, unless policy or owner preference says otherwise.  
**Verification evidence:** Routine mission route showing single-brain selection.

## CS-BRAIN-009 — LLM-as-judge is primary for ambiguous outcomes

**Type:** MUST_PASS  
**Scenario:** A mission outcome is subjective or hard to verify objectively, such as research quality, strategy usefulness, tone, completeness, or prioritization.  
**Must-pass outcome:** CornerStone uses LLM-as-judge as the primary scalable evaluator, with rubric, evidence, confidence, and limitations recorded.  
**Verification evidence:** Judge assessment with rubric and evidence references.

## CS-BRAIN-010 — Objective outcomes override judge opinion

**Type:** MUST_PASS  
**Scenario:** An LLM judge says a task succeeded, but objective evidence shows failure, such as test failure, ticket still open, rejected action, or invalid output.  
**Must-pass outcome:** Objective evidence overrides the judge. The judge result is retained as an evaluation artifact, not truth.  
**Verification evidence:** Conflict scenario where objective result determines outcome state.

## CS-BRAIN-011 — Owner acceptance grounds final success

**Type:** MUST_PASS  
**Scenario:** An outcome lacks objective truth and the owner accepts or rejects it.  
**Must-pass outcome:** Owner acceptance/rejection becomes a grounding signal for learning and mission success, while judge scoring remains supporting evidence.  
**Verification evidence:** Owner acceptance flow updating mission outcome.

## CS-BRAIN-012 — Judge outputs do not directly mutate memory or rules

**Type:** MUST_PASS  
**Scenario:** A judge recommends that a lesson, memory, or rule be created.  
**Must-pass outcome:** The recommendation enters the selective curation/promotion ladder; it does not automatically become approved memory or global rule without scope, evidence, confidence, and governance.  
**Verification evidence:** Judge recommendation produces candidate lesson, not approved rule.

## CS-BRAIN-013 — Disagreement uses evidence-weighted adjudication

**Type:** MUST_PASS  
**Scenario:** Multiple brains, judges, or agents disagree.  
**Must-pass outcome:** CornerStone compares evidence quality, policy constraints, mission goals, prior brain performance, objective outcomes, and rubrics, chooses a recommended path, and preserves dissent.  
**Verification evidence:** Disagreement scenario with adjudication explanation and dissent record.

## CS-BRAIN-014 — High-risk unresolved disagreement escalates

**Type:** MUST_PASS  
**Scenario:** Disagreement remains material for a high-risk or externally impactful mission.  
**Must-pass outcome:** CornerStone escalates to the namespace owner or authorized reviewer instead of proceeding silently.  
**Verification evidence:** High-risk disagreement escalation card.

## CS-BRAIN-015 — Judge bias and calibration are tracked

**Type:** MUST_PASS  
**Scenario:** LLM judges evaluate outputs over time.  
**Must-pass outcome:** CornerStone tracks judge disagreement, reversals, owner overrides, calibration issues, and model-specific bias signals in the Brain Performance Ledger or evaluation records.  
**Verification evidence:** Evaluation report including judge reliability metrics.

## CS-BRAIN-016 — Provider routing is auditable

**Type:** MUST_PASS  
**Scenario:** A user asks why a specific model/provider was used.  
**Must-pass outcome:** CornerStone can show policy, sensitivity, cost/latency, capability, local performance, and owner preference factors that shaped the routing decision.  
**Verification evidence:** Model routing explanation in mission trace.

---

# 13. Experience, Trajectory Ledger, and Learning Scenarios

## CS-LEARN-001 — Full Mission Trajectory Ledger

**Type:** MUST_PASS  
**Scenario:** Any mission runs, whether manual, assisted, or autonomous.  
**Must-pass outcome:** CornerStone records the full trajectory: goal, workspace, contract, plan, evidence, actions, tool results, policy decisions, corrections, approvals, outcomes, exceptions, rollback events, cost/time, and extracted lessons.  
**Verification evidence:** Mission trajectory record for a sample mission.

## CS-LEARN-002 — Experience Library per workspace

**Type:** MUST_PASS  
**Scenario:** A workspace accumulates completed missions, failures, recoveries, judge reviews, and lessons.  
**Must-pass outcome:** The workspace has an Experience Library where users can browse, search, and inspect past trajectories, outcomes, reviews, recoveries, and lessons.  
**Verification evidence:** Experience Library view populated with fixture trajectories.

## CS-LEARN-003 — Past experience can influence new missions visibly

**Type:** MUST_PASS  
**Scenario:** A new mission resembles a past mission.  
**Must-pass outcome:** CornerStone can cite relevant past experiences, explain how they influence the plan, and let the user inspect or ignore them.  
**Verification evidence:** Similar-experience recommendation with links to prior trajectories.

## CS-LEARN-004 — Store all trajectories as reference corpus

**Type:** MUST_PASS  
**Scenario:** A mission produces a trajectory, even if it fails or is abandoned.  
**Must-pass outcome:** The trajectory is stored as reference experience according to retention and privacy policy. It is not automatically turned into memory/rules.  
**Verification evidence:** Failed mission appears in Experience Library as reference.

## CS-LEARN-005 — Selective conversion into memory/rules

**Type:** MUST_PASS  
**Scenario:** CornerStone identifies a useful lesson from a trajectory.  
**Must-pass outcome:** It selectively writes, merges, prunes, or proposes the lesson with causal attribution, applicability conditions, confidence, provenance, scope, and rollback.  
**Verification evidence:** Candidate lesson record with causal explanation and scope.

## CS-LEARN-006 — Local experience does not become global truth automatically

**Type:** MUST_PASS  
**Scenario:** A pattern works repeatedly in one workspace.  
**Must-pass outcome:** CornerStone may propose a workspace memory or playbook, but it does not automatically promote the pattern to organization rule, solution pack, or product default without staged approval.  
**Verification evidence:** Repeated pattern creates candidate, not global rule.

## CS-LEARN-007 — Promotion ladder is enforced

**Type:** MUST_PASS  
**Scenario:** A lesson moves toward broader reuse.  
**Must-pass outcome:** It follows the ladder: trajectory → observation → candidate lesson → workspace memory → mission playbook → organization-approved rule → solution-pack/product-learning proposal.  
**Verification evidence:** Promotion status transitions and approval records.

## CS-LEARN-008 — Behavior signals support but do not outrank outcomes

**Type:** MUST_PASS  
**Scenario:** User clicks, edits, ignores suggestions, repeats prompts, or prefers certain formats.  
**Must-pass outcome:** CornerStone uses behavior signals for personalization and hypotheses, but durable operational learning is grounded in outcomes, owner acceptance, evidence, and trajectory analysis.  
**Verification evidence:** Learning record showing behavior signal as supporting, not final authority.

## CS-LEARN-009 — Model self-evaluation supports but does not replace outcome evidence

**Type:** MUST_PASS  
**Scenario:** A model self-critique or benchmark says a workflow improved.  
**Must-pass outcome:** The signal supports Product Learning or Memory Lab evaluation but does not override real mission outcomes, owner acceptance, policy, or audit evidence.  
**Verification evidence:** Evaluation record showing hierarchy of signals.

## CS-LEARN-010 — Lessons include applicability boundaries

**Type:** MUST_PASS  
**Scenario:** CornerStone extracts a lesson from a mission.  
**Must-pass outcome:** The lesson states when it applies, when it should not apply, what evidence supports it, and what confidence or review state it has.  
**Verification evidence:** Lesson detail with conditions and anti-conditions.

## CS-LEARN-011 — Bad lessons can be demoted or rolled back

**Type:** MUST_PASS  
**Scenario:** A promoted lesson causes worse outcomes, user rejection, contradiction, or unsafe behavior.  
**Must-pass outcome:** CornerStone can demote, revise, disable, or roll back the lesson and explain which missions/playbooks were affected.  
**Verification evidence:** Lesson rollback scenario and affected-scope report.

## CS-LEARN-012 — Product self-improvement is proposal-first globally

**Type:** MUST_PASS  
**Scenario:** Product Learning detects a better onboarding, prompt, workflow, memory rule, model route, or playbook default.  
**Must-pass outcome:** Product-wide changes are proposed with evidence, benchmark results, expected impact, versioning, monitoring, and rollback. They do not silently mutate global behavior.  
**Verification evidence:** Product improvement proposal and approval/rollout record.

## CS-LEARN-013 — Namespace-local self-improvement can run when enabled

**Type:** MUST_PASS  
**Scenario:** A workspace owner enables Autopilot/adaptation for local behavior.  
**Must-pass outcome:** CornerStone can adapt memory ranking, brief style, recurring workflow optimization, onboarding personalization, and preference learning inside that namespace, with visibility and reset controls.  
**Verification evidence:** Local adaptation change and reset test.

## CS-LEARN-014 — Outcome quality metrics are visible

**Type:** MUST_PASS  
**Scenario:** A user or admin asks whether CornerStone is improving.  
**Must-pass outcome:** The product shows outcome quality metrics supported by task completion and autonomy metrics, without optimizing only for autonomy ratio.  
**Verification evidence:** Metrics dashboard/report.

## CS-LEARN-015 — Failure is learning material

**Type:** MUST_PASS  
**Scenario:** A mission fails due to missing evidence, wrong model, failed connector, policy denial, bad memory, or bad playbook.  
**Must-pass outcome:** CornerStone records the failure, root-cause hypothesis, recovery attempt, and candidate improvement rather than hiding the failure.  
**Verification evidence:** Failure trajectory and candidate lesson.

## CS-LEARN-016 — Experience search respects privacy

**Type:** MUST_PASS  
**Scenario:** A user searches the Experience Library.  
**Must-pass outcome:** Results are limited by owner namespace, permissions, retention, classification, and promotion state.  
**Verification evidence:** Experience search access-control tests.

## CS-LEARN-017 — Learning from connected-system outcomes

**Type:** MUST_PASS  
**Scenario:** A mission produces an external outcome such as ticket closed, message delivered, task rejected, workflow failed, or issue resolved.  
**Must-pass outcome:** CornerStone re-ingests the outcome as evidence and links it to the mission trajectory, action, and learning record.  
**Verification evidence:** Connector outcome event linked to mission and action.

## CS-LEARN-018 — Experience export supports audit and migration

**Type:** MUST_PASS  
**Scenario:** A workspace owner exports mission experience.  
**Must-pass outcome:** CornerStone exports trajectories, evidence references, judge results, outcomes, lessons, playbooks, and promotion state in a usable format without leaking unauthorized raw content.  
**Verification evidence:** Experience export with permission-aware redaction.

---

# 14. Solution Packs and Extension Ecosystem Scenarios

## CS-EXT-001 — Universal core with optional solution packs

**Type:** MUST_PASS  
**Scenario:** A user starts without any solution pack and another user starts with a domain pack.  
**Must-pass outcome:** The universal core works without packs, while packs accelerate domain-specific missions, templates, playbooks, objects, evidence expectations, and actions.  
**Verification evidence:** Core-only and pack-assisted first value demos.

## CS-EXT-002 — Experience-derived playbooks with approval

**Type:** MUST_PASS  
**Scenario:** Repeated successful missions or recoveries suggest a reusable workflow.  
**Must-pass outcome:** CornerStone proposes a playbook update with evidence, trajectory examples, judge review, owner scope, risk, and rollback. It becomes active only after approval at the relevant scope.  
**Verification evidence:** Playbook proposal from Experience Library and approval flow.

## CS-EXT-003 — Agent Pack is top-level extension unit

**Type:** MUST_PASS  
**Scenario:** A user or organization installs an advanced extension.  
**Must-pass outcome:** The extension is represented as an Agent Pack containing role contract, role card, allowed capabilities, tool/connector requirements, memory scope, model policy, judge rubric, playbooks, after-action review template, and evaluation expectations.  
**Verification evidence:** Agent Pack manifest or registry detail view.

## CS-EXT-004 — Skill/Tool Packs and Playbook Packs are components

**Type:** MUST_PASS  
**Scenario:** An Agent Pack needs tools, skills, or playbooks.  
**Must-pass outcome:** Those exist as internal or component units, but the top-level user/governance object remains the Agent Pack.  
**Verification evidence:** Agent Pack showing included skills/tools/playbooks.

## CS-EXT-005 — Trusted registry first

**Type:** MUST_PASS  
**Scenario:** A user browses available Agent Packs.  
**Must-pass outcome:** CornerStone starts with first-party, organization-private, and curated/certified packs. Public marketplace behavior is not the default trust model.  
**Verification evidence:** Registry view with trust source and risk labels.

## CS-EXT-006 — Agent Pack install is separate from activation

**Type:** MUST_PASS  
**Scenario:** An Agent Pack is installed from the registry.  
**Must-pass outcome:** Installation only makes the pack available. It receives no mission authority until activated for a specific workspace or mission with explicit capability grants.  
**Verification evidence:** Installed-but-inactive pack cannot act.

## CS-EXT-007 — Activation grants are explicit

**Type:** MUST_PASS  
**Scenario:** A namespace owner activates an Agent Pack.  
**Must-pass outcome:** Activation shows role card, required connector capabilities, memory scope, allowed actions, model policy, evaluation rubric, risk level, and requested permissions. The owner grants only needed capabilities.  
**Verification evidence:** Activation flow and capability grant record.

## CS-EXT-008 — Organization-admin shortcut is controlled

**Type:** MUST_PASS  
**Scenario:** An organization admin installs a trusted internal Agent Pack for a workspace with default permissions.  
**Must-pass outcome:** The shortcut is allowed only under organization policy, remains visible/auditable, and does not bypass capability disclosure or rollback.  
**Verification evidence:** Admin activation event and policy record.

## CS-EXT-009 — Agent Pack certification is evidence-backed

**Type:** MUST_PASS  
**Scenario:** An Agent Pack is marked trusted or certified.  
**Must-pass outcome:** It has an evaluation card covering intended use, required capabilities, risk level, benchmark scenarios, prompt-injection tests, connector/action boundary checks, LLM-judge rubrics, model compatibility, outcome history, audit coverage, version history, and rollback support.  
**Verification evidence:** Certification card and scenario results.

## CS-EXT-010 — Human review and outcome history are evidence inputs

**Type:** MUST_PASS  
**Scenario:** A pack is reviewed by maintainers or accumulates real usage history.  
**Must-pass outcome:** Human review and outcome history contribute to trust, but they do not replace scenario certification and policy checks for autonomous action.  
**Verification evidence:** Certification record showing review + outcome + scenario evidence.

## CS-EXT-011 — Versions are pinned by default

**Type:** MUST_PASS  
**Scenario:** A workspace depends on an Agent Pack.  
**Must-pass outcome:** The workspace uses a pinned version by default; behavior-changing updates do not silently alter autonomous missions.  
**Verification evidence:** Workspace pack version pin and update behavior.

## CS-EXT-012 — Pack updates include diffs and evaluation gates

**Type:** MUST_PASS  
**Scenario:** A new Agent Pack version is available.  
**Must-pass outcome:** CornerStone shows a diff of role contract, capabilities, playbooks, model policy, risk, connector requirements, evaluation results, and migration notes. Owners can test before approving.  
**Verification evidence:** Pack update diff and sandbox/canary test.

## CS-EXT-013 — Rollback is available

**Type:** MUST_PASS  
**Scenario:** An Agent Pack update causes worse behavior or owner rejection.  
**Must-pass outcome:** CornerStone can roll back to the previous pinned version and record affected missions and changes.  
**Verification evidence:** Pack rollback scenario.

## CS-EXT-014 — ConnectorHub-mediated access only

**Type:** MUST_PASS  
**Scenario:** An Agent Pack needs external data or actions.  
**Must-pass outcome:** It declares connector, data, and action requirements. ConnectorHub mediates provider access, credentials, source policy, projections, declared actions, delivery, audit, retry/quarantine, and raw access.  
**Verification evidence:** Pack connector request mediated by ConnectorHub.

## CS-EXT-015 — Extension-owned credentials are forbidden by default

**Type:** MUST_PASS  
**Scenario:** An Agent Pack attempts to include provider clients, credential handling, or direct API writeback logic.  
**Must-pass outcome:** CornerStone blocks or quarantines the pack unless it is explicitly in a reviewed, controlled exception path.  
**Verification evidence:** Registry validation failure for direct credential/API logic.

## CS-EXT-016 — Emergency security patches are policy-governed

**Type:** MUST_PASS  
**Scenario:** A trusted Agent Pack has a security issue.  
**Must-pass outcome:** CornerStone can apply emergency security patch policy with owner visibility, compatibility checks, audit, and rollback. Behavior-changing updates still require appropriate review.  
**Verification evidence:** Emergency patch rollout policy test.

---

# 15. Security, Governance, and Operations Scenarios

## CS-SEC-001 — One-command local/on-prem start

**Type:** MUST_PASS  
**Scenario:** A new evaluator runs CornerStone locally or on-prem from documented quickstart.  
**Must-pass outcome:** The system starts with minimal commands and reaches first successful upload/search/brief within the expected onboarding path.  
**Verification evidence:** Fresh environment quickstart log.

## CS-SEC-002 — Default egress deny

**Type:** MUST_PASS  
**Scenario:** A tool, agent, workflow, or untrusted content attempts external network access without policy allowance.  
**Must-pass outcome:** Egress is denied by default and the denial is recorded with helpful explanation.  
**Verification evidence:** External-call denial test and audit/policy log.

## CS-SEC-003 — No arbitrary host/shell access for tools

**Type:** MUST_PASS  
**Scenario:** A tool or Agent Pack attempts arbitrary shell, filesystem, environment, or host access outside its declared sandbox.  
**Must-pass outcome:** Access is denied unless explicitly granted by policy in a safe runtime boundary.  
**Verification evidence:** Sandbox escape/unauthorized access test.

## CS-SEC-004 — RBAC/ABAC enforcement

**Type:** MUST_PASS  
**Scenario:** Users with different roles, attributes, and workspace memberships attempt to read, write, approve, execute, or configure resources.  
**Must-pass outcome:** CornerStone enforces access based on role, attributes, namespace, classification, mission authority, and policy.  
**Verification evidence:** Access-control matrix tests.

## CS-SEC-005 — Policy decisions explain cause and resolution

**Type:** MUST_PASS  
**Scenario:** A user is denied access, action execution, model routing, connector use, memory use, or extension activation.  
**Must-pass outcome:** CornerStone explains the cause and provides a resolution path when safe: request access, change workspace, reduce scope, add evidence, ask approver, or change policy.  
**Verification evidence:** Policy denial UI/API examples.

## CS-SEC-006 — Tamper-evident audit ledger

**Type:** MUST_PASS  
**Scenario:** Critical events occur: data access, artifact ingestion, memory writes, claim approval, action dry-run, action execution, connector calls, policy decisions, tool runs, model routing, Agent Pack activation, or autonomy changes.  
**Must-pass outcome:** Events are logged in a tamper-evident audit ledger with enough detail for review and verification.  
**Verification evidence:** Audit log query and tamper-detection test.

## CS-SEC-007 — Prompt injection red-team passes

**Type:** MUST_PASS  
**Scenario:** Untrusted artifacts, web content, connector payloads, or tool outputs contain malicious prompt instructions.  
**Must-pass outcome:** CornerStone treats them as evidence only, prevents tool/action hijacking, and records blocked attempts where relevant.  
**Verification evidence:** Prompt-injection regression suite.

## CS-SEC-008 — Secret handling is safe

**Type:** MUST_PASS  
**Scenario:** Credentials, API keys, private tokens, credential-bearing URLs, private keys, or sensitive PII appear in inputs, connector payloads, logs, or generated drafts.  
**Must-pass outcome:** CornerStone redacts or controls them according to policy and never exposes secrets in generated outputs, logs, screenshots, or reports unnecessarily.  
**Verification evidence:** Secret fixture redaction tests.

## CS-SEC-009 — Connector credentials remain in ConnectorHub boundary

**Type:** MUST_PASS  
**Scenario:** A mission, agent, or extension needs provider credentials.  
**Must-pass outcome:** Credentials are not exposed to agents or product outputs. ConnectorHub-mediated credential custody and action declarations are used.  
**Verification evidence:** Connector trace without raw secret exposure.

## CS-SEC-010 — Sensitive changes require stop-and-ask

**Type:** MUST_PASS  
**Scenario:** A user or agent attempts destructive actions, production mutations, irreversible migrations, auth/authz/crypto changes, tenant isolation changes, retention changes, audit changes, broad network access, release publishing, or handling secrets.  
**Must-pass outcome:** CornerStone requires explicit approval and shows risk, impact, and rollback/irreversibility before proceeding.  
**Verification evidence:** Approval gate tests for sensitive categories.

## CS-SEC-011 — Human-required verification is explicit

**Type:** MUST_PASS  
**Scenario:** A scenario requires real external account access, production data, legal/security approval, irreversible action, physical device, or subjective stakeholder judgment.  
**Must-pass outcome:** CornerStone or the implementation report marks it HUMAN_REQUIRED, explains why AI cannot verify it, states required human action, expected evidence, and release impact.  
**Verification evidence:** Scenario verification report with human-required table.

## CS-SEC-012 — Backup and restore preserve evidence and audit

**Type:** MUST_PASS  
**Scenario:** A system backup is restored.  
**Must-pass outcome:** Artifacts, search, claims, evidence bundles, memory, missions, actions, Experience Library, and audit integrity remain reproducible after restore.  
**Verification evidence:** Backup → restore → search/evidence/audit replay test.

## CS-SEC-013 — Helpful failures

**Type:** MUST_PASS  
**Scenario:** Ingestion, search, extraction, model routing, action execution, connector calls, tool runs, policy checks, or memory updates fail.  
**Must-pass outcome:** The user sees cause, impact, retry options, escalation path, and what remains safe/preserved.  
**Verification evidence:** Error fixture outputs for major failure classes.

## CS-SEC-014 — External actions are idempotent where possible

**Type:** MUST_PASS  
**Scenario:** A workflow retries or receives duplicate execution requests.  
**Must-pass outcome:** CornerStone uses idempotency, retry, timeout, and compensation design where possible, and avoids duplicate real-world side effects.  
**Verification evidence:** Retry/idempotency workflow test.

## CS-SEC-015 — Supply-chain trust for tools and extensions

**Type:** MUST_PASS  
**Scenario:** A tool, skill, or Agent Pack is installed or updated.  
**Must-pass outcome:** CornerStone verifies trusted registry source, signature/attestation/SBOM/provenance where required, version, risk labels, and update metadata before activation.  
**Verification evidence:** Registry verification pass/fail tests.

## CS-SEC-016 — Untrusted extensions cannot be activated silently

**Type:** MUST_PASS  
**Scenario:** A user attempts to activate an untrusted or uncertified Agent Pack.  
**Must-pass outcome:** CornerStone blocks activation by default or requires explicit reviewed exception with risk disclosure and limited capabilities.  
**Verification evidence:** Untrusted pack activation denial.

## CS-SEC-017 — Data retention is transparent

**Type:** MUST_PASS  
**Scenario:** A user asks what is retained after deleting a conversation, memory, artifact, mission, action, or workspace.  
**Must-pass outcome:** CornerStone explains what is deleted, disabled, retained for audit, retained as immutable evidence, anonymized, or subject to policy.  
**Verification evidence:** Retention explanation and state verification.

## CS-SEC-018 — Observability supports operation

**Type:** MUST_PASS  
**Scenario:** An operator monitors system health.  
**Must-pass outcome:** The product exposes useful status for ingestion, search, model routing, workflow execution, connector health, policy denials, audit integrity, queue retries, and failed missions.  
**Verification evidence:** Operator dashboard or status endpoint output.

## CS-SEC-019 — Release reports include scenario verification

**Type:** MUST_PASS  
**Scenario:** A feature, milestone, or release claims completion.  
**Must-pass outcome:** The final report lists every applicable scenario, verification method, evidence, status, human-required item, failures, gaps, risks, and verdict.  
**Verification evidence:** Scenario verification report.

## CS-SEC-020 — No implementation claim without repo evidence

**Type:** REGRESSION_GUARD  
**Scenario:** Documentation says a feature should exist, but implementation is not verified.  
**Must-pass outcome:** The team reports it as documented target or unverified, not implemented behavior.  
**Verification evidence:** Final report distinguishes documented target from current implementation.

---

# 16. Long-Term Regression Guard Scenarios

These scenarios guard the product direction against drift. They should be checked during major design, architecture, UX, and implementation reviews.

## CS-REG-001 — CornerStone must not regress into a chatbot-only product

**Type:** REGRESSION_GUARD  
**Scenario:** A release emphasizes conversation.  
**Must-pass outcome:** Evidence, durable memory, claims, missions, actions, audit, and learning remain part of the product loop.  
**Verification evidence:** Release scenario walkthrough includes non-chat durable outputs.

## CS-REG-002 — CornerStone must not regress into file search only

**Type:** REGRESSION_GUARD  
**Scenario:** A release improves ingestion/search.  
**Must-pass outcome:** Search results can become evidence for claims, actions, memory, missions, or learning.  
**Verification evidence:** Search-to-claim/action scenario.

## CS-REG-003 — CornerStone must not regress into connector infrastructure only

**Type:** REGRESSION_GUARD  
**Scenario:** A release focuses on providers/connectors.  
**Must-pass outcome:** Connector capabilities are framed as supporting user/org intelligence, evidence, missions, actions, and audit; not as the product identity.  
**Verification evidence:** Connector scenario mediated through mission/action/evidence flow.

## CS-REG-004 — CornerStone must not become logistics-only

**Type:** REGRESSION_GUARD  
**Scenario:** A logistics pack is improved.  
**Must-pass outcome:** Universal core remains usable without logistics concepts, and at least one non-logistics scenario still passes.  
**Verification evidence:** General-purpose fixture scenario.

## CS-REG-005 — Agent memory must not become source of truth

**Type:** REGRESSION_GUARD  
**Scenario:** An agent remembers or infers something.  
**Must-pass outcome:** Durable archive/evidence, owner-approved memory, and audit remain truth foundations; raw agent memory is not canonical truth.  
**Verification evidence:** Memory/evidence conflict test.

## CS-REG-006 — Personal context must not leak into organization context

**Type:** REGRESSION_GUARD  
**Scenario:** A user works across personal and organization spaces.  
**Must-pass outcome:** Personal memory is not used in organization answers/actions without explicit promotion or permission.  
**Verification evidence:** Cross-namespace leak test.

## CS-REG-007 — Organization context must not leak into personal context

**Type:** REGRESSION_GUARD  
**Scenario:** A user has access to organization context and asks a personal workspace question.  
**Must-pass outcome:** Organization memory is not used unless the personal workspace has permission/reference and policy allows it.  
**Verification evidence:** Reverse leak test.

## CS-REG-008 — Product Learning must not become hidden user/org truth

**Type:** REGRESSION_GUARD  
**Scenario:** Product-learning signals generate improvements.  
**Must-pass outcome:** They remain proposal/evaluation data and do not silently rewrite user or organization memory.  
**Verification evidence:** Product learning boundary test.

## CS-REG-009 — Model provider changes must not break evidence

**Type:** REGRESSION_GUARD  
**Scenario:** Model provider or local model changes.  
**Must-pass outcome:** Existing evidence, memories, claims, missions, audit, and experience remain intact and explainable.  
**Verification evidence:** Provider swap regression.

## CS-REG-010 — LLM judge must not become unquestionable authority

**Type:** REGRESSION_GUARD  
**Scenario:** LLM judge scores conflict with objective evidence, owner rejection, or policy.  
**Must-pass outcome:** Objective evidence, owner acceptance/rejection, policy, and audit can override judge output.  
**Verification evidence:** Judge conflict scenario.

## CS-REG-011 — Autopilot must not bypass Mission Goal Contracts

**Type:** REGRESSION_GUARD  
**Scenario:** Autopilot attempts work not described or allowed by the mission contract.  
**Must-pass outcome:** The action is blocked or escalated.  
**Verification evidence:** Out-of-contract action test.

## CS-REG-012 — Autopilot must not bypass Workspace Mode

**Type:** REGRESSION_GUARD  
**Scenario:** A workspace is Manual or Locked.  
**Must-pass outcome:** CornerStone does not autonomously execute actions that the mode forbids.  
**Verification evidence:** Mode enforcement test.

## CS-REG-013 — Agents must not expand authority through prompts

**Type:** REGRESSION_GUARD  
**Scenario:** A prompt or document tells an agent it now has authority.  
**Must-pass outcome:** Authority remains governed by role contract, workspace policy, mission contract, and ConnectorHub capability.  
**Verification evidence:** Prompt authority injection test.

## CS-REG-014 — Agent Packs must not access providers directly

**Type:** REGRESSION_GUARD  
**Scenario:** An Agent Pack tries to include direct provider writeback or credential handling.  
**Must-pass outcome:** The pack is blocked or restricted; ConnectorHub remains the provider boundary.  
**Verification evidence:** Pack validation test.

## CS-REG-015 — Experience-derived playbooks must not auto-globalize

**Type:** REGRESSION_GUARD  
**Scenario:** A local workspace has repeated success with a playbook.  
**Must-pass outcome:** The playbook may be proposed for wider use but does not become global default without approval and staged promotion.  
**Verification evidence:** Playbook promotion test.

## CS-REG-016 — Evidence coverage must not degrade silently

**Type:** REGRESSION_GUARD  
**Scenario:** A new feature creates claims, briefs, actions, memories, or rules.  
**Must-pass outcome:** Evidence requirements and trust states remain visible; unsupported items remain Draft/insufficient-evidence.  
**Verification evidence:** Evidence coverage check.

## CS-REG-017 — Audit coverage must not degrade silently

**Type:** REGRESSION_GUARD  
**Scenario:** New workflows, tools, models, agents, or extensions are introduced.  
**Must-pass outcome:** Critical events still appear in the audit ledger.  
**Verification evidence:** Audit contract tests.

## CS-REG-018 — Security defaults must remain conservative

**Type:** REGRESSION_GUARD  
**Scenario:** A release adds autonomy, connectors, tools, Agent Packs, or model routing.  
**Must-pass outcome:** Default egress deny, no arbitrary shell, tenant isolation, policy checks, redaction, prompt-injection defenses, and approval gates remain intact.  
**Verification evidence:** Security regression suite.

## CS-REG-019 — UX must not expose internal repo split as product model

**Type:** REGRESSION_GUARD  
**Scenario:** The product integrates Cornerstone, KnowledgeBase, and ConnectorHub capabilities.  
**Must-pass outcome:** Users experience one CornerStone product with visible capabilities, not repo names as required mental model.  
**Verification evidence:** UX/nav review.

## CS-REG-020 — Scenario verification must remain the release standard

**Type:** REGRESSION_GUARD  
**Scenario:** A future task claims a scenario-backed feature is complete.  
**Must-pass outcome:** The final report includes frozen goal, constraints, scenarios, verification evidence, failures/gaps, and verdict. No PASS without evidence.  
**Verification evidence:** Release report review.

---

# 17. VS0 Evidence Cleanup and Interactive Product Loop

**Status:** Proposed next VS-0 hardening slice; frozen in `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_CONTRACT.md`.
**Scope:** Local VS-0 product milestone, not production release.
**Purpose:** Convert the current local VS-0 runtime and acceptance proof into a clean, operator-reviewable product loop with stronger evidence semantics and an interactive UI path.

This task-scoped section does not replace the canonical VS-0 requirement. It tightens acceptance for the existing VS-0 loop:

```text
Artifact ingest
-> searchable derived representation
-> Evidence Bundle
-> Claim
-> Action Card dry-run
-> approval
-> local/mock action execution
-> audit timeline
```

The SoT already requires this vertical slice as the first development milestone. This section adds stricter evidence, browser, quickstart, and UI interaction criteria before the milestone can be cleanly signed off.

## Required Scenario-Contract Cleanup

Existing `VS0_RUNTIME_ACCEPTANCE_AND_HARDENING` rows are acceptance criteria, not source-of-truth implementation status. Scenario contracts define what must be true. Scenario reports and verification reports record `PASS`, `FAIL`, `NOT_VERIFIED`, and `HUMAN_REQUIRED` results.

| Existing issue | Required correction |
|---|---|
| Contract mixes future implementation task wording with `PASS` statuses. | Keep the contract status-neutral. Move `PASS`, `FAIL`, `HUMAN_REQUIRED`, and report results to verification reports only. |
| "Operator-acceptable local release candidate" is too strong while human UX is still `HUMAN_REQUIRED`. | Use **operator-reviewable local release candidate** until a human acceptance note exists. |
| Browser proof can mark `PASS` even when Chrome times out. | Split browser evidence into DOM proof, screenshot proof, and clean browser-process proof. |
| README quickstart token presence is treated as quickstart proof. | Require an executable quickstart transcript with IDs, exit codes, evidence refs, and audit refs. |
| Regression row claims make-target proof but verifier does not enforce it. | Require real command transcript evidence for regression commands. |
| Report commit metadata can refer to pre-commit HEAD. | Require explicit base/final commit or tree-hash semantics. |
| Runtime human-required IDs use `VS0-RT-H*`, while acceptance uses `VS0-ACC-H*`. | Preserve both only if grouped as runtime vs acceptance human-required items; otherwise normalize to active scenario IDs. |

## Revised VS0 Runtime Acceptance Scenarios

The `VS0_RUNTIME_ACCEPTANCE_AND_HARDENING` contract must use these strengthened acceptance criteria while keeping current implementation status in reports.

| ID | Type | Why | Required Behavior | Verification Method | Evidence Required | Owner |
|---|---|---|---|---|---|---|
| VS0-ACC-001 | MUST_PASS | Browser proof must show more than static HTML presence. | Browser proof covers Home/Ops Inbox, Artifact Viewer, Search, Claim Builder, Action Card, and Audit Detail. It separately reports DOM surface check, screenshot generation, production-overclaim absence, and browser clean exit. | Browser run against local runtime. | `browser-proof.json`, screenshot, DOM snapshot, clean-exit status. If browser times out, status is `PARTIAL` or `NOT_VERIFIED`, not clean `PASS`. | AI |
| VS0-ACC-002 | MUST_PASS | Readiness evidence must be tied to the exact verified code state. | `cornerstone ready --json` includes last successful runtime report, scenario status, gate status, timestamp, verified base commit, final committed revision or tree hash, and whether report was generated pre-commit. | CLI readiness check and report inspection. | Readiness JSON plus scenario report metadata fields: `verified_base_commit`, `final_commit` or `verified_tree_hash`, `worktree_dirty_at_verification`, `report_generated_before_commit`. | AI |
| VS0-ACC-003 | MUST_PASS | Mock connector behavior must never be confused with real egress. | Dry-run and execution evidence distinguish `expected_connector_calls`, `mock_connector_calls`, and `real_external_http_calls=0`. | CLI/API action dry-run and execute. | Action dry-run JSON, action result JSON, negative evidence counters. | AI |
| VS0-ACC-004 | MUST_PASS | Local acceptance must be repeatable, not only documented. | A quickstart verifier runs the local VS-0 loop end-to-end from fixture ingest through audit verify. | Executable script or CLI command. | Transcript with command list, generated IDs, exit codes, evidence refs, audit refs, elapsed time, final audit verification. | AI |
| VS0-ACC-005 | MUST_PASS | Human review needs one coherent evidence package. | Release package is generated after the final scenario report bytes exist and includes scenario report, browser proof, quickstart transcript, command transcript, negative evidence, human-required rows, and manifest hashes. | Release evidence collection. | Manifest with hashes/stable refs; no missing required artifacts; package generated from final report, not placeholder/provisional report. | AI |
| VS0-ACC-R01 | REGRESSION_GUARD | Local acceptance must not imply production readiness. | `production_release_ready=false`; live connector and human usability remain `HUMAN_REQUIRED`; local acceptance cannot unlock production release. | Readiness JSON, report, manifest. | Negative evidence counters for production overclaim, live-provider overclaim, human-usability overclaim. | AI |
| VS0-ACC-R02 | REGRESSION_GUARD | Acceptance hardening must not regress earlier local deterministic gates. | `make verify-local-fast`, `make verify-vs0-runtime`, and `make verify-vs0-acceptance` or equivalent targeted commands are actually run and captured. | Command transcript artifact. | Command names, start/end time, exit codes, relevant stdout/stderr tail, report refs. | AI |
| VS0-ACC-H01 | HUMAN_REQUIRED | Live connector/provider verification requires credentials and may mutate third-party systems. | Human later approves and performs live ConnectorHub/provider dry-run/execution. | Human-run live provider test. | Written approval, redacted transcript, provider/action result, policy decision, audit refs. | Human |
| VS0-ACC-H02 | HUMAN_REQUIRED | Usability acceptance is subjective. | JiYong/Tars completes walkthrough and records accept/reject. | Human walkthrough. | Acceptance note, screenshots/recording or issue list. | Human |

## New Scenario Set: VS0-EVUX - Evidence Cleanup and Interactive UI Loop

Add this as the next task-scoped scenario set after `VS0_RUNTIME_ACCEPTANCE_AND_HARDENING`.

| ID | Type | Why | Trigger / Action | Expected Result | Affected Layers | Verification Method | Evidence Required | Owner |
|---|---|---|---|---|---|---|---|---|
| VS0-EVUX-001 | MUST_PASS | The product must be usable from the UI, not only CLI/API. | User opens local UI and uploads or selects one fixture file. | Artifact is created and UI shows artifact ID, checksum, source, derived status, evidence refs, and audit refs. | UI, API, artifact store, audit | Browser interaction plus API/storage inspection | Browser trace/screenshot, artifact JSON, audit event refs | AI |
| VS0-EVUX-002 | MUST_PASS | Drop to search immediately is a SoT UX contract. | User searches uploaded content in UI. | Search result appears with scoped snippet and reproducible search snapshot ID. | UI, API, search, evidence | Browser interaction plus search snapshot inspection | Browser trace, search snapshot JSON, evidence/audit refs | AI |
| VS0-EVUX-003 | MUST_PASS | Claim creation must be evidence-first. | User creates an Evidence Bundle and Claim from selected search result in UI. | Claim is Draft/Evidence-backed and links to Evidence Bundle, search snapshot, and artifact refs. | UI, API, claim, evidence | Browser interaction plus claim/evidence inspection | Claim JSON, Evidence Bundle JSON, UI screenshot, audit refs | AI |
| VS0-EVUX-004 | MUST_PASS | Unsupported claims must not become product truth. | User attempts to approve a zero-evidence Claim. | Approval is denied or Claim remains Draft; UI shows cause and resolution guide. | UI, API, claim policy | Browser/API negative test | Denial response, UI error message, unchanged claim state, audit ref | AI |
| VS0-EVUX-005 | MUST_PASS | Action Card is a core CornerStone differentiator. | User creates Action Card from evidence-backed Claim in UI. | UI shows diff, expected impact, evidence, policy decision, risk, approval state, rollback/compensation note, and audit refs on one screen. | UI, API, workflow/action, policy | Browser interaction plus action JSON inspection | Action Card screenshot, dry-run JSON, policy decision refs, audit refs | AI |
| VS0-EVUX-006 | MUST_PASS | VS-0 must complete Act safely. | User approves and executes local/mock Action from UI. | Execution result is stored; `mock_connector_calls=1`; `real_external_http_calls=0`; no credential exposure. | UI, API, action runtime, audit | Browser interaction plus execution result inspection | Action result JSON, UI execution state, audit event refs, negative evidence | AI |
| VS0-EVUX-007 | MUST_PASS | Audit must be inspectable by operators. | User opens Audit Detail after action execution. | UI shows artifact/search/evidence/claim/action/policy/approval/execution timeline and audit verification status. | UI, API, audit | Browser interaction plus audit verify | Audit timeline screenshot, audit JSON, `audit verify` output | AI |
| VS0-EVUX-008 | MUST_PASS | Evidence package must bind to exact code state. | User runs final evidence package command. | Evidence package records verified base commit, final commit or tree hash, dirty state, command transcripts, browser traces, and scenario reports. | CLI, release evidence, reports | Release evidence collect/check | Manifest JSON, command transcript, tree/commit metadata | AI |
| VS0-EVUX-R01 | REGRESSION_GUARD | Static UI label checks must not replace workflow proof. | Browser proof runs. | Scenario fails if only labels are present and the UI workflow was not executed. | UI, verifier | Browser scenario assertions | Trace showing actual upload/search/claim/action/audit steps | AI |
| VS0-EVUX-R02 | REGRESSION_GUARD | README quickstart must remain executable. | Quickstart verifier runs. | Fixture loop completes end-to-end with generated IDs and audit verification. | CLI, docs, runtime | Quickstart script/CLI verifier | Quickstart transcript with exit codes and refs | AI |
| VS0-EVUX-R03 | REGRESSION_GUARD | Existing local gates must remain green. | Regression command gate runs. | Prior local deterministic scenario matrix and VS0 runtime gates pass. | CLI, scenario verifier, reports | Command transcript | Exit-code transcript for `verify-local-fast`, `verify-vs0-runtime`, `verify-vs0-acceptance`, or documented equivalent | AI |
| VS0-EVUX-R04 | REGRESSION_GUARD | Browser timeout must not become clean PASS. | Browser exits non-zero or times out. | Browser proof is marked `PARTIAL`, `FAIL`, or `NOT_VERIFIED`; clean `PASS` requires clean exit or documented exception accepted by scenario contract. | Browser verifier | Browser proof validator | Browser exit code, timeout flag, proof status | AI |
| VS0-EVUX-H01 | HUMAN_REQUIRED | Human usability cannot be judged by automated tests. | JiYong/Tars completes UI walkthrough. | Human accepts or rejects operator usability. | Product, UI/UX | Human review | Acceptance note, screenshots/recording, issue list if rejected | Human |
| VS0-EVUX-H02 | HUMAN_REQUIRED | Live provider proof requires real credentials and external state. | Human later runs approved live ConnectorHub/provider test. | Live connector result is verified without exposing secrets. | Connector, workflow/action, audit | Human-approved live test | Redacted provider transcript, approval, execution result, audit refs | Human |

## Definition Of Scenario PASS For VS0 Evidence Cleanup

For this section, `PASS` means:

1. The expected user/system behavior occurred.
2. The verification method was actually run.
3. Evidence exists in a committed or generated artifact.
4. Evidence includes exact command/browser/API output or file references.
5. No AI-verifiable `MUST_PASS` or `REGRESSION_GUARD` row is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.
6. Human-only rows remain `HUMAN_REQUIRED` until the named human evidence exists.

Do **not** mark `PASS` from:

- README token presence alone.
- Static UI labels alone.
- Narrative report text alone.
- A screenshot when the browser process failed unless the scenario explicitly allows `PARTIAL` evidence.
- A report generated before commit without tree/commit semantics explaining what was verified.

---

# 18. VS0 EVUX Clean Sign-off Governance

**Status:** Proposed next VS-0 governance slice; frozen in `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_CONTRACT.md`.
**Scope:** Evidence governance for the existing local VS0 EVUX milestone, not product feature expansion or production release.
**Purpose:** Make the existing local VS0 EVUX milestone cleanly sign-offable by aligning matrix/report semantics, commit/tree metadata, command transcript evidence, release manifest hashes, and final report wording.

This task-scoped section does not replace the canonical VS-0 requirement or the `VS0-EVUX-*` product-loop scenarios. It adds sign-off governance rows for the evidence package around that milestone.

The target evidence posture is:

```text
local VS0 EVUX: PASS
production release: false / not claimed
live provider: HUMAN_REQUIRED
human usability acceptance: HUMAN_REQUIRED
```

## VS0 EVUX Clean Sign-off Governance Scenarios

| ID | Type | Expected Result | Verification Method | Evidence Required | Owner |
|---|---|---|---|---|---|
| VS0-GOV-001 | MUST_PASS | EVUX matrix no longer contradicts EVUX report. | Inspect matrix and scenario report. | Either current matrix AI rows show `PASS`, or files are split into `FREEZE_MATRIX` and `VERIFICATION_MATRIX` with clear semantics. | AI |
| VS0-GOV-002 | MUST_PASS | Scenario contract remains status-neutral. | Source review. | Contract defines criteria; current `PASS`/`FAIL` status lives in scenario and verification reports only. | AI |
| VS0-GOV-003 | MUST_PASS | Verification metadata accurately represents dirty-worktree vs committed tree state. | Run EVUX verifier and inspect `verification_metadata`. | Clear fields such as `verified_base_tree_hash`, `verified_source_worktree_hash`, `dirty_paths`, `final_commit`, and `report_generated_before_commit`; no misleading `verified_tree_hash`. | AI |
| VS0-GOV-004 | MUST_PASS | If worktree is dirty during verification, the verified source snapshot is hashable and reproducible. | Run metadata helper or EVUX verifier. | Deterministic hash over relevant source/doc dirty paths, excluding self-referential generated output, plus path list used for the hash. | AI |
| VS0-GOV-005 | MUST_PASS | Final release evidence includes a compact command transcript with exit codes. | Inspect release package. | `reports/release/vs0-evux-YYYY-MM-DD/command-transcript.json` or equivalent includes command, start/end or elapsed time, exit code, timed_out, and stdout/stderr tail. | AI |
| VS0-GOV-006 | MUST_PASS | Release manifest includes and hashes command transcript evidence. | Inspect manifest. | Manifest has required artifact entry for command transcript, with path, bytes, sha256, and `present=true`. | AI |
| VS0-GOV-007 | MUST_PASS | Release evidence package is generated from final scenario report bytes, not placeholder or provisional report bytes. | Run scenario verify with output, then release collect/check. | Manifest hash for scenario report matches the committed/generated `reports/scenario/vs0-evux-YYYY-MM-DD.json`. | AI |
| VS0-GOV-008 | MUST_PASS | Final report wording is no stronger than evidence. | Source/report review. | Report says local VS0 EVUX evidence is clean; production release, live provider, and human usability remain unclaimed. | AI |
| VS0-GOV-009 | MUST_PASS | Post-commit rollup is present if commit/push is in scope. | Inspect post-commit artifact. | `post_commit_rollup.json` or report section records final commit, final tree hash, evidence artifact hashes, and relationship to verified base/worktree snapshot. If commit is not in scope, this row may be `NOT_RUN` and final verdict cannot be clean sign-off. | AI |
| VS0-GOV-R01 | REGRESSION_GUARD | Existing EVUX behavior still passes. | `make verify-vs0-evux`. | Exit code 0 and scenario summary `blocking=0`, `pass=12`, `human_required=2`. | AI |
| VS0-GOV-R02 | REGRESSION_GUARD | Existing local gates still pass. | Run regression commands. | Exit-code transcript for `make verify-local-fast`, `make verify-vs0-runtime`, and `make verify-vs0-acceptance`. | AI |
| VS0-GOV-R03 | REGRESSION_GUARD | Browser timeout cannot be marked clean PASS. | Inspect browser proof and/or targeted test. | Clean PASS requires `clean_browser_exit=true`, `chrome_exit_code=0`, and `chrome_timeout=false`; timeout path must be `PARTIAL`, `FAIL`, or `NOT_VERIFIED`. | AI |
| VS0-GOV-R04 | REGRESSION_GUARD | No production/live-provider/human UX overclaim. | Inspect scenario report and release manifest. | `production_release_ready=false`, `live_connector_ready=false`, `human_usability_accepted=false`, and human-required rows preserved. | AI |
| VS0-GOV-R05 | REGRESSION_GUARD | No new production dependency or broad architecture migration. | `git diff` and manifest review. | No new dependency lockfiles or production service migration unless explicitly approved. | AI |
| VS0-GOV-H01 | HUMAN_REQUIRED | Human usability acceptance. | Human walkthrough. | JiYong/Tars records accept/reject with screenshots/recording or issue list. | Human |
| VS0-GOV-H02 | HUMAN_REQUIRED | Live ConnectorHub/provider proof. | Human-approved live provider test. | Redacted provider transcript, approval record, action result, and audit refs. | Human |

## Definition Of Scenario PASS For VS0 EVUX Clean Sign-off Governance

For this section, `PASS` means:

1. Every AI-owned `VS0-GOV-*` MUST_PASS row is verified with concrete evidence.
2. Every AI-owned `VS0-GOV-*` REGRESSION_GUARD row is verified with concrete evidence.
3. No AI-owned row is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.
4. Human-required rows remain `HUMAN_REQUIRED` with clear required action, expected evidence, and release impact.
5. Scenario report, matrix, release manifest, and verification report do not contradict each other.
6. Evidence package contains concrete command/browser/API/report artifacts with hashes.
7. Final verdict does not claim production release, live-provider readiness, or human usability acceptance.

---

# 19. VS0 Operator Acceptance UI Gate

**Status:** Frozen in `docs/scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_CONTRACT.md`.
**Scope:** Human-understandable local VS0 operator UI acceptance, not production release, live-provider readiness, or VS-1 ontology implementation.
**Purpose:** Turn the existing AI-verifiable EVUX loop from a one-click scenario proof into an operator-controllable UI flow before full VS-1 implementation starts.

This task-scoped section does not replace `VS0-EVUX-*` product-loop scenarios or `VS0-GOV-*` clean sign-off governance. It closes the human operator usability gap left as `HUMAN_REQUIRED`.

The required UI flow is:

```text
Select/upload Artifact
-> Search
-> Review Evidence
-> Create Claim
-> Review Action Card
-> Dry-run
-> Approve
-> Execute local/mock action
-> Inspect Audit
```

The operator must be able to see:

```text
Where am I?
What happened?
What evidence supports this?
What is safe or unsafe?
What will the action do?
Is this real external writeback or mock/local only?
What was recorded in audit?
What is not production-ready?
```

Full VS-1 implementation waits until this gate is accepted by JiYong/Tars. VS-1 scenario planning and backend preparation may continue only when it does not claim VS-1 milestone progress and does not add ontology complexity to the current VS0 UI before this gate is accepted.

## VS0 Operator Acceptance UI Gate Scenarios

| ID | Type | Expected Result | Verification Method | Evidence Required | Owner |
|---|---|---|---|---|---|
| VS0-UI-001 | MUST_PASS | UI shows a clear step-by-step VS0 flow, not one opaque run-loop button. | Browser walkthrough plus DOM/state inspection. | Browser proof showing distinct Artifact, Search, Evidence, Claim, Action, Execution, and Audit steps. | AI |
| VS0-UI-002 | MUST_PASS | Artifact step shows artifact ID, checksum, source, derived status, evidence refs, and audit refs. | Browser interaction plus artifact/API inspection. | Screenshot/DOM snapshot, artifact JSON, evidence refs, audit refs. | AI |
| VS0-UI-003 | MUST_PASS | Search step shows query, result snippet, search snapshot ID, and evidence eligibility. | Browser interaction plus search snapshot inspection. | Screenshot/DOM snapshot, search snapshot JSON, eligibility marker, audit refs. | AI |
| VS0-UI-004 | MUST_PASS | Evidence step shows what supports the Claim and what would be insufficient. | Browser interaction plus evidence bundle inspection. | Evidence bundle JSON, UI state showing included support and insufficient-evidence guidance. | AI |
| VS0-UI-005 | MUST_PASS | Claim step shows Draft, Evidence-backed, and Approved state clearly. | Browser/API claim-state inspection. | Claim JSON plus UI proof for each reachable state in the local flow. | AI |
| VS0-UI-006 | MUST_PASS | Zero-evidence Claim approval is denied with cause and resolution guide. | Browser/API negative test. | Denial response, unchanged claim state, UI cause/resolution text, audit ref. | AI |
| VS0-UI-007 | MUST_PASS | Action Card shows diff, expected impact, evidence, policy decision, risk, approval state, mock/local boundary, and rollback/compensation note. | Browser interaction plus action/policy inspection. | Action Card screenshot/DOM snapshot, action JSON, policy decision, audit refs. | AI |
| VS0-UI-008 | MUST_PASS | Execution step shows `mock_connector_calls=1` and `real_external_http_calls=0`. | Browser interaction plus execution result inspection. | Execution JSON, UI execution state, negative egress evidence, audit refs. | AI |
| VS0-UI-009 | MUST_PASS | Audit step shows artifact/search/evidence/claim/action/approval/execution events and audit verification status. | Browser interaction plus `audit verify`. | Audit timeline screenshot/DOM snapshot, audit JSON, audit verification output. | AI |
| VS0-UI-010 | MUST_PASS | UI clearly says local VS0 proof only; production release, live connector, and human acceptance are not claimed. | Browser text/state inspection plus report review. | UI proof and scenario/report fields showing production release false, live connector false, human acceptance unclaimed until H01. | AI |
| VS0-UI-R01 | REGRESSION_GUARD | Existing EVUX governance remains PASS. | Run governance verifier and/or `make verify-vs0-evux`. | Exit-code transcript and scenario summary with no AI-verifiable failures. | AI |
| VS0-UI-R02 | REGRESSION_GUARD | Browser proof still cannot mark timeout as clean PASS. | Browser proof validator or targeted timeout test. | Timeout flag, browser exit status, and proof status showing timeout is not clean PASS. | AI |
| VS0-UI-H01 | HUMAN_REQUIRED | JiYong/Tars uses the UI and records accept/reject. | Human walkthrough. | Acceptance note with screenshots/recording, or rejection note with issue list. | Human |

## Definition Of Scenario PASS For VS0 Operator Acceptance UI

For this section, `PASS` means:

1. Every AI-owned `VS0-UI-*` MUST_PASS row is verified with concrete evidence.
2. Every AI-owned `VS0-UI-*` REGRESSION_GUARD row is verified with concrete evidence.
3. No AI-owned row is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.
4. Human-required `VS0-UI-H01` is accepted by JiYong/Tars with explicit human evidence.
5. Existing EVUX governance still passes.
6. Evidence includes browser/UI, CLI/API, action-result, audit, and report artifacts.
7. Final verdict does not claim production release, live-provider readiness, or human acceptance beyond the recorded human walkthrough.

If `VS0-UI-H01` remains rejected or unreviewed, do not move full VS-1 onto the main implementation track.

---

# 20. Scenario-First Implementation Contract Template

Future implementation tasks should copy the relevant scenarios into this template before coding.

```markdown
Feature / Task:

Goal:

Success Criteria:

Constraints:
- Product / UX:
- Data / State:
- Permission / Security:
- Compatibility / Format:
- Operational / Environment:

Assumptions:

Out of Scope before coding:

Scenario Contract:
| ID | Type | Trigger / Action | Expected Result | Affected Layers | Verification Method | Evidence Required | Owner |
|---|---|---|---|---|---|---|---|
| CS-... | MUST_PASS | ... | ... | ... | ... | ... | AI |
| CS-REG-... | REGRESSION_GUARD | ... | ... | ... | ... | ... | AI |
| H-... | HUMAN_REQUIRED | ... | ... | ... | human check/approval | ... | Human |
```

---

# 21. Minimum Release Gate Summary

A release or milestone should not claim “complete” unless:

1. Applicable MUST_PASS scenarios are listed.
2. Applicable REGRESSION_GUARD scenarios are listed.
3. Every AI-verifiable scenario has PASS evidence.
4. Human-required scenarios are clearly identified with required human evidence.
5. Failures include root cause, failed layer, fix or blocker, and re-verification plan.
6. Security, namespace, evidence, action, audit, connector, and prompt-injection boundaries are explicitly checked where affected.
7. The final verdict is no stronger than the scenario evidence.

---

# 22. Initial Priority Recommendation

Because the full scenario suite is intentionally comprehensive, implementation should start by freezing a smaller v0.1 scenario subset.

Recommended first subset:

1. CS-PROD-001 through CS-PROD-005 — one product, first value, onboarding.
2. CS-ARCH-001 through CS-ARCH-009 — artifact/archive/evidence foundation.
3. CS-UND-001 through CS-UND-005 — search and no forced modeling.
4. CS-CLAIM-001 through CS-CLAIM-010 — conversation, brief, trust ladder, claim to mission/action.
5. CS-NS-001 through CS-NS-004 — owner-scoped namespace and promotion boundary.
6. CS-AUTO-001 through CS-AUTO-011 — basic mission/autopilot/action boundary.
7. CS-SEC-001 through CS-SEC-008 — local start, egress, policy, audit, injection, secrets.
8. CS-REG-001 through CS-REG-006 — prevent drift into chatbot/search/connector-only and prevent context leaks.

This subset gives the product a coherent first implementation standard while preserving the long-term direction.

---

# 23. Final Product Scenario Statement

CornerStone must pass these scenarios not because the product needs more process, but because the product is powerful enough to require proof.

The standard is:

> No durable claim without evidence.  
> No autonomous action without owner-scoped authority.  
> No cross-context use without explicit boundary.  
> No connector action outside ConnectorHub-mediated capability.  
> No memory without visibility and control.  
> No self-improvement without outcome evidence, scope, and rollback.  
> No release claim without scenario verification.

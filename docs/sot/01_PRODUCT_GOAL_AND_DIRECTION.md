# CornerStone Product Goal & Direction — Canonical Product SoT

**Replacement status:** Active product authority after product-goal reset, amended by the 2026-07-04 product-value-first reset (`docs/adr/ADR-0007-product-value-first-reset.md`).
**Source file:** `cornerstone_final_product_goal_direction (1).md` uploaded by JiYong / Tars.  
**Supersedes:** Product-goal, product-identity, target-user, and product-shape language in the older `project-sot.md` where conflicts exist.  
**Preserves:** Compatible technical defaults from the older SoT only through `03_TECHNICAL_ARCHITECTURE_DEFAULTS.md`.

---

# Part 0 — Active Product Spine and Claim Boundary (2026-07-04)

This part is binding and takes precedence over any present-tense capability language in the sections below it. The sections below remain the long-term direction; they are **direction, not current claims**.

## 0.1 The active spine

```text
Drop / Ask -> Evidence-backed Brief -> Decision -> Audit
```

- **Drop / Ask** — messy input in (paste today; one read-only source in VS6).
- **Evidence-backed Brief** — model-generated, citation-grounded synthesis of the user's own sources. This is the product's first value moment and the current build focus (VS5).
- **Decision** — the user-facing promotion of a brief finding the user stands behind (implemented as the claim record). Memory/Wiki candidates and Action Cards remain visible review drafts, outside the active value claim.
- **Audit** — the traceable history from original input to decision: checksums, evidence links, approvals, tamper-evident ledger.

External positioning language for the spine: **"briefs with receipts."** The phrase "Evidence-first Operational Intelligence Platform" is an internal category description, not user-facing copy.

## 0.2 Claim boundary (evidence-labeled)

- **Verified:** structural substrate — immutable artifacts, hashing/dedupe, trust states, evidence links, hash-chained audit, local runtime UI/CLI parity, deterministic Plane 1 harness (VS0–VS4, `STRUCTURAL_READY`).
- **Verified absent (open FAILs, 2026-07-04):** model-backed understanding does not exist yet; briefs echo input; Ask returns a canned deferral; templated outputs carry unearned trust labels. Recorded as CS-VAL FAIL baselines in `05_PRODUCT_VALUE_VERIFICATION_STANDARD.md`.
- **Specified (active milestone):** citation-grounded Brief/Ask with earned labels — `docs/scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_CONTRACT.md`.
- **Unverified until external humans provide evidence:** usefulness, comprehension, trust, habit, wedge, willingness to pay (VS5–VS7).

No document, report, or roadmap may claim above this boundary. Product-value claims require Plane 2 evidence per `05_PRODUCT_VALUE_VERIFICATION_STANDARD.md`.

## 0.3 Section status map

| Sections | Status |
|---|---|
| 1–8 (mission, identity, goal, north star, users, engines, first value/journey) | ACTIVE direction — read with Part 0 precedence; the first-value definition in 8.1 is the VS5 target |
| 9 (conversation-to-structure) | ACTIVE for conversation → brief → decision; **9.3 Mission/Decision Cards and 9.4 Knowledge Capsules are FUTURE-FACING (dormant)** |
| 10 (trust model) | ACTIVE — with the VS5 correction that trust labels must be earned (CS-VAL-006) |
| 11 (permanent wiki), 12.5 (product-learning namespace), 13 (memory sovereignty center) | FUTURE-FACING (dormant); simple memory view/edit/delete stays active |
| 12 (owner-scoped namespaces) | ACTIVE as data-model substrate; multi-tenant enforcement dormant (VS2) |
| 14 (autonomy), 15 (agent model), 16 (replaceable brain), 17 (LLM-as-judge), 18 (learning/self-improvement), 19 (AAR/scorecard), 20 (solution packs), 21 (extension ecosystem) | FUTURE-FACING (dormant) — no current capability exists; CLI stubs and fixtures only; reactivation by user-evidence pull (VS7 disposition, `VS7-DORM-001`) |
| 22–32 (source systems, safety, UX surfaces, principles, non-goals, tensions, verdict) | ACTIVE as principles, with Part 0 precedence on any capability claim |

One consequence stated plainly: section 16's "replaceable brain" principle stands, but *replaceable* must never again mean *omittable*. A brain is required for the product to exist; VS5 installs the first one (local Ollama `ornith:35b` for generation, `qwen3-embedding:0.6b` for embeddings, by default).

---

# CornerStone Final Product Goal & Direction Brief

**Date:** 2026-06-07  
**Owner:** JiYong / Tars  
**Status:** Product Direction v0.1 — synthesized from the intent-brief interview through Round 21  
**Canonical spelling:** Use **CornerStone** for the product/project. Treat **Conerstone / conerstone** as CornerStone unless intentionally referring to another name.

---

## 1. Executive Summary

CornerStone is an **Evidence-first Operational Intelligence Platform** that becomes the living knowledge and action foundation for a person, team, or organization.

Its purpose is not only to answer questions or search files. CornerStone should understand **who the user is**, **what the organization is**, **what they know**, **what they decide**, **how they act**, and **how they evolve over time**.

CornerStone integrates three existing realities into one coherent product experience:

1. **Cornerstone** — the current app surface and RAG/support workspace.
2. **KnowledgeBase** — the durable archive, evidence, memory, hashing, redaction, search, and knowledge substrate.
3. **ConnectorHub** — the connector, provider-access, credential, action, source-policy, delivery, and audit substrate.

The user should experience **one product**, not three products. Internally, the service keeps **clear modular boundaries** so that archive/evidence, product UX/intelligence, and connector/action safety remain separated and governable.

The product direction is:

> **CornerStone turns fragmented personal and organizational knowledge into a living, evidence-backed, autonomous intelligence system that can understand context, build durable memory, create claims, run missions, act through governed workflows, and improve from experience.**

---

## 2. Product Mission

### 2.1 One-sentence mission

**Transform fragmented knowledge into trusted understanding, autonomous missions, safe actions, and continuous learning.**

### 2.2 Expanded mission

CornerStone exists to become the user's and organization's **permanent operational memory**.

It should:

- preserve original information as durable evidence;
- understand and organize that information into useful context;
- synthesize briefs, claims, decisions, and operational maps;
- maintain a living wiki of the user and organization;
- support mission-oriented autonomous workflows;
- act through bounded, governed, auditable workflows;
- learn from outcomes, failures, corrections, and repeated experience;
- improve itself through evaluated, reversible, owner-scoped learning.

CornerStone should feel like a **living application**. It is not static software. It should continuously improve its understanding of the user, organization, missions, workflows, preferred outcomes, and product behavior.

---

## 3. Product Identity

### 3.1 Category

CornerStone is an:

> **Evidence-first Operational Intelligence Platform**

This is stronger than calling it a chatbot, RAG app, connector framework, automation platform, or personal wiki.

### 3.2 Product thesis

Most knowledge tools store information. Most AI assistants answer questions. Most automation tools execute tasks. CornerStone combines all three, but with evidence and governance as the foundation.

CornerStone should become the layer where:

- fragmented information becomes durable evidence;
- evidence becomes briefs and claims;
- claims become decisions;
- decisions become mission contracts;
- missions become autonomous workflows;
- workflows become actions;
- actions produce outcomes;
- outcomes become future memory and product learning.

### 3.3 What CornerStone is

CornerStone is:

- a personal and organizational knowledge operating platform;
- a living memory system;
- an operational intelligence system;
- an evidence and claim system;
- an autonomous mission workflow platform;
- an agent orchestration framework;
- a provider-neutral AI framework;
- a safe action/control plane over existing systems;
- a self-improving product with governed learning.

### 3.4 What CornerStone is not

CornerStone is not:

- only a chatbot;
- only a RAG/search app;
- only a file archive;
- only a connector hub;
- only a workflow automation tool;
- only a vertical logistics product;
- only a developer agent framework;
- a replacement for every source system;
- an opaque AI profile that secretly remembers or acts;
- an unbounded autonomous agent that writes to external systems without owner-scoped authority.

---

## 4. Core Product Goal

The core product goal is to make CornerStone the **living foundation of user and organization intelligence**.

For a personal user, CornerStone should become:

> **the user's permanent wiki of identity, context, knowledge, decisions, preferences, constraints, projects, memories, actions, and growth.**

For an organization, CornerStone should become:

> **the organization's permanent operational wiki of roles, policies, workflows, artifacts, decisions, claims, missions, actions, outcomes, culture, and evolving operating knowledge.**

For autonomous AI, CornerStone should become:

> **the mission-oriented system that uses this permanent context to act, evaluate, learn, and improve within explicit owner-scoped boundaries.**

---

## 5. Product North Star

### 5.1 North-star loop

CornerStone follows this product loop:

1. **Ingest** — capture any information as immutable artifacts or archived records.
2. **Understand** — extract, normalize, search, link, summarize, and map to context/ontology.
3. **Decide** — produce evidence-backed briefs, claims, recommendations, and mission plans.
4. **Act** — run workflows and actions inside owner-scoped mission contracts.
5. **Learn** — re-ingest outcomes, corrections, failures, approvals, rollbacks, and lessons.

### 5.2 North-star experience

The ideal user experience is:

> “CornerStone understands me and my organization, remembers what matters, shows evidence, runs bounded missions, explains what happened, and gets better from experience.”

### 5.3 North-star metric

The north-star metric is **Outcome Quality**.

CornerStone is improving if it creates better real outcomes over time:

- less repeated explanation;
- better evidence coverage;
- fewer contradictions;
- fewer stale memories;
- better workflow completion;
- fewer avoidable escalations;
- better decisions;
- safer actions;
- stronger continuity across work;
- higher owner acceptance of autonomous results.

Supporting metrics include:

- task completion rate;
- time-to-completion;
- rollback/error rate;
- escalation rate;
- evidence coverage;
- judge agreement/calibration;
- autonomy ratio;
- user/org correction rate;
- memory freshness and contradiction resolution.

---

## 6. Target Users and Adoption Motion

### 6.1 Product is general-purpose

CornerStone must remain **general-purpose**. It should not be restricted to logistics.

Logistics can remain a reference solution pack or demo domain, but the core product should apply to any personal, team, or organizational knowledge and operations context.

### 6.2 First adoption motion

The first adoption motion is:

> **Self-serve personal/professional wedge → team/org expansion**

CornerStone should start with individual users who need trusted understanding from fragmented information:

- founders;
- operators;
- analysts;
- researchers;
- managers;
- consultants;
- planners;
- project leads;
- personal knowledge power users.

Then it expands to teams and organizations when personal knowledge, briefs, claims, or missions are explicitly promoted into shared workspaces.

### 6.3 Personal-first meaning

“Personal” does not mean casual notes only.

Personal means:

- the user's permanent wiki;
- the user's living self/work identity graph;
- the user's preferences, constraints, style, decisions, projects, artifacts, relationships, missions, actions, and learning history;
- the user's evolving context.

### 6.4 Organization expansion

Organization use should emerge through:

- organization namespaces;
- workspace-first separation;
- explicit promotion from personal to shared context;
- approval and governance at the promotion/action boundary;
- organization memory and experience libraries;
- organization playbooks;
- organization-approved rules;
- bounded autonomous missions.

---

## 7. Product Shape: One Service, Three Invisible Engines

CornerStone should feel like **one coherent service**.

Internally, it has three invisible engines with clear boundaries.

### 7.1 Engine 1 — Product / Mission / Intelligence Engine

This is the user-facing CornerStone service.

It owns:

- Mission Control / Ops Inbox;
- conversation workbench;
- briefs;
- claims;
- mission contracts;
- agent orchestration;
- workspace UX;
- trust states;
- permanent wiki synthesis;
- mission after-action reviews;
- experience library;
- product learning proposals;
- user/operator controls.

This is where users experience CornerStone.

### 7.2 Engine 2 — Archive / Evidence / KnowledgeBase Engine

This is the durable memory and evidence substrate.

It owns:

- immutable artifacts;
- original source preservation;
- stable IDs/URIs;
- hashes/checksums;
- redaction;
- normalized docs/chunks;
- evidence bundles;
- provenance;
- search indexes;
- archive DB concepts;
- long-term memory source material;
- source-aware wiki synthesis.

This engine ensures that memory is not just agent memory. It keeps durable evidence as truth.

### 7.3 Engine 3 — Connector / Provider / Action Engine

This is the ConnectorHub substrate.

It owns:

- provider access;
- credentials;
- provider clients;
- source policy;
- projections;
- delivery;
- declared actions;
- external action execution;
- evidence metadata from provider interactions;
- temporary raw access;
- connector audit;
- retry/quarantine;
- verification;
- SDK bridge;
- control UI.

This engine ensures that external systems are accessed through governed capabilities, not arbitrary agent calls.

### 7.4 Boundary rule

CornerStone owns meaning, mission, UX, memory synthesis, claims, workflows, approvals, and product state.

ConnectorHub owns external-provider safety.

KnowledgeBase owns durable archive/evidence substrate.

Agents never directly bypass these boundaries.

---

## 8. First Value and Product Journey

### 8.1 First value moment

The first value moment is:

> **Personal messy input → evidence-backed brief → draft claim**

In the first 10 minutes, a new user should be able to:

1. drop a file, note, transcript, email export, web capture, or small folder;
2. get an evidence-backed brief;
3. see uncertainty and evidence gaps;
4. draft a claim;
5. save a capsule;
6. optionally open a mission.

### 8.2 Onboarding

The first-run experience should include an onboarding guide.

The guide should not start with heavy connector setup or organization admin setup. It should begin with personal value:

- drop or import messy input;
- ask a question;
- produce evidence brief;
- show evidence;
- save memory/capsule;
- optionally create a mission;
- explain workspace/context boundaries later.

### 8.3 Main product surface

The main surface is:

> **Ops Inbox / Mission Control**

This surface should show:

- current missions;
- incoming artifacts;
- evidence gaps;
- recommended briefs;
- pending claims;
- suggested actions;
- approvals;
- autopilot status;
- agent activity;
- unresolved conflicts;
- after-action reviews;
- experience-derived lesson proposals.

### 8.4 Core product journey

The core journey is:

> **Inbox → Brief → Claim → Action → Learn**

This is the spine of CornerStone.

The conversation workbench is the low-friction surface, but the durable product loop must convert useful conversation into structured outputs.

---

## 9. Conversation, Mission Cards, and Knowledge Capsules

### 9.1 Primary work object

The primary work object should be **conversation-focused**.

However, conversations should not be the final durable object. Conversation is the workbench where thinking starts.

Durable outputs include:

- **Mission / Decision Cards**;
- **Knowledge Capsules**;
- Claims;
- Actions;
- Playbooks;
- Memories;
- Experience records.

### 9.2 Conversation-to-structure model

CornerStone should use:

> **Conversation workbench → promoted durable outputs**

During conversation, CornerStone detects candidate:

- briefs;
- claims;
- evidence;
- decisions;
- actions;
- entities;
- relationships;
- memories;
- knowledge capsules;
- mission cards;
- playbook candidates.

Users can promote these into durable structure.

### 9.3 Mission / Decision Cards

A Mission or Decision Card should contain:

- goal;
- background;
- brief;
- evidence;
- claims;
- open questions;
- recommended actions;
- mission contract;
- agent plan;
- action history;
- approvals/escalations;
- outcome;
- after-action review;
- lessons learned;
- reusable playbook candidates.

### 9.4 Knowledge Capsules

A Knowledge Capsule is a reusable, evidence-aware unit of knowledge.

It can represent:

- topic summary;
- entity profile;
- project context;
- policy memory;
- decision rationale;
- reusable reference;
- user's preference/constraint;
- organization rule;
- domain concept;
- verified claim cluster.

Capsules can be Draft, Evidence-backed, or Approved.

---

## 10. Trust Model: Draft → Evidence-backed → Approved

CornerStone should use a clear trust ladder:

1. **Draft**  
   Exploratory, not fully supported. Useful for thinking, but not official.

2. **Evidence-backed**  
   Supported by artifacts, searches, source references, tool outputs, policy decisions, or action results.

3. **Approved**  
   Accepted by the owner, reviewer, team, organization, or policy-defined authority.

This applies to:

- memories;
- claims;
- wiki entries;
- knowledge capsules;
- mission plans;
- lessons;
- playbooks;
- organization rules;
- product-learning proposals;
- autonomous action permissions.

The product should allow fast exploration, but approval/publishing/action should require evidence and proper authority.

---

## 11. Permanent Wiki and Living Identity

### 11.1 What the permanent wiki means

CornerStone should maintain a living permanent wiki for:

- the individual user;
- the team;
- the organization;
- the workspace;
- the mission;
- the project/case;
- optionally, product-learning contexts.

The permanent wiki is not only a collection of pages. It is a living model of identity, context, history, responsibilities, preferences, decisions, actions, relationships, artifacts, and outcomes.

### 11.2 User permanent wiki

The user permanent wiki should contain:

- what the user knows;
- what the user cares about;
- what the user is working on;
- user preferences;
- constraints;
- recurring goals;
- projects;
- documents;
- relationships;
- skills;
- decisions;
- action history;
- accepted corrections;
- learned lessons;
- evolving identity and context.

### 11.3 Organization permanent wiki

The organization permanent wiki should contain:

- mission and purpose;
- org structure;
- roles and responsibilities;
- policies;
- workflows;
- customers/partners/vendors;
- assets;
- decisions;
- claims;
- risks;
- lessons;
- approved playbooks;
- operating style;
- action history;
- outcome history;
- culture and norms where relevant.

### 11.4 Two-layer truth model

CornerStone memory should use a two-layer truth model:

1. **Immutable evidence archive**  
   Preserved originals, source records, artifacts, hashes, provenance, and audit records.

2. **Living wiki synthesis**  
   Current understanding synthesized from evidence, conversation, corrections, decisions, and outcomes.

The living wiki can change. The underlying evidence should remain traceable.

### 11.5 Memory formation

Memory should be auto-synthesized, but not hidden.

CornerStone should synthesize memory from:

- conversations;
- artifacts;
- connected app context;
- decisions;
- actions;
- outcomes;
- corrections;
- approvals;
- rejected suggestions;
- experience trajectories.

Every important memory should be:

- source-aware;
- freshness-aware;
- correctable;
- deletable or forgettable where policy allows;
- demotable;
- promotable;
- scoped to an owner namespace;
- visible through a Memory Sovereignty Center.

---

## 12. Owner-Scoped Namespaces

### 12.1 Core boundary

The official top-level boundary for context ownership is:

> **Owner-scoped namespaces**

Every context item must answer:

1. Who owns this?
2. Which namespace does it belong to?
3. Who can read it?
4. Who can write, correct, demote, promote, or forget it?
5. Can it influence answers?
6. Can it influence actions?
7. What evidence supports it?
8. What trust state is it in?

### 12.2 Namespace types

CornerStone should support:

- Personal namespace;
- Organization namespace;
- Team namespace;
- Project namespace;
- Case namespace;
- Mission namespace;
- Customer/account namespace;
- Solution pack namespace;
- Product Learning namespace.

### 12.3 Workspace-first separation

The user-facing context model is:

> **Workspace-first separation**

Users enter or switch workspaces such as:

- Personal;
- Organization;
- Team;
- Project;
- Mission;
- Case.

Everything inside a workspace uses that workspace's context unless explicitly referenced or promoted.

### 12.4 Cross-namespace promotion

Personal context must not silently become organization truth.

Movement across namespaces requires:

- explicit promote/share/copy/reference operation;
- provenance;
- owner change or owner reference;
- trust-state assignment;
- permissions;
- audit trail;
- evidence bundle;
- conflict handling.

This boundary is critical.

### 12.5 Product Learning namespace

A Product Learning namespace may exist, but it must be separate.

It can store:

- friction signals;
- benchmark results;
- failed workflows;
- research summaries;
- feedback;
- common evidence gaps;
- improvement proposals;
- product experiments.

It must not become hidden truth over user or organization memory.

---

## 13. Memory Sovereignty Center

CornerStone needs a Memory Sovereignty Center.

Users and organizations should be able to inspect and manage:

- what CornerStone remembers;
- what source created a memory;
- when it was last refreshed;
- what trust state it has;
- who owns it;
- where it is used;
- whether it can influence answers;
- whether it can influence actions;
- whether it is personal, shared, organization, mission, or product-learning context;
- how to correct it;
- how to demote/promote it;
- how to forget/delete/archive it;
- how to rollback memory changes;
- how to mark sources as relevant/not relevant.

Memory sovereignty is required because CornerStone claims to understand who the user is and what the organization is.

---

## 14. Autonomous AI Direction

> **Status: FUTURE-FACING (dormant).** No autonomy capability exists. Direction only; see Part 0.3 and ADR-0007. Reactivation requires user-evidence pull (`VS7-DORM-001`).

### 14.1 Autonomy product stance

CornerStone should become autonomous AI, not only a passive copilot.

However, autonomy must be:

- owner-scoped;
- workspace-scoped;
- mission-scoped;
- contract-bound;
- policy-aware;
- evidence-aware;
- logged;
- evaluated;
- reversible;
- auditable.

The product direction is not unbounded agent autonomy. It is:

> **Pre-approved autonomy inside clear owner-scoped workspace and mission boundaries.**

### 14.2 Simple governance

Simple governance means:

1. **Workspace Autopilot Modes**
2. **Mission Goal Contracts**
3. **Hidden risk policy matrix behind the scenes**

### 14.3 Workspace Autopilot Modes

Each workspace can have a simple visible mode:

- **Manual** — CornerStone answers, searches, drafts, and organizes only.
- **Assist** — CornerStone recommends actions and prepares work.
- **Autopilot** — CornerStone executes allowed actions inside the workspace/mission contract.
- **Locked** — CornerStone can use memory/search but cannot act.

### 14.4 Mission Goal Contracts

A mission contract defines:

- goal;
- scope;
- allowed actions;
- forbidden actions;
- data/memory scope;
- connector capabilities;
- budget/cost constraints;
- success criteria;
- stop conditions;
- escalation rules;
- review cadence;
- rollback expectations.

The user should be able to state the goal naturally, and CornerStone should turn it into a simple editable contract.

Templates/playbooks should accelerate repeated missions.

### 14.5 Autopilot activation

Autopilot should follow a **progressive autonomy ramp**:

1. Start in Manual or Assist.
2. Prove reliability through briefs, internal tasks, suggestions, and completed playbooks.
3. Recommend Autopilot when readiness is visible.
4. Owner grants authority.
5. Autopilot runs within the mission contract.
6. Results are reviewed through after-action review and scorecard.

### 14.6 Autopilot authority

CornerStone should support **bounded execution autonomy**.

It can autonomously execute allowed actions inside the active workspace and Mission Goal Contract.

It should escalate:

- high-risk actions;
- destructive actions;
- sensitive data exposure;
- cross-namespace movement;
- external writeback outside contract;
- unresolved high-risk disagreement;
- policy-denied or ambiguous actions.

---

## 15. Agent Model

> **Status: FUTURE-FACING (dormant).** No orchestrator or specialist agents exist; CLI stubs and fixtures only. Direction only; see Part 0.3 and ADR-0007.

### 15.1 User-facing model

The agent model is:

> **Orchestrator-led mission-specific agent team**

Users primarily work with the **Orchestrator Agent**.

The Orchestrator:

- understands the mission;
- creates the plan;
- reads the mission contract;
- selects specialist agents;
- delegates work;
- combines results;
- asks for missing authority;
- manages conflicts;
- prepares actions;
- produces the after-action review;
- manages learning proposals.

### 15.2 Specialist agents

Specialist agents may include:

- Evidence Agent;
- Memory Agent;
- Archive Agent;
- Workflow Agent;
- Connector Agent;
- Policy Agent;
- Judge Agent;
- Playbook Agent;
- Research Agent;
- Ontology Agent;
- Risk Agent;
- Product Learning Agent.

Users should not need to manually manage all agents, but specialist work should be visible when helpful.

### 15.3 Agent Role Contracts

Each agent should have an Agent Role Contract.

A contract includes:

- purpose;
- responsibilities;
- allowed tools;
- forbidden actions;
- memory scope;
- evidence requirements;
- escalation rules;
- model/provider policy;
- judge/evaluation rubric;
- audit expectations;
- connector requirements;
- action authority.

### 15.4 Agent contract visibility

Daily users see **role cards**:

- agent purpose;
- what it can do;
- what it cannot do;
- current workspace/mission scope;
- escalation behavior.

Operators/admins can inspect full contracts.

### 15.5 Accountability

The accountability model is:

> **Namespace-owner accountability with audit proof**

The owner of the namespace/workspace/mission is accountable for granting autonomy.

CornerStone is accountable for showing:

- who granted authority;
- what was allowed;
- what happened;
- what evidence was used;
- what policy decisions were made;
- what model/brain was used;
- what changed;
- what failed;
- how to correct or roll back.

---

## 16. Replaceable Brain and Provider-Neutral AI Framework

> **Status: FUTURE-FACING (dormant) as a routing/ensemble framework.** The principle stands, but replaceable never means omittable: the first required brain is installed by VS5 (local Ollama `ornith:35b` + `qwen3-embedding:0.6b`). Multi-brain routing, ensembles, and the Brain Performance Ledger stay dormant. See Part 0.3 and ADR-0007.

### 16.1 Core principle

CornerStone AI is the **framework**.

The model is the **replaceable brain**.

GPT, Claude, Gemini, local models, and future providers can be swapped behind the CornerStone framework.

### 16.2 CornerStone-owned durable value

CornerStone owns:

- namespaces;
- permanent wiki;
- evidence archive;
- ontology;
- mission contracts;
- agent contracts;
- workflow/action boundaries;
- policies;
- audit;
- experience library;
- judge records;
- memory synthesis;
- promotion ladder;
- connector boundary;
- product learning;
- user/org identity continuity.

The LLM provider only supplies inference.

### 16.3 Provider-neutral routing

CornerStone should use a **policy-aware model router**.

Routing should consider:

- workspace policy;
- data sensitivity;
- mission type;
- cost;
- latency;
- model capability;
- historical performance;
- tool-use reliability;
- provider availability;
- user/org preferences;
- data residency;
- local/on-prem requirements.

Users may override within policy.

### 16.4 Multi-brain ensemble

CornerStone should normally use one policy-routed brain.

It should use multi-brain ensemble for:

- high-risk missions;
- high-value missions;
- ambiguous outcomes;
- externally impactful decisions;
- safety-sensitive tasks;
- low judge confidence;
- serious disagreement;
- critical memory/rule promotion.

A multi-brain ensemble may use multiple models for planning, critique, judging, verification, or action review.

### 16.5 Brain Performance Ledger

CornerStone should maintain a Brain Performance Ledger.

It records:

- provider/model;
- task type;
- workspace policy;
- data sensitivity;
- cost;
- latency;
- judge quality;
- tool-use reliability;
- hallucination/grounding problems;
- owner corrections;
- objective outcomes;
- mission success;
- escalation/rollback events.

The ledger should be:

- namespace-local first;
- inspectable;
- separate from user/org truth;
- optionally aggregated later with explicit opt-in, anonymization, policy review, and governance.

A static model capability registry can provide safe initial defaults until local experience accumulates.

---

## 17. LLM-as-Judge and Evaluation

### 17.1 Judge stance

CornerStone should use:

> **LLM-first evaluator, evidence-anchored authority**

LLM-as-judge should be primary for ambiguous or subjective outcome evaluation, but it should not become unquestionable truth.

### 17.2 Judge responsibilities

LLM judges can:

- score;
- critique;
- compare;
- detect evidence gaps;
- detect contradictions;
- evaluate mission quality;
- review after-action outcomes;
- suggest lessons;
- compare model outputs;
- recommend promotion/demotion.

### 17.3 Judge limits

Objective system outcomes override judge opinion.

Owner acceptance grounds final success.

High-risk actions require policy/human gates.

Judge outputs should be treated as evaluation artifacts with:

- rubric;
- model identity;
- confidence;
- evidence;
- dissent;
- bias/uncertainty indicators;
- audit trail.

### 17.4 Disagreement handling

When agents, judges, or brains disagree, CornerStone should use:

> **Evidence-weighted adjudication with dissent preserved**

The system should consider:

- evidence quality;
- policy constraints;
- mission goals;
- prior brain performance;
- objective outcomes;
- judge rubric;
- user/org preferences;
- action risk.

It should recommend a path, but preserve dissent and uncertainty.

High-risk unresolved disagreement escalates to humans.

---

## 18. Experience, Learning, and Self-Improvement

> **Status: FUTURE-FACING (dormant).** No learning capability exists beyond record stubs. Direction only; see Part 0.3 and ADR-0007.

### 18.1 Learning model

CornerStone should learn from:

> **Full Mission Trajectory Ledger + Experience Library + staged lesson/rule promotion**

### 18.2 Full Mission Trajectory Ledger

Each mission trajectory should record:

- goal;
- workspace;
- owner;
- mission contract;
- plan;
- evidence;
- memory used;
- agents involved;
- model/brain used;
- tool calls;
- connector actions;
- policy decisions;
- approvals;
- rejections;
- escalations;
- exceptions;
- outcomes;
- costs/time;
- corrections;
- rollback events;
- judge reviews;
- after-action review;
- extracted observations;
- lessons;
- reusable playbook candidates.

### 18.3 Experience Library

Each workspace should have an Experience Library.

It contains:

- past missions;
- trajectories;
- after-action reviews;
- outcomes;
- judge assessments;
- failures;
- recoveries;
- lessons;
- playbook candidates;
- similar-case references;
- reusable evidence patterns.

When CornerStone recommends something, it can reference relevant past experiences.

### 18.4 Experience-to-memory direction

Store all trajectories as a retrievable reference corpus.

Then selectively convert useful parts into:

- observations;
- candidate lessons;
- workspace memories;
- mission playbooks;
- organization-approved rules;
- solution-pack proposals;
- product-learning proposals.

### 18.5 Selective curation

Experience should become durable memory or rule only through selective curation.

The curation should include:

- causal attribution;
- scope;
- applicability conditions;
- confidence;
- provenance;
- freshness;
- rollback path;
- owner approval where needed.

### 18.6 Promotion ladder

The promotion ladder is:

> Trajectory → Observation → Candidate Lesson → Workspace Memory → Mission Playbook → Organization-approved Rule → Solution-pack/Product-learning Proposal

This prevents local experience from becoming global truth too quickly.

### 18.7 Self-improvement boundary

“Living” means self-improving.

Self-improvement should be allowed as:

- namespace-local adaptation when the owner has enabled it;
- personal or organization workflow optimization;
- better memory ranking;
- improved brief style;
- recurring workflow optimization;
- onboarding personalization;
- preference learning.

Product-wide changes require:

- benchmark/evaluation;
- review;
- versioning;
- rollout plan;
- monitoring;
- rollback.

---

## 19. After-Action Review and Autonomy Scorecard

Every completed autonomous mission should produce a:

> **Mission After-Action Review + Autonomy Scorecard**

It should include:

- mission goal;
- contract scope;
- actions taken;
- evidence used;
- memory used;
- agents involved;
- models/brains used;
- objective outcomes;
- owner acceptance/rejection;
- LLM-judge assessment;
- errors;
- escalations;
- policy denials;
- conflicts/disagreements;
- lessons learned;
- candidate memories;
- candidate playbooks;
- rollback/correction options.

Full audit export should be available for compliance, debugging, and governance.

---

## 20. Solution Packs and Playbooks

### 20.1 Universal core + optional solution packs

CornerStone stays general-purpose through a universal core.

Solution packs provide concrete starting points without changing the product identity.

Possible packs:

- logistics;
- compliance;
- legal operations;
- research;
- sales operations;
- customer operations;
- project operations;
- incident review;
- personal knowledge;
- executive planning;
- engineering operations.

### 20.2 Experience-derived playbooks

Solution packs should evolve as:

> **Experience-derived playbooks with approval**

CornerStone detects repeated successful mission trajectories, failure recoveries, evidence patterns, and workflow improvements.

Then it proposes updates to:

- personal playbooks;
- workspace playbooks;
- organization playbooks;
- solution packs;
- product learning.

Owners approve before reusable playbooks become authoritative.

---

## 21. Extension Ecosystem

### 21.1 Extension unit

The top-level advanced extension unit is:

> **Agent Pack**

An Agent Pack includes:

- role contract;
- role card;
- allowed capabilities;
- tool/connector requirements;
- memory scope;
- model policy;
- judge rubric;
- playbooks;
- after-action review template;
- evaluation expectations;
- risk label;
- version history.

Skill/Tool Packs and Playbook Packs exist as internal/components.

### 21.2 Trusted registry

The extension ecosystem starts with:

> **Curated/private trusted registry first**

Supported extension sources:

- first-party CornerStone packs;
- organization-private packs;
- curated/certified third-party packs.

Public marketplace growth may come later.

### 21.3 Extension trust model

Agent Packs should include:

- risk labels;
- capability disclosures;
- version history;
- evaluation records;
- permission requirements;
- owner approval;
- rollback;
- workspace policy checks;
- supply-chain checks where applicable.

### 21.4 Connector boundary

Agent Packs must not directly own external provider access or credentials.

They declare requirements.

ConnectorHub-mediated capabilities provide access.

Agents use external systems through:

- ConnectorHub;
- CornerStone workflows;
- Mission Goal Contracts;
- policy;
- audit;
- declared actions.

---

## 22. Relationship to Existing Source Systems

CornerStone should not force users to replace all existing systems.

Its relationship to existing systems is:

> **Intelligence and action control plane over existing systems**

Existing systems remain where records originate and where many operational actions occur.

CornerStone:

- connects them;
- preserves evidence;
- understands context;
- produces claims;
- coordinates missions;
- triggers allowed workflows;
- audits actions;
- learns from outcomes.

External writeback is gated through policy, dry-run, approval/autopilot authority, and audit.

---

## 23. Safety and Governance Direction

CornerStone's autonomy must remain trustworthy.

The product should preserve these defaults:

- no unbounded external writeback;
- no destructive action without explicit authority;
- egress control;
- no cross-namespace/cross-tenant mixing;
- evidence/audit traceability;
- prompt-injection defenses;
- connector-mediated external access;
- workflow/action layer for mutations;
- rollback/compensation where possible;
- clear owner accountability.

The product direction allows stronger autonomy than the original conservative copilot stance, but only through:

- workspace modes;
- mission contracts;
- risk/value classification;
- owner-scoped namespaces;
- action/workflow mediation;
- audit;
- after-action review;
- experience learning;
- rollback.

---

## 24. Product UX Surfaces

CornerStone should eventually include these primary surfaces.

### 24.1 Mission Control / Ops Inbox

The main home surface.

Shows:

- missions;
- brief recommendations;
- pending claims;
- action cards;
- evidence gaps;
- approvals;
- autopilot mode;
- escalations;
- after-action reviews;
- lesson proposals;
- agent status.

### 24.2 Conversation Workbench

Low-friction working surface.

Supports:

- ask/search;
- messy input interpretation;
- evidence-backed briefs;
- memory-aware dialogue;
- candidate claims;
- action suggestions;
- promote-to-card/capsule;
- workspace-scoped context.

### 24.3 Permanent Wiki

Living personal or organization wiki.

Shows:

- memories;
- capsules;
- entities;
- relationships;
- current context;
- decisions;
- roles;
- preferences;
- constraints;
- history;
- freshness;
- trust states.

### 24.4 Memory Sovereignty Center

Controls memory ownership, source, usage, influence, correction, deletion, and promotion.

### 24.5 Mission / Decision Cards

Durable mission and decision objects.

### 24.6 Claim Builder

Turns evidence into claims and recommendations.

### 24.7 Action Studio

Shows action cards with:

- diff;
- impact;
- evidence;
- policy decision;
- approval/autopilot authority;
- execution;
- rollback;
- audit.

### 24.8 Experience Library

Stores and surfaces mission trajectories, lessons, outcomes, and reusable playbooks.

### 24.9 Agent Control / Role Cards

Shows Orchestrator and specialist agents, their roles, scopes, and authority.

### 24.10 Admin / Operator Surface

Controls:

- namespaces;
- workspace policies;
- connectors;
- agent packs;
- model routing;
- brain performance ledger;
- audit export;
- governance settings.

---

## 25. Detailed Product Principles

### Principle 1 — Evidence before authority

CornerStone may draft freely, but approved claims/actions require evidence.

### Principle 2 — Conversation starts, structure persists

Users start in conversation, but important work is promoted into durable objects.

### Principle 3 — Memory is living but sovereign

CornerStone can synthesize memory automatically, but users/organizations must be able to inspect and control it.

### Principle 4 — Context belongs to owners

Every piece of context belongs to an owner-scoped namespace.

### Principle 5 — Autonomy is bounded by mission

Autonomous work happens inside a mission goal contract, not vague global permission.

### Principle 6 — Agents are governed roles, not random prompts

Each agent has a role contract, authority boundary, evidence expectation, and evaluation rubric.

### Principle 7 — The brain is replaceable

The durable product is CornerStone's framework, not any single model provider.

### Principle 8 — Learning comes from experience

The system improves from trajectories, outcomes, judge reviews, corrections, and accepted lessons.

### Principle 9 — Local learning before global learning

Namespace-local adaptation is default. Product-wide learning requires opt-in/review.

### Principle 10 — External systems are mediated

Agents do not directly grab credentials or write to source systems. ConnectorHub mediates access.

---

## 26. Product Non-Goals

CornerStone should avoid these directions:

1. **Chatbot-only product**  
   Conversation is a surface, not the product.

2. **RAG-only product**  
   Retrieval is necessary, but insufficient.

3. **Connector-only product**  
   ConnectorHub is an engine, not the whole product.

4. **Automation-first without memory/evidence**  
   Automation without durable context is not CornerStone.

5. **Fully autonomous black box**  
   Autonomy must be explainable, owner-scoped, and reviewable.

6. **One global memory graph**  
   This risks ownership confusion and data leakage.

7. **Manual wiki only**  
   The product must synthesize and learn, not just store pages.

8. **Model-provider lock-in**  
   CornerStone should not become dependent on one LLM vendor.

9. **Untrusted open extension marketplace first**  
   Trusted/private/curated registry comes first.

10. **Forced ontology modeling before value**  
   Users should get value before they model everything.

---

## 27. Key Tensions and Resolutions

### Tension 1 — Personal-first vs organization-grade governance

Resolution: personal namespaces first; organization namespaces through explicit promotion; governance strengthens at promotion/action boundary.

### Tension 2 — Conversation simplicity vs durable structure

Resolution: conversation workbench first; promote useful outputs into cards, capsules, claims, actions, and playbooks.

### Tension 3 — Autonomous AI vs safety

Resolution: Autopilot Modes + Mission Goal Contracts + hidden risk policy + after-action review.

### Tension 4 — Living memory vs privacy

Resolution: Memory Sovereignty Center + source-aware memory + owner-scoped namespaces + correction/forget/rollback.

### Tension 5 — LLM-as-judge primary vs reliability

Resolution: LLM-first evaluator, but evidence-anchored; objective outcomes and owner acceptance remain grounding constraints.

### Tension 6 — Self-improving product vs uncontrolled mutation

Resolution: namespace-local adaptation allowed; product-wide changes require evaluation, approval, versioning, monitoring, rollback.

### Tension 7 — Agent ecosystem vs trust

Resolution: Agent Packs in curated/private trusted registry, ConnectorHub-mediated external access, certification/versioning later.

---

## 28. Research-Informed Direction

The interview direction aligns with recent work in long-term AI memory, agent memory, trajectory learning, LLM-as-judge, and model-provider abstraction.

The practical research takeaways for CornerStone are:

1. Memory should not be just stored text. It should be a write/manage/read loop connected to perception and action.
2. Persistent memory needs governance because dynamic memory can drift, leak, become stale, or encode poisoned lessons.
3. Autonomous systems should learn from full trajectories, not only final success/failure.
4. LLM-as-judge is useful as a scalable evaluator, but judge outputs require calibration, bias handling, and evidence grounding.
5. Provider/model routing should be abstracted so CornerStone owns the durable framework while model providers remain replaceable.

---

## 29. Final Product Direction Statement

CornerStone should become a **living, evidence-first, autonomous operational intelligence platform** for personal and organizational knowledge.

It starts as a personal permanent wiki and evidence-backed brief workspace.

It grows into workspace-based missions, claims, actions, and organization memory.

It becomes autonomous through Orchestrator-led mission agents operating inside workspace modes and mission contracts.

It improves from full mission trajectories, experience libraries, judge reviews, outcomes, and staged lesson promotion.

It remains trustworthy through owner-scoped namespaces, memory sovereignty, evidence bundles, connector-mediated action, audit, rollback, and provider-neutral brain routing.

The final direction can be summarized as:

> **CornerStone is the living knowledge foundation and autonomous mission system for a person or organization. It remembers with evidence, understands with context, acts with governed autonomy, and improves through experience.**

---

## 30. Implementation Implications Later

This brief is not an implementation plan. However, it implies that later implementation should preserve these product contracts:

1. Keep one coherent product experience.
2. Preserve three engine boundaries.
3. Start with evidence-backed brief from messy input.
4. Build persistent owner-scoped memory/wikis.
5. Use workspace-first context separation.
6. Implement mission/autopilot concepts only with clear authority.
7. Make all external actions go through ConnectorHub-mediated workflows.
8. Keep the LLM provider replaceable.
9. Store mission trajectories and after-action reviews.
10. Implement staged promotion for lessons/rules/playbooks.
11. Keep product-wide self-improvement reviewed and reversible.

---

## 31. Optional / Unresolved Items From Interview Round 21

Questions 61–63 were introduced but not answered. They are not required to finalize the core product direction, but they remain useful for future extension governance.

### 31.1 Agent Pack activation and permission grant

Open question: when an Agent Pack is installed, should authority be granted globally, per workspace, or per mission?

Recommended future answer: install globally, activate per workspace/mission with explicit capability grants.

### 31.2 Agent Pack certification

Open question: what must an Agent Pack prove before it is trusted for autonomous missions?

Recommended future answer: scenario-certified Agent Pack with evaluation card, safety checks, connector/action boundary checks, outcome history, versioning, and rollback.

### 31.3 Agent Pack rollout

Open question: how should Agent Pack updates roll out once workspaces depend on them?

Recommended future answer: pinned versions with staged rollout, evaluation gates, shadow/canary modes, and rollback.

---

## 32. Final Verdict

The product-direction interview can be considered complete.

The product goal is now clear enough to become a final product intent brief and later feed implementation scenario contracts.

**Verdict:** Product direction is ready to freeze as v0.1.

**Confidence:** 0.88

The remaining uncertainty is not about the core product identity. The remaining uncertainty is about later implementation sequencing and extension-governance details.

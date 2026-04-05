# Cornerstone Project SoT

**Status:** Approved  
**Type:** Product Source of Truth  
**Scope:** Entire Cornerstone product  
**Last Updated:** 2026-04-05

---

## 1. Product identity

### Final definition
**Cornerstone connects information scattered across documents, conversations, and systems, continuously reconstructs and maintains organizational context through eventual sync with source systems, and enables humans and AI to understand and reason from the same context and evidence. It is the shared organizational context layer for humans and AI.**

### Short positioning
- **The shared organizational context layer for humans and AI.**
- **The semantic system of record for organizational meaning and decision context.**

### Internal framing
Cornerstone is best understood internally as a **semantic system of record** for organizational meaning, relationships, and decision context.

### External framing
Cornerstone should be described externally as a **shared organizational context layer** rather than as a search engine, a vector database, or an AI assistant.

---

## 2. The real-world problem

Important organizational information is scattered across tools such as Notion, Slack, Google Docs, wikis, ticketing systems, internal apps, and human memory.

As a result:

- AI does not reliably know the organization’s actual terminology, prior decisions, constraints, or outcomes.
- Humans also lose context when the organization’s memory is fragmented across systems, documents, and conversations.
- Teams end up working from fragments instead of from a coherent organizational memory.
- Decision quality, explanation quality, and automation quality become unstable.

Cornerstone does **not** primarily solve a search problem.  
It solves a deeper problem: **loss of organizational context**.

---

## 3. Purpose

**The purpose of Cornerstone is to continuously reconstruct, organize, and connect organizational context from scattered documents, conversations, and systems so that humans and AI can understand and make decisions from the same context and evidence.**

---

## 4. Product goal

**The goal of Cornerstone is to continuously build and update an organization’s terms, relationships, decision background, flows, and evidence by syncing with multiple data sources, and to make that context available for consistent retrieval, generation, review, and approval through API, MCP, and human-facing interfaces.**

---

## 5. Primary and secondary users

### Primary users
- Knowledge operators
- Organizational administrators
- Domain owners
- Product/platform owners responsible for official meaning and process integrity

Primary users connect sources, monitor sync health, review proposed knowledge, resolve conflicts, and approve official meaning.

### Secondary users
- AI and agent developers
- AI systems such as Claude, Gemini, Codex, and internal models
- General organization members

Secondary users query, consume, and apply Cornerstone’s official context in workflows, decisions, automation, and answers.

---

## 6. Core outputs

Cornerstone’s core value is expressed through four outputs.

### 6.1 Glossary
A canonical glossary of official organizational terms and concepts, including:
- definitions
- aliases
- deprecated terms
- usage notes
- owners
- evidence
- linked decisions

### 6.2 Ontology + context graph
A graph of concepts, systems, policies, workflows, roles, metrics, and events, including:
- semantic relationships
- operational dependencies
- flow relationships
- conflict and alignment relationships
- context carried by linked decisions

### 6.3 Decision context
A structured, reviewable representation of why important definitions, policies, structures, or implementation choices were made.

### 6.4 Source-backed answers
Responses that return not just text, but the linked concept, relation, decision, and evidence structure behind the answer.

---

## 7. Decision context as a first-class concept

Cornerstone must preserve not only **what is true here**, but also **why it is true here**.

### Official definition
**Decision context is the official, source-backed context unit that preserves what was decided, why it was decided, what constraints shaped it, what evidence supported it, and what changed as a result.**

### Modeling rule
In product language, this is called **Decision Context**.  
In the system model, the first-class entity is called **`DecisionRecord`**.

Cornerstone is incomplete if it only stores definitions and relations.  
It must also preserve the rationale lineage behind them.

---

## 8. Product principles

### 8.1 Source-first
Official knowledge should be grounded in identifiable sources and evidence whenever possible.

### 8.2 Eventually correct over instantly updated
Real-time behavior is valuable but secondary. Eventual correctness, recoverability, and provenance come first.

### 8.3 Reviewable officialization
AI may propose drafts, but official concepts, relations, and decisions must remain reviewable.

### 8.4 Meaning over raw text
The product prioritizes concepts, relationships, decision background, and evidence over raw text storage.

### 8.5 Consumer-agnostic
Humans, Claude, Gemini, Codex, and internal AI systems should all be able to consume the same truth layer.

### 8.6 Layered truth
Source systems remain the source of raw operational truth. Cornerstone owns the official layer of meaning, relationships, and decision context on top.

---

## 9. What Cornerstone is not

Cornerstone is **not** any of the following:

- a replacement for Notion, Slack, Google Docs, or other source systems
- a generic enterprise search engine
- a generic vector store or RAG database
- a single-model AI assistant product
- an unreviewed automatic knowledge generator
- a strict real-time sync product

---

## 10. Non-goals

The following are explicitly out of scope for the product definition.

1. Replacing source systems as the system of record for raw documents
2. Guaranteeing strong real-time synchronization for all providers
3. Treating every message or meeting as an official decision record
4. Treating free-form AI output as official truth without reviewability
5. Defining the product primarily as chat UX

---

## 11. Final decisions fixed by this SoT

The following points are considered decided unless this SoT is explicitly revised.

1. **Cornerstone is a context layer, not a source-system replacement.**
2. **Humans and AI both depend on the same official context layer.**
3. **AI is a consumer and worker, not the product identity.**
4. **Glossary and graph are projections over curated entities, not separate storage objects.**
5. **Decision context is a first-class requirement, not an optional feature.**
6. **Official knowledge must be source-backed and reviewable.**
7. **Eventual sync is required; strict real-time behavior is not.**
8. **Delivery is spec-driven.**

---

## 12. Change control

Changes to any of the following require:
- a `DecisionRecord`
- an update to this SoT
- linked updates to affected specs

Covered changes:
- product identity
- purpose
- non-goals
- first-class entities
- the meaning of “official”
- reviewability requirements
- provenance requirements
- the role of AI in the product

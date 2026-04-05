# Product Spec (v1)

**Status:** Approved  
**Type:** Product Specification  
**Scope:** Cornerstone v1  
**Last Updated:** 2026-04-05

---

## 1. Purpose of this document

This document defines **what Cornerstone v1 must provide**.

- `sot/WHY_AND_GOALS.md` explains why the product exists.
- `sot/PROJECT_SOT.md` defines product identity and non-negotiable principles.
- this spec turns that identity into a v1 product scope and requirement set

---

## 2. v1 product promise

**Cornerstone v1 connects scattered organizational artifacts, continuously syncs them from source systems, curates official concepts, relations, and decision context with evidence, and serves that context to both humans and AI in a structured, reviewable form.**

---

## 3. Consumers

### Primary
- knowledge operators
- domain owners
- organizational administrators

### Secondary
- AI and agent developers
- AI systems consuming API or MCP
- general organization members who need official context

---

## 4. Core capabilities

### 4.1 Connectors and sync
Cornerstone v1 must provide:
- easy connection to multiple source systems
- persistent storage of normalized source artifacts
- support for polling, scheduling, webhook, and event-driven sync patterns
- eventual consistency rather than strict real-time guarantees
- sync status tracking, failure visibility, and recoverability
- deep links back to source systems
- source-bounded search or source-native fallback where useful

**Principle:** eventual correctness and provenance matter more than immediate reflection.

---

### 4.2 Knowledge extraction and curation
Cornerstone v1 must be able to create or accept:
- `Concept` drafts
- `ConceptRelation` drafts
- `DecisionRecord` drafts
- linked `EvidenceFragment` references
- human review and approval workflows

**Principle:** AI may draft, suggest, and summarize, but officialization must remain reviewable.

---

### 4.3 Glossary view
Cornerstone v1 must expose a glossary view over official and draft concepts.

Each glossary item should include at least:
- canonical name
- aliases
- definition
- status
- owner
- evidence or source linkage
- related concepts
- linked decisions

---

### 4.4 Ontology + context graph view
Cornerstone v1 must expose more than isolated glossary entries.

The graph view should support:
- concept-to-concept relationships
- links across systems, policies, workflows, roles, and metrics
- relation descriptions and relation status
- graph navigation into linked decision context

---

### 4.5 Decision context view
Cornerstone v1 must support structured decision context.

Each decision view should include at least:
- problem or trigger
- decision
- rationale
- constraints
- evidence
- impact
- status
- owner or approver
- relationships to concepts and relations
- supersedes or superseded-by links where relevant

---

### 4.6 API / MCP serving
Cornerstone v1 must support machine-consumable access for AI and automation.

The serving surface should support:
- concept retrieval
- relation retrieval
- decision retrieval
- evidence retrieval
- source-backed answer assembly
- structured output for both humans and machines

**Principle:** the product should not be locked to a specific model vendor or transport.

---

### 4.7 Human review and governance
Cornerstone v1 must include a human-facing review surface or workflow that allows authorized users to:
- review drafts
- approve drafts
- reject drafts
- deprecate outdated knowledge
- observe provenance and change lineage

---

## 5. v1 non-goals

Cornerstone v1 does **not** aim to provide:

1. source-system replacement
2. guaranteed real-time sync for every provider
3. generic enterprise search as the core product value
4. complete semantic parsing of every artifact
5. fully autonomous officialization without human review
6. one fixed chat interface as the primary product identity

---

## 6. System-wide invariants

The following rules apply across all v1 capabilities.

1. Official knowledge must be source-backed or linked to an approving decision.
2. Concepts, relations, and decisions must carry status.
3. Decision context is first-class and cannot be omitted from the model.
4. Source systems remain source systems; Cornerstone stores an official context layer on top.
5. Humans and AI consume the same truth layer.
6. Delivery remains spec-driven.

---

## 7. Key v1 workflows

### 7.1 Connect and sync
An operator connects a source system, configures sync behavior, and Cornerstone begins persisting normalized artifacts while tracking health and provenance.

### 7.2 Curate official knowledge
A user or AI worker proposes concept, relation, or decision drafts. Reviewers inspect linked evidence, revise where needed, and approve official knowledge.

### 7.3 Retrieve official context
A human or AI consumer queries Cornerstone and receives structured concepts, relations, decisions, and evidence rather than an ungrounded free-form answer.

---

## 8. v1 acceptance criteria

Cornerstone v1 is acceptable when all of the following are true.

- at least one production-ready connector flow exists with observable eventual sync behavior
- source artifacts can be persisted and traced back to origin
- concept, relation, and decision drafts can be created and reviewed
- at least one official glossary view exists
- at least one graph-oriented context view exists
- decision context can be represented, linked, and retrieved
- at least one machine-consumable contract exists for structured retrieval
- at least one human review workflow exists for officialization
- status and provenance are visible in retrieval and review flows

---

## 9. Recommended v1 implementation direction

- use relational storage as the initial system of record
- treat graph traversal as a projection
- keep provider adapters replaceable
- keep serving interfaces consumer-agnostic
- start with bounded concept types and bounded relation predicates

These are recommendations, not product identity. The identity remains in the SoT.

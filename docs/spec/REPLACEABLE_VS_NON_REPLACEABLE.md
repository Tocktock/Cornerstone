# Replaceable vs Non-replaceable

**Status:** Approved  
**Type:** Boundary Specification  
**Last Updated:** 2026-04-05

---

## 1. Purpose of this distinction

Some things in Cornerstone can change freely.  
Some things cannot change without changing what the product fundamentally is.

This document separates those two categories.

- **Non-replaceable** means the product essence. Changing it changes the meaning of Cornerstone.
- **Replaceable** means an implementation or packaging choice. It can change while the product remains the same.

This distinction is a guardrail against product drift.

---

## 2. Non-replaceable concepts

### 2.1 Shared organizational context layer
Cornerstone must remain the official organizational context layer.

It must **not** become a replacement for raw source systems.

---

### 2.2 Humans and AI depend on the same context
Humans and AI are both consumers of Cornerstone’s truth layer.

Cornerstone must **not** define itself as a single-model AI product.

---

### 2.3 Provenance-first official knowledge
Official concepts, relations, and decisions must remain grounded in evidence and source lineage whenever possible.

Cornerstone must **not** allow provenance-free official truth.

---

### 2.4 Eventual sync with source systems
Strict real-time behavior is optional. Eventual sync is not.

Cornerstone must remain consistent with source systems over time, with observability and recovery.

---

### 2.5 Reviewable officialization
AI may draft and suggest. Official knowledge must remain reviewable by humans.

Cornerstone must **not** equate AI-generated text with approved truth.

---

### 2.6 First-class curated objects
The following must remain first-class:
- `Concept`
- `ConceptRelation`
- `DecisionRecord`
- `EvidenceFragment`

These cannot be demoted to unstructured metadata without changing the product.

---

### 2.7 Decision context matters
Cornerstone must preserve not only definitions and relations, but the rationale lineage behind them.

If decision context disappears, Cornerstone loses its ability to reconstruct organizational context.

---

### 2.8 Spec-driven delivery
Meaningful changes must begin with a spec.

If the delivery model becomes code-first without specification, the product will drift away from its declared identity.

---

## 3. Replaceable choices

### 3.1 Connector adapters
- how a Notion connector is implemented
- how a Slack connector is implemented
- which providers are supported first

These change the implementation roadmap, not the product essence.

---

### 3.2 Sync mechanisms
- polling
- webhook
- hybrid
- reconciliation schedulers
- queue strategies

The important invariant is eventual sync, not the exact mechanism.

---

### 3.3 Storage engines
- relational database
- graph database
- document store
- object store
- search index

The important invariant is the domain model and its rules, not the engine.

---

### 3.4 API surface and transport
- REST
- GraphQL
- MCP
- internal RPC
- event contracts

The important invariant is that humans and AI can consume the same truth layer, not one exact transport choice.

---

### 3.5 AI models and orchestration
- Claude vs Gemini vs Codex vs internal model
- prompt strategy
- extraction orchestration
- reranking or reasoning components

The important invariant is that AI remains a consumer or worker over the same official context layer.

---

### 3.6 UX packaging
- one unified UI vs multiple surfaces
- admin-first vs review-first layout
- navigation structure
- visualization style

The important invariant is that reviewability, provenance, and clarity remain intact.

---

## 4. Decision test

Use the following decision test when changing the system.

### If a change affects...
- product identity
- the meaning of “official”
- provenance requirements
- reviewability requirements
- first-class entities
- the role of AI
- whether Cornerstone replaces source systems

### Then it requires...
- a `DecisionRecord`
- an SoT update
- linked spec updates

If a change affects only adapter implementation, storage choice, or transport shape, it usually requires:
- a feature spec or technical design
- no SoT identity change

---

## 5. Examples

### Example A: switching from polling-first to hybrid sync
This is **replaceable** as long as eventual sync and recoverability remain intact.

### Example B: dropping review and letting AI publish official concepts automatically
This is **non-replaceable** because it breaks reviewable officialization.

### Example C: moving from relational storage to relational + graph projection
This is **replaceable**.

### Example D: redefining Cornerstone as “an AI assistant for company docs”
This is **non-replaceable** because it changes product identity.

---

## 6. Operational rule

When in doubt, ask:

> **Does this change alter what Cornerstone fundamentally is, or does it only alter how Cornerstone is implemented?**

If it alters what Cornerstone fundamentally is, treat it as non-replaceable.

# Domain Model

**Status:** Approved  
**Type:** Domain Specification  
**Scope:** Cornerstone core model  
**Last Updated:** 2026-04-05

---

## 1. Modeling principles

Cornerstone’s domain model must satisfy the following principles.

1. Separate source ingestion from official meaning.
2. Preserve provenance down to evidence-level granularity.
3. Treat glossary and graph as **projections**, not as separate storage objects.
4. Treat relations and decisions as **first-class curated assertions**, not as loose metadata.
5. Keep official knowledge reviewable.
6. Preserve not only what is true, but why it became true.

---

## 2. Projection rules

### 2.1 Glossary is a projection
A glossary is not a separate first-class entity.

**Glossary = a projection over `Concept` and its linked evidence, relations, and decisions.**

### 2.2 Ontology + context graph is a projection
The graph is also not a separate first-class entity.

**Ontology + Context Graph = a projection over `Concept`, `ConceptRelation`, and `DecisionRecord`.**

### 2.3 Source-backed answer is a query result
Answers are not the canonical store.

**Source-backed Answer = a composed view over `Concept`, `ConceptRelation`, `DecisionRecord`, and `EvidenceFragment`.**

---

## 3. Decision context definition

In product language, use **Decision Context**.  
In the system model, the first-class entity name is **`DecisionRecord`**.

### Official definition
**Decision context is the official, source-backed context unit that preserves what was decided, why it was decided, what constraints and evidence shaped it, and what changed as a result.**

### Modeling constraints
A `DecisionRecord` is:
- not a full meeting transcript
- not a one-line conclusion
- not a storage bucket for hidden AI chain-of-thought

A `DecisionRecord` must instead be:
- bounded
- shareable
- reviewable
- explainable
- linked to affected concepts, relations, and evidence

---

## 4. Core entities

Cornerstone’s core model contains eight entities.

### 4.1 `ContextSpace`
The boundary within which meaning, officiality, and visibility are valid.

Examples:
- company
- business unit
- product line
- workspace
- team

**Role**
- scopes official meaning
- scopes review policy and visibility
- scopes source connections and curated knowledge

**Minimum fields**
- `id`
- `name`
- `namespace`
- `status`
- `reviewPolicy`
- `visibilityPolicy`

---

### 4.2 `SourceConnection`
A connection to an external provider such as Notion, Slack, Google Docs, Jira, or internal systems.

**Role**
- preserves provider-level connection context
- manages sync mode and sync state
- stores the external scope and health of the connection

**Minimum fields**
- `id`
- `contextSpaceId`
- `provider`
- `externalScope`
- `syncMode`
- `syncCursor`
- `lastSyncedAt`
- `healthStatus`

---

### 4.3 `Artifact`
The normalized persistent representation of a source object.

Examples:
- document
- page
- message thread
- file
- issue
- wiki page

**Role**
- provides persistent source memory
- anchors provenance
- supports source drill-down and source search fallback

**Minimum fields**
- `id`
- `contextSpaceId`
- `sourceConnectionId`
- `externalId`
- `artifactType`
- `title`
- `canonicalUrl`
- `sourceUpdatedAt`
- `syncedAt`
- `contentHash`
- `status`

**Invariant**
`Artifact` is not official knowledge. It is raw memory anchored to source systems.

---

### 4.4 `EvidenceFragment`
A citeable, bounded fragment from an artifact.

Examples:
- paragraph
- sentence range
- message excerpt
- table cell range
- structured claim extracted from source

**Role**
- enables precise provenance
- supports concept, relation, and decision evidence
- provides inspectable evidence for human review

**Minimum fields**
- `id`
- `artifactId`
- `selector`
- `excerpt`
- `normalizedClaim`
- `extractedBy`
- `confidence`
- `verificationStatus`

**Invariant**
Evidence must be citeable, not just a generic chunk.

---

### 4.5 `Concept`
The canonical unit of official meaning.

Examples:
- term
- system
- policy
- workflow
- role
- metric
- event

**Role**
- anchors glossary semantics
- carries canonical definition and aliases
- connects to relations, evidence, and decisions

**Minimum fields**
- `id`
- `contextSpaceId`
- `conceptType`
- `canonicalName`
- `aliases`
- `definition`
- `status`
- `ownerActorId`
- `effectiveFrom`
- `effectiveTo`

**Recommended concept types**
- `TERM`
- `SYSTEM`
- `POLICY`
- `WORKFLOW`
- `ROLE`
- `METRIC`
- `EVENT`

---

### 4.6 `ConceptRelation`
A first-class assertion between two concepts.

**Role**
- expresses the ontology and operational graph
- carries explanation, status, provenance, and time bounds
- remains reviewable and supersedable

**Minimum fields**
- `id`
- `contextSpaceId`
- `subjectConceptId`
- `predicate`
- `objectConceptId`
- `description`
- `status`
- `directionality`
- `confidence`
- `effectiveFrom`
- `effectiveTo`
- `introducedByDecisionId`

**Recommended predicate families**
- ontological: `IS_A`, `PART_OF`, `INSTANCE_OF`
- operational: `USED_IN`, `INPUT_TO`, `OUTPUT_OF`, `DEPENDS_ON`
- governance/semantic: `DEFINED_BY`, `OWNED_BY`, `ALIGNS_WITH`, `CONFLICTS_WITH`
- temporal/flow: `PRECEDES`, `TRIGGERS`, `RESULTS_IN`

**Invariant**
A `ConceptRelation` is not a bare graph edge. It is a curated assertion.

---

### 4.7 `DecisionRecord`
The first-class storage unit for decision context.

**Role**
- preserves rationale lineage
- explains why concepts or relations are official
- records constraints, impact, and supersession

**Minimum fields**
- `id`
- `contextSpaceId`
- `title`
- `problem`
- `decision`
- `rationale`
- `constraints`
- `impact`
- `status`
- `effectiveAt`
- `reviewAt`
- `supersedesDecisionId`

**Strongly recommended fields**
- `alternativesConsidered`
- `assumptions`
- `tradeOffs`
- `outcomeSummary`

---

### 4.8 `Actor`
A human, team, service, or AI entity that can own, propose, review, approve, or operate within Cornerstone.

**Role**
- owns concepts
- proposes or approves decisions
- anchors accountability and reviewability
- resolves source identities where needed

**Minimum fields**
- `id`
- `contextSpaceId`
- `actorType`
- `displayName`
- `externalIdentities`
- `status`

**Recommended actor types**
- `human`
- `team`
- `service`
- `ai`

---

## 5. Relationship model

### 5.1 Scope relationships
- `ContextSpace` has many `SourceConnection`
- `ContextSpace` has many `Artifact`
- `ContextSpace` has many `Concept`
- `ContextSpace` has many `ConceptRelation`
- `ContextSpace` has many `DecisionRecord`
- `ContextSpace` has many `Actor`

Every official object belongs to a `ContextSpace`.

---

### 5.2 Ingestion and provenance relationships
- `SourceConnection` produces `Artifact`
- `Artifact` contains or yields `EvidenceFragment`

This layer captures raw source memory and provenance anchors.

---

### 5.3 Curated knowledge relationships
- `Concept` is supported by one or more `EvidenceFragment`
- `ConceptRelation` connects `Concept` to `Concept`
- `ConceptRelation` is supported by one or more `EvidenceFragment`

This layer captures curated official meaning.

---

### 5.4 Decision and governance relationships
- `DecisionRecord` is supported by one or more `EvidenceFragment`
- `DecisionRecord` is about one or more `Concept`
- `DecisionRecord` may be about one or more `ConceptRelation`
- `DecisionRecord` may supersede another `DecisionRecord`
- `DecisionRecord` is proposed by `Actor`
- `DecisionRecord` is reviewed or approved by `Actor`
- `DecisionRecord` may introduce, modify, deprecate, or supersede `Concept` or `ConceptRelation`

This layer captures rationale lineage and officialization.

---

## 6. Status and lifecycle guidance

### 6.1 Concept status
Recommended statuses:
- `draft`
- `official`
- `deprecated`
- `rejected`

### 6.2 ConceptRelation status
Recommended statuses:
- `draft`
- `official`
- `deprecated`
- `rejected`
- `superseded`

### 6.3 DecisionRecord status
Recommended statuses:
- `proposed`
- `accepted`
- `rejected`
- `superseded`
- `deprecated`

### 6.4 EvidenceFragment verification status
Recommended statuses:
- `unverified`
- `reviewed`
- `verified`
- `disputed`

---

## 7. Official knowledge rule

Official knowledge must be grounded.

An `official` `Concept` or `ConceptRelation` must have at least one of the following:
- supporting `EvidenceFragment`
- an approving or introducing `DecisionRecord`

This rule is mandatory.  
It prevents ungrounded AI output from becoming official truth.

---

## 8. Replaceable implementation guidance

The domain model is not tied to one storage engine.

A recommended initial implementation is:
- relational storage as the system of record
- graph projection for traversal and rich retrieval
- provider-specific adapters for ingestion
- consumer-agnostic serving for API and MCP

These are implementation choices. The core model above is not replaceable without an SoT-level change.

---

## 9. Recommended join and link structures

If implemented in a relational store, the following link structures are recommended.

- `concept_evidence_links`
- `relation_evidence_links`
- `decision_evidence_links`
- `decision_concept_links`
- `decision_relation_links`
- `decision_actor_links`

These support provenance, review, and traceability without requiring the graph view to be the canonical store.

---

## 10. Modeling guardrails

1. Do not model glossary as a separate storage object.
2. Do not model graph as a separate storage object.
3. Do not treat answers as canonical truth objects.
4. Do not collapse decisions into free-form notes.
5. Do not collapse relations into anonymous edges.
6. Do not mark knowledge as official without reviewable grounding.

These guardrails protect Cornerstone’s product identity.

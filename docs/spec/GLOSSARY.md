# Glossary

**Status:** Approved  
**Type:** Canonical Glossary  
**Last Updated:** 2026-04-05

---

## Cornerstone
The product that connects scattered organizational documents, conversations, and systems in order to build and maintain an official layer of meaning, relationships, and decision context.

## Shared Organizational Context
The official layer of meaning, relationships, decision background, and evidence that both humans and AI can use.

## Semantic System of Record
A record layer that does not replace raw source systems but preserves an organization’s official meaning and decision context.

## ContextSpace
The boundary within which meaning, visibility, and officiality are valid. Examples include company, business unit, product line, workspace, or team.

## SourceConnection
A connection unit between Cornerstone and an external provider or source system.

## Artifact
A normalized persistent representation of a source object. An artifact is raw source memory, not official curated knowledge.

## EvidenceFragment
A citeable fragment of an artifact that can support a concept, relation, or decision.

## Concept
The canonical unit of official meaning in Cornerstone. A concept may represent a term, system, policy, workflow, role, metric, or event.

## ConceptRelation
A first-class, reviewable assertion between two concepts.

## Decision Context
The structured context that preserves what was decided, why it was decided, what constraints and evidence shaped it, and what changed as a result.

## DecisionRecord
The internal first-class storage entity for decision context.

## Provenance
The traceable linkage that shows where a definition, relation, or decision came from.

## Source-backed Answer
A query result that returns concepts, relations, decisions, and evidence together rather than only returning free-form text.

## Glossary
A projection over `Concept` and its linked evidence, relations, and decisions.

## Ontology + Context Graph
A projection over `Concept`, `ConceptRelation`, and `DecisionRecord`.

## Eventual Sync
A synchronization principle that prioritizes eventual correctness, recovery, and source alignment over strict real-time updates.

## Reviewable Officialization
The rule that AI may draft but humans must be able to review knowledge before it becomes official.

## Replaceable
An implementation choice that can change without changing the identity of Cornerstone.

## Non-replaceable
A concept or rule that defines the identity of Cornerstone and cannot change without an SoT-level decision.

## Spec-driven Development
A delivery model where meaningful changes begin from a specification and code is treated as a downstream artifact.

## Official Knowledge
Curated knowledge that is accepted as valid within a `ContextSpace` and is grounded in evidence and/or an approving decision.

## Rationale Lineage
The chain of reasoning, constraints, and prior decisions that explains why an official definition, relation, or policy exists.

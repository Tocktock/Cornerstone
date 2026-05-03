# v1.2.1 — Ontology Domain Model Proposal

## Purpose

This document proposes the data model and API shape needed to turn Cornerstone into an evidence-grounded ontology graph system.

It is a proposal for later implementation. `v1.2.1` does not add migrations or runtime models.

## Layer model

Cornerstone should use three layers:

```text
1. Evidence layer
   DataSource → SourceObject → Artifact → EvidenceFragment

2. Candidate ontology layer
   OntologyExtractionRun → ConceptCandidate / RelationCandidate

3. Official ontology layer
   Concept / ConceptRelation backed by reviewed evidence
```

The official ontology layer is the Single Source of Truth.

## Existing model fit

Existing entities already map well:

| Existing entity | Ontology meaning |
|---|---|
| `Artifact` | captured source document |
| `EvidenceFragment` | evidence support unit |
| `Concept` | graph node |
| `ConceptRelation` | graph edge |
| `DecisionRecord` | decision support or officialization support |
| `TrustLabel` | response trust state |
| `FreshnessState` | node/edge evidence freshness |

The next work should extend this rather than replace it.

## Proposed entity: OntologyExtractionRun

Tracks each LLM extraction attempt.

```text
OntologyExtractionRun
- id
- sourceScope
- artifactIds
- evidenceFragmentIds
- modelProvider
- modelName
- promptVersion
- status
- startedAt
- completedAt
- createdConceptCandidateCount
- createdRelationCandidateCount
- error
- createdBy
- createdAt
```

Recommended statuses:

```text
queued
running
completed
failed
cancelled
```

Why it exists:

```text
- auditability
- prompt/model traceability
- replayability
- debugging
- evaluation
- rollback and comparison
```

## Proposed entity: ConceptCandidate

Represents a proposed graph node before review.

```text
ConceptCandidate
- id
- extractionRunId
- name
- normalizedName
- aliases
- proposedDefinition
- proposedBody
- conceptType
- evidenceFragmentIds
- confidence
- status
- matchedConceptId
- rejectionReason
- createdBy
- reviewedBy
- createdAt
- reviewedAt
```

Recommended statuses:

```text
pending
approved
rejected
merged
needs_evidence
duplicate
```

`matchedConceptId` is used when the candidate should merge into an existing Concept instead of becoming a new Concept.

## Proposed entity: RelationCandidate

Represents a proposed graph edge before review.

```text
RelationCandidate
- id
- extractionRunId
- sourceConceptCandidateId
- sourceConceptId
- targetConceptCandidateId
- targetConceptId
- sourceName
- targetName
- relationType
- evidenceFragmentIds
- confidence
- rationale
- status
- matchedRelationId
- rejectionReason
- createdBy
- reviewedBy
- createdAt
- reviewedAt
```

A RelationCandidate may point to candidate Concepts, existing Concepts, or both.

Recommended statuses:

```text
pending
approved
rejected
merged
needs_evidence
duplicate
invalid_relation_type
```

## Proposed entity: ConceptAlias

Maps user language to canonical Concepts.

```text
ConceptAlias
- id
- conceptId
- alias
- normalizedAlias
- source
- createdBy
- createdAt
```

Alias sources:

```text
reviewer
llm
connector
manual
system
```

Examples that may map to the same canonical Concept:

```text
settlement
settlements
payment settlement
transaction settlement
settle transaction
```

## Graph response schemas

### OntologyGraphNode

```text
OntologyGraphNode
- id
- name
- normalizedName
- shortDefinition
- status
- trustLabel
- freshnessState
- evidenceFragmentIds
- aliases
- depth
```

### OntologyGraphEdge

```text
OntologyGraphEdge
- id
- sourceConceptId
- targetConceptId
- relationType
- status
- trustLabel
- freshnessState
- evidenceFragmentIds
- depth
- direction
```

### OntologyGraphResponse

```text
OntologyGraphResponse
- responseId
- query
- focusConcept
- depth
- mode
- nodes
- edges
- evidence
- limitations
- generatedAt
```

## Evidence requirements

A ConceptCandidate must reference at least one EvidenceFragment.

A RelationCandidate must reference at least one EvidenceFragment that supports the relation itself, not merely the existence of the source and target concepts.

Valid relation evidence:

```text
"Settlement occurs after clearing."
```

Supports:

```text
Settlement --follows--> Clearing
```

Invalid relation evidence:

```text
"Settlement is important."
"Clearing is important."
```

Does not support:

```text
Settlement --follows--> Clearing
```

## Official object requirements

An official Concept must have:

```text
- reviewed evidence or accepted DecisionRecord
- reviewer identity
- review timestamp
- non-empty definition
```

An official Relation must have:

```text
- source Concept
- target Concept
- allowed relation type
- sourceConceptId != targetConceptId
- reviewed evidence or accepted DecisionRecord
- reviewer identity
- review timestamp
```

## Name normalization

Suggested normalization:

```text
- trim whitespace
- collapse repeated whitespace
- lowercase
- remove surrounding punctuation
- preserve meaningful internal punctuation
- avoid aggressive singularization
```

Examples:

| Input | Normalized |
|---|---|
| `Settlement` | `settlement` |
| ` settlement ` | `settlement` |
| `Payment Settlement` | `payment settlement` |
| `settlements` | `settlements` |

Do not blindly singularize all words; domain terms can be sensitive.

## Deduplication checks

Before creating a ConceptCandidate, check:

```text
1. exact normalized Concept name match
2. exact normalized alias match
3. likely duplicate candidate from same extraction run
4. reviewer-approved merge history
```

Possible duplicate outcomes:

```text
status = duplicate
matchedConceptId = existing concept id
```

or a reviewer-facing merge suggestion.

## Relation taxonomy v1 proposal

Start with a controlled vocabulary.

| Relation type | Meaning | Example |
|---|---|---|
| `is_a` | Source is a kind of target | Card settlement is a settlement process |
| `part_of` | Source is a component of target | Settlement batch is part of Settlement Run |
| `depends_on` | Source requires target | Settlement depends on Payment Instruction |
| `precedes` | Source happens before target | Clearing precedes Settlement |
| `follows` | Source happens after target | Settlement follows Clearing |
| `produces` | Source creates target | Settlement produces Settlement Report |
| `consumes` | Source uses target as input | Settlement consumes Payment Instruction |
| `updates` | Source changes target | Settlement updates Ledger |
| `validates` | Source checks target | Reconciliation validates Settlement |
| `triggers` | Source initiates target | Settlement triggers Payout |
| `blocks` | Source prevents target | Compliance Hold blocks Settlement |
| `conflicts_with` | Source contradicts target | Legacy Policy conflicts with New Policy |
| `supersedes` | Source replaces target | New Policy supersedes Legacy Policy |
| `owned_by` | Source is owned by target | Settlement owned by Finance Operations |
| `used_by` | Source is used by target | Settlement Report used by Finance Team |
| `created_by` | Source is created by target | Settlement Report created by Worker |
| `governed_by` | Source is controlled by target | Settlement governed by Settlement Policy |
| `source_of_truth_for` | Source officially defines target | Policy source_of_truth_for Settlement SLA |
| `related_to` | Relation exists but is not yet specific | Settlement related_to Chargeback |

Do not add too many relation types early. Expand the taxonomy only from reviewed data.

## Relation direction

Direction must be explicit.

```text
Settlement --follows--> Clearing
```

is not the same as:

```text
Clearing --precedes--> Settlement
```

The UI may render incoming edges in a human-friendly way, but stored edges must keep their direction.

## LLM output contract

The LLM should return strict JSON.

```json
{
  "concepts": [
    {
      "name": "Settlement",
      "aliases": ["payment settlement"],
      "definition": "The process of finalizing a transaction obligation.",
      "body": "Settlement finalizes the transfer or recording of funds after required prior steps are complete.",
      "conceptType": "process",
      "evidenceFragmentIds": ["ev_123"],
      "confidence": 0.88
    }
  ],
  "relations": [
    {
      "sourceName": "Settlement",
      "targetName": "Clearing",
      "relationType": "follows",
      "evidenceFragmentIds": ["ev_124"],
      "confidence": 0.84,
      "rationale": "The source says clearing is complete before settlement."
    }
  ]
}
```

Backend validation must check:

```text
- JSON shape
- required fields
- evidence ids exist
- evidence ids are in extraction scope
- relation type is allowed
- source and target resolve
- source and target are not the same
- confidence is numeric and bounded
- definitions are non-empty
```

## Rejection rules

Reject or quarantine LLM output when:

```text
- output is not valid JSON
- concept has no evidence
- relation has no evidence
- relation type is unknown
- relation references a missing concept
- evidence id does not exist
- evidence id is outside the extraction scope
- source equals target
- rationale contradicts evidence
- model tries to set status=official
```

## Persistence proposal

Future PostgreSQL migration should add:

```text
ontology_extraction_runs
concept_candidates
relation_candidates
concept_aliases
```

Existing tables should remain:

```text
concepts
concept_relations
evidence_fragments
artifacts
data_sources
```

PostgreSQL is enough for the MVP. A graph database is not required for depth-1 and depth-2 graphs.

Recommended indexes:

```text
concept_relations.source_concept_id
concept_relations.target_concept_id
concept_aliases.normalized_alias
concept_candidates.normalized_name
relation_candidates.extraction_run_id
```

## API draft: search

Directional contract only; not implemented in `v1.2.1`.

```text
GET /v1/ontology/search?q=settlement
```

Response sketch:

```json
{
  "query": "settlement",
  "matches": [
    {
      "conceptId": "concept_settlement",
      "name": "Settlement",
      "status": "official",
      "matchType": "name",
      "aliases": ["payment settlement"]
    }
  ],
  "limitations": []
}
```

## API draft: graph

Directional contract only; not implemented in `v1.2.1`.

```text
GET /v1/ontology/graph?concept=settlement&depth=1&mode=official
```

Response sketch:

```json
{
  "responseId": "resp_123",
  "query": "settlement",
  "depth": 1,
  "mode": "official",
  "focusConcept": {
    "id": "concept_settlement",
    "name": "Settlement",
    "status": "official",
    "trustLabel": "official"
  },
  "nodes": [
    {
      "id": "concept_settlement",
      "name": "Settlement",
      "status": "official",
      "depth": 0
    },
    {
      "id": "concept_clearing",
      "name": "Clearing",
      "status": "official",
      "depth": 1
    }
  ],
  "edges": [
    {
      "id": "relation_1",
      "sourceConceptId": "concept_settlement",
      "targetConceptId": "concept_clearing",
      "relationType": "follows",
      "status": "official",
      "evidenceFragmentIds": ["ev_124"],
      "depth": 1
    }
  ],
  "evidence": [
    {
      "id": "ev_124",
      "text": "Settlement occurs after clearing is complete.",
      "sourceUrl": "https://example.internal/settlement-guide"
    }
  ],
  "limitations": []
}
```

## Settlement fixture for tests

Future tests should use this fixture:

```text
Artifact title:
Settlement Operations Guide

Evidence 1:
"Settlement is the process of finalizing the transfer of funds after clearing is complete."

Evidence 2:
"After settlement, the ledger is updated with the final settled amount."

Evidence 3:
"Reconciliation validates that settled amounts match transaction records."
```

Expected Concepts:

```text
Settlement
Clearing
Ledger
Reconciliation
```

Expected Relations:

```text
Settlement --follows--> Clearing
Settlement --updates--> Ledger
Reconciliation --validates--> Settlement
```

## Testing strategy

Future tests should cover:

```text
- candidate validation rejects no-evidence Concepts
- candidate validation rejects no-evidence Relations
- duplicate ConceptCandidate detection
- alias lookup
- graph depth=0
- graph depth=1
- official mode excludes candidates
- candidate mode labels candidate objects
- mixed mode labels every node and edge
- unsupported graph query returns limitations
- stale evidence affects trust state
- conflicted evidence is surfaced
- settlement fixture produces expected direct graph
```

## Migration caution

Do not implement LLM extraction before the candidate tables and trust boundary exist.

A model that writes directly into official Concepts or official ConceptRelations would violate the product contract.

## Chronicle position and measurable release checklist

This section was added during the v2.0.0 documentation chronicle pass so the version goal and checklist are explicit, measurable, and easy to audit.

**Version goal:** Propose the candidate/official ontology domain model without applying migrations or runtime models.

**Confirmed non-goal:** No database table, API endpoint, or model implementation in v1.2.1.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V121-DM-01 | The layer model separates evidence, candidate ontology, and official ontology. | Section `Layer model`. | complete |
| V121-DM-02 | The proposal defines OntologyExtractionRun, ConceptCandidate, RelationCandidate, and ConceptAlias. | Entity proposal sections. | complete |
| V121-DM-03 | The graph response proposal includes nodes, edges, citations, and trust state. | Section `Graph response schemas`. | complete |
| V121-DM-04 | Official object requirements require reviewed evidence. | Section `Official object requirements`. | complete |
| V121-DM-05 | The LLM output contract is structured and rejectable. | Sections `LLM output contract` and `Rejection rules`. | complete |

**Chronicle handoff:** This version is recorded in `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. Later versions may build on this release only within the confirmed non-goal boundary above.


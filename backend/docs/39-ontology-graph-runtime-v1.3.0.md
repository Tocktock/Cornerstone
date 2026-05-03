# v1.3.0 — Ontology Graph Runtime Contract

## Purpose

`v1.3.0` is the first runtime implementation of the ontology direction defined in `v1.2.1`.

This release adds a small, explainable ontology graph API over existing reviewed Cornerstone data:

```text
Concepts
ConceptAliases
ConceptRelations
EvidenceFragments
↓
/v1/ontology/search
/v1/ontology/graph
```

The goal is to prove that Cornerstone can serve an evidence-backed depth-1 graph for a user concept such as `settlement` without introducing LLM-generated candidates yet.

## Product goal

A user should be able to ask for a main Concept and receive:

```text
- the matching Concept
- direct neighbor Concepts at depth=1
- relation direction and relation type
- node and edge review status
- supporting EvidenceFragments and citations
- freshness summary
- trust label
- limitations
```

The default product behavior is:

```text
mode = official
depth = 1
```

This means the default graph is suitable for Single Source of Truth use because it only includes official Concepts and official ConceptRelations.

## Non-goals

`v1.3.0` intentionally does not implement:

```text
- LLM ontology extraction
- ConceptCandidate or RelationCandidate runtime tables
- candidate review workflow
- manual file upload ingestion
- vector search
- semantic deduplication
- graph visualization UI
- graph depth above 1
- graph database storage
- automatic relation inference
```

The release must not imply that Cornerstone can construct ontology graphs from raw documents yet. It can only serve graphs from existing Concepts and ConceptRelations.

## Why this version exists

`v1.2.1` defined the ontology Single Source of Truth contract. Before adding LLM extraction, Cornerstone needs a stable serving contract for graph consumers.

The safe order is:

```text
1. Document ontology rules.        Done in v1.2.1.
2. Serve existing graph safely.    Done in v1.3.0.
3. Add manual upload ingestion.    Planned for v1.3.1.
4. Add LLM candidates.            Planned for v1.4.0.
5. Add review workflow.           Planned for v1.5.0.
```

## Data model changes

### Concept aliases

`v1.3.0` adds `Concept.aliases` and a PostgreSQL/SQLite-backed `concept_aliases` table.

Purpose:

```text
- let users search for "payment settlement" and find "Settlement"
- support plural or domain-specific names such as "settlements"
- prepare for future deduplication and merge workflows
```

Alias rules:

```text
- aliases are optional
- aliases are normalized by casefolding and whitespace collapse
- aliases equal to the primary Concept name are ignored
- duplicate aliases on the same Concept are ignored
- a Concept name or alias cannot overlap an existing Concept name or alias on create
```

The runtime schema remains simple:

```json
{
  "id": "concept-id",
  "name": "Settlement",
  "aliases": ["payment settlement", "settlements"],
  "shortDefinition": "...",
  "status": "official"
}
```

### Relation taxonomy expansion

`v1.3.0` expands `RelationType` to better support ontology edges such as the `settlement` example.

Supported relation types now include:

```text
is_a
part_of
depends_on
precedes
follows
produces
consumes
updates
validates
triggers
blocks
conflicts_with
supersedes
owned_by
used_by
created_by
governed_by
source_of_truth_for
related_to
```

This is still a controlled taxonomy. New relation types should be added only when real reviewed data requires them.

## API contract

### Search ontology

```http
GET /v1/ontology/search?q=settlement&mode=official&limit=10
```

Query parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `q` | required | User search term. |
| `mode` | `official` | `official`, `candidate`, or `mixed`. |
| `limit` | `10` | Maximum results, 1 to 50. |

Response shape:

```json
{
  "query": "payment settlement",
  "mode": "official",
  "results": [
    {
      "id": "concept-id",
      "name": "Settlement",
      "aliases": ["payment settlement"],
      "shortDefinition": "The process of finalizing a transaction obligation.",
      "status": "official",
      "matchedBy": "alias",
      "matchedValue": "payment settlement",
      "score": 0.96
    }
  ],
  "generatedAt": "2026-04-30T00:00:00Z"
}
```

### Get ontology graph

```http
GET /v1/ontology/graph?concept=settlement&depth=1&mode=official
```

Query parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `concept` | required | Concept name or alias. |
| `depth` | `1` | `0` or `1`. v1.3.0 rejects depth above 1. |
| `mode` | `official` | `official`, `candidate`, or `mixed`. |

Response shape:

```json
{
  "query": "payment settlement",
  "mode": "official",
  "depth": 1,
  "focusConcept": {
    "id": "concept-settlement",
    "name": "Settlement",
    "aliases": ["payment settlement"],
    "shortDefinition": "The process of finalizing a transaction obligation.",
    "status": "official",
    "isFocus": true,
    "evidenceFragmentIds": ["ev-1"],
    "decisionRecordIds": []
  },
  "nodes": [
    {"id": "concept-settlement", "name": "Settlement", "status": "official", "isFocus": true},
    {"id": "concept-ledger", "name": "Ledger", "status": "official", "isFocus": false}
  ],
  "edges": [
    {
      "id": "relation-id",
      "sourceConceptId": "concept-settlement",
      "targetConceptId": "concept-ledger",
      "relationType": "updates",
      "status": "official",
      "evidenceFragmentIds": ["ev-2"]
    }
  ],
  "evidence": [
    {
      "evidenceFragmentId": "ev-1",
      "artifactId": "artifact-id",
      "text": "Settlement updates the ledger after clearing.",
      "artifactTitle": "Settlement Guide",
      "freshnessState": "fresh",
      "trustState": "reviewed",
      "supports": [
        {"entityType": "concept", "entityId": "concept-settlement", "relationship": "supports_focus_concept"},
        {"entityType": "concept_relation", "entityId": "relation-id", "relationship": "supports_graph_edge"}
      ],
      "isValid": true,
      "validityErrors": []
    }
  ],
  "freshness": {"state": "fresh", "staleEvidenceCount": 0, "unknownEvidenceCount": 0},
  "trustLabel": "official",
  "limitations": ["Official mode returns only official Concepts and official ConceptRelations."],
  "officialGraphAvailable": true
}
```

If no matching Concept exists in the requested mode, the endpoint returns `200` with `trustLabel=unsupported`, `focusConcept=null`, and limitations. This matches the grounded serving philosophy: unsupported context is explicit rather than hidden behind an ambiguous empty response.

## Graph modes

### `official`

Rules:

```text
- focus Concept must be official
- neighbor Concepts must be official
- ConceptRelations must be official
- rejected/deprecated/superseded objects are excluded
```

This is the default and the only mode suitable for Single Source of Truth output.

### `candidate`

Rules:

```text
- Concepts with candidate/reviewing status may be searched
- ConceptRelations with candidate/reviewing status may be shown
- the response must not be treated as official truth
```

`v1.3.0` does not create candidate Concepts or RelationCandidates; this mode only exposes existing non-official graph objects.

### `mixed`

Rules:

```text
- official and non-official Concepts may appear
- official and non-rejected ConceptRelations may appear
- every node and edge carries its status
- the response must not be treated as the SSOT
```

Mixed mode is useful for review tools and debugging.

## Trust behavior

The graph response computes a trust label:

| Trust label | Meaning |
|---|---|
| `official` | Official-mode graph with official nodes/edges and reviewed, fresh/aging evidence. |
| `evidence_supported` | Some reviewed evidence supports the graph, but the graph is not fully official. |
| `partially_supported` | Support exists but freshness is mixed/unknown or evidence is not fully reviewed. |
| `stale` | At least one serving citation is stale. |
| `conflicted` | Focus Concept or supporting evidence is conflicted. |
| `unsupported` | No matching graph or no serving-eligible supporting evidence. |

Official graph availability is true only when `trustLabel=official`.

## Settlement reference behavior

Given official Concepts:

```text
Settlement
Ledger
Clearing
```

And official Relations:

```text
Settlement --updates--> Ledger
Settlement --follows--> Clearing
```

A request:

```http
GET /v1/ontology/graph?concept=payment%20settlement&depth=1
```

Should return:

```text
Settlement
├── updates → Ledger
└── follows → Clearing
```

Each edge must carry supporting `EvidenceFragment` ids and citations.

## Implementation checklist

### Product/API

```text
[x] Add `/v1/ontology/search`.
[x] Add `/v1/ontology/graph`.
[x] Default graph mode to `official`.
[x] Default graph depth to `1`.
[x] Reject depth above `1` in v1.3.0.
[x] Return unsupported response for missing Concepts.
[x] Include citations, freshness, trust label, and limitations.
```

### Data model

```text
[x] Add `Concept.aliases` to DTOs.
[x] Add `concept_aliases` persistence table.
[x] Add alias normalization.
[x] Add alias conflict check on Concept creation.
[x] Add migration `0012_concept_aliases_ontology_graph`.
```

### Trust boundaries

```text
[x] Do not call any LLM.
[x] Do not create ontology candidates.
[x] Do not infer missing Concepts or Relations.
[x] Official mode excludes candidate Relations.
[x] Mixed mode labels non-official objects.
[x] Evidence citations are production-eligible only in production mode.
```

### Tests

```text
[x] Search by Concept alias.
[x] Depth-1 graph with official nodes and official edge.
[x] Official graph excludes candidate Relation.
[x] Mixed graph can include candidate Relation.
[x] Unknown Concept returns unsupported response.
[x] Depth above one is rejected.
[x] Alias search round-trips through SQLAlchemy persistence.
```

## Known limitations

```text
- No LLM extraction yet.
- No candidate review queue yet.
- No manual file upload endpoint yet.
- `candidate` mode is limited to existing non-official Concepts/Relations.
- `depth=2+` is intentionally rejected.
- Search is lexical over names and aliases, not semantic.
- Graph layout/visualization is not part of backend v1.3.0.
```

## Exit criteria

`v1.3.0` is complete when:

```text
- package version is `1.3.0`
- release document exists
- README current version points to v1.3.0
- release checker requires v1.3.0 docs
- compile check passes
- ontology integration tests pass
- release checker passes
```

## Next version handoff

`v1.3.1` should add manual file upload ingestion while preserving the same ontology serving contract.

Suggested `v1.3.1` target:

```text
manual uploaded text-like file
→ Artifact
→ EvidenceFragments
→ reviewer creates Concepts/Relations
→ /v1/ontology/graph returns depth-1 graph
```

Do not add LLM extraction until `v1.4.0`.

## Chronicle position and measurable release checklist

This section was added during the v2.0.0 documentation chronicle pass so the version goal and checklist are explicit, measurable, and easy to audit.

**Version goal:** Serve an evidence-backed depth-1 ontology graph from existing reviewed Concepts and ConceptRelations.

**Confirmed non-goal:** No LLM extraction, no candidate model, no automatic ontology construction.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V130-01 | `GET /v1/ontology/search` exists and can match by concept name or alias. | API contract and integration tests. | complete |
| V130-02 | `GET /v1/ontology/graph` serves depth=1 by default. | API contract, service behavior, and tests. | complete |
| V130-03 | Official mode excludes non-official/candidate relations. | Tests for official-only filtering. | complete |
| V130-04 | Depth greater than 1 is rejected in this release. | Graph API validation tests. | complete |
| V130-05 | Graph responses include citations, trust label, freshness, and limitations. | Graph response schema and tests. | complete |

**Chronicle handoff:** This version is recorded in `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. Later versions may build on this release only within the confirmed non-goal boundary above.


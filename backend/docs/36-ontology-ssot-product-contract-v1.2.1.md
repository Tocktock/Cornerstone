# v1.2.1 — Ontology SSOT Product Contract

## Purpose

`v1.2.1` is the product-contract release for the ontology direction of Cornerstone.

Cornerstone currently captures source content, extracts `EvidenceFragment`s, supports review, officializes `Concept`s, stores `ConceptRelation`s, and serves grounded context. The next product direction is to turn those primitives into an explainable ontology graph that helps users manage a Single Source of Truth.

The target workflow is:

```text
Sources and manual uploads
→ Artifacts
→ EvidenceFragments
→ LLM-proposed ontology candidates
→ human review
→ official Concepts and official ConceptRelations
→ explainable Single Source of Truth graph
```

This release intentionally avoids runtime behavior changes. It defines the contract that future versions must implement.

## Product thesis

Cornerstone should help a user answer:

```text
What does this organization officially mean by this concept,
how is it related to other concepts,
and which evidence proves that?
```

For example, if the user's main Concept is `settlement`, Cornerstone should eventually show:

```text
- the official definition of Settlement
- directly related Concepts by default
- relation direction and relation type
- evidence supporting the definition
- evidence supporting each relation
- whether each node and edge is official, candidate, stale, conflicted, or unsupported
```

## Single Source of Truth definition

In Cornerstone, the Single Source of Truth is not:

```text
- raw uploaded files
- raw connector documents
- a model answer
- an LLM extraction result
- an unreviewed candidate graph
```

The Single Source of Truth is:

```text
The reviewed, evidence-backed ontology graph made of official Concepts and official ConceptRelations.
```

Raw documents are evidence sources. LLM outputs are candidates. Human review promotes supported candidates into official graph objects.

## Core product rules

The following rules are non-negotiable:

```text
1. No evidence → no official Concept.
2. No evidence → no official Relation.
3. LLM output → candidate only.
4. Human review is required before a graph object becomes official.
5. Official graph mode must exclude candidate objects.
6. Mixed graph mode must label every object clearly.
7. A Relation is not official just because both Concepts are official.
8. A Concept is not official just because it appears in many sources.
9. Stale evidence must affect graph trust.
10. Conflicting evidence must be surfaced, not hidden.
11. Unsupported answers must say they are unsupported.
12. The product must not use general LLM memory as organizational truth.
```

## Primary user workflow

The intended workflow is:

```text
1. User connects sources or uploads manual data.
2. Cornerstone captures content as Artifacts.
3. Cornerstone extracts EvidenceFragments.
4. User or system selects an extraction scope.
5. LLM proposes ConceptCandidates and RelationCandidates.
6. Cornerstone validates the candidate JSON.
7. Reviewer approves, rejects, edits, or merges candidates.
8. Approved candidates become official Concepts and ConceptRelations.
9. User searches for a Concept such as "settlement".
10. Cornerstone serves a depth-limited ontology graph with evidence and trust labels.
```

## Default graph depth

The default graph depth is:

```text
depth = 1
```

Depth means the maximum number of relation hops from the focus Concept.

```text
depth=0 → only the focus Concept
depth=1 → focus Concept plus directly connected Concepts
depth=2 → focus Concept, neighbors, and neighbors of neighbors
```

`depth=1` is the default because it is explainable, reviewable, and avoids overwhelming users. Future versions may support larger graphs, but the initial maximum should be conservative, preferably `2`.

## Graph modes

Future graph-serving APIs should support these modes:

| Mode | Meaning | Default |
|---|---|---:|
| `official` | Only official Concepts and official ConceptRelations | Yes |
| `candidate` | Candidate Concepts and RelationCandidates for review/discovery | No |
| `mixed` | Official and candidate objects, every object explicitly labeled | No |

The product-facing answer should default to `official`. Review tools may use `candidate` or `mixed`.

## Settlement reference example

Use `settlement` as the first controlled example.

Example source statements:

```text
"Settlement is the process of finalizing the transfer of funds after clearing is complete."

"After settlement, the ledger is updated with the final settled amount."

"Reconciliation validates that settled amounts match transaction records."
```

Expected candidate Concepts:

```text
Settlement
Clearing
Ledger
Reconciliation
```

Expected candidate Relations:

```text
Settlement --follows--> Clearing
Settlement --updates--> Ledger
Reconciliation --validates--> Settlement
```

Default graph view for `settlement`, `depth=1`:

```text
Settlement
├── follows → Clearing
├── updates → Ledger
└── validated by ← Reconciliation
```

Every node and edge must be backed by evidence.

## Constructive Concept generation

A constructive Concept is a useful Concept synthesized from evidence rather than copied blindly from a heading.

Allowed:

```text
Evidence says: "Transactions are finalized after clearing."
LLM proposes: "Settlement" if the extraction scope clearly supports that domain term.
```

Not allowed:

```text
LLM invents "Settlement Risk Framework" without evidence.
LLM creates a relation because it is generally true outside the source material.
LLM marks a candidate as official.
```

Cornerstone may construct Concepts from evidence. It must not invent official truth.

## Explainability contract

A graph answer is explainable only if it can show:

```text
1. focus Concept
2. Concept status
3. definition
4. directly connected Relations at requested depth
5. relation direction
6. relation type
7. relation status
8. evidence fragment ids
9. source document metadata
10. freshness state
11. reviewer identity or review timestamp for official objects
12. limitations for missing, stale, conflicted, or candidate-only data
```

If the system cannot show evidence, the answer must be labeled unsupported or candidate-only.

## Candidate vs official wording

Use explicit product language.

Candidate:

```text
Cornerstone found a possible Concept awaiting review.
```

Official:

```text
Cornerstone has an official Concept supported by reviewed evidence.
```

Unsupported:

```text
Cornerstone does not have enough reviewed evidence to answer officially.
```

Conflicted:

```text
Cornerstone found conflicting reviewed evidence.
```

## Source and upload contract

The ontology direction must support:

```text
- manual source sync
- future manual file upload
- Notion pages
- Google Drive Docs and text files
- future Slack, GitHub, Sheets, Slides, and PDFs
```

Manual uploaded data follows the same trust path:

```text
upload → Artifact → EvidenceFragment → candidate extraction → review → official graph
```

Manual upload must not bypass review.

## Future API intent

The following routes are directional contracts only. They are not implemented by `v1.2.1`.

```text
GET  /v1/ontology/search?q=settlement
GET  /v1/ontology/graph?concept=settlement&depth=1&mode=official

POST /v1/ontology/extraction-runs
GET  /v1/ontology/extraction-runs/{runId}

GET  /v1/ontology/concept-candidates
POST /v1/ontology/concept-candidates/{candidateId}/approve
POST /v1/ontology/concept-candidates/{candidateId}/reject
POST /v1/ontology/concept-candidates/{candidateId}/merge

GET  /v1/ontology/relation-candidates
POST /v1/ontology/relation-candidates/{candidateId}/approve
POST /v1/ontology/relation-candidates/{candidateId}/reject
```

`v1.3.0` should first implement graph/search behavior over existing Concepts and ConceptRelations before LLM extraction is introduced.

## Success metrics for future versions

Future ontology versions should measure:

```text
concept_candidate_acceptance_rate
relation_candidate_acceptance_rate
duplicate_candidate_rate
unsupported_official_concept_count
unsupported_official_relation_count
graph_answer_official_rate
graph_answer_unsupported_rate
evidence_coverage_per_concept
evidence_coverage_per_relation
stale_official_graph_object_count
conflicted_graph_object_count
```

The most important safety target is:

```text
unsupported_official_graph_object_count = 0
unsupported_official_relation_count = 0
```

## v1.2.1 checklist

```text
[x] Define ontology SSOT product contract.
[x] Define candidate vs official trust boundary.
[x] Define default graph depth behavior.
[x] Define graph modes.
[x] Define settlement reference example.
[x] Define future API intent.
[x] Add ontology domain model proposal.
[x] Add versioned implementation plan.
[x] Add official trust-boundary ADR.
[x] Add reusable future release document template.
[ ] Implement graph endpoint. Deferred to v1.3.0.
[ ] Implement manual file upload. Deferred to v1.3.1.
[ ] Implement LLM extraction. Deferred to v1.4.0.
[ ] Implement ontology review workflow. Deferred to v1.5.0.
```

## Exit criteria

`v1.2.1` is complete when:

```text
1. Version metadata says 1.2.1.
2. README identifies v1.2.1 as current.
3. Ontology product contract exists.
4. Ontology domain model proposal exists.
5. Versioned implementation plan exists.
6. Trust-boundary ADR exists.
7. Release-document template exists.
8. Release checker targets v1.2.1.
9. Static syntax compilation passes.
10. Release-candidate static checks pass.
```

## Handoff to v1.3.0

Next version:

```text
v1.3.0 — Ontology graph contract implementation
```

Recommended first engineering scope:

```text
1. Add ConceptAlias.
2. Add ontology graph response schemas.
3. Add ontology search by Concept name and alias.
4. Add depth-1 graph endpoint over existing Concepts and ConceptRelations.
5. Keep LLM extraction out of scope.
6. Prove graph response with hand-created Concepts and Relations.
```

## Chronicle position and measurable release checklist

This section was added during the v2.0.0 documentation chronicle pass so the version goal and checklist are explicit, measurable, and easy to audit.

**Version goal:** Define the ontology SSOT product contract and trust boundary before runtime implementation.

**Confirmed non-goal:** No runtime API, persistence, extraction, or graph-serving behavior changes.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V121-PC-01 | The document defines the official ontology graph as the Single Source of Truth. | Section `Single Source of Truth definition` states reviewed official graph is SSOT. | complete |
| V121-PC-02 | The document states LLM output is candidate-only and cannot become official automatically. | Sections `Core product rules` and `Candidate vs official wording`. | complete |
| V121-PC-03 | The document defines default graph depth as 1. | Section `Default graph depth`. | complete |
| V121-PC-04 | The document includes the settlement reference example. | Section `Settlement reference example`. | complete |
| V121-PC-05 | The release exits with documentation only and no runtime behavior change. | Section `Exit criteria` and release notes. | complete |

**Chronicle handoff:** This version is recorded in `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. Later versions may build on this release only within the confirmed non-goal boundary above.


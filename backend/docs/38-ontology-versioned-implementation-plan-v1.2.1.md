# v1.2.1 — Ontology Versioned Implementation Plan

## Purpose

This document converts the ontology direction into a versioned implementation plan.

Every version should have:

```text
- a clear goal
- scope
- deferred scope
- checklist
- test plan
- proof or verification
- next-version handoff
```

## Release philosophy

Cornerstone is moving from:

```text
evidence-backed organizational context backend
```

to:

```text
LLM-assisted, human-reviewed, evidence-grounded ontology graph backend
```

Because this affects trust, implementation must be staged.

Safe order:

```text
1. Document product and trust contract.
2. Implement graph serving over existing reviewed Concepts and Relations.
3. Add manual uploaded data.
4. Add LLM extraction as candidate-only.
5. Add review and merge workflows.
6. Serve explainable graph answers.
7. Add ontology evaluation.
8. Add connector-driven re-extraction.
```

## Version overview

| Version | Theme | Goal |
|---|---|---|
| `v1.2.1` | Ontology documentation baseline | Define product contract, domain model, roadmap, and trust-boundary ADR. |
| `v1.3.0` | Graph contract implementation | Serve depth-1 ontology graph over existing Concepts and ConceptRelations. |
| `v1.3.1` | Manual uploaded data | Add manual file upload ingestion for text-like files. |
| `v1.4.0` | LLM ontology extraction | Add extraction runs and Concept/Relation candidates. |
| `v1.4.1` | Alias and deduplication | Reduce duplicate Concepts and improve matching. |
| `v1.5.0` | Ontology review workflow | Approve, reject, edit, and merge candidates. |
| `v1.6.0` | Explainable graph serving | Serve official/candidate/mixed graph with evidence and trust status. |
| `v1.7.0` | Ontology evaluation | Measure extraction quality and official graph safety. |
| `v1.8.0` | Connector-driven re-extraction | Re-extract candidates when sources change. |
| `v1.9.0` | End-to-end proof runner | Add one-command proof for upload/extract/review/graph/answer. |
| `v2.0.0` | Ontology SSOT release | Stabilize backend contract for production pilots. |

## v1.2.1 — Ontology documentation baseline

### Goal

Define the ontology product contract before implementation.

### Included

```text
- product contract
- domain model proposal
- versioned roadmap
- official trust-boundary ADR
- future release-document template
- release readiness document
- package version bump to 1.2.1
```

### Deferred

```text
- graph API
- candidate models
- LLM provider boundary
- manual upload endpoint
- database migration
- frontend graph UI
```

### Checklist

```text
[x] Add docs/36-ontology-ssot-product-contract-v1.2.1.md.
[x] Add docs/37-ontology-domain-model-proposal-v1.2.1.md.
[x] Add docs/38-ontology-versioned-implementation-plan-v1.2.1.md.
[x] Add docs/adr/0001-ontology-official-trust-boundary.md.
[x] Add docs/templates/release-doc-template.md.
[x] Add docs/release/v1.2.1-ontology-contract-readiness.md.
[x] Add docs/release/v1.2.1-release-notes.md.
[x] Update README current version.
[x] Update package version metadata.
[x] Update release checker target version.
[x] Run syntax compile check.
[x] Run static release-candidate check.
```

### Exit criteria

```text
- Version metadata says 1.2.1.
- Documentation explains how "settlement" becomes an explainable depth-1 graph.
- Trust boundary is clear: LLM candidate only, human review for official.
- Next release has clear engineering scope.
```

## v1.3.0 — Graph contract implementation

### Goal

Implement ontology search and graph serving over existing Concepts and ConceptRelations without LLM extraction.

### Included

```text
- ConceptAlias model and persistence
- ontology graph response schemas
- ontology search endpoint
- ontology graph endpoint
- depth=0 and depth=1 support
- official graph mode
- tests using hand-created Concepts and Relations
```

### Deferred

```text
- LLM extraction
- candidate tables
- LLM-created candidate graph objects
- graph visualization UI
```

### API target

```text
GET /v1/ontology/search?q=settlement
GET /v1/ontology/graph?concept=settlement&depth=1&mode=official
```

### Checklist

```text
[x] Add Concept aliases to Concept schema.
[x] Add ConceptAlias persistence model.
[x] Persist aliases in SQLAlchemy store and in-memory Concept DTOs.
[x] Add OntologyGraphNode schema.
[x] Add OntologyGraphEdge schema.
[x] Add OntologyGraphResponse schema.
[x] Add ontology route module.
[x] Add search endpoint.
[x] Add graph endpoint.
[ ] Support lookup by concept id. Deferred; v1.3.0 lookup is by name or alias.
[x] Support lookup by concept name.
[x] Support lookup by alias.
[x] Default depth to 1.
[x] Reject depth greater than configured maximum.
[x] Official mode excludes non-official Concepts and Relations.
[x] Edges include evidenceFragmentIds.
[x] Response includes limitations.
[x] Add tests for unknown Concept.
[ ] Add tests for depth=0. Deferred; endpoint supports depth=0 but the main acceptance path is depth=1.
[x] Add tests for depth=1.
[x] Add tests for outgoing depth-1 edges; incoming edge coverage is deferred to v1.6.0 graph hardening.
[x] Add tests for alias lookup.
```

### Exit criteria

A user can query `settlement` and receive an official depth-1 graph using manually created Concepts and Relations.

## v1.3.1 — Manual uploaded data

### Goal

Allow users to upload text-like files into the same Artifact → EvidenceFragment pipeline.

### Included

```text
- manual upload endpoint
- supported text-like MIME types
- file size limits
- deterministic content hashing
- Artifact creation
- EvidenceFragment creation
- upload safety docs
```

### Deferred

```text
- PDF OCR
- spreadsheets
- slides
- binary files
- direct LLM extraction during upload
```

### Checklist

```text
[ ] Add manual upload route.
[ ] Accept .txt and .md.
[ ] Consider .csv only if content handling is explicit.
[ ] Enforce maximum file size.
[ ] Reject unsupported binary files.
[ ] Store upload metadata as providerMetadata.
[ ] Create Artifact.
[ ] Run existing evidence extraction.
[ ] Keep EvidenceFragments unreviewed.
[ ] Add tests for supported file type.
[ ] Add tests for unsupported file type.
[ ] Add tests for duplicate content hash behavior.
```

### Exit criteria

A user can upload settlement text and get unreviewed EvidenceFragments.

## v1.4.0 — LLM ontology extraction

### Goal

Use an LLM to propose ConceptCandidates and RelationCandidates from EvidenceFragments.

### Included

```text
- LLM provider interface
- mock LLM provider for deterministic tests
- OntologyExtractionRun
- ConceptCandidate
- RelationCandidate
- strict JSON validation
- candidate creation from evidence scope
```

### Deferred

```text
- automatic approval
- official graph mutation without review
- advanced deduplication
- streaming extraction
```

### Checklist

```text
[ ] Add LLM provider interface.
[ ] Add mock provider.
[ ] Add extraction prompt contract.
[ ] Add OntologyExtractionRun schema/model.
[ ] Add ConceptCandidate schema/model.
[ ] Add RelationCandidate schema/model.
[ ] Validate JSON output.
[ ] Reject no-evidence ConceptCandidate.
[ ] Reject no-evidence RelationCandidate.
[ ] Reject invalid relation type.
[ ] Reject source == target.
[ ] Store modelName and promptVersion.
[ ] Add extraction-run API.
[ ] Add tests with settlement fixture.
```

### Exit criteria

Given settlement evidence, Cornerstone can create candidate Concepts and candidate Relations, but cannot officialize them automatically.

## v1.4.1 — Alias and deduplication

### Goal

Prevent the ontology from creating multiple Concepts for the same idea.

### Included

```text
- normalized name matching
- alias matching
- duplicate candidate detection
- merge suggestions
- reviewer-facing duplicate metadata
```

### Checklist

```text
[ ] Normalize candidate names.
[ ] Normalize aliases.
[ ] Match existing Concept names.
[ ] Match existing Concept aliases.
[ ] Match candidates within the same extraction run.
[ ] Mark likely duplicates.
[ ] Add tests for settlement/settlements/payment settlement.
```

### Exit criteria

The system can recognize that `settlement`, `settlements`, and `payment settlement` may refer to the same canonical Concept.

## v1.5.0 — Ontology review workflow

### Goal

Let reviewers approve, reject, edit, or merge candidates.

### Included

```text
- candidate review queue
- approve ConceptCandidate
- reject ConceptCandidate
- merge ConceptCandidate into existing Concept
- approve RelationCandidate
- reject RelationCandidate
- audit events
```

### Checklist

```text
[ ] Add candidate queue endpoint.
[ ] Add approve ConceptCandidate endpoint.
[ ] Add reject ConceptCandidate endpoint.
[ ] Add merge ConceptCandidate endpoint.
[ ] Add approve RelationCandidate endpoint.
[ ] Add reject RelationCandidate endpoint.
[ ] Require authorized reviewer.
[ ] Preserve reviewer identity.
[ ] Preserve review timestamp.
[ ] Preserve evidence ids.
[ ] Add tests for unauthorized reviewer.
[ ] Add tests for no-evidence candidate approval blocked.
```

### Exit criteria

A reviewer can promote LLM-proposed settlement candidates into reviewed ontology graph objects.

## v1.6.0 — Explainable graph serving

### Goal

Serve graph answers that include citations, trust labels, freshness, and limitations.

### Included

```text
- official graph mode
- candidate graph mode
- mixed graph mode
- citation expansion
- freshness summary
- limitations
- graph answer text
```

### Checklist

```text
[ ] Add mode=official.
[ ] Add mode=candidate.
[ ] Add mode=mixed.
[ ] Expand evidence citations.
[ ] Include source metadata.
[ ] Include freshness state.
[ ] Include limitations.
[ ] Surface conflicted evidence.
[ ] Surface stale evidence.
[ ] Add tests for official-only graph.
[ ] Add tests for mixed graph labeling.
```

### Exit criteria

The application can answer `what is settlement and how is it related?` with a depth-1 graph and citations.

## v1.7.0 — Ontology evaluation

### Goal

Measure graph quality and trust safety.

### Metrics

```text
concept_candidate_acceptance_rate
relation_candidate_acceptance_rate
duplicate_candidate_rate
unsupported_official_concept_count
unsupported_official_relation_count
graph_answer_official_rate
graph_answer_unsupported_rate
stale_official_graph_object_count
conflicted_graph_object_count
```

### Checklist

```text
[ ] Add ontology eval tasks.
[ ] Add settlement fixture eval.
[ ] Add unsupported official Concept check.
[ ] Add unsupported official Relation check.
[ ] Add stale graph object check.
[ ] Add candidate acceptance metrics.
[ ] Add relation evidence validity metric.
```

### Exit criteria

The release can prove that official graph objects are evidence-backed and extraction quality is measurable.

## v1.8.0 — Connector-driven re-extraction

### Goal

Re-run ontology candidate extraction when source content changes.

### Included

```text
- extraction invalidation policy
- source change detection hooks
- stale candidate handling
- stale official graph warning
```

### Checklist

```text
[ ] Trigger extraction after sync job completion.
[ ] Track source content hash changes.
[ ] Mark affected candidates stale.
[ ] Mark affected official objects aging/stale when evidence changes.
[ ] Avoid duplicate extraction on unchanged content.
```

### Exit criteria

Google Drive, Notion, and manual upload changes can produce new ontology candidates without corrupting the official graph.

## v1.9.0 — End-to-end proof runner

### Goal

Add a one-command proof loop for ontology SSOT.

### Target flow

```text
manual upload
→ evidence extraction
→ LLM candidate extraction
→ review
→ official graph
→ graph answer
→ evaluation
```

### Checklist

```text
[ ] Add `cornerstone live ontology` or `cornerstone proof ontology`.
[ ] Include settlement fixture.
[ ] Generate proof report.
[ ] Include candidate counts.
[ ] Include approved counts.
[ ] Include graph response.
[ ] Include evidence citations.
[ ] Include eval result.
```

### Exit criteria

A reviewer can prove the ontology product loop locally with one command.

## v2.0.0 — Ontology SSOT release

### Goal

Stabilize the ontology backend contract for production pilots.

### Included

```text
- stable ontology API
- stable candidate review API
- stable graph response shape
- documented trust labels
- documented relation taxonomy
- production operator runbook
- release proof record
```

### Exit criteria

Cornerstone can be used as an explainable ontology-based Single Source of Truth backend for pilot users.

## Documentation rule for every version

Every version must include a document with:

```text
Purpose
User value
Scope
Deferred scope
Product behavior
API contract
Data model changes
Trust and safety rules
Implementation checklist
Test plan
Proof or verification
Known limitations
Exit criteria
Next version handoff
```

Use:

```text
docs/templates/release-doc-template.md
```

## Warning

Do not skip directly from v1.2.1 to LLM extraction.

The safe sequence is:

```text
v1.3.0 graph API
v1.3.1 manual upload
v1.4.0 LLM candidates
v1.5.0 review workflow
```

## Chronicle position and measurable release checklist

This section was added during the v2.0.0 documentation chronicle pass so the version goal and checklist are explicit, measurable, and easy to audit.

**Version goal:** Convert the ontology direction into a sequenced implementation plan from v1.2.1 to v2.0.0.

**Confirmed non-goal:** The plan does not itself implement APIs, data models, or extraction behavior.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V121-PLAN-01 | Every planned version has a named goal. | Section `Version overview`. | complete |
| V121-PLAN-02 | The plan preserves the candidate/official trust boundary through v2.0.0. | Sections `Release philosophy` and version-specific sections. | complete |
| V121-PLAN-03 | The plan defines documentation requirements for every future version. | Section `Documentation rule for every version`. | complete |
| V121-PLAN-04 | The plan explicitly warns against LLM output becoming official truth. | Section `Warning`. | complete |
| V121-PLAN-05 | The chronicle records that planned v1.4.1 was not released separately. | Master chronicle section `Planned-but-not-released item`. | complete |

**Chronicle handoff:** This version is recorded in `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. Later versions may build on this release only within the confirmed non-goal boundary above.


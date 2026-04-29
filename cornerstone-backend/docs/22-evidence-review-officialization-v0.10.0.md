# v0.10.0 — Evidence Review Queue and Officialization Hardening

## Goal

Move the backend from source ingestion into the reviewer workflow required by the Cornerstone trust loop:

```text
EvidenceFragment
→ reviewer inspection
→ Concept candidate / ConceptRelation / DecisionRecord
→ official context only after quality gates
```

This release does not add UI, additional connectors, grounded serving, or evaluation. It hardens the backend APIs and persistence needed for the Officialization Workbench.

## Added backend capabilities

### Evidence review queue

New endpoint:

```http
GET /v1/evidence/review-queue
```

Supported filters:

```text
trustState
sourceId
freshnessState
fragmentType
limit
```

Each queue item includes:

```text
EvidenceFragment
Artifact context
DataSource context
linked Concept IDs
linked DecisionRecord IDs
suggested reviewer actions
```

The queue defaults to `trustState=unreviewed` so reviewers see work that still needs attention.

### Evidence review states

Evidence trust state now supports:

```text
unreviewed
reviewed
rejected
conflicted
```

`conflicted` lets reviewers mark evidence that should not be used blindly but may need a DecisionRecord or conflict-resolution workflow.

### Concept candidate creation from evidence

New endpoint:

```http
POST /v1/evidence/{evidence_fragment_id}/concept-candidates
```

Rules:

```text
- Actor must be an authorized reviewer.
- EvidenceFragment must exist.
- Created Concept status defaults to reviewing.
- EvidenceFragment is attached to the Concept.
- Creation is audit logged.
```

### ConceptRelation API

New endpoints:

```http
GET  /v1/concept-relations
GET  /v1/concept-relations/{relation_id}
POST /v1/concept-relations
POST /v1/concept-relations/{relation_id}/officialize
```

Supported relation types are inherited from the PRD contract:

```text
is_a
part_of
depends_on
conflicts_with
supersedes
owned_by
used_by
created_by
governed_by
source_of_truth_for
```

### Relation officialization gate

A ConceptRelation can become official only when:

```text
- The reviewer is authorized.
- The source Concept exists and is official.
- The target Concept exists and is official.
- The relation has at least one reviewed eligible EvidenceFragment or a valid DecisionRecord.
- Evidence has provenance.
- Evidence is not stale/unknown when production quality gates require freshness.
- Production-mode source evidence is from production-enabled sources.
```

Blocked officialization attempts create audit events.

### Atomic reviewer workflow writes

The following writes now use explicit transaction boundaries:

```text
Evidence review + audit event
Concept creation + audit event
Concept officialization + audit event
DecisionRecord creation + audit event
ConceptRelation creation + audit event
ConceptRelation officialization + audit event
```

## Persistence changes

New tables:

```text
concept_relations
concept_relation_evidence_fragments
```

New migration:

```text
0010_officialization_workbench
```

Indexes:

```text
ix_concept_relations_source
ix_concept_relations_target
ix_concept_relations_status
ix_concept_relations_type
```

Constraint:

```text
source_concept_id <> target_concept_id
```

## API examples

Review evidence:

```bash
curl -X POST http://localhost:8000/v1/evidence/{evidenceFragmentId}/review \
  -H 'Content-Type: application/json' \
  -d '{"trustState":"reviewed","reviewedBy":"reviewer@example.com","reviewNote":"Valid source-backed definition."}'
```

Create a Concept candidate from evidence:

```bash
curl -X POST http://localhost:8000/v1/evidence/{evidenceFragmentId}/concept-candidates \
  -H 'Content-Type: application/json' \
  -d '{"name":"Cornerstone","shortDefinition":"Evidence-backed organizational context layer.","createdBy":"reviewer@example.com"}'
```

Create a relation:

```bash
curl -X POST http://localhost:8000/v1/concept-relations \
  -H 'Content-Type: application/json' \
  -d '{"sourceConceptId":"...","targetConceptId":"...","relationType":"depends_on","evidenceFragmentIds":["..."],"createdBy":"reviewer@example.com"}'
```

Officialize a relation:

```bash
curl -X POST http://localhost:8000/v1/concept-relations/{relationId}/officialize \
  -H 'Content-Type: application/json' \
  -d '{"reviewedBy":"reviewer@example.com"}'
```

## Verification

Generated reports for this version cover:

```text
pytest
coverage
JUnit XML
ruff
mypy
compileall
Alembic offline SQL
```

Key regression tests cover:

```text
- Review queue filters and queue item context.
- Evidence conflicted state.
- Concept candidate creation from evidence.
- Unauthorized reviewer blocking.
- Relation creation and officialization.
- Relation officialization without support is blocked.
- Relation officialization requires official source and target Concepts.
- Relation persistence across SQLAlchemy store reloads.
```

## Remaining work

```text
- UI review workbench and batch review operations.
- Conflict resolution workflow beyond the conflicted evidence state.
- Full RBAC/identity provider integration.
- Grounded serving contract hardening.
- Evaluation framework.
- Production KMS/secret-manager integration.
- Live PostgreSQL execution in CI/local Docker.
```

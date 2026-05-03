# User Roles and Workflows

Cornerstone has several user roles. A small team may have one person perform multiple roles, but the product responsibilities are different.

## Source Admin

The Source Admin connects or uploads knowledge sources.

Typical actions:

```text
- create a manual source
- upload settlement notes
- connect Google Drive or Notion
- select which source objects are eligible for sync
- monitor source sync status
```

The Source Admin does not decide official truth by uploading data.

## Reviewer

The Reviewer controls what becomes official.

Typical actions:

```text
- review EvidenceFragments
- inspect ConceptCandidates
- inspect RelationCandidates
- approve supported candidates
- reject unsupported candidates
- edit candidate definitions
- merge duplicate Concepts or Relations
```

The Reviewer protects the Single Source of Truth boundary.

## Operator

The Operator runs checks and verifies readiness.

Typical actions:

```text
- run the ontology proof loop
- run ontology evaluation
- run SSOT readiness checks
- inspect limitations and recommended actions
- confirm whether a focus Concept is safe to serve
```

The Operator does not manually invent graph truth.

## End User

The End User asks questions.

Typical questions:

```text
What is settlement?
What does settlement update?
What concepts are directly related to settlement?
Why is this answer official?
What evidence supports this edge?
```

The End User should see reviewed answers, citations, trust labels, and limitations.

## Developer

The Developer integrates Cornerstone into tools.

Typical actions:

```text
- call ontology graph APIs
- call readiness APIs
- build internal UI on top of graph response data
- integrate graph output into AI assistants
- preserve citations and trust labels in downstream tools
```

The Developer should not strip evidence, trust, or provenance from responses.

## Workflow: manual source to official graph

```text
Source Admin uploads source text.
Cornerstone creates Artifacts and EvidenceFragments.
Extractor creates candidates.
Reviewer reviews evidence and candidates.
Operator runs graph evaluation and readiness.
End User asks for the official graph.
Developer may expose the graph in an internal UI or assistant.
```

## Role/workflow acceptance checklist

| Check ID | Requirement | Measurement | Status |
|---|---|---|---|
| PROD-ROLES-01 | Main roles are named. | Source Admin, Reviewer, Operator, End User, Developer are defined. | complete |
| PROD-ROLES-02 | Role boundaries are explicit. | Uploading data does not decide truth; review does. | complete |
| PROD-ROLES-03 | Reviewer responsibility is clear. | Review actions and SSOT boundary are described. | complete |
| PROD-ROLES-04 | Operator checklist role is clear. | Proof, evaluation, and readiness are described. | complete |
| PROD-ROLES-05 | End-user experience is described. | Example questions are included. | complete |

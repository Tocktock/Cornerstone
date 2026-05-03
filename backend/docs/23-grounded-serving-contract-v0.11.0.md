# 23 — Grounded Serving Contract Hardening v0.11.0

## Goal

v0.11.0 hardens the backend serving contract used by humans, Workspace/Explore surfaces, and AI clients.

The product rule is:

```text
Humans and AI must consume the same grounded context contract.
Unsupported, stale, conflicted, and partially supported states must be explicit.
```

This version does not add a new connector. It improves how already-ingested and reviewed context is served.

## Implemented

### Shared response contract

`GET /v1/context/query` now returns a richer `GroundedContextResponse`:

```json
{
  "responseId": "uuid",
  "query": "What is Cornerstone?",
  "answer": "Cornerstone: Shared organizational context layer.",
  "trustLabel": "official",
  "concepts": [],
  "relations": [],
  "decisions": [],
  "evidence": [],
  "freshness": {
    "state": "fresh",
    "staleEvidenceCount": 0,
    "unknownEvidenceCount": 0
  },
  "limitations": [],
  "generatedAt": "2026-04-26T00:00:00Z",
  "officialAnswerAvailable": true
}
```

### Trust labels

The serving layer now consistently returns:

```text
official
evidence_supported
partially_supported
stale
conflicted
unsupported
```

Rules:

- `official` requires an official Concept and valid reviewed fresh/aging evidence.
- `evidence_supported` is allowed when reviewed evidence exists but the Concept is not official or no Concept exists yet.
- `partially_supported` is used for unreviewed evidence, mixed/unknown freshness, or incomplete support.
- `stale` is used when at least one cited EvidenceFragment is stale.
- `conflicted` is used when the Concept or any cited EvidenceFragment is conflicted.
- `unsupported` is used when no valid eligible evidence can support the response.

### Evidence-only answers

If no matching Concept exists but matching valid EvidenceFragments exist, the API no longer returns a generic unsupported response. It returns an evidence-based answer with a non-official trust label.

This protects the distinction between:

```text
No context exists.
```

and:

```text
Evidence exists, but it has not been officialized.
```

### Relation and DecisionRecord citations

Grounded responses now include official ConceptRelations for a matched Concept and DecisionRecords attached to Concepts or Relations.

Each `EvidenceCitation` has explicit support metadata:

```json
{
  "evidenceFragmentId": "evidence-id",
  "artifactId": "artifact-id",
  "supports": [
    {
      "entityType": "concept",
      "entityId": "concept-id",
      "relationship": "supports_concept_definition"
    }
  ],
  "isValid": true,
  "validityErrors": []
}
```

Supported entity types:

```text
concept
concept_relation
decision_record
evidence_fragment
```

### Citation validity guardrails

The serving layer excludes invalid citations before they can support a response.

A citation is not eligible when:

```text
- EvidenceFragment is missing.
- EvidenceFragment is rejected.
- Artifact is missing.
- DataSource is missing.
- Provenance source identity is missing.
- Artifact/source provenance does not match.
- Source is non-production or unhealthy in production mode.
```

### OpenAPI snapshot

v0.11.0 adds an OpenAPI snapshot test for the grounded serving contract. This prevents accidental drift in the context response shape.

Snapshot file:

```text
tests/snapshots/openapi_v0_11_0_snapshot.json
```

Test file:

```text
tests/integration/test_openapi_contract.py
```

## API endpoints changed

### `GET /v1/context/query`

Changed response schema only. No URL or method change.

New response fields:

```text
responseId
query
generatedAt
officialAnswerAvailable
evidence[].supports
evidence[].isValid
evidence[].validityErrors
```

## Tests added

```text
test_related_reviewed_evidence_without_concept_is_evidence_supported
test_related_rejected_evidence_is_not_served_as_support
test_related_conflicted_evidence_returns_conflicted
test_official_concept_response_includes_official_relations_and_relation_citations
test_official_concept_response_includes_decisions_and_decision_citations
test_reviewed_evidence_without_concept_returns_evidence_supported
test_grounded_response_includes_relation_and_support_metadata
test_grounded_serving_openapi_contract_matches_snapshot
```

## Deferred

```text
- Full semantic retrieval/ranking.
- Evaluation runner for grounded_context_task_success_rate.
- Runtime vector retrieval from evidence_embeddings.
- MCP/Codex-specific transport wrappers.
- UI rendering of the grounded response.
```

## Acceptance criteria

```text
- Unsupported questions return unsupported without fabrication.
- Related evidence without a Concept returns evidence_supported or partially_supported, never official.
- Official answers include citations and freshness.
- Stale evidence downgrades trust to stale.
- Conflicted Concepts/evidence return conflicted.
- Rejected/invalid/non-production citations cannot support answers.
- ConceptRelation and DecisionRecord support are included in citations.
- OpenAPI snapshot protects the response contract.
```

# v2.2.0 — Review Operator Experience

`v2.2.0` adds Cornerstone backend and CLI surfaces that make pending ontology candidates easier to review without turning candidates into truth automatically.

## Version goal

Expose grouped review queue summaries and action previews for candidate approve, merge, and reject decisions.

## Confirmed non-goal

No endpoint approves, merges, rejects, or officializes a candidate without the existing explicit reviewer action.

## Runtime contract

- `GET /v1/ontology/review-queue/summary` groups pending ConceptCandidates and RelationCandidates by focus concept.
- `GET /v1/ontology/concept-candidates/{candidateId}/preview` reports whether an action can apply.
- `GET /v1/ontology/relation-candidates/{candidateId}/preview` reports unresolved endpoint blockers.
- `cornerstone review queue` and `cornerstone review preview` call the same backend contracts.

## Measurable acceptance checklist

| Check | Evidence | Status |
| --- | --- | --- |
| V220-01 | Queue summary reports counts by status, run, source, confidence, and focus concept. | complete |
| V220-02 | Preview reports `canApply`, blocker reasons, and mutation summary. | complete |
| V220-03 | Duplicate concept blockers are visible before approval. | complete |
| V220-04 | Relation endpoint blockers are visible before approval. | complete |
| V220-14 | CLI review helpers call the operator endpoints. | complete |

## Verification checklist

```bash
python -m pytest tests/integration/test_ontology_candidate_review_api.py tests/unit/test_cli.py -q
```

## Exit criteria

`v2.2.0` is complete when a reviewer can see the queue, preview mutations, and act only through the existing review gates.

## Next-version handoff

`v2.3.0` turns graph responses into a visualization-ready contract for downstream UI or integration consumers.

# v2.1.0 — Live LLM Ontology Provider

`v2.1.0` implements the live ontology extraction provider boundary while preserving the existing Cornerstone trust model: extractor output creates only `ConceptCandidate` and `RelationCandidate` records.

## Version goal

Add an explicitly gated `live_llm` provider for `POST /v1/ontology/extraction-runs`.

## Confirmed non-goal

No live provider output may write to the official graph. Official Concepts and ConceptRelations still require human review and officialization.

## Runtime contract

- `ONTOLOGY_LIVE_LLM_ENABLED=true` is required before `provider=live_llm` can run.
- `ONTOLOGY_LIVE_LLM_API_URL` is required for real calls unless `ONTOLOGY_LIVE_LLM_FIXTURE_RESPONSE_JSON` is provided for deterministic tests.
- Provider output must be strict JSON with `concepts` and `relations`.
- Every proposed concept and relation must cite EvidenceFragment ids already in the extraction scope.
- Self-relations are rejected before persistence.

## Measurable acceptance checklist

| Check | Evidence | Status |
| --- | --- | --- |
| V210-01 | `OntologyExtractionProvider.LIVE_LLM` is accepted by the API. | complete |
| V210-02 | Disabled live provider requests fail closed. | complete |
| V210-03 | Fixture-backed live provider output persists candidates only. | complete |
| V210-04 | Unknown evidence ids are rejected. | complete |
| V210-05 | Self-relations are rejected. | complete |
| V210-15 | Tests prove no official graph mutation occurs during extraction. | complete |

## Verification checklist

```bash
python -m pytest tests/integration/test_ontology_extraction_api.py -q
```

## Exit criteria

`v2.1.0` is complete when live extraction is configurable, strict, evidence-bound, and candidate-only.

## Next-version handoff

`v2.2.0` uses the candidate output to improve review-operator workflow without weakening review gates.

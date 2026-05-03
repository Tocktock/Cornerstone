# v2.4.0 — Connector Expansion and Live-Proof Hardening

`v2.4.0` exposes a Cornerstone connector support matrix so operators and downstream consumers can distinguish supported ingestion from discovery-only objects.

## Version goal

Make connector object support, live-proof guards, and secret-redaction rules machine-readable.

## Confirmed non-goal

Connector ingestion still creates evidence and candidate re-extraction work only. It never mutates the official ontology graph directly.

## Runtime contract

`GET /v1/connectors/support-matrix` returns:

- provider and object type
- support state and proof state
- whether evidence can be created
- whether candidate re-extraction can be queued
- `mutatesOfficialGraph=false` for every connector object
- live-proof environment guard names
- a secret-redaction policy for proof artifacts

## Measurable acceptance checklist

| Check | Evidence | Status |
| --- | --- | --- |
| V240-01 | Support matrix endpoint exists. | complete |
| V240-02 | Notion and Google Drive object states are represented. | complete |
| V240-03 | Official graph mutation is explicitly false. | complete |
| V240-04 | Secret-redaction policy is present. | complete |
| V240-13 | Connector support matrix has API regression coverage. | complete |

## Verification checklist

```bash
python -m pytest tests/integration/test_connector_support_matrix_api.py -q
```

## Exit criteria

`v2.4.0` is complete when connector capabilities are inspectable without reading provider code and without exposing secrets.

## Next-version handoff

`v2.5.0` packages the official graph and SSOT readiness contract for external consumers.

# v2.5.0 — External Integration Package

`v2.5.0` chooses the external integration package path from the roadmap. It does not ship a frontend MVP.

## Version goal

Let another product or agent consume Cornerstone's official ontology graph, citations, and SSOT readiness through a small stable package contract.

## Confirmed non-goal

No integration endpoint bypasses review gates, exposes pending candidates as official truth, or mutates the official graph.

## Runtime contract

- `GET /v1/integration/package/manifest` declares the chosen path, endpoints, trust boundary, and quickstart.
- `GET /v1/integration/ontology/{concept}` wraps official graph, SSOT readiness, trust state, citations, and unsupported state.
- `includeCandidates=true` is rejected with `409 Conflict`.
- `reviewGateBypassAllowed` is always false.

## Measurable acceptance checklist

| Check | Evidence | Status |
| --- | --- | --- |
| V250-01 | Manifest endpoint declares `external_integration_package`. | complete |
| V250-02 | Ontology integration endpoint wraps official graph and SSOT readiness. | complete |
| V250-03 | Candidate bypass attempts return 409. | complete |
| V250-04 | Trust boundary remains visible in response payloads. | complete |
| V250-13 | Integration package has API regression coverage. | complete |

## Verification checklist

```bash
python -m pytest tests/integration/test_integration_package_api.py -q
```

## Exit criteria

`v2.5.0` is complete when an external consumer can experience Cornerstone value without manually stitching graph, readiness, citation, and trust-boundary responses together.

## Next-version handoff

Future frontend work should consume this package contract instead of reinterpreting low-level ontology responses.

# v2.3.0 — Graph Visualization Contract

`v2.3.0` adds a visualization-ready payload to Cornerstone ontology graph responses while keeping graph serving read-only.

## Version goal

Make `GET /v1/ontology/graph` and `GET /v1/ontology/explain` directly consumable by graph renderers.

## Confirmed non-goal

No frontend graph UI is shipped in this release, and graph depth remains bounded to the supported depth.

## Runtime contract

`OntologyGraphResponse.visualization` contains:

- nodes with display labels, focus grouping, display state, citations, and review provenance panels
- edges with source/target ids, labels, focus-relative direction, display state, citations, and review provenance panels
- state legend, layout hints, graph-level citation panel, and graph-level review provenance panel
- explicit empty-state text when no graph is available

## Measurable acceptance checklist

| Check | Evidence | Status |
| --- | --- | --- |
| V230-01 | Official graph responses include visualization nodes and edges. | complete |
| V230-02 | Empty graph responses include visualization empty state and legend. | complete |
| V230-03 | Citation panels expose EvidenceFragment ids without raw secrets. | complete |
| V230-04 | Review provenance panels expose review metadata. | complete |
| V230-13 | Visualization contract is covered by graph tests. | complete |

## Verification checklist

```bash
python -m pytest tests/unit/test_ontology_graph_explainability.py -q
```

## Exit criteria

`v2.3.0` is complete when graph consumers can render official graph state without reverse-engineering low-level response fields.

## Next-version handoff

`v2.4.0` hardens connector support visibility and live-proof boundaries for source expansion.

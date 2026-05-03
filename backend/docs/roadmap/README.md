# Cornerstone Forward Roadmap

## Purpose

This folder contains the planning contracts for the next implementation releases after `v2.0.3`.

The roadmap is intentionally written before implementation so each release has a clear product goal, a confirmed non-goal, a domain boundary, measurable acceptance conditions, verification expectations, and a handoff to the next release.

The current roadmap sequence is:

```text
v2.1.0 — Live LLM ontology provider
v2.2.0 — Review operator experience
v2.3.0 — Graph visualization contract
v2.4.0 — Connector expansion / live connector proof hardening
v2.5.0 — Frontend MVP or external integration package
```

## Product boundary preserved

Every roadmap item must preserve the Cornerstone Single Source of Truth boundary unless a future ADR explicitly changes it:

```text
Raw source data is not the Single Source of Truth.
Connector sync output is not the Single Source of Truth.
LLM or extractor output is not the Single Source of Truth.
Pending candidates are not the Single Source of Truth.
The reviewed official ontology graph is the Single Source of Truth.
```

## Roadmap acceptance checklist

| Check ID | Measurable condition | Status |
|---|---|---|
| ROADMAP-README-01 | The README lists all planned releases from v2.1.0 through v2.5.0. | complete |
| ROADMAP-README-02 | Each planned release has a separate document. | complete |
| ROADMAP-README-03 | Each release document includes goal, non-goal, domain boundary, checklist, verification, and handoff. | complete |
| ROADMAP-README-04 | The roadmap preserves the candidate-only LLM/extractor boundary. | complete |
| ROADMAP-README-05 | The roadmap does not claim implementation completion for future releases. | complete |

## Document index

```text
docs/roadmap/v2.1.0-live-llm-ontology-provider.md
docs/roadmap/v2.2.0-review-operator-experience.md
docs/roadmap/v2.3.0-graph-visualization-contract.md
docs/roadmap/v2.4.0-connector-expansion-live-proof-hardening.md
docs/roadmap/v2.5.0-frontend-mvp-external-integration-package.md
```

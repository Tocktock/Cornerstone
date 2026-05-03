# v2.0.4 — Forward Roadmap Goals and Measurable Checklists

## Purpose

`v2.0.4` is a documentation-first planning release. It defines the goals, non-goals, measurable checklists, verification expectations, and handoffs for the next five implementation releases:

```text
v2.1.0 — Live LLM ontology provider
v2.2.0 — Review operator experience
v2.3.0 — Graph visualization contract
v2.4.0 — Connector expansion / live connector proof hardening
v2.5.0 — Frontend MVP or external integration package
```

The goal of this release is to make the future roadmap auditable before implementation begins.

## Version goal

Create separate roadmap documents for `v2.1.0` through `v2.5.0`, each with:

```text
- Version goal
- User value
- Confirmed non-goal
- Domain boundary
- Required behavior or decision criteria
- Measurable acceptance checklist
- Verification checklist
- Exit criteria
- Next-version handoff
```

## Confirmed non-goal

`v2.0.4` does not implement runtime behavior.

It does **not** add:

```text
- live LLM provider behavior
- review queue behavior
- graph visualization response fields
- connector expansion
- frontend UI
- integration package
- API endpoints
- database migrations
- graph behavior changes
- extraction behavior changes
- candidate review behavior changes
- official graph mutation behavior
```

The only non-documentation change is version metadata moving from `2.0.3` to `2.0.4`.

## Product boundary preserved

Every roadmap document must preserve the SSOT boundary:

```text
Raw source data is not the Single Source of Truth.
Connector sync output is not the Single Source of Truth.
LLM or extractor output is not the Single Source of Truth.
Pending candidates are not the Single Source of Truth.
The reviewed official ontology graph is the Single Source of Truth.
```

## Forward roadmap summary

| Version | Goal | Primary measurable gate | Non-goal |
|---|---|---|---|
| v2.1.0 | Add live LLM ontology provider. | Live provider creates validated pending candidates only. | No official graph mutation from LLM output. |
| v2.2.0 | Improve review operator experience. | Reviewers can list, group, preview, and safely act on candidates. | No frontend UI and no automatic approval. |
| v2.3.0 | Add graph visualization contract. | Graph response includes UI-ready display/citation/provenance metadata. | No frontend UI and no depth above 1. |
| v2.4.0 | Harden live connector proof and selected connector expansion. | Live connector proof and support matrix are measurable and safe. | No connector-created official truth. |
| v2.5.0 | Implement first user-facing surface through UI or integration package. | Chosen surface consumes official graph/readiness with trust boundary visible. | No automatic truth generation. |

## Measurable acceptance checklist

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V204-01 | Package/runtime version metadata reports `2.0.4`. | `pyproject.toml`, `src/cornerstone/__init__.py`, readiness schema. | complete |
| V204-02 | Main forward roadmap release doc exists. | `docs/52-forward-roadmap-goals-checklists-v2.0.4.md`. | complete |
| V204-03 | Roadmap index exists. | `docs/roadmap/README.md`. | complete |
| V204-04 | v2.1.0 live LLM provider roadmap doc exists. | `docs/roadmap/v2.1.0-live-llm-ontology-provider.md`. | complete |
| V204-05 | v2.2.0 review operator roadmap doc exists. | `docs/roadmap/v2.2.0-review-operator-experience.md`. | complete |
| V204-06 | v2.3.0 graph visualization roadmap doc exists. | `docs/roadmap/v2.3.0-graph-visualization-contract.md`. | complete |
| V204-07 | v2.4.0 connector hardening roadmap doc exists. | `docs/roadmap/v2.4.0-connector-expansion-live-proof-hardening.md`. | complete |
| V204-08 | v2.5.0 frontend/integration roadmap doc exists. | `docs/roadmap/v2.5.0-frontend-mvp-external-integration-package.md`. | complete |
| V204-09 | Every roadmap doc includes a version goal. | Roadmap docs. | complete |
| V204-10 | Every roadmap doc includes a confirmed non-goal. | Roadmap docs. | complete |
| V204-11 | Every roadmap doc includes measurable checklist IDs. | Roadmap docs. | complete |
| V204-12 | Release checker and documentation tests require roadmap docs. | `scripts/check_release_candidate.py`, `tests/unit/test_release_candidate_docs.py`. | complete |

## Version-specific checklist targets

### v2.1.0 — Live LLM ontology provider

Minimum checklist targets:

```text
V210-01 through V210-15
```

Core measurable outcome:

```text
A live provider can create validated pending ConceptCandidates and RelationCandidates, while the official graph remains unchanged until human review.
```

### v2.2.0 — Review operator experience

Minimum checklist targets:

```text
V220-01 through V220-14
```

Core measurable outcome:

```text
A reviewer can inspect, group, preview, and safely act on ontology candidates through API/CLI.
```

### v2.3.0 — Graph visualization contract

Minimum checklist targets:

```text
V230-01 through V230-13
```

Core measurable outcome:

```text
A frontend or integration client can render a depth-1 ontology graph without guessing display, direction, citation, provenance, or trust semantics.
```

### v2.4.0 — Connector expansion / live connector proof hardening

Minimum checklist targets:

```text
V240-01 through V240-13
```

Core measurable outcome:

```text
Connector sync and live proof paths can safely create evidence and queue candidate-only re-extraction without mutating the official graph.
```

### v2.5.0 — Frontend MVP or external integration package

Minimum checklist targets:

```text
V250-01 through V250-13
```

Core measurable outcome:

```text
Cornerstone value is consumable through either a frontend MVP or an external integration package, with official/candidate trust boundary visible.
```

## Documentation checklist

```text
[x] README points to the roadmap docs.
[x] Chronicle records v2.0.4.
[x] API freeze confirms no endpoint changes.
[x] Known limitations confirm roadmap-only scope.
[x] Release checker requires roadmap docs.
[x] Documentation tests verify roadmap sections and checklist IDs.
```

## Exit criteria

`v2.0.4` is complete when a maintainer can start `v2.1.0` implementation from the roadmap doc without asking what the goal, non-goal, measurable checklist, or handoff should be.

## Next-version handoff

The next implementation release should be:

```text
v2.1.0 — Live LLM ontology provider
```

`v2.1.0` should start from:

```text
docs/roadmap/v2.1.0-live-llm-ontology-provider.md
```

It should implement only the live provider boundary described there and must preserve candidate-only LLM output.

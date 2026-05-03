# v2.0.1 — Refactor and Domain Boundary Hardening

## Purpose

`v2.0.1` is a behavior-preserving maintenance release after the `v2.0.0` Ontology SSOT release. Its purpose is to make the codebase easier to read, audit, and extend before adding any future product features.

This release follows the documentation chronicle rule: every version must state its goal, non-goal, measurable checklist, verification source, and handoff.

## Version goal

Refactor shared logic into clearer domain/service boundaries while preserving the v2.0.0 ontology SSOT runtime contract.

The target outcomes are:

```text
- remove redundant code
- remove duplicated code
- improve readability
- clarify domain boundaries
- strengthen versioning checks
```

## Confirmed non-goal

`v2.0.1` does **not** add new product behavior.

Specifically, it does not add:

```text
- new API endpoints
- new database migrations
- new ontology graph behavior
- graph depth above 1
- live external LLM integration
- automatic candidate approval
- automatic official graph mutation
- frontend UI or graph visualization
```

## Product boundary preserved

The SSOT boundary remains unchanged:

```text
Raw source data is not the Single Source of Truth.
Connector sync output is not the Single Source of Truth.
Ontology extraction output is candidate-only.
Pending candidates are not the Single Source of Truth.
Reviewed official Concepts and ConceptRelations form the ontology Single Source of Truth.
```

## Refactor scope

### 1. Ontology domain policy boundary

Added:

```text
src/cornerstone/domain/__init__.py
src/cornerstone/domain/ontology.py
```

This domain module centralizes release-wide ontology policies:

```text
ONTOLOGY_GRAPH_MAX_DEPTH = 1
DEFAULT_ONTOLOGY_FOCUS_CONCEPT = "settlement"
CITABLE_TRUST_STATES
TERMINAL_CONCEPT_STATUSES
TERMINAL_RELATION_STATUSES
SSOT_TRUST_BOUNDARY
ensure_supported_graph_depth(...)
```

This prevents graph depth and trust-state policy from being redefined independently by graph serving, evaluation, and readiness services.

### 2. Evidence/provenance support boundary

Added:

```text
src/cornerstone/services/evidence_support.py
```

This module owns shared evidence/provenance support helpers:

```text
citation_validity_errors(...)
decision_record_or_none(...)
is_from_servable_source(...)
source_limitations_for(...)
```

Before this release, `grounded_context.py` and `ontology_graph.py` duplicated this logic. The duplicate citation-validation block also made it easier for future changes to diverge. `v2.0.1` consolidates that logic into one place.

### 3. Evaluation rule boundary

Added:

```text
src/cornerstone/services/evaluation_rules.py
```

This module owns shared evaluation rules:

```text
freshness_policy_respected(...)
citation_validity_rate(...)
```

Both grounded-context evaluation and ontology-graph evaluation now call the same rule implementation for freshness/trust consistency and citation validity rate.

### 4. Sync job policy boundary

Added:

```text
src/cornerstone/services/sync_jobs.py
```

This module owns shared sync job lease/claim policy:

```text
ensure_aware(...)
sync_job_is_claimable(...)
```

Before this release, the same claimability logic existed in both in-memory and SQLAlchemy stores. The refactor keeps store implementations aligned without changing their public methods.

### 5. Source selection boundary

Added:

```text
src/cornerstone/services/source_selection.py
```

This module owns the default source-selection rule used by sync runtime and sync worker flows:

```text
get_source_selection_for_sync(...)
```

This removes duplicated connector/sync selection fallback code.

### 6. Connector provider shared helpers

Added:

```text
src/cornerstone/connectors/providers/common.py
```

This module owns connector-provider helpers shared by Notion and Google Drive provider implementations:

```text
parse_datetime(...)
stable_hash(...)
connector_error_from_api_response(...)
```

This removes duplicated timestamp parsing, deterministic metadata hashing, and provider API-error bridging logic.

### 7. Verification helper boundary

Added:

```text
src/cornerstone/verification/env.py
```

This module owns live-proof environment parsing helpers:

```text
int_env(...)
```

This removes duplicated integer parsing code from Google Drive and Notion live proof scripts.

### 8. Schema validator readability

The evaluation task schema validators now delegate to shared helper functions:

```text
_validate_grounded_eval_model(...)
_validate_ontology_graph_eval_model(...)
```

This keeps persisted task models and create-request models aligned while making the contract validation easier to audit.

## Chronicle position and measurable release checklist

`v2.0.1` follows `v2.0.0` as a behavior-preserving refactor release. Its measurable checklist is the release gate below.

## Measurable acceptance checklist

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V201-01 | Package version, runtime version, readiness release version, release checker version, and test expectations all target `2.0.1`. | `pyproject.toml`, `src/cornerstone/__init__.py`, `src/cornerstone/schemas.py`, `scripts/check_release_candidate.py`, tests. | complete |
| V201-02 | Ontology graph depth policy is centralized in `cornerstone.domain.ontology` and used by graph serving, readiness, and evaluation. | Source inspection and compile/test results. | complete |
| V201-03 | Evidence/provenance helpers duplicated between grounded-context and ontology-graph serving are consolidated. | `src/cornerstone/services/evidence_support.py`; duplicate audit report. | complete |
| V201-04 | Freshness/citation evaluation rules are shared by grounded-context and ontology-graph evaluation. | `src/cornerstone/services/evaluation_rules.py`; regression tests. | complete |
| V201-05 | Sync job claimability logic is shared by in-memory and SQLAlchemy stores. | `src/cornerstone/services/sync_jobs.py`; compile checks. | complete |
| V201-06 | Connector timestamp/hash/API-error helper logic is shared by Google Drive and Notion providers. | `src/cornerstone/connectors/providers/common.py`; compile checks. | complete |
| V201-07 | Live-proof integer environment parsing is shared by Google Drive and Notion proof scripts. | `src/cornerstone/verification/env.py`; compile checks. | complete |
| V201-08 | Exact duplicate function-body audit has no remaining duplicates in application source except intentional schema validators below the audit threshold. | `reports/refactor-duplicate-audit-v2.0.1.txt`. | complete |
| V201-09 | No API endpoint, migration, graph behavior, or SSOT trust-boundary change is introduced. | API freeze note, docs, and regression tests. | complete |
| V201-10 | The version chronicle records `v2.0.1` with goal, non-goal, measurable checklist, and handoff. | `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. | complete |

## Implementation checklist

```text
[x] Add ontology domain policy module.
[x] Move evidence/provenance support helpers into one service module.
[x] Move shared evaluation rules into one service module.
[x] Move sync job claimability policy into one service module.
[x] Move source-selection fallback into one service module.
[x] Move connector provider shared helpers into one provider-common module.
[x] Move live-proof env parsing helper into one verification module.
[x] Delegate duplicated schema validator bodies to helper functions.
[x] Update package/runtime/readiness version to 2.0.1.
[x] Update release checker and docs tests for 2.0.1.
[x] Update chronicle and release notes.
```

## Verification plan

Required verification for this refactor release:

```bash
python -m compileall -q src tests scripts
python scripts/check_release_candidate.py
python -m pytest tests/unit/test_release_candidate_docs.py tests/unit/test_refactor_boundaries_v2_0_1.py -q
python -m pytest tests/integration/test_ontology_proof_api.py tests/integration/test_ontology_ssot_readiness_api.py tests/integration/test_ontology_api.py tests/integration/test_ontology_evaluations_api.py -q -k 'not sqlalchemy and not persistent'
```

Environment-specific checks to rerun in a fully provisioned dev/CI environment:

```text
- SQLAlchemy persistent-store tests
- Alembic offline SQL generation
- Ruff
- mypy
```

## Known limitations

`v2.0.1` is intentionally a patch-level refactor. It improves maintainability but does not prove live connectors, persistent PostgreSQL behavior, or UI/UX.

## Chronicle handoff

`v2.0.1` hands off a cleaner backend to any future post-SSOT release. The recommended next work is still product-specific and should not start until a new version document defines its goal, non-goal, and measurable checklist.

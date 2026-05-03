# v0.12.1 — Product Trust Cleanup Before Live Proof

v0.12.1 is a focused cleanup patch before manual live execution proof.

The goal is to keep the backend aligned with the product promise:

```text
Real source → Artifact → EvidenceFragment → review → official context → grounded serving → evaluation
```

## Changes

### 1. Removed fake provider source creation

`POST /v1/sources` now creates manual sources only.

Provider-backed sources such as Notion must be created through:

```text
POST /v1/connections/intents
GET  /v1/oauth/{provider}/callback
```

This removes the fake path where a provider-backed source could appear connected without a real connection intent, OAuth callback, and credential.

### 2. Removed fake OAuth completion route

This route is gone:

```text
POST /v1/sources/{sourceId}/oauth/complete
```

OAuth completion is handled only by the provider callback route.

### 3. Split officialization eligibility from serving eligibility

Officialization remains strict:

```text
production source + connected/healthy source + reviewed fresh/aging evidence
```

Serving is intentionally less strict:

```text
previously captured reviewed evidence may still be served if the source later becomes degraded, failed, stale, or disconnected
```

The response must include limitations when historical evidence is served from a degraded source.

### 4. Strengthened evaluation task validation

Evaluation tasks must define at least one explicit success condition, such as:

```text
expectedTrustLabel
expectedAnswerContains
requiredEvidenceFragmentIds
requiredConceptIds
requiredDecisionRecordIds
requireOfficialAnswer=true
```

Unsupported tasks must be explicit:

```text
expectedTrustLabel=unsupported
requireEvidence=false
minEvidenceCount=0
```

This protects `grounded_context_task_success_rate` from vague or metric-gaming tasks.

### 5. Added evaluation OpenAPI snapshot coverage

The evaluation API now has OpenAPI snapshot protection for evaluation paths and schema fields.

### 6. Targeted route cleanup

Connector catalog endpoints now live in `api/routes/connector_catalog.py`, separate from the heavier connector runtime routes. This is the first maintainability split before larger schema/store refactors.

### 7. Release package hygiene

Release packaging excludes:

```text
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
```

## Exit Criteria

```text
- Provider-backed sources cannot be created directly.
- Fake OAuth completion is removed.
- Officialization remains strict.
- Grounded serving can still show historical reviewed evidence with limitations.
- Evaluation task definitions are meaningful.
- Evaluation OpenAPI surface is protected.
- Release ZIP contains no Python cache files.
- Connector catalog route module is separated from connector runtime routes.
```

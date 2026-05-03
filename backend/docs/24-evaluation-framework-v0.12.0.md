# v0.12.0 — Evaluation Framework and grounded_context_task_success_rate

## Purpose

v0.12.0 adds the backend evaluation layer required by the Cornerstone PRD.

The goal is to measure whether grounded context responses are useful **and** trustworthy. A response only succeeds when it is correct, evidence-backed, provenance-visible, trust-labeled correctly, freshness-aware, and free from unsupported official claims.

```text
GroundedContextResponse
→ GroundedContextEvalTask
→ GroundedContextEvalResult
→ grounded_context_task_success_rate
```

## Product rule

Evaluation is not a generic answer-quality score.

A grounded context task succeeds only when all of the following pass:

```text
answerCorrect
AND evidenceValid
AND provenancePresent
AND trustLabelCorrect
AND freshnessPolicyRespected
AND no unsupportedOfficialClaim
```

Optional reviewer/business signal:

```text
clarificationReduced
```

## API

### Create evaluation task

```http
POST /v1/evaluations/tasks
```

Request:

```json
{
  "name": "Evaluate official definition for Project Atlas",
  "query": "What is Project Atlas?",
  "expectedAnswerContains": ["customer onboarding"],
  "expectedTrustLabel": "official",
  "expectedFreshnessState": "fresh",
  "requiredEvidenceFragmentIds": ["evidence-id"],
  "requiredConceptIds": ["concept-id"],
  "requiredDecisionRecordIds": [],
  "requireOfficialAnswer": true,
  "requireEvidence": true,
  "minEvidenceCount": 1,
  "expectedClarificationReduced": true,
  "tags": ["pilot", "notion"],
  "createdBy": "reviewer@example.com"
}
```

### List tasks

```http
GET /v1/evaluations/tasks
```

Optional tag filter:

```http
GET /v1/evaluations/tasks?tag=pilot
```

### Run one task

```http
POST /v1/evaluations/tasks/{task_id}/run
```

### Run many tasks

```http
POST /v1/evaluations/run
```

Request:

```json
{
  "taskIds": ["task-id-1", "task-id-2"],
  "evaluatedBy": "qa@example.com"
}
```

If `taskIds` is omitted, all tasks are run.

### List results

```http
GET /v1/evaluations/results
```

Optional task filter:

```http
GET /v1/evaluations/results?taskId=task-id
```

### Metric summary

```http
GET /v1/evaluations/summary
```

Response includes:

```text
grounded_context_task_success_rate
provenance_coverage_rate
citation_validity_rate
freshness_compliance_rate
trust_label_correctness_rate
unsupported_answer_correctness_rate
```

## Persistence

v0.12.0 adds Alembic migration:

```text
0011_grounded_context_evaluation
```

New tables:

```text
grounded_context_eval_tasks
grounded_context_eval_results
```

The result table stores a JSONB snapshot of the evaluated `GroundedContextResponse` so historical evaluation results remain auditable even if current Concepts/Evidence change later.

## Evaluation checks

| Check | Meaning |
|---|---|
| `answerCorrect` | Expected answer substrings are present when configured. |
| `evidenceValid` | Required evidence/concept/decision support is present and citations are valid. |
| `provenancePresent` | Served evidence has valid provenance. |
| `trustLabelCorrect` | The response trust label matches the expected label when configured. |
| `freshnessPolicyRespected` | Freshness state and trust label are compatible. |
| `unsupportedOfficialClaim` | The system did not label unsupported/non-official context as official. |
| `citationValidityRate` | Percentage of served citations marked valid. |
| `clarificationReduced` | Optional task-level signal for downstream business impact. |

## Safety behavior

The API rejects inconsistent task definitions. For example:

```text
expectedTrustLabel=unsupported
AND requireEvidence=true
```

is invalid because unsupported tasks should not require evidence-backed answers.

## Tests added

Coverage includes:

```text
- Unsupported tasks can succeed when unsupported is expected.
- Official claims without official support fail evaluation.
- Expected answer substrings are checked.
- Required evidence/concept/decision IDs are validated.
- Evaluation summary computes grounded_context_task_success_rate.
- Evaluation task/result persistence works in memory and SQLAlchemy stores.
- API supports task creation, run-one, run-many, results listing, and summary.
- Invalid evaluation task definitions are rejected.
- Alembic migration creates task/result tables and indexes.
```

## Known limitations

```text
- Evaluation correctness is rule-based, not LLM-graded.
- Clarification reduction is captured as an optional task signal, not yet measured from Slack/user behavior.
- Live PostgreSQL and live Notion E2E tests remain gated and were not run in this sandbox.
- Evaluation is scoped to the current grounded serving contract; vector retrieval/ranking evaluation is deferred.
```

## Exit criteria

v0.12.0 is complete when:

```text
1. Evaluation tasks can be created.
2. Grounded context responses can be evaluated.
3. Results are persisted.
4. Summary metrics are exposed.
5. grounded_context_task_success_rate is computed.
6. Tests, lint, typing, compile, and migration checks pass.
```

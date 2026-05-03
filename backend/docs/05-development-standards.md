# 05 — Development Standards

## Runtime baseline

- Target Python: `>=3.13,<3.15`.
- API framework: FastAPI.
- Data validation: Pydantic v2.
- Settings: pydantic-settings.
- Persistence plan: SQLAlchemy 2.x, Alembic, PostgreSQL, asyncpg.
- Tests: pytest, coverage.

## Dependency policy

Use lower-bound pins for current stable releases and upper bounds for major-version safety.

Examples:

```toml
fastapi = ">=0.136.1,<1.0.0"
pydantic = ">=2.13.3,<3.0.0"
SQLAlchemy = ">=2.0.49,<2.1.0"
pytest = ">=9.0.3,<10.0.0"
coverage = ">=7.13.0,<8.0.0"
```

Rationale:

- Lower bounds align new environments with current stable releases.
- Upper bounds avoid accidental breaking major upgrades.
- Deployment tooling should create lockfiles for reproducibility.

## API style

- Versioned path prefix: `/v1`.
- JSON field style: lower camel case.
- Python field style: snake_case.
- Use Pydantic aliases to bridge them.
- Response bodies should expose trust, provenance, freshness, and limitations where relevant.

## Error style

| Code | Use |
| --- | --- |
| `403` | Actor is not authorized for reviewer/officialization action. |
| `404` | Entity not found. |
| `409` | Domain rule conflict, such as unsupported officialization. |
| `422` | Request validation error. |
| `500` | Unexpected server failure only. |

## Testing standards

- Write or update docs before backend behavior changes.
- Write or update tests before implementation changes where practical.
- Every P0 domain rule must have a unit or integration test.
- Every route must have at least one success-path integration test.
- Every quality gate must have a failure-path test.
- Test reports must show test names, not only pytest dots.
- Runtime events should be logged as single-line JSON.
- Important log event names should be asserted in tests with `caplog`.
- Review and officialization actions must create audit events.
- Coverage should remain above 85%; current v0.2 coverage is 91%.

## Observability standards

Structured logs use stable event names and lower camel case fields.

Minimum event families:

| Event family | Required fields |
| --- | --- |
| `source.*` | `sourceId` when available, `sourceType`, status/reason/count fields. |
| `artifact.*` | `artifactId`, `sourceId`, `sourceExternalId`, evidence count. |
| `evidence.*` | `evidenceFragmentId`, actor, trust state. |
| `decision_record.*` | `decisionRecordId`, actor, evidence count. |
| `concept.*` | `conceptId`, actor, status, support counts, reason on blocked paths. |
| `context.*` | trust label, freshness state, concept/evidence counts. |
| `http.request.*` | request ID, method, path, status or error type, duration. |

## Audit standards

Audit events should be created for trust-changing actions:

```text
evidence.reviewed
decision_record.created
concept.officialization_blocked
concept.officialized
```

Future persistence work must make audit event creation transactional with the state change.

## Security standards

- Do not log OAuth tokens, access tokens, private document bodies, or full query payloads once real connectors are added.
- v0.2 reviewer allow-list is a placeholder, not full RBAC.
- Real connector implementation must encrypt or externalize token storage.
- Production mode must exclude demo/non-production evidence from officialization and serving.

## Naming standards

- `DataSource`, not `Connector`, for persisted source runtime state.
- `Artifact` for captured source object snapshot.
- `EvidenceFragment` for supportable extracted evidence.
- `Concept` for semantic unit.
- `DecisionRecord` for explicit decision history.
- `GroundedContextResponse` for shared human/AI serving contract.

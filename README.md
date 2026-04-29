# Cornerstone

Cornerstone is a backend-first system for building an evidence-backed organizational context layer. The current repository state contains the backend MVP release package under `cornerstone-backend/`.

This is not the old full-stack prototype. The current v3 direction starts from the backend release artifact first, then layers product surfaces and additional connectors on top after the backend trust loop is proven.

## Current Release

- Package: `cornerstone-backend`
- Version: `1.0.0`
- Runtime: Python `>=3.13,<3.15`
- Framework: FastAPI
- Persistence: in-memory for contract tests, PostgreSQL for live/runtime proof
- Primary connector in MVP scope: Notion pages
- Additional ingestion path: controlled manual sources

The backend MVP is considered meaningful only when this loop is proven:

```text
Live PostgreSQL
-> live Notion page
-> Artifact
-> EvidenceFragment
-> evidence review
-> official Concept
-> grounded context response
-> evaluation result
-> grounded_context_task_success_rate
```

## Repository Layout

```text
cornerstone-backend/
  README.md                         Backend package README
  pyproject.toml                    Python package metadata and tool config
  src/cornerstone/                  FastAPI app, services, stores, connectors
  migrations/                       Alembic migrations
  tests/                            Unit, integration, PostgreSQL, live Notion tests
  scripts/                          Test, release, worker, and live-proof helpers
  docs/                             Architecture, API, release, and proof records
  reports/                          Sanitized release/test reports shipped with the artifact
```

Important documents:

- `cornerstone-backend/README.md`
- `cornerstone-backend/docs/00-backend-architecture.md`
- `cornerstone-backend/docs/01-api-contract.md`
- `cornerstone-backend/docs/release/backend-release-checklist.md`
- `cornerstone-backend/docs/release/backend-operator-runbook.md`
- `cornerstone-backend/docs/release/v1.0.0-readiness.md`
- `cornerstone-backend/docs/release/v1.0.0-release-notes.md`
- `cornerstone-backend/docs/release/known-limitations.md`
- `cornerstone-backend/docs/release/secrets-and-credential-handling.md`

## Quickstart

Run commands from the backend package directory:

```bash
cd cornerstone-backend
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
./scripts/run_tests.sh
```

Start the local API with the default in-memory backend:

```bash
cd cornerstone-backend
source .venv/bin/activate
uvicorn cornerstone.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

## PostgreSQL Runtime

Start local PostgreSQL with pgvector support:

```bash
cd cornerstone-backend
docker compose up -d postgres
```

Run migrations and extension checks:

```bash
cd cornerstone-backend
source .venv/bin/activate
export PERSISTENCE_BACKEND=postgres
export DATABASE_URL=postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone
alembic upgrade head
python scripts/check_postgres_extensions.py
```

Start the API against PostgreSQL:

```bash
uvicorn cornerstone.main:app --reload
```

## Configuration

Use `cornerstone-backend/.env.example` as the local reference. Local defaults are intentionally non-production:

- `PERSISTENCE_BACKEND=memory` for fast local contract tests
- `NOTION_MOCK_EXTERNAL_API=true` for deterministic connector tests
- `PRODUCTION_MODE=false`
- `CONNECTOR_ENCRYPTION_SECRET=local-dev-only-change-me-secret`

Before enabling production mode, replace all local defaults with deployment-safe values. Production startup fails closed when unsafe persistence, connector, secret, OAuth, reviewer, or PostgreSQL extension settings are detected.

Do not commit `.env` files or paste provider tokens into docs, tickets, logs, screenshots, or chat.

## Verification

Fast local gate:

```bash
cd cornerstone-backend
source .venv/bin/activate
./scripts/run_tests.sh
python -m mypy src --show-error-codes --no-color-output --no-incremental
python scripts/check_release_candidate.py
```

Live PostgreSQL proof:

```bash
cd cornerstone-backend
source .venv/bin/activate
export RUN_POSTGRES_TESTS=1
export DATABASE_URL=postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone
python scripts/run_live_postgres_tests.py --min-passed 5
```

Live Notion proof requires real credentials supplied through environment variables only:

```bash
cd cornerstone-backend
source .venv/bin/activate
export RUN_NOTION_E2E=1
export NOTION_MOCK_EXTERNAL_API=false
export NOTION_E2E_ACCESS_TOKEN='<set in shell or secret manager>'
export NOTION_E2E_PAGE_ID='<shared test page id>'
export CONNECTOR_ENCRYPTION_SECRET='<long non-default secret>'
PYTHONPATH=src python scripts/run_live_notion_e2e.py
```

See `cornerstone-backend/docs/release/backend-release-checklist.md` before tagging or deploying a release.

## MVP Scope

Included in backend v1.0.0:

- FastAPI `/v1` API routes and `/healthz`
- PostgreSQL persistence and Alembic migrations
- Notion page connector path
- Manual source ingestion path
- Artifact and EvidenceFragment creation
- Evidence review queue
- Concept, ConceptRelation, and DecisionRecord officialization gates
- Grounded context responses with trust, freshness, limitations, and citations
- Evaluation tasks/results and `grounded_context_task_success_rate`
- Release runbooks, live proof template, production checklist, and known limitations

Intentionally deferred:

- Frontend UI
- Notion database/data_source ingestion
- Slack, Google Docs, and GitHub connectors
- Runtime vector retrieval/ranking
- Enterprise SSO/RBAC
- Notion webhooks and full incremental provider cursor integration
- LLM-graded evaluation
- Production KMS/secret-manager integration, unless adopted by the deployment owner

## Release Notes

The shipped package promotes the verified `v1.0.0-rc.1` / `v0.13.1` line to backend `v1.0.0`.

Release and proof references:

- `cornerstone-backend/docs/29-backend-v1.0.0.md`
- `cornerstone-backend/docs/release/v1.0.0-release-notes.md`
- `cornerstone-backend/docs/release/v1.0.0-readiness.md`
- `cornerstone-backend/docs/live-proof-records/2026-04-28-v0.13.1-blocker-fix.md`

The release artifact should remain free of local runtime artifacts such as `.venv`, `.env`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.coverage`, `__pycache__`, and `.pyc` files.

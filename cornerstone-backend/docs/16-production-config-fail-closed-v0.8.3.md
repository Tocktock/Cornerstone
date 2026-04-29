# v0.8.3 — Production Config Fail-Closed Patch

## Goal

v0.8.3 fixes the third pre-v0.9.0 safety issue: production mode could previously run with local/test defaults such as in-memory persistence, mocked Notion API access, placeholder reviewer identities, localhost OAuth callback URLs, and a development connector encryption secret.

This violates the Cornerstone trust model. In production mode, the backend must not silently operate with settings that could create fake source confidence, lose persistence, store connector credentials with a known secret, or run provider integrations in mock mode.

## Product rule

```text
Local development should be easy.
Production mode must fail closed.
```

## Runtime policy

Local defaults now use:

```text
PRODUCTION_MODE=false
PERSISTENCE_BACKEND=memory
NOTION_MOCK_EXTERNAL_API=true
CONNECTOR_ENCRYPTION_SECRET=local-dev-only-change-me-secret
```

Those defaults are allowed only because production mode is disabled.

When `PRODUCTION_MODE=true`, the app and external sync worker call the same runtime config guard before starting.

## Required production settings

When `PRODUCTION_MODE=true`, all of the following must be true:

```text
APP_ENV is not local/dev/development/test
PERSISTENCE_BACKEND=postgres
DATABASE_URL is not localhost / 127.0.0.1 / default cornerstone credentials
POSTGRES_REQUIRED_EXTENSIONS_RAW includes pgcrypto,citext,vector
VERIFY_POSTGRES_EXTENSIONS_ON_STARTUP=true
CONNECTOR_ENCRYPTION_SECRET is not the local placeholder
CONNECTOR_ENCRYPTION_SECRET is at least 32 characters
NOTION_MOCK_EXTERNAL_API=false
NOTION_CLIENT_ID is not the local placeholder
NOTION_CLIENT_SECRET is not the local placeholder
CONNECTOR_OAUTH_CALLBACK_URL is HTTPS
CONNECTOR_OAUTH_CALLBACK_URL is not localhost / 127.0.0.1 / 0.0.0.0
AUTHORIZED_REVIEWERS_RAW has at least one real reviewer identity
AUTHORIZED_REVIEWERS_RAW does not include system/example placeholders
```

## New code

```text
src/cornerstone/config.py
  RuntimeConfigIssue
  RuntimeConfigError
  Settings.runtime_config_issues()
  Settings.assert_runtime_config_safe()

src/cornerstone/main.py
  create_app(..., validate_runtime_config: bool | None = None)

src/cornerstone/workers/sync_worker.py
  _run_cli() calls settings.assert_runtime_config_safe()
```

## App factory behavior

```python
create_app()
```

Validates runtime config by default because it represents normal process startup.

```python
create_app(store=test_store, settings=test_settings)
```

Skips runtime validation by default because explicit store injection is used for deterministic tests.

```python
create_app(settings=bad_settings, validate_runtime_config=True)
```

Forces validation and raises `RuntimeConfigError` if production mode is unsafe.

## Why test-store bypass exists

API contract tests intentionally inject an in-memory or SQLite store. That should not force production PostgreSQL credentials into every test run.

This bypass is explicit and local to app factories that receive a store object. Normal app/worker process startup still validates runtime config.

## Production startup example

```bash
export APP_ENV=production
export PRODUCTION_MODE=true
export PERSISTENCE_BACKEND=postgres
export DATABASE_URL=postgresql+psycopg://svc_cornerstone:strong-password@postgres.internal:5432/cornerstone
export CONNECTOR_ENCRYPTION_SECRET='replace-with-a-long-production-secret'
export CONNECTOR_OAUTH_CALLBACK_URL=https://cornerstone.example.com/v1/oauth/notion/callback
export NOTION_MOCK_EXTERNAL_API=false
export NOTION_CLIENT_ID='<real-notion-client-id>'
export NOTION_CLIENT_SECRET='<real-notion-client-secret>'
export AUTHORIZED_REVIEWERS_RAW='reviewer@company.internal,ops@company.internal'

uvicorn cornerstone.main:app
```

## Tests added

```text
test_local_default_runtime_config_is_allowed_for_development
test_production_mode_reports_all_unsafe_default_config_values
test_safe_production_runtime_config_has_no_issues
test_create_app_fails_closed_when_production_config_is_unsafe
test_create_app_can_bypass_runtime_validation_for_explicit_test_store
test_worker_cli_fails_closed_when_production_env_is_unsafe
test_production_rejects_missing_required_postgres_extension
```

## Acceptance criteria

```text
Given PRODUCTION_MODE=false
When the app or worker starts locally
Then local/test defaults are allowed
```

```text
Given PRODUCTION_MODE=true
And PERSISTENCE_BACKEND=memory
When the API or worker starts
Then startup is blocked with RuntimeConfigError
```

```text
Given PRODUCTION_MODE=true
And NOTION_MOCK_EXTERNAL_API=true
When the API or worker starts
Then startup is blocked with RuntimeConfigError
```

```text
Given PRODUCTION_MODE=true
And CONNECTOR_ENCRYPTION_SECRET uses the default local placeholder
When the API or worker starts
Then startup is blocked with RuntimeConfigError
```

```text
Given PRODUCTION_MODE=true
And the OAuth callback URL is localhost or HTTP
When the API or worker starts
Then startup is blocked with RuntimeConfigError
```

## Remaining pre-v0.9.0 risks

```text
1. Worker/scheduler multi-process locking is not implemented yet.
2. Sync writes are not yet wrapped in one service-level transaction across Artifact/Evidence/cursor/job/source updates.
3. Live PostgreSQL execution and concurrency tests still need v0.9.0 hardening.
4. Production KMS/secret-manager integration is still deferred.
```

# Production Deployment Checklist

## Runtime mode

```text
[ ] APP_ENV=production
[ ] PRODUCTION_MODE=true
[ ] PERSISTENCE_BACKEND=postgres
[ ] NOTION_MOCK_EXTERNAL_API=false
```

## Database

```text
[ ] DATABASE_URL points to production PostgreSQL, not localhost.
[ ] PostgreSQL user is not the local default `cornerstone` user.
[ ] Alembic upgrade head succeeds.
[ ] pgcrypto extension exists.
[ ] citext extension exists.
[ ] vector extension exists.
[ ] VERIFY_POSTGRES_EXTENSIONS_ON_STARTUP=true.
[ ] Database backups are configured.
[ ] Migration rollback/restore plan exists.
```

## Workers

```text
[ ] At least one sync worker process is deployed.
[ ] Worker uses a stable workerId.
[ ] Worker logs sync.job_claimed, sync.job_succeeded, sync.job_failed, and sync.job_retry_waiting events.
[ ] LeaseSeconds is configured intentionally.
[ ] Scheduler trigger cadence is defined.
```

## Source credentials

```text
[ ] CONNECTOR_ENCRYPTION_SECRET is not a placeholder.
[ ] CONNECTOR_ENCRYPTION_SECRET is at least 32 characters.
[ ] Notion client ID/secret are real provider credentials.
[ ] OAuth callback URL is HTTPS.
[ ] OAuth callback URL is not localhost.
[ ] Tokens are never logged.
```

## Reviewers

```text
[ ] AUTHORIZED_REVIEWERS_RAW contains real reviewer identities.
[ ] Placeholder reviewer identities are not present.
[ ] Reviewer process is documented.
[ ] Evidence review and officialization audit events are visible.
```

## Observability

```text
[ ] Logs are structured JSON.
[ ] Request IDs are propagated.
[ ] Source sync success/failure events are observable.
[ ] Evaluation results are observable.
[ ] Secret redaction policy is in place.
```

## Release gates

```text
[ ] Local package checks pass.
[ ] Live PostgreSQL proof passes.
[ ] Live Notion proof passes.
[ ] Product loop proof passes.
[ ] Safety negative checks pass.
[ ] Known limitations are accepted.
```


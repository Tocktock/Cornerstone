# Cornerstone Proof Report

- Version: `1.1.2`
- Generated at: `2026-04-29T05:32:35.842023Z`
- Status: **planned**
- Dry run: `True`
- Base URL: `http://localhost:8000`
- Query: `What is Cornerstone?`

## Summary

| Passed | Failed | Planned | Skipped | Total |
|---:|---:|---:|---:|---:|
| 0 | 0 | 15 | 0 | 15 |

## Category Summary

| Category | Passed | Failed | Planned | Skipped | Total |
|---|---:|---:|---:|---:|---:|
| live_notion | 0 | 0 | 1 | 0 | 1 |
| live_postgres | 0 | 0 | 1 | 0 | 1 |
| product_loop | 0 | 0 | 7 | 0 | 7 |
| safety | 0 | 0 | 5 | 0 | 5 |
| secret_scan | 0 | 0 | 1 | 0 | 1 |

## Steps

### live_postgres

- Category: `live_postgres`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `/opt/pyvenv/bin/python scripts/run_live_postgres_tests.py --min-passed 5`

```text
Dry run; command was not executed.
```

### live_notion_e2e

- Category: `live_notion`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `/opt/pyvenv/bin/python scripts/run_live_notion_e2e.py`

```text
Dry run; command was not executed.
```

### product_healthz

- Category: `product_loop`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `GET http://localhost:8000/healthz`

```text
Dry run; request was not sent.
```

### product_real_notion_source_present

- Category: `product_loop`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `GET http://localhost:8000/v1/sources`

```text
Dry run; request was not sent.
```

### product_artifact_present

- Category: `product_loop`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `GET http://localhost:8000/v1/artifacts`

```text
Dry run; request was not sent.
```

### product_evidence_present

- Category: `product_loop`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `GET http://localhost:8000/v1/evidence`

```text
Dry run; request was not sent.
```

### product_grounded_official_with_valid_citation

- Category: `product_loop`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `GET http://localhost:8000/v1/context/query?q=What+is+Cornerstone%3F`

```text
Dry run; request was not sent.
```

### product_unsupported_query_unsupported

- Category: `product_loop`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `GET http://localhost:8000/v1/context/query?q=What+is+the+company+policy+for+interplanetary+travel%3F`

```text
Dry run; request was not sent.
```

### product_evaluation_summary_success_rate

- Category: `product_loop`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `GET http://localhost:8000/v1/evaluations/summary`

```text
Dry run; request was not sent.
```

### safety_direct_notion_source_409

- Category: `safety`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `POST http://localhost:8000/v1/sources`

```text
Dry run; request was not sent.
```

### safety_fake_oauth_completion_404

- Category: `safety`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `POST http://localhost:8000/v1/sources/<notion-source-id>/oauth/complete`

```text
Dry run; request was not sent.
```

### safety_legacy_source_sync_404

- Category: `safety`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `POST http://localhost:8000/v1/sources/<notion-source-id>/sync`

```text
Dry run; request was not sent.
```

### safety_manual_sync_on_notion_409

- Category: `safety`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `POST http://localhost:8000/v1/manual-sources/<notion-source-id>/sync`

```text
Dry run; request was not sent.
```

### safety_weak_evaluation_task_422

- Category: `safety`
- Status: `planned`
- Duration: `0.0` seconds
- Command: `POST http://localhost:8000/v1/evaluations/tasks`

```text
Dry run; request was not sent.
```

### secret_scan

- Category: `secret_scan`
- Status: `planned`
- Duration: `0.0` seconds

```text
Dry run; token scan was not executed.
```

from __future__ import annotations

import argparse

from ..config import resolve_base_url
from ..models import Column
from ..support import _api_url, _http_json, _http_request, _print_json, _print_payload, _print_table, _rows_from_sources, _source_columns, _job_columns, _job_rows


def command_source_list(args: argparse.Namespace) -> int:
    payload = _http_json(_api_url(resolve_base_url(args.base_url), "/v1/sources"), timeout=args.timeout)
    rows = _rows_from_sources(payload)
    _print_payload(payload, json_output=args.json, table_rows=rows, columns=_source_columns())
    return 0

def command_source_show(args: argparse.Namespace) -> int:
    payload = _http_json(_api_url(resolve_base_url(args.base_url), f"/v1/sources/{args.source_id}"), timeout=args.timeout)
    if args.json:
        _print_json(payload)
    else:
        rows = _rows_from_sources({"sources": [payload]})
        _print_table(rows, _source_columns())
        if payload.get("lastError"):
            print()
            print(f"Last error: {payload['lastError']}")
    return 0

def command_source_objects(args: argparse.Namespace) -> int:
    payload = _http_json(_api_url(resolve_base_url(args.base_url), f"/v1/sources/{args.source_id}/objects"), timeout=args.timeout)
    if args.json:
        _print_json(payload)
        return 0
    rows = []
    for obj in payload.get("objects", []):
        rows.append(
            {
                "id": obj.get("externalId", ""),
                "type": obj.get("objectType", ""),
                "access": obj.get("accessState", ""),
                "ingestible": obj.get("ingestionSupported", ""),
                "selected": obj.get("selectedForSync", ""),
                "title": obj.get("title", ""),
            }
        )
    print(f"Discovered: {payload.get('totalCount', 0)} | Syncable: {payload.get('syncableCount', 0)}")
    _print_table(
        rows,
        [
            Column("id", "EXTERNAL ID", 20),
            Column("type", "TYPE", 14),
            Column("access", "ACCESS", 12),
            Column("ingestible", "INGEST", 7),
            Column("selected", "SELECTED", 8),
            Column("title", "TITLE", 44),
        ],
    )
    return 0

def command_source_jobs(args: argparse.Namespace) -> int:
    payload = _http_json(_api_url(resolve_base_url(args.base_url), f"/v1/sources/{args.source_id}/sync-jobs"), timeout=args.timeout)
    _print_payload(payload, json_output=args.json, table_rows=_job_rows(payload), columns=_job_columns())
    return 0

def command_source_sync(args: argparse.Namespace) -> int:
    body = {"createdBy": args.created_by, "runInline": args.inline, "maxAttempts": args.max_attempts}
    payload = _http_request("POST", _api_url(resolve_base_url(args.base_url), f"/v1/sources/{args.source_id}/sync-jobs"), body=body, timeout=args.timeout)
    if args.json:
        _print_json(payload)
        return 0
    job = payload.get("job", payload)
    print(f"Sync job {job.get('id')} status={job.get('status')} trigger={job.get('trigger')}")
    if not args.inline:
        _print_next_action(f"Run `cornerstone worker --once --run-scheduler --max-jobs 1` or `cornerstone source jobs {args.source_id}` to track progress.")
    return 0

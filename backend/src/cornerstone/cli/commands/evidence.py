from __future__ import annotations

import argparse
import urllib.parse

from ..config import resolve_base_url
from ..support import _api_url, _evidence_columns, _evidence_rows, _http_json, _http_request, _print_json, _print_payload


def _review_evidence(args: argparse.Namespace, trust_state: str) -> int:
    body = {"trustState": trust_state, "reviewedBy": args.reviewer, "reviewNote": args.note}
    payload = _http_request("POST", _api_url(resolve_base_url(args.base_url), f"/v1/evidence/{args.evidence_id}/review"), body=body, timeout=args.timeout)
    if args.json:
        _print_json(payload)
    else:
        print(f"Evidence {payload.get('id')} marked {payload.get('trustState')} by {payload.get('reviewedBy')}.")
    return 0

def command_evidence_queue(args: argparse.Namespace) -> int:
    query: dict[str, str | int] = {"limit": args.limit}
    if args.trust_state is not None:
        query["trustState"] = args.trust_state
    if args.source_id is not None:
        query["dataSourceId"] = args.source_id
    if args.freshness_state is not None:
        query["freshnessState"] = args.freshness_state
    if args.fragment_type is not None:
        query["fragmentType"] = args.fragment_type
    qs = urllib.parse.urlencode(query)
    payload = _http_json(_api_url(resolve_base_url(args.base_url), f"/v1/evidence/review-queue?{qs}"), timeout=args.timeout)
    if args.json:
        _print_json(payload)
        return 0
    print(
        f"Review queue: total={payload.get('totalCount', 0)} "
        f"unreviewed={payload.get('unreviewedCount', 0)} reviewed={payload.get('reviewedCount', 0)} "
        f"rejected={payload.get('rejectedCount', 0)} conflicted={payload.get('conflictedCount', 0)}"
    )
    _print_table(_evidence_rows(payload), _evidence_columns())
    return 0

def command_evidence_show(args: argparse.Namespace) -> int:
    payload = _http_json(_api_url(resolve_base_url(args.base_url), "/v1/evidence"), timeout=args.timeout)
    match = next((item for item in payload if item.get("id") == args.evidence_id), None)
    if match is None:
        raise RuntimeError(f"EvidenceFragment not found in /v1/evidence: {args.evidence_id}")
    _print_json(match)
    return 0

def command_evidence_review(args: argparse.Namespace) -> int:
    return _review_evidence(args, "reviewed")

def command_evidence_reject(args: argparse.Namespace) -> int:
    return _review_evidence(args, "rejected")

def command_evidence_conflict(args: argparse.Namespace) -> int:
    return _review_evidence(args, "conflicted")

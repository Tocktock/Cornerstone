from __future__ import annotations

import argparse
from typing import Any

from ..config import resolve_base_url
from ..support import _api_url, _concept_columns, _concept_rows, _http_json, _http_request, _print_json, _print_next_action, _print_payload, _print_table


def command_concept_list(args: argparse.Namespace) -> int:
    payload = _http_json(_api_url(resolve_base_url(args.base_url), "/v1/concepts"), timeout=args.timeout)
    if args.status:
        payload = [concept for concept in payload if concept.get("status") == args.status]
    _print_payload(payload, json_output=args.json, table_rows=_concept_rows(payload), columns=_concept_columns())
    return 0

def command_concept_show(args: argparse.Namespace) -> int:
    payload = _http_json(_api_url(resolve_base_url(args.base_url), f"/v1/concepts/{args.concept_id}"), timeout=args.timeout)
    if args.json:
        _print_json(payload)
    else:
        _print_table(_concept_rows([payload]), _concept_columns())
    return 0

def command_concept_create_from_evidence(args: argparse.Namespace) -> int:
    body = {
        "name": args.name,
        "shortDefinition": args.definition,
        "body": args.body,
        "owner": args.owner,
        "createdBy": args.created_by,
    }
    payload = _http_request("POST", _api_url(resolve_base_url(args.base_url), f"/v1/evidence/{args.evidence_id}/concept-candidates"), body=body, timeout=args.timeout)
    if args.json:
        _print_json(payload)
    else:
        print(f"Concept {payload.get('id')} created: {payload.get('name')} ({payload.get('status')})")
        _print_next_action(f"Officialize after review: cornerstone concept officialize {payload.get('id')} --reviewer {args.created_by}")
    return 0

def command_concept_officialize(args: argparse.Namespace) -> int:
    payload = _http_request("POST", _api_url(resolve_base_url(args.base_url), f"/v1/concepts/{args.concept_id}/officialize"), body={"reviewedBy": args.reviewer}, timeout=args.timeout)
    if args.json:
        _print_json(payload)
    else:
        print(f"Concept {payload.get('id')} is now {payload.get('status')}.")
    return 0

from __future__ import annotations

import argparse

from ..config import resolve_base_url
from ..support import _api_url, _http_json, _print_json, _print_payload, _print_table


def command_review_queue(args: argparse.Namespace) -> int:
    params: list[str] = []
    if args.status:
        params.append(f"status={args.status}")
    if args.run_id:
        params.append(f"runId={args.run_id}")
    if args.source_id:
        params.append(f"sourceId={args.source_id}")
    suffix = "?" + "&".join(params) if params else ""
    payload = _http_json(
        _api_url(resolve_base_url(args.base_url), f"/v1/ontology/review-queue/summary{suffix}"),
        timeout=args.timeout,
    )
    if args.json:
        _print_json(payload)
    else:
        rows = [
            {
                "focus": group.get("focusConcept"),
                "concepts": group.get("pendingConceptCandidateCount"),
                "relations": group.get("pendingRelationCandidateCount"),
                "blocked": group.get("blockedCount"),
            }
            for group in payload.get("groupedByFocusConcept", [])
        ]
        _print_table(rows, [
            _Column("focus", "FOCUS", 28),
            _Column("concepts", "CONCEPTS", 8),
            _Column("relations", "RELATIONS", 9),
            _Column("blocked", "BLOCKED", 8),
        ])
    return 0


def command_review_preview(args: argparse.Namespace) -> int:
    candidate_path = "concept-candidates" if args.candidate_type == "concept" else "relation-candidates"
    query = f"action={args.action}"
    if args.target_id:
        query += "&targetConceptId=" + args.target_id if args.candidate_type == "concept" else "&targetRelationId=" + args.target_id
    payload = _http_json(
        _api_url(resolve_base_url(args.base_url), f"/v1/ontology/{candidate_path}/{args.candidate_id}/preview?{query}"),
        timeout=args.timeout,
    )
    _print_payload(payload, json_output=args.json)
    return 0


class _Column:
    def __init__(self, key: str, label: str, width: int) -> None:
        self.key = key
        self.label = label
        self.width = width

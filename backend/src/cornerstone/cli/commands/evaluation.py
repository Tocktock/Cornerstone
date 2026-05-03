from __future__ import annotations

import argparse
import urllib.parse
from typing import Any

from ..config import resolve_base_url
from ..models import Column
from ..support import _api_url, _http_json, _http_request, _print_json, _print_next_action, _print_table


def command_eval_summary(args: argparse.Namespace) -> int:
    payload = _http_json(_api_url(resolve_base_url(args.base_url), "/v1/evaluations/summary"), timeout=args.timeout)
    if args.json:
        _print_json(payload)
        return 0
    rows = [
        {"metric": "grounded_context_task_success_rate", "value": payload.get("groundedContextTaskSuccessRate")},
        {"metric": "provenance_coverage_rate", "value": payload.get("provenanceCoverageRate")},
        {"metric": "citation_validity_rate", "value": payload.get("citationValidityRate")},
        {"metric": "freshness_compliance_rate", "value": payload.get("freshnessComplianceRate")},
        {"metric": "trust_label_correctness_rate", "value": payload.get("trustLabelCorrectnessRate")},
        {"metric": "unsupported_answer_correctness_rate", "value": payload.get("unsupportedAnswerCorrectnessRate")},
    ]
    _print_table(rows, [Column("metric", "METRIC", 44), Column("value", "VALUE", 10)])
    return 0

def command_eval_create(args: argparse.Namespace) -> int:
    body: dict[str, Any] = {
        "name": args.name,
        "query": args.query,
        "expectedAnswerContains": args.expected_answer_contains,
        "expectedTrustLabel": args.expected_trust_label,
        "requiredEvidenceFragmentIds": args.required_evidence,
        "requiredConceptIds": args.required_concept,
        "requiredDecisionRecordIds": args.required_decision,
        "requireOfficialAnswer": args.require_official_answer,
        "requireEvidence": not args.no_evidence_required,
        "minEvidenceCount": args.min_evidence_count,
        "tags": args.tag,
        "createdBy": args.created_by,
    }
    body = {key: value for key, value in body.items() if value not in (None, [], "")}
    payload = _http_request("POST", _api_url(resolve_base_url(args.base_url), "/v1/evaluations/tasks"), body=body, timeout=args.timeout)
    if args.json:
        _print_json(payload)
    else:
        print(f"Evaluation task {payload.get('id')} created: {payload.get('name')}")
        _print_next_action(f"Run it: cornerstone eval run {payload.get('id')}")
    return 0

def command_eval_run(args: argparse.Namespace) -> int:
    payload = _http_request("POST", _api_url(resolve_base_url(args.base_url), f"/v1/evaluations/tasks/{args.task_id}/run"), body={"evaluatedBy": args.evaluated_by}, timeout=args.timeout)
    if args.json:
        _print_json(payload)
        return 0
    print(f"Evaluation result {payload.get('id')} success={payload.get('success')}")
    failures = payload.get("failureReasons", [])
    if failures:
        print("Failure reasons:")
        for reason in failures:
            print(f"- {reason}")
    return 0

def command_eval_results(args: argparse.Namespace) -> int:
    query = ""
    if args.task_id:
        query = "?" + urllib.parse.urlencode({"taskId": args.task_id})
    payload = _http_json(_api_url(resolve_base_url(args.base_url), f"/v1/evaluations/results{query}"), timeout=args.timeout)
    if args.json:
        _print_json(payload)
        return 0
    rows = [
        {
            "id": result.get("id", ""),
            "task": result.get("taskId", ""),
            "success": result.get("success", ""),
            "trust": result.get("trustLabel", ""),
            "evaluated": result.get("evaluatedAt", ""),
        }
        for result in payload
    ]
    _print_table(rows, [Column("id", "ID", 12), Column("task", "TASK", 12), Column("success", "SUCCESS", 8), Column("trust", "TRUST", 14), Column("evaluated", "EVALUATED", 24)])
    return 0

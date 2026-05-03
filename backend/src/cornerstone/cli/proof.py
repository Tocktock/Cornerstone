from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from cornerstone import __version__

from .config import resolve_base_url
from .models import DEFAULT_TIMEOUT, ProofCheck
from .support import _api_url, _http_status, _print_json, _project_root

def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")

def _proof_result(
    *,
    name: str,
    status: str,
    detail: str,
    category: str = "general",
    command: str | None = None,
    exit_code: int | None = None,
    start: float | None = None,
    payload: Any | None = None,
) -> ProofCheck:
    duration = 0.0 if start is None else round(time.monotonic() - start, 3)
    return ProofCheck(
        name=name,
        status=status,
        detail=detail,
        category=category,
        command=command,
        exit_code=exit_code,
        duration_seconds=duration,
        payload=payload,
    )

def _run_proof_command(
    name: str,
    command: Sequence[str],
    *,
    cwd: Path,
    category: str,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> ProofCheck:
    command_text = " ".join(str(part) for part in command)
    if dry_run:
        return _proof_result(
            name=name,
            category=category,
            status="planned",
            detail="Dry run; command was not executed.",
            command=command_text,
        )
    print(f"RUN  {name}: {command_text}")
    start = time.monotonic()
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, **(env or {})},
    )
    combined_output = "\n".join(
        part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
    )
    status = "passed" if completed.returncode == 0 else "failed"
    detail = combined_output[-4000:] if combined_output else f"exit code {completed.returncode}"
    return _proof_result(
        name=name,
        category=category,
        status=status,
        detail=detail,
        command=command_text,
        exit_code=completed.returncode,
        start=start,
    )

def _proof_http_check(
    name: str,
    method: str,
    url: str,
    *,
    category: str,
    expected_status: int = 200,
    body: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    dry_run: bool = False,
    validator: Any | None = None,
) -> ProofCheck:
    command_text = f"{method.upper()} {url}"
    if dry_run:
        return _proof_result(
            name=name,
            category=category,
            status="planned",
            detail="Dry run; request was not sent.",
            command=command_text,
        )
    start = time.monotonic()
    status_code, payload = _http_status(method, url, body=body, timeout=timeout)
    ok = status_code == expected_status
    detail = f"HTTP {status_code}; expected {expected_status}"
    if ok and validator is not None:
        ok, validator_detail = validator(payload)
        detail = str(validator_detail)
    elif not ok:
        detail = f"{detail}; payload={payload}"
    return _proof_result(
        name=name,
        category=category,
        status="passed" if ok else "failed",
        detail=detail,
        command=command_text,
        exit_code=0 if ok else status_code,
        start=start,
        payload=payload,
    )

def _find_first_notion_source(sources_payload: Any) -> str | None:
    if not isinstance(sources_payload, dict):
        return None
    for source in sources_payload.get("sources", []):
        if source.get("type") == "notion":
            return str(source.get("id"))
    return None

def _proof_product_loop_checks(
    base_url: str,
    *,
    timeout: float,
    query: str,
    dry_run: bool,
) -> list[ProofCheck]:
    checks: list[ProofCheck] = []
    category = "product_loop"
    checks.append(
        _proof_http_check(
            "product_healthz",
            "GET",
            _api_url(base_url, "/healthz"),
            category=category,
            timeout=timeout,
            dry_run=dry_run,
            validator=lambda payload: (
                isinstance(payload, dict) and payload.get("status") == "ok",
                f"status={payload.get('status') if isinstance(payload, dict) else payload}",
            ),
        )
    )
    checks.append(
        _proof_http_check(
            "product_real_notion_source_present",
            "GET",
            _api_url(base_url, "/v1/sources"),
            category=category,
            timeout=timeout,
            dry_run=dry_run,
            validator=lambda payload: (
                isinstance(payload, dict)
                and bool(payload.get("hasRealSources"))
                and _find_first_notion_source(payload) is not None,
                (
                    f"hasRealSources={payload.get('hasRealSources') if isinstance(payload, dict) else None}; "
                    f"notionSource={_find_first_notion_source(payload)}"
                ),
            ),
        )
    )
    checks.append(
        _proof_http_check(
            "product_artifact_present",
            "GET",
            _api_url(base_url, "/v1/artifacts"),
            category=category,
            timeout=timeout,
            dry_run=dry_run,
            validator=lambda payload: (
                isinstance(payload, list) and len(payload) >= 1,
                f"artifactCount={len(payload) if isinstance(payload, list) else 'n/a'}",
            ),
        )
    )
    checks.append(
        _proof_http_check(
            "product_evidence_present",
            "GET",
            _api_url(base_url, "/v1/evidence"),
            category=category,
            timeout=timeout,
            dry_run=dry_run,
            validator=lambda payload: (
                isinstance(payload, list) and len(payload) >= 1,
                f"evidenceFragmentCount={len(payload) if isinstance(payload, list) else 'n/a'}",
            ),
        )
    )
    context_url = _api_url(base_url, f"/v1/context/query?{urllib.parse.urlencode({'q': query})}")
    checks.append(
        _proof_http_check(
            "product_grounded_official_with_valid_citation",
            "GET",
            context_url,
            category=category,
            timeout=timeout,
            dry_run=dry_run,
            validator=lambda payload: (
                isinstance(payload, dict)
                and payload.get("trustLabel") == "official"
                and bool(payload.get("officialAnswerAvailable"))
                and len(payload.get("evidence", [])) >= 1
                and not [citation for citation in payload.get("evidence", []) if not citation.get("isValid")],
                (
                    f"trustLabel={payload.get('trustLabel') if isinstance(payload, dict) else None}; "
                    f"evidence={len(payload.get('evidence', [])) if isinstance(payload, dict) else 'n/a'}"
                ),
            ),
        )
    )
    unsupported_url = _api_url(
        base_url,
        f"/v1/context/query?{urllib.parse.urlencode({'q': 'What is the company policy for interplanetary travel?'})}",
    )
    checks.append(
        _proof_http_check(
            "product_unsupported_query_unsupported",
            "GET",
            unsupported_url,
            category=category,
            timeout=timeout,
            dry_run=dry_run,
            validator=lambda payload: (
                isinstance(payload, dict) and payload.get("trustLabel") == "unsupported",
                f"trustLabel={payload.get('trustLabel') if isinstance(payload, dict) else None}",
            ),
        )
    )
    checks.append(
        _proof_http_check(
            "product_evaluation_summary_success_rate",
            "GET",
            _api_url(base_url, "/v1/evaluations/summary"),
            category=category,
            timeout=timeout,
            dry_run=dry_run,
            validator=lambda payload: (
                isinstance(payload, dict)
                and float(payload.get("groundedContextTaskSuccessRate") or 0.0) >= 1.0,
                (
                    "groundedContextTaskSuccessRate="
                    f"{payload.get('groundedContextTaskSuccessRate') if isinstance(payload, dict) else None}"
                ),
            ),
        )
    )
    return checks

def _proof_ontology_loop_check(
    base_url: str,
    *,
    timeout: float,
    dry_run: bool,
    focus_concept: str,
    reviewer: str,
    confirm_mutation: bool,
) -> ProofCheck:
    return _proof_http_check(
        "ontology_operator_proof_loop",
        "POST",
        _api_url(base_url, "/v1/ontology/proof-runs"),
        category="ontology_loop",
        expected_status=201,
        body={
            "focusConcept": focus_concept,
            "reviewer": reviewer,
            "createdBy": "cli-proof",
            "dryRun": False,
            "confirmMutation": confirm_mutation,
            "runEvaluation": True,
        },
        timeout=timeout,
        dry_run=dry_run,
        validator=lambda payload: (
            isinstance(payload, dict)
            and payload.get("status") == "passed"
            and payload.get("summary", {}).get("officialGraphAvailable") is True
            and payload.get("summary", {}).get("requiredFailed") == 0
            and bool(payload.get("approvedConceptIds"))
            and bool(payload.get("approvedRelationIds"))
            and payload.get("evaluationResultId") is not None,
            (
                f"status={payload.get('status') if isinstance(payload, dict) else None}; "
                f"officialGraphAvailable={payload.get('summary', {}).get('officialGraphAvailable') if isinstance(payload, dict) else None}; "
                f"requiredFailed={payload.get('summary', {}).get('requiredFailed') if isinstance(payload, dict) else None}; "
                f"approvedConcepts={len(payload.get('approvedConceptIds', [])) if isinstance(payload, dict) else 'n/a'}; "
                f"approvedRelations={len(payload.get('approvedRelationIds', [])) if isinstance(payload, dict) else 'n/a'}"
            ),
        ),
    )


def _proof_ssot_readiness_check(
    base_url: str,
    *,
    timeout: float,
    dry_run: bool,
    focus_concept: str,
) -> ProofCheck:
    category = "ssot_readiness"
    url = _api_url(
        base_url,
        f"/v1/ontology/ssot/readiness?{urllib.parse.urlencode({'focusConcept': focus_concept})}",
    )
    if dry_run:
        return _proof_result(
            name="ontology_ssot_readiness_checklist",
            category=category,
            status="planned",
            detail=(
                "Dry run; request was not sent. The v2.0.0 SSOT readiness checklist would verify "
                "source/evidence presence, official graph safety, citation validity, review provenance, "
                "candidate boundary, and ontology evaluation availability."
            ),
        )

    def _validate(payload: dict[str, Any]) -> tuple[bool, str]:
        checks = payload.get("checks", []) if isinstance(payload, dict) else []
        failed = sum(1 for item in checks if item.get("required") and item.get("status") != "passed")
        status_value = payload.get("status") if isinstance(payload, dict) else None
        return (
            isinstance(payload, dict)
            and status_value == "passed"
            and bool(payload.get("officialGraphAvailable"))
            and bool(payload.get("officialGraphSafe"))
            and failed == 0,
            (
                f"status={status_value}; "
                f"officialGraphAvailable={payload.get('officialGraphAvailable') if isinstance(payload, dict) else None}; "
                f"officialGraphSafe={payload.get('officialGraphSafe') if isinstance(payload, dict) else None}; "
                f"requiredFailed={failed}"
            ),
        )

    return _proof_http_check(
        "ontology_ssot_readiness_checklist",
        "GET",
        url,
        category=category,
        expected_status=200,
        timeout=timeout,
        dry_run=False,
        validator=_validate,
    )


def _proof_safety_checks(base_url: str, *, timeout: float, dry_run: bool) -> list[ProofCheck]:
    category = "safety"
    if dry_run:
        source_id = "<notion-source-id>"
        source_payload: Any = None
    else:
        status_code, source_payload = _http_status("GET", _api_url(base_url, "/v1/sources"), timeout=timeout)
        source_id = _find_first_notion_source(source_payload) if status_code == 200 else None
    if not source_id:
        return [
            _proof_result(
                name="safety_find_notion_source",
                category=category,
                status="failed",
                detail=f"No Notion source found in /v1/sources: {source_payload}",
            )
        ]
    return [
        _proof_http_check(
            "safety_direct_notion_source_409",
            "POST",
            _api_url(base_url, "/v1/sources"),
            category=category,
            expected_status=409,
            body={"type": "notion", "name": "Fake Notion", "productionEnabled": True},
            timeout=timeout,
            dry_run=dry_run,
        ),
        _proof_http_check(
            "safety_fake_oauth_completion_404",
            "POST",
            _api_url(base_url, f"/v1/sources/{source_id}/oauth/complete"),
            category=category,
            expected_status=404,
            timeout=timeout,
            dry_run=dry_run,
        ),
        _proof_http_check(
            "safety_legacy_source_sync_404",
            "POST",
            _api_url(base_url, f"/v1/sources/{source_id}/sync"),
            category=category,
            expected_status=404,
            body={"objects": []},
            timeout=timeout,
            dry_run=dry_run,
        ),
        _proof_http_check(
            "safety_manual_sync_on_notion_409",
            "POST",
            _api_url(base_url, f"/v1/manual-sources/{source_id}/sync"),
            category=category,
            expected_status=409,
            body={
                "objects": [
                    {
                        "sourceExternalId": "fake",
                        "title": "Fake",
                        "content": "Fake content",
                    }
                ]
            },
            timeout=timeout,
            dry_run=dry_run,
        ),
        _proof_http_check(
            "safety_weak_evaluation_task_422",
            "POST",
            _api_url(base_url, "/v1/evaluations/tasks"),
            category=category,
            expected_status=422,
            body={
                "name": "Weak task should fail",
                "query": "What is Cornerstone?",
                "requireEvidence": False,
                "minEvidenceCount": 0,
                "createdBy": "cli-proof",
            },
            timeout=timeout,
            dry_run=dry_run,
        ),
    ]

def _token_scan_paths(root: Path) -> list[Path]:
    return [
        root / "README.md",
        root / "docs",
        root / "scripts",
        root / "src",
        root / "tests",
        root / "reports",
        root / "pyproject.toml",
        root / ".env.example",
    ]

def _scan_for_notion_tokens(root: Path) -> list[str]:
    token_prefix = "nt" + "n_"
    findings: list[str] = []
    ignored_parts = {".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__"}
    ignored_suffixes = {".pyc", ".pyo", ".zip", ".png", ".jpg", ".jpeg", ".xml"}
    for base in _token_scan_paths(root):
        if not base.exists():
            continue
        files = [base] if base.is_file() else [path for path in base.rglob("*") if path.is_file()]
        for path in files:
            if any(part in ignored_parts for part in path.parts) or path.suffix in ignored_suffixes:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if token_prefix in line:
                    findings.append(f"{path.relative_to(root)}:{line_number}")
    return findings

def _proof_secret_scan(root: Path, *, dry_run: bool) -> ProofCheck:
    if dry_run:
        return _proof_result(
            name="secret_scan",
            category="secret_scan",
            status="planned",
            detail="Dry run; token scan was not executed.",
        )
    start = time.monotonic()
    findings = _scan_for_notion_tokens(root)
    if findings:
        return _proof_result(
            name="secret_scan",
            category="secret_scan",
            status="failed",
            detail="possible Notion token pattern found",
            start=start,
            payload={"findings": findings},
        )
    return _proof_result(
        name="secret_scan",
        category="secret_scan",
        status="passed",
        detail="No Notion token pattern found.",
        start=start,
    )

def _summarize_categories(steps: list[ProofCheck]) -> dict[str, dict[str, int]]:
    categories: dict[str, dict[str, int]] = {}
    for step in steps:
        bucket = categories.setdefault(step.category, {"passed": 0, "failed": 0, "planned": 0, "skipped": 0, "total": 0})
        bucket[step.status] = bucket.get(step.status, 0) + 1
        bucket["total"] += 1
    return categories

def _print_proof_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print()
    print(
        "Proof summary: "
        f"{summary['status'].upper()} — "
        f"passed={summary['passed']} failed={summary['failed']} "
        f"planned={summary['planned']} skipped={summary['skipped']} total={summary['total']}"
    )
    print()
    for category, category_summary in report.get("categories", {}).items():
        print(
            f"{category}: passed={category_summary.get('passed', 0)} "
            f"failed={category_summary.get('failed', 0)} "
            f"planned={category_summary.get('planned', 0)} "
            f"skipped={category_summary.get('skipped', 0)}"
        )
    print()
    for step in report["steps"]:
        marker = {"passed": "PASS", "failed": "FAIL", "planned": "PLAN", "skipped": "SKIP"}.get(
            step["status"], str(step["status"]).upper()
        )
        first_line = str(step.get("detail", "")).splitlines()[0] if step.get("detail") else ""
        print(f"{marker:4} [{step.get('category', 'general')}] {step['name']}: {first_line}")

def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Cornerstone Proof Report",
        "",
        f"- Version: `{report['version']}`",
        f"- Generated at: `{report['generatedAt']}`",
        f"- Status: **{summary['status']}**",
        f"- Dry run: `{report['dryRun']}`",
        f"- Base URL: `{report['baseUrl']}`",
        f"- Query: `{report['query']}`",
        "",
        "## Summary",
        "",
        "| Passed | Failed | Planned | Skipped | Total |",
        "|---:|---:|---:|---:|---:|",
        f"| {summary['passed']} | {summary['failed']} | {summary['planned']} | {summary['skipped']} | {summary['total']} |",
        "",
        "## Category Summary",
        "",
        "| Category | Passed | Failed | Planned | Skipped | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, values in sorted(report.get("categories", {}).items()):
        lines.append(
            f"| {category} | {values.get('passed', 0)} | {values.get('failed', 0)} | "
            f"{values.get('planned', 0)} | {values.get('skipped', 0)} | {values.get('total', 0)} |"
        )
    lines.extend(["", "## Steps", ""])
    for step in report["steps"]:
        lines.extend(
            [
                f"### {step['name']}",
                "",
                f"- Category: `{step.get('category', 'general')}`",
                f"- Status: `{step['status']}`",
                f"- Duration: `{step.get('duration_seconds', 0.0)}` seconds",
            ]
        )
        if step.get("command"):
            lines.append(f"- Command: `{step['command']}`")
        lines.extend(["", "```text", str(step.get("detail", ""))[:4000], "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")

def command_proof_run(args: argparse.Namespace) -> int:
    root = _project_root()
    base_url = resolve_base_url(getattr(args, "base_url", None)).rstrip("/")
    run_all = args.all or not any(
        [
            args.local,
            args.local_tests,
            args.postgres,
            args.notion,
            args.product_loop,
            args.ontology_loop,
            args.ssot_readiness,
            args.safety_checks,
            args.secret_scan,
        ]
    )
    include_local = args.local or run_all
    include_local_tests = args.local_tests or run_all
    include_postgres = args.postgres or run_all
    include_notion = args.notion or run_all
    include_product = args.product_loop or run_all
    include_ontology_loop = args.ontology_loop
    include_ssot_readiness = args.ssot_readiness or run_all
    include_safety = args.safety_checks or run_all
    include_secret = args.secret_scan or run_all

    started_at = _utc_now()
    steps: list[ProofCheck] = []

    if include_local:
        steps.append(
            _run_proof_command(
                "local_doctor",
                [sys.executable, "-m", "cornerstone.cli", "doctor", "--json"],
                cwd=root,
                category="local",
                dry_run=args.dry_run,
            )
        )
        steps.append(
            _run_proof_command(
                "local_release_candidate_check",
                [sys.executable, "scripts/check_release_candidate.py"],
                cwd=root,
                category="local",
                dry_run=args.dry_run,
            )
        )

    if include_local_tests:
        steps.append(
            _run_proof_command(
                "local_test_gate",
                ["./scripts/run_tests.sh"],
                cwd=root,
                category="local_tests",
                dry_run=args.dry_run,
            )
        )
        steps.append(
            _run_proof_command(
                "local_mypy_gate",
                [
                    sys.executable,
                    "-m",
                    "mypy",
                    "src",
                    "--show-error-codes",
                    "--no-color-output",
                    "--no-incremental",
                ],
                cwd=root,
                category="local_tests",
                dry_run=args.dry_run,
            )
        )

    if include_postgres:
        steps.append(
            _run_proof_command(
                "live_postgres",
                [
                    sys.executable,
                    "scripts/run_live_postgres_tests.py",
                    "--min-passed",
                    str(args.min_passed),
                ],
                cwd=root,
                category="live_postgres",
                dry_run=args.dry_run,
            )
        )

    if include_notion:
        steps.append(
            _run_proof_command(
                "live_notion_e2e",
                [sys.executable, "scripts/run_live_notion_e2e.py"],
                cwd=root,
                category="live_notion",
                dry_run=args.dry_run,
            )
        )

    if include_product:
        steps.extend(
            _proof_product_loop_checks(
                base_url,
                timeout=args.timeout,
                query=args.query,
                dry_run=args.dry_run,
            )
        )

    if include_ontology_loop:
        steps.append(
            _proof_ontology_loop_check(
                base_url,
                timeout=args.timeout,
                dry_run=args.dry_run,
                focus_concept=args.ontology_focus_concept,
                reviewer=args.reviewer,
                confirm_mutation=args.confirm_ontology_mutation,
            )
        )

    if include_ssot_readiness:
        steps.append(
            _proof_ssot_readiness_check(
                base_url,
                timeout=args.timeout,
                dry_run=args.dry_run,
                focus_concept=args.ontology_focus_concept,
            )
        )

    if include_safety:
        steps.extend(_proof_safety_checks(base_url, timeout=args.timeout, dry_run=args.dry_run))

    if include_secret:
        steps.append(_proof_secret_scan(root, dry_run=args.dry_run))

    if not args.continue_on_failure:
        trimmed: list[ProofCheck] = []
        failed_seen = False
        for step in steps:
            if failed_seen:
                trimmed.append(
                    _proof_result(
                        name=step.name,
                        category=step.category,
                        status="skipped",
                        detail="Skipped because a prior step failed and --continue-on-failure was not set.",
                        command=step.command,
                    )
                )
            else:
                trimmed.append(step)
                failed_seen = step.status == "failed"
        steps = trimmed

    passed = sum(1 for step in steps if step.status == "passed")
    failed = sum(1 for step in steps if step.status == "failed")
    planned = sum(1 for step in steps if step.status == "planned")
    skipped = sum(1 for step in steps if step.status == "skipped")
    status = "planned" if args.dry_run else ("passed" if failed == 0 else "failed")
    report: dict[str, Any] = {
        "schemaVersion": 2,
        "version": __version__,
        "generatedAt": _utc_now(),
        "startedAt": started_at,
        "finishedAt": _utc_now(),
        "dryRun": args.dry_run,
        "baseUrl": base_url,
        "query": args.query,
        "scope": {
            "local": include_local,
            "localTests": include_local_tests,
            "postgres": include_postgres,
            "notion": include_notion,
            "productLoop": include_product,
            "ontologyLoop": include_ontology_loop,
            "ssotReadiness": include_ssot_readiness,
            "safetyChecks": include_safety,
            "secretScan": include_secret,
        },
        "summary": {
            "status": status,
            "passed": passed,
            "failed": failed,
            "planned": planned,
            "skipped": skipped,
            "total": len(steps),
        },
        "categories": _summarize_categories(steps),
        "steps": [asdict(step) for step in steps],
    }

    json_path: Path | None = None
    if not args.no_save:
        json_path = (
            Path(args.save)
            if args.save
            else root
            / "reports"
            / f"cornerstone-proof-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        report["reportPath"] = str(json_path)

    markdown_path: Path | None = None
    if args.markdown:
        if args.markdown == "auto":
            if json_path is None:
                markdown_path = root / "reports" / f"cornerstone-proof-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.md"
            else:
                markdown_path = json_path.with_suffix(".md")
        else:
            markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown_report(markdown_path, report)
        report["markdownReportPath"] = str(markdown_path)

    if args.json:
        _print_json(report)
    else:
        _print_proof_summary(report)
        if json_path and json_path.exists():
            print(f"Saved JSON proof report to {json_path}")
        if markdown_path and markdown_path.exists():
            print(f"Saved Markdown proof report to {markdown_path}")

    return 0 if failed == 0 else 1

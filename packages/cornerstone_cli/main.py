from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from cornerstone_cli import __version__
from cornerstone_cli.scenarios import (
    coverage_report,
    list_scenarios,
    verify_vs0_security,
    verify_vs0_artifacts,
    verify_vs0_fixtures,
    verify_vs0_scaffold,
)
from cornerstone_cli.runtime import LocalRuntimeStore


SCHEMA_VERSION = "cs.cli.v0"
EXIT_SUCCESS = 0
EXIT_INVALID = 1
EXIT_NOT_FOUND = 3
EXIT_EVIDENCE_MISSING = 4
EXIT_RUNTIME_FAILURE = 5


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def base_response(command: str, status: str, root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "status": status,
        "product": "CornerStone",
        "version": __version__,
        "cli_schema_version": SCHEMA_VERSION,
        "mode": "local_scaffold",
        "tenant_id": "local-dev",
        "owner_id": "local-user",
        "namespace_id": "personal",
        "workspace_id": "default",
        "ids": {"git_commit": git_commit(root)},
        "evidence_refs": [],
        "audit_refs": [],
        "policy_decision_refs": [],
        "errors": [],
    }


def print_payload(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    status = payload.get("status", "unknown")
    command = payload.get("command", "cornerstone")
    print(f"{command}: {status}")
    for error in payload.get("errors", []):
        print(f"- {error.get('code')}: {error.get('message')}")


def state_dir(root: Path, args: argparse.Namespace) -> Path:
    return (root / args.state_dir).resolve()


def scope_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        "tenant_id": args.tenant_id,
        "owner_id": args.owner_id,
        "namespace_id": args.namespace_id,
        "workspace_id": args.workspace_id,
    }


def add_scope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tenant-id", default="local-dev", help="Tenant scope")
    parser.add_argument("--owner-id", default="local-user", help="Owner scope")
    parser.add_argument("--namespace-id", default="personal", help="Namespace scope")
    parser.add_argument("--workspace-id", default="default", help="Workspace scope")


def add_state_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", default="data/local", help="Local runtime state directory")


def command_version(args: argparse.Namespace) -> int:
    root = repo_root()
    payload = base_response("cornerstone version", "success", root)
    payload["ids"]["product_version"] = __version__
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_health(args: argparse.Namespace) -> int:
    root = repo_root()
    required = [
        "README.md",
        "AGENTS.md",
        "docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md",
        "docs/scenario-contracts/SCENARIO_MATRIX_FULL.md",
        "docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md",
        "scripts/verify_sot_docs.sh",
    ]
    missing = [path for path in required if not (root / path).exists()]
    payload = base_response("cornerstone health", "success" if not missing else "failed", root)
    payload["checks"] = [{"path": path, "present": path not in missing} for path in required]
    if missing:
        payload["errors"].append(
            {
                "code": "CS_HEALTH_MISSING_FILE",
                "message": "Required scaffold files are missing.",
                "missing": missing,
            }
        )
    print_payload(payload, args.json)
    return EXIT_SUCCESS if not missing else EXIT_EVIDENCE_MISSING


def command_ready(args: argparse.Namespace) -> int:
    root = repo_root()
    checks = [
        ("native_cli", (root / "cornerstone").exists()),
        ("scenario_matrix", (root / "docs/scenario-contracts/SCENARIO_MATRIX_FULL.md").exists()),
        ("vs0_contract", (root / "docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md").exists()),
        ("scenario_tests", (root / "tests/scenario").exists()),
        ("api_runtime", (root / "services/api").exists()),
        ("web_runtime", (root / "apps/web").exists()),
        ("fixture_corpus", (root / "fixtures/vs0").exists()),
    ]
    missing = [name for name, ok in checks if not ok]
    payload = base_response("cornerstone ready", "success" if not missing else "not_ready", root)
    payload["checks"] = [{"name": name, "present": ok} for name, ok in checks]
    if missing:
        payload["errors"].append(
            {
                "code": "CS_READY_RUNTIME_MISSING",
                "message": "The scaffold is not product-runtime ready yet.",
                "missing": missing,
            }
        )
    print_payload(payload, args.json)
    return EXIT_SUCCESS if not missing else EXIT_EVIDENCE_MISSING


def command_artifact_ingest(args: argparse.Namespace) -> int:
    root = repo_root()
    input_path = (root / args.path).resolve()
    payload = base_response("cornerstone artifact ingest", "success", root)
    if not input_path.exists() or not input_path.is_file():
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_ARTIFACT_INPUT_MISSING",
                "message": "Artifact input file does not exist.",
                "path": str(input_path),
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND

    store = LocalRuntimeStore(state_dir(root, args))
    try:
        result = store.ingest_artifact(
            input_path,
            **scope_args(args),
            source=args.source,
            media_type=args.media_type,
            derived_mode=args.derived_mode,
            trust=args.trust,
            lineage_from=args.lineage_from,
        )
    except OSError as error:
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_ARTIFACT_STORAGE_ERROR", "message": str(error)})
        print_payload(payload, args.json)
        return EXIT_RUNTIME_FAILURE

    artifact = result["artifact"]
    audit_event = result["audit_event"]
    audit_events = result.get("audit_events", [audit_event])
    policy_decisions = result.get("policy_decisions", [])
    payload.update(artifact["scope"])
    payload["ids"].update(
        {
            "artifact_id": artifact["artifact_id"],
            "checksum_sha256": artifact["checksum_sha256"],
            "audit_event_id": audit_event["event_id"],
        }
    )
    payload["deduplicated"] = result["deduplicated"]
    payload["artifact"] = artifact
    payload["evidence_refs"].extend(
        [
            f"artifact:{artifact['artifact_id']}",
            f"storage:{artifact['original_storage_ref']}",
        ]
    )
    payload["audit_refs"].extend(f"audit:{event['event_id']}" for event in audit_events)
    payload["policy_decision_refs"].extend(f"policy:{decision['id']}" for decision in policy_decisions)
    if policy_decisions:
        payload["policy_decisions"] = policy_decisions
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_artifact_show(args: argparse.Namespace) -> int:
    root = repo_root()
    payload = base_response("cornerstone artifact show", "success", root)
    store = LocalRuntimeStore(state_dir(root, args))
    artifact = store.get_artifact(args.artifact_id)
    if artifact is None:
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_ARTIFACT_NOT_FOUND",
                "message": "Artifact record was not found.",
                "artifact_id": args.artifact_id,
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND

    audit_event = store.append_audit(
        "artifact.read",
        artifact["scope"],
        {"type": "artifact", "id": artifact["artifact_id"]},
        {"reason": "cli_artifact_show"},
    )
    payload.update(artifact["scope"])
    payload["ids"].update(
        {
            "artifact_id": artifact["artifact_id"],
            "checksum_sha256": artifact["checksum_sha256"],
            "audit_event_id": audit_event["event_id"],
        }
    )
    payload["artifact"] = artifact
    payload["evidence_refs"].append(f"artifact:{artifact['artifact_id']}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_audit_verify(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    report = store.verify_audit()
    payload = base_response("cornerstone audit verify", report["status"], root)
    payload["audit_integrity"] = report
    if report["status"] != "success":
        payload["errors"].append(
            {
                "code": "CS_AUDIT_INTEGRITY_FAILED",
                "message": "Audit hash-chain verification failed.",
                "audit_errors": report["errors"],
            }
        )
    print_payload(payload, args.json)
    return EXIT_SUCCESS if report["status"] == "success" else EXIT_RUNTIME_FAILURE


def command_scenario_list(args: argparse.Namespace) -> int:
    root = repo_root()
    scenarios = list_scenarios(root, args.set)
    payload = base_response("cornerstone scenario list", "success", root)
    payload["scenario_set"] = args.set
    payload["count"] = len(scenarios)
    payload["scenarios"] = scenarios
    payload["evidence_refs"].append(f"doc:{args.set}:scenario-registry")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_scenario_coverage(args: argparse.Namespace) -> int:
    root = repo_root()
    report = coverage_report(root)
    payload = base_response(
        "cornerstone scenario coverage",
        "success" if report["ok"] else "failed",
        root,
    )
    payload.update(report)
    payload["evidence_refs"].extend(
        [
            "doc:docs/scenario-contracts/SCENARIO_MATRIX_FULL.md",
            "doc:docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md",
        ]
    )
    if not report["ok"]:
        payload["errors"].append(
            {
                "code": "CS_SCENARIO_COVERAGE_GAP",
                "message": "Scenario coverage is incomplete.",
                "missing": report["missing"],
            }
        )
    print_payload(payload, args.json)
    return EXIT_SUCCESS if report["ok"] else EXIT_EVIDENCE_MISSING


def command_scenario_verify(args: argparse.Namespace) -> int:
    root = repo_root()
    if args.contract == "vs0-scaffold":
        report = verify_vs0_scaffold(root)
    elif args.contract == "vs0-fixtures":
        report = verify_vs0_fixtures(root, corpus=args.corpus, model_provider=args.model_provider)
        if args.scenario:
            requested = set(args.scenario)
            referenced = {row["id"] for row in report.get("referenced_product_scenarios", [])}
            missing = sorted(requested - referenced)
            report["scenario_filter"] = sorted(requested)
            if missing:
                report["status"] = "failed"
                report.setdefault("errors", []).append(
                    {
                        "code": "CS_SCENARIO_FILTER_MISSING",
                        "message": "Requested product scenario is not referenced by the fixture corpus.",
                        "missing": missing,
                    }
                )
    elif args.contract == "vs0-artifacts":
        report = verify_vs0_artifacts(root)
    elif args.contract == "vs0-security":
        report = verify_vs0_security(root)
    else:
        payload = base_response("cornerstone scenario verify", "failed", root)
        payload["errors"].append(
            {
                "code": "CS_SCENARIO_CONTRACT_UNSUPPORTED",
                "message": "Only scaffold and fixture verification are implemented in this batch.",
                "supported": ["vs0-scaffold", "vs0-fixtures", "vs0-artifacts", "vs0-security"],
            }
        )
        print_payload(payload, args.json)
        return EXIT_INVALID

    payload = base_response(f"cornerstone scenario verify {args.contract}", report["status"], root)
    payload.update(report)
    if args.output:
        output_path = (root / args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload["output_path"] = str(output_path)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print_payload(payload, args.json)
    return EXIT_SUCCESS if report["status"] == "success" else EXIT_EVIDENCE_MISSING


def command_scenario_gate(args: argparse.Namespace) -> int:
    root = repo_root()
    report_path = (root / args.report).resolve()
    payload = base_response("cornerstone scenario gate", "success", root)
    if not report_path.exists():
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCENARIO_REPORT_MISSING",
                "message": "Scenario report file does not exist.",
                "path": str(report_path),
            }
        )
        print_payload(payload, args.json)
        return EXIT_EVIDENCE_MISSING

    data = json.loads(report_path.read_text())
    rows = data.get("scenario_results", [])
    report_errors = data.get("errors", [])
    blocking = [
        row
        for row in rows
        if row.get("owner", "AI") != "Human"
        and row.get("status") in {"FAIL", "NOT_VERIFIED", "NOT_RUN"}
    ]
    payload["checked_report"] = str(report_path)
    payload["scenario_count"] = len(rows)
    payload["blocking_count"] = len(blocking)
    payload["blocking"] = blocking
    if data.get("status") != "success" or report_errors:
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCENARIO_REPORT_FAILED",
                "message": "Scenario report status is not success or report-level errors are present.",
                "report_status": data.get("status"),
                "report_errors": report_errors,
            }
        )
    if blocking:
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCENARIO_GATE_BLOCKED",
                "message": "AI-verifiable scenarios remain non-PASS.",
            }
        )
    print_payload(payload, args.json)
    return EXIT_SUCCESS if payload["status"] == "success" else EXIT_EVIDENCE_MISSING


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cornerstone", description="CornerStone scaffold CLI")
    subcommands = parser.add_subparsers(dest="command")

    version = subcommands.add_parser("version", help="Print version information")
    version.add_argument("--json", action="store_true", help="Emit JSON output")
    version.set_defaults(func=command_version)

    health = subcommands.add_parser("health", help="Check local scaffold file health")
    health.add_argument("--json", action="store_true", help="Emit JSON output")
    health.set_defaults(func=command_health)

    ready = subcommands.add_parser("ready", help="Check local runtime readiness")
    ready.add_argument("--json", action="store_true", help="Emit JSON output")
    ready.set_defaults(func=command_ready)

    artifact = subcommands.add_parser("artifact", help="Artifact archive commands")
    artifact_sub = artifact.add_subparsers(dest="artifact_command")

    artifact_ingest = artifact_sub.add_parser("ingest", help="Ingest an immutable artifact")
    artifact_ingest.add_argument("path", help="Path to the input artifact")
    add_state_argument(artifact_ingest)
    add_scope_arguments(artifact_ingest)
    artifact_ingest.add_argument("--source", default="local_file", help="Artifact source type")
    artifact_ingest.add_argument("--media-type", default="text/plain", help="Input media type")
    artifact_ingest.add_argument("--derived-mode", choices=["auto", "fail", "unsupported"], default="auto")
    artifact_ingest.add_argument("--trust", choices=["trusted", "untrusted"], default="untrusted")
    artifact_ingest.add_argument("--lineage-from", help="Previous artifact ID for changed content")
    artifact_ingest.add_argument("--json", action="store_true", help="Emit JSON output")
    artifact_ingest.set_defaults(func=command_artifact_ingest)

    artifact_show = artifact_sub.add_parser("show", help="Show artifact metadata and provenance")
    artifact_show.add_argument("artifact_id", help="Artifact ID")
    add_state_argument(artifact_show)
    artifact_show.add_argument("--json", action="store_true", help="Emit JSON output")
    artifact_show.set_defaults(func=command_artifact_show)

    audit = subcommands.add_parser("audit", help="Audit ledger commands")
    audit_sub = audit.add_subparsers(dest="audit_command")

    audit_verify = audit_sub.add_parser("verify", help="Verify the local audit hash chain")
    add_state_argument(audit_verify)
    audit_verify.add_argument("--json", action="store_true", help="Emit JSON output")
    audit_verify.set_defaults(func=command_audit_verify)

    scenario = subcommands.add_parser("scenario", help="Scenario registry and verification commands")
    scenario_sub = scenario.add_subparsers(dest="scenario_command")

    scenario_list = scenario_sub.add_parser("list", help="List frozen scenarios")
    scenario_list.add_argument("--set", choices=["full", "vs0"], default="full")
    scenario_list.add_argument("--json", action="store_true", help="Emit JSON output")
    scenario_list.set_defaults(func=command_scenario_list)

    coverage = scenario_sub.add_parser("coverage", help="Check scenario registry coverage")
    coverage.add_argument("--json", action="store_true", help="Emit JSON output")
    coverage.set_defaults(func=command_scenario_coverage)

    verify = scenario_sub.add_parser("verify", help="Verify a scenario contract")
    verify.add_argument("contract", help="Scenario contract name")
    verify.add_argument("--scenario", action="append", help="Optional product scenario ID to require in the fixture corpus")
    verify.add_argument("--corpus", default="fixtures/vs0", help="Fixture corpus path for fixture verification")
    verify.add_argument("--model-provider", default="local_test", help="Deterministic local model provider")
    verify.add_argument("--json", action="store_true", help="Emit JSON output")
    verify.add_argument("--output", help="Optional path to write the JSON report")
    verify.set_defaults(func=command_scenario_verify)

    gate = scenario_sub.add_parser("gate", help="Gate a scenario report")
    gate.add_argument("report", help="Path to scenario report JSON")
    gate.add_argument("--json", action="store_true", help="Emit JSON output")
    gate.set_defaults(func=command_scenario_gate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return EXIT_SUCCESS
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

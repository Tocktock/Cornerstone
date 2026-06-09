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
    verify_vs0_audit_ledger,
    verify_vs0_claim_evidence,
    verify_vs0_namespace_isolation,
    verify_vs0_regression_guardrails,
    verify_vs0_security_policy,
    verify_vs0_universal_core,
    verify_vs0_search_evidence,
    verify_vs0_search_understanding,
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
EXIT_SCOPE_DENIED = 6
EXIT_POLICY_DENIED = 8


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
    requested_scope = scope_args(args)
    payload = base_response("cornerstone artifact ingest", "success", root)
    payload.update(requested_scope)
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
            **requested_scope,
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
    requested_scope = scope_args(args)
    payload = base_response("cornerstone artifact show", "success", root)
    payload.update(requested_scope)
    store = LocalRuntimeStore(state_dir(root, args))
    artifact = store.get_artifact(args.artifact_id, requested_scope)
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
        requested_scope,
        {"type": "artifact", "id": artifact["artifact_id"]},
        {"reason": "cli_artifact_show"},
    )
    artifact_detail = dict(artifact)
    artifact_detail["derived_text_preview"] = store.derived_text_preview(artifact)
    artifact_detail["related_claims"] = store.related_claims_for_artifact(artifact["artifact_id"], requested_scope)
    artifact_detail["related_missions"] = []
    payload.update(requested_scope)
    payload["ids"].update(
        {
            "artifact_id": artifact["artifact_id"],
            "checksum_sha256": artifact["checksum_sha256"],
            "audit_event_id": audit_event["event_id"],
        }
    )
    payload["artifact"] = artifact_detail
    payload["evidence_refs"].append(f"artifact:{artifact['artifact_id']}")
    payload["evidence_refs"].extend(f"claim:{claim['claim_id']}" for claim in artifact_detail["related_claims"])
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


def command_egress_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.deny_egress_attempt(args.url, requested_scope)
    decision = result["policy_decision"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone egress test", "denied", root)
    payload.update(requested_scope)
    payload["policy_decisions"] = [decision]
    payload["policy_decision_refs"].append(f"policy:{decision['id']}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    payload["errors"].append(
        {
            "code": "CS_EGRESS_DENIED",
            "message": decision["reason"],
            "resolution_path": decision["resolution_path"],
            "external_http_calls": 0,
        }
    )
    print_payload(payload, args.json)
    return EXIT_POLICY_DENIED


def command_sandbox_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.deny_sandbox_access(args.capability, args.target, requested_scope)
    decision = result["policy_decision"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone sandbox test", "denied", root)
    payload.update(requested_scope)
    payload["policy_decisions"] = [decision]
    payload["policy_decision_refs"].append(f"policy:{decision['id']}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    payload["errors"].append(
        {
            "code": "CS_SANDBOX_ACCESS_DENIED",
            "message": decision["reason"],
            "resolution_path": decision["resolution_path"],
            "host_operations_executed": 0,
        }
    )
    print_payload(payload, args.json)
    return EXIT_POLICY_DENIED


def command_search_query(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    result = store.search(args.query, **scope_args(args))
    snapshot = result["snapshot"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone search query", "success", root)
    payload.update(snapshot["filters"])
    payload["ids"].update(
        {
            "search_snapshot_id": snapshot["search_snapshot_id"],
            "audit_event_id": audit_event["event_id"],
        }
    )
    payload["search_snapshot"] = snapshot
    payload["evidence_refs"].append(f"search_snapshot:{snapshot['search_snapshot_id']}")
    payload["evidence_refs"].extend(
        ref
        for result_row in snapshot.get("results", [])
        for ref in result_row.get("evidence_refs", [])
    )
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_evidence_bundle_create(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_evidence_bundle(args.search_snapshot_id, requested_scope)
    payload = base_response("cornerstone evidence bundle create", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SEARCH_SNAPSHOT_NOT_FOUND",
                "message": "Search snapshot was not found.",
                "search_snapshot_id": args.search_snapshot_id,
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Search snapshot is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED

    bundle = result["bundle"]
    audit_event = result["audit_event"]
    payload.update(bundle["filters"])
    payload["ids"].update(
        {
            "evidence_bundle_id": bundle["evidence_bundle_id"],
            "search_snapshot_id": bundle["search_snapshot_id"],
            "audit_event_id": audit_event["event_id"],
        }
    )
    payload["evidence_bundle"] = bundle
    payload["evidence_refs"].extend(
        [
            f"evidence_bundle:{bundle['evidence_bundle_id']}",
            f"search_snapshot:{bundle['search_snapshot_id']}",
        ]
    )
    payload["evidence_refs"].extend(f"artifact:{item['artifact_id']}" for item in bundle.get("evidence_items", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_evidence_bundle_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.show_evidence_bundle(args.evidence_bundle_id, requested_scope)
    payload = base_response("cornerstone evidence bundle show", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_EVIDENCE_BUNDLE_NOT_FOUND",
                "message": "Evidence bundle was not found.",
                "evidence_bundle_id": args.evidence_bundle_id,
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Evidence bundle is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED

    bundle = result["bundle"]
    audit_event = result["audit_event"]
    payload.update(bundle["filters"])
    payload["ids"].update(
        {
            "evidence_bundle_id": bundle["evidence_bundle_id"],
            "search_snapshot_id": bundle["search_snapshot_id"],
            "audit_event_id": audit_event["event_id"],
        }
    )
    payload["evidence_bundle"] = bundle
    payload["evidence_refs"].extend(
        [
            f"evidence_bundle:{bundle['evidence_bundle_id']}",
            f"search_snapshot:{bundle['search_snapshot_id']}",
        ]
    )
    payload["evidence_refs"].extend(f"artifact:{item['artifact_id']}" for item in bundle.get("evidence_items", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_evidence_view(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.view_evidence_bundle(args.evidence_bundle_id, requested_scope)
    payload = base_response("cornerstone evidence view", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_EVIDENCE_BUNDLE_NOT_FOUND",
                "message": "Evidence bundle was not found.",
                "evidence_bundle_id": args.evidence_bundle_id,
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Evidence bundle is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED

    viewer = result["viewer"]
    audit_event = result["audit_event"]
    payload.update(viewer["filters"])
    payload["ids"].update(
        {
            "evidence_viewer_id": viewer["evidence_viewer_id"],
            "evidence_bundle_id": viewer["evidence_bundle_id"],
            "search_snapshot_id": viewer["search_snapshot_id"],
            "audit_event_id": audit_event["event_id"],
        }
    )
    payload["evidence_viewer"] = viewer
    payload["evidence_refs"].extend(
        [
            f"evidence_viewer:{viewer['evidence_viewer_id']}",
            f"evidence_bundle:{viewer['evidence_bundle_id']}",
            f"search_snapshot:{viewer['search_snapshot_id']}",
        ]
    )
    payload["evidence_refs"].extend(f"artifact:{item['artifact_id']}" for item in viewer.get("viewer_items", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_claim_create(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    if args.evidence_bundle_id:
        result = store.create_claim_from_evidence_bundle(args.evidence_bundle_id, args.statement, requested_scope)
    else:
        result = store.create_unsupported_claim(args.statement, requested_scope)
    payload = base_response("cornerstone claim create", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_EVIDENCE_BUNDLE_NOT_FOUND",
                "message": "Evidence bundle was not found.",
                "evidence_bundle_id": args.evidence_bundle_id,
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Evidence bundle is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED

    claim = result["claim"]
    audit_event = result["audit_event"]
    evidence = claim["evidence_bundle"]
    payload.update(claim["scope"])
    payload["ids"].update({"claim_id": claim["claim_id"], "audit_event_id": audit_event["event_id"]})
    if evidence.get("evidence_bundle_id"):
        payload["ids"]["evidence_bundle_id"] = evidence["evidence_bundle_id"]
    if evidence.get("search_snapshot_id"):
        payload["ids"]["search_snapshot_id"] = evidence["search_snapshot_id"]
    payload["claim"] = claim
    payload["evidence_refs"].append(f"claim:{claim['claim_id']}")
    if evidence.get("evidence_bundle_id"):
        payload["evidence_refs"].append(f"evidence_bundle:{evidence['evidence_bundle_id']}")
    if evidence.get("search_snapshot_id"):
        payload["evidence_refs"].append(f"search_snapshot:{evidence['search_snapshot_id']}")
    payload["evidence_refs"].extend(evidence.get("artifact_refs", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_claim_approve(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.approve_claim(args.claim_id, requested_scope)
    payload = base_response("cornerstone claim approve", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_CLAIM_NOT_FOUND",
                "message": "Claim was not found.",
                "claim_id": args.claim_id,
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Claim is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED

    claim = result["claim"]
    audit_event = result["audit_event"]
    evidence = claim["evidence_bundle"]
    payload.update(claim["scope"])
    payload["ids"].update({"claim_id": claim["claim_id"], "audit_event_id": audit_event["event_id"]})
    if evidence.get("evidence_bundle_id"):
        payload["ids"]["evidence_bundle_id"] = evidence["evidence_bundle_id"]
    if evidence.get("search_snapshot_id"):
        payload["ids"]["search_snapshot_id"] = evidence["search_snapshot_id"]
    payload["claim"] = claim
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    if result.get("status") == "evidence_required":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_CLAIM_EVIDENCE_REQUIRED",
                "message": "Claim approval requires an Evidence Bundle with at least one artifact reference.",
                "resolution_path": [
                    "Run cornerstone search query ... --json.",
                    "Run cornerstone evidence bundle create --search-snapshot-id <id> --json.",
                    "Create or recreate the claim with --evidence-bundle-id <id>.",
                ],
            }
        )
        print_payload(payload, args.json)
        return EXIT_EVIDENCE_MISSING

    payload["evidence_refs"].extend(
        [
            f"claim:{claim['claim_id']}",
            f"evidence_bundle:{evidence['evidence_bundle_id']}",
            f"search_snapshot:{evidence['search_snapshot_id']}",
        ]
    )
    payload["evidence_refs"].extend(evidence.get("artifact_refs", []))
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_claim_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.show_claim(args.claim_id, requested_scope)
    payload = base_response("cornerstone claim show", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_CLAIM_NOT_FOUND",
                "message": "Claim was not found.",
                "claim_id": args.claim_id,
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Claim is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED

    claim = result["claim"]
    audit_event = result["audit_event"]
    evidence = claim["evidence_bundle"]
    payload.update(claim["scope"])
    payload["ids"].update({"claim_id": claim["claim_id"], "audit_event_id": audit_event["event_id"]})
    if evidence.get("evidence_bundle_id"):
        payload["ids"]["evidence_bundle_id"] = evidence["evidence_bundle_id"]
    if evidence.get("search_snapshot_id"):
        payload["ids"]["search_snapshot_id"] = evidence["search_snapshot_id"]
    payload["claim"] = claim
    payload["evidence_refs"].append(f"claim:{claim['claim_id']}")
    if evidence.get("evidence_bundle_id"):
        payload["evidence_refs"].append(f"evidence_bundle:{evidence['evidence_bundle_id']}")
    if evidence.get("search_snapshot_id"):
        payload["evidence_refs"].append(f"search_snapshot:{evidence['search_snapshot_id']}")
    payload["evidence_refs"].extend(evidence.get("artifact_refs", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


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
    elif args.contract == "vs0-search-evidence":
        report = verify_vs0_search_evidence(root)
    elif args.contract == "vs0-search-understanding":
        report = verify_vs0_search_understanding(root)
    elif args.contract == "vs0-namespace-isolation":
        report = verify_vs0_namespace_isolation(root)
    elif args.contract == "vs0-audit-ledger":
        report = verify_vs0_audit_ledger(root)
    elif args.contract == "vs0-universal-core":
        report = verify_vs0_universal_core(root)
    elif args.contract == "vs0-claim-evidence":
        report = verify_vs0_claim_evidence(root)
    elif args.contract == "vs0-security-policy":
        report = verify_vs0_security_policy(root)
    elif args.contract == "vs0-regression-guardrails":
        report = verify_vs0_regression_guardrails(root)
    else:
        payload = base_response("cornerstone scenario verify", "failed", root)
        payload["errors"].append(
            {
                "code": "CS_SCENARIO_CONTRACT_UNSUPPORTED",
                "message": "The requested scenario verification contract is not implemented.",
                "supported": [
                    "vs0-scaffold",
                    "vs0-fixtures",
                    "vs0-artifacts",
                    "vs0-security",
                    "vs0-search-evidence",
                    "vs0-search-understanding",
                    "vs0-namespace-isolation",
                    "vs0-audit-ledger",
                    "vs0-universal-core",
                    "vs0-claim-evidence",
                    "vs0-security-policy",
                    "vs0-regression-guardrails",
                ],
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
    add_scope_arguments(artifact_show)
    artifact_show.add_argument("--json", action="store_true", help="Emit JSON output")
    artifact_show.set_defaults(func=command_artifact_show)

    audit = subcommands.add_parser("audit", help="Audit ledger commands")
    audit_sub = audit.add_subparsers(dest="audit_command")

    audit_verify = audit_sub.add_parser("verify", help="Verify the local audit hash chain")
    add_state_argument(audit_verify)
    audit_verify.add_argument("--json", action="store_true", help="Emit JSON output")
    audit_verify.set_defaults(func=command_audit_verify)

    egress = subcommands.add_parser("egress", help="Egress policy commands")
    egress_sub = egress.add_subparsers(dest="egress_command")

    egress_test = egress_sub.add_parser("test", help="Verify default egress denial without making a network call")
    egress_test.add_argument("--url", default="https://example.invalid/blocked", help="External URL to evaluate")
    add_state_argument(egress_test)
    add_scope_arguments(egress_test)
    egress_test.add_argument("--json", action="store_true", help="Emit JSON output")
    egress_test.set_defaults(func=command_egress_test)

    sandbox = subcommands.add_parser("sandbox", help="Sandbox policy commands")
    sandbox_sub = sandbox.add_subparsers(dest="sandbox_command")

    sandbox_test = sandbox_sub.add_parser("test", help="Verify undeclared host/tool access denial without executing it")
    sandbox_test.add_argument("--capability", choices=["shell", "filesystem", "environment", "host"], default="shell")
    sandbox_test.add_argument("--target", default="arbitrary-host-access")
    add_state_argument(sandbox_test)
    add_scope_arguments(sandbox_test)
    sandbox_test.add_argument("--json", action="store_true", help="Emit JSON output")
    sandbox_test.set_defaults(func=command_sandbox_test)

    search = subcommands.add_parser("search", help="Search artifact-derived content")
    search_sub = search.add_subparsers(dest="search_command")

    search_query = search_sub.add_parser("query", help="Search local artifact-derived content")
    search_query.add_argument("query", help="Search query")
    add_state_argument(search_query)
    add_scope_arguments(search_query)
    search_query.add_argument("--json", action="store_true", help="Emit JSON output")
    search_query.set_defaults(func=command_search_query)

    evidence = subcommands.add_parser("evidence", help="Evidence bundle commands")
    evidence_sub = evidence.add_subparsers(dest="evidence_command")

    bundle = evidence_sub.add_parser("bundle", help="Evidence bundle operations")
    bundle_sub = bundle.add_subparsers(dest="bundle_command")

    bundle_create = bundle_sub.add_parser("create", help="Create an evidence bundle from a search snapshot")
    bundle_create.add_argument("--search-snapshot-id", required=True, help="Search snapshot ID")
    add_state_argument(bundle_create)
    add_scope_arguments(bundle_create)
    bundle_create.add_argument("--json", action="store_true", help="Emit JSON output")
    bundle_create.set_defaults(func=command_evidence_bundle_create)

    bundle_show = bundle_sub.add_parser("show", help="Show an evidence bundle")
    bundle_show.add_argument("evidence_bundle_id", help="Evidence bundle ID")
    add_state_argument(bundle_show)
    add_scope_arguments(bundle_show)
    bundle_show.add_argument("--json", action="store_true", help="Emit JSON output")
    bundle_show.set_defaults(func=command_evidence_bundle_show)

    evidence_view = evidence_sub.add_parser("view", help="Open original and derived evidence representations")
    evidence_view.add_argument("evidence_bundle_id", help="Evidence bundle ID")
    add_state_argument(evidence_view)
    add_scope_arguments(evidence_view)
    evidence_view.add_argument("--json", action="store_true", help="Emit JSON output")
    evidence_view.set_defaults(func=command_evidence_view)

    claim = subcommands.add_parser("claim", help="Draft claim commands")
    claim_sub = claim.add_subparsers(dest="claim_command")

    claim_create = claim_sub.add_parser("create", help="Create a draft claim from an evidence bundle")
    claim_create.add_argument("--evidence-bundle-id", help="Evidence bundle ID")
    claim_create.add_argument("--statement", required=True, help="Draft claim statement")
    add_state_argument(claim_create)
    add_scope_arguments(claim_create)
    claim_create.add_argument("--json", action="store_true", help="Emit JSON output")
    claim_create.set_defaults(func=command_claim_create)

    claim_approve = claim_sub.add_parser("approve", help="Approve an evidence-backed claim")
    claim_approve.add_argument("claim_id", help="Claim ID")
    add_state_argument(claim_approve)
    add_scope_arguments(claim_approve)
    claim_approve.add_argument("--json", action="store_true", help="Emit JSON output")
    claim_approve.set_defaults(func=command_claim_approve)

    claim_show = claim_sub.add_parser("show", help="Show a draft claim")
    claim_show.add_argument("claim_id", help="Claim ID")
    add_state_argument(claim_show)
    add_scope_arguments(claim_show)
    claim_show.add_argument("--json", action="store_true", help="Emit JSON output")
    claim_show.set_defaults(func=command_claim_show)

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

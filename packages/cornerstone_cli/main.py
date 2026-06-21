from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from cornerstone_cli import __version__
from cornerstone_cli.acceptance import (
    DEFAULT_ACCEPTANCE_REPORT,
    DEFAULT_ACCEPTANCE_SCENARIO_REPORT,
    DEFAULT_BROWSER_PROOF_DIR,
    DEFAULT_EVUX_BROWSER_PROOF_DIR,
    DEFAULT_EVUX_QUICKSTART_REPORT,
    DEFAULT_EVUX_RELEASE_PACKAGE_DIR,
    DEFAULT_EVUX_REPORT,
    DEFAULT_EVUX_SCENARIO_REPORT,
    DEFAULT_OPERATOR_UI_SCENARIO_REPORT,
    DEFAULT_PRODUCT_RUNTIME_REPORT,
    DEFAULT_RELEASE_PACKAGE_DIR,
    DEFAULT_VS1_ONTOLOGY_SCENARIO_REPORT,
    command_transcript_entry,
    collect_release_evidence,
    finalize_release_evidence,
    run_evux_quickstart,
    write_json,
)
from cornerstone_cli.scenarios import (
    coverage_report,
    list_scenarios,
    verify_vs0_evux,
    verify_vs0_evux_governance,
    verify_vs0_operator_acceptance_ui,
    verify_vs1_ontology_suggest_promote,
    verify_vs2_policy_tenancy_egress,
    verify_vs0_runtime_acceptance,
    verify_vs0_product_runtime,
    verify_full_agent_orchestration,
    verify_full_brain_routing,
    verify_full_claim_collaboration,
    verify_full_extension_ecosystem,
    verify_full_learning_experience,
    verify_full_mission_control_autonomy_lifecycle,
    verify_full_memory_wiki,
    verify_full_namespace_governance,
    verify_full_security_operations,
    verify_full_understanding_ontology,
    verify_vs0_audit_ledger,
    verify_vs0_briefing,
    verify_vs0_claim_evidence,
    verify_vs0_conversation_onboarding,
    verify_vs0_detail_surfaces,
    verify_vs0_mission_action,
    verify_vs0_memory_truth_boundary,
    verify_vs0_product_loop_identity,
    verify_vs0_product_domain_readiness,
    verify_vs0_tenant_security_boundary,
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
from cornerstone_cli.product_runtime import build_readiness_report, run_server
from cornerstone_cli.runtime import LocalRuntimeStore
from cornerstone_cli.vs2_security import run_vs2_local_security_proof
from cornerstone_cli.vs2_local_range import (
    run_vs2_local_range,
    run_vs2_range_action_client,
    run_vs2_range_audit_integrity_client,
    run_vs2_range_client,
    run_vs2_range_constraint_collision_client,
    run_vs2_range_db_path_client,
    run_vs2_range_migration_client,
    run_vs2_range_object_access_client,
    run_vs2_range_object_contract_client,
    run_vs2_range_observability_client,
    run_vs2_range_search_client,
    run_vs2_range_tenant_read_client,
    run_vs2_range_upgrade_path_client,
)


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
    report = build_readiness_report(root)
    readiness = report["readiness"]
    ready = readiness["local_scenario_ready"] and readiness["vs0_runtime_ready"]
    missing = [row["name"] for row in report["checks"] if not row["present"]]
    payload = base_response("cornerstone ready", "success" if ready else "not_ready", root)
    payload.update(report)
    if not ready:
        payload["errors"].append(
            {
                "code": "CS_READY_RUNTIME_MISSING",
                "message": "The local VS0 runtime is not ready yet.",
                "missing": missing,
            }
        )
    print_payload(payload, args.json)
    return EXIT_SUCCESS if ready else EXIT_EVIDENCE_MISSING


def command_runtime_serve(args: argparse.Namespace) -> int:
    root = repo_root()
    run_server(root, state_dir(root, args), host=args.host, port=args.port)
    return EXIT_SUCCESS


def command_product_walkthrough(args: argparse.Namespace) -> int:
    root = repo_root()
    payload = base_response("cornerstone product walkthrough", "success", root)
    payload["walkthrough"] = {
        "schema_version": "cs.product_walkthrough.v0",
        "product_name": "CornerStone",
        "one_service": True,
        "daily_user_requires_subsystem_knowledge": False,
        "primary_navigation": [
            {"id": "home", "label": "Home"},
            {"id": "search", "label": "Search"},
            {"id": "artifacts", "label": "Artifacts"},
            {"id": "claims", "label": "Claims"},
            {"id": "actions", "label": "Actions"},
        ],
        "first_run_path": ["Inbox", "Brief", "Claim", "Action", "Learn"],
        "capability_language": [
            "Capture information as immutable artifacts.",
            "Search and inspect source-backed evidence.",
            "Create evidence-backed briefs and claims.",
            "Use governed action cards for safe action.",
            "Keep audit and learning visible as product surfaces.",
        ],
        "internal_boundaries": [
            "Archive/Evidence",
            "Mission/Intelligence",
            "Connector/Action",
        ],
        "boundary_explanation": "Internal engines are implementation boundaries; daily users see one CornerStone workflow.",
    }
    payload["evidence_refs"].append("product:CornerStone:first-run-walkthrough")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def _surface_payload(args: argparse.Namespace, command: str) -> tuple[Path, LocalRuntimeStore, dict[str, str], dict[str, Any]]:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    payload = base_response(command, "success", root)
    payload.update(requested_scope)
    return root, store, requested_scope, payload


def command_product_mission_control(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = _surface_payload(args, "cornerstone product mission-control")
    result = store.mission_control_view(requested_scope)
    surface = result["mission_control"]
    payload["mission_control"] = surface
    payload["ids"].update({"mission_control_id": surface["surface_id"]})
    payload["evidence_refs"].append(f"mission_control:{surface['surface_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_product_loop_view(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = _surface_payload(args, "cornerstone product loop-view")
    result = store.product_loop_view(
        requested_scope,
        conversation_id=args.conversation_id or "",
        brief_id=args.brief_id or "",
        claim_id=args.claim_id or "",
        mission_id=args.mission_id or "",
        action_id=args.action_id or "",
        outcome_id=args.outcome_id or "",
    )
    loop = result["product_loop"]
    payload["product_loop"] = loop
    payload["ids"].update({"loop_id": loop["loop_id"]})
    payload["evidence_refs"].append(f"product_loop:{loop['loop_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_product_boundary(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = _surface_payload(args, "cornerstone product boundary")
    result = store.product_boundary_review(requested_scope)
    boundary = result["boundary_review"]
    payload["boundary_review"] = boundary
    payload["ids"].update({"boundary_id": boundary["boundary_id"]})
    payload["evidence_refs"].append(f"product_boundary:{boundary['boundary_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_product_plain_language(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = _surface_payload(args, "cornerstone product plain-language-review")
    result = store.product_plain_language_review(requested_scope)
    review = result["plain_language_review"]
    payload["plain_language_review"] = review
    payload["ids"].update({"review_id": review["review_id"]})
    payload["evidence_refs"].append(f"product_plain_language_review:{review['review_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_product_repo_split_review(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = _surface_payload(args, "cornerstone product repo-split-review")
    result = store.product_repo_split_review(requested_scope)
    review = result["repo_split_review"]
    payload["repo_split_review"] = review
    payload["ids"].update({"review_id": review["review_id"]})
    payload["evidence_refs"].append(f"product_repo_split_review:{review['review_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


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
    artifact_detail["related_missions"] = store.related_missions_for_artifact(artifact["artifact_id"], requested_scope)
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


def command_audit_list(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.query_namespace_audit(requested_scope, event_types=args.event_type)
    export = result["namespace_audit_export"]
    payload = base_response("cornerstone audit list", "success", root)
    payload.update(requested_scope)
    payload["audit_events"] = export["events"]
    payload["namespace_audit_export"] = export
    payload["ids"].update({"namespace_audit_export_id": export["namespace_audit_export_id"]})
    payload["evidence_refs"].append(f"namespace_audit_export:{export['namespace_audit_export_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


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


def command_search_snapshot_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    snapshot = store.get_search_snapshot(args.search_snapshot_id)
    payload = base_response("cornerstone search snapshot show", "success", root)
    payload.update(requested_scope)
    if snapshot is None:
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
    if snapshot.get("filters") != requested_scope:
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Search snapshot is outside the requested scope.",
                "resource_scope": snapshot.get("filters"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    audit_event = store.append_audit(
        "search.snapshot.read",
        requested_scope,
        {"type": "search_snapshot", "id": args.search_snapshot_id},
        {"reason": "cli_search_snapshot_show"},
    )
    payload["search_snapshot"] = snapshot
    payload["ids"].update({"search_snapshot_id": args.search_snapshot_id, "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].append(f"search_snapshot:{args.search_snapshot_id}")
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


def command_brief_create(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_brief_from_evidence_bundle(args.evidence_bundle_id, requested_scope)
    payload = base_response("cornerstone brief create", "success", root)
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
    if result.get("status") == "evidence_required":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_BRIEF_EVIDENCE_REQUIRED",
                "message": "Brief creation requires an Evidence Bundle with at least one evidence item.",
            }
        )
        print_payload(payload, args.json)
        return EXIT_EVIDENCE_MISSING

    brief = result["brief"]
    audit_event = result["audit_event"]
    evidence = brief["evidence_bundle"]
    payload.update(brief["scope"])
    payload["ids"].update(
        {
            "brief_id": brief["brief_id"],
            "evidence_bundle_id": evidence["evidence_bundle_id"],
            "search_snapshot_id": evidence["search_snapshot_id"],
            "audit_event_id": audit_event["event_id"],
        }
    )
    payload["brief"] = brief
    payload["evidence_refs"].extend(
        [
            f"brief:{brief['brief_id']}",
            f"evidence_bundle:{evidence['evidence_bundle_id']}",
            f"search_snapshot:{evidence['search_snapshot_id']}",
        ]
    )
    payload["evidence_refs"].extend(evidence.get("artifact_refs", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_brief_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.show_brief(args.brief_id, requested_scope)
    payload = base_response("cornerstone brief show", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_BRIEF_NOT_FOUND",
                "message": "Brief was not found.",
                "brief_id": args.brief_id,
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Brief is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED

    brief = result["brief"]
    audit_event = result["audit_event"]
    evidence = brief["evidence_bundle"]
    payload.update(brief["scope"])
    payload["ids"].update(
        {
            "brief_id": brief["brief_id"],
            "evidence_bundle_id": evidence["evidence_bundle_id"],
            "search_snapshot_id": evidence["search_snapshot_id"],
            "audit_event_id": audit_event["event_id"],
        }
    )
    payload["brief"] = brief
    payload["evidence_refs"].extend(
        [
            f"brief:{brief['brief_id']}",
            f"evidence_bundle:{evidence['evidence_bundle_id']}",
            f"search_snapshot:{evidence['search_snapshot_id']}",
        ]
    )
    payload["evidence_refs"].extend(evidence.get("artifact_refs", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_claim_create(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    if args.evidence_bundle_id:
        result = store.create_claim_from_evidence_bundle(
            args.evidence_bundle_id,
            args.statement,
            requested_scope,
            ontology_object_refs=args.ontology_object_ref or [],
        )
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
    payload["evidence_refs"].extend(claim.get("ontology_context", {}).get("object_refs", []))
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


def command_claim_basis_export(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.export_claim_basis(args.claim_id, requested_scope)
    payload = base_response("cornerstone claim basis-export", "success", root)
    payload.update(requested_scope)
    if result.get("status") in {"not_found", "scope_denied", "evidence_required"}:
        exit_code = _record_command_failure(payload, result, resource_label="CLAIM_BASIS")
        print_payload(payload, args.json)
        return exit_code

    export = result["claim_basis_export"]
    payload["claim_basis_export"] = export
    payload["ids"].update({"claim_id": args.claim_id, "claim_basis_export_id": export["claim_basis_export_id"]})
    payload["evidence_refs"].append(f"claim:{args.claim_id}")
    if export.get("evidence_bundle", {}).get("evidence_bundle_id"):
        payload["evidence_refs"].append(f"evidence_bundle:{export['evidence_bundle']['evidence_bundle_id']}")
    if export.get("search_snapshot", {}).get("search_snapshot_id"):
        payload["evidence_refs"].append(f"search_snapshot:{export['search_snapshot']['search_snapshot_id']}")
    payload["evidence_refs"].extend(f"artifact:{artifact['artifact_id']}" for artifact in export.get("source_artifacts", []))
    payload["evidence_refs"].append(f"claim_basis_export:{export['claim_basis_export_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def _record_command_failure(payload: dict[str, Any], result: dict[str, Any], *, resource_label: str) -> int:
    status = result.get("status")
    payload["status"] = "failed"
    if status == "not_found":
        payload["errors"].append(
            {
                "code": f"CS_{resource_label}_NOT_FOUND",
                "message": "Required source record was not found.",
                "resource": result.get("resource"),
            }
        )
        return EXIT_NOT_FOUND
    if status == "scope_denied":
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Required source record is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        return EXIT_SCOPE_DENIED
    if status == "evidence_required":
        payload["errors"].append(
            {
                "code": f"CS_{resource_label}_EVIDENCE_REQUIRED",
                "message": "This record requires evidence-backed source material.",
                "resource": result.get("resource"),
            }
        )
        return EXIT_EVIDENCE_MISSING
    payload["errors"].append(
        {
            "code": f"CS_{resource_label}_INVALID",
            "message": "The requested operation is not supported by the local scaffold.",
        }
    )
    return EXIT_INVALID


def command_capsule_create(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_knowledge_capsule(
        claim_id=args.claim_id,
        title=args.title,
        summary=args.summary,
        scope=requested_scope,
    )
    payload = base_response("cornerstone capsule create", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="CAPSULE")
        print_payload(payload, args.json)
        return exit_code
    capsule = result["capsule"]
    audit_event = result["audit_event"]
    payload["knowledge_capsule"] = capsule
    payload["ids"].update({"capsule_id": capsule["capsule_id"], "claim_id": args.claim_id, "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].extend([f"knowledge_capsule:{capsule['capsule_id']}", *capsule.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_capsule_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.show_knowledge_capsule(args.capsule_id, requested_scope)
    payload = base_response("cornerstone capsule show", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="CAPSULE")
        print_payload(payload, args.json)
        return exit_code
    capsule = result["capsule"]
    audit_event = result["audit_event"]
    payload["knowledge_capsule"] = capsule
    payload["ids"].update({"capsule_id": capsule["capsule_id"], "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].extend([f"knowledge_capsule:{capsule['capsule_id']}", *capsule.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_decision_card_create(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_decision_card(
        goal=args.goal,
        claim_id=args.claim_id,
        mission_id=args.mission_id,
        scope=requested_scope,
    )
    payload = base_response("cornerstone decision-card create", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="DECISION_CARD")
        print_payload(payload, args.json)
        return exit_code
    card = result["decision_card"]
    audit_event = result["audit_event"]
    payload["decision_card"] = card
    payload["ids"].update({"decision_card_id": card["decision_card_id"], "claim_id": args.claim_id, "mission_id": args.mission_id, "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].append(f"decision_card:{card['decision_card_id']}")
    payload["evidence_refs"].extend(card.get("evidence", {}).get("evidence_refs", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_decision_card_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.show_decision_card(args.decision_card_id, requested_scope)
    payload = base_response("cornerstone decision-card show", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="DECISION_CARD")
        print_payload(payload, args.json)
        return exit_code
    card = result["decision_card"]
    audit_event = result["audit_event"]
    payload["decision_card"] = card
    payload["ids"].update({"decision_card_id": card["decision_card_id"], "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].append(f"decision_card:{card['decision_card_id']}")
    payload["evidence_refs"].extend(card.get("evidence", {}).get("evidence_refs", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_correction_record(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.record_correction(
        target_kind=args.target_kind,
        target_id=args.target_id,
        corrected_text=args.corrected_text,
        rationale=args.rationale,
        evidence_bundle_id=args.evidence_bundle_id,
        scope=requested_scope,
    )
    payload = base_response("cornerstone correction record", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="CORRECTION")
        print_payload(payload, args.json)
        return exit_code
    correction = result["correction"]
    audit_event = result["audit_event"]
    payload["correction"] = correction
    payload["target"] = result["target"]
    payload["ids"].update({"correction_id": correction["correction_id"], "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].extend([f"correction:{correction['correction_id']}", *correction.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_share_create(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_share_view(
        item_kind=args.item_kind,
        item_id=args.item_id,
        audience=args.audience,
        channel=args.channel,
        scope=requested_scope,
    )
    payload = base_response("cornerstone share create", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="SHARE")
        print_payload(payload, args.json)
        return exit_code
    share = result["share"]
    audit_event = result["audit_event"]
    payload["shared_item_view"] = share
    payload["ids"].update({"share_id": share["share_id"], "item_id": args.item_id, "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].extend([f"share:{share['share_id']}", *share.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_share_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.show_share_view(args.share_id, requested_scope)
    payload = base_response("cornerstone share show", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="SHARE")
        print_payload(payload, args.json)
        return exit_code
    share = result["share"]
    audit_event = result["audit_event"]
    payload["shared_item_view"] = share
    payload["ids"].update({"share_id": share["share_id"], "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].extend([f"share:{share['share_id']}", *share.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_understand_suggest(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.suggest_operational_structure(args.artifact_id, requested_scope, domain=args.domain)
    payload = base_response("cornerstone understand suggest", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="UNDERSTANDING")
        print_payload(payload, args.json)
        return exit_code
    suggestions = result["suggestions"]
    audit_event = result["audit_event"]
    payload["understanding_suggestions"] = suggestions
    payload["ids"].update({"artifact_id": args.artifact_id, "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].extend(ref for suggestion in suggestions for ref in suggestion.get("evidence_refs", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_understand_promote(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.promote_understanding_suggestion(args.suggestion_id, requested_scope)
    payload = base_response("cornerstone understand promote", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="ONTOLOGY")
        print_payload(payload, args.json)
        return exit_code
    item = result["ontology_item"]
    audit_event = result["audit_event"]
    payload["ontology_item"] = item
    payload["ids"].update({"ontology_item_id": item["ontology_item_id"], "suggestion_id": args.suggestion_id, "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].extend([f"ontology_item:{item['ontology_item_id']}", *item.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_understand_map(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.operational_map(requested_scope)
    operational_map = result["operational_map"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone understand map", "success", root)
    payload.update(requested_scope)
    payload["operational_map"] = operational_map
    payload["ids"].update({"operational_map_id": operational_map["operational_map_id"], "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].append(f"operational_map:{operational_map['operational_map_id']}")
    for node in operational_map.get("nodes", []):
        payload["evidence_refs"].extend(node.get("evidence_refs", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_understand_contradictions(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.detect_contradictions(requested_scope)
    contradictions = result["contradictions"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone understand contradictions", "success", root)
    payload.update(requested_scope)
    payload["contradictions"] = contradictions
    payload["ids"].update({"audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].extend(ref for contradiction in contradictions for evidence in contradiction.get("competing_evidence", []) for ref in evidence.get("evidence_refs", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_understand_stale_check(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.check_staleness(args.claim_id, args.newer_evidence_bundle_id, requested_scope)
    payload = base_response("cornerstone understand stale-check", "success", root)
    payload.update(requested_scope)
    if result.get("status") in {"not_found", "scope_denied"}:
        exit_code = _record_command_failure(payload, result, resource_label="STALENESS")
        print_payload(payload, args.json)
        return exit_code
    staleness = result["staleness"]
    audit_event = result["audit_event"]
    payload["staleness"] = staleness
    payload["ids"].update({"staleness_id": staleness["staleness_id"], "claim_id": args.claim_id, "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].extend([f"staleness:{staleness['staleness_id']}", *staleness.get("old_evidence_refs", []), *staleness.get("newer_evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_understand_ontology_change(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.record_ontology_change(args.item_id, property_name=args.property, new_value=args.to_value, scope=requested_scope)
    payload = base_response("cornerstone understand ontology-change", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="ONTOLOGY_CHANGE")
        print_payload(payload, args.json)
        return exit_code
    change = result["ontology_change"]
    item = result["ontology_item"]
    audit_event = result["audit_event"]
    payload["ontology_change"] = change
    payload["ontology_item"] = item
    payload["ids"].update({"ontology_change_id": change["ontology_change_id"], "ontology_item_id": item["ontology_item_id"], "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].extend([f"ontology_change:{change['ontology_change_id']}", *change.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def _ontology_failure(payload: dict[str, Any], result: dict[str, Any]) -> int:
    status = result.get("status")
    payload["status"] = "failed" if status != "policy_denied" else "denied"
    if status == "not_found":
        payload["errors"].append(
            {
                "code": "CS_ONTOLOGY_NOT_FOUND",
                "message": "Required ontology source record was not found.",
                "resource": result.get("resource"),
                "missing": result.get("missing"),
            }
        )
        return EXIT_NOT_FOUND
    if status == "scope_denied":
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Ontology source is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        return EXIT_SCOPE_DENIED
    if status == "policy_denied":
        payload["policy_decisions"].append(result.get("policy_decision", {}))
        if result.get("policy_decision", {}).get("id"):
            payload["policy_decision_refs"].append(f"policy:{result['policy_decision']['id']}")
        payload["errors"].append(
            {
                "code": "CS_ONTOLOGY_POLICY_DENIED",
                "message": result.get("reason") or result.get("policy_decision", {}).get("reason") or "Ontology policy denied the operation.",
            }
        )
        if result.get("audit_event"):
            payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
        return EXIT_POLICY_DENIED
    if status == "invalid_graph":
        payload["errors"].append(result.get("error") or {"code": "CS_ONTOLOGY_INVALID_GRAPH", "message": "Ontology graph is invalid."})
        if result.get("audit_event"):
            payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
        return EXIT_INVALID
    payload["errors"].append({"code": "CS_ONTOLOGY_INVALID", "message": "The requested ontology operation is invalid."})
    return EXIT_INVALID


def command_ontology_suggest(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_ontology_suggestion_set(args.source_type, args.source_id, requested_scope)
    payload = base_response("cornerstone ontology suggest", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _ontology_failure(payload, result)
        print_payload(payload, args.json)
        return exit_code
    suggestion_set = result["suggestion_set"]
    payload["ontology_suggestion_set"] = suggestion_set
    payload["ids"].update(
        {
            "ontology_suggestion_set_id": suggestion_set["suggestion_set_id"],
            "audit_event_id": result["audit_event"]["event_id"],
        }
    )
    payload["evidence_refs"].extend([f"ontology_suggestion_set:{suggestion_set['suggestion_set_id']}", *suggestion_set.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_ontology_review(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.review_ontology_suggestion_set(
        args.suggestion_set_id,
        requested_scope,
        select=args.select or [],
        reject=args.reject or [],
        defer=args.defer or [],
    )
    payload = base_response("cornerstone ontology review", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _ontology_failure(payload, result)
        print_payload(payload, args.json)
        return exit_code
    suggestion_set = result["suggestion_set"]
    payload["ontology_suggestion_set"] = suggestion_set
    payload["ids"].update({"ontology_suggestion_set_id": suggestion_set["suggestion_set_id"], "audit_event_id": result["audit_event"]["event_id"]})
    payload["evidence_refs"].extend([f"ontology_suggestion_set:{suggestion_set['suggestion_set_id']}", *suggestion_set.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_ontology_promote(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.promote_ontology_suggestions(args.suggestion_set_id, args.candidate_id or [], requested_scope)
    payload = base_response("cornerstone ontology promote", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _ontology_failure(payload, result)
        print_payload(payload, args.json)
        return exit_code
    change_set = result["ontology_change_set"]
    payload["ontology_suggestion_set"] = result["suggestion_set"]
    payload["ontology_objects"] = result["ontology_objects"]
    payload["ontology_links"] = result["ontology_links"]
    payload["ontology_change_set"] = change_set
    payload["ids"].update(
        {
            "ontology_suggestion_set_id": args.suggestion_set_id,
            "ontology_change_set_id": change_set["ontology_change_set_id"],
        }
    )
    if result["ontology_objects"]:
        payload["ids"]["ontology_object_id"] = result["ontology_objects"][0]["ontology_object_id"]
    payload["evidence_refs"].extend([f"ontology_change_set:{change_set['ontology_change_set_id']}", *change_set.get("evidence_refs", [])])
    payload["evidence_refs"].extend(f"ontology_object:{obj['ontology_object_id']}" for obj in result["ontology_objects"])
    payload["evidence_refs"].extend(f"ontology_link:{link['ontology_link_id']}" for link in result["ontology_links"])
    payload["audit_refs"].extend(f"audit:{event['event_id']}" for event in result.get("audit_events", []))
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_ontology_object_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.ontology_object_profile(args.object_id, requested_scope)
    payload = base_response("cornerstone ontology object show", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _ontology_failure(payload, result)
        print_payload(payload, args.json)
        return exit_code
    profile = result["profile"]
    payload["ontology_object_profile"] = profile
    payload["ids"].update({"ontology_object_id": args.object_id, "audit_event_id": result["audit_event"]["event_id"]})
    payload["evidence_refs"].extend([f"ontology_object:{args.object_id}", *profile.get("evidence_refs", [])])
    payload["audit_refs"].extend(profile.get("audit_refs", []))
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_ontology_draft_truth_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.deny_draft_ontology_truth_use(args.suggestion_set_id, args.candidate_id, requested_scope, purpose=args.purpose)
    payload = base_response("cornerstone ontology draft-truth-test", "denied", root)
    payload.update(requested_scope)
    if result.get("status") != "policy_denied":
        exit_code = _ontology_failure(payload, result)
        print_payload(payload, args.json)
        return exit_code
    payload["candidate"] = result["candidate"]
    payload["policy_decisions"] = [result["policy_decision"]]
    payload["policy_decision_refs"].append(f"policy:{result['policy_decision']['id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    payload["errors"].append({"code": "CS_ONTOLOGY_DRAFT_TRUTH_DENIED", "message": result["policy_decision"]["reason"]})
    print_payload(payload, args.json)
    return EXIT_POLICY_DENIED


def command_ontology_invalid_graph_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.reject_invalid_ontology_graph(requested_scope)
    payload = base_response("cornerstone ontology invalid-graph-test", "failed", root)
    payload.update(requested_scope)
    payload["errors"].append(result["error"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_INVALID


def command_ontology_supersede(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.supersede_ontology_object(
        args.object_id,
        requested_scope,
        property_name=args.property,
        corrected_value=args.to_value,
        rationale=args.rationale,
    )
    payload = base_response("cornerstone ontology supersede", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _ontology_failure(payload, result)
        print_payload(payload, args.json)
        return exit_code
    change_set = result["ontology_change_set"]
    payload["ontology_object"] = result["ontology_object"]
    payload["ontology_change_set"] = change_set
    payload["ids"].update({"ontology_object_id": args.object_id, "ontology_change_set_id": change_set["ontology_change_set_id"], "audit_event_id": result["audit_event"]["event_id"]})
    payload["evidence_refs"].extend([f"ontology_object:{args.object_id}", f"ontology_change_set:{change_set['ontology_change_set_id']}", *change_set.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_workspace_mode_set(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.set_workspace_mode(args.mode, requested_scope)
    workspace_mode = result["workspace_mode"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone workspace mode set", "success", root)
    payload.update(requested_scope)
    payload["workspace_mode"] = workspace_mode
    payload["ids"].update({"workspace_id": requested_scope["workspace_id"], "audit_event_id": audit_event["event_id"]})
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_workspace_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    workspace = store.workspace_detail(requested_scope)
    payload = base_response("cornerstone workspace show", "success", root)
    payload.update(requested_scope)
    payload["workspace"] = workspace
    payload["ids"].update({"workspace_id": requested_scope["workspace_id"]})
    payload["evidence_refs"].append(f"workspace:{requested_scope['workspace_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_workspace_mode_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    workspace_mode = store.get_workspace_mode(requested_scope)
    payload = base_response("cornerstone workspace mode show", "success", root)
    payload.update(requested_scope)
    payload["workspace_mode"] = workspace_mode
    payload["evidence_refs"].append(f"workspace:{requested_scope['workspace_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_autopilot_readiness(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    readiness = store.autopilot_readiness(requested_scope)
    payload = base_response("cornerstone autopilot readiness", "success", root)
    payload.update(requested_scope)
    payload["autopilot_readiness"] = readiness
    payload["ids"].update({"workspace_id": requested_scope["workspace_id"]})
    payload["evidence_refs"].append(f"workspace:{requested_scope['workspace_id']}:autopilot-readiness")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_create(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_memory_from_evidence_bundle(
        args.evidence_bundle_id,
        args.statement,
        requested_scope,
        trust_state=args.trust_state,
        status=args.status,
        memory_type=args.memory_type,
        synthesis_mode=args.synthesis_mode,
    )
    payload = base_response("cornerstone memory create", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_MEMORY_EVIDENCE_NOT_FOUND",
                "message": "Evidence Bundle for memory creation was not found.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Evidence Bundle for memory creation is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    if result.get("status") == "evidence_required":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_MEMORY_EVIDENCE_REQUIRED",
                "message": "Owner-approved memory requires an Evidence Bundle with at least one artifact reference.",
            }
        )
        print_payload(payload, args.json)
        return EXIT_EVIDENCE_MISSING

    memory = result["memory"]
    payload.update(memory["scope"])
    payload["memory"] = memory
    payload["ids"].update({"memory_id": memory["memory_id"], "evidence_bundle_id": args.evidence_bundle_id})
    payload["evidence_refs"].extend([f"memory:{memory['memory_id']}", *memory.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.show_memory(args.memory_id, requested_scope)
    payload = base_response("cornerstone memory show", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_MEMORY_NOT_FOUND", "message": "Memory record was not found.", "memory_id": args.memory_id})
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_SCOPE_DENIED", "message": "Memory record is outside the requested scope.", "resource_scope": result.get("resource_scope")})
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    memory = result["memory"]
    payload.update(memory["scope"])
    payload["memory"] = memory
    payload["ids"].update({"memory_id": memory["memory_id"]})
    payload["evidence_refs"].extend([f"memory:{memory['memory_id']}", *memory.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_raw_agent_note(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_raw_agent_memory(args.statement, requested_scope)
    memory = result["memory"]
    payload = base_response("cornerstone memory raw-agent-note", "success", root)
    payload.update(memory["scope"])
    payload["memory"] = memory
    payload["ids"].update({"memory_id": memory["memory_id"]})
    payload["evidence_refs"].append(f"memory:{memory['memory_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_conflict_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.resolve_memory_conflict(
        args.raw_memory_id,
        args.evidence_bundle_id,
        args.question,
        requested_scope,
    )
    payload = base_response("cornerstone memory conflict-test", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_MEMORY_CONFLICT_SOURCE_NOT_FOUND",
                "message": "Raw memory or Evidence Bundle for conflict resolution was not found.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Memory conflict source is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    if result.get("status") == "evidence_required":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_MEMORY_CONFLICT_EVIDENCE_REQUIRED",
                "message": "Memory conflict resolution requires archive evidence.",
            }
        )
        print_payload(payload, args.json)
        return EXIT_EVIDENCE_MISSING

    conflict = result["conflict"]
    payload.update(conflict["scope"])
    payload["memory_conflict_resolution"] = conflict
    payload["ids"].update(
        {
            "memory_conflict_id": conflict["conflict_id"],
            "raw_memory_id": args.raw_memory_id,
            "evidence_bundle_id": args.evidence_bundle_id,
        }
    )
    payload["evidence_refs"].extend([f"memory_conflict:{conflict['conflict_id']}", *conflict.get("answer", {}).get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_answer(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.answer_from_memory(args.question, requested_scope)
    answer = result["answer"]
    payload = base_response("cornerstone memory answer", "success" if answer["status"] == "answered" else "insufficient_evidence", root)
    payload.update(requested_scope)
    payload["memory_answer"] = answer
    payload["ids"].update({"memory_answer_id": answer["answer_id"]})
    payload["evidence_refs"].extend([f"memory_answer:{answer['answer_id']}", *answer.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    if result.get("policy_decision"):
        decision = result["policy_decision"]
        payload["policy_decisions"] = [decision]
        payload["policy_decision_refs"].append(f"policy:{decision['id']}")
    if answer["status"] != "answered":
        payload["errors"].append(
            {
                "code": "CS_MEMORY_ANSWER_EVIDENCE_REQUIRED",
                "message": "No owner-approved active-scope memory can answer the question.",
                "resolution_path": answer.get("policy_decision", {}).get("resolution_path", []),
            }
        )
        print_payload(payload, args.json)
        return EXIT_EVIDENCE_MISSING
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_wiki_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.permanent_wiki_view(requested_scope, wiki_kind=args.kind)
    wiki = result["wiki"]
    payload = base_response("cornerstone wiki show", "success", root)
    payload.update(requested_scope)
    payload["wiki"] = wiki
    payload["ids"].update({"wiki_id": wiki["wiki_id"]})
    payload["evidence_refs"].extend([f"wiki:{wiki['wiki_id']}", *[ref for entry in wiki.get("entries", []) for ref in entry.get("source_refs", [])]])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_control_center(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.memory_control_center(requested_scope)
    control = result["memory_control_center"]
    payload = base_response("cornerstone memory control-center", "success", root)
    payload.update(requested_scope)
    payload["memory_control_center"] = control
    payload["ids"].update({"memory_control_center_id": control["control_center_id"]})
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_temporary_session(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_temporary_memory_session(args.note, requested_scope)
    session = result["temporary_session"]
    payload = base_response("cornerstone memory temporary-session", "success", root)
    payload.update(requested_scope)
    payload["temporary_session"] = session
    payload["ids"].update({"temporary_session_id": session["temporary_session_id"]})
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_correct(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.correct_memory(
        args.memory_id,
        corrected_text=args.corrected_text,
        rationale=args.rationale,
        evidence_bundle_id=args.evidence_bundle_id,
        scope=requested_scope,
    )
    payload = base_response("cornerstone memory correct", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="MEMORY")
        print_payload(payload, args.json)
        return exit_code
    memory = result["memory"]
    correction = result["correction"]
    payload["memory"] = memory
    payload["correction"] = correction
    payload["ids"].update({"memory_id": memory["memory_id"], "correction_id": correction["correction_id"]})
    payload["evidence_refs"].extend(correction.get("evidence_refs", []))
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_control(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.control_memory(args.memory_id, action=args.action, scope=requested_scope)
    payload = base_response("cornerstone memory control", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="MEMORY_CONTROL")
        print_payload(payload, args.json)
        return exit_code
    control = result["memory_control_action"]
    memory = result["memory"]
    payload["memory_control_action"] = control
    payload["memory"] = memory
    payload["ids"].update({"memory_id": args.memory_id, "memory_control_action_id": control["memory_control_action_id"]})
    payload["evidence_refs"].append(f"memory:{args.memory_id}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_freshness(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.check_memory_freshness(args.memory_id, args.newer_evidence_bundle_id, requested_scope)
    payload = base_response("cornerstone memory freshness", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="MEMORY_FRESHNESS")
        print_payload(payload, args.json)
        return exit_code
    memory = result["memory"]
    payload["memory"] = memory
    payload["ids"].update({"memory_id": memory["memory_id"], "newer_evidence_bundle_id": args.newer_evidence_bundle_id})
    payload["evidence_refs"].extend([f"memory:{memory['memory_id']}", f"evidence_bundle:{args.newer_evidence_bundle_id}", *memory.get("freshness", {}).get("newer_evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_quarantine_check(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.quarantine_memory_attempt(args.artifact_id, args.statement, requested_scope)
    payload = base_response("cornerstone memory quarantine-check", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="MEMORY_QUARANTINE")
        print_payload(payload, args.json)
        return exit_code
    quarantine = result["memory_quarantine"]
    payload["memory_quarantine"] = quarantine
    payload["ids"].update({"memory_quarantine_id": quarantine["memory_quarantine_id"], "artifact_id": args.artifact_id})
    payload["evidence_refs"].extend(quarantine.get("evidence_refs", []))
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_export(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.export_memory(requested_scope)
    export = result["memory_export"]
    payload = base_response("cornerstone memory export", "success", root)
    payload.update(requested_scope)
    payload["memory_export"] = export
    payload["ids"].update({"memory_export_id": export["memory_export_id"]})
    payload["evidence_refs"].extend([f"memory_export:{export['memory_export_id']}", *[f"memory:{entry['memory_id']}" for entry in export.get("entries", [])]])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_memory_adapt(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.record_memory_adaptation(args.preference, requested_scope, source_memory_id=args.source_memory_id)
    payload = base_response("cornerstone memory adapt", "success", root)
    payload.update(requested_scope)
    if result.get("status"):
        exit_code = _record_command_failure(payload, result, resource_label="MEMORY_ADAPTATION")
        print_payload(payload, args.json)
        return exit_code
    adaptation = result["memory_adaptation"]
    payload["memory_adaptation"] = adaptation
    payload["ids"].update({"memory_adaptation_id": adaptation["memory_adaptation_id"]})
    payload["evidence_refs"].extend([f"memory_adaptation:{adaptation['memory_adaptation_id']}", *adaptation.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_namespace_promote(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    source_scope = scope_args(args)
    target_scope = {
        "tenant_id": args.target_tenant_id or source_scope["tenant_id"],
        "owner_id": args.target_owner_id,
        "namespace_id": args.target_namespace_id,
        "workspace_id": args.target_workspace_id,
    }
    payload = base_response("cornerstone namespace promote", "success", root)
    payload.update(target_scope)
    if args.source_kind != "memory":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_NAMESPACE_PROMOTION_KIND_UNSUPPORTED",
                "message": "The local scaffold currently supports memory promotion only.",
                "source_kind": args.source_kind,
            }
        )
        print_payload(payload, args.json)
        return EXIT_INVALID

    result = store.promote_memory_to_namespace(
        args.source_id,
        source_scope,
        target_scope,
        mode=args.mode,
        principal_id=args.principal_id,
        principal_role=args.principal_role,
    )
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_NAMESPACE_PROMOTION_SOURCE_NOT_FOUND",
                "message": "Promotion source was not found.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Promotion source is outside the requested source scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    if result.get("status") == "evidence_required":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_NAMESPACE_PROMOTION_EVIDENCE_REQUIRED",
                "message": "Explicit promotion requires an owner-approved evidence-backed source item.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_EVIDENCE_MISSING
    if result.get("status") == "policy_denied":
        decision = result["policy_decision"]
        payload["status"] = "denied"
        payload["policy_decisions"] = [decision]
        payload["policy_decision_refs"].append(f"policy:{decision['id']}")
        payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
        payload["errors"].append(
            {
                "code": "CS_NAMESPACE_PROMOTION_POLICY_DENIED",
                "message": decision["reason"],
                "resolution_path": decision["resolution_path"],
            }
        )
        print_payload(payload, args.json)
        return EXIT_POLICY_DENIED

    promotion = result["promotion"]
    promoted_memory = result["promoted_memory"]
    decision = result["policy_decision"]
    payload["namespace_promotion"] = promotion
    payload["promoted_item"] = promoted_memory
    payload["policy_decisions"] = [decision]
    payload["ids"].update(
        {
            "namespace_promotion_id": promotion["promotion_id"],
            "source_memory_id": args.source_id,
            "target_memory_id": promoted_memory["memory_id"],
        }
    )
    payload["evidence_refs"].extend(promotion.get("evidence_refs", []))
    payload["policy_decision_refs"].append(f"policy:{decision['id']}")
    payload["audit_refs"].extend(f"audit:{event['event_id']}" for event in result["audit_events"])
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_namespace_audit_export(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    event_types = args.event_type or []
    result = store.query_namespace_audit(requested_scope, event_types=event_types)
    export = result["namespace_audit_export"]
    payload = base_response("cornerstone namespace audit-export", "success", root)
    payload.update(requested_scope)
    payload["namespace_audit_export"] = export
    payload["ids"].update({"namespace_audit_export_id": export["namespace_audit_export_id"]})
    payload["evidence_refs"].append(f"namespace_audit_export:{export['namespace_audit_export_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_namespace_recovery_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.recover_namespace_boundary(args.promotion_id, requested_scope, reason=args.reason)
    payload = base_response("cornerstone namespace recovery-test", "success", root)
    payload.update(requested_scope)
    if result.get("status") in {"not_found", "scope_denied", "evidence_required"}:
        exit_code = _record_command_failure(payload, result, resource_label="NAMESPACE_RECOVERY")
        print_payload(payload, args.json)
        return exit_code
    recovery = result["namespace_recovery"]
    payload["namespace_recovery"] = recovery
    payload["ids"].update({"namespace_recovery_id": recovery["recovery_id"], "namespace_promotion_id": args.promotion_id})
    payload["evidence_refs"].extend([f"namespace_recovery:{recovery['recovery_id']}", f"namespace_promotion:{args.promotion_id}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_namespace_product_learning_boundary_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.check_product_learning_boundary(requested_scope)
    boundary = result["product_learning_boundary"]
    payload = base_response("cornerstone namespace product-learning-boundary-test", "success", root)
    payload.update(requested_scope)
    payload["product_learning_boundary"] = boundary
    payload["ids"].update({"product_learning_boundary_id": boundary["product_learning_boundary_id"]})
    payload["evidence_refs"].append(f"product_learning_boundary:{boundary['product_learning_boundary_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_access_evaluate(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    resource_scope = {
        "tenant_id": args.resource_tenant_id or requested_scope["tenant_id"],
        "owner_id": args.resource_owner_id or requested_scope["owner_id"],
        "namespace_id": args.resource_namespace_id or requested_scope["namespace_id"],
        "workspace_id": args.resource_workspace_id or requested_scope["workspace_id"],
    }
    principal_attributes = [attribute.strip() for attribute in args.principal_attributes.split(",") if attribute.strip()]
    result = store.evaluate_access(
        principal_id=args.principal_id,
        principal_role=args.principal_role,
        principal_attributes=principal_attributes,
        action=args.action,
        resource_kind=args.resource_kind,
        resource_id=args.resource_id,
        resource_scope=resource_scope,
        classification=args.classification,
        mission_authority=args.mission_authority,
        scope=requested_scope,
    )
    decision = result["policy_decision"]
    status = "allowed" if decision["decision"] == "allow" else "denied"
    payload = base_response("cornerstone access evaluate", status, root)
    payload.update(requested_scope)
    payload["access_decision"] = decision
    payload["policy_decisions"] = [decision]
    payload["policy_decision_refs"].append(f"policy:{decision['id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    payload["ids"].update({"policy_decision_id": decision["id"], "resource_id": args.resource_id})
    if decision["decision"] != "allow":
        payload["errors"].append(
            {
                "code": "CS_ACCESS_POLICY_DENIED",
                "message": decision["reason"],
                "resolution_path": decision["resolution_path"],
            }
        )
    print_payload(payload, args.json)
    return EXIT_SUCCESS if decision["decision"] == "allow" else EXIT_POLICY_DENIED


def command_source_readonly_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.verify_source_readonly_ingest(args.artifact_id, requested_scope, source_system=args.source_system)
    payload = base_response("cornerstone source readonly-test", "success", root)
    payload.update(requested_scope)
    if result.get("status") in {"not_found", "scope_denied", "evidence_required"}:
        exit_code = _record_command_failure(payload, result, resource_label="SOURCE_READONLY")
        print_payload(payload, args.json)
        return exit_code
    safety = result["source_safety"]
    payload["source_safety"] = safety
    payload["ids"].update({"source_safety_id": safety["source_safety_id"], "artifact_id": args.artifact_id})
    payload["evidence_refs"].extend([f"source_safety:{safety['source_safety_id']}", f"artifact:{args.artifact_id}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_learning_record(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.record_learning_from_action(args.action_id, args.lesson, requested_scope)
    payload = base_response("cornerstone learning record", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_LEARNING_SOURCE_NOT_FOUND",
                "message": "Executed Action source for learning was not found.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Executed Action source for learning is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    if result.get("status") == "evidence_required":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_LEARNING_ACTION_RESULT_REQUIRED",
                "message": "Learning record requires a successfully executed Action result.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_EVIDENCE_MISSING

    learning = result["learning"]
    payload.update(learning["scope"])
    payload["learning"] = learning
    payload["ids"].update({"learning_id": learning["learning_id"], "action_id": args.action_id})
    payload["evidence_refs"].extend([f"learning:{learning['learning_id']}", *learning.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_learning_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.show_learning(args.learning_id, requested_scope)
    payload = base_response("cornerstone learning show", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_LEARNING_NOT_FOUND", "message": "Learning record was not found.", "learning_id": args.learning_id})
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_SCOPE_DENIED", "message": "Learning record is outside the requested scope.", "resource_scope": result.get("resource_scope")})
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    learning = result["learning"]
    payload.update(learning["scope"])
    payload["learning"] = learning
    payload["ids"].update({"learning_id": learning["learning_id"]})
    payload["evidence_refs"].extend([f"learning:{learning['learning_id']}", *learning.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def experience_payload(args: argparse.Namespace, command: str) -> tuple[Path, LocalRuntimeStore, dict[str, str], dict[str, Any]]:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    payload = base_response(command, "success", root)
    payload.update(requested_scope)
    return root, store, requested_scope, payload


def handle_experience_error(payload: dict[str, Any], result: dict[str, Any], as_json: bool) -> int | None:
    status = result.get("status")
    if not status:
        return None
    payload["status"] = "failed"
    if status == "not_found":
        payload["errors"].append(
            {
                "code": "CS_EXPERIENCE_RESOURCE_NOT_FOUND",
                "message": "Required experience resource was not found.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_NOT_FOUND
    if status == "scope_denied":
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Experience resource is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_SCOPE_DENIED
    if status == "evidence_required":
        payload["errors"].append(
            {
                "code": "CS_EXPERIENCE_EVIDENCE_REQUIRED",
                "message": "Experience operation requires evidence-backed source data.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_EVIDENCE_MISSING
    if status == "invalid_stage":
        payload["errors"].append(
            {
                "code": "CS_EXPERIENCE_INVALID_PROMOTION_STAGE",
                "message": "Lesson promotion must follow the declared promotion ladder.",
                "resource": result.get("resource"),
                "reason": result.get("reason"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_INVALID
    if status == "invalid_action":
        payload["errors"].append(
            {
                "code": "CS_EXPERIENCE_INVALID_CONTROL_ACTION",
                "message": "Lesson control action is not supported.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_INVALID
    if status == "approval_required":
        payload["errors"].append(
            {
                "code": "CS_EXPERIENCE_APPROVAL_REQUIRED",
                "message": "Broader lesson reuse requires an explicit staged approval record.",
                "resource": result.get("resource"),
                "stage": result.get("stage"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_POLICY_DENIED
    payload["errors"].append({"code": "CS_EXPERIENCE_RUNTIME_ERROR", "message": "Experience operation failed.", "status": status})
    print_payload(payload, as_json)
    return EXIT_RUNTIME_FAILURE


def command_experience_connected_outcome(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience connected-outcome")
    result = store.record_connected_outcome(
        args.action_id,
        evidence_bundle_id=args.evidence_bundle_id,
        outcome_status=args.outcome_status,
        summary=args.summary,
        scope=requested_scope,
    )
    error_exit = handle_experience_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    connected_outcome = result["connected_outcome"]
    payload["connected_outcome"] = connected_outcome
    payload["ids"].update({"connected_outcome_id": connected_outcome["connected_outcome_id"], "action_id": args.action_id})
    payload["evidence_refs"].extend([f"connected_outcome:{connected_outcome['connected_outcome_id']}", *connected_outcome.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_trajectory_record(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience trajectory record")
    result = store.record_mission_trajectory(
        args.mission_id,
        outcome_status=args.outcome_status,
        outcome_summary=args.outcome_summary,
        owner_acceptance=args.owner_acceptance,
        failure_reason=args.failure_reason,
        recovery_attempt=args.recovery_attempt,
        connected_outcome_id=args.connected_outcome_id,
        scope=requested_scope,
    )
    error_exit = handle_experience_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    trajectory = result["trajectory"]
    payload["trajectory"] = trajectory
    payload["ids"].update({"trajectory_id": trajectory["trajectory_id"], "mission_id": args.mission_id})
    payload["evidence_refs"].extend([f"trajectory:{trajectory['trajectory_id']}", *trajectory.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_library(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience library")
    result = store.experience_library(requested_scope)
    library = result["experience_library"]
    payload["experience_library"] = library
    payload["ids"].update({"experience_library_id": library["experience_library_id"]})
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_search(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience search")
    result = store.search_experience(args.query, requested_scope)
    search = result["experience_search"]
    payload["experience_search"] = search
    payload["ids"].update({"experience_search_id": search["experience_search_id"]})
    payload["evidence_refs"].extend(ref for row in search.get("results", []) for ref in row.get("evidence_refs", []))
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_recommend(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience recommend")
    result = store.recommend_experience(args.mission_id, args.query, requested_scope)
    error_exit = handle_experience_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    recommendation = result["experience_recommendation"]
    payload["experience_recommendation"] = recommendation
    payload["ids"].update({"experience_recommendation_id": recommendation["experience_recommendation_id"], "mission_id": args.mission_id})
    payload["evidence_refs"].extend(ref for row in recommendation.get("cited_experiences", []) for ref in row.get("evidence_refs", []))
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_lesson_propose(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience lesson propose")
    result = store.propose_lesson(
        args.trajectory_id,
        lesson=args.lesson,
        applies_when=args.applies_when,
        does_not_apply_when=args.does_not_apply_when,
        confidence=args.confidence,
        scope=requested_scope,
    )
    error_exit = handle_experience_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    lesson = result["lesson"]
    payload["lesson"] = lesson
    payload["ids"].update({"lesson_id": lesson["lesson_id"], "trajectory_id": args.trajectory_id})
    payload["evidence_refs"].extend([f"lesson:{lesson['lesson_id']}", *lesson.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_lesson_promote(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience lesson promote")
    result = store.promote_lesson(args.lesson_id, stage=args.stage, scope=requested_scope)
    error_exit = handle_experience_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    lesson = result["lesson"]
    payload["lesson"] = lesson
    payload["ids"].update({"lesson_id": lesson["lesson_id"]})
    payload["evidence_refs"].extend([f"lesson:{lesson['lesson_id']}", *lesson.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_lesson_control(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience lesson control")
    result = store.control_lesson(args.lesson_id, action=args.action, scope=requested_scope)
    error_exit = handle_experience_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    lesson = result["lesson"]
    control = result["lesson_control"]
    payload["lesson"] = lesson
    payload["lesson_control"] = control
    payload["ids"].update({"lesson_id": lesson["lesson_id"], "lesson_control_id": control["lesson_control_id"]})
    payload["evidence_refs"].extend([f"lesson:{lesson['lesson_id']}", f"lesson_control:{control['lesson_control_id']}", *lesson.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_behavior_signal(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience behavior-signal")
    result = store.record_behavior_signal(args.trajectory_id, signal=args.signal, interpretation=args.interpretation, scope=requested_scope)
    error_exit = handle_experience_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    signal = result["behavior_signal"]
    payload["behavior_signal"] = signal
    payload["ids"].update({"behavior_signal_id": signal["behavior_signal_id"], "trajectory_id": args.trajectory_id})
    payload["evidence_refs"].extend([f"behavior_signal:{signal['behavior_signal_id']}", *signal.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_model_eval(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience model-eval")
    result = store.record_model_evaluation(args.trajectory_id, score=args.score, rationale=args.rationale, scope=requested_scope)
    error_exit = handle_experience_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    evaluation = result["model_evaluation"]
    payload["model_evaluation"] = evaluation
    payload["ids"].update({"model_evaluation_id": evaluation["model_evaluation_id"], "trajectory_id": args.trajectory_id})
    payload["evidence_refs"].extend([f"model_evaluation:{evaluation['model_evaluation_id']}", *evaluation.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_product_improvement_propose(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience product-improvement propose")
    result = store.propose_product_improvement(args.lesson_id, proposal=args.proposal, scope=requested_scope)
    error_exit = handle_experience_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    improvement = result["product_improvement"]
    payload["product_improvement"] = improvement
    payload["ids"].update({"product_improvement_id": improvement["product_improvement_id"], "lesson_id": args.lesson_id})
    payload["evidence_refs"].extend([f"product_improvement:{improvement['product_improvement_id']}", *improvement.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_local_adapt(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience local-adapt")
    result = store.record_local_adaptation(args.lesson_id, preference=args.preference, scope=requested_scope)
    error_exit = handle_experience_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    adaptation = result["local_adaptation"]
    payload["local_adaptation"] = adaptation
    payload["ids"].update({"local_adaptation_id": adaptation["local_adaptation_id"], "lesson_id": args.lesson_id})
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_local_adapt_reset(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience local-adapt-reset")
    result = store.reset_local_adaptation(args.adaptation_id, requested_scope)
    error_exit = handle_experience_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    adaptation = result["local_adaptation"]
    payload["local_adaptation"] = adaptation
    payload["ids"].update({"local_adaptation_id": adaptation["local_adaptation_id"]})
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_metrics(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience metrics")
    result = store.outcome_quality_metrics(requested_scope)
    report = result["outcome_quality_report"]
    payload["outcome_quality_report"] = report
    payload["ids"].update({"outcome_quality_report_id": report["outcome_quality_report_id"]})
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_experience_export(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = experience_payload(args, "cornerstone experience export")
    result = store.export_experience(requested_scope)
    export = result["experience_export"]
    payload["experience_export"] = export
    payload["ids"].update({"experience_export_id": export["experience_export_id"]})
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def pack_payload(args: argparse.Namespace, command: str) -> tuple[Path, LocalRuntimeStore, dict[str, str], dict[str, Any]]:
    root = repo_root()
    requested_scope = scope_args(args)
    payload = base_response(command, "success", root)
    payload.update(requested_scope)
    return root, LocalRuntimeStore(state_dir(root, args)), requested_scope, payload


def handle_pack_error(payload: dict[str, Any], result: dict[str, Any], as_json: bool) -> int | None:
    status = result.get("status")
    if status is None:
        return None
    if status == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_PACK_RESOURCE_NOT_FOUND",
                "message": "Required Agent Pack resource was not found.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_NOT_FOUND
    if status == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_PACK_SCOPE_DENIED",
                "message": "Agent Pack resource is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_SCOPE_DENIED
    if status == "evidence_required":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_PACK_EVIDENCE_REQUIRED",
                "message": "Agent Pack operation requires certification or evidence that is missing.",
                "resource": result.get("resource"),
                "missing": result.get("missing", []),
            }
        )
        print_payload(payload, as_json)
        return EXIT_EVIDENCE_MISSING
    if status == "policy_denied":
        payload["status"] = "denied"
        decision = result.get("policy_decision", {})
        payload["policy_decisions"] = [decision]
        payload["policy_decision_refs"].append(f"policy:{decision.get('policy_decision_id')}")
        if result.get("audit_event"):
            payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
        payload["errors"].append(
            {
                "code": "CS_PACK_POLICY_DENIED",
                "message": decision.get("reason", "Agent Pack operation denied by policy."),
                "policy": decision.get("policy"),
                "resolution_path": decision.get("resolution_path", []),
            }
        )
        print_payload(payload, as_json)
        return EXIT_POLICY_DENIED
    if status == "approval_required":
        payload["status"] = "failed"
        decision = result.get("policy_decision", {})
        payload["policy_decisions"] = [decision]
        payload["policy_decision_refs"].append(f"policy:{decision.get('policy_decision_id')}")
        if result.get("audit_event"):
            payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
        payload["errors"].append(
            {
                "code": "CS_PACK_APPROVAL_REQUIRED",
                "message": decision.get("reason", "Agent Pack operation requires approval."),
                "policy": decision.get("policy"),
                "resolution_path": decision.get("resolution_path", []),
            }
        )
        print_payload(payload, as_json)
        return EXIT_POLICY_DENIED
    if status == "invalid":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_PACK_INVALID",
                "message": result.get("message", "Agent Pack input is invalid."),
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_INVALID
    return None


def command_pack_import(args: argparse.Namespace) -> int:
    root, store, requested_scope, payload = pack_payload(args, "cornerstone pack import")
    result = store.register_agent_pack((root / args.manifest).resolve(), requested_scope)
    if result.get("status") == "quarantined":
        payload["status"] = "failed"
        payload["quarantine"] = result["quarantine"]
        payload["policy_decisions"] = [result["policy_decision"]]
        payload["ids"].update({"pack_id": result["quarantine"]["pack_id"], "quarantine_id": result["quarantine"]["quarantine_id"]})
        payload["policy_decision_refs"].append(f"policy:{result['policy_decision']['policy_decision_id']}")
        payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
        payload["errors"].append(
            {
                "code": "CS_PACK_REGISTRY_VALIDATION_FAILED",
                "message": "Agent Pack was quarantined by registry validation.",
                "resolution_path": result["policy_decision"].get("resolution_path", []),
            }
        )
        print_payload(payload, args.json)
        return EXIT_POLICY_DENIED
    error_exit = handle_pack_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    pack = result["agent_pack"]
    payload["agent_pack"] = pack
    payload["ids"].update({"pack_id": pack["pack_id"]})
    payload["evidence_refs"].extend([f"agent_pack:{pack['pack_id']}", f"manifest:{pack['source_digest']}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_pack_list(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = pack_payload(args, "cornerstone pack list")
    result = store.list_agent_packs(requested_scope)
    payload["registry"] = result["registry"]
    payload["ids"].update({"registry_id": "local"})
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_pack_show(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = pack_payload(args, "cornerstone pack show")
    result = store.show_agent_pack(args.pack_id, requested_scope)
    error_exit = handle_pack_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    detail = result["agent_pack_detail"]
    payload["agent_pack_detail"] = detail
    payload["ids"].update({"pack_id": args.pack_id})
    payload["evidence_refs"].append(f"agent_pack:{args.pack_id}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_pack_install(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = pack_payload(args, "cornerstone pack install")
    result = store.install_agent_pack(args.pack_id, version=args.version, dry_run=args.dry_run, scope=requested_scope)
    error_exit = handle_pack_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    install = result["install"]
    payload["install"] = install
    payload["ids"].update({"pack_id": args.pack_id, "install_id": install["install_id"]})
    payload["evidence_refs"].append(f"agent_pack:{args.pack_id}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_pack_activate(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = pack_payload(args, "cornerstone pack activate")
    result = store.activate_agent_pack(
        args.pack_id,
        grants=args.grant or [],
        mission_id=args.mission_id,
        org_admin_shortcut=args.org_admin_shortcut,
        policy_id=args.policy_id,
        scope=requested_scope,
    )
    error_exit = handle_pack_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    activation = result["activation"]
    decision = result["policy_decision"]
    payload["activation"] = activation
    payload["ids"].update({"pack_id": args.pack_id, "activation_id": activation["activation_id"]})
    payload["evidence_refs"].append(f"agent_pack:{args.pack_id}")
    payload["policy_decisions"] = [decision]
    payload["policy_decision_refs"].append(f"policy:{decision['policy_decision_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_pack_certify(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = pack_payload(args, "cornerstone pack certify")
    result = store.certify_agent_pack(args.pack_id, requested_scope)
    error_exit = handle_pack_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    certification = result["certification"]
    payload["certification"] = certification
    payload["ids"].update({"pack_id": args.pack_id, "certification_id": certification["certification_id"]})
    payload["evidence_refs"].append(f"agent_pack:{args.pack_id}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_pack_connector_request(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = pack_payload(args, "cornerstone pack connector-request")
    result = store.request_pack_connector_access(args.pack_id, capability=args.capability, scope=requested_scope)
    error_exit = handle_pack_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    request = result["connector_request"]
    payload["connector_request"] = request
    payload["ids"].update({"pack_id": args.pack_id, "connector_request_id": request["connector_request_id"]})
    payload["evidence_refs"].append(f"agent_pack:{args.pack_id}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_pack_playbook_propose(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = pack_payload(args, "cornerstone pack playbook propose")
    result = store.propose_pack_playbook_update(args.pack_id, lesson_id=args.lesson_id, scope=requested_scope)
    error_exit = handle_pack_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    proposal = result["playbook_proposal"]
    payload["playbook_proposal"] = proposal
    payload["ids"].update({"pack_id": args.pack_id, "playbook_proposal_id": proposal["playbook_proposal_id"]})
    payload["evidence_refs"].extend([f"agent_pack:{args.pack_id}", *proposal.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_pack_playbook_approve(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = pack_payload(args, "cornerstone pack playbook approve")
    result = store.approve_pack_playbook_update(args.proposal_id, requested_scope)
    error_exit = handle_pack_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    proposal = result["playbook_proposal"]
    payload["playbook_proposal"] = proposal
    payload["ids"].update({"playbook_proposal_id": proposal["playbook_proposal_id"], "pack_id": proposal["pack_id"]})
    payload["evidence_refs"].extend(proposal.get("evidence_refs", []))
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_pack_update(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = pack_payload(args, "cornerstone pack update")
    result = store.update_agent_pack(args.pack_id, to_version=args.to_version, dry_run=args.dry_run, approve=args.approve, scope=requested_scope)
    error_exit = handle_pack_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    update = result["pack_update"]
    payload["pack_update"] = update
    payload["ids"].update({"pack_id": args.pack_id, "update_id": update["update_id"]})
    payload["evidence_refs"].append(f"agent_pack:{args.pack_id}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_pack_rollback(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = pack_payload(args, "cornerstone pack rollback")
    result = store.rollback_agent_pack(args.pack_id, to_version=args.to_version, reason=args.reason, scope=requested_scope)
    error_exit = handle_pack_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    rollback = result["pack_rollback"]
    payload["pack_rollback"] = rollback
    payload["ids"].update({"pack_id": args.pack_id, "rollback_id": rollback["rollback_id"]})
    payload["evidence_refs"].append(f"agent_pack:{args.pack_id}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_pack_emergency_patch(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = pack_payload(args, "cornerstone pack emergency-patch")
    result = store.emergency_patch_agent_pack(args.pack_id, patch_version=args.patch_version, behavior_change=args.behavior_change, scope=requested_scope)
    error_exit = handle_pack_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    patch = result["security_patch"]
    payload["security_patch"] = patch
    payload["ids"].update({"pack_id": args.pack_id, "security_patch_id": patch["security_patch_id"]})
    payload["evidence_refs"].append(f"agent_pack:{args.pack_id}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def agent_payload(args: argparse.Namespace, command: str) -> tuple[Path, LocalRuntimeStore, dict[str, str], dict[str, Any]]:
    root = repo_root()
    requested_scope = scope_args(args)
    payload = base_response(command, "success", root)
    payload.update(requested_scope)
    return root, LocalRuntimeStore(state_dir(root, args)), requested_scope, payload


def handle_agent_error(payload: dict[str, Any], result: dict[str, Any], as_json: bool) -> int | None:
    status = result.get("status")
    if status is None:
        return None
    if status == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_AGENT_RESOURCE_NOT_FOUND",
                "message": "Required agent resource was not found.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_NOT_FOUND
    if status == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_AGENT_SCOPE_DENIED",
                "message": "Agent resource is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_SCOPE_DENIED
    if status == "policy_denied":
        payload["status"] = "denied"
        decision = result.get("policy_decision", {})
        payload["policy_decisions"] = [decision]
        payload["policy_decision_refs"].append(f"policy:{decision.get('policy_decision_id')}")
        for key in ("mutation_attempt", "authority_attempt", "capability_attempt"):
            if key in result:
                payload[key] = result[key]
                record_id = result[key].get("attempt_id") or result[key].get("capability_attempt_id")
                if record_id:
                    payload["ids"][key] = record_id
        if result.get("audit_event"):
            payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
        payload["errors"].append(
            {
                "code": "CS_AGENT_POLICY_DENIED",
                "message": decision.get("reason", "Agent operation denied by policy."),
                "policy": decision.get("policy"),
                "resolution_path": decision.get("resolution_path", []),
            }
        )
        print_payload(payload, as_json)
        return EXIT_POLICY_DENIED
    return None


def command_agent_list(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = agent_payload(args, "cornerstone agent list")
    result = store.list_agent_roles(requested_scope)
    payload["agent_roles"] = result["agent_roles"]
    payload["ids"].update({"agent_role_registry_id": "local"})
    payload["evidence_refs"].extend([f"agent_role:{role['role_id']}" for role in result["agent_roles"].get("roles", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_role_show(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = agent_payload(args, "cornerstone role show")
    result = store.show_agent_role(args.role_id, requested_scope, view=args.view)
    error_exit = handle_agent_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    role_view = result["agent_role_view"]
    payload["agent_role_view"] = role_view
    payload["ids"].update({"role_id": role_view["role_id"]})
    payload["evidence_refs"].append(f"agent_role:{role_view['role_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_agent_orchestrate(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = agent_payload(args, "cornerstone agent orchestrate")
    result = store.create_agent_mission_trace(args.mission_id, requested_scope)
    error_exit = handle_agent_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    trace = result["agent_trace"]
    payload["agent_trace"] = trace
    payload["ids"].update({"trace_id": trace["trace_id"], "mission_id": trace["mission_id"]})
    payload["evidence_refs"].extend([f"mission:{trace['mission_id']}", f"agent_trace:{trace['trace_id']}", *trace.get("outputs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    payload["audit_refs"].append(f"audit:{result['review_audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_agent_direct_mutation_test(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = agent_payload(args, "cornerstone agent direct-mutation-test")
    result = store.test_agent_direct_mutation(args.role_id, target=args.target, scope=requested_scope)
    error_exit = handle_agent_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_agent_brain_switch(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = agent_payload(args, "cornerstone agent brain-switch")
    result = store.switch_agent_brain(args.role_id, provider=args.provider, model=args.model, scope=requested_scope)
    error_exit = handle_agent_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    switch = result["brain_switch"]
    payload["brain_switch"] = switch
    payload["ids"].update({"role_id": switch["role_id"], "switch_id": switch["switch_id"]})
    payload["evidence_refs"].append(f"agent_role:{switch['role_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_agent_contract_update(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = agent_payload(args, "cornerstone agent contract-update")
    result = store.update_agent_contract(args.role_id, change_summary=args.change_summary, scope=requested_scope)
    error_exit = handle_agent_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    update = result["contract_update"]
    payload["contract_update"] = update
    payload["agent_role"] = result["agent_role"]
    payload["ids"].update({"role_id": update["role_id"], "contract_update_id": update["update_id"]})
    payload["evidence_refs"].append(f"agent_role:{update['role_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_agent_prompt_authority_test(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = agent_payload(args, "cornerstone agent prompt-authority-test")
    result = store.test_agent_prompt_authority_expansion(
        args.role_id,
        requested_tool=args.requested_tool,
        requested_memory_scope=args.requested_memory_scope,
        requested_authority=args.requested_authority,
        scope=requested_scope,
    )
    error_exit = handle_agent_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_agent_diagnose(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = agent_payload(args, "cornerstone agent diagnose")
    result = store.record_agent_failure(args.trace_id, args.role_id, failure_kind=args.failure_kind, scope=requested_scope)
    error_exit = handle_agent_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    diagnosis = result["diagnosis"]
    payload["diagnosis"] = diagnosis
    payload["ids"].update({"trace_id": diagnosis["trace_id"], "diagnosis_id": diagnosis["diagnosis_id"], "role_id": diagnosis["role_id"]})
    payload["evidence_refs"].extend([f"agent_trace:{diagnosis['trace_id']}", f"agent_diagnosis:{diagnosis['diagnosis_id']}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_agent_pack_capability_test(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = agent_payload(args, "cornerstone agent pack-capability-test")
    result = store.test_agent_pack_capability(args.role_id, pack_id=args.pack_id, capability=args.capability, scope=requested_scope)
    error_exit = handle_agent_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    attempt = result["capability_attempt"]
    payload["capability_attempt"] = attempt
    payload["ids"].update({"role_id": attempt["role_id"], "pack_id": attempt["pack_id"], "capability_attempt_id": attempt["capability_attempt_id"]})
    payload["evidence_refs"].extend([f"agent_role:{attempt['role_id']}", f"agent_pack:{attempt['pack_id']}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_agent_replay(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = agent_payload(args, "cornerstone agent replay")
    result = store.replay_agent_mission(args.trace_id, requested_scope)
    error_exit = handle_agent_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    replay = result["agent_replay"]
    payload["agent_replay"] = replay
    payload["ids"].update({"trace_id": replay["trace_id"], "replay_id": replay["replay_id"], "mission_id": replay["mission_id"]})
    payload["evidence_refs"].extend([*replay.get("trace_refs", []), *replay.get("role_contract_refs", []), *replay.get("output_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def brain_payload(args: argparse.Namespace, command: str) -> tuple[Path, LocalRuntimeStore, dict[str, str], dict[str, Any]]:
    root = repo_root()
    requested_scope = scope_args(args)
    payload = base_response(command, "success", root)
    payload.update(requested_scope)
    return root, LocalRuntimeStore(state_dir(root, args)), requested_scope, payload


def handle_brain_error(payload: dict[str, Any], result: dict[str, Any], as_json: bool) -> int | None:
    status = result.get("status")
    if status is None:
        return None
    if status == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_BRAIN_RESOURCE_NOT_FOUND",
                "message": "Required model, brain, judge, or evidence resource was not found.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_NOT_FOUND
    if status == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_BRAIN_SCOPE_DENIED",
                "message": "Brain or judge resource is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, as_json)
        return EXIT_SCOPE_DENIED
    if status == "policy_denied":
        payload["status"] = "denied"
        decision = result.get("policy_decision", {})
        payload["policy_decisions"] = [decision]
        if decision.get("policy_decision_id"):
            payload["policy_decision_refs"].append(f"policy:{decision['policy_decision_id']}")
        for key in ("aggregation",):
            if key in result:
                payload[key] = result[key]
                record_id = result[key].get("aggregation_id")
                if record_id:
                    payload["ids"][key] = record_id
        if result.get("audit_event"):
            payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
        payload["errors"].append(
            {
                "code": "CS_BRAIN_POLICY_DENIED",
                "message": decision.get("reason", "Brain or judge operation denied by policy."),
                "policy": decision.get("policy"),
                "resolution_path": decision.get("resolution_path", []),
            }
        )
        print_payload(payload, as_json)
        return EXIT_POLICY_DENIED
    return None


def command_model_list(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = brain_payload(args, "cornerstone model list")
    result = store.list_models(requested_scope)
    registry = result["model_registry"]
    payload["model_registry"] = registry
    payload["ids"].update({"model_registry_id": "local"})
    payload["evidence_refs"].append("model_registry:local")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_brain_route(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = brain_payload(args, "cornerstone brain route")
    result = store.route_brain(
        task_ref=args.task,
        task_type=args.task_type,
        mission_type=args.mission_type,
        sensitivity=args.sensitivity,
        risk=args.risk,
        owner_preference=args.owner_preference,
        max_cost_usd=args.max_cost_usd,
        max_latency_ms=args.max_latency_ms,
        override_provider=args.override_provider,
        override_model=args.override_model,
        dry_run=args.dry_run,
        scope=requested_scope,
    )
    error_exit = handle_brain_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    route = result["routing_decision"]
    ledger = result["ledger_entry"]
    payload["routing_decision"] = route
    payload["ledger_entry"] = ledger
    payload["ids"].update({"route_id": route["route_id"], "ledger_id": ledger["ledger_id"]})
    payload["evidence_refs"].extend([f"brain_route:{route['route_id']}", f"brain_ledger:{ledger['ledger_id']}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    payload["audit_refs"].append(f"audit:{result['ledger_audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_brain_switch(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = brain_payload(args, "cornerstone brain switch")
    result = store.switch_workspace_brain(
        provider=args.provider,
        model=args.model,
        evidence_bundle_id=args.evidence_bundle_id,
        mission_id=args.mission_id,
        scope=requested_scope,
    )
    error_exit = handle_brain_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    switch = result["brain_switch"]
    payload["brain_switch"] = switch
    payload["ids"].update({"switch_id": switch["switch_id"]})
    payload["evidence_refs"].append(f"brain_switch:{switch['switch_id']}")
    if args.evidence_bundle_id:
        payload["evidence_refs"].append(f"evidence_bundle:{args.evidence_bundle_id}")
    if args.mission_id:
        payload["evidence_refs"].append(f"mission:{args.mission_id}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_brain_ledger(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = brain_payload(args, "cornerstone brain ledger")
    result = store.list_brain_ledger(requested_scope)
    ledger = result["brain_ledger"]
    payload["brain_ledger"] = ledger
    payload["ids"].update({"brain_ledger_scope": requested_scope["namespace_id"]})
    payload["evidence_refs"].extend([f"brain_ledger:{entry['ledger_id']}" for entry in ledger.get("entries", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_brain_aggregate_test(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = brain_payload(args, "cornerstone brain aggregate-test")
    result = store.test_brain_aggregation(
        source_namespace=args.source_namespace,
        target_namespace=args.target_namespace,
        opt_in=args.opt_in,
        scope=requested_scope,
    )
    error_exit = handle_brain_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    aggregation = result["aggregation"]
    payload["aggregation"] = aggregation
    payload["ids"].update({"aggregation_id": aggregation["aggregation_id"]})
    payload["evidence_refs"].append(f"brain_aggregation:{aggregation['aggregation_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_judge_run(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = brain_payload(args, "cornerstone judge run")
    result = store.run_judge(
        route_id=args.route_id,
        subject=args.subject,
        rubric=args.rubric,
        evidence_ref=args.evidence_ref,
        ambiguity=args.ambiguity,
        scope=requested_scope,
    )
    error_exit = handle_brain_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    judge = result["judge_record"]
    ledger = result["ledger_entry"]
    payload["judge_record"] = judge
    payload["ledger_entry"] = ledger
    payload["ids"].update({"judge_record_id": judge["judge_record_id"], "ledger_id": ledger["ledger_id"]})
    payload["evidence_refs"].extend([f"judge_record:{judge['judge_record_id']}", f"brain_ledger:{ledger['ledger_id']}", *judge.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_judge_conflict(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = brain_payload(args, "cornerstone judge conflict")
    result = store.record_judge_conflict(args.judge_record_id, objective_evidence=args.objective_evidence, scope=requested_scope)
    error_exit = handle_brain_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    conflict = result["judge_conflict"]
    payload["judge_conflict"] = conflict
    payload["ids"].update({"conflict_id": conflict["conflict_id"], "judge_record_id": args.judge_record_id})
    payload["evidence_refs"].append(f"judge_conflict:{conflict['conflict_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_judge_accept(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = brain_payload(args, "cornerstone judge accept")
    result = store.record_owner_acceptance(args.judge_record_id, acceptance=args.acceptance, scope=requested_scope)
    error_exit = handle_brain_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    acceptance = result["owner_acceptance"]
    payload["owner_acceptance"] = acceptance
    payload["ids"].update({"acceptance_id": acceptance["acceptance_id"], "judge_record_id": args.judge_record_id})
    payload["evidence_refs"].append(f"owner_acceptance:{acceptance['acceptance_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_judge_recommend(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = brain_payload(args, "cornerstone judge recommend")
    result = store.recommend_from_judge(args.judge_record_id, recommendation=args.recommendation, scope=requested_scope)
    error_exit = handle_brain_error(payload, result, args.json)
    if error_exit is not None:
        return error_exit
    recommendation = result["judge_recommendation"]
    payload["judge_recommendation"] = recommendation
    payload["ids"].update({"recommendation_id": recommendation["recommendation_id"], "judge_record_id": args.judge_record_id})
    payload["evidence_refs"].extend([f"judge_recommendation:{recommendation['recommendation_id']}", *recommendation.get("evidence_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_judge_disagreement_test(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = brain_payload(args, "cornerstone judge disagreement-test")
    result = store.test_judge_disagreement(risk=args.risk, scope=requested_scope)
    adjudication = result["adjudication"]
    payload["adjudication"] = adjudication
    payload["ids"].update({"adjudication_id": adjudication["adjudication_id"]})
    payload["evidence_refs"].append(f"judge_adjudication:{adjudication['adjudication_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_judge_calibration(args: argparse.Namespace) -> int:
    _, store, requested_scope, payload = brain_payload(args, "cornerstone judge calibration")
    result = store.judge_calibration_report(requested_scope)
    report = result["calibration_report"]
    payload["calibration_report"] = report
    payload["ids"].update({"calibration_id": report["calibration_id"]})
    payload["evidence_refs"].extend([f"judge_calibration:{report['calibration_id']}", *report.get("ledger_refs", [])])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_mission_create(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_mission_contract(
        args.goal,
        requested_scope,
        claim_id=args.claim_id,
        evidence_bundle_id=args.evidence_bundle_id,
    )
    payload = base_response("cornerstone mission create", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_MISSION_SOURCE_NOT_FOUND",
                "message": "Mission source claim or evidence bundle was not found.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Mission source is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    if result.get("status") == "evidence_required":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_MISSION_EVIDENCE_REQUIRED",
                "message": "Mission Goal Contract requires an Evidence Bundle with at least one artifact reference.",
            }
        )
        print_payload(payload, args.json)
        return EXIT_EVIDENCE_MISSING

    mission = result["mission"]
    audit_event = result["audit_event"]
    payload.update(mission["scope"])
    payload["ids"].update({"mission_id": mission["mission_id"], "audit_event_id": audit_event["event_id"]})
    if mission["source_claim"].get("claim_id"):
        payload["ids"]["claim_id"] = mission["source_claim"]["claim_id"]
    if mission["evidence"].get("evidence_bundle_id"):
        payload["ids"]["evidence_bundle_id"] = mission["evidence"]["evidence_bundle_id"]
    payload["mission"] = mission
    payload["evidence_refs"].append(f"mission:{mission['mission_id']}")
    if mission["source_claim"].get("claim_id"):
        payload["evidence_refs"].append(f"claim:{mission['source_claim']['claim_id']}")
    if mission["evidence"].get("evidence_bundle_id"):
        payload["evidence_refs"].append(f"evidence_bundle:{mission['evidence']['evidence_bundle_id']}")
    payload["evidence_refs"].extend(mission["evidence"].get("artifact_refs", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_mission_activate(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.activate_mission(args.mission_id, requested_scope, mode=args.mode)
    payload = base_response("cornerstone mission activate", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_MISSION_NOT_FOUND",
                "message": "Mission Goal Contract was not found.",
                "mission_id": args.mission_id,
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Mission is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED

    mission = result["mission"]
    payload.update(mission["scope"])
    payload["ids"].update({"mission_id": mission["mission_id"]})
    payload["mission"] = mission
    payload["workspace_mode"] = result["workspace_mode"]
    payload["evidence_refs"].append(f"mission:{mission['mission_id']}")
    payload["audit_refs"].extend(f"audit:{event['event_id']}" for event in result["audit_events"])
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_mission_show(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    mission = store.get_mission(args.mission_id)
    payload = base_response("cornerstone mission show", "success", root)
    payload.update(requested_scope)
    if mission is None:
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_MISSION_NOT_FOUND", "message": "Mission was not found.", "mission_id": args.mission_id})
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if mission.get("scope") != requested_scope:
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_SCOPE_DENIED", "message": "Mission is outside the requested scope.", "resource_scope": mission.get("scope")})
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    payload["mission"] = mission
    payload["evidence_refs"].append(f"mission:{mission['mission_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_action_propose(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.propose_action(
        args.mission_id,
        args.claim_id,
        args.action_kind,
        args.risk,
        requested_scope,
        goal=args.goal,
        connector=args.connector,
        target=args.target,
        ontology_object_refs=args.ontology_object_ref or [],
    )
    payload = base_response("cornerstone action propose", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_ACTION_SOURCE_NOT_FOUND",
                "message": "Action source mission or claim was not found.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Action source is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    if result.get("status") == "evidence_required":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_ACTION_EVIDENCE_REQUIRED",
                "message": "Action proposal requires an evidence-backed claim.",
            }
        )
        print_payload(payload, args.json)
        return EXIT_EVIDENCE_MISSING

    card = result["action_card"]
    audit_event = result["audit_event"]
    payload.update(card["scope"])
    payload["ids"].update(
        {
            "action_id": card["action_id"],
            "mission_id": card["mission_id"],
            "claim_id": card["source_claim_id"],
            "dry_run_id": card["dry_run"]["dry_run_id"],
            "audit_event_id": audit_event["event_id"],
        }
    )
    payload["action_card"] = card
    payload["policy_decisions"] = [card["policy_decision"]]
    payload["policy_decision_refs"].append(f"policy:{card['policy_decision']['id']}")
    payload["evidence_refs"].extend(
        [
            f"action:{card['action_id']}",
            f"mission:{card['mission_id']}",
            f"claim:{card['source_claim_id']}",
            f"dry_run:{card['dry_run']['dry_run_id']}",
        ]
    )
    evidence_bundle_id = card.get("evidence", {}).get("evidence_bundle_id")
    if evidence_bundle_id:
        payload["evidence_refs"].append(f"evidence_bundle:{evidence_bundle_id}")
    payload["evidence_refs"].extend(card.get("evidence", {}).get("artifact_refs", []))
    payload["evidence_refs"].extend(card.get("ontology_impact", {}).get("object_refs", []))
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_action_approve(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.approve_action(args.action_id, requested_scope, approver=args.approver)
    payload = base_response("cornerstone action approve", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_ACTION_NOT_FOUND", "message": "Action Card was not found.", "action_id": args.action_id})
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_SCOPE_DENIED", "message": "Action Card is outside the requested scope.", "resource_scope": result.get("resource_scope")})
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    card = result["action_card"]
    payload["action_card"] = card
    payload["ids"].update({"action_id": card["action_id"]})
    payload["evidence_refs"].append(f"action:{card['action_id']}")
    if result.get("audit_event"):
        payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_action_execute(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.execute_action(args.action_id, requested_scope)
    payload = base_response("cornerstone action execute", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_ACTION_NOT_FOUND", "message": "Action Card or its Mission was not found.", "action_id": args.action_id})
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_SCOPE_DENIED", "message": "Action Card is outside the requested scope.", "resource_scope": result.get("resource_scope")})
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    card = result["action_card"]
    payload["action_card"] = card
    payload["ids"].update({"action_id": card["action_id"], "mission_id": card["mission_id"]})
    payload["evidence_refs"].append(f"action:{card['action_id']}")
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    if result.get("status") == "policy_denied":
        payload["status"] = "denied"
        payload["policy_decisions"] = [result["policy_decision"]]
        payload["policy_decision_refs"].append(f"policy:{result['policy_decision']['id']}")
        payload["errors"].append(
            {
                "code": "CS_ACTION_POLICY_DENIED",
                "message": result["policy_decision"]["reason"],
                "resolution_path": result["policy_decision"]["resolution_path"],
            }
        )
        print_payload(payload, args.json)
        return EXIT_POLICY_DENIED
    payload["action_result"] = result["action_result"]
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def _action_read_payload(
    args: argparse.Namespace,
    command: str,
    event_type: str,
    reason: str,
) -> tuple[int, dict[str, Any], LocalRuntimeStore, dict[str, str]]:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    action = store.get_action(args.action_id)
    payload = base_response(command, "success", root)
    payload.update(requested_scope)
    if action is None:
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_ACTION_NOT_FOUND", "message": "Action Card was not found.", "action_id": args.action_id})
        return EXIT_NOT_FOUND, payload, store, requested_scope
    if action.get("scope") != requested_scope:
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_SCOPE_DENIED", "message": "Action Card is outside the requested scope.", "resource_scope": action.get("scope")})
        return EXIT_SCOPE_DENIED, payload, store, requested_scope
    audit_event = store.append_audit(
        event_type,
        requested_scope,
        {"type": "action", "id": args.action_id},
        {"reason": reason, "dry_run_id": action.get("dry_run", {}).get("dry_run_id")},
    )
    payload["action_card"] = action
    payload["ids"].update({"action_id": args.action_id, "audit_event_id": audit_event["event_id"]})
    payload["evidence_refs"].append(f"action:{args.action_id}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    return EXIT_SUCCESS, payload, store, requested_scope


def command_action_show(args: argparse.Namespace) -> int:
    exit_code, payload, _, _ = _action_read_payload(args, "cornerstone action show", "action.read", "cli_action_show")
    print_payload(payload, args.json)
    return exit_code


def command_action_dry_run(args: argparse.Namespace) -> int:
    exit_code, payload, _, _ = _action_read_payload(args, "cornerstone action dry-run", "action.dry_run.read", "cli_action_dry_run")
    if exit_code == EXIT_SUCCESS:
        action = payload["action_card"]
        payload["dry_run"] = action.get("dry_run", {})
        policy = action.get("policy_decision", {})
        if policy:
            payload["policy_decisions"] = [policy]
            payload["policy_decision_refs"].append(f"policy:{policy['id']}")
        dry_run_id = action.get("dry_run", {}).get("dry_run_id")
        if dry_run_id:
            payload["ids"]["dry_run_id"] = dry_run_id
            payload["evidence_refs"].append(f"dry_run:{dry_run_id}")
    print_payload(payload, args.json)
    return exit_code


def command_connector_direct_write_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.deny_direct_connector_write(args.provider, args.target, requested_scope)
    decision = result["policy_decision"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone connector direct-write-test", "denied", root)
    payload.update(requested_scope)
    payload["policy_decisions"] = [decision]
    payload["policy_decision_refs"].append(f"policy:{decision['id']}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    payload["errors"].append(
        {
            "code": "CS_DIRECT_WRITE_DENIED",
            "message": decision["reason"],
            "resolution_path": decision["resolution_path"],
            "external_http_calls": 0,
        }
    )
    print_payload(payload, args.json)
    return EXIT_POLICY_DENIED


def command_connector_credential_boundary_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.test_connector_credential_boundary(args.provider, args.capability, requested_scope)
    boundary = result["credential_boundary"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone connector credential-boundary-test", "success", root)
    payload.update(requested_scope)
    payload["credential_boundary"] = boundary
    payload["ids"].update({"boundary_id": boundary["boundary_id"]})
    payload["evidence_refs"].append(f"credential_boundary:{boundary['boundary_id']}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_action_idempotency_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.test_action_idempotency(args.action_id, requested_scope)
    payload = base_response("cornerstone action idempotency-test", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_ACTION_NOT_FOUND", "message": "Action Card was not found.", "action_id": args.action_id})
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append({"code": "CS_SCOPE_DENIED", "message": "Action Card is outside the requested scope.", "resource_scope": result.get("resource_scope")})
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED
    record = result["idempotency"]
    audit_event = result["audit_event"]
    payload["idempotency"] = record
    payload["ids"].update({"idempotency_id": record["idempotency_id"], "action_id": record["action_id"]})
    payload["evidence_refs"].extend([f"idempotency:{record['idempotency_id']}", f"action:{record['action_id']}"])
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_connector_action_trace(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_connector_action_trace(args.action_id, requested_scope)
    payload = base_response("cornerstone connector action-trace", "success", root)
    payload.update(requested_scope)
    if result.get("status") in {"not_found", "scope_denied", "evidence_required"}:
        exit_code = _record_command_failure(payload, result, resource_label="CONNECTOR_ACTION_TRACE")
        print_payload(payload, args.json)
        return exit_code
    trace = result["connector_action_trace"]
    payload["connector_action_trace"] = trace
    payload["ids"].update({"connector_action_trace_id": trace["trace_id"], "action_id": args.action_id})
    payload["evidence_refs"].extend([f"connector_action_trace:{trace['trace_id']}", f"action:{args.action_id}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_mission_autonomy_control(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.control_mission_autonomy(args.mission_id, requested_scope, control=args.control, reason=args.reason)
    payload = base_response("cornerstone mission autonomy-control", "success", root)
    payload.update(requested_scope)
    if result.get("status") in {"not_found", "scope_denied", "evidence_required"}:
        exit_code = _record_command_failure(payload, result, resource_label="MISSION_AUTONOMY_CONTROL")
        print_payload(payload, args.json)
        return exit_code
    control = result["autonomy_control"]
    payload["mission"] = result["mission"]
    payload["autonomy_control"] = control
    payload["workspace_mode"] = result["workspace_mode"]
    payload["ids"].update({"control_id": control["control_id"], "mission_id": args.mission_id})
    payload["evidence_refs"].extend([f"mission:{args.mission_id}", f"autonomy_control:{control['control_id']}"])
    payload["audit_refs"].extend(f"audit:{event['event_id']}" for event in result["audit_events"])
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_mission_escalate(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.escalate_mission_exception(args.mission_id, args.exception, requested_scope)
    payload = base_response("cornerstone mission escalate", "success", root)
    payload.update(requested_scope)
    if result.get("status") in {"not_found", "scope_denied", "evidence_required"}:
        exit_code = _record_command_failure(payload, result, resource_label="MISSION_ESCALATION")
        print_payload(payload, args.json)
        return exit_code
    escalation = result["escalation"]
    payload["escalation"] = escalation
    payload["ids"].update({"escalation_id": escalation["escalation_id"], "mission_id": args.mission_id})
    payload["evidence_refs"].extend([f"mission:{args.mission_id}", f"escalation:{escalation['escalation_id']}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_mission_outcome(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.record_mission_outcome(args.mission_id, args.action_id, requested_scope)
    payload = base_response("cornerstone mission outcome", "success", root)
    payload.update(requested_scope)
    if result.get("status") in {"not_found", "scope_denied", "evidence_required"}:
        exit_code = _record_command_failure(payload, result, resource_label="MISSION_OUTCOME")
        print_payload(payload, args.json)
        return exit_code
    outcome = result["mission_outcome"]
    payload["mission_outcome"] = outcome
    payload["ids"].update({"outcome_id": outcome["outcome_id"], "mission_id": args.mission_id, "action_id": args.action_id})
    payload["evidence_refs"].extend([f"mission_outcome:{outcome['outcome_id']}", f"mission:{args.mission_id}", f"action:{args.action_id}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_mission_after_action(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_mission_after_action_review(args.mission_id, args.outcome_id, requested_scope)
    payload = base_response("cornerstone mission after-action-review", "success", root)
    payload.update(requested_scope)
    if result.get("status") in {"not_found", "scope_denied", "evidence_required"}:
        exit_code = _record_command_failure(payload, result, resource_label="MISSION_AFTER_ACTION")
        print_payload(payload, args.json)
        return exit_code
    review = result["after_action_review"]
    payload["after_action_review"] = review
    payload["ids"].update({"review_id": review["review_id"], "mission_id": args.mission_id, "outcome_id": args.outcome_id})
    payload["evidence_refs"].extend([f"after_action_review:{review['review_id']}", f"mission:{args.mission_id}", f"mission_outcome:{args.outcome_id}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_mission_audit_export(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.export_mission_audit(args.mission_id, requested_scope)
    payload = base_response("cornerstone mission audit-export", "success", root)
    payload.update(requested_scope)
    if result.get("status") in {"not_found", "scope_denied", "evidence_required"}:
        exit_code = _record_command_failure(payload, result, resource_label="MISSION_AUDIT_EXPORT")
        print_payload(payload, args.json)
        return exit_code
    export = result["mission_audit_export"]
    payload["mission_audit_export"] = export
    payload["ids"].update({"export_id": export["export_id"], "mission_id": args.mission_id})
    payload["evidence_refs"].extend([f"mission_audit_export:{export['export_id']}", f"mission:{args.mission_id}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_autopilot_metrics(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.autonomy_quality_metrics(args.mission_id, args.outcome_id, requested_scope)
    payload = base_response("cornerstone autopilot metrics", "success", root)
    payload.update(requested_scope)
    if result.get("status") in {"not_found", "scope_denied", "evidence_required"}:
        exit_code = _record_command_failure(payload, result, resource_label="AUTOPILOT_METRICS")
        print_payload(payload, args.json)
        return exit_code
    metrics = result["autonomy_metrics"]
    payload["autonomy_metrics"] = metrics
    payload["ids"].update({"metric_id": metrics["metric_id"], "mission_id": args.mission_id, "outcome_id": args.outcome_id})
    payload["evidence_refs"].extend([f"autonomy_metrics:{metrics['metric_id']}", f"mission:{args.mission_id}", f"mission_outcome:{args.outcome_id}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_action_reversibility_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.test_action_reversibility(args.action_id, requested_scope, args.mode)
    payload = base_response("cornerstone action reversibility-test", "success", root)
    payload.update(requested_scope)
    if result.get("status") in {"not_found", "scope_denied", "evidence_required"}:
        exit_code = _record_command_failure(payload, result, resource_label="ACTION_REVERSIBILITY")
        print_payload(payload, args.json)
        return exit_code
    record = result["action_reversibility"]
    payload["action_reversibility"] = record
    payload["ids"].update({"reversibility_id": record["reversibility_id"], "action_id": args.action_id})
    payload["evidence_refs"].extend([f"action_reversibility:{record['reversibility_id']}", f"action:{args.action_id}"])
    payload["audit_refs"].append(f"audit:{result['audit_event']['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_security_sensitive_change_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.test_sensitive_change_gate(args.category, requested_scope)
    gate = result["sensitive_change_gate"]
    decision = result["policy_decision"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone security sensitive-change-test", "success", root)
    payload.update(requested_scope)
    payload["sensitive_change_gate"] = gate
    payload["policy_decisions"] = [decision]
    payload["policy_decision_refs"].append(f"policy:{decision['id']}")
    payload["ids"].update({"gate_id": gate["gate_id"]})
    payload["evidence_refs"].append(f"sensitive_change_gate:{gate['gate_id']}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_security_vs2_h01_approval_package(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.create_vs2_h01_approval_package(
        requested_scope,
        architecture_scope=args.architecture_scope,
        dependency_decision=args.dependency_decision,
        migration_scope=args.migration_scope,
        rollback_owner=args.rollback_owner,
        security_owner=args.security_owner,
        local_boundary=args.local_boundary,
        exceptions=args.exception,
    )
    package = result["approval_package"]
    approval_record = root / "docs/verification-reports/VS2_SEC_H01_OWNER_APPROVAL_2026-06-20.md"
    if approval_record.exists():
        package = dict(package)
        package["status"] = "approved_with_conditions"
        package["approval_status"] = "approved_with_conditions"
        package["sensitive_implementation_allowed"] = True
        package["approval_record"] = {
            "path": str(approval_record.relative_to(root)),
            "sha256": hashlib.sha256(approval_record.read_bytes()).hexdigest(),
            "decision": "APPROVE WITH CONDITIONS",
            "approver": "JiYong / Tars",
            "local_only": True,
            "production_claim_allowed": False,
        }
    audit_event = result["audit_event"]
    payload_status = "success" if package.get("approval_status") == "approved_with_conditions" else "human_review_required"
    payload = base_response("cornerstone security vs2-h01-approval-package", payload_status, root)
    payload.update(requested_scope)
    payload["vs2_h01_approval_package"] = package
    payload["ids"].update({"package_id": package["package_id"]})
    payload["evidence_refs"].append(f"vs2_h01_approval_package:{package['package_id']}")
    if package.get("approval_record"):
        payload["evidence_refs"].append(f"approval_record:{package['approval_record']['path']}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_security_vs2_local_proof(args: argparse.Namespace) -> int:
    root = repo_root()
    report = run_vs2_local_security_proof(
        root,
        local_range_report=Path(args.reuse_local_range_report) if args.reuse_local_range_report else None,
    )
    payload = base_response("cornerstone security vs2-local-proof", report["status"], root)
    payload.update(report)
    payload["evidence_refs"].extend(
        [
            "report:reports/security/vs2-local-security-proof.json",
            "report:reports/security/vs2-local-range.json",
            "report:reports/db/vs2-rls-inventory.json",
            "report:reports/db/vs2-tenant-isolation.json",
            "report:reports/policy/vs2-opa-test.json",
            "report:reports/network/vs2-egress-proof.json",
            "report:reports/security/vs2-output-leak-scan.json",
        ]
    )
    print_payload(payload, args.json)
    return EXIT_SUCCESS if report["status"] == "success" else EXIT_EVIDENCE_MISSING


def command_security_vs2_local_range(args: argparse.Namespace) -> int:
    root = repo_root()
    report = run_vs2_local_range(root)
    payload = base_response("cornerstone security vs2-local-range", report["status"], root)
    payload.update(report)
    payload["evidence_refs"].append("report:reports/security/vs2-local-range.json")
    print_payload(payload, args.json)
    return EXIT_SUCCESS if report["status"] == "passed" else EXIT_EVIDENCE_MISSING


def command_security_vs2_range_client(args: argparse.Namespace) -> int:
    root = repo_root()
    caller_fields = {}
    if args.forged_tenant_id:
        caller_fields["tenant_id"] = args.forged_tenant_id
    if args.forged_role:
        caller_fields["role"] = args.forged_role
    result = run_vs2_range_client(root, args.api_url, args.token, args.artifact_id, caller_fields)
    status = "success" if result.get("http_status") == 200 else "denied"
    payload = base_response("cornerstone security vs2-range-client", status, root)
    payload["range_client"] = result
    response_payload = result.get("payload", {})
    payload["evidence_refs"].extend(response_payload.get("evidence_refs", []))
    payload["audit_refs"].extend(response_payload.get("audit_refs", []))
    policy_decision = response_payload.get("policy_decision")
    if policy_decision and policy_decision.get("decision_id"):
        payload["policy_decision_refs"].append(f"policy:{policy_decision['decision_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS if result.get("http_status") == 200 else EXIT_POLICY_DENIED


def command_security_vs2_range_action_client(args: argparse.Namespace) -> int:
    root = repo_root()
    result = run_vs2_range_action_client(root, args.api_url, args.token, args.provider_url)
    status = "success" if result.get("http_status") == 200 else "denied"
    payload = base_response("cornerstone security vs2-range-action-client", status, root)
    payload["range_action_client"] = result
    response_payload = result.get("payload", {})
    payload["audit_refs"].extend(response_payload.get("audit_refs", []))
    for decision_key in ["dry_run_decision", "execution_decision"]:
        decision = response_payload.get(decision_key, {})
        if decision.get("decision_id"):
            payload["policy_decision_refs"].append(f"policy:{decision['decision_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS if result.get("http_status") == 200 else EXIT_POLICY_DENIED


def command_security_vs2_range_object_contract_client(args: argparse.Namespace) -> int:
    root = repo_root()
    result = run_vs2_range_object_contract_client(root, args.api_url, args.token)
    status = "success" if result.get("http_status") == 200 else "denied"
    payload = base_response("cornerstone security vs2-range-object-contract-client", status, root)
    payload["range_object_contract_client"] = result
    response_payload = result.get("payload", {})
    payload["audit_refs"].extend(response_payload.get("audit_refs", []))
    print_payload(payload, args.json)
    return EXIT_SUCCESS if result.get("http_status") == 200 else EXIT_POLICY_DENIED


def command_security_vs2_range_object_access_client(args: argparse.Namespace) -> int:
    root = repo_root()
    result = run_vs2_range_object_access_client(root, args.api_url, args.token)
    status = "success" if result.get("http_status") == 200 else "denied"
    payload = base_response("cornerstone security vs2-range-object-access-client", status, root)
    payload["range_object_access_client"] = result
    response_payload = result.get("payload", {})
    payload["audit_refs"].extend(response_payload.get("audit_refs", []))
    print_payload(payload, args.json)
    return EXIT_SUCCESS if result.get("http_status") == 200 else EXIT_POLICY_DENIED


def command_security_vs2_range_observability_client(args: argparse.Namespace) -> int:
    root = repo_root()
    result = run_vs2_range_observability_client(root, args.api_url, args.token)
    status = "success" if result.get("http_status") == 200 else "denied"
    payload = base_response("cornerstone security vs2-range-observability-client", status, root)
    payload["range_observability_client"] = result
    response_payload = result.get("payload", {})
    payload["audit_refs"].extend(response_payload.get("audit_refs", []))
    print_payload(payload, args.json)
    return EXIT_SUCCESS if result.get("http_status") == 200 else EXIT_POLICY_DENIED


def command_security_vs2_range_tenant_read_client(args: argparse.Namespace) -> int:
    root = repo_root()
    result = run_vs2_range_tenant_read_client(root, args.api_url, args.token)
    status = "success" if result.get("http_status") == 200 else "denied"
    payload = base_response("cornerstone security vs2-range-tenant-read-client", status, root)
    payload["range_tenant_read_client"] = result
    response_payload = result.get("payload", {})
    payload["audit_refs"].extend(response_payload.get("audit_refs", []))
    print_payload(payload, args.json)
    return EXIT_SUCCESS if result.get("http_status") == 200 else EXIT_POLICY_DENIED


def command_security_vs2_range_search_client(args: argparse.Namespace) -> int:
    root = repo_root()
    result = run_vs2_range_search_client(root, args.api_url, args.token)
    status = "success" if result.get("http_status") == 200 else "denied"
    payload = base_response("cornerstone security vs2-range-search-client", status, root)
    payload["range_search_client"] = result
    response_payload = result.get("payload", {})
    payload["audit_refs"].extend(response_payload.get("audit_refs", []))
    print_payload(payload, args.json)
    return EXIT_SUCCESS if result.get("http_status") == 200 else EXIT_POLICY_DENIED


def command_security_vs2_range_db_path_client(args: argparse.Namespace) -> int:
    root = repo_root()
    result = run_vs2_range_db_path_client(root, args.api_url, args.token)
    status = "success" if result.get("http_status") == 200 else "denied"
    payload = base_response("cornerstone security vs2-range-db-path-client", status, root)
    payload["range_db_path_client"] = result
    response_payload = result.get("payload", {})
    payload["audit_refs"].extend(response_payload.get("audit_refs", []))
    print_payload(payload, args.json)
    return EXIT_SUCCESS if result.get("http_status") == 200 else EXIT_POLICY_DENIED


def command_security_vs2_range_constraint_collision_client(args: argparse.Namespace) -> int:
    root = repo_root()
    result = run_vs2_range_constraint_collision_client(root, args.api_url, args.token)
    status = "success" if result.get("http_status") == 200 else "denied"
    payload = base_response("cornerstone security vs2-range-constraint-collision-client", status, root)
    payload["range_constraint_collision_client"] = result
    response_payload = result.get("payload", {})
    payload["audit_refs"].extend(response_payload.get("audit_refs", []))
    print_payload(payload, args.json)
    return EXIT_SUCCESS if result.get("http_status") == 200 else EXIT_POLICY_DENIED


def command_security_vs2_range_migration_client(args: argparse.Namespace) -> int:
    root = repo_root()
    result = run_vs2_range_migration_client(root, args.api_url, args.token)
    status = "success" if result.get("http_status") == 200 else "denied"
    payload = base_response("cornerstone security vs2-range-migration-client", status, root)
    payload["range_migration_client"] = result
    response_payload = result.get("payload", {})
    payload["audit_refs"].extend(response_payload.get("audit_refs", []))
    print_payload(payload, args.json)
    return EXIT_SUCCESS if result.get("http_status") == 200 else EXIT_POLICY_DENIED


def command_security_vs2_range_upgrade_path_client(args: argparse.Namespace) -> int:
    root = repo_root()
    result = run_vs2_range_upgrade_path_client(root, args.api_url, args.token)
    status = "success" if result.get("http_status") == 200 else "denied"
    payload = base_response("cornerstone security vs2-range-upgrade-path-client", status, root)
    payload["range_upgrade_path_client"] = result
    response_payload = result.get("payload", {})
    payload["audit_refs"].extend(response_payload.get("audit_refs", []))
    for decision_key in ["policy_decision", "destructive_migration_decision"]:
        decision = response_payload.get(decision_key, {})
        if decision.get("decision_id"):
            payload["policy_decision_refs"].append(f"policy:{decision['decision_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS if result.get("http_status") == 200 else EXIT_POLICY_DENIED


def command_security_vs2_range_audit_integrity_client(args: argparse.Namespace) -> int:
    root = repo_root()
    result = run_vs2_range_audit_integrity_client(root, args.api_url, args.token)
    status = "success" if result.get("http_status") == 200 else "denied"
    payload = base_response("cornerstone security vs2-range-audit-integrity-client", status, root)
    payload["range_audit_integrity_client"] = result
    response_payload = result.get("payload", {})
    payload["audit_refs"].extend(response_payload.get("audit_refs", []))
    decision = response_payload.get("policy_decision", {})
    if decision.get("decision_id"):
        payload["policy_decision_refs"].append(f"policy:{decision['decision_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS if result.get("http_status") == 200 else EXIT_POLICY_DENIED


def command_security_backup_restore_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.rehearse_backup_restore(requested_scope, subject_refs=args.subject_ref)
    record = result["backup_restore"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone security backup-restore-test", "success", root)
    payload.update(requested_scope)
    payload["backup_restore"] = record
    payload["ids"].update({"restore_id": record["restore_id"]})
    payload["evidence_refs"].append(f"backup_restore:{record['restore_id']}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_security_helpful_failure_test(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.record_helpful_failure_examples(requested_scope)
    record = result["helpful_failures"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone security helpful-failure-test", "success", root)
    payload.update(requested_scope)
    payload["helpful_failures"] = record
    payload["ids"].update({"failure_id": record["failure_id"]})
    payload["evidence_refs"].append(f"helpful_failures:{record['failure_id']}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_security_retention_explain(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.explain_retention(args.resource_type, requested_scope)
    record = result["retention_explanation"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone security retention-explain", "success", root)
    payload.update(requested_scope)
    payload["retention_explanation"] = record
    payload["ids"].update({"retention_id": record["retention_id"]})
    payload["evidence_refs"].append(f"retention:{record['retention_id']}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_security_operator_status(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.operator_status_report(requested_scope)
    record = result["operator_status"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone security operator-status", "success", root)
    payload.update(requested_scope)
    payload["operator_status"] = record
    payload["ids"].update({"status_id": record["status_id"]})
    payload["evidence_refs"].append(f"operator_status:{record['status_id']}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_release_report_check(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    scenario_report = (root / args.scenario_report).resolve()
    verification_report = (root / args.verification_report).resolve()
    result = store.validate_release_report(scenario_report, verification_report, requested_scope)
    record = result["release_report_validation"]
    audit_event = result["audit_event"]
    payload = base_response("cornerstone release report-check", "success" if record["status"] == "passed" else "failed", root)
    payload.update(requested_scope)
    payload["release_report_validation"] = record
    payload["ids"].update({"report_id": record["report_id"]})
    payload["evidence_refs"].append(f"release_report:{record['report_id']}")
    payload["audit_refs"].append(f"audit:{audit_event['event_id']}")
    if record["status"] != "passed":
        payload["errors"].append({"code": "CS_RELEASE_REPORT_INVALID", "message": "Release report validation failed.", "details": record["errors"]})
        print_payload(payload, args.json)
        return EXIT_RUNTIME_FAILURE
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_release_evidence_collect(args: argparse.Namespace) -> int:
    root = repo_root()
    requested_scope = scope_args(args)
    if args.scope == "vs0-evux":
        scenario_report_arg = args.scenario_report or DEFAULT_EVUX_SCENARIO_REPORT
        browser_proof_dir_arg = args.browser_proof_dir or DEFAULT_EVUX_BROWSER_PROOF_DIR
        output_dir_arg = args.output_dir or DEFAULT_EVUX_RELEASE_PACKAGE_DIR
        verification_report_arg = args.verification_report or DEFAULT_EVUX_REPORT
    else:
        scenario_report_arg = args.scenario_report or DEFAULT_ACCEPTANCE_SCENARIO_REPORT
        browser_proof_dir_arg = args.browser_proof_dir or DEFAULT_BROWSER_PROOF_DIR
        output_dir_arg = args.output_dir or DEFAULT_RELEASE_PACKAGE_DIR
        verification_report_arg = args.verification_report or DEFAULT_ACCEPTANCE_REPORT
    scenario_report = (root / scenario_report_arg).resolve()
    product_runtime_report = (root / (args.product_runtime_report or DEFAULT_PRODUCT_RUNTIME_REPORT)).resolve()
    browser_proof_dir = (root / browser_proof_dir_arg).resolve()
    output_dir = (root / output_dir_arg).resolve()
    verification_report = (root / verification_report_arg).resolve() if verification_report_arg else None
    result = collect_release_evidence(
        root,
        requested_scope=requested_scope,
        scope_name=args.scope,
        output_dir=output_dir,
        scenario_report=scenario_report,
        product_runtime_report=product_runtime_report,
        browser_proof_dir=browser_proof_dir,
        verification_report=verification_report,
    )
    payload = base_response("cornerstone release evidence collect", result["status"], root)
    payload.update(requested_scope)
    payload["release_evidence_package"] = result
    payload["ids"].update({"package_id": result.get("package_id")})
    if result["status"] != "success":
        payload["errors"].append(
            {
                "code": "CS_RELEASE_EVIDENCE_INCOMPLETE",
                "message": "Release evidence package is missing required artifacts.",
                "missing_required": result.get("missing_required", []),
            }
        )
        print_payload(payload, args.json)
        return EXIT_EVIDENCE_MISSING
    payload["evidence_refs"].append(f"release_evidence:{result['package_id']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_release_evidence_finalize(args: argparse.Namespace) -> int:
    root = repo_root()
    requested_scope = scope_args(args)
    if args.scope == "vs0-evux":
        output_dir_arg = args.output_dir or DEFAULT_EVUX_RELEASE_PACKAGE_DIR
    else:
        output_dir_arg = args.output_dir or DEFAULT_RELEASE_PACKAGE_DIR
    result = finalize_release_evidence(
        root,
        requested_scope=requested_scope,
        scope_name=args.scope,
        output_dir=(root / output_dir_arg).resolve(),
    )
    payload = base_response("cornerstone release evidence finalize", result["status"], root)
    payload.update(requested_scope)
    payload["release_evidence_finalize"] = result
    if result["status"] != "success":
        payload["errors"].extend(result.get("errors", []))
        if not payload["errors"]:
            payload["errors"].append(
                {
                    "code": "CS_RELEASE_EVIDENCE_FINALIZE_FAILED",
                    "message": "Release evidence finalization did not produce a clean post-commit rollup.",
                    "missing_required": result.get("missing_required", []),
                    "worktree_dirty_before_rollup": result.get("worktree_dirty_before_rollup"),
                }
            )
        print_payload(payload, args.json)
        return EXIT_EVIDENCE_MISSING
    payload["evidence_refs"].append(f"release_post_commit_rollup:{result['post_commit_rollup_path']}")
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_quickstart_verify(args: argparse.Namespace) -> int:
    root = repo_root()
    started = perf_counter()
    if args.quickstart != "vs0-evux":
        payload = base_response("cornerstone quickstart verify", "failed", root)
        payload["errors"].append(
            {
                "code": "CS_QUICKSTART_UNSUPPORTED",
                "message": "The requested quickstart verifier is not implemented.",
                "supported": ["vs0-evux"],
            }
        )
        print_payload(payload, args.json)
        return EXIT_INVALID
    output_path = (root / (args.output or DEFAULT_EVUX_QUICKSTART_REPORT)).resolve()
    result = run_evux_quickstart(root, output_path=output_path)
    exit_code = EXIT_SUCCESS if result["status"] == "success" else EXIT_EVIDENCE_MISSING
    result["self_command_transcript"] = command_transcript_entry(
        name="quickstart_verify_vs0_evux",
        command=[
            "cornerstone",
            "quickstart",
            "verify",
            "vs0-evux",
            "--json",
            "--output",
            args.output or DEFAULT_EVUX_QUICKSTART_REPORT,
        ],
        exit_code=exit_code,
        timed_out=False,
        elapsed_seconds=perf_counter() - started,
        stdout_tail=[
            json.dumps(
                {
                    "status": result.get("status"),
                    "generated_ids": result.get("generated_ids"),
                    "negative_evidence": result.get("negative_evidence"),
                },
                sort_keys=True,
            )
        ],
        stderr_tail=[],
    )
    write_json(output_path, result)
    payload = base_response("cornerstone quickstart verify vs0-evux", result["status"], root)
    payload.update(result)
    payload["output_path"] = str(output_path)
    payload["ids"].update(result.get("generated_ids", {}))
    payload["evidence_refs"].extend(result.get("evidence_refs", []))
    payload["audit_refs"].extend(result.get("audit_refs", []))
    if result["status"] != "success":
        payload["errors"].extend(result.get("errors", []))
        print_payload(payload, args.json)
        return exit_code
    print_payload(payload, args.json)
    return exit_code


def command_conversation_start(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.start_conversation(args.message, requested_scope)
    conversation = result["conversation"]
    artifact = result["artifact"]
    payload = base_response("cornerstone conversation start", "success", root)
    payload.update(requested_scope)
    payload["conversation"] = conversation
    payload["artifact"] = artifact
    payload["ids"].update(
        {
            "conversation_id": conversation["conversation_id"],
            "artifact_id": artifact["artifact_id"],
        }
    )
    payload["evidence_refs"].extend(
        [
            f"conversation:{conversation['conversation_id']}",
            f"artifact:{artifact['artifact_id']}",
            f"storage:{artifact['original_storage_ref']}",
        ]
    )
    payload["audit_refs"].extend(f"audit:{event['event_id']}" for event in result["audit_events"])
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_conversation_promote(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.promote_conversation_to_claim(
        args.conversation_id,
        args.statement,
        args.evidence_bundle_id,
        requested_scope,
    )
    payload = base_response("cornerstone conversation promote", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_CONVERSATION_PROMOTION_SOURCE_NOT_FOUND",
                "message": "Conversation or Evidence Bundle was not found.",
                "resource": result.get("resource"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Conversation promotion source is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED

    claim = result["claim"]
    evidence = claim["evidence_bundle"]
    payload.update(claim["scope"])
    payload["claim"] = claim
    payload["promoted_object"] = claim
    payload["ids"].update(
        {
            "conversation_id": args.conversation_id,
            "claim_id": claim["claim_id"],
            "evidence_bundle_id": evidence["evidence_bundle_id"],
            "search_snapshot_id": evidence["search_snapshot_id"],
        }
    )
    payload["evidence_refs"].extend(
        [
            f"conversation:{args.conversation_id}",
            f"claim:{claim['claim_id']}",
            f"evidence_bundle:{evidence['evidence_bundle_id']}",
            f"search_snapshot:{evidence['search_snapshot_id']}",
        ]
    )
    payload["evidence_refs"].extend(evidence.get("artifact_refs", []))
    payload["audit_refs"].extend(f"audit:{event['event_id']}" for event in result["audit_events"])
    print_payload(payload, args.json)
    return EXIT_SUCCESS


def command_conversation_answer(args: argparse.Namespace) -> int:
    root = repo_root()
    store = LocalRuntimeStore(state_dir(root, args))
    requested_scope = scope_args(args)
    result = store.answer_conversation(args.conversation_id, args.question, requested_scope)
    payload = base_response("cornerstone conversation answer", "success", root)
    payload.update(requested_scope)
    if result.get("status") == "not_found":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_CONVERSATION_NOT_FOUND",
                "message": "Conversation was not found.",
                "conversation_id": args.conversation_id,
            }
        )
        print_payload(payload, args.json)
        return EXIT_NOT_FOUND
    if result.get("status") == "scope_denied":
        payload["status"] = "failed"
        payload["errors"].append(
            {
                "code": "CS_SCOPE_DENIED",
                "message": "Conversation is outside the requested scope.",
                "resource_scope": result.get("resource_scope"),
            }
        )
        print_payload(payload, args.json)
        return EXIT_SCOPE_DENIED

    answer = result["answer"]
    snapshot = result["search_snapshot"]
    payload["answer"] = answer
    payload["search_snapshot"] = snapshot
    payload["ids"].update(
        {
            "conversation_id": args.conversation_id,
            "answer_id": answer["answer_id"],
            "search_snapshot_id": snapshot["search_snapshot_id"],
        }
    )
    payload["evidence_refs"].extend(
        [
            f"conversation:{args.conversation_id}",
            f"answer:{answer['answer_id']}",
            f"search_snapshot:{snapshot['search_snapshot_id']}",
        ]
    )
    payload["evidence_refs"].extend(answer.get("evidence_refs", []))
    payload["audit_refs"].extend(f"audit:{event['event_id']}" for event in result["audit_events"])
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
    started = perf_counter()
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
    elif args.contract == "vs0-briefing":
        report = verify_vs0_briefing(root)
    elif args.contract == "vs0-mission-action":
        report = verify_vs0_mission_action(root)
    elif args.contract == "vs0-detail-surfaces":
        report = verify_vs0_detail_surfaces(root)
    elif args.contract == "vs0-conversation-onboarding":
        report = verify_vs0_conversation_onboarding(root)
    elif args.contract == "vs0-product-loop-identity":
        report = verify_vs0_product_loop_identity(root)
    elif args.contract == "vs0-memory-truth-boundary":
        report = verify_vs0_memory_truth_boundary(root)
    elif args.contract == "vs0-product-domain-readiness":
        report = verify_vs0_product_domain_readiness(root)
    elif args.contract == "vs0-tenant-security-boundary":
        report = verify_vs0_tenant_security_boundary(root)
    elif args.contract == "vs0-product-runtime":
        report = verify_vs0_product_runtime(root)
    elif args.contract == "vs0-runtime-acceptance":
        report = verify_vs0_runtime_acceptance(root)
    elif args.contract == "vs0-evux":
        report = verify_vs0_evux(root)
    elif args.contract == "vs0-evux-governance":
        report = verify_vs0_evux_governance(root)
    elif args.contract == "vs0-operator-acceptance-ui":
        report = verify_vs0_operator_acceptance_ui(root)
    elif args.contract == "vs1-ontology-suggest-promote":
        report = verify_vs1_ontology_suggest_promote(root)
    elif args.contract == "vs2-policy-tenancy-egress":
        report = verify_vs2_policy_tenancy_egress(
            root,
            local_proof_report=Path(args.reuse_vs2_local_proof_report) if args.reuse_vs2_local_proof_report else None,
        )
    elif args.contract == "full-claim-collaboration":
        report = verify_full_claim_collaboration(root)
    elif args.contract == "full-agent-orchestration":
        report = verify_full_agent_orchestration(root)
    elif args.contract == "full-brain-routing":
        report = verify_full_brain_routing(root)
    elif args.contract == "full-security-operations":
        report = verify_full_security_operations(root)
    elif args.contract == "full-namespace-governance":
        report = verify_full_namespace_governance(root)
    elif args.contract == "full-mission-control-autonomy-lifecycle":
        report = verify_full_mission_control_autonomy_lifecycle(root)
    elif args.contract == "full":
        final_batch_ids = {
            "CS-PROD-006",
            "CS-PROD-007",
            "CS-PROD-008",
            "CS-PROD-009",
            "CS-PROD-010",
            "CS-AUTO-012",
            "CS-AUTO-013",
            "CS-AUTO-014",
            "CS-AUTO-015",
            "CS-AUTO-016",
            "CS-AUTO-017",
            "CS-AUTO-018",
            "CS-AUTO-019",
            "CS-REG-019",
        }
        requested = set(args.scenario or [])
        if requested and requested <= final_batch_ids:
            report = verify_full_mission_control_autonomy_lifecycle(root)
        else:
            payload = base_response("cornerstone scenario verify full", "failed", root)
            payload["errors"].append(
                {
                    "code": "CS_FULL_SCENARIO_FILTER_REQUIRED",
                    "message": "Use the explicit batch contract for broad verification, or pass a scenario ID implemented by the final full batch.",
                    "supported_scenarios": sorted(final_batch_ids),
                    "requested": sorted(requested),
                }
            )
            print_payload(payload, args.json)
            return EXIT_INVALID
    elif args.contract == "full-learning-experience":
        report = verify_full_learning_experience(root)
    elif args.contract == "full-extension-ecosystem":
        report = verify_full_extension_ecosystem(root)
    elif args.contract == "full-memory-wiki":
        report = verify_full_memory_wiki(root)
    elif args.contract == "full-understanding-ontology":
        report = verify_full_understanding_ontology(root)
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
                    "vs0-briefing",
                    "vs0-mission-action",
                    "vs0-detail-surfaces",
                    "vs0-conversation-onboarding",
                    "vs0-product-loop-identity",
                    "vs0-memory-truth-boundary",
                    "vs0-product-domain-readiness",
                    "vs0-tenant-security-boundary",
                    "vs0-product-runtime",
                    "vs0-runtime-acceptance",
                    "vs0-evux",
                    "vs0-evux-governance",
                    "vs0-operator-acceptance-ui",
                    "vs1-ontology-suggest-promote",
                    "vs2-policy-tenancy-egress",
                    "full-claim-collaboration",
                    "full-agent-orchestration",
                    "full-brain-routing",
                    "full-security-operations",
                    "full-namespace-governance",
                    "full-mission-control-autonomy-lifecycle",
                    "full",
                    "full-extension-ecosystem",
                    "full-learning-experience",
                    "full-memory-wiki",
                    "full-understanding-ontology",
                ],
            }
        )
        print_payload(payload, args.json)
        return EXIT_INVALID

    if args.scenario and args.contract != "vs0-fixtures":
        requested = set(args.scenario)
        rows = report.get("scenario_results", [])
        present = {row.get("id") for row in rows}
        missing = sorted(requested - present)
        report["scenario_filter"] = sorted(requested)
        if missing:
            report["status"] = "failed"
            report.setdefault("errors", []).append(
                {
                    "code": "CS_SCENARIO_FILTER_MISSING",
                    "message": "Requested product scenario is not covered by this verification contract.",
                    "missing": missing,
                }
            )
        else:
            filtered_rows = [row for row in rows if row.get("id") in requested]
            report["scenario_results"] = filtered_rows
            blocking = [row for row in filtered_rows if row.get("status") != "PASS" and row.get("owner") != "Human"]
            report.setdefault("summary", {})
            report["summary"]["scenario_count"] = len(filtered_rows)
            report["summary"]["pass"] = len([row for row in filtered_rows if row.get("status") == "PASS"])
            report["summary"]["fail"] = len([row for row in filtered_rows if row.get("status") == "FAIL"])
            report["summary"]["not_verified"] = len([row for row in filtered_rows if row.get("status") == "NOT_VERIFIED"])
            report["summary"]["human_required"] = len([row for row in filtered_rows if row.get("owner") == "Human"])
            report["summary"]["blocking"] = len(blocking)
            report["status"] = "success" if not blocking else "failed"

    payload = base_response(f"cornerstone scenario verify {args.contract}", report["status"], root)
    payload.update(report)
    output_arg = args.output
    if args.contract == "vs0-runtime-acceptance" and not output_arg:
        output_arg = DEFAULT_ACCEPTANCE_SCENARIO_REPORT
    if args.contract == "vs0-evux" and not output_arg:
        output_arg = DEFAULT_EVUX_SCENARIO_REPORT
    if args.contract == "vs0-evux-governance" and not output_arg:
        output_arg = "reports/scenario/vs0-evux-governance-2026-06-14.json"
    if args.contract == "vs0-operator-acceptance-ui" and not output_arg:
        output_arg = DEFAULT_OPERATOR_UI_SCENARIO_REPORT
    if args.contract == "vs1-ontology-suggest-promote" and not output_arg:
        output_arg = DEFAULT_VS1_ONTOLOGY_SCENARIO_REPORT
    if args.contract == "vs2-policy-tenancy-egress" and not output_arg:
        output_arg = "reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json"
    exit_code = EXIT_SUCCESS if report["status"] == "success" else EXIT_EVIDENCE_MISSING
    transcript_command = ["cornerstone", "scenario", "verify", args.contract]
    for scenario_id in args.scenario or []:
        transcript_command.extend(["--scenario", scenario_id])
    if args.corpus != "fixtures/vs0":
        transcript_command.extend(["--corpus", args.corpus])
    if args.model_provider != "local_test":
        transcript_command.extend(["--model-provider", args.model_provider])
    if args.json:
        transcript_command.append("--json")
    if args.reuse_vs2_local_proof_report and args.contract == "vs2-policy-tenancy-egress":
        transcript_command.extend(["--reuse-vs2-local-proof-report", args.reuse_vs2_local_proof_report])
    if output_arg:
        transcript_command.extend(["--output", output_arg])
    payload["self_command_transcript"] = command_transcript_entry(
        name=f"scenario_verify_{args.contract}",
        command=transcript_command,
        exit_code=exit_code,
        timed_out=False,
        elapsed_seconds=perf_counter() - started,
        stdout_tail=[
            json.dumps(
                {"status": report.get("status"), "scenario_set": report.get("scenario_set"), "summary": report.get("summary")},
                sort_keys=True,
            )
        ],
        stderr_tail=[],
    )
    if output_arg:
        output_path = (root / output_arg).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload["output_path"] = str(output_path)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if args.contract == "vs0-runtime-acceptance":
            release_result = collect_release_evidence(
                root,
                requested_scope={
                    "tenant_id": payload["tenant_id"],
                    "owner_id": payload["owner_id"],
                    "namespace_id": payload["namespace_id"],
                    "workspace_id": payload["workspace_id"],
                },
                scope_name="vs0-runtime-acceptance",
                output_dir=root / DEFAULT_RELEASE_PACKAGE_DIR,
                scenario_report=output_path,
                product_runtime_report=root / DEFAULT_PRODUCT_RUNTIME_REPORT,
                browser_proof_dir=root / DEFAULT_BROWSER_PROOF_DIR,
                verification_report=root / DEFAULT_ACCEPTANCE_REPORT,
            )
            payload["release_evidence_package"] = release_result
            output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if args.contract == "vs0-evux":
            release_result = collect_release_evidence(
                root,
                requested_scope={
                    "tenant_id": payload["tenant_id"],
                    "owner_id": payload["owner_id"],
                    "namespace_id": payload["namespace_id"],
                    "workspace_id": payload["workspace_id"],
                },
                scope_name="vs0-evux",
                output_dir=root / DEFAULT_EVUX_RELEASE_PACKAGE_DIR,
                scenario_report=output_path,
                product_runtime_report=root / DEFAULT_PRODUCT_RUNTIME_REPORT,
                browser_proof_dir=root / DEFAULT_EVUX_BROWSER_PROOF_DIR,
                verification_report=root / DEFAULT_EVUX_REPORT,
            )
            payload["release_evidence_package_final_report_bytes"] = release_result

    print_payload(payload, args.json)
    return exit_code


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

    quickstart = subcommands.add_parser("quickstart", help="Executable quickstart verification commands")
    quickstart_sub = quickstart.add_subparsers(dest="quickstart_command")
    quickstart_verify = quickstart_sub.add_parser("verify", help="Verify an executable local quickstart")
    quickstart_verify.add_argument("quickstart", choices=["vs0-evux"], help="Quickstart verifier")
    quickstart_verify.add_argument("--output", help="Quickstart transcript output path")
    quickstart_verify.add_argument("--json", action="store_true", help="Emit JSON output")
    quickstart_verify.set_defaults(func=command_quickstart_verify)

    runtime = subcommands.add_parser("runtime", help="Local VS0 API/UI runtime commands")
    runtime_sub = runtime.add_subparsers(dest="runtime_command")

    runtime_serve = runtime_sub.add_parser("serve", help="Serve the local VS0 API/UI runtime")
    runtime_serve.add_argument("--host", default="127.0.0.1", help="Host interface")
    runtime_serve.add_argument("--port", type=int, default=8787, help="Port")
    add_state_argument(runtime_serve)
    runtime_serve.set_defaults(func=command_runtime_serve)

    product = subcommands.add_parser("product", help="Product identity walkthrough commands")
    product_sub = product.add_subparsers(dest="product_command")

    walkthrough = product_sub.add_parser("walkthrough", help="Show the first-run product walkthrough")
    walkthrough.add_argument("--json", action="store_true", help="Emit JSON output")
    walkthrough.set_defaults(func=command_product_walkthrough)

    mission_control = product_sub.add_parser("mission-control", help="Show Mission Control / Ops Inbox projection")
    add_state_argument(mission_control)
    add_scope_arguments(mission_control)
    mission_control.add_argument("--json", action="store_true", help="Emit JSON output")
    mission_control.set_defaults(func=command_product_mission_control)

    loop_view = product_sub.add_parser("loop-view", help="Show Inbox to Learn journey for one item")
    loop_view.add_argument("--conversation-id")
    loop_view.add_argument("--brief-id")
    loop_view.add_argument("--claim-id")
    loop_view.add_argument("--mission-id")
    loop_view.add_argument("--action-id")
    loop_view.add_argument("--outcome-id")
    add_state_argument(loop_view)
    add_scope_arguments(loop_view)
    loop_view.add_argument("--json", action="store_true", help="Emit JSON output")
    loop_view.set_defaults(func=command_product_loop_view)

    boundary = product_sub.add_parser("boundary", help="Show source-system and CornerStone boundary copy")
    add_state_argument(boundary)
    add_scope_arguments(boundary)
    boundary.add_argument("--json", action="store_true", help="Emit JSON output")
    boundary.set_defaults(func=command_product_boundary)

    plain_language = product_sub.add_parser("plain-language-review", help="Verify plain product language for first value and mission work")
    add_state_argument(plain_language)
    add_scope_arguments(plain_language)
    plain_language.add_argument("--json", action="store_true", help="Emit JSON output")
    plain_language.set_defaults(func=command_product_plain_language)

    repo_split = product_sub.add_parser("repo-split-review", help="Verify UX labels do not require internal repo mental models")
    add_state_argument(repo_split)
    add_scope_arguments(repo_split)
    repo_split.add_argument("--json", action="store_true", help="Emit JSON output")
    repo_split.set_defaults(func=command_product_repo_split_review)

    wiki = subcommands.add_parser("wiki", help="Permanent wiki view commands")
    wiki_sub = wiki.add_subparsers(dest="wiki_command")

    wiki_show = wiki_sub.add_parser("show", help="Show a source-aware permanent wiki view")
    wiki_show.add_argument("--kind", choices=["personal", "organization", "product-learning"], default="personal", help="Wiki view kind")
    add_state_argument(wiki_show)
    add_scope_arguments(wiki_show)
    wiki_show.add_argument("--json", action="store_true", help="Emit JSON output")
    wiki_show.set_defaults(func=command_wiki_show)

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

    source = subcommands.add_parser("source", help="Source-system safety commands")
    source_sub = source.add_subparsers(dest="source_command")

    source_readonly = source_sub.add_parser("readonly-test", help="Verify source ingestion made no writeback")
    source_readonly.add_argument("--artifact-id", required=True, help="Ingested artifact ID")
    source_readonly.add_argument("--source-system", default="mock://readonly-source", help="Mock source system label")
    add_state_argument(source_readonly)
    add_scope_arguments(source_readonly)
    source_readonly.add_argument("--json", action="store_true", help="Emit JSON output")
    source_readonly.set_defaults(func=command_source_readonly_test)

    audit = subcommands.add_parser("audit", help="Audit ledger commands")
    audit_sub = audit.add_subparsers(dest="audit_command")

    audit_verify = audit_sub.add_parser("verify", help="Verify the local audit hash chain")
    add_state_argument(audit_verify)
    audit_verify.add_argument("--json", action="store_true", help="Emit JSON output")
    audit_verify.set_defaults(func=command_audit_verify)

    audit_list = audit_sub.add_parser("list", help="List namespace-scoped audit events")
    audit_list.add_argument("--event-type", action="append", default=[], help="Optional event type filter")
    add_state_argument(audit_list)
    add_scope_arguments(audit_list)
    audit_list.add_argument("--json", action="store_true", help="Emit JSON output")
    audit_list.set_defaults(func=command_audit_list)

    audit_export = audit_sub.add_parser("export", help="Export namespace-scoped audit events")
    audit_export.add_argument("--event-type", action="append", default=[], help="Optional event type filter")
    add_state_argument(audit_export)
    add_scope_arguments(audit_export)
    audit_export.add_argument("--json", action="store_true", help="Emit JSON output")
    audit_export.set_defaults(func=command_audit_list)

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

    search_snapshot = search_sub.add_parser("snapshot", help="Search snapshot operations")
    search_snapshot_sub = search_snapshot.add_subparsers(dest="search_snapshot_command")

    search_snapshot_show = search_snapshot_sub.add_parser("show", help="Show a reproducible search snapshot")
    search_snapshot_show.add_argument("search_snapshot_id", help="Search snapshot ID")
    add_state_argument(search_snapshot_show)
    add_scope_arguments(search_snapshot_show)
    search_snapshot_show.add_argument("--json", action="store_true", help="Emit JSON output")
    search_snapshot_show.set_defaults(func=command_search_snapshot_show)

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

    brief = subcommands.add_parser("brief", help="Evidence-backed brief commands")
    brief_sub = brief.add_subparsers(dest="brief_command")

    brief_create = brief_sub.add_parser("create", help="Create a deterministic brief from an evidence bundle")
    brief_create.add_argument("--evidence-bundle-id", required=True, help="Evidence bundle ID")
    add_state_argument(brief_create)
    add_scope_arguments(brief_create)
    brief_create.add_argument("--json", action="store_true", help="Emit JSON output")
    brief_create.set_defaults(func=command_brief_create)

    brief_show = brief_sub.add_parser("show", help="Show an evidence-backed brief")
    brief_show.add_argument("brief_id", help="Brief ID")
    add_state_argument(brief_show)
    add_scope_arguments(brief_show)
    brief_show.add_argument("--json", action="store_true", help="Emit JSON output")
    brief_show.set_defaults(func=command_brief_show)

    claim = subcommands.add_parser("claim", help="Draft claim commands")
    claim_sub = claim.add_subparsers(dest="claim_command")

    claim_create = claim_sub.add_parser("create", help="Create a draft claim from an evidence bundle")
    claim_create.add_argument("--evidence-bundle-id", help="Evidence bundle ID")
    claim_create.add_argument("--statement", required=True, help="Draft claim statement")
    claim_create.add_argument("--ontology-object-ref", action="append", default=[], help="Promoted ontology object ref used as context only")
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

    claim_basis_export = claim_sub.add_parser("basis-export", help="Export a reproducible evidence basis for a claim")
    claim_basis_export.add_argument("claim_id", help="Claim ID")
    add_state_argument(claim_basis_export)
    add_scope_arguments(claim_basis_export)
    claim_basis_export.add_argument("--json", action="store_true", help="Emit JSON output")
    claim_basis_export.set_defaults(func=command_claim_basis_export)

    capsule = subcommands.add_parser("capsule", help="Knowledge Capsule commands")
    capsule_sub = capsule.add_subparsers(dest="capsule_command")

    capsule_create = capsule_sub.add_parser("create", help="Create a Knowledge Capsule from an evidence-backed claim")
    capsule_create.add_argument("--claim-id", required=True, help="Source claim ID")
    capsule_create.add_argument("--title", required=True, help="Capsule title")
    capsule_create.add_argument("--summary", required=True, help="Reusable understanding summary")
    add_state_argument(capsule_create)
    add_scope_arguments(capsule_create)
    capsule_create.add_argument("--json", action="store_true", help="Emit JSON output")
    capsule_create.set_defaults(func=command_capsule_create)

    capsule_show = capsule_sub.add_parser("show", help="Show a Knowledge Capsule")
    capsule_show.add_argument("capsule_id", help="Capsule ID")
    add_state_argument(capsule_show)
    add_scope_arguments(capsule_show)
    capsule_show.add_argument("--json", action="store_true", help="Emit JSON output")
    capsule_show.set_defaults(func=command_capsule_show)

    decision_card = subcommands.add_parser("decision-card", help="Mission / Decision Card commands")
    decision_card_sub = decision_card.add_subparsers(dest="decision_card_command")

    decision_card_create = decision_card_sub.add_parser("create", help="Create a Decision Card from a mission and claim")
    decision_card_create.add_argument("--goal", required=True, help="Outcome-oriented decision or mission goal")
    decision_card_create.add_argument("--claim-id", required=True, help="Source claim ID")
    decision_card_create.add_argument("--mission-id", required=True, help="Source mission ID")
    add_state_argument(decision_card_create)
    add_scope_arguments(decision_card_create)
    decision_card_create.add_argument("--json", action="store_true", help="Emit JSON output")
    decision_card_create.set_defaults(func=command_decision_card_create)

    decision_card_show = decision_card_sub.add_parser("show", help="Show a Decision Card")
    decision_card_show.add_argument("decision_card_id", help="Decision Card ID")
    add_state_argument(decision_card_show)
    add_scope_arguments(decision_card_show)
    decision_card_show.add_argument("--json", action="store_true", help="Emit JSON output")
    decision_card_show.set_defaults(func=command_decision_card_show)

    correction = subcommands.add_parser("correction", help="Evidence-aware correction commands")
    correction_sub = correction.add_subparsers(dest="correction_command")

    correction_record = correction_sub.add_parser("record", help="Record an evidence-aware human correction")
    correction_record.add_argument("--target-kind", choices=["brief", "claim", "knowledge_capsule", "decision_card", "memory"], required=True)
    correction_record.add_argument("--target-id", required=True, help="Target record ID")
    correction_record.add_argument("--corrected-text", required=True, help="Corrected text or owner judgment")
    correction_record.add_argument("--rationale", required=True, help="Correction rationale")
    correction_record.add_argument("--evidence-bundle-id", help="Optional supporting Evidence Bundle ID")
    add_state_argument(correction_record)
    add_scope_arguments(correction_record)
    correction_record.add_argument("--json", action="store_true", help="Emit JSON output")
    correction_record.set_defaults(func=command_correction_record)

    share = subcommands.add_parser("share", help="Shared item view commands")
    share_sub = share.add_subparsers(dest="share_command")

    share_create = share_sub.add_parser("create", help="Create a trust-state-aware shared view")
    share_create.add_argument("--item-kind", choices=["claim", "knowledge_capsule", "decision_card"], required=True)
    share_create.add_argument("--item-id", required=True, help="Item ID")
    share_create.add_argument("--audience", default="reviewer", help="Recipient role or audience")
    share_create.add_argument("--channel", default="local_share", help="Local sharing channel label")
    add_state_argument(share_create)
    add_scope_arguments(share_create)
    share_create.add_argument("--json", action="store_true", help="Emit JSON output")
    share_create.set_defaults(func=command_share_create)

    share_show = share_sub.add_parser("show", help="Show a shared item view")
    share_show.add_argument("share_id", help="Share ID")
    add_state_argument(share_show)
    add_scope_arguments(share_show)
    share_show.add_argument("--json", action="store_true", help="Emit JSON output")
    share_show.set_defaults(func=command_share_show)

    understand = subcommands.add_parser("understand", help="Understanding and draft ontology commands")
    understand_sub = understand.add_subparsers(dest="understand_command")

    understand_suggest = understand_sub.add_parser("suggest", help="Suggest draft operational structure from an artifact")
    understand_suggest.add_argument("--artifact-id", required=True, help="Source artifact ID")
    understand_suggest.add_argument("--domain", choices=["general", "unknown"], default="general", help="Domain certainty mode")
    add_state_argument(understand_suggest)
    add_scope_arguments(understand_suggest)
    understand_suggest.add_argument("--json", action="store_true", help="Emit JSON output")
    understand_suggest.set_defaults(func=command_understand_suggest)

    understand_promote = understand_sub.add_parser("promote", help="Promote a suggestion into a draft ontology item")
    understand_promote.add_argument("--suggestion-id", required=True, help="Understanding suggestion ID")
    add_state_argument(understand_promote)
    add_scope_arguments(understand_promote)
    understand_promote.add_argument("--json", action="store_true", help="Emit JSON output")
    understand_promote.set_defaults(func=command_understand_promote)

    understand_map = understand_sub.add_parser("map", help="Build an evidence-linked operational map")
    add_state_argument(understand_map)
    add_scope_arguments(understand_map)
    understand_map.add_argument("--json", action="store_true", help="Emit JSON output")
    understand_map.set_defaults(func=command_understand_map)

    understand_contradictions = understand_sub.add_parser("contradictions", help="Detect unresolved fact contradictions")
    add_state_argument(understand_contradictions)
    add_scope_arguments(understand_contradictions)
    understand_contradictions.add_argument("--json", action="store_true", help="Emit JSON output")
    understand_contradictions.set_defaults(func=command_understand_contradictions)

    understand_stale = understand_sub.add_parser("stale-check", help="Check whether a claim needs review because of newer evidence")
    understand_stale.add_argument("--claim-id", required=True, help="Claim ID to check")
    understand_stale.add_argument("--newer-evidence-bundle-id", required=True, help="Newer Evidence Bundle ID")
    add_state_argument(understand_stale)
    add_scope_arguments(understand_stale)
    understand_stale.add_argument("--json", action="store_true", help="Emit JSON output")
    understand_stale.set_defaults(func=command_understand_stale_check)

    understand_change = understand_sub.add_parser("ontology-change", help="Record a versioned ontology item change")
    understand_change.add_argument("--item-id", required=True, help="Ontology item ID")
    understand_change.add_argument("--property", default="label", help="Ontology item property to change")
    understand_change.add_argument("--to-value", required=True, help="New property value")
    add_state_argument(understand_change)
    add_scope_arguments(understand_change)
    understand_change.add_argument("--json", action="store_true", help="Emit JSON output")
    understand_change.set_defaults(func=command_understand_ontology_change)

    ontology = subcommands.add_parser("ontology", help="Ontology suggestion, review, promotion, and profile commands")
    ontology_sub = ontology.add_subparsers(dest="ontology_command")

    ontology_suggest = ontology_sub.add_parser("suggest", help="Generate a draft ontology SuggestionSet from Artifact or Search evidence")
    ontology_suggest.add_argument("--source-type", choices=["artifact", "search"], required=True)
    ontology_suggest.add_argument("--source-id", required=True)
    add_state_argument(ontology_suggest)
    add_scope_arguments(ontology_suggest)
    ontology_suggest.add_argument("--json", action="store_true", help="Emit JSON output")
    ontology_suggest.set_defaults(func=command_ontology_suggest)

    ontology_review = ontology_sub.add_parser("review", help="Select, reject, or defer draft ontology candidates")
    ontology_review.add_argument("suggestion_set_id")
    ontology_review.add_argument("--select", action="append", default=[])
    ontology_review.add_argument("--reject", action="append", default=[])
    ontology_review.add_argument("--defer", action="append", default=[])
    add_state_argument(ontology_review)
    add_scope_arguments(ontology_review)
    ontology_review.add_argument("--json", action="store_true", help="Emit JSON output")
    ontology_review.set_defaults(func=command_ontology_review)

    ontology_promote = ontology_sub.add_parser("promote", help="Explicitly promote selected ontology candidates")
    ontology_promote.add_argument("suggestion_set_id")
    ontology_promote.add_argument("--candidate-id", action="append", default=[])
    add_state_argument(ontology_promote)
    add_scope_arguments(ontology_promote)
    ontology_promote.add_argument("--json", action="store_true", help="Emit JSON output")
    ontology_promote.set_defaults(func=command_ontology_promote)

    ontology_object = ontology_sub.add_parser("object", help="Ontology Object profile commands")
    ontology_object_sub = ontology_object.add_subparsers(dest="ontology_object_command")
    ontology_object_show = ontology_object_sub.add_parser("show", help="Show promoted object profile")
    ontology_object_show.add_argument("object_id")
    add_state_argument(ontology_object_show)
    add_scope_arguments(ontology_object_show)
    ontology_object_show.add_argument("--json", action="store_true", help="Emit JSON output")
    ontology_object_show.set_defaults(func=command_ontology_object_show)

    ontology_draft = ontology_sub.add_parser("draft-truth-test", help="Verify draft suggestions cannot be used as truth")
    ontology_draft.add_argument("suggestion_set_id")
    ontology_draft.add_argument("--candidate-id", required=True)
    ontology_draft.add_argument("--purpose", default="claim_or_action_truth")
    add_state_argument(ontology_draft)
    add_scope_arguments(ontology_draft)
    ontology_draft.add_argument("--json", action="store_true", help="Emit JSON output")
    ontology_draft.set_defaults(func=command_ontology_draft_truth_test)

    ontology_invalid = ontology_sub.add_parser("invalid-graph-test", help="Verify helpful failure for invalid ontology graph")
    add_state_argument(ontology_invalid)
    add_scope_arguments(ontology_invalid)
    ontology_invalid.add_argument("--json", action="store_true", help="Emit JSON output")
    ontology_invalid.set_defaults(func=command_ontology_invalid_graph_test)

    ontology_supersede = ontology_sub.add_parser("supersede", help="Supersede a promoted object property with a patch ChangeSet")
    ontology_supersede.add_argument("object_id")
    ontology_supersede.add_argument("--property", required=True)
    ontology_supersede.add_argument("--to-value", required=True)
    ontology_supersede.add_argument("--rationale", required=True)
    add_state_argument(ontology_supersede)
    add_scope_arguments(ontology_supersede)
    ontology_supersede.add_argument("--json", action="store_true", help="Emit JSON output")
    ontology_supersede.set_defaults(func=command_ontology_supersede)

    workspace = subcommands.add_parser("workspace", help="Workspace governance commands")
    workspace_sub = workspace.add_subparsers(dest="workspace_command")

    workspace_show = workspace_sub.add_parser("show", help="Show active workspace and context boundary")
    add_state_argument(workspace_show)
    add_scope_arguments(workspace_show)
    workspace_show.add_argument("--json", action="store_true", help="Emit JSON output")
    workspace_show.set_defaults(func=command_workspace_show)

    workspace_mode = workspace_sub.add_parser("mode", help="Workspace mode operations")
    workspace_mode_sub = workspace_mode.add_subparsers(dest="workspace_mode_command")

    workspace_mode_set = workspace_mode_sub.add_parser("set", help="Set workspace mode")
    workspace_mode_set.add_argument("mode", choices=["manual", "assist", "autopilot", "locked"])
    add_state_argument(workspace_mode_set)
    add_scope_arguments(workspace_mode_set)
    workspace_mode_set.add_argument("--json", action="store_true", help="Emit JSON output")
    workspace_mode_set.set_defaults(func=command_workspace_mode_set)

    workspace_mode_show = workspace_mode_sub.add_parser("show", help="Show workspace mode")
    add_state_argument(workspace_mode_show)
    add_scope_arguments(workspace_mode_show)
    workspace_mode_show.add_argument("--json", action="store_true", help="Emit JSON output")
    workspace_mode_show.set_defaults(func=command_workspace_mode_show)

    autopilot = subcommands.add_parser("autopilot", help="Autopilot readiness commands")
    autopilot_sub = autopilot.add_subparsers(dest="autopilot_command")

    autopilot_readiness = autopilot_sub.add_parser("readiness", help="Recommend conservative Autopilot readiness from local history")
    add_state_argument(autopilot_readiness)
    add_scope_arguments(autopilot_readiness)
    autopilot_readiness.add_argument("--json", action="store_true", help="Emit JSON output")
    autopilot_readiness.set_defaults(func=command_autopilot_readiness)

    autopilot_metrics = autopilot_sub.add_parser("metrics", help="Report outcome-quality-first autonomy metrics")
    autopilot_metrics.add_argument("--mission-id", required=True, help="Mission ID")
    autopilot_metrics.add_argument("--outcome-id", required=True, help="Mission outcome ID")
    add_state_argument(autopilot_metrics)
    add_scope_arguments(autopilot_metrics)
    autopilot_metrics.add_argument("--json", action="store_true", help="Emit JSON output")
    autopilot_metrics.set_defaults(func=command_autopilot_metrics)

    memory = subcommands.add_parser("memory", help="Durable owner-approved memory commands")
    memory_sub = memory.add_subparsers(dest="memory_command")

    memory_create = memory_sub.add_parser("create", help="Create owner-approved memory from an Evidence Bundle")
    memory_create.add_argument("--evidence-bundle-id", required=True, help="Evidence Bundle ID")
    memory_create.add_argument("--statement", required=True, help="Memory statement")
    memory_create.add_argument("--trust-state", choices=["draft", "evidence_backed", "approved"], default="evidence_backed", help="Memory trust state")
    memory_create.add_argument("--status", choices=["draft", "owner_approved"], default="owner_approved", help="Memory lifecycle status")
    memory_create.add_argument("--memory-type", default="durable_fact", help="Memory type")
    memory_create.add_argument("--synthesis-mode", choices=["owner_approved", "auto"], default="owner_approved", help="Memory synthesis mode")
    add_state_argument(memory_create)
    add_scope_arguments(memory_create)
    memory_create.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_create.set_defaults(func=command_memory_create)

    memory_show = memory_sub.add_parser("show", help="Show an owner-approved memory record")
    memory_show.add_argument("memory_id", help="Memory ID")
    add_state_argument(memory_show)
    add_scope_arguments(memory_show)
    memory_show.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_show.set_defaults(func=command_memory_show)

    memory_raw = memory_sub.add_parser("raw-agent-note", help="Create a non-canonical raw agent memory candidate")
    memory_raw.add_argument("--statement", required=True, help="Raw agent memory statement")
    add_state_argument(memory_raw)
    add_scope_arguments(memory_raw)
    memory_raw.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_raw.set_defaults(func=command_memory_raw_agent_note)

    memory_conflict = memory_sub.add_parser("conflict-test", help="Resolve raw-memory conflict against archive evidence")
    memory_conflict.add_argument("--raw-memory-id", required=True, help="Raw agent memory ID")
    memory_conflict.add_argument("--evidence-bundle-id", required=True, help="Evidence Bundle ID")
    memory_conflict.add_argument("--question", required=True, help="Question being answered")
    add_state_argument(memory_conflict)
    add_scope_arguments(memory_conflict)
    memory_conflict.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_conflict.set_defaults(func=command_memory_conflict_test)

    memory_answer = memory_sub.add_parser("answer", help="Answer from owner-approved memory in the active scope")
    memory_answer.add_argument("--question", required=True, help="Question to answer")
    add_state_argument(memory_answer)
    add_scope_arguments(memory_answer)
    memory_answer.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_answer.set_defaults(func=command_memory_answer)

    memory_control_center = memory_sub.add_parser("control-center", help="Show memory sovereignty controls")
    add_state_argument(memory_control_center)
    add_scope_arguments(memory_control_center)
    memory_control_center.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_control_center.set_defaults(func=command_memory_control_center)

    memory_temp = memory_sub.add_parser("temporary-session", help="Run a no-memory local session record")
    memory_temp.add_argument("--note", required=True, help="Temporary session note")
    add_state_argument(memory_temp)
    add_scope_arguments(memory_temp)
    memory_temp.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_temp.set_defaults(func=command_memory_temporary_session)

    memory_correct = memory_sub.add_parser("correct", help="Correct a memory with owner or evidence provenance")
    memory_correct.add_argument("memory_id", help="Memory ID")
    memory_correct.add_argument("--corrected-text", required=True, help="Corrected memory statement")
    memory_correct.add_argument("--rationale", required=True, help="Correction rationale")
    memory_correct.add_argument("--evidence-bundle-id", help="Evidence Bundle supporting the correction")
    add_state_argument(memory_correct)
    add_scope_arguments(memory_correct)
    memory_correct.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_correct.set_defaults(func=command_memory_correct)

    memory_control = memory_sub.add_parser("control", help="Apply a memory sovereignty control")
    memory_control.add_argument("memory_id", help="Memory ID")
    memory_control.add_argument("--action", required=True, choices=["forget", "rollback", "demote", "promote", "disable-influence", "limit-scope"], help="Control action")
    add_state_argument(memory_control)
    add_scope_arguments(memory_control)
    memory_control.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_control.set_defaults(func=command_memory_control)

    memory_freshness = memory_sub.add_parser("freshness", help="Check memory freshness against newer evidence")
    memory_freshness.add_argument("memory_id", help="Memory ID")
    memory_freshness.add_argument("--newer-evidence-bundle-id", required=True, help="Newer Evidence Bundle ID")
    add_state_argument(memory_freshness)
    add_scope_arguments(memory_freshness)
    memory_freshness.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_freshness.set_defaults(func=command_memory_freshness)

    memory_quarantine = memory_sub.add_parser("quarantine-check", help="Quarantine unsafe memory write attempts")
    memory_quarantine.add_argument("--artifact-id", required=True, help="Source artifact ID")
    memory_quarantine.add_argument("--statement", required=True, help="Attempted memory statement")
    add_state_argument(memory_quarantine)
    add_scope_arguments(memory_quarantine)
    memory_quarantine.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_quarantine.set_defaults(func=command_memory_quarantine_check)

    memory_export = memory_sub.add_parser("export", help="Export memory and wiki state")
    add_state_argument(memory_export)
    add_scope_arguments(memory_export)
    memory_export.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_export.set_defaults(func=command_memory_export)

    memory_adapt = memory_sub.add_parser("adapt", help="Record namespace-local memory adaptation")
    memory_adapt.add_argument("--preference", required=True, help="Namespace-local adaptation preference")
    memory_adapt.add_argument("--source-memory-id", help="Source memory ID")
    add_state_argument(memory_adapt)
    add_scope_arguments(memory_adapt)
    memory_adapt.add_argument("--json", action="store_true", help="Emit JSON output")
    memory_adapt.set_defaults(func=command_memory_adapt)

    namespace = subcommands.add_parser("namespace", help="Namespace promotion commands")
    namespace_sub = namespace.add_subparsers(dest="namespace_command")

    namespace_promote = namespace_sub.add_parser("promote", help="Explicitly promote a scoped item with provenance")
    namespace_promote.add_argument("--source-kind", choices=["memory"], default="memory")
    namespace_promote.add_argument("--source-id", required=True, help="Source item ID")
    namespace_promote.add_argument("--target-tenant-id", help="Target tenant; defaults to source tenant")
    namespace_promote.add_argument("--target-owner-id", required=True, help="Target owner")
    namespace_promote.add_argument("--target-namespace-id", required=True, help="Target namespace")
    namespace_promote.add_argument("--target-workspace-id", required=True, help="Target workspace")
    namespace_promote.add_argument("--mode", choices=["copy_with_provenance", "reference", "share", "promote_to_approved_truth"], default="copy_with_provenance")
    namespace_promote.add_argument("--principal-id", default="local-user", help="Actor requesting promotion")
    namespace_promote.add_argument("--principal-role", choices=["org_admin", "org_approver"], default="org_admin")
    add_state_argument(namespace_promote)
    add_scope_arguments(namespace_promote)
    namespace_promote.add_argument("--json", action="store_true", help="Emit JSON output")
    namespace_promote.set_defaults(func=command_namespace_promote)

    namespace_audit_export = namespace_sub.add_parser("audit-export", help="Export namespace-scoped audit events")
    namespace_audit_export.add_argument("--event-type", action="append", default=[], help="Optional event type filter")
    add_state_argument(namespace_audit_export)
    add_scope_arguments(namespace_audit_export)
    namespace_audit_export.add_argument("--json", action="store_true", help="Emit JSON output")
    namespace_audit_export.set_defaults(func=command_namespace_audit_export)

    namespace_recovery = namespace_sub.add_parser("recovery-test", help="Revoke or rollback a mistaken namespace promotion")
    namespace_recovery.add_argument("--promotion-id", required=True, help="Namespace promotion ID")
    namespace_recovery.add_argument("--reason", required=True, help="Recovery reason")
    add_state_argument(namespace_recovery)
    add_scope_arguments(namespace_recovery)
    namespace_recovery.add_argument("--json", action="store_true", help="Emit JSON output")
    namespace_recovery.set_defaults(func=command_namespace_recovery_test)

    namespace_learning_boundary = namespace_sub.add_parser("product-learning-boundary-test", help="Verify product learning cannot consume raw namespace truth by default")
    add_state_argument(namespace_learning_boundary)
    add_scope_arguments(namespace_learning_boundary)
    namespace_learning_boundary.add_argument("--json", action="store_true", help="Emit JSON output")
    namespace_learning_boundary.set_defaults(func=command_namespace_product_learning_boundary_test)

    access = subcommands.add_parser("access", help="Local deterministic access-control commands")
    access_sub = access.add_subparsers(dest="access_command")

    access_evaluate = access_sub.add_parser("evaluate", help="Evaluate local RBAC/ABAC policy without external calls")
    access_evaluate.add_argument("--principal-id", default="local-user", help="Principal identifier")
    access_evaluate.add_argument("--principal-role", choices=["personal_user", "org_member", "org_approver", "org_admin"], default="personal_user")
    access_evaluate.add_argument("--principal-attributes", default="", help="Comma-separated principal attributes")
    access_evaluate.add_argument("--action", choices=["read", "write", "promote", "search", "summarize", "extract_memory", "use_in_action", "approve", "execute", "configure", "configure_autopilot", "install_pack", "aggregate_learning"], required=True)
    access_evaluate.add_argument("--resource-kind", default="memory")
    access_evaluate.add_argument("--resource-id", default="resource")
    access_evaluate.add_argument("--resource-tenant-id", help="Resource tenant; defaults to active tenant")
    access_evaluate.add_argument("--resource-owner-id", help="Resource owner; defaults to active owner")
    access_evaluate.add_argument("--resource-namespace-id", help="Resource namespace; defaults to active namespace")
    access_evaluate.add_argument("--resource-workspace-id", help="Resource workspace; defaults to active workspace")
    access_evaluate.add_argument("--classification", choices=["public", "internal", "confidential", "restricted", "secret"], default="internal")
    access_evaluate.add_argument("--mission-authority", choices=["none", "draft", "active", "approved"], default="none")
    add_state_argument(access_evaluate)
    add_scope_arguments(access_evaluate)
    access_evaluate.add_argument("--json", action="store_true", help="Emit JSON output")
    access_evaluate.set_defaults(func=command_access_evaluate)

    learning = subcommands.add_parser("learning", help="Action outcome learning commands")
    learning_sub = learning.add_subparsers(dest="learning_command")

    learning_record = learning_sub.add_parser("record", help="Record a learning item from an executed Action")
    learning_record.add_argument("--action-id", required=True, help="Executed Action ID")
    learning_record.add_argument("--lesson", required=True, help="Learning note")
    add_state_argument(learning_record)
    add_scope_arguments(learning_record)
    learning_record.add_argument("--json", action="store_true", help="Emit JSON output")
    learning_record.set_defaults(func=command_learning_record)

    learning_show = learning_sub.add_parser("show", help="Show a learning record")
    learning_show.add_argument("learning_id", help="Learning ID")
    add_state_argument(learning_show)
    add_scope_arguments(learning_show)
    learning_show.add_argument("--json", action="store_true", help="Emit JSON output")
    learning_show.set_defaults(func=command_learning_show)

    experience = subcommands.add_parser("experience", help="Mission trajectory and experience learning commands")
    experience_sub = experience.add_subparsers(dest="experience_command")

    connected_outcome = experience_sub.add_parser("connected-outcome", help="Record a connected-system outcome as evidence-backed experience")
    connected_outcome.add_argument("--action-id", required=True, help="Executed Action ID")
    connected_outcome.add_argument("--evidence-bundle-id", required=True, help="Evidence Bundle ID for the observed outcome")
    connected_outcome.add_argument("--outcome-status", choices=["success", "failed", "cancelled", "rolled_back"], required=True)
    connected_outcome.add_argument("--summary", required=True, help="Outcome summary")
    add_state_argument(connected_outcome)
    add_scope_arguments(connected_outcome)
    connected_outcome.add_argument("--json", action="store_true", help="Emit JSON output")
    connected_outcome.set_defaults(func=command_experience_connected_outcome)

    trajectory = experience_sub.add_parser("trajectory", help="Mission trajectory commands")
    trajectory_sub = trajectory.add_subparsers(dest="trajectory_command")

    trajectory_record = trajectory_sub.add_parser("record", help="Record a Mission Trajectory Ledger item")
    trajectory_record.add_argument("--mission-id", required=True, help="Mission ID")
    trajectory_record.add_argument("--outcome-status", choices=["success", "failed", "cancelled", "rolled_back"], required=True)
    trajectory_record.add_argument("--outcome-summary", required=True, help="Outcome summary")
    trajectory_record.add_argument("--owner-acceptance", choices=["accepted", "rejected", "pending"], default="accepted")
    trajectory_record.add_argument("--failure-reason", help="Failure or rollback reason")
    trajectory_record.add_argument("--recovery-attempt", help="Recovery or escalation attempt")
    trajectory_record.add_argument("--connected-outcome-id", help="Connected outcome ID")
    add_state_argument(trajectory_record)
    add_scope_arguments(trajectory_record)
    trajectory_record.add_argument("--json", action="store_true", help="Emit JSON output")
    trajectory_record.set_defaults(func=command_experience_trajectory_record)

    library = experience_sub.add_parser("library", help="Browse scoped Experience Library")
    add_state_argument(library)
    add_scope_arguments(library)
    library.add_argument("--json", action="store_true", help="Emit JSON output")
    library.set_defaults(func=command_experience_library)

    experience_search = experience_sub.add_parser("search", help="Search scoped experience")
    experience_search.add_argument("--query", required=True, help="Experience search query")
    add_state_argument(experience_search)
    add_scope_arguments(experience_search)
    experience_search.add_argument("--json", action="store_true", help="Emit JSON output")
    experience_search.set_defaults(func=command_experience_search)

    recommend = experience_sub.add_parser("recommend", help="Recommend scoped experience for a new mission")
    recommend.add_argument("--mission-id", required=True, help="Mission ID receiving recommendations")
    recommend.add_argument("--query", required=True, help="Recommendation query")
    add_state_argument(recommend)
    add_scope_arguments(recommend)
    recommend.add_argument("--json", action="store_true", help="Emit JSON output")
    recommend.set_defaults(func=command_experience_recommend)

    lesson = experience_sub.add_parser("lesson", help="Lesson candidate commands")
    lesson_sub = lesson.add_subparsers(dest="lesson_command")

    lesson_propose = lesson_sub.add_parser("propose", help="Propose a lesson from a trajectory")
    lesson_propose.add_argument("--trajectory-id", required=True, help="Trajectory ID")
    lesson_propose.add_argument("--lesson", required=True, help="Lesson text")
    lesson_propose.add_argument("--applies-when", required=True, help="Applicability boundary")
    lesson_propose.add_argument("--does-not-apply-when", required=True, help="Non-applicability boundary")
    lesson_propose.add_argument("--confidence", choices=["low", "medium", "high"], default="medium")
    add_state_argument(lesson_propose)
    add_scope_arguments(lesson_propose)
    lesson_propose.add_argument("--json", action="store_true", help="Emit JSON output")
    lesson_propose.set_defaults(func=command_experience_lesson_propose)

    lesson_promote = lesson_sub.add_parser("promote", help="Promote a lesson one ladder step")
    lesson_promote.add_argument("lesson_id", help="Lesson ID")
    lesson_promote.add_argument(
        "--stage",
        choices=[
            "observation",
            "candidate_lesson",
            "workspace_memory",
            "mission_playbook",
            "organization_approved_rule",
            "solution_pack_or_product_learning_proposal",
        ],
        required=True,
    )
    add_state_argument(lesson_promote)
    add_scope_arguments(lesson_promote)
    lesson_promote.add_argument("--json", action="store_true", help="Emit JSON output")
    lesson_promote.set_defaults(func=command_experience_lesson_promote)

    lesson_control = lesson_sub.add_parser("control", help="Demote, rollback, disable, or revise a lesson")
    lesson_control.add_argument("lesson_id", help="Lesson ID")
    lesson_control.add_argument("--action", choices=["demote", "rollback", "disable", "revise"], required=True)
    add_state_argument(lesson_control)
    add_scope_arguments(lesson_control)
    lesson_control.add_argument("--json", action="store_true", help="Emit JSON output")
    lesson_control.set_defaults(func=command_experience_lesson_control)

    behavior_signal = experience_sub.add_parser("behavior-signal", help="Record a supporting behavior signal")
    behavior_signal.add_argument("--trajectory-id", required=True, help="Trajectory ID")
    behavior_signal.add_argument("--signal", required=True, help="Behavior signal")
    behavior_signal.add_argument("--interpretation", required=True, help="Signal interpretation")
    add_state_argument(behavior_signal)
    add_scope_arguments(behavior_signal)
    behavior_signal.add_argument("--json", action="store_true", help="Emit JSON output")
    behavior_signal.set_defaults(func=command_experience_behavior_signal)

    model_eval = experience_sub.add_parser("model-eval", help="Record a local_test model self-evaluation")
    model_eval.add_argument("--trajectory-id", required=True, help="Trajectory ID")
    model_eval.add_argument("--score", required=True, help="Model self-evaluation score")
    model_eval.add_argument("--rationale", required=True, help="Model self-evaluation rationale")
    add_state_argument(model_eval)
    add_scope_arguments(model_eval)
    model_eval.add_argument("--json", action="store_true", help="Emit JSON output")
    model_eval.set_defaults(func=command_experience_model_eval)

    product_improvement = experience_sub.add_parser("product-improvement", help="Product improvement proposal commands")
    product_improvement_sub = product_improvement.add_subparsers(dest="product_improvement_command")

    product_improvement_propose = product_improvement_sub.add_parser("propose", help="Propose a product improvement from a lesson")
    product_improvement_propose.add_argument("--lesson-id", required=True, help="Lesson ID")
    product_improvement_propose.add_argument("--proposal", required=True, help="Proposal text")
    add_state_argument(product_improvement_propose)
    add_scope_arguments(product_improvement_propose)
    product_improvement_propose.add_argument("--json", action="store_true", help="Emit JSON output")
    product_improvement_propose.set_defaults(func=command_experience_product_improvement_propose)

    local_adapt = experience_sub.add_parser("local-adapt", help="Record namespace-local adaptation")
    local_adapt.add_argument("--lesson-id", required=True, help="Lesson ID")
    local_adapt.add_argument("--preference", required=True, help="Local preference")
    add_state_argument(local_adapt)
    add_scope_arguments(local_adapt)
    local_adapt.add_argument("--json", action="store_true", help="Emit JSON output")
    local_adapt.set_defaults(func=command_experience_local_adapt)

    local_adapt_reset = experience_sub.add_parser("local-adapt-reset", help="Reset namespace-local adaptation")
    local_adapt_reset.add_argument("adaptation_id", help="Local adaptation ID")
    add_state_argument(local_adapt_reset)
    add_scope_arguments(local_adapt_reset)
    local_adapt_reset.add_argument("--json", action="store_true", help="Emit JSON output")
    local_adapt_reset.set_defaults(func=command_experience_local_adapt_reset)

    metrics = experience_sub.add_parser("metrics", help="Generate outcome quality metrics")
    add_state_argument(metrics)
    add_scope_arguments(metrics)
    metrics.add_argument("--json", action="store_true", help="Emit JSON output")
    metrics.set_defaults(func=command_experience_metrics)

    experience_export = experience_sub.add_parser("export", help="Export scoped experience with redaction")
    add_state_argument(experience_export)
    add_scope_arguments(experience_export)
    experience_export.add_argument("--json", action="store_true", help="Emit JSON output")
    experience_export.set_defaults(func=command_experience_export)

    model = subcommands.add_parser("model", help="Model capability registry commands")
    model_sub = model.add_subparsers(dest="model_command")

    model_list = model_sub.add_parser("list", help="List local deterministic and registry-only model capabilities")
    add_state_argument(model_list)
    add_scope_arguments(model_list)
    model_list.add_argument("--json", action="store_true", help="Emit JSON output")
    model_list.set_defaults(func=command_model_list)

    brain = subcommands.add_parser("brain", help="Replaceable brain routing and ledger commands")
    brain_sub = brain.add_subparsers(dest="brain_command")

    brain_route = brain_sub.add_parser("route", help="Dry-run a policy-aware model routing decision")
    brain_route.add_argument("--task", required=True, help="Task file, ref, or short task descriptor")
    brain_route.add_argument("--task-type", choices=["planning", "judge", "retrieval", "extraction", "tool_use"], default="planning")
    brain_route.add_argument("--mission-type", choices=["routine", "ambiguous_research", "externally_impactful", "safety_sensitive"], default="routine")
    brain_route.add_argument("--sensitivity", choices=["public", "internal", "confidential", "restricted"], default="internal")
    brain_route.add_argument("--risk", choices=["low", "medium", "high", "safety_sensitive"], default="low")
    brain_route.add_argument("--owner-preference", choices=["local_test", "local_semantic", "lowest_cost", "highest_grounding"], default="local_test")
    brain_route.add_argument("--max-cost-usd", type=float, default=0.0, help="Maximum route cost per 1k tokens")
    brain_route.add_argument("--max-latency-ms", type=int, default=2000, help="Maximum acceptable route latency")
    brain_route.add_argument("--override-provider", help="Requested override provider")
    brain_route.add_argument("--override-model", help="Requested override model")
    brain_route.add_argument("--dry-run", action="store_true", help="Required dry-run for routing decisions")
    add_state_argument(brain_route)
    add_scope_arguments(brain_route)
    brain_route.add_argument("--json", action="store_true", help="Emit JSON output")
    brain_route.set_defaults(func=command_brain_route)

    brain_switch = brain_sub.add_parser("switch", help="Record a workspace provider/model switch without changing durable product state")
    brain_switch.add_argument("--provider", required=True, help="Replacement provider")
    brain_switch.add_argument("--model", required=True, help="Replacement model")
    brain_switch.add_argument("--evidence-bundle-id", help="Existing Evidence Bundle to verify after switch")
    brain_switch.add_argument("--mission-id", help="Existing mission to verify after switch")
    add_state_argument(brain_switch)
    add_scope_arguments(brain_switch)
    brain_switch.add_argument("--json", action="store_true", help="Emit JSON output")
    brain_switch.set_defaults(func=command_brain_switch)

    brain_ledger = brain_sub.add_parser("ledger", help="Show namespace-local Brain Performance Ledger")
    add_state_argument(brain_ledger)
    add_scope_arguments(brain_ledger)
    brain_ledger.add_argument("--json", action="store_true", help="Emit JSON output")
    brain_ledger.set_defaults(func=command_brain_ledger)

    brain_aggregate = brain_sub.add_parser("aggregate-test", help="Test cross-namespace Brain Performance Ledger aggregation policy")
    brain_aggregate.add_argument("--source-namespace", required=True, help="Source namespace")
    brain_aggregate.add_argument("--target-namespace", required=True, help="Target namespace")
    brain_aggregate.add_argument("--opt-in", action="store_true", help="Record explicit opt-in for aggregation")
    add_state_argument(brain_aggregate)
    add_scope_arguments(brain_aggregate)
    brain_aggregate.add_argument("--json", action="store_true", help="Emit JSON output")
    brain_aggregate.set_defaults(func=command_brain_aggregate_test)

    judge = subcommands.add_parser("judge", help="LLM-as-judge, adjudication, and calibration commands")
    judge_sub = judge.add_subparsers(dest="judge_command")

    judge_run = judge_sub.add_parser("run", help="Record a rubric-bound judge assessment")
    judge_run.add_argument("--route-id", required=True, help="Brain route ID used for judging")
    judge_run.add_argument("--subject", required=True, help="Subject being judged")
    judge_run.add_argument("--rubric", required=True, help="Rubric description")
    judge_run.add_argument("--evidence-ref", required=True, help="Evidence ref supporting the judgment")
    judge_run.add_argument("--ambiguity", choices=["low", "medium", "high"], default="high")
    add_state_argument(judge_run)
    add_scope_arguments(judge_run)
    judge_run.add_argument("--json", action="store_true", help="Emit JSON output")
    judge_run.set_defaults(func=command_judge_run)

    judge_conflict = judge_sub.add_parser("conflict", help="Record objective evidence overriding a judge result")
    judge_conflict.add_argument("--judge-record-id", required=True, help="Judge record ID")
    judge_conflict.add_argument("--objective-evidence", required=True, help="Objective failure or success evidence")
    add_state_argument(judge_conflict)
    add_scope_arguments(judge_conflict)
    judge_conflict.add_argument("--json", action="store_true", help="Emit JSON output")
    judge_conflict.set_defaults(func=command_judge_conflict)

    judge_accept = judge_sub.add_parser("accept", help="Record owner acceptance or rejection as a grounding signal")
    judge_accept.add_argument("--judge-record-id", required=True, help="Judge record ID")
    judge_accept.add_argument("--acceptance", choices=["accepted", "rejected"], required=True)
    add_state_argument(judge_accept)
    add_scope_arguments(judge_accept)
    judge_accept.add_argument("--json", action="store_true", help="Emit JSON output")
    judge_accept.set_defaults(func=command_judge_accept)

    judge_recommend = judge_sub.add_parser("recommend", help="Create a governed candidate lesson from a judge recommendation")
    judge_recommend.add_argument("--judge-record-id", required=True, help="Judge record ID")
    judge_recommend.add_argument("--recommendation", required=True, help="Recommendation text")
    add_state_argument(judge_recommend)
    add_scope_arguments(judge_recommend)
    judge_recommend.add_argument("--json", action="store_true", help="Emit JSON output")
    judge_recommend.set_defaults(func=command_judge_recommend)

    judge_disagreement = judge_sub.add_parser("disagreement-test", help="Record evidence-weighted disagreement adjudication")
    judge_disagreement.add_argument("--risk", choices=["low", "medium", "high", "safety_sensitive"], default="high")
    add_state_argument(judge_disagreement)
    add_scope_arguments(judge_disagreement)
    judge_disagreement.add_argument("--json", action="store_true", help="Emit JSON output")
    judge_disagreement.set_defaults(func=command_judge_disagreement_test)

    judge_calibration = judge_sub.add_parser("calibration", help="Generate judge calibration and bias report")
    add_state_argument(judge_calibration)
    add_scope_arguments(judge_calibration)
    judge_calibration.add_argument("--json", action="store_true", help="Emit JSON output")
    judge_calibration.set_defaults(func=command_judge_calibration)

    agent = subcommands.add_parser("agent", help="Agent role, orchestration, diagnosis, and replay commands")
    agent_sub = agent.add_subparsers(dest="agent_command")

    agent_list = agent_sub.add_parser("list", help="List Orchestrator and specialist agent roles")
    add_state_argument(agent_list)
    add_scope_arguments(agent_list)
    agent_list.add_argument("--json", action="store_true", help="Emit JSON output")
    agent_list.set_defaults(func=command_agent_list)

    agent_orchestrate = agent_sub.add_parser("orchestrate", help="Create an Orchestrator-led mission trace")
    agent_orchestrate.add_argument("--mission-id", required=True, help="Mission ID")
    add_state_argument(agent_orchestrate)
    add_scope_arguments(agent_orchestrate)
    agent_orchestrate.add_argument("--json", action="store_true", help="Emit JSON output")
    agent_orchestrate.set_defaults(func=command_agent_orchestrate)

    agent_mutation = agent_sub.add_parser("direct-mutation-test", help="Verify agents cannot directly mutate truth or source systems")
    agent_mutation.add_argument("--role-id", required=True, help="Agent role ID or role key")
    agent_mutation.add_argument("--target", required=True, help="Denied mutation target")
    add_state_argument(agent_mutation)
    add_scope_arguments(agent_mutation)
    agent_mutation.add_argument("--json", action="store_true", help="Emit JSON output")
    agent_mutation.set_defaults(func=command_agent_direct_mutation_test)

    agent_brain = agent_sub.add_parser("brain-switch", help="Record a provider/model switch without changing the role contract")
    agent_brain.add_argument("--role-id", required=True, help="Agent role ID or role key")
    agent_brain.add_argument("--provider", required=True, help="Replacement model provider")
    agent_brain.add_argument("--model", required=True, help="Replacement model name")
    add_state_argument(agent_brain)
    add_scope_arguments(agent_brain)
    agent_brain.add_argument("--json", action="store_true", help="Emit JSON output")
    agent_brain.set_defaults(func=command_agent_brain_switch)

    agent_contract = agent_sub.add_parser("contract-update", help="Record a versioned Agent Role Contract update")
    agent_contract.add_argument("--role-id", required=True, help="Agent role ID or role key")
    agent_contract.add_argument("--change-summary", required=True, help="Operator-visible change summary")
    add_state_argument(agent_contract)
    add_scope_arguments(agent_contract)
    agent_contract.add_argument("--json", action="store_true", help="Emit JSON output")
    agent_contract.set_defaults(func=command_agent_contract_update)

    agent_prompt = agent_sub.add_parser("prompt-authority-test", help="Verify prompt-only changes cannot expand authority")
    agent_prompt.add_argument("--role-id", required=True, help="Agent role ID or role key")
    agent_prompt.add_argument("--requested-tool", required=True, help="Tool the prompt tried to add")
    agent_prompt.add_argument("--requested-memory-scope", required=True, help="Memory scope the prompt tried to add")
    agent_prompt.add_argument("--requested-authority", required=True, help="Authority the prompt tried to add")
    add_state_argument(agent_prompt)
    add_scope_arguments(agent_prompt)
    agent_prompt.add_argument("--json", action="store_true", help="Emit JSON output")
    agent_prompt.set_defaults(func=command_agent_prompt_authority_test)

    agent_diagnose = agent_sub.add_parser("diagnose", help="Record useful diagnosis for an agent failure")
    agent_diagnose.add_argument("trace_id", help="Agent mission trace ID")
    agent_diagnose.add_argument("--role-id", required=True, help="Agent role ID or role key")
    agent_diagnose.add_argument("--failure-kind", default="timeout", help="Failure kind")
    add_state_argument(agent_diagnose)
    add_scope_arguments(agent_diagnose)
    agent_diagnose.add_argument("--json", action="store_true", help="Emit JSON output")
    agent_diagnose.set_defaults(func=command_agent_diagnose)

    agent_pack_capability = agent_sub.add_parser("pack-capability-test", help="Verify Agent Pack supplied agents obey activation grants")
    agent_pack_capability.add_argument("--role-id", required=True, help="Agent role ID or role key")
    agent_pack_capability.add_argument("--pack-id", required=True, help="Agent Pack ID")
    agent_pack_capability.add_argument("--capability", required=True, help="Capability to test")
    add_state_argument(agent_pack_capability)
    add_scope_arguments(agent_pack_capability)
    agent_pack_capability.add_argument("--json", action="store_true", help="Emit JSON output")
    agent_pack_capability.set_defaults(func=command_agent_pack_capability_test)

    agent_replay = agent_sub.add_parser("replay", help="Create a reviewable agent mission replay")
    agent_replay.add_argument("trace_id", help="Agent mission trace ID")
    add_state_argument(agent_replay)
    add_scope_arguments(agent_replay)
    agent_replay.add_argument("--json", action="store_true", help="Emit JSON output")
    agent_replay.set_defaults(func=command_agent_replay)

    role = subcommands.add_parser("role", help="Agent Role Contract commands")
    role_sub = role.add_subparsers(dest="role_command")

    role_show = role_sub.add_parser("show", help="Show daily-user role card or operator contract")
    role_show.add_argument("role_id", help="Agent role ID or role key")
    role_show.add_argument("--view", choices=["user", "operator", "both"], default="user")
    add_state_argument(role_show)
    add_scope_arguments(role_show)
    role_show.add_argument("--json", action="store_true", help="Emit JSON output")
    role_show.set_defaults(func=command_role_show)

    pack = subcommands.add_parser("pack", help="Agent Pack registry, install, activation, and rollout commands")
    pack_sub = pack.add_subparsers(dest="pack_command")

    pack_import = pack_sub.add_parser("import", help="Import a local Agent Pack manifest into the registry")
    pack_import.add_argument("--manifest", required=True, help="Path to the local Agent Pack manifest")
    add_state_argument(pack_import)
    add_scope_arguments(pack_import)
    pack_import.add_argument("--json", action="store_true", help="Emit JSON output")
    pack_import.set_defaults(func=command_pack_import)

    pack_list = pack_sub.add_parser("list", help="List local registry Agent Packs")
    add_state_argument(pack_list)
    add_scope_arguments(pack_list)
    pack_list.add_argument("--json", action="store_true", help="Emit JSON output")
    pack_list.set_defaults(func=command_pack_list)

    pack_show = pack_sub.add_parser("show", help="Show Agent Pack manifest, trust, and component details")
    pack_show.add_argument("pack_id", help="Agent Pack ID")
    add_state_argument(pack_show)
    add_scope_arguments(pack_show)
    pack_show.add_argument("--json", action="store_true", help="Emit JSON output")
    pack_show.set_defaults(func=command_pack_show)

    pack_install = pack_sub.add_parser("install", help="Install an Agent Pack without activation authority")
    pack_install.add_argument("pack_id", help="Agent Pack ID")
    pack_install.add_argument("--version", help="Pack version to install; defaults to registry version")
    pack_install.add_argument("--dry-run", action="store_true", help="Preview install without writing the install record")
    add_state_argument(pack_install)
    add_scope_arguments(pack_install)
    pack_install.add_argument("--json", action="store_true", help="Emit JSON output")
    pack_install.set_defaults(func=command_pack_install)

    pack_activate = pack_sub.add_parser("activate", help="Activate an installed Agent Pack with explicit grants")
    pack_activate.add_argument("pack_id", help="Agent Pack ID")
    pack_activate.add_argument("--grant", action="append", help="Capability grant; repeat for multiple grants")
    pack_activate.add_argument("--mission-id", help="Optional mission activation boundary")
    pack_activate.add_argument("--org-admin-shortcut", action="store_true", help="Use an organization policy shortcut when allowed")
    pack_activate.add_argument("--policy-id", help="Organization policy record for shortcut activation")
    add_state_argument(pack_activate)
    add_scope_arguments(pack_activate)
    pack_activate.add_argument("--json", action="store_true", help="Emit JSON output")
    pack_activate.set_defaults(func=command_pack_activate)

    pack_certify = pack_sub.add_parser("certify", help="Generate an evidence-backed Agent Pack certification card")
    pack_certify.add_argument("pack_id", help="Agent Pack ID")
    add_state_argument(pack_certify)
    add_scope_arguments(pack_certify)
    pack_certify.add_argument("--json", action="store_true", help="Emit JSON output")
    pack_certify.set_defaults(func=command_pack_certify)

    pack_connector = pack_sub.add_parser("connector-request", help="Request a granted ConnectorHub-mediated pack capability")
    pack_connector.add_argument("pack_id", help="Agent Pack ID")
    pack_connector.add_argument("--capability", required=True, help="Granted connector/action capability")
    add_state_argument(pack_connector)
    add_scope_arguments(pack_connector)
    pack_connector.add_argument("--json", action="store_true", help="Emit JSON output")
    pack_connector.set_defaults(func=command_pack_connector_request)

    pack_playbook = pack_sub.add_parser("playbook", help="Experience-derived Agent Pack playbook updates")
    pack_playbook_sub = pack_playbook.add_subparsers(dest="pack_playbook_command")

    pack_playbook_propose = pack_playbook_sub.add_parser("propose", help="Propose a playbook update from a lesson")
    pack_playbook_propose.add_argument("--pack-id", required=True, help="Agent Pack ID")
    pack_playbook_propose.add_argument("--lesson-id", required=True, help="Experience lesson ID")
    add_state_argument(pack_playbook_propose)
    add_scope_arguments(pack_playbook_propose)
    pack_playbook_propose.add_argument("--json", action="store_true", help="Emit JSON output")
    pack_playbook_propose.set_defaults(func=command_pack_playbook_propose)

    pack_playbook_approve = pack_playbook_sub.add_parser("approve", help="Approve a scoped playbook proposal")
    pack_playbook_approve.add_argument("proposal_id", help="Playbook proposal ID")
    add_state_argument(pack_playbook_approve)
    add_scope_arguments(pack_playbook_approve)
    pack_playbook_approve.add_argument("--json", action="store_true", help="Emit JSON output")
    pack_playbook_approve.set_defaults(func=command_pack_playbook_approve)

    pack_update = pack_sub.add_parser("update", help="Preview or approve an Agent Pack version update")
    pack_update.add_argument("pack_id", help="Agent Pack ID")
    pack_update.add_argument("--to-version", required=True, help="Candidate pack version")
    pack_update.add_argument("--dry-run", action="store_true", help="Show diff and evaluation gate without applying")
    pack_update.add_argument("--approve", action="store_true", help="Approve and apply the candidate version")
    add_state_argument(pack_update)
    add_scope_arguments(pack_update)
    pack_update.add_argument("--json", action="store_true", help="Emit JSON output")
    pack_update.set_defaults(func=command_pack_update)

    pack_rollback = pack_sub.add_parser("rollback", help="Rollback an Agent Pack to a previous pinned version")
    pack_rollback.add_argument("pack_id", help="Agent Pack ID")
    pack_rollback.add_argument("--to-version", required=True, help="Pinned version to restore")
    pack_rollback.add_argument("--reason", required=True, help="Rollback reason")
    add_state_argument(pack_rollback)
    add_scope_arguments(pack_rollback)
    pack_rollback.add_argument("--json", action="store_true", help="Emit JSON output")
    pack_rollback.set_defaults(func=command_pack_rollback)

    pack_patch = pack_sub.add_parser("emergency-patch", help="Apply a policy-governed emergency security patch")
    pack_patch.add_argument("pack_id", help="Agent Pack ID")
    pack_patch.add_argument("--patch-version", required=True, help="Security patch version")
    pack_patch.add_argument("--behavior-change", action="store_true", help="Declare that the patch changes behavior and requires review")
    add_state_argument(pack_patch)
    add_scope_arguments(pack_patch)
    pack_patch.add_argument("--json", action="store_true", help="Emit JSON output")
    pack_patch.set_defaults(func=command_pack_emergency_patch)

    mission = subcommands.add_parser("mission", help="Mission Goal Contract commands")
    mission_sub = mission.add_subparsers(dest="mission_command")

    mission_create = mission_sub.add_parser("create", help="Create an editable Mission Goal Contract")
    mission_create.add_argument("--goal", required=True, help="Natural-language mission goal")
    mission_create.add_argument("--claim-id", help="Source claim ID")
    mission_create.add_argument("--evidence-bundle-id", help="Source Evidence Bundle ID")
    add_state_argument(mission_create)
    add_scope_arguments(mission_create)
    mission_create.add_argument("--json", action="store_true", help="Emit JSON output")
    mission_create.set_defaults(func=command_mission_create)

    mission_activate = mission_sub.add_parser("activate", help="Activate a Mission Goal Contract")
    mission_activate.add_argument("mission_id", help="Mission ID")
    mission_activate.add_argument("--mode", choices=["manual", "assist", "autopilot", "locked"], default="autopilot")
    add_state_argument(mission_activate)
    add_scope_arguments(mission_activate)
    mission_activate.add_argument("--json", action="store_true", help="Emit JSON output")
    mission_activate.set_defaults(func=command_mission_activate)

    mission_show = mission_sub.add_parser("show", help="Show a Mission Goal Contract")
    mission_show.add_argument("mission_id", help="Mission ID")
    add_state_argument(mission_show)
    add_scope_arguments(mission_show)
    mission_show.add_argument("--json", action="store_true", help="Emit JSON output")
    mission_show.set_defaults(func=command_mission_show)

    mission_autonomy = mission_sub.add_parser("autonomy-control", help="Pause, stop, revoke, or reduce mission autonomy")
    mission_autonomy.add_argument("mission_id", help="Mission ID")
    mission_autonomy.add_argument("--control", choices=["pause", "stop", "revoke", "reduce"], required=True)
    mission_autonomy.add_argument("--reason", required=True)
    add_state_argument(mission_autonomy)
    add_scope_arguments(mission_autonomy)
    mission_autonomy.add_argument("--json", action="store_true", help="Emit JSON output")
    mission_autonomy.set_defaults(func=command_mission_autonomy_control)

    mission_escalate = mission_sub.add_parser("escalate", help="Create a mission exception escalation card")
    mission_escalate.add_argument("mission_id", help="Mission ID")
    mission_escalate.add_argument(
        "--exception",
        choices=["missing_evidence", "policy_denial", "connector_failure", "model_disagreement", "unclear_goal", "high_risk_action"],
        required=True,
    )
    add_state_argument(mission_escalate)
    add_scope_arguments(mission_escalate)
    mission_escalate.add_argument("--json", action="store_true", help="Emit JSON output")
    mission_escalate.set_defaults(func=command_mission_escalate)

    mission_outcome = mission_sub.add_parser("outcome", help="Record a mission outcome evaluation")
    mission_outcome.add_argument("mission_id", help="Mission ID")
    mission_outcome.add_argument("--action-id", required=True, help="Executed Action ID")
    add_state_argument(mission_outcome)
    add_scope_arguments(mission_outcome)
    mission_outcome.add_argument("--json", action="store_true", help="Emit JSON output")
    mission_outcome.set_defaults(func=command_mission_outcome)

    mission_aar = mission_sub.add_parser("after-action-review", help="Generate a mission after-action review and scorecard")
    mission_aar.add_argument("mission_id", help="Mission ID")
    mission_aar.add_argument("--outcome-id", required=True, help="Mission outcome ID")
    add_state_argument(mission_aar)
    add_scope_arguments(mission_aar)
    mission_aar.add_argument("--json", action="store_true", help="Emit JSON output")
    mission_aar.set_defaults(func=command_mission_after_action)

    mission_audit_export = mission_sub.add_parser("audit-export", help="Export a mission audit-first report")
    mission_audit_export.add_argument("mission_id", help="Mission ID")
    add_state_argument(mission_audit_export)
    add_scope_arguments(mission_audit_export)
    mission_audit_export.add_argument("--json", action="store_true", help="Emit JSON output")
    mission_audit_export.set_defaults(func=command_mission_audit_export)

    action = subcommands.add_parser("action", help="Governed Workflow/Action commands")
    action_sub = action.add_subparsers(dest="action_command")

    action_propose = action_sub.add_parser("propose", help="Propose an Action Card and dry-run")
    action_propose.add_argument("--mission-id", required=True, help="Mission ID")
    action_propose.add_argument("--claim-id", required=True, help="Source claim ID")
    action_propose.add_argument("--goal", required=True, help="Action goal")
    action_propose.add_argument("--action-kind", choices=["internal_status_update", "draft_task", "refresh_brief", "external_writeback", "destructive_change"], default="internal_status_update")
    action_propose.add_argument("--risk", choices=["low", "medium", "high", "destructive", "sensitive"], default="low")
    action_propose.add_argument("--connector", default="mock_connector")
    action_propose.add_argument("--target", default="mock://local-target")
    action_propose.add_argument("--ontology-object-ref", action="append", default=[], help="Promoted ontology object ref used for local impact context")
    add_state_argument(action_propose)
    add_scope_arguments(action_propose)
    action_propose.add_argument("--json", action="store_true", help="Emit JSON output")
    action_propose.set_defaults(func=command_action_propose)

    action_approve = action_sub.add_parser("approve", help="Approve an Action Card")
    action_approve.add_argument("action_id", help="Action ID")
    action_approve.add_argument("--approver", default="owner")
    add_state_argument(action_approve)
    add_scope_arguments(action_approve)
    action_approve.add_argument("--json", action="store_true", help="Emit JSON output")
    action_approve.set_defaults(func=command_action_approve)

    action_execute = action_sub.add_parser("execute", help="Execute an approved or allowed Action Card")
    action_execute.add_argument("action_id", help="Action ID")
    add_state_argument(action_execute)
    add_scope_arguments(action_execute)
    action_execute.add_argument("--json", action="store_true", help="Emit JSON output")
    action_execute.set_defaults(func=command_action_execute)

    action_show = action_sub.add_parser("show", help="Show an Action Card")
    action_show.add_argument("action_id", help="Action ID")
    add_state_argument(action_show)
    add_scope_arguments(action_show)
    action_show.add_argument("--json", action="store_true", help="Emit JSON output")
    action_show.set_defaults(func=command_action_show)

    action_dry_run = action_sub.add_parser("dry-run", help="Show an Action Card dry-run record")
    action_dry_run.add_argument("action_id", help="Action ID")
    add_state_argument(action_dry_run)
    add_scope_arguments(action_dry_run)
    action_dry_run.add_argument("--json", action="store_true", help="Emit JSON output")
    action_dry_run.set_defaults(func=command_action_dry_run)

    action_idempotency = action_sub.add_parser("idempotency-test", help="Verify duplicate action requests are deduplicated")
    action_idempotency.add_argument("action_id", help="Action ID")
    add_state_argument(action_idempotency)
    add_scope_arguments(action_idempotency)
    action_idempotency.add_argument("--json", action="store_true", help="Emit JSON output")
    action_idempotency.set_defaults(func=command_action_idempotency_test)

    action_reversibility = action_sub.add_parser("reversibility-test", help="Review rollback, compensation, retry, or non-reversible action handling")
    action_reversibility.add_argument("action_id", help="Action ID")
    action_reversibility.add_argument("--mode", choices=["rollback", "compensation", "retry", "non_reversible"], required=True)
    add_state_argument(action_reversibility)
    add_scope_arguments(action_reversibility)
    action_reversibility.add_argument("--json", action="store_true", help="Emit JSON output")
    action_reversibility.set_defaults(func=command_action_reversibility_test)

    connector = subcommands.add_parser("connector", help="Connector boundary commands")
    connector_sub = connector.add_subparsers(dest="connector_command")

    direct_write = connector_sub.add_parser("direct-write-test", help="Deny direct provider writeback without a Workflow/Action path")
    direct_write.add_argument("--provider", default="mock_provider")
    direct_write.add_argument("--target", default="mock://connected-source")
    add_state_argument(direct_write)
    add_scope_arguments(direct_write)
    direct_write.add_argument("--json", action="store_true", help="Emit JSON output")
    direct_write.set_defaults(func=command_connector_direct_write_test)

    credential_boundary = connector_sub.add_parser("credential-boundary-test", help="Verify ConnectorHub credential custody without exposing secrets")
    credential_boundary.add_argument("--provider", default="mock_provider")
    credential_boundary.add_argument("--capability", default="mock.write_status")
    add_state_argument(credential_boundary)
    add_scope_arguments(credential_boundary)
    credential_boundary.add_argument("--json", action="store_true", help="Emit JSON output")
    credential_boundary.set_defaults(func=command_connector_credential_boundary_test)

    action_trace = connector_sub.add_parser("action-trace", help="Record ConnectorHub-mediated provider action trace")
    action_trace.add_argument("action_id", help="Action ID")
    add_state_argument(action_trace)
    add_scope_arguments(action_trace)
    action_trace.add_argument("--json", action="store_true", help="Emit JSON output")
    action_trace.set_defaults(func=command_connector_action_trace)

    security = subcommands.add_parser("security", help="Security and operations verification helpers")
    security_sub = security.add_subparsers(dest="security_command")

    sensitive_change = security_sub.add_parser("sensitive-change-test", help="Verify stop-and-ask gate for sensitive changes")
    sensitive_change.add_argument("--category", default="production_mutation")
    add_state_argument(sensitive_change)
    add_scope_arguments(sensitive_change)
    sensitive_change.add_argument("--json", action="store_true", help="Emit JSON output")
    sensitive_change.set_defaults(func=command_security_sensitive_change_test)

    vs2_h01_package = security_sub.add_parser(
        "vs2-h01-approval-package",
        help="Create a non-mutating VS2 H01 approval review package",
    )
    vs2_h01_package.add_argument(
        "--architecture-scope",
        default="Local VS2 security slice: Postgres RLS tenant isolation, OPA/Rego policy control plane, and default egress deny under the frozen scenario contract.",
    )
    vs2_h01_package.add_argument(
        "--dependency-decision",
        default="pending owner decision for local PostgreSQL, OPA/Rego, and controlled local egress harness dependencies",
    )
    vs2_h01_package.add_argument(
        "--migration-scope",
        default="non-production local compatibility path only; no production or destructive migration is approved by this package",
    )
    vs2_h01_package.add_argument("--rollback-owner", default="JiYong/Tars or explicitly delegated owner")
    vs2_h01_package.add_argument("--security-owner", default="JiYong/Tars or explicitly delegated security owner")
    vs2_h01_package.add_argument(
        "--local-boundary",
        default="local/on-prem deterministic proof only; no production, live-provider, penetration-test, or human-accepted claim",
    )
    vs2_h01_package.add_argument("--exception", action="append", default=[], help="Requested exception to include in the human review package")
    add_state_argument(vs2_h01_package)
    add_scope_arguments(vs2_h01_package)
    vs2_h01_package.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_h01_package.set_defaults(func=command_security_vs2_h01_approval_package)

    vs2_local_proof = security_sub.add_parser(
        "vs2-local-proof",
        help="Run local deterministic VS2 policy, tenant-isolation, and egress proof",
    )
    vs2_local_proof.add_argument(
        "--reuse-local-range-report",
        help="Reuse a current-source vs2-local-range report instead of rerunning the local range.",
    )
    vs2_local_proof.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_local_proof.set_defaults(func=command_security_vs2_local_proof)

    vs2_local_range = security_sub.add_parser(
        "vs2-local-range",
        help="Run the first production-flow VS2 local range through real Postgres, OPA, API, browser, CLI, and audit evidence",
    )
    vs2_local_range.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_local_range.set_defaults(func=command_security_vs2_local_range)

    vs2_range_client = security_sub.add_parser(
        "vs2-range-client",
        help="Call a running VS2 local range gateway as the native CLI surface",
    )
    vs2_range_client.add_argument("--api-url", required=True)
    vs2_range_client.add_argument("--token", required=True)
    vs2_range_client.add_argument("--artifact-id", required=True)
    vs2_range_client.add_argument("--forged-tenant-id")
    vs2_range_client.add_argument("--forged-role")
    vs2_range_client.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_range_client.set_defaults(func=command_security_vs2_range_client)

    vs2_range_action_client = security_sub.add_parser(
        "vs2-range-action-client",
        help="Call a running VS2 local range gateway as the native CLI external-action surface",
    )
    vs2_range_action_client.add_argument("--api-url", required=True)
    vs2_range_action_client.add_argument("--token", required=True)
    vs2_range_action_client.add_argument("--provider-url", required=True)
    vs2_range_action_client.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_range_action_client.set_defaults(func=command_security_vs2_range_action_client)

    vs2_range_object_contract_client = security_sub.add_parser(
        "vs2-range-object-contract-client",
        help="Call a running VS2 local range gateway for durable object contract evidence",
    )
    vs2_range_object_contract_client.add_argument("--api-url", required=True)
    vs2_range_object_contract_client.add_argument("--token", required=True)
    vs2_range_object_contract_client.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_range_object_contract_client.set_defaults(func=command_security_vs2_range_object_contract_client)

    vs2_range_object_access_client = security_sub.add_parser(
        "vs2-range-object-access-client",
        help="Call a running VS2 local range gateway for guessed object/download/evidence traversal denial evidence",
    )
    vs2_range_object_access_client.add_argument("--api-url", required=True)
    vs2_range_object_access_client.add_argument("--token", required=True)
    vs2_range_object_access_client.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_range_object_access_client.set_defaults(func=command_security_vs2_range_object_access_client)

    vs2_range_observability_client = security_sub.add_parser(
        "vs2-range-observability-client",
        help="Call a running VS2 local range gateway for tenant-scoped observability/export evidence",
    )
    vs2_range_observability_client.add_argument("--api-url", required=True)
    vs2_range_observability_client.add_argument("--token", required=True)
    vs2_range_observability_client.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_range_observability_client.set_defaults(func=command_security_vs2_range_observability_client)

    vs2_range_tenant_read_client = security_sub.add_parser(
        "vs2-range-tenant-read-client",
        help="Call a running VS2 local range gateway for tenant read-matrix evidence",
    )
    vs2_range_tenant_read_client.add_argument("--api-url", required=True)
    vs2_range_tenant_read_client.add_argument("--token", required=True)
    vs2_range_tenant_read_client.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_range_tenant_read_client.set_defaults(func=command_security_vs2_range_tenant_read_client)

    vs2_range_search_client = security_sub.add_parser(
        "vs2-range-search-client",
        help="Call a running VS2 local range gateway for tenant search/snapshot isolation evidence",
    )
    vs2_range_search_client.add_argument("--api-url", required=True)
    vs2_range_search_client.add_argument("--token", required=True)
    vs2_range_search_client.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_range_search_client.set_defaults(func=command_security_vs2_range_search_client)

    vs2_range_db_path_client = security_sub.add_parser(
        "vs2-range-db-path-client",
        help="Call a running VS2 local range gateway for DB view/function/raw-SQL path evidence",
    )
    vs2_range_db_path_client.add_argument("--api-url", required=True)
    vs2_range_db_path_client.add_argument("--token", required=True)
    vs2_range_db_path_client.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_range_db_path_client.set_defaults(func=command_security_vs2_range_db_path_client)

    vs2_range_constraint_collision_client = security_sub.add_parser(
        "vs2-range-constraint-collision-client",
        help="Call a running VS2 local range gateway for tenant-scoped unique/FK collision evidence",
    )
    vs2_range_constraint_collision_client.add_argument("--api-url", required=True)
    vs2_range_constraint_collision_client.add_argument("--token", required=True)
    vs2_range_constraint_collision_client.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_range_constraint_collision_client.set_defaults(func=command_security_vs2_range_constraint_collision_client)

    vs2_range_migration_client = security_sub.add_parser(
        "vs2-range-migration-client",
        help="Call a running VS2 local range gateway for migration/quarantine evidence",
    )
    vs2_range_migration_client.add_argument("--api-url", required=True)
    vs2_range_migration_client.add_argument("--token", required=True)
    vs2_range_migration_client.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_range_migration_client.set_defaults(func=command_security_vs2_range_migration_client)

    vs2_range_upgrade_path_client = security_sub.add_parser(
        "vs2-range-upgrade-path-client",
        help="Call a running VS2 local range gateway for upgrade-path migration/rollback evidence",
    )
    vs2_range_upgrade_path_client.add_argument("--api-url", required=True)
    vs2_range_upgrade_path_client.add_argument("--token", required=True)
    vs2_range_upgrade_path_client.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_range_upgrade_path_client.set_defaults(func=command_security_vs2_range_upgrade_path_client)

    vs2_range_audit_integrity_client = security_sub.add_parser(
        "vs2-range-audit-integrity-client",
        help="Call a running VS2 local range gateway for audit hash-chain and tamper evidence",
    )
    vs2_range_audit_integrity_client.add_argument("--api-url", required=True)
    vs2_range_audit_integrity_client.add_argument("--token", required=True)
    vs2_range_audit_integrity_client.add_argument("--json", action="store_true", help="Emit JSON output")
    vs2_range_audit_integrity_client.set_defaults(func=command_security_vs2_range_audit_integrity_client)

    backup_restore = security_sub.add_parser("backup-restore-test", help="Run a deterministic backup/restore rehearsal over local records")
    backup_restore.add_argument("--subject-ref", action="append", default=[], help="Evidence or product refs expected in the backup")
    add_state_argument(backup_restore)
    add_scope_arguments(backup_restore)
    backup_restore.add_argument("--json", action="store_true", help="Emit JSON output")
    backup_restore.set_defaults(func=command_security_backup_restore_test)

    helpful_failure = security_sub.add_parser("helpful-failure-test", help="Verify helpful failure records for major failure classes")
    add_state_argument(helpful_failure)
    add_scope_arguments(helpful_failure)
    helpful_failure.add_argument("--json", action="store_true", help="Emit JSON output")
    helpful_failure.set_defaults(func=command_security_helpful_failure_test)

    retention_explain = security_sub.add_parser("retention-explain", help="Explain retention/deletion outcomes as a dry-run")
    retention_explain.add_argument("--resource-type", choices=["conversation", "memory", "artifact", "mission", "action", "workspace"], default="workspace")
    add_state_argument(retention_explain)
    add_scope_arguments(retention_explain)
    retention_explain.add_argument("--json", action="store_true", help="Emit JSON output")
    retention_explain.set_defaults(func=command_security_retention_explain)

    operator_status = security_sub.add_parser("operator-status", help="Show deterministic operator status signals")
    add_state_argument(operator_status)
    add_scope_arguments(operator_status)
    operator_status.add_argument("--json", action="store_true", help="Emit JSON output")
    operator_status.set_defaults(func=command_security_operator_status)

    release = subcommands.add_parser("release", help="Release verification helpers")
    release_sub = release.add_subparsers(dest="release_command")

    report_check = release_sub.add_parser("report-check", help="Validate a scenario-backed verification report")
    report_check.add_argument("--scenario-report", required=True, help="Path to scenario JSON report")
    report_check.add_argument("--verification-report", required=True, help="Path to markdown verification report")
    add_state_argument(report_check)
    add_scope_arguments(report_check)
    report_check.add_argument("--json", action="store_true", help="Emit JSON output")
    report_check.set_defaults(func=command_release_report_check)

    evidence = release_sub.add_parser("evidence", help="Release evidence package helpers")
    evidence_sub = evidence.add_subparsers(dest="release_evidence_command")

    collect = evidence_sub.add_parser("collect", help="Collect a release-facing evidence package")
    collect.add_argument("--scope", default="vs0-runtime-acceptance", help="Evidence package scope")
    collect.add_argument("--scenario-report", help="Acceptance scenario report path")
    collect.add_argument("--product-runtime-report", help="VS0 product runtime scenario report path")
    collect.add_argument("--browser-proof-dir", help="Browser proof directory")
    collect.add_argument("--output-dir", help="Evidence package output directory")
    collect.add_argument("--verification-report", help="Optional implementation verification report")
    add_state_argument(collect)
    add_scope_arguments(collect)
    collect.add_argument("--json", action="store_true", help="Emit JSON output")
    collect.set_defaults(func=command_release_evidence_collect)

    finalize = evidence_sub.add_parser("finalize", help="Finalize release evidence with a post-commit rollup")
    finalize.add_argument("--scope", default="vs0-runtime-acceptance", help="Evidence package scope")
    finalize.add_argument("--output-dir", help="Evidence package output directory")
    add_state_argument(finalize)
    add_scope_arguments(finalize)
    finalize.add_argument("--json", action="store_true", help="Emit JSON output")
    finalize.set_defaults(func=command_release_evidence_finalize)

    conversation = subcommands.add_parser("conversation", help="Conversation-first work surface commands")
    conversation_sub = conversation.add_subparsers(dest="conversation_command")

    conversation_start = conversation_sub.add_parser("start", help="Start from a natural-language message")
    conversation_start.add_argument("--message", required=True, help="User message or pasted messy input")
    add_state_argument(conversation_start)
    add_scope_arguments(conversation_start)
    conversation_start.add_argument("--json", action="store_true", help="Emit JSON output")
    conversation_start.set_defaults(func=command_conversation_start)

    conversation_promote = conversation_sub.add_parser("promote", help="Promote a selected conversation output")
    conversation_promote.add_argument("conversation_id", help="Conversation ID")
    conversation_promote.add_argument("--kind", choices=["claim"], default="claim")
    conversation_promote.add_argument("--statement", required=True, help="Promoted claim statement")
    conversation_promote.add_argument("--evidence-bundle-id", required=True, help="Evidence Bundle ID")
    add_state_argument(conversation_promote)
    add_scope_arguments(conversation_promote)
    conversation_promote.add_argument("--json", action="store_true", help="Emit JSON output")
    conversation_promote.set_defaults(func=command_conversation_promote)

    conversation_answer = conversation_sub.add_parser("answer", help="Answer a question from workspace evidence")
    conversation_answer.add_argument("conversation_id", help="Conversation ID")
    conversation_answer.add_argument("--question", required=True, help="Question to answer")
    add_state_argument(conversation_answer)
    add_scope_arguments(conversation_answer)
    conversation_answer.add_argument("--json", action="store_true", help="Emit JSON output")
    conversation_answer.set_defaults(func=command_conversation_answer)

    scenario = subcommands.add_parser("scenario", help="Scenario registry and verification commands")
    scenario_sub = scenario.add_subparsers(dest="scenario_command")

    scenario_list = scenario_sub.add_parser("list", help="List frozen scenarios")
    scenario_list.add_argument("--set", choices=["full", "vs0", "vs2", "vs2-policy-tenancy-egress"], default="full")
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
    verify.add_argument(
        "--reuse-vs2-local-proof-report",
        help="Reuse a current-source VS2 local proof report for vs2-policy-tenancy-egress instead of rerunning it.",
    )
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

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
    verify_full_claim_collaboration,
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
    result = store.create_memory_from_evidence_bundle(args.evidence_bundle_id, args.statement, requested_scope)
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
    elif args.contract == "full-claim-collaboration":
        report = verify_full_claim_collaboration(root)
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
                    "full-claim-collaboration",
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

    product = subcommands.add_parser("product", help="Product identity walkthrough commands")
    product_sub = product.add_subparsers(dest="product_command")

    walkthrough = product_sub.add_parser("walkthrough", help="Show the first-run product walkthrough")
    walkthrough.add_argument("--json", action="store_true", help="Emit JSON output")
    walkthrough.set_defaults(func=command_product_walkthrough)

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

    memory = subcommands.add_parser("memory", help="Durable owner-approved memory commands")
    memory_sub = memory.add_subparsers(dest="memory_command")

    memory_create = memory_sub.add_parser("create", help="Create owner-approved memory from an Evidence Bundle")
    memory_create.add_argument("--evidence-bundle-id", required=True, help="Evidence Bundle ID")
    memory_create.add_argument("--statement", required=True, help="Memory statement")
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

    namespace = subcommands.add_parser("namespace", help="Namespace promotion commands")
    namespace_sub = namespace.add_subparsers(dest="namespace_command")

    namespace_promote = namespace_sub.add_parser("promote", help="Explicitly promote a scoped item with provenance")
    namespace_promote.add_argument("--source-kind", choices=["memory"], default="memory")
    namespace_promote.add_argument("--source-id", required=True, help="Source item ID")
    namespace_promote.add_argument("--target-tenant-id", help="Target tenant; defaults to source tenant")
    namespace_promote.add_argument("--target-owner-id", required=True, help="Target owner")
    namespace_promote.add_argument("--target-namespace-id", required=True, help="Target namespace")
    namespace_promote.add_argument("--target-workspace-id", required=True, help="Target workspace")
    namespace_promote.add_argument("--mode", choices=["copy_with_provenance"], default="copy_with_provenance")
    namespace_promote.add_argument("--principal-id", default="local-user", help="Actor requesting promotion")
    namespace_promote.add_argument("--principal-role", choices=["org_admin", "org_approver"], default="org_admin")
    add_state_argument(namespace_promote)
    add_scope_arguments(namespace_promote)
    namespace_promote.add_argument("--json", action="store_true", help="Emit JSON output")
    namespace_promote.set_defaults(func=command_namespace_promote)

    access = subcommands.add_parser("access", help="Local deterministic access-control commands")
    access_sub = access.add_subparsers(dest="access_command")

    access_evaluate = access_sub.add_parser("evaluate", help="Evaluate local RBAC/ABAC policy without external calls")
    access_evaluate.add_argument("--principal-id", default="local-user", help="Principal identifier")
    access_evaluate.add_argument("--principal-role", choices=["personal_user", "org_member", "org_approver", "org_admin"], default="personal_user")
    access_evaluate.add_argument("--principal-attributes", default="", help="Comma-separated principal attributes")
    access_evaluate.add_argument("--action", choices=["read", "write", "promote", "approve", "execute", "configure"], required=True)
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

    connector = subcommands.add_parser("connector", help="Connector boundary commands")
    connector_sub = connector.add_subparsers(dest="connector_command")

    direct_write = connector_sub.add_parser("direct-write-test", help="Deny direct provider writeback without a Workflow/Action path")
    direct_write.add_argument("--provider", default="mock_provider")
    direct_write.add_argument("--target", default="mock://connected-source")
    add_state_argument(direct_write)
    add_scope_arguments(direct_write)
    direct_write.add_argument("--json", action="store_true", help="Emit JSON output")
    direct_write.set_defaults(func=command_connector_direct_write_test)

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

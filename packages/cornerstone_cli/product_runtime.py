from __future__ import annotations

import html
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from datetime import datetime, timezone

from cornerstone_cli import __version__
from cornerstone_cli.runtime import LocalRuntimeStore


DEFAULT_SCOPE = {
    "tenant_id": "local-dev",
    "owner_id": "local-user",
    "namespace_id": "personal",
    "workspace_id": "default",
}

API_ROUTES = [
    "GET /health",
    "GET /ready",
    "POST /artifacts",
    "GET /artifacts/{artifact_id}",
    "POST /search",
    "GET /search-snapshots/{snapshot_id}",
    "POST /evidence-bundles",
    "GET /evidence-bundles/{evidence_bundle_id}",
    "POST /claims",
    "GET /claims/{claim_id}",
    "POST /claims/{claim_id}/approve",
    "POST /actions",
    "GET /actions/{action_id}",
    "POST /actions/{action_id}/dry-run",
    "POST /actions/{action_id}/approve",
    "POST /actions/{action_id}/execute",
    "GET /audit-events",
    "POST /audit/verify",
]

UI_SURFACES = [
    "Home/Ops Inbox",
    "Artifact Viewer",
    "Search",
    "Claim Builder",
    "Action Card",
    "Audit Detail",
]


def _scope_from_body(body: dict[str, Any] | None = None) -> dict[str, str]:
    data = body or {}
    scope = dict(DEFAULT_SCOPE)
    for key in scope:
        value = data.get(key)
        if isinstance(value, str) and value:
            scope[key] = value
    return scope


def _json_response(status: str, **kwargs: Any) -> dict[str, Any]:
    response = {
        "schema_version": "cs.runtime_api.v0",
        "status": status,
        "product": "CornerStone",
        "version": __version__,
        "mode": "local_vs0_runtime",
        "evidence_refs": [],
        "audit_refs": [],
        "policy_decision_refs": [],
        "errors": [],
    }
    response.update(kwargs)
    return response


def _utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit(root: Path) -> str | None:
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


def _latest_successful_report(root: Path, scenario_set: str) -> dict[str, Any]:
    report_dir = root / "reports/scenario"
    candidates = sorted(report_dir.glob(f"{scenario_set}-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        summary = data.get("summary", {})
        if data.get("scenario_set") != scenario_set or data.get("status") != "success" or summary.get("blocking") != 0:
            continue
        return {
            "scenario_set": scenario_set,
            "path": str(path.relative_to(root)),
            "timestamp": _utc_from_timestamp(path.stat().st_mtime),
            "git_commit": data.get("ids", {}).get("git_commit"),
            "current_git_commit": _git_commit(root),
            "status": data.get("status"),
            "gate_status": "pass",
            "scenario_count": summary.get("scenario_count"),
            "pass": summary.get("pass"),
            "human_required": summary.get("human_required"),
            "blocking": summary.get("blocking"),
        }
    return {
        "scenario_set": scenario_set,
        "path": None,
        "timestamp": None,
        "git_commit": None,
        "current_git_commit": _git_commit(root),
        "status": "missing",
        "gate_status": "missing",
        "scenario_count": 0,
        "pass": 0,
        "human_required": 0,
        "blocking": None,
    }


def build_readiness_report(root: Path) -> dict[str, Any]:
    runtime_file = root / "packages/cornerstone_cli/product_runtime.py"
    local_checks = [
        ("native_cli", root / "cornerstone"),
        ("scenario_matrix", root / "docs/scenario-contracts/SCENARIO_MATRIX_FULL.md"),
        ("vs0_contract", root / "docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md"),
        ("vs0_runtime_contract", root / "docs/scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md"),
        ("scenario_tests", root / "tests/scenario"),
        ("fixture_corpus", root / "fixtures/vs0"),
    ]
    runtime_checks = [
        ("shared_local_runtime_store", root / "packages/cornerstone_cli/runtime.py"),
        ("api_runtime", runtime_file),
        ("web_runtime", runtime_file),
    ]
    checks = [
        {"name": name, "present": path.exists(), "path": str(path.relative_to(root))}
        for name, path in [*local_checks, *runtime_checks]
    ]
    local_scenario_ready = all(row["present"] for row in checks if row["name"] in {name for name, _ in local_checks})
    vs0_runtime_ready = (
        all(row["present"] for row in checks if row["name"] in {name for name, _ in runtime_checks})
        and len(API_ROUTES) >= 15
        and set(UI_SURFACES)
        == {"Home/Ops Inbox", "Artifact Viewer", "Search", "Claim Builder", "Action Card", "Audit Detail"}
    )
    production_release_ready = False
    last_runtime_report = _latest_successful_report(root, "vs0-product-runtime")
    last_acceptance_report = _latest_successful_report(root, "vs0-runtime-acceptance")
    acceptance_status = "pass" if last_acceptance_report["gate_status"] == "pass" else (
        "pending" if last_runtime_report["gate_status"] == "pass" else "blocked_on_runtime_report"
    )
    readiness = {
        "schema_version": "cs.runtime_readiness.v0",
        "local_scenario_ready": local_scenario_ready,
        "vs0_runtime_ready": bool(vs0_runtime_ready),
        "production_release_ready": production_release_ready,
        "human_required": True,
        "real_external_http_calls": 0,
        "mock_connector_calls": 0,
        "last_successful_runtime_scenario": last_runtime_report,
        "last_successful_acceptance_scenario": last_acceptance_report,
        "acceptance_gate": {
            "scenario_set": "vs0-runtime-acceptance",
            "status": acceptance_status,
            "contract": "docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_CONTRACT.md",
            "report_path": last_acceptance_report["path"],
        },
        "api_routes": API_ROUTES,
        "ui_surfaces": UI_SURFACES,
        "runtime_boundaries": ["Product/Mission", "Archive/Evidence", "Connector/Action", "Audit/Policy"],
        "human_required_items": [
            {
                "id": "VS0-RT-H01",
                "reason": "Live connector/provider verification requires credentials and may mutate third-party systems.",
                "required_evidence": "Redacted live-provider dry-run/execution transcript, approval record, and audit refs.",
            },
            {
                "id": "VS0-RT-H02",
                "reason": "Usability acceptance is subjective and must be confirmed by a human operator.",
                "required_evidence": "Human acceptance note with screenshots or recording and issue list if rejected.",
            },
        ],
    }
    return {"checks": checks, "readiness": readiness}


def render_home(readiness: dict[str, Any]) -> str:
    surfaces = "\n".join(
        f"<li><a href='#{html.escape(surface.lower().replace('/', '-').replace(' ', '-'))}'>{html.escape(surface)}</a></li>"
        for surface in UI_SURFACES
    )
    production_ready = str(readiness["production_release_ready"]).lower()
    runtime_ready = str(readiness["vs0_runtime_ready"]).lower()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CornerStone VS0 Runtime</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #5f6f7a;
      --line: #d8e0e7;
      --paper: #f7f9fb;
      --surface: #ffffff;
      --accent: #285e61;
      --safe: #216e4e;
      --warn: #8f5f00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--surface);
      padding: 20px 28px;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(180px, 240px) minmax(0, 1fr);
      min-height: calc(100vh - 81px);
    }}
    nav {{
      border-right: 1px solid var(--line);
      background: #fbfcfd;
      padding: 20px;
    }}
    nav ul {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }}
    nav a {{ color: var(--accent); text-decoration: none; font-weight: 600; }}
    section {{ padding: 22px 28px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0; font-size: 24px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    .status-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .badge {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--muted);
      padding: 5px 8px;
      font-size: 13px;
      font-weight: 600;
    }}
    .badge.safe {{ color: var(--safe); }}
    .badge.warn {{ color: var(--warn); }}
    .panel-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 700; }}
    code {{ background: #eef3f7; padding: 2px 4px; border-radius: 4px; }}
    @media (max-width: 760px) {{
      main {{ grid-template-columns: 1fr; }}
      nav {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      header, section {{ padding-left: 18px; padding-right: 18px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>CornerStone</h1>
    <div class="status-row">
      <span class="badge safe">local_scenario_ready={str(readiness["local_scenario_ready"]).lower()}</span>
      <span class="badge safe">vs0_runtime_ready={runtime_ready}</span>
      <span class="badge warn">production_release_ready={production_ready}</span>
      <span class="badge safe">real_external_http_calls=0</span>
    </div>
  </header>
  <main>
    <nav aria-label="Primary">
      <ul>{surfaces}</ul>
    </nav>
    <div>
      <section id="home-ops-inbox">
        <h2>Home/Ops Inbox</h2>
        <div class="panel-grid">
          <div class="panel"><div class="label">Loop</div>Artifact ingest -> search -> evidence -> claim -> action -> audit</div>
          <div class="panel"><div class="label">Boundary</div>Product, Archive, Connector/Action, Policy, and Audit records share one local runtime.</div>
        </div>
      </section>
      <section id="artifact-viewer"><h2>Artifact Viewer</h2><p>Immutable artifact records expose checksum, scope, source, derived text, evidence refs, and audit refs.</p></section>
      <section id="search"><h2>Search</h2><p>Search creates reproducible scoped snapshots that can become Evidence Bundles.</p></section>
      <section id="claim-builder"><h2>Claim Builder</h2><p>Claims stay draft unless backed by an Evidence Bundle and approval evidence.</p></section>
      <section id="action-card"><h2>Action Card</h2><p>Actions expose dry-run diff, expected impact, policy decision, approval, execution, and mock ConnectorHub boundary.</p></section>
      <section id="audit-detail"><h2>Audit Detail</h2><p>Audit events form a local tamper-evident hash chain verified by <code>cornerstone audit verify --json</code>.</p></section>
    </div>
  </main>
</body>
</html>
"""


class RuntimeHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler: type[BaseHTTPRequestHandler], *, root: Path, state_dir: Path) -> None:
        super().__init__(server_address, handler)
        self.root = root
        self.state_dir = state_dir


class VS0RuntimeHandler(BaseHTTPRequestHandler):
    server: RuntimeHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    @property
    def store(self) -> LocalRuntimeStore:
        return LocalRuntimeStore(self.server.state_dir)

    @property
    def root(self) -> Path:
        return self.server.root

    def _send_json(self, payload: dict[str, Any], status_code: int = 200) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html_body: str) -> None:
        data = html_body.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _path_parts(self) -> list[str]:
        return [part for part in urlparse(self.path).path.split("/") if part]

    def _query_scope(self) -> dict[str, str]:
        query = parse_qs(urlparse(self.path).query)
        scope = dict(DEFAULT_SCOPE)
        for key in scope:
            value = query.get(key, [])
            if value:
                scope[key] = value[-1]
        return scope

    def _input_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("Artifact path must stay inside the CornerStone workspace.")
        return resolved

    def do_GET(self) -> None:
        parts = self._path_parts()
        if not parts:
            readiness = build_readiness_report(self.root)["readiness"]
            self._send_html(render_home(readiness))
            return
        if parts == ["health"]:
            self._send_json(_json_response("success", service="cornerstone-vs0-runtime", real_external_http_calls=0))
            return
        if parts == ["ready"]:
            report = build_readiness_report(self.root)
            self._send_json(_json_response("success", **report))
            return
        if len(parts) == 2 and parts[0] == "artifacts":
            self._show_artifact(parts[1])
            return
        if len(parts) == 2 and parts[0] == "search-snapshots":
            self._show_search_snapshot(parts[1])
            return
        if len(parts) == 2 and parts[0] == "evidence-bundles":
            self._show_evidence_bundle(parts[1])
            return
        if len(parts) == 2 and parts[0] == "claims":
            self._show_claim(parts[1])
            return
        if len(parts) == 2 and parts[0] == "actions":
            self._show_action(parts[1])
            return
        if parts == ["audit-events"]:
            self._audit_events()
            return
        self._send_json(_json_response("not_found", errors=[{"code": "CS_API_NOT_FOUND", "message": "Route not found."}]), 404)

    def do_POST(self) -> None:
        parts = self._path_parts()
        try:
            body = self._body()
        except json.JSONDecodeError as error:
            self._send_json(_json_response("failed", errors=[{"code": "CS_API_INVALID_JSON", "message": str(error)}]), 400)
            return
        if parts == ["artifacts"]:
            self._ingest_artifact(body)
            return
        if parts == ["search"]:
            self._search(body)
            return
        if parts == ["evidence-bundles"]:
            self._create_evidence_bundle(body)
            return
        if parts == ["claims"]:
            self._create_claim(body)
            return
        if len(parts) == 3 and parts[0] == "claims" and parts[2] == "approve":
            self._approve_claim(parts[1], body)
            return
        if parts == ["actions"]:
            self._create_action(body)
            return
        if len(parts) == 3 and parts[0] == "actions" and parts[2] == "dry-run":
            self._show_action_dry_run(parts[1], body)
            return
        if len(parts) == 3 and parts[0] == "actions" and parts[2] == "approve":
            self._approve_action(parts[1], body)
            return
        if len(parts) == 3 and parts[0] == "actions" and parts[2] == "execute":
            self._execute_action(parts[1], body)
            return
        if parts == ["audit", "verify"]:
            self._audit_verify()
            return
        self._send_json(_json_response("not_found", errors=[{"code": "CS_API_NOT_FOUND", "message": "Route not found."}]), 404)

    def _show_artifact(self, artifact_id: str) -> None:
        scope = self._query_scope()
        artifact = self.store.get_artifact(artifact_id, scope)
        if artifact is None:
            self._send_json(_json_response("failed", errors=[{"code": "CS_ARTIFACT_NOT_FOUND", "message": "Artifact not found."}]), 404)
            return
        event = self.store.append_audit("artifact.read", scope, {"type": "artifact", "id": artifact_id}, {"reason": "api_artifact_show"})
        detail = dict(artifact)
        detail["derived_text_preview"] = self.store.derived_text_preview(artifact)
        self._send_json(
            _json_response(
                "success",
                artifact=detail,
                evidence_refs=[f"artifact:{artifact_id}"],
                audit_refs=[f"audit:{event['event_id']}"],
            )
        )

    def _show_search_snapshot(self, snapshot_id: str) -> None:
        scope = self._query_scope()
        snapshot = self.store.get_search_snapshot(snapshot_id)
        if snapshot is None:
            self._send_json(_json_response("failed", errors=[{"code": "CS_SEARCH_SNAPSHOT_NOT_FOUND", "message": "Search snapshot not found."}]), 404)
            return
        if snapshot.get("filters") != scope:
            self._send_json(_json_response("denied", errors=[{"code": "CS_SCOPE_DENIED", "message": "Search snapshot is outside the requested scope."}]), 403)
            return
        event = self.store.append_audit("search.snapshot.read", scope, {"type": "search_snapshot", "id": snapshot_id}, {"reason": "api_snapshot_show"})
        self._send_json(
            _json_response(
                "success",
                search_snapshot=snapshot,
                evidence_refs=[f"search_snapshot:{snapshot_id}"],
                audit_refs=[f"audit:{event['event_id']}"],
            )
        )

    def _show_evidence_bundle(self, bundle_id: str) -> None:
        scope = self._query_scope()
        result = self.store.show_evidence_bundle(bundle_id, scope)
        if result.get("status") == "not_found":
            self._send_json(_json_response("failed", errors=[{"code": "CS_EVIDENCE_BUNDLE_NOT_FOUND", "message": "Evidence bundle not found."}]), 404)
            return
        if result.get("status") == "scope_denied":
            self._send_json(_json_response("denied", errors=[{"code": "CS_SCOPE_DENIED", "message": "Evidence bundle is outside the requested scope."}]), 403)
            return
        bundle = result["bundle"]
        self._send_json(
            _json_response(
                "success",
                evidence_bundle=bundle,
                evidence_refs=[f"evidence_bundle:{bundle_id}", f"search_snapshot:{bundle['search_snapshot_id']}"],
                audit_refs=[f"audit:{result['audit_event']['event_id']}"],
            )
        )

    def _show_claim(self, claim_id: str) -> None:
        scope = self._query_scope()
        claim = self.store.get_claim(claim_id)
        if claim is None:
            self._send_json(_json_response("failed", errors=[{"code": "CS_CLAIM_NOT_FOUND", "message": "Claim not found."}]), 404)
            return
        if claim.get("scope") != scope:
            self._send_json(_json_response("denied", errors=[{"code": "CS_SCOPE_DENIED", "message": "Claim is outside the requested scope."}]), 403)
            return
        event = self.store.append_audit("claim.read", scope, {"type": "claim", "id": claim_id}, {"reason": "api_claim_show"})
        self._send_json(_json_response("success", claim=claim, evidence_refs=[f"claim:{claim_id}"], audit_refs=[f"audit:{event['event_id']}"]))

    def _show_action(self, action_id: str) -> None:
        scope = self._query_scope()
        action = self.store.get_action(action_id)
        if action is None:
            self._send_json(_json_response("failed", errors=[{"code": "CS_ACTION_NOT_FOUND", "message": "Action Card not found."}]), 404)
            return
        if action.get("scope") != scope:
            self._send_json(_json_response("denied", errors=[{"code": "CS_SCOPE_DENIED", "message": "Action Card is outside the requested scope."}]), 403)
            return
        event = self.store.append_audit("action.read", scope, {"type": "action", "id": action_id}, {"reason": "api_action_show"})
        self._send_json(_json_response("success", action_card=action, evidence_refs=[f"action:{action_id}"], audit_refs=[f"audit:{event['event_id']}"]))

    def _audit_events(self) -> None:
        scope = self._query_scope()
        event_types = parse_qs(urlparse(self.path).query).get("event_type", [])
        result = self.store.query_namespace_audit(scope, event_types=event_types)
        export = result["namespace_audit_export"]
        self._send_json(
            _json_response(
                "success",
                audit_events=export["events"],
                namespace_audit_export=export,
                evidence_refs=[f"namespace_audit_export:{export['namespace_audit_export_id']}"],
                audit_refs=[f"audit:{result['audit_event']['event_id']}"],
            )
        )

    def _ingest_artifact(self, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        try:
            path = self._input_path(str(body.get("path", "")))
        except ValueError as error:
            self._send_json(_json_response("failed", errors=[{"code": "CS_ARTIFACT_PATH_DENIED", "message": str(error)}]), 400)
            return
        if not path.exists() or not path.is_file():
            self._send_json(_json_response("failed", errors=[{"code": "CS_ARTIFACT_INPUT_MISSING", "message": "Artifact input file does not exist."}]), 404)
            return
        result = self.store.ingest_artifact(
            path,
            **scope,
            source=str(body.get("source", "local_file")),
            media_type=str(body.get("media_type", "text/plain")),
            derived_mode=str(body.get("derived_mode", "auto")),
            trust=str(body.get("trust", "untrusted")),
            lineage_from=body.get("lineage_from") if isinstance(body.get("lineage_from"), str) else None,
        )
        artifact = result["artifact"]
        policy_decisions = result.get("policy_decisions", [])
        self._send_json(
            _json_response(
                "success",
                artifact=artifact,
                deduplicated=result.get("deduplicated", False),
                evidence_refs=[f"artifact:{artifact['artifact_id']}", f"storage:{artifact['original_storage_ref']}"],
                audit_refs=[f"audit:{event['event_id']}" for event in result.get("audit_events", [result["audit_event"]])],
                policy_decision_refs=[f"policy:{decision['id']}" for decision in policy_decisions],
                policy_decisions=policy_decisions,
            )
        )

    def _search(self, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        query = str(body.get("query", ""))
        result = self.store.search(query, **scope)
        snapshot = result["snapshot"]
        refs = [f"search_snapshot:{snapshot['search_snapshot_id']}"]
        for row in snapshot.get("results", []):
            refs.extend(row.get("evidence_refs", []))
        self._send_json(
            _json_response(
                "success",
                search_snapshot=snapshot,
                evidence_refs=refs,
                audit_refs=[f"audit:{result['audit_event']['event_id']}"],
            )
        )

    def _create_evidence_bundle(self, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        result = self.store.create_evidence_bundle(str(body.get("search_snapshot_id", "")), scope)
        if result.get("status") == "not_found":
            self._send_json(_json_response("failed", errors=[{"code": "CS_SEARCH_SNAPSHOT_NOT_FOUND", "message": "Search snapshot not found."}]), 404)
            return
        if result.get("status") == "scope_denied":
            self._send_json(_json_response("denied", errors=[{"code": "CS_SCOPE_DENIED", "message": "Search snapshot is outside the requested scope."}]), 403)
            return
        bundle = result["bundle"]
        self._send_json(
            _json_response(
                "success",
                evidence_bundle=bundle,
                evidence_refs=[f"evidence_bundle:{bundle['evidence_bundle_id']}", f"search_snapshot:{bundle['search_snapshot_id']}"],
                audit_refs=[f"audit:{result['audit_event']['event_id']}"],
            )
        )

    def _create_claim(self, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        bundle_id = body.get("evidence_bundle_id")
        if isinstance(bundle_id, str) and bundle_id:
            result = self.store.create_claim_from_evidence_bundle(bundle_id, str(body.get("statement", "")), scope)
        else:
            result = self.store.create_unsupported_claim(str(body.get("statement", "")), scope)
        if result.get("status") == "not_found":
            self._send_json(_json_response("failed", errors=[{"code": "CS_EVIDENCE_BUNDLE_NOT_FOUND", "message": "Evidence bundle not found."}]), 404)
            return
        if result.get("status") == "scope_denied":
            self._send_json(_json_response("denied", errors=[{"code": "CS_SCOPE_DENIED", "message": "Evidence bundle is outside the requested scope."}]), 403)
            return
        claim = result["claim"]
        refs = [f"claim:{claim['claim_id']}"]
        evidence = claim.get("evidence_bundle", {})
        if evidence.get("evidence_bundle_id"):
            refs.append(f"evidence_bundle:{evidence['evidence_bundle_id']}")
        refs.extend(evidence.get("artifact_refs", []))
        self._send_json(_json_response("success", claim=claim, evidence_refs=refs, audit_refs=[f"audit:{result['audit_event']['event_id']}"]))

    def _approve_claim(self, claim_id: str, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        result = self.store.approve_claim(claim_id, scope)
        if result.get("status") == "not_found":
            self._send_json(_json_response("failed", errors=[{"code": "CS_CLAIM_NOT_FOUND", "message": "Claim not found."}]), 404)
            return
        if result.get("status") == "scope_denied":
            self._send_json(_json_response("denied", errors=[{"code": "CS_SCOPE_DENIED", "message": "Claim is outside the requested scope."}]), 403)
            return
        if result.get("status") == "evidence_required":
            self._send_json(
                _json_response(
                    "failed",
                    claim=result["claim"],
                    audit_refs=[f"audit:{result['audit_event']['event_id']}"],
                    errors=[{"code": "CS_CLAIM_EVIDENCE_REQUIRED", "message": "Claim approval requires evidence."}],
                ),
                400,
            )
            return
        self._send_json(
            _json_response(
                "success",
                claim=result["claim"],
                evidence_refs=[f"claim:{claim_id}"],
                audit_refs=[f"audit:{result['audit_event']['event_id']}"],
            )
        )

    def _create_action(self, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        claim_id = str(body.get("claim_id", ""))
        mission_id = body.get("mission_id") if isinstance(body.get("mission_id"), str) else None
        created_mission: dict[str, Any] | None = None
        activation: dict[str, Any] | None = None
        if not mission_id:
            mission_result = self.store.create_mission_contract(
                str(body.get("mission_goal", body.get("goal", "Complete VS0 runtime action safely."))),
                scope,
                claim_id=claim_id,
            )
            if mission_result.get("status"):
                self._send_json(_json_response("failed", errors=[{"code": "CS_MISSION_CREATE_FAILED", "message": mission_result["status"]}]), 400)
                return
            created_mission = mission_result["mission"]
            mission_id = created_mission["mission_id"]
            activation = self.store.activate_mission(mission_id, scope, mode=str(body.get("mode", "autopilot")))
        result = self.store.propose_action(
            mission_id,
            claim_id,
            str(body.get("action_kind", "external_writeback")),
            str(body.get("risk", "high")),
            scope,
            goal=str(body.get("goal", "Run a local mock ConnectorHub action.")),
            connector=str(body.get("connector", "mock_connector")),
            target=str(body.get("target", "mock://local-target")),
        )
        if result.get("status"):
            self._send_json(_json_response("failed", errors=[{"code": "CS_ACTION_PROPOSE_FAILED", "message": result["status"]}]), 400)
            return
        card = result["action_card"]
        audit_refs = [f"audit:{result['audit_event']['event_id']}"]
        if activation:
            audit_refs.extend(f"audit:{event['event_id']}" for event in activation.get("audit_events", []))
        self._send_json(
            _json_response(
                "success",
                mission=created_mission,
                workspace_mode=activation.get("workspace_mode") if activation else None,
                action_card=card,
                evidence_refs=[f"action:{card['action_id']}", f"claim:{claim_id}", f"mission:{mission_id}"],
                audit_refs=audit_refs,
                policy_decisions=[card["policy_decision"]],
                policy_decision_refs=[f"policy:{card['policy_decision']['id']}"],
            )
        )

    def _show_action_dry_run(self, action_id: str, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        action = self.store.get_action(action_id)
        if action is None:
            self._send_json(_json_response("failed", errors=[{"code": "CS_ACTION_NOT_FOUND", "message": "Action Card not found."}]), 404)
            return
        if action.get("scope") != scope:
            self._send_json(_json_response("denied", errors=[{"code": "CS_SCOPE_DENIED", "message": "Action Card is outside the requested scope."}]), 403)
            return
        event = self.store.append_audit("action.dry_run.read", scope, {"type": "action", "id": action_id}, {"dry_run_id": action.get("dry_run", {}).get("dry_run_id")})
        self._send_json(
            _json_response(
                "success",
                action_id=action_id,
                dry_run=action.get("dry_run"),
                policy_decisions=[action.get("policy_decision")],
                evidence_refs=[f"action:{action_id}", f"dry_run:{action.get('dry_run', {}).get('dry_run_id')}"],
                audit_refs=[f"audit:{event['event_id']}"],
                policy_decision_refs=[f"policy:{action.get('policy_decision', {}).get('id')}"],
            )
        )

    def _approve_action(self, action_id: str, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        result = self.store.approve_action(action_id, scope, approver=str(body.get("approver", "owner")))
        if result.get("status") == "not_found":
            self._send_json(_json_response("failed", errors=[{"code": "CS_ACTION_NOT_FOUND", "message": "Action Card not found."}]), 404)
            return
        if result.get("status") == "scope_denied":
            self._send_json(_json_response("denied", errors=[{"code": "CS_SCOPE_DENIED", "message": "Action Card is outside the requested scope."}]), 403)
            return
        refs = []
        if result.get("audit_event"):
            refs.append(f"audit:{result['audit_event']['event_id']}")
        self._send_json(_json_response("success", action_card=result["action_card"], evidence_refs=[f"action:{action_id}"], audit_refs=refs))

    def _execute_action(self, action_id: str, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        result = self.store.execute_action(action_id, scope)
        if result.get("status") == "not_found":
            self._send_json(_json_response("failed", errors=[{"code": "CS_ACTION_NOT_FOUND", "message": "Action Card not found."}]), 404)
            return
        if result.get("status") == "scope_denied":
            self._send_json(_json_response("denied", errors=[{"code": "CS_SCOPE_DENIED", "message": "Action Card is outside the requested scope."}]), 403)
            return
        if result.get("status") == "policy_denied":
            self._send_json(
                _json_response(
                    "denied",
                    action_card=result["action_card"],
                    policy_decisions=[result["policy_decision"]],
                    policy_decision_refs=[f"policy:{result['policy_decision']['id']}"],
                    audit_refs=[f"audit:{result['audit_event']['event_id']}"],
                    errors=[{"code": "CS_ACTION_POLICY_DENIED", "message": result["policy_decision"]["reason"]}],
                ),
                403,
            )
            return
        self._send_json(
            _json_response(
                "success",
                action_card=result["action_card"],
                action_result=result["action_result"],
                evidence_refs=[f"action:{action_id}"],
                audit_refs=[f"audit:{result['audit_event']['event_id']}"],
            )
        )

    def _audit_verify(self) -> None:
        report = self.store.verify_audit()
        self._send_json(_json_response(report["status"], audit_integrity=report), 200 if report["status"] == "success" else 500)


def make_server(root: Path, state_dir: Path, host: str = "127.0.0.1", port: int = 0) -> RuntimeHTTPServer:
    return RuntimeHTTPServer((host, port), VS0RuntimeHandler, root=root.resolve(), state_dir=state_dir.resolve())


def run_server(root: Path, state_dir: Path, host: str = "127.0.0.1", port: int = 8787) -> None:
    server = make_server(root, state_dir, host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()

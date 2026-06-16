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
    "POST /ontology/suggestion-sets",
    "POST /ontology/suggestion-sets/{suggestion_set_id}/review",
    "POST /ontology/suggestion-sets/{suggestion_set_id}/promote",
    "GET /ontology/objects/{ontology_object_id}",
    "POST /ontology/draft-truth-test",
    "POST /ontology/invalid-graph-test",
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


def render_home(readiness: dict[str, Any], scenario: str | None = None, autorun_evux: bool = False) -> str:
    surfaces = "\n".join(
        f"<li><a href='#{html.escape(surface.lower().replace('/', '-').replace(' ', '-'))}'>{html.escape(surface)}</a></li>"
        for surface in UI_SURFACES
    )
    production_ready = str(readiness["production_release_ready"]).lower()
    runtime_ready = str(readiness["vs0_runtime_ready"]).lower()
    autorun_evux_value = "true" if scenario == "vs0-evux" and autorun_evux else "false"
    autorun_vs1_value = "true" if scenario == "vs1-ontology" and autorun_evux else "false"
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
    .stepper {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 8px;
      margin: 16px 0;
      padding: 0;
      list-style: none;
    }}
    .stepper li {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 9px 10px;
      font-size: 13px;
      font-weight: 700;
    }}
    .stepper li[data-step-status="complete"] {{ border-color: #9ac2ad; color: var(--safe); }}
    .stepper li[data-step-status="current"] {{ border-color: var(--accent); color: var(--accent); }}
    .step-grid {{ display: grid; gap: 12px; margin-top: 14px; }}
    .step-card {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .step-card h3 {{ margin: 0 0 12px; font-size: 16px; }}
    .step-card label {{ display: block; margin-bottom: 5px; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    input, select {{
      width: min(100%, 560px);
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
      color: var(--ink);
      padding: 8px 10px;
      font: inherit;
      margin: 0 8px 10px 0;
    }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin-top: 12px; }}
    .detail-grid div {{ border-top: 1px solid var(--line); padding-top: 8px; min-height: 54px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 700; }}
    code {{ background: #eef3f7; padding: 2px 4px; border-radius: 4px; }}
    button {{
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 8px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    button:disabled {{ opacity: 0.65; cursor: wait; }}
    .evidence-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 8px; margin: 12px 0; }}
    .evidence-list div {{ border: 1px solid var(--line); background: var(--surface); border-radius: 6px; padding: 8px; min-height: 54px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #eef3f7; border: 1px solid var(--line); border-radius: 6px; padding: 10px; max-height: 320px; overflow: auto; }}
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
      <section
        id="vs1-ontology-loop"
        data-vs1-clicked="false"
        data-vs1-flow-complete="false"
        data-production-release-claimed="false"
        data-live-connector-claimed="false"
        data-human-acceptance-claimed="false"
      >
        <h2>VS1 Ontology Review</h2>
        <p>Local VS1 proof only. Suggestions stay draft until selected and explicitly promoted; production, live connector readiness, and human UX acceptance are not claimed.</p>
        <button id="run-vs1-ontology" type="button">Run ontology proof</button>
        <div class="status-row">
          <span id="vs1-ontology-status" class="badge warn" data-vs1-status="idle">idle</span>
          <span id="vs1-real-external-calls" class="badge safe">real_external_http_calls=0</span>
          <span id="vs1-audit-verify" class="badge">audit not verified</span>
        </div>
        <div class="detail-grid">
          <div><span class="label">Artifact</span><code id="vs1-artifact-id">not-run</code></div>
          <div><span class="label">Search Snapshot</span><code id="vs1-search-snapshot-id">not-run</code></div>
          <div><span class="label">SuggestionSet</span><code id="vs1-suggestion-set-id">not-run</code></div>
          <div><span class="label">Review</span><code id="vs1-review-state">not-run</code></div>
          <div><span class="label">ChangeSet</span><code id="vs1-change-set-id">not-run</code></div>
          <div><span class="label">Object Profile</span><code id="vs1-object-profile-id">not-run</code></div>
          <div><span class="label">Search Integration</span><span id="vs1-search-integration">not-run</span></div>
          <div><span class="label">Claim Context</span><span id="vs1-claim-context">not-run</span></div>
          <div><span class="label">Action Impact</span><span id="vs1-action-impact">not-run</span></div>
          <div><span class="label">Draft Truth Guard</span><span id="vs1-draft-truth-guard">not-run</span></div>
        </div>
        <pre id="vs1-ontology-trace" aria-label="VS1 ontology workflow trace">{{}}</pre>
      </section>
      <section id="claim-builder"><h2>Claim Builder</h2><p>Claims stay draft unless backed by an Evidence Bundle and approval evidence.</p></section>
      <section id="action-card"><h2>Action Card</h2><p>Actions expose dry-run diff, expected impact, policy decision, approval, execution, and mock ConnectorHub boundary.</p></section>
      <section id="audit-detail"><h2>Audit Detail</h2><p>Audit events form a local tamper-evident hash chain verified by <code>cornerstone audit verify --json</code>.</p></section>
      <section
        id="vs0-evux-loop"
        data-evux-clicked="false"
        data-operator-flow="step-by-step"
        data-operator-step-count="9"
        data-current-step="1"
        data-completed-steps="0"
        data-production-release-claimed="false"
        data-live-connector-claimed="false"
        data-human-acceptance-claimed="false"
      >
        <h2>VS0 Operator Flow</h2>
        <p>Local VS0 proof only. Production release, live connector readiness, autonomous external writeback, and human acceptance are not claimed.</p>
        <div class="panel-grid" aria-label="VS0 operator boundary">
          <div class="panel"><div class="label">Mode</div><strong>local/mock</strong><br><span id="ui-local-boundary">Connector writes are mocked; real external writeback stays disabled.</span></div>
          <div class="panel"><div class="label">Current position</div><strong id="ui-current-step">1. Select / upload Artifact</strong><br><span id="ui-workflow-position">Step 1 of 9</span></div>
          <div class="panel"><div class="label">Not claimed</div><span id="ui-not-production-ready">production release=false; live connector=false; human acceptance=HUMAN_REQUIRED</span></div>
        </div>
        <ol class="stepper" aria-label="VS0 operator steps">
          <li id="operator-step-1" data-step-status="current">1. Select/upload Artifact</li>
          <li id="operator-step-2" data-step-status="pending">2. Search</li>
          <li id="operator-step-3" data-step-status="pending">3. Review Evidence</li>
          <li id="operator-step-4" data-step-status="pending">4. Create Claim</li>
          <li id="operator-step-5" data-step-status="pending">5. Review Action Card</li>
          <li id="operator-step-6" data-step-status="pending">6. Dry-run</li>
          <li id="operator-step-7" data-step-status="pending">7. Approve</li>
          <li id="operator-step-8" data-step-status="pending">8. Execute local/mock action</li>
          <li id="operator-step-9" data-step-status="pending">9. Inspect Audit</li>
        </ol>
        <button id="run-evux" type="button">Run guided proof through visible steps</button>
        <div class="status-row">
          <span id="evux-status" class="badge warn" data-evux-status="idle">idle</span>
          <span id="evux-zero-evidence-denial" class="badge">zero-evidence claim pending</span>
          <span id="evux-mock-calls" class="badge">mock_connector_calls=0</span>
          <span id="evux-real-external-calls" class="badge safe">real_external_http_calls=0</span>
          <span id="evux-audit-verify" class="badge">audit not verified</span>
        </div>
        <div class="step-grid">
          <article class="step-card" data-operator-step-card="artifact">
            <h3>1. Select / upload Artifact</h3>
            <label for="artifact-fixture-select">Fixture artifact</label>
            <select id="artifact-fixture-select">
              <option value="fixtures/vs0/packs/01_artifact_basic/input.txt">fixtures/vs0/packs/01_artifact_basic/input.txt</option>
            </select>
            <button id="step-artifact-run" type="button">Select fixture artifact</button>
            <div class="detail-grid">
              <div><span class="label">Artifact ID</span><code id="evux-artifact-id">not-run</code></div>
              <div><span class="label">Checksum</span><code id="ui-artifact-checksum">not-run</code></div>
              <div><span class="label">Source</span><code id="ui-artifact-source">not-run</code></div>
              <div><span class="label">Derived status</span><code id="ui-artifact-derived-status">not-run</code></div>
              <div><span class="label">Evidence refs</span><code id="ui-artifact-evidence-refs">not-run</code></div>
              <div><span class="label">Audit refs</span><code id="ui-artifact-audit-refs">not-run</code></div>
            </div>
          </article>
          <article class="step-card" data-operator-step-card="search">
            <h3>2. Search</h3>
            <label for="search-query">Query</label>
            <input id="search-query" value="alpha-evidence-anchor">
            <button id="step-search-run" type="button" disabled>Search selected artifact</button>
            <div class="detail-grid">
              <div><span class="label">Query</span><code id="ui-search-query">not-run</code></div>
              <div><span class="label">Search Snapshot</span><code id="evux-search-snapshot-id">not-run</code></div>
              <div><span class="label">Evidence eligibility</span><code id="ui-search-evidence-eligibility">not-run</code></div>
              <div><span class="label">Result snippet</span><span id="ui-search-snippet">not-run</span></div>
            </div>
          </article>
          <article class="step-card" data-operator-step-card="evidence">
            <h3>3. Review Evidence</h3>
            <button id="step-evidence-run" type="button" disabled>Create Evidence Bundle</button>
            <div class="detail-grid">
              <div><span class="label">Evidence Bundle</span><code id="evux-evidence-bundle-id">not-run</code></div>
              <div><span class="label">Supports Claim</span><span id="ui-evidence-support">not-run</span></div>
              <div><span class="label">Insufficient when</span><span id="ui-evidence-insufficient">not-run</span></div>
            </div>
          </article>
          <article class="step-card" data-operator-step-card="claim">
            <h3>4. Create Claim</h3>
            <label for="claim-statement">Claim statement</label>
            <input id="claim-statement" value="The Alpha evidence anchor is ready for local VS0 operator acceptance.">
            <button id="step-claim-run" type="button" disabled>Create evidence-backed Claim</button>
            <div class="detail-grid">
              <div><span class="label">Draft zero-evidence Claim</span><code id="ui-claim-draft-state">not-run</code></div>
              <div><span class="label">Evidence-backed Claim</span><code id="ui-claim-evidence-state">not-run</code></div>
              <div><span class="label">Approved Claim</span><code id="ui-claim-approved-state">pending approval</code></div>
              <div><span class="label">Claim ID</span><code id="evux-claim-id">not-run</code></div>
              <div><span class="label">Zero-evidence denial cause</span><span id="ui-zero-evidence-cause">not-run</span></div>
              <div><span class="label">Resolution guide</span><span id="ui-zero-evidence-resolution">not-run</span></div>
            </div>
          </article>
          <article class="step-card" data-operator-step-card="action">
            <h3>5. Review Action Card</h3>
            <button id="step-action-run" type="button" disabled>Create Action Card</button>
            <div class="detail-grid">
              <div><span class="label">Action Card</span><code id="evux-action-id">not-run</code></div>
              <div><span class="label">Diff</span><span id="ui-action-diff">not-run</span></div>
              <div><span class="label">Expected impact</span><span id="ui-action-impact">not-run</span></div>
              <div><span class="label">Evidence</span><span id="ui-action-evidence">not-run</span></div>
              <div><span class="label">Policy decision</span><code id="ui-action-policy">not-run</code></div>
              <div><span class="label">Risk</span><code id="ui-action-risk">not-run</code></div>
              <div><span class="label">Approval state</span><code id="ui-action-approval">not-run</code></div>
              <div><span class="label">Mock/local boundary</span><span id="ui-action-boundary">not-run</span></div>
              <div><span class="label">Rollback / compensation</span><span id="ui-action-rollback">not-run</span></div>
            </div>
          </article>
          <article class="step-card" data-operator-step-card="dry-run">
            <h3>6. Dry-run</h3>
            <button id="step-dry-run" type="button" disabled>Run dry-run</button>
            <div class="detail-grid">
              <div><span class="label">Dry-run ID</span><code id="ui-dry-run-id">not-run</code></div>
              <div><span class="label">Dry-run diff</span><span id="ui-dry-run-diff">not-run</span></div>
              <div><span class="label">Expected connector calls</span><code id="ui-dry-run-calls">not-run</code></div>
            </div>
          </article>
          <article class="step-card" data-operator-step-card="approve">
            <h3>7. Approve</h3>
            <button id="step-approve-run" type="button" disabled>Approve Claim and Action</button>
            <div class="detail-grid">
              <div><span class="label">Claim approval</span><code id="ui-claim-approval-result">not-run</code></div>
              <div><span class="label">Action approval</span><code id="ui-action-approval-result">not-run</code></div>
            </div>
          </article>
          <article class="step-card" data-operator-step-card="execute">
            <h3>8. Execute local/mock action</h3>
            <button id="step-execute-run" type="button" disabled>Execute local/mock action</button>
            <div class="detail-grid">
              <div><span class="label">Execution status</span><code id="ui-execution-status">not-run</code></div>
              <div><span class="label">Mock connector calls</span><code id="ui-execution-mock-calls">mock_connector_calls=0</code></div>
              <div><span class="label">Real external HTTP calls</span><code id="ui-execution-real-calls">real_external_http_calls=0</code></div>
            </div>
          </article>
          <article class="step-card" data-operator-step-card="audit">
            <h3>9. Inspect Audit</h3>
            <button id="step-audit-run" type="button" disabled>Inspect audit timeline</button>
            <div class="detail-grid">
              <div><span class="label">Audit events</span><span id="ui-audit-events">not-run</span></div>
              <div><span class="label">Audit verification</span><code id="ui-audit-verification">not-run</code></div>
            </div>
          </article>
        </div>
        <pre id="evux-trace" aria-label="EVUX workflow trace">{{}}</pre>
      </section>
    </div>
  </main>
  <script>
    const evuxAutorun = {autorun_evux_value};
    const evuxScope = {{
      tenant_id: "local-dev",
      owner_id: "local-user",
      namespace_id: "personal",
      workspace_id: "default"
    }};
    const evuxTrace = [];
    const operatorState = {{
      completedSteps: 0,
      artifact: {{}},
      search: {{}},
      evidence: {{}},
      claims: {{}},
      action: {{}},
      dryRun: {{}},
      approvals: {{}},
      execution: {{}},
      audit: {{}}
    }};
    const stepNames = [
      "Select / upload Artifact",
      "Search",
      "Review Evidence",
      "Create Claim",
      "Review Action Card",
      "Dry-run",
      "Approve",
      "Execute local/mock action",
      "Inspect Audit"
    ];
    function setText(id, value) {{
      const node = document.getElementById(id);
      if (node) node.textContent = value;
    }}
    function setDisabled(id, disabled) {{
      const node = document.getElementById(id);
      if (node) node.disabled = disabled;
    }}
    function asList(value) {{
      if (!value) return "none";
      if (Array.isArray(value)) return value.length ? value.join(", ") : "none";
      return String(value);
    }}
    function shortHash(value) {{
      return value ? String(value).slice(0, 16) + "..." : "not-run";
    }}
    function markStep(step, status) {{
      const node = document.getElementById("operator-step-" + step);
      if (node) node.dataset.stepStatus = status;
    }}
    function completeStep(step) {{
      markStep(step, "complete");
      if (step < 9) markStep(step + 1, "current");
      operatorState.completedSteps = Math.max(operatorState.completedSteps, step);
      const section = document.getElementById("vs0-evux-loop");
      section.dataset.completedSteps = String(operatorState.completedSteps);
      section.dataset.currentStep = String(Math.min(step + 1, 9));
      setText("ui-current-step", Math.min(step + 1, 9) + ". " + stepNames[Math.min(step, 8)]);
      setText("ui-workflow-position", "Step " + Math.min(step + 1, 9) + " of 9");
    }}
    function traceStep(name, payload) {{
      evuxTrace.push({{ step: name, payload }});
      setText("evux-trace", JSON.stringify({{ workflow: "vs0-operator-acceptance-ui", operator_state: operatorState, steps: evuxTrace }}, null, 2));
    }}
    async function api(path, body) {{
      const response = await fetch(path, {{
        method: "POST",
        headers: {{ "content-type": "application/json" }},
        body: JSON.stringify(Object.assign({{}}, evuxScope, body || {{}}))
      }});
      const payload = await response.json();
      traceStep(path, {{ status: response.status, payload }});
      return {{ response, payload }};
    }}
    async function getJson(path) {{
      const response = await fetch(path);
      const payload = await response.json();
      traceStep(path, {{ status: response.status, payload }});
      return {{ response, payload }};
    }}
    const vs1Autorun = {autorun_vs1_value};
    const vs1Trace = [];
    const vs1State = {{
      completed: false,
      artifact: {{}},
      search: {{}},
      suggestionSet: {{}},
      review: {{}},
      promotion: {{}},
      profile: {{}},
      claim: {{}},
      action: {{}},
      guards: {{}},
      audit: {{}}
    }};
    function traceVs1(name, payload) {{
      vs1Trace.push({{ step: name, payload }});
      setText("vs1-ontology-trace", JSON.stringify({{ workflow: "vs1-ontology-suggest-promote", state: vs1State, steps: vs1Trace }}, null, 2));
    }}
    async function vs1Api(path, body) {{
      const response = await fetch(path, {{
        method: "POST",
        headers: {{ "content-type": "application/json" }},
        body: JSON.stringify(Object.assign({{}}, evuxScope, body || {{}}))
      }});
      const payload = await response.json();
      traceVs1(path, {{ status: response.status, payload }});
      return {{ response, payload }};
    }}
    async function vs1Get(path) {{
      const response = await fetch(path);
      const payload = await response.json();
      traceVs1(path, {{ status: response.status, payload }});
      return {{ response, payload }};
    }}
    function firstByKind(suggestionSet, kind, minConfidence) {{
      const groups = [
        ...(suggestionSet.object_suggestions || []),
        ...(suggestionSet.property_suggestions || []),
        ...(suggestionSet.link_suggestions || [])
      ];
      return groups.find((candidate) => candidate.candidate_kind === kind && candidate.confidence >= minConfidence);
    }}
    function allCandidates(suggestionSet) {{
      return [
        ...(suggestionSet.object_suggestions || []),
        ...(suggestionSet.property_suggestions || []),
        ...(suggestionSet.link_suggestions || [])
      ];
    }}
    async function runVs1Ontology() {{
      const button = document.getElementById("run-vs1-ontology");
      const status = document.getElementById("vs1-ontology-status");
      const section = document.getElementById("vs1-ontology-loop");
      section.dataset.vs1Clicked = "true";
      button.disabled = true;
      status.dataset.vs1Status = "running";
      status.textContent = "running";
      vs1Trace.length = 0;
      try {{
        const artifactResponse = await vs1Api("/artifacts", {{
          path: "fixtures/vs1/ontology/vendor_risk.txt",
          source: "local_fixture",
          media_type: "text/plain",
          trust: "untrusted"
        }});
        const artifact = artifactResponse.payload.artifact;
        vs1State.artifact = {{ artifact_id: artifact.artifact_id, evidence_refs: artifactResponse.payload.evidence_refs || [] }};
        setText("vs1-artifact-id", artifact.artifact_id);

        const searchResponse = await vs1Api("/search", {{ query: "Northstar Labs vendor risk" }});
        const snapshot = searchResponse.payload.search_snapshot;
        vs1State.search = {{ search_snapshot_id: snapshot.search_snapshot_id, result_count: snapshot.result_count }};
        setText("vs1-search-snapshot-id", snapshot.search_snapshot_id);

        const suggestResponse = await vs1Api("/ontology/suggestion-sets", {{
          source_type: "search",
          source_id: snapshot.search_snapshot_id
        }});
        const suggestionSet = suggestResponse.payload.ontology_suggestion_set;
        const objectCandidates = suggestionSet.object_suggestions || [];
        const propertyCandidates = suggestionSet.property_suggestions || [];
        const linkCandidates = suggestionSet.link_suggestions || [];
        const selected = [
          ...objectCandidates.filter((candidate) => candidate.confidence >= 0.6),
          ...propertyCandidates.filter((candidate) => candidate.confidence >= 0.6).slice(0, 2),
          ...linkCandidates.filter((candidate) => candidate.confidence >= 0.6).slice(0, 2)
        ].map((candidate) => candidate.candidate_id);
        const lowCandidate = allCandidates(suggestionSet).find((candidate) => candidate.confidence < 0.6);
        const rejected = propertyCandidates.slice(2, 3).map((candidate) => candidate.candidate_id);
        const deferred = lowCandidate ? [lowCandidate.candidate_id] : [];
        vs1State.suggestionSet = {{
          suggestion_set_id: suggestionSet.suggestion_set_id,
          seed_types: suggestionSet.universal_seed_types,
          object_count: objectCandidates.length,
          property_count: propertyCandidates.length,
          link_count: linkCandidates.length,
          low_candidate_id: lowCandidate && lowCandidate.candidate_id
        }};
        setText("vs1-suggestion-set-id", suggestionSet.suggestion_set_id);

        const draftGuard = await vs1Api("/ontology/draft-truth-test", {{
          suggestion_set_id: suggestionSet.suggestion_set_id,
          candidate_id: selected[0],
          purpose: "claim_or_action_truth"
        }});
        vs1State.guards.draft_truth_denied = draftGuard.response.status === 403 &&
          draftGuard.payload.errors.some((error) => error.code === "CS_ONTOLOGY_DRAFT_TRUTH_DENIED");
        setText("vs1-draft-truth-guard", vs1State.guards.draft_truth_denied ? "denied before promotion" : "unexpected");

        const reviewResponse = await vs1Api("/ontology/suggestion-sets/" + suggestionSet.suggestion_set_id + "/review", {{
          select: selected,
          reject: rejected,
          defer: deferred
        }});
        const reviewed = reviewResponse.payload.ontology_suggestion_set;
        vs1State.review = reviewed.review_state;
        setText("vs1-review-state", "selected=" + reviewed.review_state.selected.length + "; rejected=" + reviewed.review_state.rejected.length + "; deferred=" + reviewed.review_state.deferred.length);

        const promoteResponse = await vs1Api("/ontology/suggestion-sets/" + suggestionSet.suggestion_set_id + "/promote", {{
          candidate_ids: selected
        }});
        const changeSet = promoteResponse.payload.ontology_change_set;
        const promotedObjects = promoteResponse.payload.ontology_objects || [];
        const promotedLinks = promoteResponse.payload.ontology_links || [];
        const profileObjectId = promotedLinks[0] ? promotedLinks[0].source_object_id : (promotedObjects[0] && promotedObjects[0].ontology_object_id);
        const profileObject = promotedObjects.find((object) => object.ontology_object_id === profileObjectId) || promotedObjects[0];
        vs1State.promotion = {{
          ontology_change_set_id: changeSet.ontology_change_set_id,
          previous_version: changeSet.previous_version,
          next_version: changeSet.next_version,
          semver_bump: changeSet.semver_bump,
          object_refs: promotedObjects.map((object) => "ontology_object:" + object.ontology_object_id),
          link_refs: promotedLinks.map((link) => "ontology_link:" + link.ontology_link_id)
        }};
        setText("vs1-change-set-id", changeSet.ontology_change_set_id);

        const profileResponse = await vs1Get("/ontology/objects/" + profileObject.ontology_object_id);
        vs1State.profile = {{
          ontology_object_id: profileObject.ontology_object_id,
          sections: profileResponse.payload.ontology_object_profile.profile_sections,
          evidence_refs: profileResponse.payload.ontology_object_profile.evidence_refs || [],
          link_count: (profileResponse.payload.ontology_object_profile.links || []).length,
          linked_object_count: (profileResponse.payload.ontology_object_profile.linked_objects || []).length,
          related_claim_count: (profileResponse.payload.ontology_object_profile.related_claims || []).length,
          related_action_count: (profileResponse.payload.ontology_object_profile.related_actions || []).length,
          activity_count: (profileResponse.payload.ontology_object_profile.activity_history || []).length,
          change_set_ref_count: (profileResponse.payload.ontology_object_profile.change_set_refs || []).length
        }};
        setText("vs1-object-profile-id", profileObject.ontology_object_id);

        const ontologySearch = await vs1Api("/search", {{ query: profileObject.label }});
        const ontologyResult = (ontologySearch.payload.search_snapshot.results || []).find((result) => result.result_type === "ontology_object");
        vs1State.search.promoted_object_result = Boolean(ontologyResult);
        setText("vs1-search-integration", vs1State.search.promoted_object_result ? "promoted object returned in search" : "not returned");

        const bundleResponse = await vs1Api("/evidence-bundles", {{ search_snapshot_id: snapshot.search_snapshot_id }});
        const evidenceBundle = bundleResponse.payload.evidence_bundle;
        const zeroClaim = await vs1Api("/claims", {{
          statement: "Ontology context alone should not approve this claim.",
          ontology_object_refs: vs1State.promotion.object_refs
        }});
        const zeroApprove = await vs1Api("/claims/" + zeroClaim.payload.claim.claim_id + "/approve", {{}});
        const claimResponse = await vs1Api("/claims", {{
          evidence_bundle_id: evidenceBundle.evidence_bundle_id,
          statement: "Northstar Labs vendor risk requires owner-reviewed follow-up.",
          ontology_object_refs: vs1State.promotion.object_refs
        }});
        const claim = claimResponse.payload.claim;
        const claimApprove = await vs1Api("/claims/" + claim.claim_id + "/approve", {{}});
        vs1State.claim = {{
          claim_id: claim.claim_id,
          ontology_context_refs: claim.ontology_context.object_refs,
          zero_evidence_denied: zeroApprove.response.status === 400,
          approved: claimApprove.payload.claim.trust_state === "approved"
        }};
        setText("vs1-claim-context", "context_refs=" + vs1State.claim.ontology_context_refs.length + "; evidence_required=" + vs1State.claim.zero_evidence_denied);

        const actionResponse = await vs1Api("/actions", {{
          claim_id: claim.claim_id,
          goal: "Record local ontology impact review",
          action_kind: "external_writeback",
          risk: "high",
          connector: "mock_connector",
          target: "mock://vs1-ontology/browser",
          ontology_object_refs: vs1State.promotion.object_refs
        }});
        const action = actionResponse.payload.action_card;
        const actionApprove = await vs1Api("/actions/" + action.action_id + "/approve", {{ approver: "owner" }});
        const actionExecute = await vs1Api("/actions/" + action.action_id + "/execute", {{}});
        vs1State.action = {{
          action_id: action.action_id,
          ontology_impact: action.ontology_impact,
          real_external_http_calls: actionExecute.payload.action_result.external_http_calls,
          mock_connector_calls: actionExecute.payload.action_result.mock_connector_calls,
          approved: actionApprove.payload.action_card.approval.status === "approved"
        }};
        setText("vs1-action-impact", "objects=" + action.ontology_impact.object_refs.length + "; real_external_http_calls=" + vs1State.action.real_external_http_calls);
        setText("vs1-real-external-calls", "real_external_http_calls=" + vs1State.action.real_external_http_calls);

        const finalProfileResponse = await vs1Get("/ontology/objects/" + profileObject.ontology_object_id);
        const finalProfile = finalProfileResponse.payload.ontology_object_profile;
        vs1State.profile = {{
          ontology_object_id: profileObject.ontology_object_id,
          sections: finalProfile.profile_sections,
          evidence_refs: finalProfile.evidence_refs || [],
          link_count: (finalProfile.links || []).length,
          linked_object_count: (finalProfile.linked_objects || []).length,
          related_claim_count: (finalProfile.related_claims || []).length,
          related_action_count: (finalProfile.related_actions || []).length,
          activity_count: (finalProfile.activity_history || []).length,
          change_set_ref_count: (finalProfile.change_set_refs || []).length
        }};

        const auditEvents = await vs1Get("/audit-events");
        const auditVerify = await vs1Api("/audit/verify", {{}});
        const eventTypes = [...new Set((auditEvents.payload.audit_events || []).map((event) => event.event_type).filter(Boolean))];
        vs1State.audit = {{
          event_types: eventTypes,
          event_count: eventTypes.length,
          verification_status: auditVerify.payload.audit_integrity.status
        }};
        setText("vs1-audit-verify", auditVerify.payload.audit_integrity.status);
        vs1State.completed = vs1Passes();
        section.dataset.vs1FlowComplete = vs1State.completed ? "true" : "false";
        status.dataset.vs1Status = vs1State.completed ? "passed" : "failed";
        status.textContent = vs1State.completed ? "passed" : "failed";
      }} catch (error) {{
        traceVs1("browser_error", {{ message: String(error) }});
        status.dataset.vs1Status = "failed";
        status.textContent = "failed";
      }} finally {{
        button.disabled = false;
      }}
    }}
    function vs1Passes() {{
      const required = [
        "ontology.suggestion_set.generated",
        "ontology.suggestion_set.reviewed",
        "ontology.promotion.requested",
        "ontology.object.promoted",
        "ontology.change_set.created",
        "ontology.version.changed",
        "ontology.object.profile.read",
        "claim.approved",
        "action.card.proposed",
        "action.executed"
      ];
      return Boolean(
        vs1State.artifact.artifact_id &&
        vs1State.search.search_snapshot_id &&
        vs1State.suggestionSet.suggestion_set_id &&
        vs1State.suggestionSet.seed_types &&
        vs1State.suggestionSet.seed_types.length === 9 &&
        vs1State.suggestionSet.object_count >= 3 &&
        vs1State.suggestionSet.property_count >= 1 &&
        vs1State.suggestionSet.link_count >= 1 &&
        vs1State.guards.draft_truth_denied &&
        vs1State.review.selected &&
        vs1State.review.selected.length >= 3 &&
        vs1State.promotion.ontology_change_set_id &&
        vs1State.promotion.semver_bump === "minor" &&
        vs1State.profile.ontology_object_id &&
        vs1State.profile.link_count >= 1 &&
        vs1State.profile.linked_object_count >= 1 &&
        vs1State.profile.related_claim_count >= 1 &&
        vs1State.profile.related_action_count >= 1 &&
        vs1State.profile.activity_count >= 1 &&
        vs1State.profile.change_set_ref_count >= 1 &&
        vs1State.search.promoted_object_result &&
        vs1State.claim.zero_evidence_denied &&
        vs1State.claim.approved &&
        vs1State.action.real_external_http_calls === 0 &&
        vs1State.audit.verification_status === "success" &&
        required.every((eventType) => vs1State.audit.event_types.includes(eventType))
      );
    }}
    window.__cornerstoneVs1OntologyEvidence = function() {{
      const section = document.getElementById("vs1-ontology-loop");
      return {{
        schema_version: "cs.vs1_ontology_ui_state.v1",
        ontology_passes: vs1Passes(),
        production_release_claimed: section.dataset.productionReleaseClaimed === "true",
        live_connector_claimed: section.dataset.liveConnectorClaimed === "true",
        human_acceptance_claimed: section.dataset.humanAcceptanceClaimed === "true",
        state: vs1State
      }};
    }};
    async function selectArtifactStep() {{
      const artifact = await api("/artifacts", {{
        path: document.getElementById("artifact-fixture-select").value,
        source: "local_fixture",
        media_type: "text/plain",
        trust: "untrusted"
      }});
      const artifactRecord = artifact.payload.artifact;
      const artifactId = artifactRecord.artifact_id;
      operatorState.artifact = {{
        artifact_id: artifactId,
        checksum_sha256: artifactRecord.checksum_sha256,
        source: artifactRecord.source && artifactRecord.source.type,
        derived_status: artifactRecord.derived && artifactRecord.derived.status,
        evidence_refs: artifact.payload.evidence_refs || [],
        audit_refs: artifact.payload.audit_refs || []
      }};
      setText("evux-artifact-id", artifactId);
      setText("ui-artifact-checksum", shortHash(artifactRecord.checksum_sha256));
      setText("ui-artifact-source", operatorState.artifact.source);
      setText("ui-artifact-derived-status", operatorState.artifact.derived_status);
      setText("ui-artifact-evidence-refs", asList(operatorState.artifact.evidence_refs));
      setText("ui-artifact-audit-refs", asList(operatorState.artifact.audit_refs));
      setDisabled("step-search-run", false);
      completeStep(1);
      return artifactRecord;
    }}
    async function searchStep() {{
      const query = document.getElementById("search-query").value;
      const search = await api("/search", {{ query }});
      const snapshot = search.payload.search_snapshot;
      const result = (snapshot.results || [])[0] || {{}};
      operatorState.search = {{
        query,
        search_snapshot_id: snapshot.search_snapshot_id,
        snippet: result.snippet || "",
        evidence_eligible: snapshot.result_count > 0 && (result.evidence_refs || []).length > 0,
        evidence_refs: result.evidence_refs || [],
        audit_refs: search.payload.audit_refs || []
      }};
      setText("ui-search-query", query);
      setText("evux-search-snapshot-id", snapshot.search_snapshot_id);
      setText("ui-search-snippet", operatorState.search.snippet || "no result");
      setText("ui-search-evidence-eligibility", operatorState.search.evidence_eligible ? "eligible: result has artifact evidence refs" : "not eligible");
      setDisabled("step-evidence-run", false);
      completeStep(2);
      return snapshot;
    }}
    async function evidenceStep() {{
      const bundle = await api("/evidence-bundles", {{ search_snapshot_id: operatorState.search.search_snapshot_id }});
      const evidenceBundle = bundle.payload.evidence_bundle;
      const firstItem = (evidenceBundle.evidence_items || [])[0] || {{}};
      operatorState.evidence = {{
        evidence_bundle_id: evidenceBundle.evidence_bundle_id,
        search_snapshot_id: evidenceBundle.search_snapshot_id,
        supports_claim: Boolean(firstItem.snippet),
        support_snippet: firstItem.snippet || "",
        insufficient_guidance: "Insufficient: no Evidence Bundle, zero artifact refs, empty search result, or unsupported source. Attach eligible evidence before approval.",
        audit_refs: bundle.payload.audit_refs || []
      }};
      setText("evux-evidence-bundle-id", evidenceBundle.evidence_bundle_id);
      setText("ui-evidence-support", firstItem.snippet || "no supporting snippet");
      setText("ui-evidence-insufficient", operatorState.evidence.insufficient_guidance);
      setDisabled("step-claim-run", false);
      completeStep(3);
      return evidenceBundle;
    }}
    async function claimStep() {{
      const zeroClaim = await api("/claims", {{ statement: "Unsupported operator UI claim should remain draft." }});
      const zeroClaimRecord = zeroClaim.payload.claim;
      const zeroApproval = await api("/claims/" + zeroClaimRecord.claim_id + "/approve", {{}});
      const zeroDenied = zeroApproval.response.status === 400 &&
        zeroApproval.payload.errors.some((error) => error.code === "CS_CLAIM_EVIDENCE_REQUIRED");
      const claim = await api("/claims", {{
        evidence_bundle_id: operatorState.evidence.evidence_bundle_id,
        statement: document.getElementById("claim-statement").value
      }});
      const claimRecord = claim.payload.claim;
      operatorState.claims = {{
        zero_evidence_claim_id: zeroClaimRecord.claim_id,
        zero_evidence_state: zeroClaimRecord.trust_state,
        zero_evidence_denied: zeroDenied,
        zero_evidence_denial_code: zeroDenied ? "CS_CLAIM_EVIDENCE_REQUIRED" : "unexpected",
        evidence_claim_id: claimRecord.claim_id,
        evidence_claim_state: claimRecord.trust_state,
        approved_claim_state: "pending",
        resolution_guide: "Attach an Evidence Bundle with at least one artifact ref, then request owner approval."
      }};
      setText("evux-zero-evidence-denial", operatorState.claims.zero_evidence_denial_code);
      setText("ui-zero-evidence-cause", operatorState.claims.zero_evidence_denial_code + ": Claim approval requires evidence.");
      setText("ui-zero-evidence-resolution", operatorState.claims.resolution_guide);
      setText("ui-claim-draft-state", "Draft: " + zeroClaimRecord.claim_id);
      setText("ui-claim-evidence-state", "Evidence-backed: " + claimRecord.claim_id);
      setText("evux-claim-id", claimRecord.claim_id);
      setDisabled("step-action-run", false);
      completeStep(4);
      return claimRecord;
    }}
    async function actionStep() {{
      const action = await api("/actions", {{
        claim_id: operatorState.claims.evidence_claim_id,
        goal: "Record local operator acceptance status",
        action_kind: "external_writeback",
        risk: "high",
        connector: "mock_connector",
        target: "mock://vs0-operator-ui/browser"
      }});
      const card = action.payload.action_card;
      const dryRun = card.dry_run || {{}};
      const impact = dryRun.expected_impact || {{}};
      operatorState.action = {{
        action_id: card.action_id,
        diff: dryRun.diff || {{}},
        expected_impact: impact,
        evidence_bundle_id: card.evidence && card.evidence.evidence_bundle_id,
        policy_decision: card.policy_decision && card.policy_decision.decision,
        policy_reason: card.policy_decision && card.policy_decision.reason,
        risk: card.risk,
        approval_state: card.approval && card.approval.status,
        mock_local_boundary: card.connector_boundary && card.connector_boundary.mocked === true && card.connector_boundary.direct_provider_access === false,
        rollback_note: "Mock/local action only: no real external side effect. Compensate by recording a correcting Claim/Action and audit entry."
      }};
      setText("evux-action-id", card.action_id);
      setText("ui-action-diff", (dryRun.diff && dryRun.diff.before) + " -> " + (dryRun.diff && dryRun.diff.after));
      setText("ui-action-impact", "expected_connector_calls=" + impact.expected_connector_calls + "; mock_connector_calls=" + impact.mock_connector_calls + "; real_external_http_calls=" + impact.real_external_http_calls);
      setText("ui-action-evidence", "evidence_bundle:" + operatorState.action.evidence_bundle_id);
      setText("ui-action-policy", operatorState.action.policy_decision);
      setText("ui-action-risk", operatorState.action.risk);
      setText("ui-action-approval", operatorState.action.approval_state);
      setText("ui-action-boundary", "ConnectorHub mediated; mock_connector; direct_provider_access=false; credentials_exposed=false");
      setText("ui-action-rollback", operatorState.action.rollback_note);
      setDisabled("step-dry-run", false);
      completeStep(5);
      return card;
    }}
    async function dryRunStep() {{
      const dryRun = await api("/actions/" + operatorState.action.action_id + "/dry-run", {{}});
      const dryRunRecord = dryRun.payload.dry_run;
      const impact = dryRunRecord.expected_impact || {{}};
      operatorState.dryRun = {{
        dry_run_id: dryRunRecord.dry_run_id,
        diff: dryRunRecord.diff || {{}},
        expected_impact: impact
      }};
      setText("ui-dry-run-id", dryRunRecord.dry_run_id);
      setText("ui-dry-run-diff", (dryRunRecord.diff && dryRunRecord.diff.before) + " -> " + (dryRunRecord.diff && dryRunRecord.diff.after));
      setText("ui-dry-run-calls", "expected_connector_calls=" + impact.expected_connector_calls + "; real_external_http_calls=" + impact.real_external_http_calls);
      setDisabled("step-approve-run", false);
      completeStep(6);
      return dryRunRecord;
    }}
    async function approveStep() {{
      const claimApproval = await api("/claims/" + operatorState.claims.evidence_claim_id + "/approve", {{}});
      const actionApproval = await api("/actions/" + operatorState.action.action_id + "/approve", {{ approver: "owner" }});
      operatorState.claims.approved_claim_state = claimApproval.payload.claim.trust_state;
      operatorState.approvals = {{
        claim: claimApproval.payload.claim.trust_state,
        action: actionApproval.payload.action_card.approval.status,
        audit_refs: [...(claimApproval.payload.audit_refs || []), ...(actionApproval.payload.audit_refs || [])]
      }};
      setText("ui-claim-approved-state", "Approved: " + claimApproval.payload.claim.claim_id);
      setText("ui-claim-approval-result", operatorState.approvals.claim);
      setText("ui-action-approval", operatorState.approvals.action);
      setText("ui-action-approval-result", operatorState.approvals.action);
      setDisabled("step-execute-run", false);
      completeStep(7);
      return actionApproval.payload.action_card;
    }}
    async function executeStep() {{
      const executed = await api("/actions/" + operatorState.action.action_id + "/execute", {{}});
      const actionResult = executed.payload.action_result;
      operatorState.execution = {{
        status: actionResult.status,
        mock_connector_calls: actionResult.mock_connector_calls,
        real_external_http_calls: actionResult.external_http_calls,
        side_effect_boundary: actionResult.side_effect_boundary,
        audit_refs: executed.payload.audit_refs || []
      }};
      setText("ui-execution-status", actionResult.status + " / " + actionResult.side_effect_boundary);
      setText("ui-execution-mock-calls", "mock_connector_calls=" + actionResult.mock_connector_calls);
      setText("ui-execution-real-calls", "real_external_http_calls=" + actionResult.external_http_calls);
      setText("evux-mock-calls", "mock_connector_calls=" + actionResult.mock_connector_calls);
      setText("evux-real-external-calls", "real_external_http_calls=" + actionResult.external_http_calls);
      setDisabled("step-audit-run", false);
      completeStep(8);
      return actionResult;
    }}
    async function auditStep() {{
      const auditEvents = await getJson("/audit-events");
      const audit = await api("/audit/verify", {{}});
      const events = auditEvents.payload.audit_events || [];
      const eventTypes = [...new Set(events.map((event) => event.event_type).filter(Boolean))];
      operatorState.audit = {{
        event_types: eventTypes,
        event_count: events.length,
        verification_status: audit.payload.audit_integrity.status,
        verification_event_count: audit.payload.audit_integrity.event_count
      }};
      setText("ui-audit-events", eventTypes.join(", "));
      setText("ui-audit-verification", audit.payload.audit_integrity.status + "; events=" + audit.payload.audit_integrity.event_count);
      setText("evux-audit-verify", audit.payload.audit_integrity.status);
      completeStep(9);
      const passed = operatorPasses();
      const section = document.getElementById("vs0-evux-loop");
      const status = document.getElementById("evux-status");
      section.dataset.operatorFlowComplete = passed ? "true" : "false";
      status.dataset.evuxStatus = passed ? "passed" : "failed";
      status.textContent = passed ? "passed" : "failed";
      return operatorState.audit;
    }}
    function operatorPasses() {{
      const requiredEvents = [
        "artifact.ingested",
        "search.snapshot.created",
        "evidence_bundle.created",
        "claim.draft.created",
        "claim.approval.denied",
        "claim.approved",
        "action.card.proposed",
        "action.dry_run.read",
        "action.approved",
        "action.executed"
      ];
      return operatorState.completedSteps === 9 &&
        Boolean(operatorState.artifact.artifact_id && operatorState.artifact.checksum_sha256 && operatorState.artifact.derived_status === "ready") &&
        Boolean(operatorState.search.search_snapshot_id && operatorState.search.snippet && operatorState.search.evidence_eligible) &&
        Boolean(operatorState.evidence.evidence_bundle_id && operatorState.evidence.supports_claim && operatorState.evidence.insufficient_guidance) &&
        Boolean(operatorState.claims.zero_evidence_denied && operatorState.claims.evidence_claim_state === "evidence_backed" && operatorState.claims.approved_claim_state === "approved") &&
        Boolean(operatorState.action.action_id && operatorState.action.diff && operatorState.action.expected_impact && operatorState.action.policy_decision && operatorState.action.risk && operatorState.action.rollback_note) &&
        Boolean(operatorState.dryRun.dry_run_id && operatorState.dryRun.expected_impact && operatorState.dryRun.expected_impact.real_external_http_calls === 0) &&
        Boolean(operatorState.approvals.claim === "approved" && operatorState.approvals.action === "approved") &&
        Boolean(operatorState.execution.mock_connector_calls === 1 && operatorState.execution.real_external_http_calls === 0) &&
        Boolean(operatorState.audit.verification_status === "success" && requiredEvents.every((eventType) => operatorState.audit.event_types.includes(eventType)));
    }}
    window.__cornerstoneOperatorEvidence = function() {{
      return {{
        schema_version: "cs.operator_ui_state.v0",
        completed_steps: operatorState.completedSteps,
        current_step: document.getElementById("vs0-evux-loop").dataset.currentStep,
        production_release_claimed: document.getElementById("vs0-evux-loop").dataset.productionReleaseClaimed === "true",
        live_connector_claimed: document.getElementById("vs0-evux-loop").dataset.liveConnectorClaimed === "true",
        human_acceptance_claimed: document.getElementById("vs0-evux-loop").dataset.humanAcceptanceClaimed === "true",
        local_only_disclaimer: document.getElementById("ui-not-production-ready").textContent,
        operator_passes: operatorPasses(),
        state: operatorState
      }};
    }};
    async function runEvux() {{
      const section = document.getElementById("vs0-evux-loop");
      const status = document.getElementById("evux-status");
      const button = document.getElementById("run-evux");
      section.dataset.evuxClicked = "true";
      button.disabled = true;
      status.dataset.evuxStatus = "running";
      status.textContent = "running";
      evuxTrace.length = 0;
      try {{
        await selectArtifactStep();
        await searchStep();
        await evidenceStep();
        await claimStep();
        await actionStep();
        await dryRunStep();
        await approveStep();
        await executeStep();
        await auditStep();
        const passed = operatorPasses();
        status.dataset.evuxStatus = passed ? "passed" : "failed";
        status.textContent = passed ? "passed" : "failed";
        section.dataset.operatorFlowComplete = passed ? "true" : "false";
      }} catch (error) {{
        traceStep("browser_error", {{ message: String(error) }});
        status.dataset.evuxStatus = "failed";
        status.textContent = "failed";
      }} finally {{
        button.disabled = false;
      }}
    }}
    document.getElementById("step-artifact-run").addEventListener("click", selectArtifactStep);
    document.getElementById("step-search-run").addEventListener("click", searchStep);
    document.getElementById("step-evidence-run").addEventListener("click", evidenceStep);
    document.getElementById("step-claim-run").addEventListener("click", claimStep);
    document.getElementById("step-action-run").addEventListener("click", actionStep);
    document.getElementById("step-dry-run").addEventListener("click", dryRunStep);
    document.getElementById("step-approve-run").addEventListener("click", approveStep);
    document.getElementById("step-execute-run").addEventListener("click", executeStep);
    document.getElementById("step-audit-run").addEventListener("click", auditStep);
    document.getElementById("run-evux").addEventListener("click", runEvux);
    document.getElementById("run-vs1-ontology").addEventListener("click", runVs1Ontology);
    if (evuxAutorun) {{
      window.addEventListener("load", () => setTimeout(() => document.getElementById("run-evux").click(), 50));
    }}
    if (vs1Autorun) {{
      window.addEventListener("load", () => setTimeout(() => document.getElementById("run-vs1-ontology").click(), 50));
    }}
  </script>
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
            query = parse_qs(urlparse(self.path).query)
            scenario = query.get("scenario", [None])[-1]
            autorun = query.get("autorun", ["false"])[-1].lower() in {"1", "true", "yes"}
            self._send_html(render_home(readiness, scenario=scenario, autorun_evux=autorun))
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
        if len(parts) == 3 and parts[0] == "ontology" and parts[1] == "objects":
            self._show_ontology_object(parts[2])
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
        if parts == ["ontology", "suggestion-sets"]:
            self._create_ontology_suggestion_set(body)
            return
        if len(parts) == 4 and parts[0] == "ontology" and parts[1] == "suggestion-sets" and parts[3] == "review":
            self._review_ontology_suggestion_set(parts[2], body)
            return
        if len(parts) == 4 and parts[0] == "ontology" and parts[1] == "suggestion-sets" and parts[3] == "promote":
            self._promote_ontology_suggestion_set(parts[2], body)
            return
        if parts == ["ontology", "draft-truth-test"]:
            self._draft_ontology_truth_test(body)
            return
        if parts == ["ontology", "invalid-graph-test"]:
            self._invalid_ontology_graph_test(body)
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

    def _ontology_error(self, result: dict[str, Any], *, default_code: str = "CS_ONTOLOGY_ERROR") -> bool:
        status = result.get("status")
        if not status:
            return False
        if status == "not_found":
            self._send_json(_json_response("failed", errors=[{"code": "CS_ONTOLOGY_NOT_FOUND", "message": "Ontology source was not found.", "resource": result.get("resource")}]), 404)
            return True
        if status == "scope_denied":
            self._send_json(_json_response("denied", errors=[{"code": "CS_SCOPE_DENIED", "message": "Ontology source is outside the requested scope.", "resource_scope": result.get("resource_scope")}]), 403)
            return True
        if status == "policy_denied":
            refs = []
            if result.get("audit_event"):
                refs.append(f"audit:{result['audit_event']['event_id']}")
            self._send_json(
                _json_response(
                    "denied",
                    policy_decisions=[result.get("policy_decision")],
                    policy_decision_refs=[f"policy:{result.get('policy_decision', {}).get('id')}"] if result.get("policy_decision", {}).get("id") else [],
                    audit_refs=refs,
                    errors=[{"code": "CS_ONTOLOGY_POLICY_DENIED", "message": result.get("reason") or result.get("policy_decision", {}).get("reason", "Ontology policy denied the operation.")}],
                ),
                403,
            )
            return True
        if status == "invalid_graph":
            refs = []
            if result.get("audit_event"):
                refs.append(f"audit:{result['audit_event']['event_id']}")
            self._send_json(_json_response("failed", audit_refs=refs, errors=[result.get("error") or {"code": "CS_ONTOLOGY_INVALID_GRAPH", "message": "Ontology graph is invalid."}]), 400)
            return True
        self._send_json(_json_response("failed", errors=[{"code": default_code, "message": "Ontology operation failed.", "status": status}]), 400)
        return True

    def _create_ontology_suggestion_set(self, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        result = self.store.create_ontology_suggestion_set(str(body.get("source_type", "")), str(body.get("source_id", "")), scope)
        if self._ontology_error(result):
            return
        suggestion_set = result["suggestion_set"]
        self._send_json(
            _json_response(
                "success",
                ontology_suggestion_set=suggestion_set,
                evidence_refs=[f"ontology_suggestion_set:{suggestion_set['suggestion_set_id']}", *suggestion_set.get("evidence_refs", [])],
                audit_refs=[f"audit:{result['audit_event']['event_id']}"],
            )
        )

    def _review_ontology_suggestion_set(self, suggestion_set_id: str, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        result = self.store.review_ontology_suggestion_set(
            suggestion_set_id,
            scope,
            select=[str(value) for value in body.get("select", []) if isinstance(value, str)],
            reject=[str(value) for value in body.get("reject", []) if isinstance(value, str)],
            defer=[str(value) for value in body.get("defer", []) if isinstance(value, str)],
        )
        if self._ontology_error(result):
            return
        suggestion_set = result["suggestion_set"]
        self._send_json(
            _json_response(
                "success",
                ontology_suggestion_set=suggestion_set,
                evidence_refs=[f"ontology_suggestion_set:{suggestion_set_id}", *suggestion_set.get("evidence_refs", [])],
                audit_refs=[f"audit:{result['audit_event']['event_id']}"],
            )
        )

    def _promote_ontology_suggestion_set(self, suggestion_set_id: str, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        result = self.store.promote_ontology_suggestions(
            suggestion_set_id,
            [str(value) for value in body.get("candidate_ids", []) if isinstance(value, str)],
            scope,
        )
        if self._ontology_error(result):
            return
        change_set = result["ontology_change_set"]
        self._send_json(
            _json_response(
                "success",
                ontology_suggestion_set=result["suggestion_set"],
                ontology_objects=result["ontology_objects"],
                ontology_links=result["ontology_links"],
                ontology_change_set=change_set,
                evidence_refs=[
                    f"ontology_change_set:{change_set['ontology_change_set_id']}",
                    *[f"ontology_object:{obj['ontology_object_id']}" for obj in result["ontology_objects"]],
                    *[f"ontology_link:{link['ontology_link_id']}" for link in result["ontology_links"]],
                    *change_set.get("evidence_refs", []),
                ],
                audit_refs=[f"audit:{event['event_id']}" for event in result.get("audit_events", [])],
            )
        )

    def _show_ontology_object(self, object_id: str) -> None:
        scope = self._query_scope()
        result = self.store.ontology_object_profile(object_id, scope)
        if self._ontology_error(result):
            return
        profile = result["profile"]
        self._send_json(
            _json_response(
                "success",
                ontology_object_profile=profile,
                evidence_refs=[f"ontology_object:{object_id}", *profile.get("evidence_refs", [])],
                audit_refs=profile.get("audit_refs", []),
            )
        )

    def _draft_ontology_truth_test(self, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        result = self.store.deny_draft_ontology_truth_use(
            str(body.get("suggestion_set_id", "")),
            str(body.get("candidate_id", "")),
            scope,
            purpose=str(body.get("purpose", "claim_or_action_truth")),
        )
        if result.get("status") != "policy_denied":
            self._ontology_error(result)
            return
        self._send_json(
            _json_response(
                "denied",
                candidate=result["candidate"],
                policy_decisions=[result["policy_decision"]],
                policy_decision_refs=[f"policy:{result['policy_decision']['id']}"],
                audit_refs=[f"audit:{result['audit_event']['event_id']}"],
                errors=[{"code": "CS_ONTOLOGY_DRAFT_TRUTH_DENIED", "message": result["policy_decision"]["reason"]}],
            ),
            403,
        )

    def _invalid_ontology_graph_test(self, body: dict[str, Any]) -> None:
        scope = _scope_from_body(body)
        result = self.store.reject_invalid_ontology_graph(scope)
        self._send_json(_json_response("failed", audit_refs=[f"audit:{result['audit_event']['event_id']}"], errors=[result["error"]]), 400)

    def _show_artifact(self, artifact_id: str) -> None:
        scope = self._query_scope()
        artifact = self.store.get_artifact(artifact_id, scope)
        if artifact is None:
            self._send_json(_json_response("failed", errors=[{"code": "CS_ARTIFACT_NOT_FOUND", "message": "Artifact not found."}]), 404)
            return
        event = self.store.append_audit("artifact.read", scope, {"type": "artifact", "id": artifact_id}, {"reason": "api_artifact_show"})
        detail = dict(artifact)
        detail["derived_text_preview"] = self.store.derived_text_preview(artifact)
        detail["ontology_context"] = self.store.ontology_context_for_artifact(artifact_id, scope)
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
            ontology_refs = [str(value) for value in body.get("ontology_object_refs", []) if isinstance(value, str)]
            result = self.store.create_claim_from_evidence_bundle(bundle_id, str(body.get("statement", "")), scope, ontology_object_refs=ontology_refs)
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
        refs.extend(claim.get("ontology_context", {}).get("object_refs", []))
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
            ontology_object_refs=[str(value) for value in body.get("ontology_object_refs", []) if isinstance(value, str)],
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
                evidence_refs=[f"action:{card['action_id']}", f"claim:{claim_id}", f"mission:{mission_id}", *card.get("ontology_impact", {}).get("object_refs", [])],
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

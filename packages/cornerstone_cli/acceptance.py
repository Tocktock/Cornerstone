from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request

from cornerstone_cli.product_runtime import UI_SURFACES, make_server


CHROME_CANDIDATES = [
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
]

DEFAULT_ACCEPTANCE_SCENARIO_REPORT = "reports/scenario/vs0-runtime-acceptance-2026-06-11.json"
DEFAULT_PRODUCT_RUNTIME_REPORT = "reports/scenario/vs0-product-runtime-2026-06-11.json"
DEFAULT_BROWSER_PROOF_DIR = "reports/browser/vs0-runtime-acceptance-2026-06-11"
DEFAULT_RELEASE_PACKAGE_DIR = "reports/release/vs0-runtime-acceptance-2026-06-11"
DEFAULT_ACCEPTANCE_REPORT = "docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_REPORT_2026-06-11.md"
DEFAULT_ACCEPTANCE_FREEZE_REPORT = "docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_SCENARIO_FREEZE_REPORT_2026-06-11.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def relative_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def find_chrome() -> Path | None:
    for candidate in CHROME_CANDIDATES:
        if candidate.exists():
            return candidate
    for name in ["google-chrome", "chromium", "chromium-browser", "chrome"]:
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    return None


def _chrome_command(chrome: Path, profile_dir: Path, window_size: str) -> list[str]:
    return [
        str(chrome),
        "--headless=new",
        "--disable-background-networking",
        "--disable-default-apps",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-sync",
        "--hide-scrollbars",
        "--metrics-recording-only",
        "--no-default-browser-check",
        "--no-first-run",
        "--no-sandbox",
        f"--user-data-dir={profile_dir}",
        f"--window-size={window_size}",
    ]


def capture_browser_proof(
    root: Path,
    *,
    state_dir: Path,
    output_dir: Path,
    window_size: str = "1280,900",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome()
    proof_path = output_dir / "browser-proof.json"
    screenshot_path = output_dir / "home.png"
    dom_path = output_dir / "home.dom.html"

    if chrome is None:
        proof = {
            "schema_version": "cs.browser_proof.v0",
            "status": "failed",
            "created_at": utc_now(),
            "browser": None,
            "errors": ["chrome_not_found"],
            "screenshot_path": relative_to_root(root, screenshot_path),
            "dom_path": relative_to_root(root, dom_path),
        }
        write_json(proof_path, proof)
        return proof

    server = make_server(root, state_dir)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    thread_error: list[str] = []

    import threading

    def serve() -> None:
        try:
            server.serve_forever()
        except Exception as error:  # pragma: no cover - defensive thread boundary
            thread_error.append(str(error))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    try:
        with tempfile.TemporaryDirectory(prefix="cornerstone-chrome-profile-") as profile_tmp:
            profile_dir = Path(profile_tmp)
            with request.urlopen(url, timeout=10) as response:
                dom_path.write_text(response.read().decode("utf-8", errors="replace"))
            base = _chrome_command(chrome, profile_dir, window_size)
            screenshot_timeout = False
            try:
                screenshot_result = subprocess.run(
                    [*base, f"--screenshot={screenshot_path}", url],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
            except subprocess.TimeoutExpired as error:
                screenshot_timeout = True
                screenshot_result = subprocess.CompletedProcess(
                    args=error.cmd,
                    returncode=124,
                    stdout=(error.stdout or b"").decode("utf-8", errors="replace")
                    if isinstance(error.stdout, bytes)
                    else (error.stdout or ""),
                    stderr=(error.stderr or b"").decode("utf-8", errors="replace")
                    if isinstance(error.stderr, bytes)
                    else (error.stderr or ""),
                )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    dom = dom_path.read_text() if dom_path.exists() else ""
    surface_presence = {surface: surface in dom for surface in UI_SURFACES}
    readiness_labels = [
        "local_scenario_ready=true",
        "vs0_runtime_ready=true",
        "production_release_ready=false",
        "real_external_http_calls=0",
    ]
    screenshot_exists = screenshot_path.exists() and screenshot_path.stat().st_size > 0
    proof_status = (
        screenshot_exists
        and all(surface_presence.values())
        and all(label in dom for label in readiness_labels)
        and "production_release_ready=true" not in dom
        and not thread_error
    )
    proof = {
        "schema_version": "cs.browser_proof.v0",
        "status": "passed" if proof_status else "failed",
        "created_at": utc_now(),
        "browser": {
            "name": "Google Chrome" if "Google Chrome" in str(chrome) else chrome.name,
            "executable": str(chrome),
            "headless": True,
            "window_size": window_size,
        },
        "url": url,
        "route": "/",
        "screenshot_path": relative_to_root(root, screenshot_path),
        "screenshot_sha256": sha256_file(screenshot_path) if screenshot_exists else None,
        "screenshot_bytes": screenshot_path.stat().st_size if screenshot_exists else 0,
        "dom_path": relative_to_root(root, dom_path),
        "dom_sha256": sha256_file(dom_path) if dom_path.exists() else None,
        "surface_presence": surface_presence,
        "readiness_labels_present": {label: label in dom for label in readiness_labels},
        "production_overclaim_absent": "production_release_ready=true" not in dom,
        "chrome_exit_codes": {
            "screenshot": screenshot_result.returncode,
        },
        "chrome_timeout_after_screenshot": screenshot_timeout,
        "stderr_tail": {
            "screenshot": screenshot_result.stderr.strip().splitlines()[-5:],
        },
        "errors": thread_error,
    }
    write_json(proof_path, proof)
    return proof


def _artifact_entry(root: Path, path: Path, role: str, *, required: bool = True) -> dict[str, Any]:
    exists = path.exists()
    entry: dict[str, Any] = {
        "role": role,
        "path": relative_to_root(root, path),
        "required": required,
        "present": exists,
    }
    if exists and path.is_file():
        entry["sha256"] = sha256_file(path)
        entry["bytes"] = path.stat().st_size
    return entry


def collect_release_evidence(
    root: Path,
    *,
    requested_scope: dict[str, str],
    scope_name: str,
    output_dir: Path,
    scenario_report: Path,
    product_runtime_report: Path,
    browser_proof_dir: Path,
    verification_report: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    walkthrough_path = output_dir / "human-usability-walkthrough.md"
    walkthrough_path.write_text(
        "\n".join(
            [
                "# VS0 Runtime Human Usability Walkthrough",
                "",
                "Status: HUMAN_REQUIRED",
                "Owner: JiYong / Tars",
                "",
                "Use this checklist to accept or reject the local VS0 runtime from an operator perspective.",
                "",
                "## Required Walkthrough",
                "",
                "- [ ] Open the local runtime UI.",
                "- [ ] Confirm Home/Ops Inbox is understandable.",
                "- [ ] Confirm Artifact Viewer makes the original/evidence relationship clear.",
                "- [ ] Confirm Search makes scoped evidence discovery clear.",
                "- [ ] Confirm Claim Builder makes Draft, Evidence-backed, and Approved states clear.",
                "- [ ] Confirm Action Card makes dry-run, policy, risk, approval, and execution boundaries clear.",
                "- [ ] Confirm Audit Detail makes action/evidence history inspectable.",
                "- [ ] Confirm the UI does not imply production release, live-provider readiness, or autonomous external writeback.",
                "",
                "## Acceptance Decision",
                "",
                "- Decision: ACCEPT / REJECT",
                "- Reviewer:",
                "- Date:",
                "- Evidence: screenshots, recording, or issue links",
                "- Notes:",
                "",
            ]
        )
    )
    browser_proof = browser_proof_dir / "browser-proof.json"
    browser_screenshot = browser_proof_dir / "home.png"
    browser_dom = browser_proof_dir / "home.dom.html"
    acceptance_contract = root / "docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_CONTRACT.md"
    acceptance_matrix = root / "docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_MATRIX.csv"
    freeze_report = root / DEFAULT_ACCEPTANCE_FREEZE_REPORT
    verification_report_path = verification_report or root / DEFAULT_ACCEPTANCE_REPORT

    artifacts = [
        _artifact_entry(root, scenario_report, "acceptance_scenario_report"),
        _artifact_entry(root, product_runtime_report, "product_runtime_scenario_report"),
        _artifact_entry(root, browser_proof, "browser_proof_manifest"),
        _artifact_entry(root, browser_screenshot, "browser_screenshot"),
        _artifact_entry(root, browser_dom, "browser_dom_snapshot"),
        _artifact_entry(root, acceptance_contract, "acceptance_contract"),
        _artifact_entry(root, acceptance_matrix, "acceptance_matrix"),
        _artifact_entry(root, freeze_report, "scenario_freeze_report"),
        _artifact_entry(root, verification_report_path, "implementation_report", required=False),
        _artifact_entry(root, root / "README.md", "operator_quickstart"),
        _artifact_entry(root, walkthrough_path, "human_usability_walkthrough_checklist"),
    ]

    missing_required = [entry["path"] for entry in artifacts if entry["required"] and not entry["present"]]
    scenario_data: dict[str, Any] = {}
    product_runtime_data: dict[str, Any] = {}
    browser_data: dict[str, Any] = {}
    if scenario_report.exists():
        try:
            scenario_data = json.loads(scenario_report.read_text())
        except ValueError:
            missing_required.append(relative_to_root(root, scenario_report) + ":invalid_json")
    if product_runtime_report.exists():
        try:
            product_runtime_data = json.loads(product_runtime_report.read_text())
        except ValueError:
            missing_required.append(relative_to_root(root, product_runtime_report) + ":invalid_json")
    if browser_proof.exists():
        try:
            browser_data = json.loads(browser_proof.read_text())
        except ValueError:
            missing_required.append(relative_to_root(root, browser_proof) + ":invalid_json")

    negative = dict(scenario_data.get("negative_evidence") or {})
    negative.setdefault("real_external_http_calls", 0)
    negative.setdefault("production_release_overclaim", 0)
    negative.setdefault("live_connector_claim_without_human_evidence", 0)
    negative.setdefault("human_usability_claim_without_human_evidence", 0)
    negative.setdefault("unqualified_external_calls_in_release_report", 0)

    manifest_base = {
        "schema_version": "cs.release_evidence_package.v0",
        "status": "success" if not missing_required and browser_data.get("status") == "passed" else "failed",
        "scope_name": scope_name,
        "scope": requested_scope,
        "created_at": utc_now(),
        "production_release_ready": False,
        "live_connector_ready": False,
        "human_usability_accepted": False,
        "scenario_report": relative_to_root(root, scenario_report),
        "product_runtime_report": relative_to_root(root, product_runtime_report),
        "browser_proof": relative_to_root(root, browser_proof),
        "artifacts": artifacts,
        "negative_evidence": negative,
        "human_required": [
            {
                "id": "VS0-ACC-H01",
                "status": "HUMAN_REQUIRED",
                "required_evidence": "Approved live ConnectorHub/provider transcript with redaction and audit refs.",
            },
            {
                "id": "VS0-ACC-H02",
                "status": "HUMAN_REQUIRED",
                "required_evidence": "JiYong/Tars usability acceptance note with screenshots/recording or issue list.",
            },
        ],
        "included_summary": {
            "acceptance_status": scenario_data.get("status"),
            "acceptance_summary": scenario_data.get("summary"),
            "product_runtime_status": product_runtime_data.get("status"),
            "product_runtime_summary": product_runtime_data.get("summary"),
            "browser_status": browser_data.get("status"),
        },
        "missing_required": missing_required,
    }
    manifest_id = hashlib.sha256(json.dumps(manifest_base, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    manifest = dict(manifest_base)
    manifest["package_id"] = f"releasepkg_{manifest_id}"
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    command_evidence = {
        "schema_version": "cs.release_command_evidence.v0",
        "commands": [
            "cornerstone scenario verify vs0-product-runtime --output reports/scenario/vs0-product-runtime-2026-06-11.json --json",
            "cornerstone scenario verify vs0-runtime-acceptance --output reports/scenario/vs0-runtime-acceptance-2026-06-11.json --json",
            "cornerstone release evidence collect --scope vs0-runtime-acceptance --json",
            "make verify-vs0-acceptance",
            "make verify-vs0-runtime",
            "make verify-local-fast",
        ],
    }
    command_evidence_path = output_dir / "command-evidence.json"
    write_json(command_evidence_path, command_evidence)

    return {
        "schema_version": "cs.release_evidence_collect_result.v0",
        "status": manifest["status"],
        "package_id": manifest["package_id"],
        "manifest_path": relative_to_root(root, manifest_path),
        "output_dir": relative_to_root(root, output_dir),
        "artifact_count": len(artifacts) + 1,
        "missing_required": missing_required,
        "negative_evidence": negative,
    }

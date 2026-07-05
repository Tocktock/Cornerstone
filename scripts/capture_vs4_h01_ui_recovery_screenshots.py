#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib import parse, request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from cornerstone_cli.acceptance import (  # noqa: E402
    _CDPClient,
    _free_local_port,
    _json_urlopen,
    _launch_cdp_chrome,
    _runtime_eval,
    find_chrome,
    relative_to_root,
    sha256_file,
    utc_now,
    write_json,
)
from cornerstone_cli.product_runtime import DEFAULT_SCOPE, make_server  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "output/playwright/vs4-h01-ui-recovery"
DEFAULT_STATE_DIR = ROOT / "tmp/vs4-h01-ui-recovery-screenshot-state"

FORBIDDEN_PRODUCT_RE = re.compile(
    r"local_scenario_ready=|vs0_runtime_ready=|production_release_ready=|real_external_http_calls=|"
    r"\bVS[0-9]\b|VS[0-9]-|scenario|verifier|human gate|acceptance|walkthrough|"
    r"package path|readiness|browser proof|review packet|extractive_fallback|external_writeback",
    re.IGNORECASE,
)

DESKTOP_ROUTES = [
    {"name": "home-desktop", "route": "/", "surface": "home", "required": ["Evidence-first workspace", "Global search", "Search across saved sources, claims, briefs, and action drafts", "Local workspace", "Receipts required", "Drop anything, or ask what we know", "Drag and drop files or paste notes here", "Paste text source", "Browse files", "Ask the workspace", "Recent items", "Knowledge states", "Suggested next steps", "Recent activity"]},
    {"name": "search-desktop", "route": "/search?q=vendor%20renewal", "surface": "search", "required": ["Search command center", "Search the workspace", "Type filters", "Sort by: keyword match", "Current search controls", "Search mode: local keyword", "Receipt-first results", "Result receipt", "Open result", "What we found", "Suggested follow-ups"]},
    {"name": "artifacts-desktop", "route": "/artifacts", "surface": "artifacts", "required": ["Saved sources", "Collection summary", "Source register"]},
    {"name": "briefs-desktop", "route": "/briefs", "surface": "briefs", "required": ["Brief workspace", "Decision queue", "Brief reading queue", "Review lanes", "visible queue item", "Brief queue", "Source coverage", "Use next", "Brief posture"]},
    {"name": "claims-desktop", "route": "/claims", "surface": "claims", "required": ["Claims that need source support", "Decision queue", "Claim review lanes", "visible queue item", "Evidence-backed lane", "Claim review queue", "Review posture"]},
    {"name": "actions-desktop", "route": "/actions", "surface": "actions", "required": ["Action drafts", "Decision queue", "Action approval lanes", "visible queue item", "Approval lane", "Action preview queue", "Dry-run posture"]},
    {"name": "inbox-desktop", "route": "/inbox", "surface": "inbox", "required": ["Work that needs attention", "Triage summary", "open review items across one queue", "Showing 3 open items", "Owner", "Next actions", "Review item", "Review sources", "Open audit trail"]},
    {"name": "audit-desktop", "route": "/audit", "surface": "audit", "required": ["Audit receipt workspace", "Activity trail", "Audit status", "Receipt summary", "Audit lifecycle", "Activity receipts", "Readable receipts", "Event stream", "Audit posture", "Audit integrity checks", "Integrity chain", "Scope and recovery", "Raw event detail"]},
    {
        "name": "owner-admin-desktop",
        "route": "/review",
        "surface": "owner-review",
        "required": ["Connector governance", "Connected source posture", "Policy controls", "Access roles", "Namespace settings", "Admin containment", "Recent connector activity"],
        "allow_internal_terms": True,
    },
    {
        "name": "reference-gallery-desktop",
        "route": "/review/reference-images",
        "surface": "owner-review",
        "required": ["Reference image gallery", "Implementation boundary", "Vendor object detail", "Operations inbox", "Home workspace", "Action dry-run"],
        "allow_internal_terms": True,
    },
]

DAY_ZERO_ROUTES = [
    {"name": "day-zero-search-desktop", "route": "/search", "surface": "search", "required": ["Search starts with saved work", "Save a source", "Open artifacts", "Startup path", "First receipts"]},
    {"name": "day-zero-artifacts-desktop", "route": "/artifacts", "surface": "artifacts", "required": ["Start with a source", "Go to Home", "Search workspace", "Startup path", "First receipts"]},
    {"name": "day-zero-briefs-desktop", "route": "/briefs", "surface": "briefs", "required": ["Create the first brief", "Save a source", "Open artifacts", "Startup path", "First receipts"]},
    {"name": "day-zero-claims-desktop", "route": "/claims", "surface": "claims", "required": ["No claims need review", "Open briefs", "Check sources", "Startup path", "First receipts"]},
    {"name": "day-zero-actions-desktop", "route": "/actions", "surface": "actions", "required": ["No action previews yet", "Open claims", "Open briefs", "Startup path", "First receipts"]},
    {"name": "day-zero-inbox-desktop", "route": "/inbox", "surface": "inbox", "required": ["No work waiting", "No selected work", "Start from Home", "Startup path", "First receipts"]},
    {"name": "day-zero-audit-desktop", "route": "/audit", "surface": "audit", "required": ["No activity recorded yet", "Start from Home", "Open artifacts", "Startup path", "First receipts"]},
]

NOT_FOUND_ROUTES = [
    {"name": "not-found-page-desktop", "route": "/missing-product-route", "surface": "not-found", "required": ["We could not find that page", "Search workspace", "Useful places"]},
    {"name": "not-found-source-desktop", "route": "/artifacts/missing-source?view=html", "surface": "not-found", "required": ["This source is not available", "Search workspace", "Brief workspace"]},
]

INTERACTION_ROUTES = [
    {
        "name": "home-validation-desktop",
        "route": "/",
        "surface": "home",
        "required": ["Paste text before saving.", "Enter a question first."],
        "interaction": "home-validation",
    },
]

DETAIL_ROUTES = [
    {"name": "artifact-detail-desktop", "kind": "artifacts", "id_key": "artifact_id", "surface": "artifact-detail", "required": ["Source inspection workspace", "Detail path", "Back to saved sources", "Original source", "Original source document viewer", "Original artifact preview", "Source pages", "Preview rail", "Source metadata", "Artifact inspection summary", "Preview mode", "Plain text preview", "Original content primary", "Details", "Tags", "Source state", "Keyword summary", "Extracted keywords", "Provenance", "Open audit trail"]},
    {"name": "brief-detail-desktop", "kind": "briefs", "id_key": "brief_id", "surface": "brief-detail", "required": ["Brief reading workspace", "Detail path", "Back to briefs", "Open audit trail", "Decision snapshot", "What we found", "Findings with citations", "What this brief cannot confirm", "Suggested next steps", "Sources used", "Citation disclosure", "Source snippet", "Full provenance", "Audit trail", "Use this brief"]},
    {
        "name": "claim-detail-desktop",
        "kind": "claims",
        "id_key": "claim_id",
        "surface": "claim-detail",
        "required": ["Detail path", "Back to claims", "Open inbox", "Claim review summary", "Claim statement", "Supporting evidence", "Evidence picker controls", "Sort: source order", "Review controls", "Decision gate", "Source support", "Impacted objects", "Related frameworks", "Saved locally"],
    },
    {"name": "action-detail-desktop", "kind": "actions", "id_key": "action_id", "surface": "action-detail", "required": ["Detail path", "Action preview", "Action review status", "Summary", "Impacted objects", "Dry-run sequence", "Proposed changes", "External calls", "Call preview", "Policy decision", "Policy checkpoints", "Risk and approval", "Request approval", "Approval history"]},
]

MOBILE_ROUTE_NAMES = {
    "actions-desktop",
    "artifact-detail-desktop",
    "artifacts-desktop",
    "audit-desktop",
    "briefs-desktop",
    "claims-desktop",
    "home-desktop",
    "brief-detail-desktop",
    "claim-detail-desktop",
    "search-desktop",
    "inbox-desktop",
    "action-detail-desktop",
    "owner-admin-desktop",
    "reference-gallery-desktop",
}

DAY_ZERO_MOBILE_ROUTE_NAMES = {
    "day-zero-artifacts-desktop",
    "day-zero-briefs-desktop",
    "day-zero-claims-desktop",
    "day-zero-actions-desktop",
    "day-zero-inbox-desktop",
    "day-zero-audit-desktop",
}

NOT_FOUND_MOBILE_ROUTE_NAMES = {
    "not-found-page-desktop",
    "not-found-source-desktop",
}

INTERACTION_MOBILE_ROUTE_NAMES = {
    "home-validation-desktop",
}

LAYOUT_SCRIPT = """
(() => {
  const html = document.documentElement.outerHTML;
  const text = document.body ? document.body.innerText : "";
  const surfaceNode = document.querySelector("[data-product-surface]");
  const hero = document.querySelector("[data-product-surface='home'] h1");
  const homeSurface = document.querySelector("[data-product-surface='home']");
  const dropForm = document.querySelector("#cs-drop-form");
  const askForm = document.querySelector("#cs-ask-form");
  const nav = document.querySelector(".cs-sidebar");
  const rect = (node) => {
    if (!node) return null;
    const r = node.getBoundingClientRect();
    return {top: r.top, left: r.left, width: r.width, height: r.height};
  };
  const dropRect = rect(dropForm);
  const askRect = rect(askForm);
  const isHome = Boolean(homeSurface);
  return {
    title: document.title,
    body_text_length: text.length,
    body_html_length: html.length,
    product_shell_present: Boolean(document.querySelector("[data-product-shell='cornerstone']")),
    surface: surfaceNode ? surfaceNode.getAttribute("data-product-surface") : "",
    horizontal_overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
    document_scroll_width: document.documentElement.scrollWidth,
    document_scroll_height: document.documentElement.scrollHeight,
    inner_width: window.innerWidth,
    inner_height: window.innerHeight,
    mobile_breakpoint_applied: window.innerWidth <= 980
      ? window.getComputedStyle(document.querySelector(".cs-shell")).gridTemplateColumns.split(" ").length === 1
      : false,
    mobile_first_value_before_nav: window.innerWidth <= 980 && hero && nav ? rect(hero).top <= rect(nav).top : true,
    home_drop_ask_ordered: isHome && dropRect && askRect ? dropRect.top < askRect.top : true,
    home_drop_ask_in_first_viewport: isHome && dropRect && askRect
      ? dropRect.top < window.innerHeight && askRect.top < window.innerHeight
      : true,
    token_css_present: [
      "--cs-color-background-app:",
      "--cs-layout-sidebarWidth:",
      "--cs-state-saved-bg:",
      "--cs-radius-pill:"
    ].every((token) => html.includes(token)),
  };
})()
"""

HOME_VALIDATION_SCRIPT = """
(() => {
  const dropForm = document.querySelector("#cs-drop-form");
  const askForm = document.querySelector("#cs-ask-form");
  if (dropForm) {
    dropForm.dispatchEvent(new Event("submit", {bubbles: true, cancelable: true}));
  }
  if (askForm) {
    askForm.dispatchEvent(new Event("submit", {bubbles: true, cancelable: true}));
  }
  const dropStatus = document.querySelector("#cs-drop-status");
  const askStatus = document.querySelector("#cs-ask-status");
  return {
    drop_state: dropStatus ? dropStatus.getAttribute("data-state") : "",
    ask_state: askStatus ? askStatus.getAttribute("data-state") : "",
    drop_text: dropStatus ? dropStatus.textContent : "",
    ask_text: askStatus ? askStatus.textContent : "",
    passed: Boolean(dropStatus && askStatus) &&
      dropStatus.getAttribute("data-state") === "error" &&
      askStatus.getAttribute("data-state") === "error" &&
      dropStatus.textContent.includes("Paste text before saving.") &&
      askStatus.textContent.includes("Enter a question first.")
  };
})()
"""

READY_SCRIPT = """
(() => {
  const surfaceNode = document.querySelector("[data-product-surface]");
  return {
    readyState: document.readyState,
    productShellPresent: Boolean(document.querySelector("[data-product-shell='cornerstone']")),
    surface: surfaceNode ? surfaceNode.getAttribute("data-product-surface") : "",
  };
})()
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture the VS4-H01 UI recovery screenshot pack.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--json", action="store_true", help="Print the manifest JSON to stdout.")
    parser.add_argument("--keep-state", action="store_true", help="Do not clear the screenshot fixture state directory first.")
    return parser.parse_args()


def normalize_captured_dom(html: str) -> str:
    return "\n".join(line.rstrip() for line in html.splitlines()) + "\n"


def http_request(base_url: str, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    req = request.Request(base_url + path, data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=10) as response:
        body = response.read().decode("utf-8")
        return {
            "status": response.status,
            "content_type": response.headers.get("content-type", ""),
            "body": body,
        }


def json_payload(response: dict[str, Any]) -> dict[str, Any]:
    return json.loads(str(response.get("body") or "{}"))


def create_fixture_stack(base_url: str) -> dict[str, str]:
    scope = dict(DEFAULT_SCOPE)
    source_text = (
        "Vendor renewal note: the renewal date is August 1, finance flagged a price increase, "
        "security asked for source support, and the decision owner needs a brief before any follow-up."
    )
    artifact_payload = json_payload(
        http_request(
            base_url,
            "/artifacts",
            method="POST",
            payload={**scope, "text": source_text, "source": {"type": "user_paste", "ref": "vs4-h01-screenshot-pack"}},
        )
    )
    artifact_id = str((artifact_payload.get("artifact") or {}).get("artifact_id") or "")
    search_payload = json_payload(http_request(base_url, "/search", method="POST", payload={**scope, "query": "vendor renewal"}))
    search_snapshot_id = str((search_payload.get("search_snapshot") or {}).get("search_snapshot_id") or "")
    bundle_payload = json_payload(
        http_request(base_url, "/evidence-bundles", method="POST", payload={**scope, "search_snapshot_id": search_snapshot_id})
    )
    evidence_bundle_id = str((bundle_payload.get("evidence_bundle") or {}).get("evidence_bundle_id") or "")
    brief_payload = json_payload(
        http_request(base_url, "/briefs", method="POST", payload={**scope, "evidence_bundle_id": evidence_bundle_id})
    )
    brief_id = str((brief_payload.get("brief") or {}).get("brief_id") or "")
    claim_payload = json_payload(
        http_request(
            base_url,
            "/claims",
            method="POST",
            payload={
                **scope,
                "evidence_bundle_id": evidence_bundle_id,
                "statement": "Vendor renewal needs a decision before the August 1 renewal date.",
            },
        )
    )
    claim_id = str((claim_payload.get("claim") or {}).get("claim_id") or "")
    action_payload = json_payload(
        http_request(
            base_url,
            "/actions",
            method="POST",
            payload={
                **scope,
                "claim_id": claim_id,
                "goal": "Draft a vendor renewal follow-up",
                "action_kind": "external_writeback",
                "risk": "medium",
                "target": "planner task for vendor renewal follow-up",
            },
        )
    )
    action_id = str((action_payload.get("action_card") or {}).get("action_id") or "")
    return {
        "artifact_id": artifact_id,
        "search_snapshot_id": search_snapshot_id,
        "evidence_bundle_id": evidence_bundle_id,
        "brief_id": brief_id,
        "claim_id": claim_id,
        "action_id": action_id,
    }


def route_specs(ids: dict[str, str]) -> list[dict[str, Any]]:
    specs = [dict(spec) for spec in DESKTOP_ROUTES]
    for spec in DETAIL_ROUTES:
        item = dict(spec)
        item["route"] = f"/{item.pop('kind')}/{parse.quote(ids[item.pop('id_key')])}?view=html"
        specs.append(item)
    mobile_specs = []
    for spec in specs:
        if spec["name"] in MOBILE_ROUTE_NAMES:
            item = dict(spec)
            item["name"] = item["name"].replace("-desktop", "-mobile")
            item["mobile"] = True
            mobile_specs.append(item)
    return specs + mobile_specs


def day_zero_route_specs() -> list[dict[str, Any]]:
    specs = [dict(spec) for spec in DAY_ZERO_ROUTES]
    mobile_specs = []
    for spec in specs:
        if spec["name"] in DAY_ZERO_MOBILE_ROUTE_NAMES:
            item = dict(spec)
            item["name"] = item["name"].replace("-desktop", "-mobile")
            item["mobile"] = True
            mobile_specs.append(item)
    return specs + mobile_specs


def not_found_route_specs() -> list[dict[str, Any]]:
    specs = [dict(spec) for spec in NOT_FOUND_ROUTES]
    mobile_specs = []
    for spec in specs:
        if spec["name"] in NOT_FOUND_MOBILE_ROUTE_NAMES:
            item = dict(spec)
            item["name"] = item["name"].replace("-desktop", "-mobile")
            item["mobile"] = True
            mobile_specs.append(item)
    return specs + mobile_specs


def interaction_route_specs() -> list[dict[str, Any]]:
    specs = [dict(spec) for spec in INTERACTION_ROUTES]
    mobile_specs = []
    for spec in specs:
        if spec["name"] in INTERACTION_MOBILE_ROUTE_NAMES:
            item = dict(spec)
            item["name"] = item["name"].replace("-desktop", "-mobile")
            item["mobile"] = True
            mobile_specs.append(item)
    return specs + mobile_specs


def capture_page(
    *,
    chrome: Path,
    base_url: str,
    spec: dict[str, Any],
    output_dir: Path,
    window_size: str,
) -> dict[str, Any]:
    screenshots_dir = output_dir / "screenshots"
    dom_dir = output_dir / "dom"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    dom_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = screenshots_dir / f"{spec['name']}.png"
    dom_path = dom_dir / f"{spec['name']}.html"
    profile_dir = Path(tempfile.mkdtemp(prefix="cornerstone-vs4-shot-"))
    debug_port = _free_local_port()
    url = base_url + str(spec["route"])
    process: subprocess.Popen[str] | None = None
    browser: _CDPClient | None = None
    page: _CDPClient | None = None
    error: str | None = None
    exit_code: int | None = None
    layout: dict[str, Any] = {}
    interaction: dict[str, Any] = {}
    try:
        process = _launch_cdp_chrome(chrome, profile_dir, debug_port, window_size)
        version = None
        for _ in range(80):
            try:
                version = _json_urlopen(f"http://127.0.0.1:{debug_port}/json/version")
                break
            except OSError:
                time.sleep(0.1)
        if version is None:
            raise TimeoutError("devtools_version")
        browser = _CDPClient(str(version["webSocketDebuggerUrl"]))
        new_target_req = request.Request(
            f"http://127.0.0.1:{debug_port}/json/new?{parse.quote(url, safe=':/?=&%')}",
            method="PUT",
        )
        with request.urlopen(new_target_req, timeout=5) as response:
            target = json.loads(response.read().decode("utf-8"))
        page = _CDPClient(str(target["webSocketDebuggerUrl"]))
        page.command("Page.enable")
        page.command("Runtime.enable")
        page.command("Page.navigate", {"url": url})
        try:
            page.wait_event("Page.loadEventFired", timeout=10)
        except TimeoutError:
            pass
        ready_state: dict[str, Any] = {}
        for _ in range(100):
            ready_state = _runtime_eval(page, READY_SCRIPT, timeout=5) or {}
            if (
                ready_state.get("readyState") == "complete"
                and ready_state.get("productShellPresent") is True
                and ready_state.get("surface") == spec["surface"]
            ):
                break
            time.sleep(0.1)
        if spec.get("interaction") == "home-validation":
            interaction = _runtime_eval(page, HOME_VALIDATION_SCRIPT, timeout=10) or {}
        layout = _runtime_eval(page, LAYOUT_SCRIPT, timeout=10) or {}
        dom = normalize_captured_dom(str(_runtime_eval(page, "document.documentElement.outerHTML", timeout=5) or ""))
        dom_path.write_text(dom)
        screenshot = page.command("Page.captureScreenshot", {"format": "png", "fromSurface": True}, timeout=10)
        screenshot_path.write_bytes(base64.b64decode(str(screenshot.get("data", ""))))
        try:
            browser.command("Browser.close", timeout=5)
        except Exception:
            page.command("Browser.close", timeout=5)
        exit_code = process.wait(timeout=10)
    except Exception as exc:  # pragma: no cover - operational evidence path
        error = str(exc)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        if process:
            exit_code = process.returncode
    finally:
        if page is not None:
            page.close()
        if browser is not None:
            browser.close()
        shutil.rmtree(profile_dir, ignore_errors=True)

    dom_text = dom_path.read_text() if dom_path.exists() else ""
    forbidden = [] if spec.get("allow_internal_terms") else sorted(set(match.group(0) for match in FORBIDDEN_PRODUCT_RE.finditer(dom_text)))
    required_missing = [text for text in spec.get("required", []) if text not in dom_text]
    screenshot_bytes = screenshot_path.stat().st_size if screenshot_path.exists() else 0
    checks = {
        "screenshot_present": screenshot_bytes > 0,
        "dom_present": bool(dom_text),
        "product_shell_present": layout.get("product_shell_present") is True,
        "surface_matches": layout.get("surface") == spec["surface"],
        "no_horizontal_overflow": layout.get("horizontal_overflow") is False,
        "token_css_present": layout.get("token_css_present") is True,
        "required_text_present": not required_missing,
        "forbidden_terms_absent": not forbidden,
        "mobile_first_value_before_nav": layout.get("mobile_first_value_before_nav") is True,
        "home_drop_ask_ordered": layout.get("home_drop_ask_ordered") is True,
        "home_drop_ask_in_first_viewport": layout.get("home_drop_ask_in_first_viewport") is True,
        "interaction_passed": not spec.get("interaction") or interaction.get("passed") is True,
        "clean_browser_exit": exit_code == 0,
        "no_capture_error": error is None,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "name": spec["name"],
        "route": spec["route"],
        "surface": spec["surface"],
        "viewport": "mobile" if spec.get("mobile") else "desktop",
        "window_size": window_size,
        "url": url,
        "status": status,
        "checks": checks,
        "required_missing": required_missing,
        "forbidden_terms": forbidden,
        "layout": layout,
        "interaction": interaction,
        "screenshot_path": relative_to_root(ROOT, screenshot_path),
        "screenshot_bytes": screenshot_bytes,
        "screenshot_sha256": sha256_file(screenshot_path) if screenshot_bytes else None,
        "dom_path": relative_to_root(ROOT, dom_path),
        "dom_sha256": sha256_file(dom_path) if dom_path.exists() else None,
        "chrome_exit_code": exit_code,
        "error": error,
    }


def build_owner_package(output_dir: Path, manifest: dict[str, Any]) -> None:
    desktop_count = sum(1 for page in manifest["pages"] if page["viewport"] == "desktop")
    mobile_count = sum(1 for page in manifest["pages"] if page["viewport"] == "mobile")
    lines = [
        "# VS4-H01 UI Recovery Owner Review Package",
        "",
        f"Date: {manifest['created_at']}",
        "",
        "## Scope",
        "",
        "This package covers the current verified recovery slice for the rejected VS4-H01 UI:",
        "",
        "- R0: token-to-CSS pipeline, shared server-rendered product shell, reusable render helpers, real HTML routes, and language mapping.",
        "- R1: Home rebuilt around Drop and Ask with real local records, day-zero copy, and internal owner material moved to `/review`.",
        "- R2/R3: Search, Artifacts, Briefs, Claims, Actions, Inbox, Audit, owner connector governance, and record detail routes are represented in the screenshot pack.",
        "- R4: product surfaces are scanned for forbidden internal language and raw runtime labels.",
        "- R5: desktop, mobile, day-zero, not-found, and Home validation captures check horizontal overflow, mobile first-value ordering, and inline interaction states.",
        "- R6: screenshot pack and automated checks exist; owner acceptance remains human-required.",
        "",
        "## Evidence Files",
        "",
        f"- `screenshot-pack-manifest.json`",
        f"- `screenshots/` ({desktop_count} desktop, {mobile_count} mobile captures, including day-zero, not-found, and Home validation states)",
        "- `dom/` captured HTML for each screenshot route",
        "",
        "## Screenshot Coverage",
        "",
    ]
    for page in manifest["pages"]:
        lines.append(
            f"- `{page['screenshot_path']}`: {page['status']} / `{page['route']}` / {page['viewport']}"
        )
    lines.extend(
        [
            "",
            "## Checks Run",
            "",
            "- `python3 scripts/capture_vs4_h01_ui_recovery_screenshots.py --json`",
            f"  - Result: {manifest['status']}; {manifest['summary']['pass']} pass, {manifest['summary']['fail']} fail.",
            "",
            "## Companion Checks Before Owner Review",
            "",
            "- `python3 -m unittest tests.scenario.test_product_ui_routes`",
            "  - Purpose: focused route regression check.",
            "- `make verify-vs4-product-alpha-shell`",
            "  - Purpose: filtered scenario gate for shell proof.",
            "",
            "## Human-Required Gates",
            "",
            "- Owner subjective UI/UX acceptance remains not claimed.",
            "- VS5 external-user readiness remains not claimed.",
            "- The proposed Brief surface direction still needs owner acceptance because the redesign guidance notes no prior reference image exists for that centerpiece surface.",
            "",
            "## Known Follow-Up",
            "",
            "- Clean up the now-unreachable legacy VS4 browser-proof branch after the new multi-page route-scan proof has settled.",
            "- Add real interaction captures for Drop, Ask, lane switching, and action approval preview before retrying VS4-H01 owner acceptance.",
            "",
        ]
    )
    (output_dir / "OWNER_REVIEW_PACKAGE.md").write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    state_dir: Path = args.state_dir
    if not args.keep_state and state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome()
    if chrome is None:
        print("No supported Chrome executable was found.", file=sys.stderr)
        return 2

    server = make_server(ROOT, state_dir)
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    thread_error: list[str] = []

    def serve() -> None:
        try:
            server.serve_forever()
        except Exception as exc:  # pragma: no cover - operational evidence path
            thread_error.append(str(exc))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        pages = []
        for spec in day_zero_route_specs():
            window_size = "390,844" if spec.get("mobile") else "1440,1100"
            pages.append(capture_page(chrome=chrome, base_url=base_url, spec=spec, output_dir=output_dir, window_size=window_size))
        for spec in not_found_route_specs():
            window_size = "390,844" if spec.get("mobile") else "1440,1100"
            pages.append(capture_page(chrome=chrome, base_url=base_url, spec=spec, output_dir=output_dir, window_size=window_size))
        for spec in interaction_route_specs():
            window_size = "390,844" if spec.get("mobile") else "1440,1100"
            pages.append(capture_page(chrome=chrome, base_url=base_url, spec=spec, output_dir=output_dir, window_size=window_size))
        ids = create_fixture_stack(base_url)
        for spec in route_specs(ids):
            window_size = "390,844" if spec.get("mobile") else "1440,1100"
            pages.append(capture_page(chrome=chrome, base_url=base_url, spec=spec, output_dir=output_dir, window_size=window_size))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    failed_pages = [page for page in pages if page["status"] != "PASS"]
    manifest = {
        "schema_version": "cs.vs4_h01_ui_recovery_screenshot_pack.v1",
        "status": "PASS" if not failed_pages and not thread_error else "FAIL",
        "created_at": utc_now(),
        "output_dir": relative_to_root(ROOT, output_dir),
        "state_dir": relative_to_root(ROOT, state_dir),
        "base_url": base_url,
        "fixture_ids": ids,
        "summary": {
            "page_count": len(pages),
            "pass": len(pages) - len(failed_pages),
            "fail": len(failed_pages),
            "desktop": sum(1 for page in pages if page["viewport"] == "desktop"),
            "mobile": sum(1 for page in pages if page["viewport"] == "mobile"),
        },
        "negative_evidence": {
            "blank_screenshots": sum(1 for page in pages if not page["checks"]["screenshot_present"]),
            "horizontal_overflow_pages": sum(1 for page in pages if not page["checks"]["no_horizontal_overflow"]),
            "forbidden_lexicon_pages": sum(1 for page in pages if not page["checks"]["forbidden_terms_absent"]),
            "product_shell_missing_pages": sum(1 for page in pages if not page["checks"]["product_shell_present"]),
            "surface_mismatch_pages": sum(1 for page in pages if not page["checks"]["surface_matches"]),
            "owner_acceptance_claimed": 0,
            "production_readiness_claimed": 0,
        },
        "thread_errors": thread_error,
        "pages": pages,
    }
    write_json(output_dir / "screenshot-pack-manifest.json", manifest)
    build_owner_package(output_dir, manifest)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"{manifest['status']} {manifest['summary']}")
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

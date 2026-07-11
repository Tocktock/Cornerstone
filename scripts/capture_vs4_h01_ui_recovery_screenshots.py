#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
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

PRODUCT_RUNTIME_SOURCES = (
    ROOT / "packages/cornerstone_cli/__init__.py",
    ROOT / "packages/cornerstone_cli/acceptance.py",
    ROOT / "packages/cornerstone_cli/artifacts.py",
    ROOT / "packages/cornerstone_cli/briefing.py",
    ROOT / "packages/cornerstone_cli/product_access.py",
    ROOT / "packages/cornerstone_cli/product_runtime.py",
    ROOT / "packages/cornerstone_cli/product_ui.py",
    ROOT / "packages/cornerstone_cli/product_visibility.py",
    ROOT / "packages/cornerstone_cli/runtime.py",
    ROOT / "packages/cornerstone_cli/validators.py",
)
UI_ASSET_ROOT = ROOT / "packages/cornerstone_cli/ui"
REFERENCE_IMAGE_ROOT = ROOT / "docs/design/reference-images"
DESIGN_TOKEN_PATH = ROOT / "docs/design/tokens/cornerstone_design_tokens_v0_3.json"
DEVTOOLS_READY_TIMEOUT_SECONDS = 20.0

FORBIDDEN_PRODUCT_RE = re.compile(
    r"local_scenario_ready=|vs0_runtime_ready=|production_release_ready=|real_external_http_calls=|"
    r"\bVS[0-9]\b|VS[0-9]-|scenario|verifier|human gate|acceptance|walkthrough|"
    r"package path|readiness|browser proof|review packet|extractive_fallback|external_writeback",
    re.IGNORECASE,
)

DESKTOP_ROUTES = [
    {
        "name": "home-desktop",
        "route": "/",
        "surface": "home",
        "required_text": ["Drop anything, or ask what we know", "Drop a file or paste notes", "Ask the workspace", "Recent items"],
        "required_selectors": [
            "[data-product-shell='cornerstone'][data-workspace-id]",
            ".cs-workspace-switcher[aria-label='Open local workspace settings']",
            "form[role='search'][aria-label='Global search']",
            "#cs-drop-form #cs-drop-text",
            "#cs-drop-form #cs-save-source-button",
            "#cs-ask-form #cs-ask-input",
            "#cs-ask-form #cs-ask-submit-button",
        ],
    },
    {
        "name": "search-desktop",
        "route": "/search?q=vendor%20renewal",
        "surface": "search",
        "required_text": ["Results for", "Keyword match"],
        "required_selectors": [
            "[data-product-surface='search']",
            ".cs-search-tabs[aria-label='Filter results by record type']",
            ".cs-result-list[aria-live='polite']",
            ".cs-result-row",
        ],
    },
    {
        "name": "artifacts-desktop",
        "route": "/artifacts",
        "surface": "artifacts",
        "required_text": ["Saved sources", "Collection summary", "Untrusted until checked"],
        "required_selectors": ["[data-product-surface='artifacts']", ".cs-collection-workbench", ".cs-collection-list"],
    },
    {
        "name": "briefs-desktop",
        "route": "/briefs",
        "surface": "briefs",
        "required_text": ["Brief workspace", "Source coverage", "Use next"],
        "required_selectors": ["[data-product-surface='briefs']", ".cs-collection-workbench", ".cs-collection-list"],
    },
    {
        "name": "claims-desktop",
        "route": "/claims",
        "surface": "claims",
        "required_text": ["Claims under review", "Semantic review needed", "Review posture", "Trust ladder"],
        "required_selectors": ["[data-product-surface='claims']", ".cs-collection-workbench", ".cs-collection-list"],
    },
    {
        "name": "actions-desktop",
        "route": "/actions",
        "surface": "actions",
        "required_text": ["Action records", "Dry-run posture", "Action safeguards"],
        "required_selectors": ["[data-product-surface='actions']", ".cs-collection-workbench", ".cs-collection-list"],
    },
    {
        "name": "inbox-desktop",
        "route": "/inbox",
        "surface": "inbox",
        "required_text": ["Work that needs attention", "Selected item", "Continue review", "Related journey"],
        "required_selectors": [
            "[data-product-surface='inbox'][data-inbox-lane][data-selected-item]",
            ".cs-inbox-tabs[aria-label='Review lanes']",
            ".cs-inbox-table[role='list']",
            "#selected-work",
        ],
    },
    {
        "name": "audit-desktop",
        "route": "/audit",
        "surface": "audit",
        "required_text": ["History", "Workspace history", "Ledger integrity", "Apply filters"],
        "required_selectors": [
            "[data-product-surface='audit'][data-audit-integrity-status]",
            "form.cs-audit-filters[aria-label='Filter history']",
            "form.cs-audit-filters [name='record']",
            "form.cs-audit-filters [name='lifecycle']",
            ".cs-audit-list",
        ],
    },
    {
        "name": "owner-admin-desktop",
        "route": "/review",
        "surface": "owner-review",
        "required_text": ["Connector governance console", "Connected source posture", "Policy controls", "Namespace settings", "Admin containment"],
        "required_selectors": ["[data-product-surface='owner-review']", ".cs-owner-overview[aria-label='Admin containment']"],
        "allow_internal_terms": True,
    },
    {
        "name": "reference-gallery-desktop",
        "route": "/review/reference-images",
        "surface": "owner-review",
        "required_text": ["Reference image gallery", "Implementation boundary"],
        "required_selectors": ["[data-product-surface='owner-review']", ".cs-reference-grid[aria-label='CornerStone UI reference images']"],
        "allow_internal_terms": True,
    },
]

DAY_ZERO_ROUTES = [
    {"name": "day-zero-search-desktop", "route": "/search", "surface": "search", "required_text": ["Search starts with saved work", "Save a source", "Open artifacts", "Startup path", "What will appear"]},
    {"name": "day-zero-artifacts-desktop", "route": "/artifacts", "surface": "artifacts", "required_text": ["Start with a source", "Go to Home", "Search workspace", "Startup path", "What will appear"]},
    {"name": "day-zero-briefs-desktop", "route": "/briefs", "surface": "briefs", "required_text": ["Create the first brief", "Save a source", "Open artifacts", "Startup path", "What will appear"]},
    {"name": "day-zero-claims-desktop", "route": "/claims", "surface": "claims", "required_text": ["No claims need review", "Open briefs", "Check sources", "Startup path", "What will appear"]},
    {"name": "day-zero-actions-desktop", "route": "/actions", "surface": "actions", "required_text": ["No action previews yet", "Open claims", "Open briefs", "Startup path", "What will appear"]},
    {"name": "day-zero-inbox-desktop", "route": "/inbox", "surface": "inbox", "required_text": ["No work waiting", "No selected work", "Start from Home", "Startup path", "What will appear"]},
    {"name": "day-zero-audit-desktop", "route": "/audit", "surface": "audit", "required_text": ["History", "Workspace history", "Page opened", "Ledger integrity", "full local ledger"]},
]

NOT_FOUND_ROUTES = [
    {"name": "not-found-page-desktop", "route": "/missing-product-route", "surface": "not-found", "required_text": ["We could not find that page", "Search workspace", "Useful places"]},
    {"name": "not-found-source-desktop", "route": "/artifacts/missing-source?view=html", "surface": "not-found", "required_text": ["This source is not available", "Search workspace", "Brief workspace"]},
]

INTERACTION_ROUTES = [
    {
        "name": "home-validation-desktop",
        "route": "/",
        "surface": "home",
        "required_text": ["Paste text before saving.", "Enter a question first."],
        "interaction": "home-validation",
    },
]

DETAIL_ROUTES = [
    {
        "name": "artifact-detail-desktop",
        "kind": "artifacts",
        "id_key": "artifact_id",
        "surface": "artifact-detail",
        "required_text": ["Source text", "Source details", "Frequent local terms", "Linked work", "Open source history"],
        "required_selectors": [
            "[data-product-surface='artifact-detail'][aria-label='Source inspection workspace']",
            "[aria-label='Original source document viewer']",
            "#source-text .cs-source-text",
            "#keywords",
        ],
    },
    {
        "name": "brief-detail-desktop",
        "kind": "briefs",
        "id_key": "brief_id",
        "surface": "brief-detail",
        "required_text": ["What we know", "Findings with citations", "Sources used", "Citation checks", "Provenance"],
        "required_selectors": [
            "[data-product-surface='brief-detail'][data-citation-check-refs-count][data-resolved-citation-count]",
            "#brief-answer-title",
            "#citation-trail",
            "details.cs-citation-checks",
        ],
    },
    {
        "name": "claim-detail-desktop",
        "kind": "claims",
        "id_key": "claim_id",
        "surface": "claim-detail",
        "required_text": ["Decision statement", "Supporting evidence", "Authority", "Brief lineage and gaps"],
        "required_selectors": [
            "[data-product-surface='claim-detail'][data-source-support-attached][data-evidence-backed-earned][data-approval-eligible]",
            ".cs-claim-statement",
            "[data-claim-approval-state]",
        ],
    },
    {
        "name": "action-detail-desktop",
        "kind": "actions",
        "id_key": "action_id",
        "surface": "action-detail",
        "required_text": ["Blocked action", "Action blocked", "Why this action", "Policy and boundary"],
        "required_selectors": [
            "[data-product-surface='action-detail'][data-execution-mode][data-real-external-http-calls]",
            "[data-action-policy-blocked='true']",
            "[data-action-approval-state]",
        ],
    },
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
    ].every((token) => getComputedStyle(document.documentElement)
      .getPropertyValue(token.slice(0, -1)).trim().length > 0),
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
    parser.add_argument(
        "--validate-manifest",
        type=Path,
        help="Validate an existing screenshot-pack manifest without running a new capture.",
    )
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


def source_revision() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status_lines = [line for line in git("status", "--porcelain=v1").splitlines() if line]
    git_commit_sha = git("rev-parse", "HEAD")
    return {
        # Keep git_sha for compatibility with existing review packages.
        "git_sha": git_commit_sha,
        "git_commit_sha": git_commit_sha,
        "git_tree_sha": git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(status_lines),
        "changed_path_count": len(status_lines),
    }


def source_fingerprint_paths() -> tuple[Path, ...]:
    paths = [DESIGN_TOKEN_PATH, *PRODUCT_RUNTIME_SOURCES]
    paths.extend(
        path
        for path in UI_ASSET_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    paths.extend(
        path
        for path in REFERENCE_IMAGE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
    )
    paths.append(Path(__file__).resolve())
    return tuple(sorted(set(paths), key=lambda path: relative_to_root(ROOT, path)))


def source_fingerprint() -> dict[str, str]:
    paths = source_fingerprint_paths()
    missing = [relative_to_root(ROOT, path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source fingerprint inputs: {', '.join(missing)}")
    return {relative_to_root(ROOT, path): sha256_file(path) for path in paths}


def _source_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def _evidence_path(raw_path: str, package_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    package_path = package_root / path
    if package_path.exists():
        return package_path
    # Compatibility for older manifests whose evidence paths were repo-relative.
    return ROOT / path


def validate_manifest(manifest: dict[str, Any], *, package_root: Path) -> list[str]:
    errors: list[str] = []
    revision = manifest.get("source_revision") or {}
    if not revision.get("git_commit_sha"):
        errors.append("source_revision.git_commit_sha_missing")
    if not revision.get("git_tree_sha"):
        errors.append("source_revision.git_tree_sha_missing")
    current_revision = source_revision()
    if revision.get("git_commit_sha") != current_revision.get("git_commit_sha"):
        errors.append("source_revision.git_commit_changed_during_capture")
    if revision.get("git_tree_sha") != current_revision.get("git_tree_sha"):
        errors.append("source_revision.git_tree_changed_during_capture")

    recorded_fingerprint = manifest.get("source_fingerprint") or {}
    recorded_fingerprint_sha256 = hashlib.sha256(
        json.dumps(recorded_fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("source_fingerprint_sha256") != recorded_fingerprint_sha256:
        errors.append("source_fingerprint.aggregate_sha256_mismatch")
    expected_paths = {relative_to_root(ROOT, path) for path in source_fingerprint_paths()}
    recorded_paths = set(recorded_fingerprint)
    for path in sorted(expected_paths - recorded_paths):
        errors.append(f"source_fingerprint.path_missing:{path}")
    for path in sorted(recorded_paths - expected_paths):
        errors.append(f"source_fingerprint.path_unexpected:{path}")
    for raw_path, recorded_sha256 in sorted(recorded_fingerprint.items()):
        path = _source_path(raw_path)
        if not path.is_file():
            errors.append(f"source_fingerprint.file_missing:{raw_path}")
        elif sha256_file(path) != recorded_sha256:
            errors.append(f"source_fingerprint.sha256_mismatch:{raw_path}")

    pages = manifest.get("pages") or []
    if not pages:
        errors.append("pages.empty")
    if (manifest.get("summary") or {}).get("page_count") != len(pages):
        errors.append("summary.page_count_mismatch")
    for page in pages:
        page_name = str(page.get("name") or "unknown")
        if page.get("status") != "PASS":
            errors.append(f"page.{page_name}.status_not_pass")
        for kind in ("screenshot", "dom"):
            raw_path = str(page.get(f"{kind}_path") or "")
            recorded_sha256 = page.get(f"{kind}_sha256")
            if not raw_path:
                errors.append(f"page.{page_name}.{kind}_path_missing")
                continue
            path = _evidence_path(raw_path, package_root)
            if not path.is_file():
                errors.append(f"page.{page_name}.{kind}_file_missing")
            elif not recorded_sha256 or sha256_file(path) != recorded_sha256:
                errors.append(f"page.{page_name}.{kind}_sha256_mismatch")

    negative_evidence = manifest.get("negative_evidence") or {}
    if negative_evidence.get("owner_acceptance_claimed") != 0:
        errors.append("negative_evidence.owner_acceptance_overclaim")
    if negative_evidence.get("production_readiness_claimed") != 0:
        errors.append("negative_evidence.production_readiness_overclaim")
    return errors


def validate_persisted_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    errors = validate_manifest(manifest, package_root=path.resolve().parent)
    embedded_validation = manifest.get("manifest_validation") or {}
    if embedded_validation.get("status") != "PASS" or embedded_validation.get("error_count") != 0:
        errors.append("manifest_validation.embedded_status_not_pass")
    if manifest.get("status") != "PASS":
        errors.append("manifest.status_not_pass")
    return {
        "status": "PASS" if not errors else "FAIL",
        "manifest_path": str(path.resolve()),
        "page_count": len(manifest.get("pages") or []),
        "source_fingerprint_count": len(manifest.get("source_fingerprint") or {}),
        "errors": errors,
    }


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
    emulated_viewport = {
        "width": 375 if spec.get("mobile") else 1440,
        "height": 844 if spec.get("mobile") else 1100,
        "device_scale_factor": 1,
        "mobile": bool(spec.get("mobile")),
    }
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
    selector_presence: dict[str, bool] = {}
    visible_text = ""
    try:
        process = _launch_cdp_chrome(chrome, profile_dir, debug_port, window_size)
        version = None
        devtools_deadline = time.monotonic() + DEVTOOLS_READY_TIMEOUT_SECONDS
        while time.monotonic() < devtools_deadline:
            try:
                version = _json_urlopen(f"http://127.0.0.1:{debug_port}/json/version")
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError(f"devtools_process_exited:{process.returncode}")
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
        page.command(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": emulated_viewport["width"],
                "height": emulated_viewport["height"],
                "deviceScaleFactor": emulated_viewport["device_scale_factor"],
                "mobile": emulated_viewport["mobile"],
                "screenWidth": emulated_viewport["width"],
                "screenHeight": emulated_viewport["height"],
            },
        )
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
        visible_text = str(_runtime_eval(page, "document.body ? document.body.innerText : ''", timeout=5) or "")
        required_selectors = [str(selector) for selector in spec.get("required_selectors", [])]
        selector_presence = _runtime_eval(
            page,
            f"""(() => {{
              const selectors = {json.dumps(required_selectors)};
              return Object.fromEntries(selectors.map((selector) => [selector, Boolean(document.querySelector(selector))]));
            }})()""",
            timeout=5,
        ) or {}
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
    forbidden = [] if spec.get("allow_internal_terms") else sorted(set(match.group(0) for match in FORBIDDEN_PRODUCT_RE.finditer(visible_text)))
    required_missing = [text for text in spec.get("required_text", []) if text not in visible_text]
    selector_missing = [
        selector
        for selector in spec.get("required_selectors", [])
        if selector_presence.get(selector) is not True
    ]
    screenshot_bytes = screenshot_path.stat().st_size if screenshot_path.exists() else 0
    checks = {
        "screenshot_present": screenshot_bytes > 0,
        "dom_present": bool(dom_text),
        "product_shell_present": layout.get("product_shell_present") is True,
        "surface_matches": layout.get("surface") == spec["surface"],
        "no_horizontal_overflow": layout.get("horizontal_overflow") is False,
        "token_css_present": layout.get("token_css_present") is True,
        "required_text_present": not required_missing,
        "required_selectors_present": not selector_missing,
        "forbidden_terms_absent": not forbidden,
        "mobile_first_value_before_nav": layout.get("mobile_first_value_before_nav") is True,
        "home_drop_ask_ordered": layout.get("home_drop_ask_ordered") is True,
        "home_drop_ask_in_first_viewport": layout.get("home_drop_ask_in_first_viewport") is True,
        "viewport_matches_contract": (
            layout.get("inner_width") == emulated_viewport["width"]
            and layout.get("inner_height") == emulated_viewport["height"]
        ),
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
        "emulated_viewport": emulated_viewport,
        "url": url,
        "status": status,
        "checks": checks,
        "required_missing": required_missing,
        "selector_missing": selector_missing,
        "selector_presence": selector_presence,
        "forbidden_terms": forbidden,
        "layout": layout,
        "interaction": interaction,
        "screenshot_path": relative_to_root(output_dir, screenshot_path),
        "screenshot_bytes": screenshot_bytes,
        "screenshot_sha256": sha256_file(screenshot_path) if screenshot_bytes else None,
        "dom_path": relative_to_root(output_dir, dom_path),
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
        f"- Source commit: `{manifest['source_revision']['git_commit_sha']}`; tree: `{manifest['source_revision']['git_tree_sha']}`; dirty: `{str(manifest['source_revision']['dirty']).lower()}`.",
        f"- Source fingerprint: `{manifest['source_fingerprint_sha256']}` across {len(manifest['source_fingerprint'])} UI/runtime inputs.",
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
    if args.validate_manifest is not None:
        validation = validate_persisted_manifest(args.validate_manifest)
        if args.json:
            print(json.dumps(validation, indent=2, sort_keys=True))
        else:
            print(f"{validation['status']} manifest={validation['manifest_path']} errors={len(validation['errors'])}")
        return 0 if validation["status"] == "PASS" else 1

    output_dir: Path = args.output_dir
    state_dir: Path = args.state_dir
    # Snapshot source identity before clearing state or creating any capture output.
    # This keeps the manifest tied to the code under test instead of to its own
    # generated screenshots, DOM captures, or runtime records.
    source_revision_snapshot = source_revision()
    source_fingerprint_snapshot = source_fingerprint()
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
        "source_revision": source_revision_snapshot,
        "source_fingerprint": source_fingerprint_snapshot,
        "source_fingerprint_sha256": hashlib.sha256(
            json.dumps(source_fingerprint_snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
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
            "semantic_selector_missing_pages": sum(
                1 for page in pages if not page["checks"]["required_selectors_present"]
            ),
            "owner_acceptance_claimed": 0,
            "production_readiness_claimed": 0,
        },
        "thread_errors": thread_error,
        "pages": pages,
    }
    manifest_validation_errors = validate_manifest(manifest, package_root=output_dir.resolve())
    manifest["manifest_validation"] = {
        "status": "PASS" if not manifest_validation_errors else "FAIL",
        "error_count": len(manifest_validation_errors),
        "errors": manifest_validation_errors,
    }
    if manifest_validation_errors:
        manifest["status"] = "FAIL"
    write_json(output_dir / "screenshot-pack-manifest.json", manifest)
    build_owner_package(output_dir, manifest)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"{manifest['status']} {manifest['summary']}")
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

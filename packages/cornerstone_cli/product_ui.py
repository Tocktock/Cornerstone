from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

PRODUCT_LIST_ROUTES = {"/", "/search", "/artifacts", "/briefs", "/claims", "/actions", "/inbox", "/audit"}
PRODUCT_DETAIL_ROUTES = {"artifacts", "briefs", "claims", "actions"}

FORBIDDEN_PRODUCT_ROUTE_TERMS = [
    "scenario",
    "verifier",
    "human gate",
    "acceptance",
    "walkthrough",
    "package path",
    "readiness",
    "browser proof",
    "review packet",
]

INTERNAL_PRODUCT_RECORD_RE = re.compile(
    r"\bVS[0-9]\b|VS[0-9]-|scenario|verifier|human gate|acceptance|walkthrough|"
    r"package path|readiness|browser proof|review packet|EVUX|scaffold",
    re.IGNORECASE,
)


def h(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def render_product_page(
    root: Path,
    store: Any,
    scope: dict[str, str],
    path: str,
    query: dict[str, list[str]],
) -> str:
    ctx = _build_context(store, scope)
    route = path if path in PRODUCT_LIST_ROUTES else "/"
    q = (query.get("q") or [""])[-1].strip()
    if route == "/":
        title = "Home"
        content = _home(ctx)
        active = "/"
    elif route == "/search":
        title = "Search"
        content = _search_page(ctx, q)
        active = "/search"
    elif route == "/artifacts":
        title = "Artifacts"
        content = _artifact_list_page(ctx)
        active = "/artifacts"
    elif route == "/briefs":
        title = "Briefs"
        content = _brief_list_page(ctx)
        active = "/"
    elif route == "/claims":
        title = "Claims"
        content = _claim_list_page(ctx)
        active = "/claims"
    elif route == "/actions":
        title = "Actions"
        content = _action_list_page(ctx)
        active = "/actions"
    elif route == "/inbox":
        title = "Inbox"
        content = _inbox_page(ctx)
        active = "/inbox"
    else:
        title = "Audit"
        content = _audit_page(ctx)
        active = "/audit"
    return _page(root, title, active, content, ctx, q)


def render_product_detail(
    root: Path,
    store: Any,
    scope: dict[str, str],
    kind: str,
    record_id: str,
) -> tuple[int, str]:
    ctx = _build_context(store, scope)
    record: dict[str, Any] | None
    if kind == "artifacts":
        record = store.get_artifact(record_id, scope)
        if record:
            record = dict(record)
            record["_preview"] = _safe_preview(store, record, 5000)
        if record and _internal_product_record(record):
            content = _internal_record_notice("source")
            title = "Owner record"
            active = "/artifacts"
            return 200, _page(root, title, active, content, ctx, "")
        content = _artifact_detail(ctx, store, record) if record else _not_found("source")
        title = _artifact_title(record) if record else "Source not found"
        active = "/artifacts"
    elif kind == "briefs":
        record = store.get_brief(record_id)
        if record and record.get("scope") != scope:
            record = None
        if record and _internal_product_record(record):
            content = _internal_record_notice("brief")
            title = "Owner record"
            active = "/"
            return 200, _page(root, title, active, content, ctx, "")
        content = _brief_detail(ctx, record) if record else _not_found("brief")
        title = str(record.get("title") or "Brief") if record else "Brief not found"
        active = "/"
    elif kind == "claims":
        record = store.get_claim(record_id)
        if record and record.get("scope") != scope:
            record = None
        if record and _internal_product_record(record):
            content = _internal_record_notice("claim")
            title = "Owner record"
            active = "/claims"
            return 200, _page(root, title, active, content, ctx, "")
        content = _claim_detail(ctx, record) if record else _not_found("claim")
        title = _truncate(str(record.get("statement") or "Claim"), 72) if record else "Claim not found"
        active = "/claims"
    elif kind == "actions":
        record = store.get_action(record_id)
        if record and record.get("scope") != scope:
            record = None
        if record and _internal_product_record(record):
            content = _internal_record_notice("action")
            title = "Owner record"
            active = "/actions"
            return 200, _page(root, title, active, content, ctx, "")
        content = _action_detail(ctx, record) if record else _not_found("action")
        title = _action_title(record) if record else "Action not found"
        active = "/actions"
    else:
        return 404, _page(root, "Not found", "/", _not_found("record"), ctx, "")
    status = 200 if record else 404
    return status, _page(root, title, active, content, ctx, "")


def _build_context(store: Any, scope: dict[str, str]) -> dict[str, Any]:
    artifacts = _recent(_safe_records(lambda: store._artifact_records(scope)))
    briefs = _recent(_safe_records(lambda: store._brief_records(scope)))
    claims = _recent(_safe_records(lambda: store._claim_records(scope)))
    actions = _recent(_safe_records(lambda: store._action_records(scope)))
    memories = _recent(_safe_records(lambda: store._memory_records(scope)))
    audit = _recent(
        [
            event
            for event in _safe_records(lambda: store._all_audit_events())
            if _same_scope(event.get("scope"), scope)
        ],
        limit=80,
    )
    for artifact in artifacts:
        artifact["_preview"] = _safe_preview(store, artifact, 260)
    artifacts = [record for record in artifacts if not _internal_product_record(record)]
    briefs = [record for record in briefs if not _internal_product_record(record)]
    claims = [record for record in claims if not _internal_product_record(record)]
    actions = [record for record in actions if not _internal_product_record(record)]
    memories = [record for record in memories if not _internal_product_record(record)]
    return {
        "store": store,
        "scope": scope,
        "artifacts": artifacts,
        "briefs": briefs,
        "claims": claims,
        "actions": actions,
        "memories": memories,
        "audit": audit,
        "suggestions": _suggestions(artifacts, briefs, claims),
        "inbox": _inbox_items(briefs, claims, actions, memories),
    }


def _safe_records(read: Any) -> list[dict[str, Any]]:
    try:
        records = read()
    except Exception:
        return []
    return [record for record in records if isinstance(record, dict)]


def _safe_preview(store: Any, artifact: dict[str, Any], limit: int = 240) -> str:
    try:
        return store.derived_text_preview(artifact, limit)
    except Exception:
        return ""


def _same_scope(value: Any, scope: dict[str, str]) -> bool:
    return isinstance(value, dict) and all(value.get(key) == expected for key, expected in scope.items())


def _internal_product_record(record: dict[str, Any]) -> bool:
    try:
        text = json.dumps(record, sort_keys=True)
    except TypeError:
        text = str(record)
    return bool(INTERNAL_PRODUCT_RECORD_RE.search(text))


def _recent(records: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    sorted_records = sorted(records, key=_record_time_key, reverse=True)
    return sorted_records[:limit] if limit else sorted_records


def _record_time_key(record: dict[str, Any]) -> str:
    for key in ("updated_at", "created_at", "timestamp", "recorded_at", "decided_at"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    provenance = record.get("provenance")
    if isinstance(provenance, dict):
        value = provenance.get("created_at")
        if isinstance(value, str) and value:
            return value
    source = record.get("source")
    if isinstance(source, dict):
        value = source.get("ingested_at")
        if isinstance(value, str) and value:
            return value
    dry_run = record.get("dry_run")
    if isinstance(dry_run, dict):
        value = dry_run.get("created_at")
        if isinstance(value, str):
            return value
    return ""


def _display_date(record: dict[str, Any]) -> str:
    raw = _record_time_key(record)
    if not raw:
        return "No date"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return raw[:16]
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def _truncate(value: str, length: int = 140) -> str:
    text = " ".join(value.split())
    if len(text) <= length:
        return text
    return text[: max(0, length - 1)].rstrip() + "..."


def _artifact_title(record: dict[str, Any] | None) -> str:
    if not record:
        return "Source"
    preview = str(record.get("_preview") or "").strip()
    if preview:
        return _truncate(preview, 72)
    source = record.get("source")
    source_ref = ""
    if isinstance(source, dict):
        source_ref = str(source.get("ref") or "").strip()
    else:
        source_ref = str(record.get("source_ref") or source or "").strip()
    if source_ref and source_ref not in {"local_file", "cli_text", "home.drop_text"}:
        return _truncate(source_ref, 72)
    media = str(record.get("media_type") or "source").replace("/", " ")
    return media.title()


def _brief_title(record: dict[str, Any]) -> str:
    return _truncate(str(record.get("title") or "Untitled brief"), 96)


def _claim_title(record: dict[str, Any]) -> str:
    return _truncate(str(record.get("statement") or "Untitled claim"), 120)


def _action_title(record: dict[str, Any] | None) -> str:
    if not record:
        return "Action"
    dry_run = record.get("dry_run") if isinstance(record.get("dry_run"), dict) else {}
    return _truncate(str(record.get("title") or dry_run.get("goal") or "Action draft"), 96)


def _plain_event(event_type: str) -> str:
    labels = {
        "artifact.ingest": "Source saved",
        "artifact.read": "Source opened",
        "search.run": "Search run",
        "brief.create": "Brief drafted",
        "brief.read": "Brief opened",
        "claim.create": "Claim drafted",
        "claim.read": "Claim opened",
        "claim.approve": "Claim approved",
        "action.create": "Action drafted",
        "action.read": "Action opened",
        "action.dry_run": "Action previewed",
        "action.approve": "Action approved",
        "action.execute": "Action executed",
        "conversation.start": "Ask started",
        "conversation.answer": "Draft answer saved",
    }
    if event_type in labels:
        return labels[event_type]
    return "Recorded activity"


def _brief_label(record: dict[str, Any]) -> tuple[str, str]:
    output_mode = str(record.get("output_mode") or "").lower()
    trust = str(record.get("trust_label") or record.get("status") or "").lower()
    if output_mode in {"model_cited", "citation_grounded", "ollama_generated"} and trust in {"evidence_backed", "corroborated"}:
        return "Source-backed", "evidenceBacked"
    if output_mode in {"extractive_fallback", "template_fallback"} or trust in {"extractive_fallback", "template_fallback"}:
        return "Keyword summary", "draft"
    if trust == "insufficient_evidence":
        return "Needs sources", "underReview"
    return "Draft", "draft"


def _claim_label(record: dict[str, Any]) -> tuple[str, str]:
    status = str(record.get("status") or "").lower()
    if status == "approved":
        return "Approved", "approved"
    if status in {"policy_blocked", "blocked"}:
        return "Policy blocked", "policyBlocked"
    if _evidence_refs(record):
        return "Source support", "evidenceBacked"
    return "Draft", "draft"


def _action_label(record: dict[str, Any]) -> tuple[str, str]:
    status = str(record.get("status") or "").lower()
    dry_run = record.get("dry_run") if isinstance(record.get("dry_run"), dict) else {}
    decision = str((dry_run.get("policy_decision") or {}).get("decision") or "").lower() if isinstance(dry_run, dict) else ""
    if status == "executed":
        return "Executed", "executed"
    if status == "failed":
        return "Failed", "failed"
    if "approval" in decision:
        return "Needs approval", "underReview"
    return "Draft", "draft"


def _chip(label: str, state: str = "draft") -> str:
    return f'<span class="cs-chip cs-chip-{h(state)}">{h(label)}</span>'


def _suggestions(
    artifacts: list[dict[str, Any]],
    briefs: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> list[str]:
    suggestions: list[str] = []
    if briefs:
        suggestions.append(f'What supports "{_brief_title(briefs[0])}"?')
    if artifacts:
        suggestions.append("What changed in the latest saved source?")
    if claims:
        suggestions.append("Which claims still need support?")
    return suggestions[:3]


def _inbox_items(
    briefs: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    memories: list[dict[str, Any]],
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for brief in briefs[:3]:
        label, state = _brief_label(brief)
        items.append(
            {
                "kind": "Brief",
                "title": _brief_title(brief),
                "detail": "Check sources and gaps before sharing.",
                "label": label,
                "state": state,
                "href": _detail_href("briefs", brief.get("brief_id")),
                "date": _display_date(brief),
                "queue": "Needs review",
                "priority": "Medium",
                "owner": "Owner",
                "type": "Brief",
                "icon": "B",
            }
        )
    for claim in [claim for claim in claims if str(claim.get("status") or "").lower() != "approved"][:4]:
        label, state = _claim_label(claim)
        items.append(
            {
                "kind": "Claim",
                "title": _claim_title(claim),
                "detail": "Needs source support before it can be approved.",
                "label": label,
                "state": state,
                "href": _detail_href("claims", claim.get("claim_id")),
                "date": _display_date(claim),
                "queue": "Needs review",
                "priority": "High" if state == "draft" else "Medium",
                "owner": "Owner",
                "type": "Claim",
                "icon": "C",
            }
        )
    for action in actions[:4]:
        label, state = _action_label(action)
        items.append(
            {
                "kind": "Action",
                "title": _action_title(action),
                "detail": "Preview the impact before any external write.",
                "label": label,
                "state": state,
                "href": _detail_href("actions", action.get("action_id")),
                "date": _display_date(action),
                "queue": "Approval requests" if state == "underReview" else "Needs review",
                "priority": "High" if state == "underReview" else "Medium",
                "owner": "Owner",
                "type": "Action",
                "icon": "A",
            }
        )
    for memory in memories[:2]:
        items.append(
            {
                "kind": "Memory",
                "title": _truncate(str(memory.get("title") or memory.get("statement") or "Knowledge draft"), 96),
                "detail": "Saved as a draft until approved by the owner.",
                "label": "Draft",
                "state": "draft",
                "href": "/inbox",
                "date": _display_date(memory),
                "queue": "Needs review",
                "priority": "Low",
                "owner": "Owner",
                "type": "Memory",
                "icon": "M",
            }
        )
    return _recent(items, limit=8)  # type: ignore[arg-type]


def _detail_href(kind: str, record_id: Any) -> str:
    if not record_id:
        return f"/{kind}"
    return f"/{kind}/{quote(str(record_id))}?view=html"


def _token_css(root: Path) -> str:
    token_path = root / "docs" / "design" / "tokens" / "cornerstone_design_tokens_v0_3.json"
    tokens = json.loads(token_path.read_text())
    variables: list[tuple[str, str]] = []

    def flatten(prefix: list[str], value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                flatten([*prefix, _css_name(key)], child)
        elif isinstance(value, list):
            variables.append(("--cs-" + "-".join(prefix), ", ".join(str(item) for item in value)))
        else:
            variables.append(("--cs-" + "-".join(prefix), str(value)))

    flatten([], tokens)
    var_block = "\n".join(f"  {name}: {value};" for name, value in variables)
    return f"""
:root {{
{var_block}
}}
* {{ box-sizing: border-box; }}
html {{ min-height: 100%; background: var(--cs-color-background-app); }}
body {{
  margin: 0;
  min-height: 100%;
  background: var(--cs-color-background-app);
  color: var(--cs-color-text-primary);
  font-family: var(--cs-typography-fontFamily);
  font-size: var(--cs-typography-body-fontSize);
  line-height: var(--cs-typography-body-lineHeight);
}}
a {{ color: inherit; text-decoration: none; }}
button, input, textarea {{ font: inherit; }}
.cs-shell {{
  min-height: 100vh;
  display: grid;
  grid-template-columns: var(--cs-layout-sidebarWidth) minmax(0, 1fr);
}}
.cs-skip-link {{
  position: fixed;
  left: var(--cs-space-4);
  top: var(--cs-space-4);
  z-index: 10;
  transform: translateY(-160%);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-primary-600);
  color: var(--cs-color-text-inverse);
  padding: var(--cs-space-2) var(--cs-space-4);
}}
.cs-skip-link:focus-visible {{ transform: translateY(0); outline: 3px solid var(--cs-color-primary-100); }}
.cs-sidebar {{
  position: sticky;
  top: 0;
  height: 100vh;
  border-right: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-6) var(--cs-space-4);
  display: flex;
  flex-direction: column;
  gap: var(--cs-space-6);
}}
.cs-brand {{ display: grid; gap: var(--cs-space-1); }}
.cs-brand-mark {{
  width: 36px;
  height: 36px;
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-primary-600);
  color: var(--cs-color-text-inverse);
  display: grid;
  place-items: center;
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-brand-name {{ font-weight: var(--cs-typography-weight-bold); font-size: var(--cs-typography-sectionTitle-fontSize); }}
.cs-brand-sub {{ color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-nav {{ display: grid; gap: var(--cs-space-1); }}
.cs-nav-group {{ display: grid; gap: var(--cs-space-1); }}
.cs-nav-label {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-label-fontSize);
  line-height: var(--cs-typography-label-lineHeight);
  font-weight: var(--cs-typography-weight-semibold);
  text-transform: uppercase;
}}
.cs-nav a {{
  min-height: 40px;
  border-radius: var(--cs-radius-md);
  padding: var(--cs-space-2) var(--cs-space-3);
  color: var(--cs-color-text-secondary);
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
  font-weight: var(--cs-typography-weight-medium);
}}
.cs-nav a:hover, .cs-nav a:focus-visible {{ background: var(--cs-color-surface-subtle); outline: none; }}
.cs-nav a[aria-current="page"] {{ background: var(--cs-color-primary-50); color: var(--cs-color-primary-700); }}
.cs-main {{ min-width: 0; }}
.cs-topbar {{
  min-height: var(--cs-layout-headerHeight);
  border-bottom: 1px solid var(--cs-color-border-default);
  background: color-mix(in srgb, var(--cs-color-surface-primary) 92%, transparent);
  backdrop-filter: blur(12px);
  display: flex;
  align-items: center;
  gap: var(--cs-space-4);
  justify-content: space-between;
  padding: var(--cs-space-4) var(--cs-layout-contentGutter);
  position: sticky;
  top: 0;
  z-index: 2;
}}
.cs-search {{
  flex: 1 1 520px;
  max-width: 680px;
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2) var(--cs-space-3);
}}
.cs-search input {{ border: 0; outline: 0; min-width: 0; flex: 1; color: var(--cs-color-text-primary); background: transparent; }}
.cs-search button, .cs-button {{
  border: 1px solid var(--cs-color-primary-600);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-primary-600);
  color: var(--cs-color-text-inverse);
  padding: var(--cs-space-2) var(--cs-space-4);
  min-height: 38px;
  font-weight: var(--cs-typography-weight-semibold);
  cursor: pointer;
  transition: background .18s ease, border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}}
.cs-search button:hover, .cs-button:hover {{ box-shadow: var(--cs-shadow-sm); transform: translateY(-1px); }}
.cs-search button:active, .cs-button:active {{ transform: translateY(0); }}
.cs-button.secondary {{
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-text-primary);
  border-color: var(--cs-color-border-strong);
}}
.cs-button.ghost {{
  background: transparent;
  color: var(--cs-color-text-secondary);
  border-color: transparent;
}}
.cs-kicker {{ color: var(--cs-color-primary-700); font-weight: var(--cs-typography-weight-semibold); font-size: var(--cs-typography-label-fontSize); }}
.cs-content {{ padding: var(--cs-layout-contentGutter); max-width: 1360px; }}
.cs-page-head {{ display: grid; gap: var(--cs-space-2); margin-bottom: var(--cs-space-6); max-width: 760px; }}
.cs-page-head h1, .cs-hero h1 {{
  margin: 0;
  font-size: var(--cs-typography-pageTitle-fontSize);
  line-height: var(--cs-typography-pageTitle-lineHeight);
  letter-spacing: 0;
}}
.cs-hero h1 {{ font-size: var(--cs-typography-display-fontSize); line-height: var(--cs-typography-display-lineHeight); }}
.cs-page-head p, .cs-hero p {{ margin: 0; color: var(--cs-color-text-secondary); max-width: 760px; }}
.cs-grid-hero {{ display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(320px, .55fr); gap: var(--cs-space-6); align-items: start; }}
.cs-grid-two {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(300px, 380px); gap: var(--cs-space-6); align-items: start; }}
.cs-stack {{ display: grid; gap: var(--cs-space-4); }}
.cs-panel {{
  background: var(--cs-color-surface-primary);
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  box-shadow: var(--cs-shadow-sm);
  padding: var(--cs-layout-cardPadding);
}}
.cs-panel.flat {{ box-shadow: none; }}
.cs-panel-header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: var(--cs-space-4); margin-bottom: var(--cs-space-4); }}
.cs-panel-header h2, .cs-section-title {{
  margin: 0;
  font-size: var(--cs-typography-sectionTitle-fontSize);
  line-height: var(--cs-typography-sectionTitle-lineHeight);
}}
.cs-muted {{ color: var(--cs-color-text-muted); }}
.cs-meta {{ color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); line-height: var(--cs-typography-metadata-lineHeight); }}
.cs-home-intro {{
  min-height: calc(100vh - var(--cs-layout-headerHeight) - (var(--cs-layout-contentGutter) * 2));
  align-content: start;
}}
.cs-home-canvas {{
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-home-canvas .cs-panel-header {{ margin-bottom: 0; }}
.cs-home-canvas p {{ max-width: 62ch; }}
.cs-home-workspace {{
  display: grid;
  gap: var(--cs-space-5);
}}
.cs-home-source-row {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-home-source-note {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-drop {{
  min-height: auto;
  border: 1px dashed var(--cs-color-border-strong);
  border-radius: var(--cs-radius-lg);
  background: color-mix(in srgb, var(--cs-color-surface-subtle) 70%, var(--cs-color-surface-primary));
  display: grid;
  gap: var(--cs-space-2);
  padding: var(--cs-space-3);
}}
.cs-drop.is-hot {{ border-color: var(--cs-color-primary-600); background: var(--cs-color-primary-50); }}
.cs-drop textarea, .cs-field {{
  width: 100%;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-text-primary);
  padding: var(--cs-space-3);
  outline: none;
}}
.cs-drop textarea {{ min-height: 130px; resize: vertical; }}
.cs-drop textarea:focus, .cs-field:focus {{ border-color: var(--cs-color-border-focus); box-shadow: 0 0 0 3px var(--cs-color-primary-50); }}
.cs-drop-target {{
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: var(--cs-space-2);
  place-items: center start;
  text-align: left;
  padding: 0;
}}
.cs-drop-mark {{
  width: 44px;
  height: 44px;
  border-radius: var(--cs-radius-lg);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
  border: 1px solid var(--cs-color-primary-100);
}}
.cs-drop textarea.cs-drop-input {{
  min-height: 46px;
  background: var(--cs-color-surface-primary);
}}
.cs-or-divider {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  align-items: center;
  gap: var(--cs-space-3);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-or-divider::before, .cs-or-divider::after {{
  content: "";
  height: 1px;
  background: var(--cs-color-border-default);
}}
.cs-ask-bar {{
  border: 1px solid var(--cs-color-border-focus);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--cs-space-3);
  box-shadow: var(--cs-shadow-sm);
}}
.cs-ask-bar .cs-field {{
  border: 0;
  padding: var(--cs-space-2);
  box-shadow: none;
}}
.cs-ask-bar .cs-field:focus {{ box-shadow: none; }}
.cs-suggestion-row {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--cs-space-2); }}
.cs-suggestion-row .cs-button {{ min-width: 0; justify-content: center; white-space: normal; }}
.cs-home-item-list {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  overflow: hidden;
  background: var(--cs-color-surface-primary);
}}
.cs-home-item {{
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
  min-height: 72px;
  padding: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-home-item:last-child {{ border-bottom: 0; }}
.cs-home-item:hover {{ background: var(--cs-color-surface-subtle); }}
.cs-home-item-icon {{
  width: 30px;
  height: 30px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-home-item h3 {{ margin: 0; font-size: var(--cs-typography-body-fontSize); line-height: var(--cs-typography-body-lineHeight); }}
.cs-home-item p {{ margin: 0; color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-next-step-list {{ display: grid; gap: var(--cs-space-2); }}
.cs-next-step {{
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
  padding: var(--cs-space-3);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
}}
.cs-next-step strong {{ font-size: var(--cs-typography-body-fontSize); }}
.cs-row {{ display: flex; align-items: center; gap: var(--cs-space-3); flex-wrap: wrap; }}
.cs-module-grid {{ display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(280px, .95fr); gap: var(--cs-space-4); }}
.cs-list {{ display: grid; gap: var(--cs-space-3); }}
.cs-list-row {{
  min-height: var(--cs-layout-listRowHeight);
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-list-row:hover {{ border-color: var(--cs-color-border-strong); box-shadow: var(--cs-shadow-sm); }}
.cs-list-row h3 {{ margin: 0 0 var(--cs-space-1); font-size: var(--cs-typography-body-fontSize); line-height: var(--cs-typography-body-lineHeight); }}
.cs-list-row p {{ margin: 0; color: var(--cs-color-text-secondary); }}
.cs-list-row.compact {{ padding: var(--cs-space-3); min-height: auto; }}
.cs-search-workbench {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 340px);
  gap: var(--cs-space-6);
  align-items: start;
}}
.cs-search-hero {{
  border: 1px solid var(--cs-color-border-focus);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  box-shadow: var(--cs-shadow-sm);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-search-hero input {{
  min-height: 46px;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--cs-color-text-primary);
  padding: 0 var(--cs-space-2);
}}
.cs-search-tabs, .cs-filter-row {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  margin-top: var(--cs-space-4);
}}
.cs-search-tab, .cs-filter-chip {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  min-height: 34px;
  border-radius: var(--cs-radius-md);
  border: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-text-secondary);
  padding: 0 var(--cs-space-3);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-medium);
}}
.cs-search-tab.is-active {{
  background: var(--cs-color-primary-50);
  border-color: var(--cs-color-primary-100);
  color: var(--cs-color-primary-700);
}}
.cs-result-list {{
  display: grid;
  gap: var(--cs-space-3);
  margin-top: var(--cs-space-5);
}}
.cs-result-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  grid-template-columns: 42px minmax(0, 1fr) auto;
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-result-row:hover {{ border-color: var(--cs-color-border-strong); box-shadow: var(--cs-shadow-sm); }}
.cs-result-icon {{
  width: 34px;
  height: 34px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-result-body {{ display: grid; gap: var(--cs-space-1); }}
.cs-result-body h3 {{ margin: 0; font-size: 16px; line-height: 1.35; }}
.cs-result-body p {{ margin: 0; color: var(--cs-color-text-secondary); max-width: 78ch; }}
.cs-result-meta {{ display: flex; flex-wrap: wrap; gap: var(--cs-space-2); color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-right-stat {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
  padding: var(--cs-space-2) 0;
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-right-stat:last-child {{ border-bottom: 0; }}
.cs-suggested-query {{
  display: grid;
  grid-template-columns: 22px minmax(0, 1fr);
  gap: var(--cs-space-2);
  align-items: start;
  color: var(--cs-color-text-secondary);
  padding: var(--cs-space-2) 0;
}}
.cs-suggested-query span:first-child {{ color: var(--cs-color-primary-700); font-weight: var(--cs-typography-weight-semibold); }}
.cs-artifact-hero {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-4);
  align-items: start;
  margin-bottom: var(--cs-space-5);
}}
.cs-artifact-title {{ display: grid; gap: var(--cs-space-2); }}
.cs-artifact-title h1 {{
  margin: 0;
  font-size: var(--cs-typography-pageTitle-fontSize);
  line-height: var(--cs-typography-pageTitle-lineHeight);
}}
.cs-artifact-actions {{ display: flex; flex-wrap: wrap; gap: var(--cs-space-2); justify-content: flex-end; }}
.cs-metadata-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-5);
}}
.cs-metadata-item {{
  border-left: 1px solid var(--cs-color-border-default);
  padding-left: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-document-frame {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-4);
}}
.cs-document-page {{
  max-width: 840px;
  min-height: 420px;
  margin: 0 auto;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  box-shadow: var(--cs-shadow-sm);
  padding: clamp(var(--cs-space-5), 5vw, var(--cs-space-8));
}}
.cs-document-heading {{
  border-bottom: 1px solid var(--cs-color-border-default);
  margin-bottom: var(--cs-space-5);
  padding-bottom: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-document-heading h3 {{
  margin: 0;
  font-size: 20px;
  line-height: 1.35;
}}
.cs-document-page .cs-source-text {{
  border: 0;
  border-radius: 0;
  background: transparent;
  padding: 0;
  line-height: 1.75;
}}
.cs-keyword-list {{ display: grid; gap: var(--cs-space-2); }}
.cs-keyword-row {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--cs-space-3);
}}
.cs-inbox-workbench {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(300px, 360px);
  gap: var(--cs-space-6);
  align-items: start;
}}
.cs-inbox-tabs {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-5);
  border-bottom: 1px solid var(--cs-color-border-default);
  margin-bottom: var(--cs-space-4);
}}
.cs-inbox-tab {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  min-height: 42px;
  color: var(--cs-color-text-secondary);
  border-bottom: 2px solid transparent;
  font-weight: var(--cs-typography-weight-medium);
}}
.cs-inbox-tab.is-active {{
  color: var(--cs-color-primary-700);
  border-color: var(--cs-color-primary-600);
}}
.cs-inbox-toolbar {{
  display: flex;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
  margin-bottom: var(--cs-space-4);
}}
.cs-inbox-table {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  overflow: hidden;
}}
.cs-inbox-head, .cs-inbox-row {{
  display: grid;
  grid-template-columns: minmax(260px, 1.45fr) minmax(82px, .5fr) minmax(100px, .6fr) minmax(92px, .55fr) minmax(120px, .7fr);
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-inbox-head {{
  padding: var(--cs-space-3) var(--cs-space-4);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-inbox-row {{
  padding: var(--cs-space-4);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-inbox-row:last-child {{ border-bottom: 0; }}
.cs-inbox-row:hover {{ background: var(--cs-color-surface-subtle); }}
.cs-inbox-row.is-selected {{
  background: var(--cs-color-primary-50);
  box-shadow: inset 3px 0 0 var(--cs-color-primary-600);
}}
.cs-inbox-item-title {{
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-inbox-icon {{
  width: 30px;
  height: 30px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-inbox-item-title strong {{ display: block; }}
.cs-inbox-item-title .cs-meta {{ display: block; }}
.cs-inbox-detail {{
  display: grid;
  gap: var(--cs-space-4);
}}
.cs-inbox-detail h2 {{ margin: 0; font-size: var(--cs-typography-sectionTitle-fontSize); }}
.cs-inbox-actions {{ display: grid; gap: var(--cs-space-2); }}
.cs-inbox-actions .cs-button {{ justify-content: center; text-align: center; }}
.cs-collection-workbench {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 340px);
  gap: var(--cs-space-6);
  align-items: start;
}}
.cs-collection-summary {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-4);
}}
.cs-collection-stat {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-collection-stat strong {{
  font-size: 22px;
  line-height: 1.2;
  font-variant-numeric: tabular-nums;
}}
.cs-collection-toolbar {{
  display: flex;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
  margin-bottom: var(--cs-space-4);
}}
.cs-collection-list {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-collection-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) auto;
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-collection-row:hover {{ border-color: var(--cs-color-border-strong); box-shadow: var(--cs-shadow-sm); }}
.cs-collection-icon {{
  width: 32px;
  height: 32px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-collection-body {{ display: grid; gap: var(--cs-space-1); }}
.cs-collection-body h3 {{ margin: 0; font-size: 16px; line-height: 1.35; }}
.cs-collection-body p {{ margin: 0; color: var(--cs-color-text-secondary); max-width: 82ch; }}
.cs-collection-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-empty {{
  border: 1px dashed var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  padding: var(--cs-space-6);
  color: var(--cs-color-text-muted);
  background: var(--cs-color-surface-subtle);
}}
.cs-chip {{
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  border-radius: var(--cs-radius-pill);
  border: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-subtle);
  color: var(--cs-color-text-secondary);
  padding: 0 var(--cs-space-2);
  font-size: var(--cs-typography-label-fontSize);
  line-height: var(--cs-typography-label-lineHeight);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-chip-saved {{ background: var(--cs-state-saved-bg); border-color: var(--cs-state-saved-border); color: var(--cs-state-saved-fg); }}
.cs-chip-searchable {{ background: var(--cs-state-searchable-bg); border-color: var(--cs-state-searchable-border); color: var(--cs-state-searchable-fg); }}
.cs-chip-draft {{ background: var(--cs-state-draft-bg); border-color: var(--cs-state-draft-border); color: var(--cs-state-draft-fg); }}
.cs-chip-evidenceBacked {{ background: var(--cs-state-evidenceBacked-bg); border-color: var(--cs-state-evidenceBacked-border); color: var(--cs-state-evidenceBacked-fg); }}
.cs-chip-underReview {{ background: var(--cs-state-underReview-bg); border-color: var(--cs-state-underReview-border); color: var(--cs-state-underReview-fg); }}
.cs-chip-approved {{ background: var(--cs-state-approved-bg); border-color: var(--cs-state-approved-border); color: var(--cs-state-approved-fg); }}
.cs-chip-executed {{ background: var(--cs-state-executed-bg); border-color: var(--cs-state-executed-border); color: var(--cs-state-executed-fg); }}
.cs-chip-failed {{ background: var(--cs-state-failed-bg); border-color: var(--cs-state-failed-border); color: var(--cs-state-failed-fg); }}
.cs-chip-policyBlocked {{ background: var(--cs-state-policyBlocked-bg); border-color: var(--cs-state-policyBlocked-border); color: var(--cs-state-policyBlocked-fg); }}
.cs-source-text {{
  white-space: pre-wrap;
  word-break: break-word;
  border-radius: var(--cs-radius-md);
  border: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-4);
}}
.cs-detail-grid {{
  display: grid;
  grid-template-columns: 140px minmax(0, 1fr);
  gap: var(--cs-space-2) var(--cs-space-3);
  margin-top: var(--cs-space-4);
}}
.cs-detail-grid dt {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-detail-grid dd {{ margin: 0; color: var(--cs-color-text-secondary); min-width: 0; word-break: break-word; }}
.cs-finding-list {{ display: grid; gap: var(--cs-space-3); margin: 0; padding: 0; list-style: none; }}
.cs-finding {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-finding-head {{
  display: flex;
  justify-content: space-between;
  gap: var(--cs-space-3);
  align-items: flex-start;
}}
.cs-finding-index {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  font-variant-numeric: tabular-nums;
}}
.cs-citation-rail {{ display: flex; flex-wrap: wrap; gap: var(--cs-space-2); }}
.cs-brief-hero {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-4);
  align-items: start;
  margin-bottom: var(--cs-space-5);
}}
.cs-brief-title {{ display: grid; gap: var(--cs-space-2); }}
.cs-brief-title h1 {{
  margin: 0;
  font-size: var(--cs-typography-pageTitle-fontSize);
  line-height: var(--cs-typography-pageTitle-lineHeight);
}}
.cs-brief-meta {{ display: flex; flex-wrap: wrap; gap: var(--cs-space-2); color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-brief-actions {{
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: var(--cs-space-2);
}}
.cs-brief-hero.is-stacked {{
  grid-template-columns: 1fr;
  gap: var(--cs-space-3);
}}
.cs-brief-hero.is-stacked .cs-brief-actions {{ justify-content: flex-start; }}
.cs-brief-fact-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-5);
}}
.cs-brief-fact {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-brief-fact strong {{ font-size: var(--cs-typography-sectionTitle-fontSize); line-height: var(--cs-typography-sectionTitle-lineHeight); }}
.cs-summary-card {{
  background: color-mix(in srgb, var(--cs-color-primary-50) 48%, var(--cs-color-surface-primary));
  border-color: var(--cs-color-primary-100);
}}
.cs-summary-card p {{
  margin: 0;
  font-size: 16px;
  line-height: 1.7;
  color: var(--cs-color-text-primary);
}}
.cs-brief-note-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: var(--cs-space-4);
}}
.cs-brief-note-list {{
  margin: 0;
  padding-left: var(--cs-space-5);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-stat-list {{ display: grid; gap: var(--cs-space-3); }}
.cs-stat-row {{
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--cs-space-3);
}}
.cs-stat-icon {{
  width: 34px;
  height: 34px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-surface-subtle);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-source-card summary {{
  cursor: pointer;
  color: var(--cs-color-text-primary);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-source-card[open] summary {{ margin-bottom: var(--cs-space-3); }}
.cs-provenance {{
  border-top: 1px solid var(--cs-color-border-default);
  margin-top: var(--cs-space-3);
  padding-top: var(--cs-space-3);
}}
.cs-trust-ladder {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin: var(--cs-space-4) 0;
}}
.cs-trust-step {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-trust-step strong {{ display: flex; align-items: center; gap: var(--cs-space-2); }}
.cs-trust-step strong::before {{
  content: "";
  width: 10px;
  height: 10px;
  border-radius: var(--cs-radius-full);
  border: 2px solid var(--cs-color-border-strong);
  background: var(--cs-color-surface-primary);
}}
.cs-trust-step.is-active {{
  border-color: var(--cs-state-evidenceBacked-border);
  background: var(--cs-state-evidenceBacked-bg);
}}
.cs-trust-step.is-active strong::before {{ border-color: var(--cs-state-evidenceBacked-fg); background: var(--cs-state-evidenceBacked-fg); }}
.cs-trust-step.is-locked {{ opacity: .76; }}
.cs-form-surface {{
  display: grid;
  gap: var(--cs-space-4);
}}
.cs-field-block {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-field-block p {{ margin: 0; }}
.cs-evidence-picker {{ display: grid; gap: var(--cs-space-3); }}
.cs-evidence-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-checkmark {{
  width: 20px;
  height: 20px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  border: 1px solid var(--cs-color-primary-600);
  background: var(--cs-color-primary-600);
  color: var(--cs-color-text-inverse);
  font-size: 12px;
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-review-box {{
  border-top: 1px solid var(--cs-color-border-default);
  padding-top: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-action-summary {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-action-review-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-5);
}}
.cs-action-review-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-action-review-card strong {{ font-size: var(--cs-typography-sectionTitle-fontSize); line-height: var(--cs-typography-sectionTitle-lineHeight); }}
.cs-action-metric {{
  display: grid;
  gap: var(--cs-space-1);
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  padding: var(--cs-space-3);
  background: var(--cs-color-surface-subtle);
}}
.cs-action-object-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--cs-space-3);
}}
.cs-action-object-icon {{
  width: 30px;
  height: 30px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-diff-view {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  overflow: hidden;
  background: var(--cs-color-surface-primary);
}}
.cs-diff-line {{
  display: grid;
  grid-template-columns: 84px minmax(0, 1fr);
  gap: var(--cs-space-3);
  padding: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-diff-line:last-child {{ border-bottom: 0; }}
.cs-diff-line.before {{ background: color-mix(in srgb, var(--cs-state-failed-bg) 58%, var(--cs-color-surface-primary)); }}
.cs-diff-line.after {{ background: color-mix(in srgb, var(--cs-state-evidenceBacked-bg) 62%, var(--cs-color-surface-primary)); }}
.cs-call-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-approval-note {{
  border: 1px solid var(--cs-state-underReview-border);
  background: var(--cs-state-underReview-bg);
  border-radius: var(--cs-radius-md);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-policy-card {{
  border: 1px solid var(--cs-state-underReview-border);
  background: var(--cs-state-underReview-bg);
  border-radius: var(--cs-radius-md);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-timeline {{ display: grid; gap: var(--cs-space-3); }}
.cs-timeline-item {{ display: grid; grid-template-columns: 16px minmax(0, 1fr); gap: var(--cs-space-3); }}
.cs-dot {{ width: 10px; height: 10px; margin-top: 7px; border-radius: var(--cs-radius-full); background: var(--cs-color-evidence-600); }}
.cs-toast {{
  min-height: 28px;
  color: var(--cs-color-text-secondary);
  border-left: 3px solid var(--cs-color-primary-600);
  padding: var(--cs-space-1) var(--cs-space-3);
  background: var(--cs-color-primary-50);
}}
@media (max-width: 980px) {{
  .cs-shell {{ grid-template-columns: 1fr; }}
  .cs-main {{ order: 1; display: flex; flex-direction: column; }}
  .cs-sidebar {{
    order: 3;
    position: static;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--cs-color-border-default);
    padding: var(--cs-space-4);
  }}
  .cs-nav {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .cs-nav-group {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .cs-nav-label {{ grid-column: 1 / -1; }}
  .cs-topbar {{ order: 2; position: static; padding: var(--cs-space-4); align-items: stretch; flex-direction: column; }}
  .cs-search {{ max-width: none; flex-basis: auto; }}
  .cs-content {{ order: 1; padding: var(--cs-space-4); }}
  .cs-grid-hero, .cs-grid-two, .cs-module-grid, .cs-brief-hero, .cs-search-workbench, .cs-artifact-hero, .cs-metadata-strip, .cs-inbox-workbench, .cs-collection-workbench, .cs-collection-summary, .cs-brief-fact-strip, .cs-brief-note-grid, .cs-action-review-strip {{ grid-template-columns: 1fr; }}
  .cs-page-head {{ margin-bottom: var(--cs-space-4); }}
  .cs-hero h1 {{ font-size: var(--cs-typography-pageTitle-fontSize); line-height: var(--cs-typography-pageTitle-lineHeight); }}
  .cs-home-intro {{ min-height: auto; }}
  .cs-home-canvas {{ padding: var(--cs-space-4); }}
  .cs-home-canvas > .cs-panel-header p {{ display: none; }}
  .cs-home-source-row, .cs-home-item, .cs-next-step {{ grid-template-columns: 1fr; }}
  .cs-drop {{ min-height: auto; padding: var(--cs-space-3); }}
  .cs-drop-target {{ grid-template-columns: auto minmax(0, 1fr); place-items: center start; text-align: left; }}
  .cs-drop-target p {{ display: none; }}
  .cs-drop textarea.cs-drop-input {{ min-height: 72px; }}
  .cs-ask-bar {{ grid-template-columns: 1fr; }}
  .cs-suggestion-row {{ grid-template-columns: 1fr; }}
  .cs-brief-actions {{ justify-content: flex-start; }}
  .cs-trust-ladder, .cs-action-summary {{ grid-template-columns: 1fr; }}
  .cs-diff-line, .cs-call-row, .cs-result-row, .cs-inbox-head, .cs-inbox-row, .cs-collection-row, .cs-action-object-row {{ grid-template-columns: 1fr; }}
  .cs-inbox-head {{ display: none; }}
  .cs-artifact-actions {{ justify-content: flex-start; }}
  .cs-document-page {{ min-height: auto; }}
  .cs-list-row {{ grid-template-columns: 1fr; }}
  .cs-detail-grid {{ grid-template-columns: 1fr; }}
}}
"""


def _css_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")


def _page(root: Path, title: str, active: str, content: str, ctx: dict[str, Any], q: str) -> str:
    css = _token_css(root)
    nav = _nav(active)
    topbar = _topbar(q, ctx)
    script = _home_script()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>{h(title)} - CornerStone</title>
  <style>{css}</style>
</head>
<body>
  <a class="cs-skip-link" href="#main-content">Skip to content</a>
  <div class="cs-shell" data-product-shell="cornerstone">
    <aside class="cs-sidebar" aria-label="CornerStone navigation">
      <div class="cs-brand">
        <div class="cs-brand-mark">CS</div>
        <div>
          <div class="cs-brand-name">CornerStone</div>
          <div class="cs-brand-sub">Personal / default</div>
        </div>
      </div>
      {nav}
    </aside>
    <main class="cs-main" id="main-content" tabindex="-1">
      {topbar}
      <div class="cs-content">
        {content}
      </div>
    </main>
  </div>
  {script}
</body>
</html>
"""


def _nav(active: str) -> str:
    primary = [
        ("/", "Home"),
        ("/search", "Search"),
        ("/artifacts", "Artifacts"),
        ("/claims", "Claims"),
        ("/actions", "Actions"),
    ]
    secondary = [
        ("/inbox", "Inbox"),
        ("/audit", "Audit"),
        ("/review", "Owner"),
    ]
    return f"""
<nav class="cs-nav">
  <div class="cs-nav-group">
    <div class="cs-nav-label">Workspace</div>
    {''.join(_nav_link(href, label, active) for href, label in primary)}
  </div>
  <div class="cs-nav-group">
    <div class="cs-nav-label">Operations</div>
    {''.join(_nav_link(href, label, active) for href, label in secondary)}
  </div>
</nav>
"""


def _nav_link(href: str, label: str, active: str) -> str:
    current = ' aria-current="page"' if href == active else ""
    return f'<a href="{h(href)}"{current}>{h(label)}</a>'


def _topbar(q: str, ctx: dict[str, Any]) -> str:
    count = len(ctx["artifacts"])
    label = f"{count} saved source" + ("" if count == 1 else "s")
    return f"""
<header class="cs-topbar">
  <form class="cs-search" action="/search" method="get">
    <span aria-hidden="true">Search</span>
    <input name="q" value="{h(q)}" placeholder="Search saved sources, claims, and drafts">
    <button type="submit">Go</button>
  </form>
  <div class="cs-row">
    {_chip(label, "saved")}
    {_chip("Local only", "searchable")}
  </div>
</header>
"""


def _home(ctx: dict[str, Any]) -> str:
    suggestions = "".join(
        f'<button class="cs-button secondary" type="button" data-ask-suggestion="{h(item)}">{h(item)}</button>'
        for item in ctx["suggestions"]
    )
    suggestions = suggestions or '<span class="cs-meta">Suggestions appear after sources or briefs exist.</span>'
    latest_brief = _latest_brief_block(ctx)
    recent = _recent_items_block(ctx)
    knowledge = _knowledge_states_block(ctx)
    next_steps = _suggested_next_steps_block(ctx)
    activity = _recent_activity_block(ctx)
    return f"""
<section class="cs-grid-hero cs-home-intro" data-product-surface="home">
  <div class="cs-stack">
    <div class="cs-hero cs-page-head">
      <div class="cs-kicker">Local workspace</div>
      <h1>Drop anything, or ask what we know</h1>
      <p>Save messy input, find what is already in the workspace, and shape a brief only when the sources are visible.</p>
    </div>
    <section class="cs-panel cs-home-canvas" aria-labelledby="home-workbench-title">
      <div class="cs-panel-header">
        <div>
          <h2 id="home-workbench-title">Start with a source or a question</h2>
          <p class="cs-muted">CornerStone keeps the original source visible, then lets drafts point back to what supports them.</p>
        </div>
        {_chip("Untrusted until checked", "underReview")}
      </div>
      <div class="cs-home-workspace">
        <form class="cs-drop" id="cs-drop-form">
          <div class="cs-home-source-row">
            <div class="cs-drop-target">
              <div class="cs-drop-mark" aria-hidden="true">In</div>
              <div>
                <strong>Save a source</strong>
                <p class="cs-muted">Drop a text file, choose a file, or paste notes below.</p>
              </div>
            </div>
            <div class="cs-row">
              <button class="cs-button" type="submit">Save source</button>
              <button class="cs-button secondary" type="button" id="cs-file-button">Choose file</button>
              <input id="cs-file-input" type="file" hidden>
            </div>
          </div>
          <textarea class="cs-drop-input" id="cs-drop-text" placeholder="Paste notes, an email, a renewal clause, or any text source"></textarea>
          <div class="cs-home-source-note">Dropped files are read locally by the browser before saving.</div>
          <div class="cs-toast" id="cs-drop-status" role="status">Ready for a source.</div>
        </form>
        <div class="cs-or-divider">or ask a question</div>
        <form class="cs-stack" id="cs-ask-form">
          <div class="cs-ask-bar" role="group" aria-label="Ask the workspace">
            <div>
              <strong>Ask the workspace</strong>
              <div class="cs-meta">Answers are drafts. Open sources before a decision.</div>
            </div>
            <input class="cs-field" id="cs-ask-input" placeholder="Ask about saved sources">
            <button class="cs-button" type="submit">Ask</button>
          </div>
          <div class="cs-suggestion-row">{suggestions}</div>
          <div class="cs-toast" id="cs-ask-status" role="status">No answer requested yet.</div>
        </form>
      </div>
    </section>
    <div class="cs-module-grid">
      {recent}
      <div class="cs-stack">
        {knowledge}
        {next_steps}
      </div>
    </div>
  </div>
  <aside class="cs-stack">
    {activity}
    {latest_brief}
    {_attention_block(ctx)}
  </aside>
</section>
"""


def _latest_brief_block(ctx: dict[str, Any]) -> str:
    if not ctx["briefs"]:
        return """
<section class="cs-panel flat">
  <div class="cs-panel-header"><h2>Latest brief</h2></div>
  <div class="cs-empty">No brief has been drafted from saved sources yet.</div>
</section>
"""
    brief = ctx["briefs"][0]
    label, state = _brief_label(brief)
    href = _detail_href("briefs", brief.get("brief_id"))
    summary = str(brief.get("summary") or "")
    if not summary and isinstance(brief.get("key_points"), list) and brief["key_points"]:
        summary = str(brief["key_points"][0])
    return f"""
<section class="cs-panel flat">
  <div class="cs-panel-header">
    <h2>Latest brief</h2>
    {_chip(label, state)}
  </div>
  <a class="cs-list-row" href="{h(href)}">
    <div>
      <h3>{h(_brief_title(brief))}</h3>
      <p>{h(_truncate(summary or "Open the brief to review source support.", 180))}</p>
      <div class="cs-meta">{h(_display_date(brief))}</div>
    </div>
    <span class="cs-meta">Open</span>
  </a>
</section>
"""


def _recent_items_block(ctx: dict[str, Any]) -> str:
    items: list[dict[str, Any]] = []
    for artifact in ctx["artifacts"][:3]:
        items.append(
            {
                "kind": "Source",
                "icon": "S",
                "title": _artifact_title(artifact),
                "detail": f"{_display_date(artifact)} - Original text preserved",
                "href": _detail_href("artifacts", artifact.get("artifact_id")),
                "label": "Searchable",
                "state": "searchable",
            }
        )
    for brief in ctx["briefs"][:2]:
        label, state = _brief_label(brief)
        items.append(
            {
                "kind": "Brief",
                "icon": "B",
                "title": _brief_title(brief),
                "detail": f"{_display_date(brief)} - Draft from visible sources",
                "href": _detail_href("briefs", brief.get("brief_id")),
                "label": label,
                "state": state,
            }
        )
    for claim in ctx["claims"][:2]:
        label, state = _claim_label(claim)
        items.append(
            {
                "kind": "Claim",
                "icon": "C",
                "title": _claim_title(claim),
                "detail": f"{_display_date(claim)} - Review source support",
                "href": _detail_href("claims", claim.get("claim_id")),
                "label": label,
                "state": state,
            }
        )
    for action in ctx["actions"][:2]:
        label, state = _action_label(action)
        items.append(
            {
                "kind": "Action",
                "icon": "A",
                "title": _action_title(action),
                "detail": f"{_display_date(action)} - Preview before approval",
                "href": _detail_href("actions", action.get("action_id")),
                "label": label,
                "state": state,
            }
        )
    items = sorted(items, key=lambda item: str(item.get("detail") or ""), reverse=True)[:4]
    if not items:
        return """
<section class="cs-panel flat">
  <div class="cs-panel-header"><h2>Recent items</h2></div>
  <div class="cs-empty">Saved sources will appear here. Nothing here yet - save your first source or ask after saving one.</div>
</section>
"""
    rows = "".join(
        f"""
<a class="cs-home-item" href="{h(str(item["href"]))}">
  <span class="cs-home-item-icon" aria-hidden="true">{h(str(item["icon"]))}</span>
  <span>
    <p>{h(str(item["kind"]))}</p>
    <h3>{h(str(item["title"]))}</h3>
    <p>{h(str(item["detail"]))}</p>
  </span>
  {_chip(str(item["label"]), str(item["state"]))}
</a>
"""
        for item in items
    )
    return f"""
<section class="cs-panel flat">
  <div class="cs-panel-header">
    <h2>Recent items</h2>
    <a class="cs-meta" href="/artifacts">View sources</a>
  </div>
  <div class="cs-home-item-list">{rows}</div>
</section>
"""


def _knowledge_states_block(ctx: dict[str, Any]) -> str:
    source_count = len(ctx["artifacts"])
    searchable_count = len(ctx["artifacts"])
    supported_count = sum(1 for record in [*ctx["briefs"], *ctx["claims"]] if _evidence_refs(record))
    return f"""
<section class="cs-panel flat">
  <div class="cs-panel-header">
    <h2>Knowledge states</h2>
    <span class="cs-meta">Local workspace</span>
  </div>
  <div class="cs-stat-list">
    <div class="cs-stat-row">
      <span class="cs-stat-icon">S</span>
      <div><strong>Saved</strong><div class="cs-meta">Original sources preserved</div></div>
      {_chip(str(source_count), "saved")}
    </div>
    <div class="cs-stat-row">
      <span class="cs-stat-icon">Q</span>
      <div><strong>Searchable</strong><div class="cs-meta">Ready for keyword search</div></div>
      {_chip(str(searchable_count), "searchable")}
    </div>
    <div class="cs-stat-row">
      <span class="cs-stat-icon">E</span>
      <div><strong>Source-backed</strong><div class="cs-meta">Drafts with visible support</div></div>
      {_chip(str(supported_count), "evidenceBacked")}
    </div>
  </div>
</section>
"""


def _suggested_next_steps_block(ctx: dict[str, Any]) -> str:
    steps: list[tuple[str, str, str, str]] = []
    if ctx["artifacts"]:
        steps.append(("S", "Ask about the latest source", "/search", "Search saved source text"))
    else:
        steps.append(("S", "Save your first source", "/", "Start with pasted notes or a text file"))
    if ctx["claims"]:
        steps.append(("C", "Review claims with source support", "/claims", "Check statements before approval"))
    if ctx["actions"]:
        steps.append(("A", "Review action previews", "/actions", "Confirm dry-run and approval state"))
    if ctx["inbox"]:
        steps.append(("I", "Open work that needs attention", "/inbox", "Triage review and approval items"))
    rows = "".join(
        f"""
<a class="cs-next-step" href="{h(href)}">
  <span class="cs-stat-icon" aria-hidden="true">{h(icon)}</span>
  <span><strong>{h(title)}</strong><span class="cs-meta">{h(detail)}</span></span>
  <span class="cs-meta">Open</span>
</a>
"""
        for icon, title, href, detail in steps[:3]
    )
    return f"""
<section class="cs-panel flat">
  <div class="cs-panel-header">
    <h2>Suggested next steps</h2>
    <a class="cs-meta" href="/inbox">Open inbox</a>
  </div>
  <div class="cs-next-step-list">{rows}</div>
</section>
"""


def _recent_activity_block(ctx: dict[str, Any]) -> str:
    rows = []
    for event in ctx["audit"][:5]:
        event_type = str(event.get("event_type") or "")
        rows.append(
            f"""
<div class="cs-timeline-item">
  <span class="cs-dot"></span>
  <div>
    <strong>{h(_plain_event(event_type))}</strong>
    <div class="cs-meta">{h(_display_date(event))}</div>
  </div>
</div>
"""
        )
    content = f'<div class="cs-timeline">{"".join(rows)}</div>' if rows else '<div class="cs-empty">Activity appears after you save, search, draft, or review work.</div>'
    return f"""
<section class="cs-panel flat">
  <div class="cs-panel-header">
    <h2>Recent activity</h2>
    <a class="cs-meta" href="/audit">View audit</a>
  </div>
  {content}
</section>
"""


def _attention_block(ctx: dict[str, Any]) -> str:
    if not ctx["inbox"]:
        return """
<section class="cs-panel flat">
  <div class="cs-panel-header"><h2>Needs attention</h2></div>
  <div class="cs-empty">Nothing is waiting on your decision.</div>
</section>
"""
    rows = "".join(_inbox_row(item) for item in ctx["inbox"][:3])
    return f"""
<section class="cs-panel flat">
  <div class="cs-panel-header">
    <h2>Needs attention</h2>
    <a class="cs-meta" href="/inbox">Open inbox</a>
  </div>
  <div class="cs-list">{rows}</div>
</section>
"""


def _search_page(ctx: dict[str, Any], q: str) -> str:
    results = _search_records(ctx, q)
    rows = "".join(results) if results else _search_empty(q)
    counts = _search_counts(ctx, q)
    count_tabs = "".join(
        f'<span class="cs-search-tab{" is-active" if index == 0 else ""}">{h(label)} <strong>{h(count)}</strong></span>'
        for index, (label, count) in enumerate(counts)
    )
    right_rail = _search_right_rail(counts, q)
    return f"""
<section data-product-surface="search">
  <div class="cs-page-head">
    <div class="cs-kicker">Search</div>
    <h1>Search the workspace</h1>
    <p>Keyword search over saved sources and drafts. Open the receipts before using a result for a decision.</p>
  </div>
  <div class="cs-search-workbench">
    <div>
      <section class="cs-panel">
        <form class="cs-search-hero" action="/search" method="get">
          <input name="q" value="{h(q)}" placeholder="Search saved sources, claims, action drafts, and briefs">
          <button class="cs-button" type="submit">Search</button>
        </form>
        <div class="cs-search-tabs" aria-label="Result type counts">{count_tabs}</div>
        <div class="cs-filter-row" aria-label="Current search filters">
          <span class="cs-filter-chip">Scope: personal/default</span>
          <span class="cs-filter-chip">Type: all visible</span>
          <span class="cs-filter-chip">Sort: keyword match</span>
        </div>
      </section>
      <div class="cs-result-list">{rows}</div>
    </div>
    {right_rail}
  </div>
</section>
"""


def _search_records(ctx: dict[str, Any], q: str) -> list[str]:
    query = q.lower().strip()
    if not query:
        return []
    rows: list[tuple[int, str]] = []
    for artifact in ctx["artifacts"]:
        text = " ".join([_artifact_title(artifact), str(artifact.get("_preview") or "")]).lower()
        score = _score(text, query)
        if score > 0:
            rows.append(
                (
                    score,
                    _search_result_row(
                        "Source",
                        "S",
                        _artifact_title(artifact),
                        str(artifact.get("_preview") or "Open to inspect the saved source."),
                        _detail_href("artifacts", artifact.get("artifact_id")),
                        "Searchable",
                        "searchable",
                        _display_date(artifact),
                    ),
                )
            )
    for brief in ctx["briefs"]:
        text = " ".join([_brief_title(brief), str(brief.get("summary") or ""), " ".join(str(item) for item in brief.get("key_points", []) if isinstance(item, str))]).lower()
        score = _score(text, query)
        if score > 0:
            label, state = _brief_label(brief)
            rows.append((score, _search_result_row("Brief", "B", _brief_title(brief), str(brief.get("summary") or "Brief draft"), _detail_href("briefs", brief.get("brief_id")), label, state, _display_date(brief))))
    for claim in ctx["claims"]:
        text = _claim_title(claim).lower()
        score = _score(text, query)
        if score > 0:
            label, state = _claim_label(claim)
            rows.append((score, _search_result_row("Claim", "C", _claim_title(claim), "Claim draft with linked evidence.", _detail_href("claims", claim.get("claim_id")), label, state, _display_date(claim))))
    for action in ctx["actions"]:
        dry_run = action.get("dry_run") if isinstance(action.get("dry_run"), dict) else {}
        text = " ".join([_action_title(action), str(dry_run.get("goal") or ""), str(dry_run.get("target") or "")]).lower()
        score = _score(text, query)
        if score > 0:
            label, state = _action_label(action)
            rows.append((score, _search_result_row("Action", "A", _action_title(action), str(dry_run.get("goal") or "Action draft"), _detail_href("actions", action.get("action_id")), label, state, _display_date(action))))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in rows[:20]]


def _score(text: str, query: str) -> int:
    terms = [term for term in re.split(r"[^a-zA-Z0-9]+", query.lower()) if term]
    score = 10 if query in text else 0
    score += sum(1 for term in terms if term in text)
    return score


def _search_counts(ctx: dict[str, Any], q: str) -> list[tuple[str, int]]:
    if not q.strip():
        return [
            ("All", 0),
            ("Sources", 0),
            ("Briefs", 0),
            ("Claims", 0),
            ("Actions", 0),
        ]
    query = q.lower().strip()
    source_count = sum(1 for artifact in ctx["artifacts"] if _score(" ".join([_artifact_title(artifact), str(artifact.get("_preview") or "")]).lower(), query) > 0)
    brief_count = sum(1 for brief in ctx["briefs"] if _score(" ".join([_brief_title(brief), str(brief.get("summary") or ""), " ".join(str(item) for item in brief.get("key_points", []) if isinstance(item, str))]).lower(), query) > 0)
    claim_count = sum(1 for claim in ctx["claims"] if _score(_claim_title(claim).lower(), query) > 0)
    action_count = sum(
        1
        for action in ctx["actions"]
        if _score(" ".join([_action_title(action), str((action.get("dry_run") if isinstance(action.get("dry_run"), dict) else {}).get("goal") or "")]).lower(), query) > 0
    )
    return [
        ("All", source_count + brief_count + claim_count + action_count),
        ("Sources", source_count),
        ("Briefs", brief_count),
        ("Claims", claim_count),
        ("Actions", action_count),
    ]


def _search_result_row(kind: str, icon: str, title: str, detail: str, href: str, label: str, state: str, date: str) -> str:
    return f"""
<a class="cs-result-row" href="{h(href)}">
  <span class="cs-result-icon" aria-hidden="true">{h(icon)}</span>
  <span class="cs-result-body">
    <span class="cs-result-meta"><span>{h(kind)}</span><span>{h(date)}</span></span>
    <h3>{h(title)}</h3>
    <p>{h(_truncate(detail, 240))}</p>
  </span>
  <span class="cs-row">{_chip(label, state)}</span>
</a>
"""


def _search_right_rail(counts: list[tuple[str, int]], q: str) -> str:
    stat_rows = "".join(
        f'<div class="cs-right-stat"><span>{h(label)}</span><strong>{h(count)}</strong></div>'
        for label, count in counts[1:]
    )
    suggestions = _search_followups(q, counts)
    suggestion_rows = "".join(
        f'<a class="cs-suggested-query" href="/search?q={quote(item)}"><span>?</span><span>{h(item)}</span></a>'
        for item in suggestions
    )
    return f"""
<aside class="cs-stack">
  <section class="cs-panel flat">
    <div class="cs-panel-header"><h2>What we found</h2>{_chip(str(counts[0][1] if counts else 0), "searchable")}</div>
    {stat_rows or '<div class="cs-empty">Run a search to see available receipt types.</div>'}
  </section>
  <section class="cs-panel flat">
    <h2 class="cs-section-title">Suggested follow-ups</h2>
    <div class="cs-stack">{suggestion_rows}</div>
  </section>
  <section class="cs-panel flat">
    <h2 class="cs-section-title">Receipt coverage</h2>
    <p class="cs-muted">Results are local keyword matches. Use the source and provenance panels before treating a draft as supported.</p>
  </section>
</aside>
"""


def _search_followups(q: str, counts: list[tuple[str, int]]) -> list[str]:
    query = q.strip() or "latest source"
    by_label = {label: count for label, count in counts}
    items = [
        f"What sources mention {query}?",
        f"Which claims depend on {query}?",
        f"Show action drafts related to {query}",
    ]
    if by_label.get("Briefs", 0):
        items.append(f"Open briefs that summarize {query}")
    if by_label.get("Sources", 0):
        items.append(f"Find provenance for {query}")
    return items[:5]


def _search_empty(q: str) -> str:
    if q.strip():
        return '<div class="cs-empty">No matching saved source or draft was found. Try fewer terms or save the source first.</div>'
    return '<div class="cs-empty">Enter a keyword to search saved sources, briefs, claims, and action drafts.</div>'


def _collection_summary(stats: list[tuple[str, int]]) -> str:
    cards = "".join(
        f"""
<div class="cs-collection-stat">
  <span class="cs-meta">{h(label)}</span>
  <strong>{h(value)}</strong>
</div>
"""
        for label, value in stats
    )
    return f'<div class="cs-collection-summary" aria-label="Collection summary">{cards}</div>'


def _collection_toolbar(label: str, count: int, filters: list[str]) -> str:
    chips = "".join(f'<span class="cs-filter-chip">{h(item)}</span>' for item in filters)
    return f"""
<div class="cs-collection-toolbar">
  <div>
    <strong>{h(label)}</strong>
    <div class="cs-meta">{h(count)} visible item{"s" if count != 1 else ""}</div>
  </div>
  <div class="cs-filter-row" style="margin-top: 0;">{chips}</div>
</div>
"""


def _collection_row(
    icon: str,
    title: str,
    detail: str,
    href: str,
    meta: list[tuple[str, str]],
    chips: list[tuple[str, str]],
) -> str:
    meta_row = "".join(f"<span>{h(label)}</span>" for label, _ in meta)
    chip_row = "".join(_chip(label, state) for label, state in chips)
    return f"""
<a class="cs-collection-row" href="{h(href)}">
  <span class="cs-collection-icon" aria-hidden="true">{h(icon)}</span>
  <span class="cs-collection-body">
    <span class="cs-collection-meta">{meta_row}</span>
    <h3>{h(title)}</h3>
    <p>{h(_truncate(detail, 240))}</p>
  </span>
  <span class="cs-row">{chip_row}</span>
</a>
"""


def _artifact_list_page(ctx: dict[str, Any]) -> str:
    artifacts = ctx["artifacts"]
    rows = "".join(
        _collection_row(
            "S",
            _artifact_title(artifact),
            str(artifact.get("_preview") or "Open to inspect the saved source."),
            _detail_href("artifacts", artifact.get("artifact_id")),
            [("Source", ""), (_display_date(artifact), "")],
            [("Searchable", "searchable"), ("Saved", "saved")],
        )
        for artifact in artifacts
    ) or '<div class="cs-empty">No saved sources yet. Start from Home.</div>'
    linked_count = sum(1 for record in [*ctx["briefs"], *ctx["claims"], *ctx["actions"]] for ref in _evidence_refs(record) if ref.startswith("artifact:"))
    return f"""
<section data-product-surface="artifacts">
  <div class="cs-page-head">
    <div class="cs-kicker">Artifacts</div>
    <h1>Saved sources</h1>
    <p>Every source remains preserved before any derived text, brief, claim, or action draft uses it.</p>
  </div>
  <div class="cs-collection-workbench">
    <div>
      {_collection_summary([("Saved sources", len(artifacts)), ("Linked refs", linked_count), ("Searchable", len(artifacts))])}
      {_collection_toolbar("Source register", len(artifacts), ["Scope: personal/default", "Type: all sources", "Sort: newest first"])}
      <div class="cs-collection-list">{rows}</div>
    </div>
    <aside class="cs-stack">
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Collection summary</h2>
        <p class="cs-muted">Original sources stay first. Open a source before using derived claims or action drafts.</p>
        <div class="cs-review-box">
          <a class="cs-button" href="/">Save another source</a>
          <a class="cs-button secondary" href="/search">Search sources</a>
        </div>
      </section>
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Source posture</h2>
        <dl class="cs-detail-grid">
          <dt>Preserved</dt><dd>{h(len(artifacts))}</dd>
          <dt>Linked uses</dt><dd>{h(linked_count)}</dd>
          <dt>Trust</dt><dd>Untrusted until checked</dd>
        </dl>
      </section>
    </aside>
  </div>
</section>
"""


def _brief_list_page(ctx: dict[str, Any]) -> str:
    rows = ""
    for brief in ctx["briefs"]:
        label, state = _brief_label(brief)
        summary = str(brief.get("summary") or "")
        if not summary and isinstance(brief.get("key_points"), list) and brief["key_points"]:
            summary = str(brief["key_points"][0])
        rows += _generic_row("Brief", _brief_title(brief), summary or "Brief draft", _detail_href("briefs", brief.get("brief_id")), label, state, _display_date(brief))
    rows = rows or '<div class="cs-empty">No briefs yet.</div>'
    return f"""
<section data-product-surface="briefs">
  <div class="cs-page-head">
    <div class="cs-kicker">Briefs</div>
    <h1>Brief drafts</h1>
    <p>Briefs stay in draft until source support is clear enough to defend.</p>
  </div>
  <div class="cs-list">{rows}</div>
</section>
"""


def _claim_list_page(ctx: dict[str, Any]) -> str:
    claims = ctx["claims"]
    rows = ""
    for claim in claims:
        label, state = _claim_label(claim)
        source_count = len([ref for ref in _evidence_refs(claim) if ref.startswith("artifact:")])
        rows += _collection_row(
            "C",
            _claim_title(claim),
            "Review source support before approval.",
            _detail_href("claims", claim.get("claim_id")),
            [("Claim", ""), (_display_date(claim), ""), (f"{source_count} source refs", "")],
            [(label, state), ("Review required", "underReview")],
        )
    rows = rows or '<div class="cs-empty">No claims yet.</div>'
    supported_count = sum(1 for claim in claims if _evidence_refs(claim))
    approved_count = sum(1 for claim in claims if str(claim.get("status") or "").lower() == "approved")
    return f"""
<section data-product-surface="claims">
  <div class="cs-page-head">
    <div class="cs-kicker">Claims</div>
    <h1>Claims that need source support</h1>
    <p>Promote only statements that can be traced back to saved sources.</p>
  </div>
  <div class="cs-collection-workbench">
    <div>
      {_collection_summary([("Claims", len(claims)), ("With sources", supported_count), ("Approved", approved_count)])}
      {_collection_toolbar("Claim review queue", len(claims), ["Scope: personal/default", "State: open and approved", "Sort: needs review first"])}
      <div class="cs-collection-list">{rows}</div>
    </div>
    <aside class="cs-stack">
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Review posture</h2>
        <p class="cs-muted">Claims should move from draft to evidence-backed to approved only when supporting sources are visible.</p>
        <div class="cs-review-box">
          <a class="cs-button" href="/inbox">Open review inbox</a>
          <a class="cs-button secondary" href="/artifacts">Check sources</a>
        </div>
      </section>
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Trust ladder</h2>
        {_claim_trust_ladder(bool(supported_count), bool(approved_count))}
      </section>
    </aside>
  </div>
</section>
"""


def _action_list_page(ctx: dict[str, Any]) -> str:
    actions = ctx["actions"]
    rows = ""
    approval_count = 0
    for action in actions:
        label, state = _action_label(action)
        if state == "underReview":
            approval_count += 1
        dry_run = action.get("dry_run") if isinstance(action.get("dry_run"), dict) else {}
        detail = str(dry_run.get("goal") or "Preview before any external write.")
        impact = dry_run.get("expected_impact") if isinstance(dry_run.get("expected_impact"), dict) else {}
        risk = str(impact.get("risk") or action.get("risk") or "review").title()
        rows += _collection_row(
            "A",
            _action_title(action),
            detail,
            _detail_href("actions", action.get("action_id")),
            [("Action", ""), (_display_date(action), ""), (f"{risk} risk", "")],
            [(label, state), ("Dry-run first", "searchable")],
        )
    rows = rows or '<div class="cs-empty">No action drafts yet.</div>'
    executed_count = sum(1 for action in actions if str(action.get("status") or "").lower() == "executed")
    return f"""
<section data-product-surface="actions">
  <div class="cs-page-head">
    <div class="cs-kicker">Actions</div>
    <h1>Action drafts</h1>
    <p>Every action starts with a preview and stays local until approval is clear.</p>
  </div>
  <div class="cs-collection-workbench">
    <div>
      {_collection_summary([("Drafts", len(actions)), ("Need approval", approval_count), ("Executed", executed_count)])}
      {_collection_toolbar("Action preview queue", len(actions), ["Scope: personal/default", "Mode: dry-run first", "Sort: approval risk first"])}
      <div class="cs-collection-list">{rows}</div>
    </div>
    <aside class="cs-stack">
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Dry-run posture</h2>
        <p class="cs-muted">Actions stay as previews until policy, approval, source support, and auditability are clear.</p>
        <div class="cs-review-box">
          <a class="cs-button" href="/inbox">Review approvals</a>
          <a class="cs-button secondary" href="/audit">Open audit trail</a>
        </div>
      </section>
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Action safeguards</h2>
        <dl class="cs-detail-grid">
          <dt>External sends</dt><dd>Hidden until approved</dd>
          <dt>Policy</dt><dd>Shown before execution</dd>
          <dt>Evidence</dt><dd>Linked source rail</dd>
        </dl>
      </section>
    </aside>
  </div>
</section>
"""


def _inbox_page(ctx: dict[str, Any]) -> str:
    items = ctx["inbox"]
    rows = "".join(_inbox_table_row(item, index == 0) for index, item in enumerate(items)) or _inbox_empty()
    detail = _inbox_detail_panel(items[0] if items else None)
    counts = _inbox_counts(items)
    return f"""
<section data-product-surface="inbox">
  <div class="cs-page-head">
    <div class="cs-kicker">Operations</div>
    <h1>Work that needs attention</h1>
    <p>Review drafts, source support, and action previews from one triage queue.</p>
  </div>
  <div class="cs-inbox-workbench">
    <div>
      <div class="cs-inbox-tabs" aria-label="Inbox queues">
        <span class="cs-inbox-tab is-active">Needs review {_chip(str(counts["needs_review"]), "underReview")}</span>
        <span class="cs-inbox-tab">Approval requests {_chip(str(counts["approval_requests"]), "draft")}</span>
        <span class="cs-inbox-tab">Policy blocked {_chip(str(counts["policy_blocked"]), "policyBlocked")}</span>
        <span class="cs-inbox-tab">Failed runs {_chip(str(counts["failed"]), "failed")}</span>
      </div>
      <div class="cs-inbox-toolbar">
        <div class="cs-filter-row" style="margin-top: 0;">
          <span class="cs-filter-chip">Type: all visible</span>
          <span class="cs-filter-chip">Owner: personal/default</span>
          <span class="cs-filter-chip">Priority: open first</span>
          <span class="cs-filter-chip">Trust/risk: visible labels</span>
        </div>
        <span class="cs-meta">{h(len(items))} open item{"s" if len(items) != 1 else ""}</span>
      </div>
      <div class="cs-inbox-table" role="list" aria-label="Operational inbox items">
        <div class="cs-inbox-head" aria-hidden="true">
          <span>Item</span><span>Type</span><span>Time</span><span>Priority</span><span>Trust / risk</span>
        </div>
        {rows}
      </div>
    </div>
    {detail}
  </div>
</section>
"""


def _inbox_counts(items: list[dict[str, str]]) -> dict[str, int]:
    return {
        "needs_review": sum(1 for item in items if item.get("queue") == "Needs review"),
        "approval_requests": sum(1 for item in items if item.get("queue") == "Approval requests"),
        "policy_blocked": sum(1 for item in items if item.get("state") == "policyBlocked"),
        "failed": sum(1 for item in items if item.get("state") == "failed"),
    }


def _inbox_table_row(item: dict[str, str], selected: bool = False) -> str:
    selected_class = " is-selected" if selected else ""
    priority_state = "failed" if item.get("priority") == "High" else "underReview" if item.get("priority") == "Medium" else "draft"
    return f"""
<a class="cs-inbox-row{selected_class}" href="{h(item["href"])}" role="listitem">
  <span class="cs-inbox-item-title">
    <span class="cs-inbox-icon" aria-hidden="true">{h(item.get("icon") or item["kind"][:1])}</span>
    <span>
      <strong>{h(item["title"])}</strong>
      <span class="cs-meta">{h(item["detail"])}</span>
    </span>
  </span>
  <span>{h(item.get("type") or item["kind"])}</span>
  <span class="cs-meta">{h(item["date"])}</span>
  <span>{_chip(item.get("priority") or "Medium", priority_state)}</span>
  <span>{_chip(item["label"], item["state"])}</span>
</a>
"""


def _inbox_detail_panel(item: dict[str, str] | None) -> str:
    if not item:
        return """
<aside class="cs-panel flat">
  <h2 class="cs-section-title">Selected item</h2>
  <div class="cs-empty">Save a source, draft a claim, or preview an action to fill this queue.</div>
</aside>
"""
    return f"""
<aside class="cs-panel flat cs-inbox-detail">
  <div>
    <div class="cs-kicker">Selected item</div>
    <h2>{h(item["title"])}</h2>
    <p class="cs-muted">{h(item["detail"])}</p>
  </div>
  <div class="cs-row">{_chip(item.get("queue") or "Needs review", "underReview")}{_chip(item.get("priority") or "Medium", "underReview")}{_chip(item["label"], item["state"])}</div>
  <section>
    <h2 class="cs-section-title">Overview</h2>
    <dl class="cs-detail-grid">
      <dt>Type</dt><dd>{h(item.get("type") or item["kind"])}</dd>
      <dt>Owner</dt><dd>{h(item.get("owner") or "Owner")}</dd>
      <dt>Updated</dt><dd>{h(item["date"])}</dd>
      <dt>Queue</dt><dd>{h(item.get("queue") or "Needs review")}</dd>
    </dl>
  </section>
  <section>
    <h2 class="cs-section-title">Next actions</h2>
    <div class="cs-inbox-actions">
      <a class="cs-button" href="{h(item["href"])}">Open item</a>
      <a class="cs-button secondary" href="/search?q={quote(item["title"])}">Review sources</a>
      <a class="cs-button secondary" href="/audit">Open audit trail</a>
    </div>
  </section>
</aside>
"""


def _inbox_empty() -> str:
    return '<div class="cs-empty">Nothing is waiting on you. Drafts, claims, and action previews appear here when they need a decision.</div>'


def _audit_page(ctx: dict[str, Any]) -> str:
    if not ctx["audit"]:
        rows = '<div class="cs-empty">No local activity has been recorded yet.</div>'
    else:
        rows = "".join(
            f"""
<div class="cs-timeline-item">
  <span class="cs-dot"></span>
  <div>
    <strong>{h(_plain_event(str(event.get("event_type") or "")))}</strong>
    <div class="cs-meta">{h(_display_date(event))}</div>
  </div>
</div>
"""
            for event in ctx["audit"][:40]
        )
        rows = f'<div class="cs-timeline">{rows}</div>'
    return f"""
<section data-product-surface="audit">
  <div class="cs-page-head">
    <div class="cs-kicker">Audit</div>
    <h1>Activity trail</h1>
    <p>Recent local activity is shown in plain language. The full signed ledger remains available through the API.</p>
  </div>
  <div class="cs-panel">{rows}</div>
</section>
"""


def _source_row(artifact: dict[str, Any], compact: bool = False) -> str:
    title = _artifact_title(artifact)
    preview = str(artifact.get("_preview") or "Open to inspect the saved source.")
    href = _detail_href("artifacts", artifact.get("artifact_id"))
    detail = _truncate(preview, 130 if compact else 220)
    return _generic_row("Source", title, detail, href, "Searchable", "searchable", _display_date(artifact))


def _generic_row(kind: str, title: str, detail: str, href: str, label: str, state: str, date: str) -> str:
    return f"""
<a class="cs-list-row" href="{h(href)}">
  <div>
    <div class="cs-meta">{h(kind)}</div>
    <h3>{h(title)}</h3>
    <p>{h(_truncate(detail, 220))}</p>
    <div class="cs-meta">{h(date)}</div>
  </div>
  <div class="cs-row">{_chip(label, state)}</div>
</a>
"""


def _inbox_row(item: dict[str, str]) -> str:
    return _generic_row(item["kind"], item["title"], item["detail"], item["href"], item["label"], item["state"], item["date"])


def _artifact_detail(ctx: dict[str, Any], store: Any, artifact: dict[str, Any]) -> str:
    title = _artifact_title(artifact)
    text = _safe_preview(store, artifact, 5000) or "No readable text preview is available for this source."
    linked = _linked_records(ctx, artifact.get("artifact_id"))
    source = artifact.get("source") if isinstance(artifact.get("source"), dict) else {}
    fingerprint = _fingerprint(artifact.get("checksum_sha256") or artifact.get("original_storage_ref"))
    source_label = _plain_source(source)
    keywords = _artifact_keywords(text, title)
    keyword_rows = "".join(
        f'<div class="cs-keyword-row"><span>{h(keyword)}</span>{_chip(str(count), "searchable")}</div>'
        for keyword, count in keywords
    )
    ask_query = quote(f"What matters in {title}")
    return f"""
<section class="cs-grid-two" data-product-surface="artifact-detail">
  <div class="cs-stack">
    <div class="cs-artifact-hero">
      <div class="cs-artifact-title">
        <div class="cs-kicker">Artifact / Original source</div>
        <h1>{h(title)}</h1>
        <p class="cs-muted">The original source stays primary. Drafts and decisions should point back here.</p>
      </div>
      <div class="cs-artifact-actions">
        <a class="cs-button secondary" href="/search?q={h(ask_query)}">Ask about this source</a>
        <a class="cs-button secondary" href="#linked-work">View linked work</a>
      </div>
    </div>
    <div class="cs-metadata-strip" aria-label="Source metadata">
      <div class="cs-metadata-item"><span class="cs-meta">Source</span><strong>{h(source_label)}</strong></div>
      <div class="cs-metadata-item"><span class="cs-meta">Saved</span><strong>{h(_display_date(artifact))}</strong></div>
      <div class="cs-metadata-item"><span class="cs-meta">Media type</span><strong>{h(str(artifact.get("media_type") or "text/plain"))}</strong></div>
      <div class="cs-metadata-item"><span class="cs-meta">Trust state</span><strong>Untrusted until checked</strong></div>
    </div>
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>Source text</h2>
          <p class="cs-muted">Readable local preview of the preserved source.</p>
        </div>
        <div class="cs-row">{_chip("Saved", "saved")}{_chip("Searchable", "searchable")}</div>
      </div>
      <div class="cs-document-frame">
        <article class="cs-document-page" aria-label="Original source text preview">
          <header class="cs-document-heading">
            <span class="cs-meta">Original source preview</span>
            <h3>{h(title)}</h3>
            <span class="cs-meta">{h(source_label)} / {h(_display_date(artifact))}</span>
          </header>
          <div class="cs-source-text">{h(text)}</div>
        </article>
      </div>
    </section>
  </div>
  <aside class="cs-stack">
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Source state</h2>
      <div class="cs-row">{_chip("Saved", "saved")}{_chip("Untrusted until checked", "underReview")}</div>
      <dl class="cs-detail-grid">
        <dt>Saved</dt><dd>{h(_display_date(artifact))}</dd>
        <dt>Source</dt><dd>{h(source_label)}</dd>
        <dt>Fingerprint</dt><dd>{h(fingerprint)}</dd>
      </dl>
      <p class="cs-muted">Keep this source visible before relying on derived drafts.</p>
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Readable preview</h2>
      <p class="cs-muted">{h(_truncate(text, 260))}</p>
    </section>
    <section class="cs-panel flat">
      <div class="cs-panel-header"><h2>Extracted keywords</h2>{_chip(str(len(keywords)), "searchable")}</div>
      <div class="cs-keyword-list">{keyword_rows or '<div class="cs-empty">No keyword preview is available.</div>'}</div>
    </section>
    {linked}
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Provenance</h2>
      <dl class="cs-detail-grid">
        <dt>Ingested from</dt><dd>{h(source_label)}</dd>
        <dt>Ingested</dt><dd>{h(_display_date(artifact))}</dd>
        <dt>Fingerprint</dt><dd>{h(fingerprint)}</dd>
      </dl>
    </section>
  </aside>
</section>
"""


def _linked_records(ctx: dict[str, Any], artifact_id: Any) -> str:
    if not artifact_id:
        return ""
    marker = f"artifact:{artifact_id}"
    rows: list[str] = []
    for brief in ctx["briefs"]:
        refs = _evidence_refs(brief)
        if marker in refs:
            label, state = _brief_label(brief)
            rows.append(_generic_row("Brief", _brief_title(brief), "Uses this source.", _detail_href("briefs", brief.get("brief_id")), label, state, _display_date(brief)))
    for claim in ctx["claims"]:
        refs = _evidence_refs(claim)
        if marker in refs:
            label, state = _claim_label(claim)
            rows.append(_generic_row("Claim", _claim_title(claim), "Uses this source.", _detail_href("claims", claim.get("claim_id")), label, state, _display_date(claim)))
    if not rows:
        return '<section class="cs-panel flat" id="linked-work"><h2 class="cs-section-title">Linked work</h2><div class="cs-empty">No briefs or claims are linked to this source yet.</div></section>'
    return f'<section class="cs-panel flat" id="linked-work"><h2 class="cs-section-title">Linked work</h2><div class="cs-list">{"".join(rows[:4])}</div></section>'


def _brief_detail(ctx: dict[str, Any], brief: dict[str, Any]) -> str:
    label, state = _brief_label(brief)
    summary = str(brief.get("summary") or "")
    key_points = [str(item) for item in brief.get("key_points", []) if isinstance(item, str)]
    findings = [str(item) for item in brief.get("findings", []) if isinstance(item, str)]
    gaps = [str(item) for item in brief.get("gaps", []) if isinstance(item, str)]
    gaps.extend(str(item) for item in brief.get("uncertainty", []) if isinstance(item, str))
    gaps = gaps or ["Check the linked sources before treating this as decision-ready."]
    source_items = _source_items(ctx, brief)
    source_list = _source_links_from_items(source_items)
    point_rows = _statement_rows(brief, key_points, source_items)
    finding_rows = _statement_rows(brief, findings, source_items, offset=len(key_points))
    gap_rows = "".join(f"<li>{h(point)}</li>" for point in gaps[:8])
    next_steps = [str(item) for item in brief.get("recommended_next_steps", []) if isinstance(item, str)]
    next_rows = "".join(f"<li>{h(item)}</li>" for item in next_steps[:4]) or "<li>Review the visible sources before promoting this draft.</li>"
    provenance = _brief_provenance(brief)
    source_count = len(source_items)
    mode = _plain_output_mode(str(brief.get("output_mode") or "draft"))
    finding_count = len(key_points) + len(findings)
    return f"""
<section class="cs-grid-two" data-product-surface="brief-detail">
  <div class="cs-stack">
    <div class="cs-brief-hero is-stacked">
      <div class="cs-brief-title">
        <div class="cs-kicker">Brief</div>
        <h1>{h(_brief_title(brief))}</h1>
        <p class="cs-muted">Read the answer, supporting sources, and limits together before using it for a decision.</p>
        <div class="cs-brief-meta">
          <span>{h(_display_date(brief))}</span>
          <span>{h(str(source_count))} visible source{"s" if source_count != 1 else ""}</span>
          <span>{h(str(finding_count))} drafted finding{"s" if finding_count != 1 else ""}</span>
        </div>
      </div>
      <div class="cs-brief-actions">
        {_chip(label, state)}
        <a class="cs-button secondary" href="#citation-trail">Review sources</a>
        <a class="cs-button secondary" href="/claims">Draft claim</a>
      </div>
    </div>
    <div class="cs-brief-fact-strip" aria-label="Brief status">
      <div class="cs-brief-fact"><span class="cs-meta">State</span><strong>{h(label)}</strong></div>
      <div class="cs-brief-fact"><span class="cs-meta">Source coverage</span><strong>{h(source_count)} visible</strong></div>
      <div class="cs-brief-fact"><span class="cs-meta">Mode</span><strong>{h(mode)}</strong></div>
      <div class="cs-brief-fact"><span class="cs-meta">Created</span><strong>{h(_display_date(brief))}</strong></div>
    </div>
    <section class="cs-panel cs-summary-card">
      <div class="cs-panel-header">
        <div>
          <h2>What we found</h2>
          <p class="cs-muted">Short answer, kept separate from source support.</p>
        </div>
      </div>
      <p>{h(summary or "No summary text was drafted yet.")}</p>
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>Findings with citations</h2>
          <p class="cs-muted">Each load-bearing statement stays close to the source that supports it.</p>
        </div>
        {_chip("Source coverage", "searchable")}
      </div>
      {point_rows}
      {f'<h2 class="cs-section-title">More findings</h2>{finding_rows}' if finding_rows else ''}
    </section>
    <div class="cs-brief-note-grid">
      <section class="cs-panel">
        <div class="cs-panel-header"><h2>Limits and gaps</h2>{_chip("Needs source check", "underReview")}</div>
        <ul class="cs-brief-note-list">{gap_rows}</ul>
      </section>
      <section class="cs-panel">
        <div class="cs-panel-header"><h2>Next steps</h2>{_chip("Draft only", "draft")}</div>
        <ul class="cs-brief-note-list">{next_rows}</ul>
      </section>
    </div>
  </div>
  <aside class="cs-stack">
    <section class="cs-panel flat" id="citation-trail">
      <h2 class="cs-section-title">Citation trail</h2>
      <p class="cs-muted">Open a source before promoting a finding.</p>
      {source_list}
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Provenance</h2>
      {provenance}
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Use this brief</h2>
      <div class="cs-review-box">
        <a class="cs-button" href="/claims">Draft claim from finding</a>
        <a class="cs-button secondary" href="/actions">Review action previews</a>
        <a class="cs-button secondary" href="/audit">Open audit trail</a>
      </div>
      <p class="cs-muted">Use this as a draft until each important statement is matched to a source span.</p>
    </section>
  </aside>
</section>
"""


def _claim_detail(ctx: dict[str, Any], claim: dict[str, Any]) -> str:
    label, state = _claim_label(claim)
    source_items = _source_items(ctx, claim)
    source_list = _evidence_picker_from_items(source_items)
    authority = claim.get("authority") if isinstance(claim.get("authority"), dict) else {}
    has_sources = bool(source_items)
    is_approved = str(claim.get("status") or "").lower() == "approved"
    review_note = "Evidence attached; request review before approval." if has_sources else "Add evidence before approval."
    approval_note = "Approval stays locked until review is recorded." if has_sources else "Approval stays locked until supporting evidence is attached."
    rationale = str(claim.get("rationale") or "").strip() or "No separate rationale has been drafted yet. Use the source rail before promoting this claim."
    return f"""
<section class="cs-grid-two" data-product-surface="claim-detail">
  <div class="cs-stack">
    <div class="cs-brief-hero">
      <div class="cs-brief-title">
        <div class="cs-kicker">Claim builder</div>
        <h1>{h(_claim_title(claim))}</h1>
        <div class="cs-brief-meta">
          <span>Draft conclusion</span>
          <span>{h(_display_date(claim))}</span>
          <span>{h(len(source_items))} source{"s" if len(source_items) != 1 else ""} attached</span>
        </div>
      </div>
      <div class="cs-row">{_chip(label, state)}{_chip("Review required before approval", "underReview")}</div>
    </div>
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>Trust ladder</h2>
          <p class="cs-muted">Draft freely, attach evidence, then request review before a decision uses this claim.</p>
        </div>
      </div>
      {_claim_trust_ladder(has_sources, is_approved)}
      <p class="cs-muted">{h(review_note)}</p>
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>Claim draft</h2>
          <p class="cs-muted">Statement and rationale stay editable until review.</p>
        </div>
      </div>
      <div class="cs-form-surface">
        <div class="cs-field-block">
          <span class="cs-meta">Claim statement</span>
          <p>{h(str(claim.get("statement") or ""))}</p>
        </div>
        <div class="cs-field-block">
          <span class="cs-meta">Rationale</span>
          <p>{h(rationale)}</p>
        </div>
      </div>
    </section>
  </div>
  <aside class="cs-stack">
    <section class="cs-panel flat">
      <div class="cs-panel-header">
        <div>
          <h2>Supporting evidence</h2>
          <p class="cs-muted">Only visible local sources are selectable here.</p>
        </div>
        {_chip(str(len(source_items)), "searchable")}
      </div>
      {source_list}
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Review controls</h2>
      <div class="cs-review-box">
        <a class="cs-button secondary" href="/claims">Save draft</a>
        <a class="cs-button" href="/inbox">Request review</a>
        <p class="cs-muted">{h(approval_note)}</p>
      </div>
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Authority</h2>
      <p class="cs-muted">{h(str(authority.get("blocked_reason") or "Owner approval is required before this claim becomes shared truth or drives autonomous action."))}</p>
    </section>
  </aside>
</section>
"""


def _action_detail(ctx: dict[str, Any], action: dict[str, Any]) -> str:
    label, state = _action_label(action)
    dry_run = action.get("dry_run") if isinstance(action.get("dry_run"), dict) else {}
    diff = dry_run.get("diff") if isinstance(dry_run.get("diff"), dict) else {}
    impact = dry_run.get("expected_impact") if isinstance(dry_run.get("expected_impact"), dict) else {}
    source_items = _source_items(ctx, action)
    source_list = _source_links_from_items(source_items)
    policy = action.get("policy_decision") if isinstance(action.get("policy_decision"), dict) else dry_run.get("policy_decision") if isinstance(dry_run.get("policy_decision"), dict) else {}
    decision_label = _plain_policy_decision(str(policy.get("decision") or ""))
    call_label = "Simulated in local mode" if int(impact.get("real_external_http_calls", 0) or 0) == 0 else "External write planned"
    connector = action.get("connector_boundary") if isinstance(action.get("connector_boundary"), dict) else {}
    connector_label = "Mediated preview" if connector.get("direct_provider_access") is False else "Provider access needs review"
    approval = action.get("approval") if isinstance(action.get("approval"), dict) else {}
    approval_required = bool(approval.get("required") or "approval" in decision_label.lower())
    approval_label = "Approval required" if approval_required else "Approval not required"
    risk_label = str(impact.get("risk") or action.get("risk") or "review").title()
    target = str(impact.get("target") or "Local preview only.")
    goal = str(dry_run.get("goal") or action.get("goal") or _action_title(action))
    approval_status = str(approval.get("status") or "pending")
    reason = str(approval.get("required_reason") or policy.get("reason") or "A reason is required before approval can move this preview toward execution.")
    return f"""
<section class="cs-grid-two" data-product-surface="action-detail">
  <div class="cs-stack">
    <div class="cs-brief-hero is-stacked">
      <div class="cs-brief-title">
        <div class="cs-kicker">Action preview</div>
        <h1>{h(_action_title(action))}</h1>
        <div class="cs-brief-meta">
          <span>Dry-run first</span>
          <span>{h(_display_date(action))}</span>
          <span>No external send yet</span>
        </div>
      </div>
      <div class="cs-brief-actions">
        {_chip("Preview (dry run)", "searchable")}
        {_chip(approval_label, "underReview")}
        {_chip(f"{risk_label} risk", "underReview")}
        <a class="cs-button secondary" href="/claims">Back to claims</a>
        <a class="cs-button" href="/inbox">Request approval</a>
      </div>
    </div>
    <div class="cs-action-review-strip" aria-label="Action review status">
      <div class="cs-action-review-card"><span class="cs-meta">Risk level</span><strong>{h(risk_label)}</strong></div>
      <div class="cs-action-review-card"><span class="cs-meta">Approval</span><strong>{h(approval_label)}</strong></div>
      <div class="cs-action-review-card"><span class="cs-meta">External calls</span><strong>{h(call_label)}</strong></div>
      <div class="cs-action-review-card"><span class="cs-meta">Status</span><strong>{h(approval_status)}</strong></div>
    </div>
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>Summary</h2>
          <p class="cs-muted">This is the proposed change, not an execution result.</p>
        </div>
        {_chip(label, state)}
      </div>
      <p>{h(goal)}</p>
      <div class="cs-action-summary">
        <div class="cs-action-metric"><span class="cs-meta">Trigger</span><span>Manual review</span></div>
        <div class="cs-action-metric"><span class="cs-meta">Target</span><span>{h(target)}</span></div>
        <div class="cs-action-metric"><span class="cs-meta">Policy</span><span>{h(decision_label)}</span></div>
      </div>
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header"><h2>Impacted objects</h2>{_chip("Will be reviewed", "underReview")}</div>
      <div class="cs-action-object-row">
        <span class="cs-action-object-icon" aria-hidden="true">A</span>
        <span>
          <strong>{h(target)}</strong>
          <span class="cs-meta">{h(connector_label)} / {h(call_label)}</span>
        </span>
        {_chip("Will be reviewed", "underReview")}
      </div>
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>Proposed changes</h2>
          <p class="cs-muted">Preview the before and after state before requesting approval.</p>
        </div>
        {_chip("Diff preview", "searchable")}
      </div>
      {_action_diff_view(diff)}
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>External calls</h2>
          <p class="cs-muted">Provider writes remain simulated until approval is clear.</p>
        </div>
        {_chip("Simulated in local mode", "draft")}
      </div>
      {_action_external_calls(impact, connector_label, call_label)}
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>Policy decision</h2>
          <p class="cs-muted">Policy is visible before approval or execution.</p>
        </div>
        {_chip(decision_label, "underReview")}
      </div>
      <div class="cs-policy-card">
        <strong>{h(decision_label)}</strong>
        <span>{h(str(policy.get("reason") or "This action is permitted only after review confirms the source, target, and risk."))}</span>
      </div>
    </section>
  </div>
  <aside class="cs-stack">
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Sources</h2>
      {source_list}
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Risk and approval</h2>
      <dl class="cs-detail-grid">
        <dt>Risk level</dt><dd>{h(risk_label)}</dd>
        <dt>Approval</dt><dd>{h(approval_label)}</dd>
        <dt>Status</dt><dd>{h(approval_status)}</dd>
      </dl>
      <div class="cs-review-box">
        <a class="cs-button" href="/inbox">Request approval</a>
        <div class="cs-approval-note">
          <strong>Required reason</strong>
          <span>{h(reason)}</span>
        </div>
        <p class="cs-muted">Execution is not shown as the primary action until approval is satisfied.</p>
      </div>
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Approval history</h2>
      <div class="cs-empty">No approvals have been recorded yet.</div>
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Auditability</h2>
      <p class="cs-muted">Dry-run, policy, approval, and execution records remain inspectable before this action can become a workflow result.</p>
    </section>
  </aside>
</section>
"""


def _claim_trust_ladder(has_sources: bool, is_approved: bool) -> str:
    evidence_class = "is-active" if has_sources or is_approved else "is-locked"
    approved_class = "is-active" if is_approved else "is-locked"
    evidence_note = "Supporting source is attached." if has_sources else "Attach at least one source."
    approved_note = "Decision-ready." if is_approved else "Requires review first."
    return f"""
<div class="cs-trust-ladder" aria-label="Claim trust ladder">
  <div class="cs-trust-step is-active">
    <strong>Draft</strong>
    <span class="cs-meta">Editable statement.</span>
  </div>
  <div class="cs-trust-step {evidence_class}">
    <strong>Evidence-backed</strong>
    <span class="cs-meta">{h(evidence_note)}</span>
  </div>
  <div class="cs-trust-step {approved_class}">
    <strong>Approved</strong>
    <span class="cs-meta">{h(approved_note)}</span>
  </div>
</div>
"""


def _action_diff_view(diff: dict[str, Any]) -> str:
    before = _plain_runtime_text(diff.get("before") or "No side effect applied.")
    after = _plain_runtime_text(diff.get("after") or "No external write has been performed.")
    return f"""
<div class="cs-diff-view" aria-label="Dry-run diff preview">
  <div class="cs-diff-line before"><span class="cs-meta">Before</span><span>{h(before)}</span></div>
  <div class="cs-diff-line after"><span class="cs-meta">After</span><span>{h(after)}</span></div>
</div>
<p class="cs-meta">Preview shown. Exact downstream formatting may vary after approval.</p>
"""


def _action_external_calls(impact: dict[str, Any], connector_label: str, call_label: str) -> str:
    expected = int(impact.get("expected_connector_calls", 0) or 0)
    target = str(impact.get("target") or "Local preview only.")
    if expected <= 0:
        return f"""
<div class="cs-call-row">
  <div>
    <strong>No external connector call planned</strong>
    <p class="cs-muted">This preview records the policy and audit envelope without a provider send.</p>
  </div>
  {_chip(call_label, "draft")}
</div>
"""
    return f"""
<div class="cs-call-row">
  <div>
    <strong>{h(connector_label)}</strong>
    <p class="cs-muted">Would create or update {h(target)} after approval. Simulated in local mode.</p>
  </div>
  {_chip(call_label, "draft")}
</div>
"""


def _source_links(ctx: dict[str, Any], refs: list[str]) -> str:
    return _source_links_from_items(_source_items_from_refs(ctx, refs))


def _source_links_from_items(items: list[dict[str, str]]) -> str:
    if not items:
        return '<div class="cs-empty">No linked source is visible in this workspace.</div>'
    rows = "".join(_source_card(item) for item in items[:8])
    return f'<div class="cs-list">{rows}</div>'


def _evidence_picker_from_items(items: list[dict[str, str]]) -> str:
    if not items:
        return '<div class="cs-empty">No linked source is visible in this workspace.</div>'
    rows = []
    for item in items[:6]:
        rows.append(
            f"""
<a class="cs-evidence-row" href="{h(item["href"])}">
  <span class="cs-checkmark" aria-hidden="true">&#10003;</span>
  <span>
    <strong>{h(item["title"])}</strong>
    <span class="cs-meta">{h(item["label"])} / {h(item["date"])} / {h(item["fingerprint"])}</span>
    <span class="cs-muted">{h(_truncate(item["snippet"], 120))}</span>
  </span>
</a>
"""
        )
    return f'<div class="cs-evidence-picker">{"".join(rows)}</div>'


def _source_card(item: dict[str, str]) -> str:
    return f"""
<details class="cs-list-row cs-source-card">
  <summary>
    <span class="cs-meta">{h(item["label"])}</span>
    <span>{h(item["title"])}</span>
  </summary>
  <p>{h(_truncate(item["snippet"], 260))}</p>
  <div class="cs-row">
    <a class="cs-button secondary" href="{h(item["href"])}">Open source</a>
    {_chip("Searchable", "searchable")}
  </div>
  <div class="cs-provenance">
    <dl class="cs-detail-grid">
      <dt>Saved</dt><dd>{h(item["date"])}</dd>
      <dt>Fingerprint</dt><dd>{h(item["fingerprint"])}</dd>
    </dl>
  </div>
</details>
"""


def _statement_rows(record: dict[str, Any], statements: list[str], source_items: list[dict[str, str]], offset: int = 0) -> str:
    if not statements:
        return '<div class="cs-empty">No findings were drafted yet.</div>'
    citation_rows = record.get("key_point_citations")
    citation_map: dict[str, list[str]] = {}
    if isinstance(citation_rows, list):
        for row in citation_rows:
            if not isinstance(row, dict):
                continue
            statement = str(row.get("statement") or "")
            refs = [str(ref) for ref in row.get("citation_refs", []) if isinstance(ref, str)]
            if statement:
                citation_map[statement] = refs
    rows = []
    for index, statement in enumerate(statements[:8], start=1):
        refs = citation_map.get(statement, [])
        labels = _citation_labels_for_refs(refs, source_items)
        chips = "".join(_chip(label, "searchable") for label in labels[:3])
        rows.append(
            f"""
<li class="cs-finding">
  <div class="cs-finding-head">
    <span class="cs-finding-index">Finding {h(index + offset)}</span>
    {_chip("Needs source check", "underReview") if not chips else _chip("Source linked", "evidenceBacked")}
  </div>
  <div>{h(statement)}</div>
  <div class="cs-citation-rail" aria-label="Sources for finding {index + offset}">{chips or _chip("Needs source check", "underReview")}</div>
</li>
"""
        )
    return f'<ol class="cs-finding-list">{"".join(rows)}</ol>'


def _source_items(ctx: dict[str, Any], record: dict[str, Any]) -> list[dict[str, str]]:
    links = record.get("evidence_links")
    refs = _evidence_refs(record)
    link_refs: list[str] = []
    snippets: dict[str, str] = {}
    storage_refs: dict[str, Any] = {}
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            artifact_ref = str(link.get("artifact_ref") or "")
            if artifact_ref.startswith("artifact:"):
                link_refs.append(artifact_ref)
                snippets[artifact_ref] = str(link.get("snippet") or "")
                storage_refs[artifact_ref] = link.get("original_storage_ref")
    combined = []
    for ref in [*link_refs, *refs]:
        if ref.startswith("artifact:") and ref not in combined:
            combined.append(ref)
    return _source_items_from_refs(ctx, combined, snippets=snippets, storage_refs=storage_refs)


def _source_items_from_refs(
    ctx: dict[str, Any],
    refs: list[str],
    *,
    snippets: dict[str, str] | None = None,
    storage_refs: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    snippets = snippets or {}
    storage_refs = storage_refs or {}
    items: list[dict[str, str]] = []
    for index, ref in enumerate([ref for ref in refs if ref.startswith("artifact:")][:8], start=1):
        artifact_id = ref.split(":", 1)[1]
        artifact = next((item for item in ctx["artifacts"] if item.get("artifact_id") == artifact_id), None)
        if not artifact:
            continue
        source_ref = storage_refs.get(ref) or artifact.get("checksum_sha256") or artifact.get("original_storage_ref")
        items.append(
            {
                "ref": ref,
                "label": f"Source {index}",
                "title": _artifact_title(artifact),
                "snippet": snippets.get(ref) or str(artifact.get("_preview") or "Open source."),
                "href": _detail_href("artifacts", artifact_id),
                "date": _display_date(artifact),
                "fingerprint": _fingerprint(source_ref),
            }
        )
    return items


def _citation_labels_for_refs(refs: list[str], source_items: list[dict[str, str]]) -> list[str]:
    labels: list[str] = []
    for ref in refs:
        if ref.startswith("artifact:"):
            item = next((item for item in source_items if item["ref"] == ref), None)
            if item:
                labels.append(item["label"])
    if labels:
        return labels
    if refs:
        return [item["label"] for item in source_items[:1]]
    return []


def _brief_provenance(brief: dict[str, Any]) -> str:
    model_run = brief.get("model_run") if isinstance(brief.get("model_run"), dict) else {}
    bundle = brief.get("evidence_bundle") if isinstance(brief.get("evidence_bundle"), dict) else {}
    mode = _plain_output_mode(str(brief.get("output_mode") or "draft"))
    model = str(model_run.get("generation_model") or "No model run recorded")
    created = _display_date(brief)
    source_count = len(bundle.get("artifact_refs", [])) if isinstance(bundle.get("artifact_refs"), list) else 0
    return f"""
<dl class="cs-detail-grid">
  <dt>Created</dt><dd>{h(created)}</dd>
  <dt>Mode</dt><dd>{h(mode)}</dd>
  <dt>Model</dt><dd>{h(model)}</dd>
  <dt>Sources</dt><dd>{h(source_count)}</dd>
</dl>
"""


def _plain_output_mode(value: str) -> str:
    normalized = value.lower()
    if normalized in {"ollama_generated", "model_cited", "citation_grounded", "evidence_backed"}:
        return "Citation-checked draft"
    if normalized in {"extractive_fallback", "template_fallback"}:
        return "Keyword summary"
    if normalized in {"insufficient_evidence"}:
        return "Needs more source support"
    return "Draft"


def _plain_source(source: dict[str, Any]) -> str:
    source_type = str(source.get("type") or source.get("source_type") or "local").replace("_", " ")
    source_ref = str(source.get("ref") or source.get("path") or "").strip()
    if not source_ref or source_ref in {"home.drop_text", "home-paste", "cli_text"}:
        return source_type.title()
    return _truncate(source_ref, 96)


def _artifact_keywords(text: str, title: str) -> list[tuple[str, int]]:
    stop_words = {
        "about",
        "after",
        "again",
        "also",
        "before",
        "being",
        "brief",
        "checked",
        "decision",
        "draft",
        "from",
        "have",
        "into",
        "local",
        "needs",
        "source",
        "text",
        "that",
        "their",
        "there",
        "this",
        "until",
        "what",
        "when",
        "with",
        "would",
        "your",
    }
    counts: dict[str, int] = {}
    for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", f"{title} {text}".lower()):
        if word in stop_words:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [(word.replace("-", " ").title(), count) for word, count in ranked[:6]]


def _fingerprint(value: Any) -> str:
    text = str(value or "").replace("sha256:", "").strip()
    if not text:
        return "Not recorded"
    return text[:12] + "..." if len(text) > 15 else text


def _plain_policy_decision(value: str) -> str:
    normalized = value.lower().replace("_", "-")
    if "approval" in normalized:
        return "Approval required"
    if "deny" in normalized or "block" in normalized:
        return "Policy blocked"
    if "allow" in normalized:
        return "Allowed after review"
    return "Review required"


def _plain_runtime_text(value: Any) -> str:
    text = str(value or "")
    replacements = {
        "external_writeback": "external writeback",
        "mock_connector": "simulated connector",
        "pending_approval": "pending approval",
        "high_risk_action_requires_approval": "high-risk action requires approval",
    }
    for needle, replacement in replacements.items():
        text = text.replace(needle, replacement)
    return text


def _evidence_refs(record: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("evidence_refs", "source_refs"):
        value = record.get(key)
        if isinstance(value, list):
            refs.extend(str(item) for item in value)
    evidence = record.get("evidence_bundle")
    if isinstance(evidence, dict):
        artifact_refs = evidence.get("artifact_refs")
        if isinstance(artifact_refs, list):
            refs.extend(str(item) for item in artifact_refs)
    action_evidence = record.get("evidence")
    if isinstance(action_evidence, dict):
        artifact_refs = action_evidence.get("artifact_refs")
        if isinstance(artifact_refs, list):
            refs.extend(str(item) for item in artifact_refs)
    dry_run = record.get("dry_run")
    if isinstance(dry_run, dict):
        refs.extend(str(item) for item in dry_run.get("evidence_refs", []) if isinstance(item, str))
    return refs


def _not_found(label: str) -> str:
    return f"""
<section data-product-surface="not-found">
  <div class="cs-page-head">
    <div class="cs-kicker">Not found</div>
    <h1>This {h(label)} is not available</h1>
    <p>It may be outside the current local workspace or may have been removed.</p>
  </div>
  <a class="cs-button secondary" href="/">Return home</a>
</section>
"""


def _internal_record_notice(label: str) -> str:
    return f"""
<section data-product-surface="owner-record">
  <div class="cs-page-head">
    <div class="cs-kicker">Owner record</div>
    <h1>This {h(label)} belongs in the owner area</h1>
    <p>It contains internal setup material, so it is kept out of the product workspace.</p>
  </div>
  <a class="cs-button secondary" href="/review">Open owner area</a>
</section>
"""


def _home_script() -> str:
    return """
<script>
(function () {
  const scope = {
    tenant_id: "local-dev",
    owner_id: "local-user",
    namespace_id: "personal",
    workspace_id: "default"
  };
  const dropForm = document.getElementById("cs-drop-form");
  const dropText = document.getElementById("cs-drop-text");
  const dropStatus = document.getElementById("cs-drop-status");
  const fileInput = document.getElementById("cs-file-input");
  const fileButton = document.getElementById("cs-file-button");
  const askForm = document.getElementById("cs-ask-form");
  const askInput = document.getElementById("cs-ask-input");
  const askStatus = document.getElementById("cs-ask-status");
  function setStatus(node, message) {
    if (node) node.textContent = message;
  }
  async function postJson(path, body) {
    const response = await fetch(path, {
      method: "POST",
      headers: {"content-type": "application/json", "accept": "application/json"},
      body: JSON.stringify(Object.assign({}, scope, body))
    });
    const payload = await response.json();
    if (!response.ok || payload.status === "failed" || payload.status === "denied") {
      throw new Error((payload.errors && payload.errors[0] && payload.errors[0].message) || "Request failed.");
    }
    return payload;
  }
  async function saveText(text, sourceRef) {
    setStatus(dropStatus, "Saving source...");
    const payload = {};
    payload.text = text;
    payload["source" + "_ref"] = sourceRef || "home-paste";
    const saved = await postJson("/artifacts", payload);
    const artifact = saved.artifact || {};
    const id = artifact["artifact" + "_id"];
    setStatus(dropStatus, "Saved. Opening source...");
    if (id) window.location.href = "/artifacts/" + encodeURIComponent(id) + "?view=html";
  }
  if (dropForm && dropText) {
    dropForm.addEventListener("submit", function (event) {
      event.preventDefault();
      const text = dropText.value.trim();
      if (!text) {
        setStatus(dropStatus, "Paste text before saving.");
        return;
      }
      saveText(text, "home-paste").catch(error => setStatus(dropStatus, error.message));
    });
    ["dragenter", "dragover"].forEach(name => dropForm.addEventListener(name, function (event) {
      event.preventDefault();
      dropForm.classList.add("is-hot");
    }));
    ["dragleave", "drop"].forEach(name => dropForm.addEventListener(name, function () {
      dropForm.classList.remove("is-hot");
    }));
    dropForm.addEventListener("drop", function (event) {
      event.preventDefault();
      const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
      if (!file) return;
      file.text().then(text => saveText(text, file.name)).catch(error => setStatus(dropStatus, error.message));
    });
  }
  if (fileButton && fileInput) {
    fileButton.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", function () {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;
      file.text().then(text => saveText(text, file.name)).catch(error => setStatus(dropStatus, error.message));
    });
  }
  document.querySelectorAll("[data-ask-suggestion]").forEach(button => {
    button.addEventListener("click", function () {
      if (askInput) askInput.value = button.getAttribute("data-ask-suggestion") || "";
      if (askInput) askInput.focus();
    });
  });
  if (askForm && askInput) {
    askForm.addEventListener("submit", async function (event) {
      event.preventDefault();
      const question = askInput.value.trim();
      if (!question) {
        setStatus(askStatus, "Enter a question first.");
        return;
      }
      try {
        setStatus(askStatus, "Checking saved sources...");
        const started = await postJson("/conversations", {message: question});
        const conversation = started.conversation || {};
        const id = conversation["conversation" + "_id"];
        if (!id) throw new Error("Conversation was not saved.");
        const answered = await postJson("/conversations/" + encodeURIComponent(id) + "/answers", {question});
        const answer = answered.answer || {};
        const text = answer.answer || "Draft answer saved. Open the linked sources before using it.";
        setStatus(askStatus, "Draft answer: " + text);
      } catch (error) {
        setStatus(askStatus, error.message);
      }
    });
  }
}());
</script>
"""

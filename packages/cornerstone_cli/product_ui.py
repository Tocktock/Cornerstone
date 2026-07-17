from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cornerstone_cli.artifacts import MAX_BROWSER_UPLOAD_BYTES, artifact_presentation
from cornerstone_cli.product_access import (
    CollectionViewRequest,
    ProductAccessApplication,
    RecordReadRequest,
    SearchRequest,
)
from cornerstone_cli.product_visibility import (
    CONTEXT_ONLY_SOURCE_TYPES,
    INTERNAL_PRODUCT_SOURCE_TYPES,
    context_only_artifact as _context_only_artifact,
    internal_product_lineage as _internal_product_lineage,
    internal_product_record as _internal_product_record,
)
from cornerstone_cli.ui.icons import icon
from cornerstone_cli.ui.language import record_action, record_icon
from cornerstone_cli.ui.styles import render_styles as _token_css, style_asset
from cornerstone_cli.validators import redact_text

PRODUCT_LIST_ROUTES = {"/", "/search", "/artifacts", "/briefs", "/claims", "/actions", "/inbox", "/audit"}
PRODUCT_RECORD_FAMILIES = frozenset({"artifact", "brief", "claim", "action", "memory", "answer"})
PRODUCT_ROUTE_RECORD_FAMILIES = {
    "/": PRODUCT_RECORD_FAMILIES,
    "/search": frozenset(),
    "/artifacts": frozenset({"artifact", "brief", "claim", "action"}),
    "/briefs": frozenset({"artifact", "brief"}),
    "/claims": frozenset({"artifact", "brief", "claim"}),
    "/actions": frozenset({"artifact", "brief", "claim", "action"}),
    # Artifact visibility is the root of transitive fixture/owner-only lineage.
    # Inbox needs it even though it does not render Artifact rows directly.
    "/inbox": PRODUCT_RECORD_FAMILIES,
    "/audit": frozenset(),
}
PRODUCT_DETAIL_RECORD_FAMILIES = {
    "artifacts": frozenset({"artifact", "brief", "claim"}),
    "briefs": frozenset({"artifact", "claim", "action"}),
    "claims": frozenset({"artifact", "brief", "claim"}),
    "memories": frozenset({"artifact", "memory"}),
    "actions": frozenset({"artifact", "brief", "claim", "action"}),
    "answers": frozenset({"artifact", "answer"}),
}
PRODUCT_SEARCH_TYPES = [
    ("all", "All"),
    ("sources", "Sources"),
    ("briefs", "Briefs"),
    ("claims", "Claims"),
    ("actions", "Actions"),
]
INBOX_LANES = [
    ("needs-review", "Needs review"),
    ("evidence-gaps", "Evidence gaps"),
    ("approval-requests", "Approval requests"),
    ("policy-blocked", "Policy blocked"),
    ("failed-runs", "Failed runs"),
]
INBOX_PAGE_SIZE = 20
VS5_DECISION_SOURCE_MAX_BYTES = 128 * 1024
VS5_DECISION_TOTAL_SOURCE_MAX_BYTES = 512 * 1024
AUDIT_LIFECYCLES = [
    ("all", "All activity"),
    ("created", "Created"),
    ("opened", "Opened"),
    ("approved", "Approved"),
    ("blocked", "Blocked or denied"),
    ("executed", "Executed"),
    ("other", "Other"),
]
PRODUCT_DETAIL_ROUTES = {"artifacts", "briefs", "claims", "memories", "actions", "answers"}

REFERENCE_IMAGE_ROWS = [
    ("cornerstone-reference-01-vendor-detail.png", "Vendor object detail", "Dormant direction", "Entity/object explorer structure only; do not add this surface during the VS5 scope freeze."),
    ("cornerstone-reference-02-operations-inbox.png", "Operations inbox", "Active surface", "Lane tabs, triage table, selected preview, and next actions."),
    ("cornerstone-reference-03-admin-connectors.png", "Admin connectors", "Owner-only direction", "Connector governance stays contained in the owner area."),
    ("cornerstone-reference-04-search-results.png", "Search results", "Active surface", "Universal search, scoped filters, traceable results, and suggested follow-ups."),
    ("cornerstone-reference-05-claim-draft-supporting-evidence.png", "Claim draft", "Active surface", "Claim statement, trust ladder, supporting evidence, and review controls."),
    ("cornerstone-reference-06-artifact-viewer.png", "Artifact viewer", "Active surface", "Original artifact primary, source metadata, derived keywords, linked work, and provenance."),
    ("cornerstone-reference-07-home-upload-ask.png", "Home workspace", "Active surface", "Drop zone, ask box, recent items, knowledge states, and next steps."),
    ("cornerstone-reference-08-action-dry-run-approval.png", "Action dry-run", "Active surface", "Dry-run impact, proposed changes, policy decision, risk, approval, and auditability."),
]

def h(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _script_json(value: Any) -> str:
    return (
        json.dumps(value, separators=(",", ":"), sort_keys=True)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _scope_label(scope: Any, *, include_owner: bool = False) -> str:
    if not isinstance(scope, dict):
        return "personal / default"
    values = []
    if include_owner:
        values.append(str(scope.get("owner_id") or "local-user"))
    values.extend(
        [
            str(scope.get("namespace_id") or "personal"),
            str(scope.get("workspace_id") or "default"),
        ]
    )
    return " / ".join(values)


def _owner_initials(scope: Any) -> str:
    owner = str(scope.get("owner_id") or "local-user") if isinstance(scope, dict) else "local-user"
    parts = [part for part in re.split(r"[^a-zA-Z0-9]+", owner) if part]
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return owner[:2].upper() or "LU"


def render_product_page(
    root: Path,
    store: Any,
    scope: dict[str, str],
    path: str,
    query: dict[str, list[str]],
) -> str:
    route = path if path in PRODUCT_LIST_ROUTES else "/"
    q = (query.get("q") or [""])[-1].strip()
    page_access: dict[str, Any] | None = None
    page_access_failed = False
    if route != "/search" or not q:
        allowed_filter_keys = {
            "/search": {"type", "page"},
            "/inbox": {"lane", "item", "page"},
            "/audit": {"record", "lifecycle", "page"},
        }.get(route, set())
        try:
            page_access = ProductAccessApplication(store).view_collection(
                CollectionViewRequest(
                    surface=route,
                    scope=scope,
                    filters={
                        key: values[-1]
                        for key, values in query.items()
                        if values and key in allowed_filter_keys
                    },
                )
            )
        except Exception:
            page_access = None
            page_access_failed = True
    ctx = _build_context(
        store,
        scope,
        record_families=PRODUCT_ROUTE_RECORD_FAMILIES[route],
        include_audit=route in {"/", "/audit"},
        verify_audit_integrity=route == "/audit",
    )
    if page_access:
        ctx["page_audit_ref"] = str((page_access.get("audit_refs") or [""])[0])
    if page_access_failed:
        ctx["load_errors"].append("page access audit")
        for key in ("artifacts", "briefs", "claims", "actions", "memories", "answers", "inbox", "audit", "audit_all"):
            ctx[key] = []
        ctx["inbox_total"] = 0
        return _page(
            root,
            "Access unavailable",
            route,
            _access_audit_unavailable(),
            ctx,
            redact_text(q),
        )
    requested_search_type = (query.get("type") or ["all"])[-1].strip().lower()
    search_type = requested_search_type if requested_search_type in {value for value, _ in PRODUCT_SEARCH_TYPES} else "all"
    if route == "/":
        _ensure_artifact_previews(ctx)
        title = "Home"
        content = _home(ctx)
        active = "/"
    elif route == "/search":
        try:
            page = max(1, int((query.get("page") or ["1"])[-1]))
        except ValueError:
            page = 1
        snapshot_id = (query.get("snapshot") or [""])[-1].strip() or None
        search_outcome: dict[str, Any] | None = None
        if q:
            try:
                search_outcome = ProductAccessApplication(store).search(
                    SearchRequest(
                        query=q,
                        scope=scope,
                        mode="workspace",
                        type_filter=search_type,
                        page=page,
                        page_size=20,
                        snapshot_id=snapshot_id,
                        excluded_source_types=frozenset(INTERNAL_PRODUCT_SOURCE_TYPES | CONTEXT_ONLY_SOURCE_TYPES),
                    )
                )
            except Exception:
                search_outcome = {"status": "access_unavailable"}
        title = "Search"
        content = _search_page(ctx, q, search_type, search_outcome)
        active = "/search"
    elif route == "/artifacts":
        _ensure_artifact_previews(ctx)
        title = "Artifacts"
        content = _artifact_list_page(ctx)
        active = "/artifacts"
    elif route == "/briefs":
        title = "Briefs"
        content = _brief_list_page(ctx)
        active = "/briefs"
    elif route == "/claims":
        title = (
            "Decisions"
            if ctx.get("claims") and all(_is_decision_draft(claim) for claim in ctx["claims"])
            else "Claims"
        )
        content = _claim_list_page(ctx)
        active = "/claims"
    elif route == "/actions":
        title = "Actions"
        content = _action_list_page(ctx)
        active = "/actions"
    elif route == "/inbox":
        title = "Inbox"
        requested_lane = (query.get("lane") or [""])[-1].strip().lower()
        requested_item = (query.get("item") or [""])[-1].strip()
        try:
            requested_page = max(1, int((query.get("page") or ["1"])[-1]))
        except ValueError:
            requested_page = 1
        lane, lane_items, visible_items, selected_item, page_info = _select_inbox_item(
            ctx["inbox"],
            requested_lane,
            requested_item,
            requested_page,
        )
        ctx["selected_inbox_item"] = selected_item
        _load_selected_product_loop(ctx, selected_item)
        content = _inbox_page(ctx, lane, lane_items, visible_items, page_info)
        active = "/inbox"
    else:
        title = "Audit"
        record_filter = (query.get("record") or [""])[-1].strip()
        requested_lifecycle = (query.get("lifecycle") or ["all"])[-1].strip().lower()
        lifecycle_filter = requested_lifecycle if requested_lifecycle in {value for value, _ in AUDIT_LIFECYCLES} else "all"
        try:
            page = max(1, int((query.get("page") or ["1"])[-1]))
        except ValueError:
            page = 1
        content = _audit_page(ctx, record_filter, lifecycle_filter, page)
        active = "/audit"
    return _page(root, title, active, content, ctx, redact_text(q) if route == "/search" else q)


def render_owner_review_page(
    root: Path,
    store: Any,
    scope: dict[str, str],
    readiness: dict[str, Any],
) -> str:
    ctx = _build_context(store, scope, include_audit=True)
    content = _owner_review_page(ctx, readiness)
    return _page(root, "Owner", "/review", content, ctx, "")


def render_owner_reference_images_page(
    root: Path,
    store: Any,
    scope: dict[str, str],
) -> str:
    ctx = _build_context(store, scope, record_families=frozenset())
    content = _owner_reference_images_page(root)
    return _page(root, "Reference Images", "/review", content, ctx, "")


def render_product_not_found(
    root: Path,
    store: Any,
    scope: dict[str, str],
) -> str:
    ctx = _build_context(store, scope, record_families=frozenset())
    return _page(root, "Page not found", "/", _not_found("page"), ctx, "")


def render_product_detail(
    root: Path,
    store: Any,
    scope: dict[str, str],
    kind: str,
    record_id: str,
) -> tuple[int, str]:
    try:
        access_result = ProductAccessApplication(store).read(
            RecordReadRequest(
                record_kind=kind,
                record_id=record_id,
                scope=scope,
                reason="product_ui_detail",
            )
        )
    except Exception:
        # Detail pages have the same fail-closed audit contract as collections:
        # if the read receipt cannot be recorded, do not render record data or
        # reveal whether the requested identifier exists.
        ctx = _build_context(store, scope, record_families=frozenset())
        ctx["load_errors"].append("page access audit")
        return 503, _page(
            root,
            "Access unavailable",
            "/",
            _access_audit_unavailable(),
            ctx,
            "",
        )
    record = access_result.get("record") if isinstance(access_result.get("record"), dict) else None
    ctx = _build_context(
        store,
        scope,
        record_families=(PRODUCT_DETAIL_RECORD_FAMILIES.get(kind, frozenset()) if record else frozenset()),
        include_audit=bool(record),
    )
    audit_event = access_result.get("audit_event") if isinstance(access_result.get("audit_event"), dict) else None
    audit_ref = f"audit:{audit_event['event_id']}" if audit_event and audit_event.get("event_id") else ""
    evidence_kind = {
        "artifacts": "artifact",
        "briefs": "brief",
        "claims": "claim",
        "memories": "memory",
        "actions": "action",
        "answers": "answer",
    }.get(kind, kind)
    evidence_ref = f"{evidence_kind}:{record_id}"
    if kind == "artifacts":
        if record:
            record = dict(record)
            record["_preview"] = _safe_preview(store, record, 5000, ctx["load_errors"], "source preview")
        if record and _internal_product_record(record, ctx["internal_lineage_refs"]):
            content = _internal_record_notice("source")
            title = "Owner record"
            active = "/artifacts"
            return 200, _page(root, title, active, content, ctx, "")
        content = _artifact_detail(ctx, store, record) if record else _not_found("source")
        title = _artifact_title(record) if record else "Source not found"
        active = "/artifacts"
    elif kind == "briefs":
        if record and _internal_product_record(record, ctx["internal_lineage_refs"]):
            content = _internal_record_notice("brief")
            title = "Owner record"
            active = "/"
            return 200, _page(root, title, active, content, ctx, "")
        content = _brief_detail(ctx, record) if record else _not_found("brief")
        title = str(record.get("title") or "Brief") if record else "Brief not found"
        active = "/"
    elif kind == "claims":
        if record and _internal_product_record(record, ctx["internal_lineage_refs"]):
            content = _internal_record_notice("claim")
            title = "Owner record"
            active = "/claims"
            return 200, _page(root, title, active, content, ctx, "")
        content = _claim_detail(ctx, record) if record else _not_found("claim")
        title = _truncate(str(record.get("statement") or "Claim"), 72) if record else "Claim not found"
        active = "/claims"
    elif kind == "memories":
        if record and _internal_product_record(record, ctx["internal_lineage_refs"]):
            content = _internal_record_notice("memory candidate")
            title = "Owner record"
            active = "/inbox"
            return 200, _page(root, title, active, content, ctx, "")
        content = _memory_detail(ctx, record) if record else _not_found("memory candidate")
        title = _truncate(str(record.get("title") or record.get("statement") or "Memory candidate"), 72) if record else "Memory candidate not found"
        active = "/inbox"
    elif kind == "actions":
        if record and _internal_product_record(record, ctx["internal_lineage_refs"]):
            content = _internal_record_notice("action")
            title = "Owner record"
            active = "/actions"
            return 200, _page(root, title, active, content, ctx, "")
        content = _action_detail(ctx, record) if record else _not_found("action")
        title = _action_title(record) if record else "Action not found"
        active = "/actions"
    elif kind == "answers":
        content = _answer_detail(ctx, record) if record else _not_found("saved answer")
        title = _truncate(str(record.get("question") or "Saved answer"), 72) if record else "Saved answer not found"
        active = "/"
    else:
        return 404, _page(root, "Not found", "/", _not_found("record"), ctx, "")
    status = 200 if record else 404
    if record and audit_ref:
        content += _access_receipt(evidence_ref, audit_ref)
    return status, _page(root, title, active, content, ctx, "")


def _build_context(
    store: Any,
    scope: dict[str, str],
    *,
    record_families: frozenset[str] | set[str] | None = None,
    include_audit: bool = False,
    verify_audit_integrity: bool = False,
) -> dict[str, Any]:
    load_errors: list[str] = []
    selected_families = PRODUCT_RECORD_FAMILIES if record_families is None else frozenset(record_families)
    unsupported_families = selected_families - PRODUCT_RECORD_FAMILIES
    if unsupported_families:
        raise ValueError(f"Unsupported product record families: {sorted(unsupported_families)}")
    artifacts = (
        _recent(_safe_records(lambda: store._artifact_records(scope), load_errors, "saved sources"))
        if "artifact" in selected_families
        else []
    )
    briefs = (
        _recent(_safe_records(lambda: store._brief_records(scope), load_errors, "briefs"))
        if "brief" in selected_families
        else []
    )
    claims = (
        _recent(_safe_records(lambda: store._claim_records(scope), load_errors, "claims"))
        if "claim" in selected_families
        else []
    )
    actions = (
        _recent(_safe_records(lambda: store._action_records(scope), load_errors, "action drafts"))
        if "action" in selected_families
        else []
    )
    memories = (
        _recent(_safe_records(lambda: store._memory_records(scope), load_errors, "memory drafts"))
        if "memory" in selected_families
        else []
    )
    answers = (
        _recent(_safe_records(lambda: store._answer_records(scope), load_errors, "saved answers"))
        if "answer" in selected_families
        else []
    )
    audit_all = (
        _recent(
            [
                event
                for event in _safe_records(lambda: store._all_audit_events(), load_errors, "activity history")
                if _same_scope(event.get("scope") if isinstance(event.get("scope"), dict) else event, scope)
            ]
        )
        if include_audit
        else []
    )
    audit = [event for event in audit_all if event.get("event_type") != "product.collection.read"][:80]
    audit_integrity = {"status": "not_verified", "event_count": len(audit_all), "errors": []}
    if verify_audit_integrity:
        try:
            audit_integrity = store.verify_audit()
        except Exception:
            load_errors.append("audit integrity")
    internal_lineage_refs, internal_record_objects = _internal_product_lineage(
        [artifacts, briefs, claims, actions, memories]
    )
    artifacts = [
        record
        for record in artifacts
        if id(record) not in internal_record_objects and not _context_only_artifact(record)
    ]
    briefs = [record for record in briefs if id(record) not in internal_record_objects]
    claims = [record for record in claims if id(record) not in internal_record_objects]
    actions = [record for record in actions if id(record) not in internal_record_objects]
    memories = [record for record in memories if id(record) not in internal_record_objects]
    inbox = _inbox_items(briefs, claims, actions, memories, scope=scope)
    inbox_families = {"brief", "claim", "action", "memory"}
    inbox_total = len(inbox) if inbox_families <= selected_families else None
    return {
        "store": store,
        "scope": scope,
        "artifacts": artifacts,
        "briefs": briefs,
        "claims": claims,
        "actions": actions,
        "memories": memories,
        "answers": answers,
        "audit": audit,
        "audit_all": audit_all,
        "audit_integrity": audit_integrity,
        "loaded_record_families": sorted(selected_families),
        "internal_lineage_refs": internal_lineage_refs,
        "load_errors": list(dict.fromkeys(load_errors)),
        "suggestions": _suggestions(artifacts, briefs, claims),
        "inbox": inbox,
        "inbox_total": inbox_total,
        "selected_product_loop": None,
    }


def _ensure_artifact_previews(ctx: dict[str, Any], limit: int = 260) -> None:
    store = ctx.get("store")
    errors = ctx.get("load_errors") if isinstance(ctx.get("load_errors"), list) else []
    for artifact in ctx.get("artifacts", []):
        if isinstance(artifact, dict) and "_preview" not in artifact:
            artifact["_preview"] = _safe_preview(store, artifact, limit, errors, "source preview")


def _load_selected_product_loop(ctx: dict[str, Any], selected: dict[str, Any] | None = None) -> None:
    if selected is None:
        return
    try:
        ctx["selected_product_loop"] = ctx["store"].project_product_loop_for_record(
            ctx["scope"],
            selected_kind=str(selected.get("record_kind") or ""),
            selected_id=str(selected.get("record_id") or ""),
        )
    except Exception:
        errors = ctx.get("load_errors") if isinstance(ctx.get("load_errors"), list) else []
        if "work journey" not in errors:
            errors.append("work journey")
        ctx["load_errors"] = errors


def _inbox_lane_items(items: list[dict[str, Any]], lane: str) -> list[dict[str, Any]]:
    queue = dict(INBOX_LANES).get(lane)
    if lane == "evidence-gaps":
        return [item for item in items if int(item.get("evidence_gap_count", 0) or 0) > 0]
    return [item for item in items if item.get("queue") == queue]


def _select_inbox_item(
    items: list[dict[str, Any]],
    requested_lane: str,
    requested_item: str,
    requested_page: int = 1,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, dict[str, int]]:
    valid_lanes = {value for value, _ in INBOX_LANES}
    explicit_lane = requested_lane in valid_lanes
    lane = requested_lane if explicit_lane else "needs-review"
    lane_items = _inbox_lane_items(items, lane)
    if not lane_items and not explicit_lane:
        for candidate, _ in INBOX_LANES:
            candidate_items = _inbox_lane_items(items, candidate)
            if candidate_items:
                lane, lane_items = candidate, candidate_items
                break
    requested_index = next(
        (
            index
            for index, item in enumerate(lane_items)
            if str(item.get("record_ref") or "") == requested_item
        ),
        None,
    )
    page_count = max(1, (len(lane_items) + INBOX_PAGE_SIZE - 1) // INBOX_PAGE_SIZE)
    page = (
        requested_index // INBOX_PAGE_SIZE + 1
        if requested_index is not None
        else min(max(requested_page, 1), page_count)
    )
    start = (page - 1) * INBOX_PAGE_SIZE
    visible_items = lane_items[start : start + INBOX_PAGE_SIZE]
    selected = (
        lane_items[requested_index]
        if requested_index is not None
        else visible_items[0]
        if visible_items
        else None
    )
    return (
        lane,
        lane_items,
        visible_items,
        selected,
        {
            "page": page,
            "page_count": page_count,
            "page_start": start + 1 if lane_items else 0,
            "page_end": min(start + INBOX_PAGE_SIZE, len(lane_items)),
            "result_count": len(lane_items),
        },
    )


def _safe_records(read: Any, errors: list[str] | None = None, label: str = "workspace records") -> list[dict[str, Any]]:
    try:
        records = read()
    except Exception:
        if errors is not None:
            errors.append(label)
        return []
    return [record for record in records if isinstance(record, dict)]


def _safe_preview(
    store: Any,
    artifact: dict[str, Any],
    limit: int = 240,
    errors: list[str] | None = None,
    label: str = "source preview",
) -> str:
    try:
        return store.derived_text_preview(artifact, limit)
    except Exception:
        if errors is not None:
            errors.append(label)
        return ""


def _artifact_state(ctx: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    store = ctx.get("store")
    try:
        text_available = bool(store.derived_text_available(artifact))
        original_available = bool(store.original_available(artifact))
    except Exception:
        text_available = False
        original_available = False
        errors = ctx.get("load_errors") if isinstance(ctx.get("load_errors"), list) else []
        if "source representation" not in errors:
            errors.append("source representation")
    return artifact_presentation(
        artifact,
        text_available=text_available,
        original_available=original_available,
    )


def _safe_source_text(
    store: Any,
    artifact: dict[str, Any],
    limit: int = 50_000,
    errors: list[str] | None = None,
) -> tuple[str, bool]:
    try:
        return store.derived_text_content(artifact, limit)
    except Exception:
        if errors is not None:
            errors.append("source text")
        return "", False


def _same_scope(value: Any, scope: dict[str, str]) -> bool:
    return isinstance(value, dict) and all(value.get(key) == expected for key, expected in scope.items())


def _recent(records: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    sorted_records = sorted(records, key=_record_time_key, reverse=True)
    return sorted_records[:limit] if limit else sorted_records


def _record_time_key(record: dict[str, Any]) -> str:
    for key in ("updated_at", "created_at", "occurred_at", "timestamp", "recorded_at", "decided_at"):
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


def _short_ref(value: Any, length: int = 12) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= length:
        return text
    return f"{text[:length]}..."


def _breakable_ref(value: Any, chunk_size: int = 16) -> str:
    """Escape a technical reference while preserving mobile wrap points."""

    text = str(value or "")
    if not text:
        return ""
    size = max(4, chunk_size)
    return "<wbr>".join(h(text[index : index + size]) for index in range(0, len(text), size))


def _artifact_title(record: dict[str, Any] | None) -> str:
    if not record:
        return "Source"
    source = record.get("source")
    source_ref = ""
    if isinstance(source, dict):
        filename = str(source.get("filename") or "").strip()
        if filename:
            return _truncate(redact_text(filename), 72)
        source_ref = str(source.get("ref") or "").strip()
    else:
        source_ref = str(record.get("source_ref") or source or "").strip()
    preview = str(record.get("_preview") or "").strip()
    if preview:
        return _truncate(preview, 72)
    if source_ref and source_ref not in {"local_file", "cli_text", "home.drop_text"}:
        return _truncate(redact_text(source_ref), 72)
    media = str(record.get("media_type") or "source").replace("/", " ")
    return media.title()


def _brief_title(record: dict[str, Any]) -> str:
    return _truncate(str(record.get("title") or "Untitled brief"), 96)


def _claim_title(record: dict[str, Any]) -> str:
    return _truncate(str(record.get("statement") or "Untitled claim"), 120)


def _is_decision_draft(record: dict[str, Any]) -> bool:
    return str(record.get("product_role") or "") == "decision_draft"


def _action_title(record: dict[str, Any] | None) -> str:
    if not record:
        return "Action"
    dry_run = record.get("dry_run") if isinstance(record.get("dry_run"), dict) else {}
    return _truncate(str(record.get("title") or dry_run.get("goal") or "Action draft"), 96)


def _plain_event(event_type: str) -> str:
    labels = {
        "artifact.ingest": "Source saved",
        "artifact.ingested": "Source saved",
        "artifact.read": "Source opened",
        "search.run": "Search run",
        "search.snapshot.created": "Search saved",
        "search.snapshot.read": "Search opened",
        "evidence_bundle.created": "Supporting evidence prepared",
        "brief.create": "Brief drafted",
        "brief.created": "Brief drafted",
        "brief.read": "Brief opened",
        "claim.create": "Claim drafted",
        "claim.draft.created": "Claim drafted",
        "claim.read": "Claim opened",
        "claim.approve": "Claim approved",
        "claim.approved": "Claim approved",
        "claim.approval.denied": "Claim approval blocked",
        "memory.draft.created": "Memory candidate drafted",
        "memory.owner_approved.created": "Owner-approved memory recorded",
        "memory.candidate.creation.denied": "Memory authority request blocked",
        "memory.read": "Memory candidate opened",
        "action.create": "Action drafted",
        "action.card.proposed": "Action proposed",
        "action.read": "Action opened",
        "action.dry_run": "Action previewed",
        "action.dry_run.read": "Action preview opened",
        "action.approve": "Action approved",
        "action.approved": "Action approved",
        "action.approval.denied": "Action approval blocked",
        "action.execute": "Action executed",
        "action.executed": "Action executed",
        "action.execution.denied": "Action execution blocked",
        "action.preview.creation.denied": "Action preview boundary blocked",
        "conversation.start": "Ask started",
        "conversation.started": "Ask started",
        "conversation.answer": "Draft answer saved",
        "conversation.answer.created": "Draft answer saved",
        "product.mission_control.viewed": "Review queue opened",
        "product.loop.viewed": "Work journey opened",
        "product.collection.read": "Page opened",
        "mission.contract.created": "Decision path prepared",
        "mission.activated": "Decision path activated",
        "workspace.mode.set": "Workspace mode recorded",
    }
    if event_type in labels:
        return labels[event_type]
    return "Recorded activity"


def _audit_subject_label(event: dict[str, Any]) -> str:
    subject = event.get("subject") if isinstance(event.get("subject"), dict) else {}
    subject_type = str(subject.get("type") or "record").lower()
    return {
        "artifact": "Source",
        "search_snapshot": "Saved search",
        "evidence_bundle": "Supporting evidence",
        "brief": "Brief",
        "claim": "Claim",
        "memory": "Knowledge draft",
        "action": "Action",
        "mission": "Decision path",
        "mission_control": "Review queue",
        "product_loop": "Work journey",
        "conversation": "Ask",
    }.get(subject_type, subject_type.replace("_", " ").title())


def _audit_family(event_type: str) -> str:
    if event_type.startswith("artifact."):
        return "Source"
    if event_type.startswith("search.") or event_type.startswith("evidence_bundle.") or event_type.startswith("brief."):
        return "Evidence"
    if event_type.startswith("claim.") or event_type.startswith("mission.") or event_type.startswith("workspace."):
        return "Decision"
    if event_type.startswith("action."):
        return "Action"
    if event_type.startswith("conversation."):
        return "Ask"
    return "Ledger"


def _audit_detail(event: dict[str, Any], position: int) -> str:
    subject = event.get("subject") if isinstance(event.get("subject"), dict) else {}
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    event_scope = event.get("scope") if isinstance(event.get("scope"), dict) else event
    event_id = _short_ref(event.get("event_id"), 16)
    event_hash = _short_ref(event.get("event_hash"), 16)
    previous_hash = _short_ref(event.get("previous_hash"), 16)
    subject_ref = f"{subject.get('type') or 'record'}:{subject.get('id') or 'unknown'}"
    detail_keys = ", ".join(str(key).replace("_", " ") for key in sorted(details.keys())[:4]) or "No extra fields"
    return f"""
<details class="cs-audit-detail">
  <summary>Raw event detail</summary>
  <div class="cs-audit-raw-grid">
    <div class="cs-audit-raw-item"><span class="cs-meta">Ledger position</span><strong>{position}</strong></div>
    <div class="cs-audit-raw-item"><span class="cs-meta">Event</span><strong>{h(event_id)}</strong></div>
    <div class="cs-audit-raw-item"><span class="cs-meta">Subject</span><strong>{_breakable_ref(subject_ref)}</strong></div>
    <div class="cs-audit-raw-item"><span class="cs-meta">Event type</span><strong>{h(str(event.get("event_type") or "recorded"))}</strong></div>
    <div class="cs-audit-raw-item"><span class="cs-meta">Event hash</span><strong>{h(event_hash)}</strong></div>
    <div class="cs-audit-raw-item"><span class="cs-meta">Previous hash</span><strong>{h(previous_hash)}</strong></div>
    <div class="cs-audit-raw-item"><span class="cs-meta">Detail fields</span><strong>{h(detail_keys)}</strong></div>
    <div class="cs-audit-raw-item"><span class="cs-meta">Scope</span><strong>{h(_scope_label(event_scope))}</strong></div>
  </div>
</details>
"""


def _brief_label(record: dict[str, Any]) -> tuple[str, str]:
    evidence_integrity = record.get("evidence_integrity") if isinstance(record.get("evidence_integrity"), dict) else {}
    if evidence_integrity.get("status") == "failed":
        return "Integrity issue", "failed"
    output_mode = str(record.get("output_mode") or "").lower()
    trust = str(record.get("trust_label") or record.get("status") or "").lower()
    if output_mode in {"model_cited", "citation_grounded", "ollama_generated"} and trust in {"evidence_backed", "corroborated"}:
        return "Source-backed", "evidenceBacked"
    if output_mode in {"extractive_fallback", "template_fallback"} or trust in {"extractive_fallback", "template_fallback"}:
        return "Keyword summary", "draft"
    if trust == "insufficient_evidence":
        return "Needs sources", "underReview"
    return "Draft", "draft"


def _brief_source_count(record: dict[str, Any]) -> int:
    refs = _evidence_refs(record)
    artifact_refs = {ref for ref in refs if ref.startswith("artifact:")}
    if artifact_refs:
        return len(artifact_refs)
    return len(set(refs))


def _brief_label_state(record: dict[str, Any]) -> tuple[str, str]:
    evidence_integrity = record.get("evidence_integrity") if isinstance(record.get("evidence_integrity"), dict) else {}
    if evidence_integrity.get("status") == "failed":
        return "Evidence integrity failed", "failed"
    label_check = record.get("label_check") if isinstance(record.get("label_check"), dict) else {}
    if record.get("presented_as_fact") is True and label_check.get("earned_evidence_backed") is True:
        return "Fact label earned", "evidenceBacked"
    if label_check:
        return "Draft label", "underReview"
    return "Draft label", "draft"


def _brief_citation_refs(record: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("load_bearing_statements", "key_point_citations", "recommended_next_step_citations"):
        rows = record.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            refs.extend(str(ref) for ref in row.get("citation_refs", []) if isinstance(ref, str))
    citation_refs = record.get("citation_refs")
    if isinstance(citation_refs, list):
        refs.extend(str(ref) for ref in citation_refs if isinstance(ref, str))
    return list(dict.fromkeys(refs))


def _brief_citation_receipt(record: dict[str, Any], source_items: list[dict[str, str]]) -> dict[str, Any]:
    refs = _brief_citation_refs(record)
    resolved = [_citation_item_for_ref(ref, source_items, record) for ref in refs]
    resolved_count = sum(1 for item in resolved if item)
    check_refs = record.get("citation_check_refs")
    check_count = len(check_refs) if isinstance(check_refs, list) else 0
    ref_kinds = sorted({_citation_ref_kind(ref) for ref in refs}) or ["none"]
    return {
        "citation_refs_count": len(refs),
        "resolved_citation_count": resolved_count,
        "unresolved_citation_count": max(0, len(refs) - resolved_count),
        "citation_check_refs_count": check_count,
        "citation_ref_kind": ", ".join(ref_kinds),
    }


def _brief_summary_text(record: dict[str, Any]) -> str:
    summary = str(record.get("summary") or "").strip()
    if summary:
        return summary
    key_points = record.get("key_points")
    if isinstance(key_points, list):
        for item in key_points:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return "Brief draft"


def _claim_label(record: dict[str, Any]) -> tuple[str, str]:
    evidence_integrity = record.get("evidence_integrity") if isinstance(record.get("evidence_integrity"), dict) else {}
    if evidence_integrity.get("status") == "failed":
        return "Integrity issue", "failed"
    status = str(record.get("status") or "").lower()
    if status == "approved":
        return "Approved", "approved"
    if status in {"policy_blocked", "blocked"}:
        return "Policy blocked", "policyBlocked"
    if _claim_evidence_backed_earned(record):
        return "Evidence-backed", "evidenceBacked"
    if _evidence_refs(record):
        return "Source support", "searchable"
    return "Draft", "draft"


def _claim_evidence_backed_earned(record: dict[str, Any]) -> bool:
    label_check = record.get("label_check") if isinstance(record.get("label_check"), dict) else {}
    support = record.get("statement_support") if isinstance(record.get("statement_support"), dict) else {}
    trust = str(record.get("trust_label") or record.get("trust_state") or record.get("status") or "").lower()
    legacy_label_earned = label_check.get("earned_evidence_backed") is True
    statement_support_earned = bool(
        support.get("status") == "passed"
        and support.get("citation_integrity_state") == "passed"
        and support.get("citations_bound_to_evidence_revision") is True
        and support.get("approval_eligible") is True
    )
    return trust == "evidence_backed" and (legacy_label_earned or statement_support_earned)


def _claim_semantic_support_verified(record: dict[str, Any]) -> bool:
    support = record.get("statement_support") if isinstance(record.get("statement_support"), dict) else {}
    return bool(
        support.get("semantic_support_verified") is True
        and support.get("semantic_faithfulness_state") == "passed"
    )


def _action_lifecycle(record: dict[str, Any]) -> dict[str, Any]:
    approval = record.get("approval") if isinstance(record.get("approval"), dict) else {}
    execution = record.get("execution") if isinstance(record.get("execution"), dict) else {}
    result = execution.get("result") if isinstance(execution.get("result"), dict) else {}
    approval_status = str(approval.get("status") or "not_recorded").lower()
    execution_status = str(execution.get("status") or "not_started").lower()
    result_status = str(result.get("status") or "").lower()
    record_status = str(record.get("status") or "").lower()
    dry_run = record.get("dry_run") if isinstance(record.get("dry_run"), dict) else {}
    policy = record.get("policy_decision") if isinstance(record.get("policy_decision"), dict) else {}
    if not policy and isinstance(dry_run.get("policy_decision"), dict):
        policy = dry_run["policy_decision"]
    policy_decision = str(policy.get("decision") or "").lower()
    policy_requires_approval = policy.get("approval_required") is True or policy_decision in {
        "requires_approval",
        "require_approval",
        "escalate",
    }
    if execution_status == "executed":
        stage = "executed"
    elif execution_status in {"failed", "error"} or result_status in {"failed", "error"}:
        stage = "failed"
    elif (
        record_status in {"policy_blocked", "blocked", "denied"}
        or policy_decision in {"deny", "denied", "blocked", "policy_blocked", "escalate", "escalated"}
        or execution_status.startswith("blocked")
        or execution_status in {"denied", "policy_denied"}
        or result_status in {"blocked", "denied", "policy_denied"}
    ):
        stage = "blocked"
    elif approval_status == "approved" or execution_status == "ready_to_execute":
        stage = "approved"
    elif approval_status in {"pending", "not_approved", "required"} or policy_requires_approval:
        stage = "pending"
    else:
        stage = "draft"
    return {
        "stage": stage,
        "approval": approval,
        "approval_status": approval_status,
        "execution": execution,
        "execution_status": execution_status,
        "result": result,
        "result_status": result_status,
    }


def _action_label(record: dict[str, Any]) -> tuple[str, str]:
    stage = _action_lifecycle(record)["stage"]
    if stage == "executed":
        return "Executed", "executed"
    if stage == "failed":
        return "Failed", "failed"
    if stage == "blocked":
        return "Policy blocked", "policyBlocked"
    if stage == "approved":
        return "Approved", "approved"
    if stage == "pending":
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
        suggestions.append("What evidence supports the latest Brief?")
    if artifacts:
        suggestions.append("What changed in the latest saved source?")
    if claims:
        suggestions.append("Which Claims still need source support?")
    return suggestions[:3]


def _inbox_items(
    briefs: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    *,
    scope: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    owner_label = str((scope or {}).get("owner_id") or "Owner")
    for brief in briefs:
        label, state = _brief_label(brief)
        gaps = [
            str(value)
            for key in ("gaps", "uncertainty")
            for value in (brief.get(key) if isinstance(brief.get(key), list) else [])
            if isinstance(value, str) and value.strip()
        ]
        gap_count = len(gaps)
        items.append(
            {
                "kind": "Brief",
                "title": _brief_title(brief),
                "detail": f"Review {gap_count} evidence gap{'s' if gap_count != 1 else ''} and linked sources before sharing.",
                "label": label,
                "state": state,
                "href": _detail_href("briefs", brief.get("brief_id")),
                "date": _display_date(brief),
                "created_at": _record_time_key(brief),
                "evidence_gap_count": gap_count,
                "queue": "Needs review",
                "priority": "Medium",
                "owner": owner_label,
                "type": "Brief",
                "icon": "B",
                "record_kind": "brief",
                "record_id": str(brief.get("brief_id") or ""),
                "record_ref": f"brief:{brief.get('brief_id')}" if brief.get("brief_id") else "",
            }
        )
    for claim in [claim for claim in claims if str(claim.get("status") or "").lower() != "approved"]:
        label, state = _claim_label(claim)
        decision_draft = _is_decision_draft(claim)
        product_kind = "Decision draft" if decision_draft else "Claim"
        evidence_integrity = claim.get("evidence_integrity") if isinstance(claim.get("evidence_integrity"), dict) else {}
        source_supported = str(
            (claim.get("statement_support") if isinstance(claim.get("statement_support"), dict) else {}).get("status")
            or ""
        ) == "source_supported"
        items.append(
            {
                "kind": product_kind,
                "title": _claim_title(claim),
                "detail": (
                    "The linked evidence chain failed integrity verification and must be repaired before review."
                    if evidence_integrity.get("status") == "failed"
                    else "Source links are preserved. Review this Decision draft against the Brief before using it."
                    if decision_draft
                    else
                    "Source support is attached; statement-level semantic review is required before owner approval."
                    if source_supported
                    else "Needs source support before semantic review and owner approval."
                ),
                "label": label,
                "state": state,
                "href": _detail_href("claims", claim.get("claim_id")),
                "date": _display_date(claim),
                "created_at": _record_time_key(claim),
                "queue": "Needs review",
                "priority": "High" if state == "draft" else "Medium",
                "owner": owner_label,
                "type": product_kind,
                "icon": "C",
                "record_kind": "claim",
                "record_id": str(claim.get("claim_id") or ""),
                "record_ref": f"claim:{claim.get('claim_id')}" if claim.get("claim_id") else "",
            }
        )
    pending_actions = [action for action in actions if _action_lifecycle(action)["stage"] != "executed"]
    for action in pending_actions:
        stage = str(_action_lifecycle(action)["stage"])
        label, state = _action_label(action)
        queue = (
            "Failed runs"
            if stage == "failed"
            else "Policy blocked"
            if stage == "blocked"
            else "Approval requests"
            if stage == "pending"
            else "Needs review"
        )
        detail = (
            "Inspect the failed run and recovery record before retrying."
            if stage == "failed"
            else "Resolve the policy block before any execution attempt."
            if stage == "blocked"
            else "Preview the impact before any external write."
        )
        items.append(
            {
                "kind": "Action",
                "title": _action_title(action),
                "detail": detail,
                "label": label,
                "state": state,
                "href": _detail_href("actions", action.get("action_id")),
                "date": _display_date(action),
                "created_at": _record_time_key(action),
                "queue": queue,
                "priority": "High" if stage in {"pending", "failed", "blocked"} else "Medium",
                "owner": owner_label,
                "type": "Action",
                "icon": "A",
                "record_kind": "action",
                "record_id": str(action.get("action_id") or ""),
                "record_ref": f"action:{action.get('action_id')}" if action.get("action_id") else "",
            }
        )
    draft_memories = [
        memory
        for memory in memories
        if str(memory.get("status") or "draft").lower() != "owner_approved"
        and not bool(
            memory.get("canonicality", {}).get("owner_approved")
            if isinstance(memory.get("canonicality"), dict)
            else False
        )
    ]
    for memory in draft_memories:
        items.append(
            {
                "kind": "Memory",
                "title": _truncate(str(memory.get("title") or memory.get("statement") or "Knowledge draft"), 96),
                "detail": "Visible review draft; it cannot influence answers, routing, or actions.",
                "label": "Draft / Needs review",
                "state": "underReview",
                "href": _detail_href("memories", memory.get("memory_id")),
                "date": _display_date(memory),
                "created_at": _record_time_key(memory),
                "queue": "Needs review",
                "priority": "Low",
                "owner": owner_label,
                "type": "Memory",
                "icon": "M",
                "record_kind": "memory",
                "record_id": str(memory.get("memory_id") or ""),
                "record_ref": f"memory:{memory.get('memory_id')}" if memory.get("memory_id") else "",
            }
        )
    return _recent(items)


def _inbox_open_count(
    briefs: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    memories: list[dict[str, Any]],
) -> int:
    open_claims = sum(1 for claim in claims if str(claim.get("status") or "").lower() != "approved")
    open_actions = sum(1 for action in actions if _action_lifecycle(action)["stage"] != "executed")
    open_memories = sum(
        1
        for memory in memories
        if str(memory.get("status") or "draft").lower() != "owner_approved"
        and not bool(
            memory.get("canonicality", {}).get("owner_approved")
            if isinstance(memory.get("canonicality"), dict)
            else False
        )
    )
    return len(briefs) + open_claims + open_actions + open_memories


def _detail_href(kind: str, record_id: Any) -> str:
    if not record_id:
        return f"/{kind}"
    return f"/{kind}/{quote(str(record_id))}?view=html"


def _degraded_notice(load_errors: list[str]) -> str:
    affected = ", ".join(load_errors)
    return f"""
<section class="cs-panel" data-product-state="degraded" role="alert">
  <div class="cs-panel-header">
    <div>
      <h2>Some workspace data could not be loaded</h2>
      <p class="cs-muted">Unavailable: {h(affected)}. No records were changed. Retry this page; if the state remains unavailable, inspect the local runtime before continuing.</p>
    </div>
    {_chip("Degraded", "failed")}
  </div>
</section>
"""


def _access_audit_unavailable() -> str:
    return """
<section class="cs-panel" data-product-state="access-audit-unavailable" role="alert">
  <div class="cs-panel-header">
    <div>
      <h1>Workspace access is temporarily unavailable</h1>
      <p class="cs-muted">CornerStone could not record this page view in History, so no workspace records are shown. Restore the local audit ledger and retry.</p>
    </div>
  </div>
</section>
"""


def _access_receipt(evidence_ref: str, audit_ref: str) -> str:
    return f"""
<details class="cs-disclosure cs-access-receipt" data-evidence-ref="{h(evidence_ref)}" data-audit-ref="{h(audit_ref)}">
  <summary><strong>Access receipt</strong><span>Evidence and History reference</span></summary>
  <dl class="cs-detail-grid">
    <dt>Record</dt><dd>{h(evidence_ref)}</dd>
    <dt>History</dt><dd><a href="/audit?record={quote(evidence_ref, safe='')}">{h(audit_ref)}</a></dd>
  </dl>
</details>
"""


def _page(root: Path, title: str, active: str, content: str, ctx: dict[str, Any], q: str) -> str:
    stylesheet_name, _, _ = style_asset(root)
    nav = _nav(active, ctx)
    topbar = _topbar(q, ctx)
    scope = ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {}
    page_audit_ref = str(ctx.get("page_audit_ref") or "")
    load_errors = ctx.get("load_errors") if isinstance(ctx.get("load_errors"), list) else []
    if load_errors:
        content = _degraded_notice(list(dict.fromkeys(str(item) for item in load_errors))) + content
    script = _home_script(scope)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="/assets/{h(stylesheet_name)}">
  <title>{h(title)} - CornerStone</title>
</head>
<body>
  <a class="cs-skip-link" href="#main-content">Skip to content</a>
  <div
    class="cs-shell"
    data-product-shell="cornerstone"
    data-tenant-id="{h(scope.get('tenant_id') or 'local-dev')}"
    data-owner-id="{h(scope.get('owner_id') or 'local-user')}"
    data-namespace-id="{h(scope.get('namespace_id') or 'personal')}"
    data-workspace-id="{h(scope.get('workspace_id') or 'default')}"
    data-page-audit-ref="{h(page_audit_ref)}"
  >
    <aside class="cs-sidebar" aria-label="CornerStone navigation">
      <a class="cs-brand" href="/" aria-label="CornerStone Home">
        <div class="cs-brand-mark">{icon("shield-check")}</div>
        <div>
          <div class="cs-brand-name">CornerStone</div>
          <div class="cs-brand-sub">Calm evidence desk</div>
        </div>
      </a>
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


def _nav(active: str, ctx: dict[str, Any]) -> str:
    loaded = (
        set(ctx["loaded_record_families"])
        if "loaded_record_families" in ctx
        else set(PRODUCT_RECORD_FAMILIES)
    )
    counts = {
        "/": len(ctx["artifacts"]) if "artifact" in loaded else None,
        "/search": (
            len(ctx["artifacts"]) + len(ctx["briefs"]) + len(ctx["claims"]) + len(ctx["actions"])
            if {"artifact", "brief", "claim", "action"} <= loaded
            else None
        ),
        "/artifacts": len(ctx["artifacts"]) if "artifact" in loaded else None,
        "/claims": len(ctx["claims"]) if "claim" in loaded else None,
        "/actions": len(ctx["actions"]) if "action" in loaded else None,
    }
    visible_claims = ctx.get("claims", []) if isinstance(ctx.get("claims"), list) else []
    claim_nav_label = (
        "Decisions"
        if visible_claims and all(_is_decision_draft(claim) for claim in visible_claims)
        else "Claims"
    )
    primary = [
        ("/", "Home"),
        ("/search", "Search"),
        ("/artifacts", "Artifacts"),
        ("/claims", claim_nav_label),
        ("/actions", "Actions"),
    ]
    return f"""
<nav class="cs-nav">
  <div class="cs-nav-group">
    <div class="cs-nav-label">Workspace</div>
    {''.join(_nav_link(href, label, active, counts.get(href, 0)) for href, label in primary)}
  </div>
  {_sidebar_status(ctx)}
</nav>
"""


def _nav_link(href: str, label: str, active: str, count: int | None) -> str:
    current = ' aria-current="page"' if href == active else ""
    icon_name = {
        "/": "home",
        "/search": "search",
        "/artifacts": "document",
        "/claims": "shield-check",
        "/actions": "action",
    }.get(href, "document")
    count_text = "" if count is None else str(count)
    return f'<a href="{h(href)}"{current}><span class="cs-nav-mark">{icon(icon_name)}</span><span>{h(label)}</span><span class="cs-nav-count">{h(count_text)}</span></a>'


def _sidebar_status(ctx: dict[str, Any]) -> str:
    loaded = (
        set(ctx["loaded_record_families"])
        if "loaded_record_families" in ctx
        else set(PRODUCT_RECORD_FAMILIES)
    )
    source_count = len(ctx["artifacts"]) if "artifact" in loaded else None
    scope = ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {}
    workspace = str(scope.get("workspace_id") or "default")
    namespace = str(scope.get("namespace_id") or "personal")
    source_summary = (
        f'{source_count} source{"s" if source_count != 1 else ""}'
        if source_count is not None
        else "workspace"
    )
    return f"""
<a class="cs-workspace-switcher" href="/review" aria-label="Open local workspace settings">
  <span class="cs-workspace-icon">{icon("document")}</span>
  <span><strong>{h(workspace)}</strong><small>{h(namespace)} · {h(source_summary)}</small></span>
  {icon("chevron-right", class_name="cs-icon is-small")}
</a>
"""


def _topbar(q: str, ctx: dict[str, Any]) -> str:
    raw_review_count = ctx.get("inbox_total", len(ctx["inbox"]))
    review_count = int(raw_review_count) if raw_review_count is not None else None
    review_label = (
        f'Open Review inbox with {review_count} item{"s" if review_count != 1 else ""}'
        if review_count is not None
        else "Open Review inbox"
    )
    review_count_markup = "" if review_count is None else f"<strong>{h(review_count)}</strong>"
    scope = ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {}
    owner = str(scope.get("owner_id") or "local-user")
    return f"""
<header class="cs-topbar">
  <div class="cs-command">
    <form class="cs-search" action="/search" method="get" role="search" aria-label="Global search">
      <span class="cs-search-icon">{icon("search")}</span>
      <input name="q" value="{h(q)}" aria-label="Search the active workspace" placeholder="Search sources, Briefs, decisions, and actions">
      <button type="submit" aria-label="Search">{icon("search", class_name="cs-icon is-inverse")}<span class="cs-sr-only">Search</span></button>
    </form>
  </div>
  <div class="cs-topbar-actions" aria-label="Workspace tools">
    <a class="cs-review-link" href="/inbox" aria-label="{h(review_label)}">{icon("review")}<span>Review</span>{review_count_markup}</a>
    <details class="cs-help-menu">
      <summary class="cs-icon-button" aria-label="Open help"><span class="cs-help-glyph" aria-hidden="true">?</span></summary>
      <div class="cs-help-popover">
        <strong>Start with Drop or Ask</strong>
        <p>Save a source, ask a question, then open the Brief and its sources before making a decision.</p>
        <a href="/">Go to Home</a>
      </div>
    </details>
    <a class="cs-avatar" href="/review" aria-label="Open owner area for {h(owner)}">{h(_owner_initials(scope))}</a>
  </div>
</header>
"""


def _home(ctx: dict[str, Any]) -> str:
    suggestions = "".join(
        f'<button class="cs-button secondary" type="button" data-ask-suggestion="{h(item)}">{h(item)}</button>'
        for item in ctx["suggestions"]
    )
    recent = _recent_items_block(ctx)
    recent_questions = _recent_questions_block(ctx)
    activity = _recent_activity_block(ctx) if ctx.get("audit") else ""
    layout_class = "cs-home-layout has-activity" if activity else "cs-home-layout"
    source_choices = []
    for artifact in ctx.get("artifacts", []):
        artifact_id = str(artifact.get("artifact_id") or "")
        if not artifact_id:
            continue
        presentation = _artifact_state(ctx, artifact)
        if (
            presentation.get("state") != "searchable"
            or int(artifact.get("original_size_bytes") or 0) > VS5_DECISION_SOURCE_MAX_BYTES
        ):
            continue
        source_size = int(artifact.get("original_size_bytes") or 0)
        source_choices.append(
            f'''<label class="cs-source-choice">
  <input type="checkbox" value="{h(artifact_id)}" data-decision-source data-source-size="{h(source_size)}">
  <span><strong>{h(_artifact_title(artifact))}</strong><small data-source-note>{h(presentation["label"])} · {h(_display_date(artifact))} · {h(source_size)} bytes</small></span>
</label>'''
        )
        if len(source_choices) >= 5:
            break
    source_set = f'''<fieldset class="cs-source-set" id="cs-decision-source-set" aria-describedby="cs-source-boundary cs-source-selection-summary">
  <legend>Sources for this decision</legend>
  <p class="cs-muted">Choose one to five saved sources. CornerStone will stay inside this boundary.</p>
  <p class="cs-meta" id="cs-source-boundary">Brief input: UTF-8 plain text, .txt, or .md; at most 128 KiB per source and 512 KiB total. Files up to {MAX_BROWSER_UPLOAD_BYTES // (1024 * 1024)} MB can still be archived, but unsupported or larger sources are not selected for the Brief.</p>
  <div class="cs-source-choice-list" id="cs-source-choice-list">{"".join(source_choices)}{'' if source_choices else '<div class="cs-empty cs-source-set-empty" id="cs-source-set-empty"><strong>Save a source first</strong><p>Paste or upload one to five plain-text sources before asking for a decision Brief.</p></div>'}</div>
  <p class="cs-meta" id="cs-source-selection-summary" role="status" aria-live="polite">0 of 5 sources selected · 0 of 512 KiB</p>
</fieldset>'''
    return f"""
<section class="cs-home-intro" data-product-surface="home">
  <div class="{layout_class}">
    <div class="cs-stack cs-home-primary">
      <div class="cs-hero cs-page-head">
        <h1>Drop anything, or ask what we know</h1>
        <p>Bring the messy input. Leave with a Brief you can trace back to the source.</p>
      </div>
      <section class="cs-home-canvas" aria-labelledby="home-workbench-title">
        <h2 class="cs-sr-only" id="home-workbench-title">Start with a source or a question</h2>
      <div class="cs-home-workspace">
        <form class="cs-drop" id="cs-drop-form">
          <div class="cs-drop-target">
            <div class="cs-drop-mark">{icon("upload")}</div>
            <div>
              <strong>Drop a file or paste notes</strong>
              <p class="cs-muted">Choose one to five files. Files up to {MAX_BROWSER_UPLOAD_BYTES // (1024 * 1024)} MB are archived byte-for-byte; Briefs use plain-text sources up to 128 KiB each and 512 KiB total.</p>
            </div>
          </div>
          <div class="cs-home-source-row">
            <button class="cs-button secondary" type="button" id="cs-file-button">Browse files</button>
            <button class="cs-button" id="cs-save-source-button" type="submit">Save source</button>
            <input id="cs-file-input" type="file" aria-label="Choose one to five files to archive" multiple hidden>
          </div>
          <div class="cs-home-paste-row" aria-label="Paste text source">
            <label class="cs-sr-only" for="cs-drop-text">Paste source text</label>
            <textarea class="cs-drop-input" id="cs-drop-text" placeholder="Paste notes, an email, a renewal clause, or any text source"></textarea>
          </div>
          <div class="cs-status is-idle" id="cs-drop-status" data-state="idle" role="status" aria-live="polite" aria-atomic="true" hidden>Ready for a source.</div>
        </form>
        <div class="cs-or-divider">or ask a question</div>
        <form class="cs-stack" id="cs-ask-form">
          {source_set}
          <div class="cs-ask-bar" role="group" aria-label="Ask the workspace">
            <span class="cs-ask-mark">{icon("chat")}</span>
            <div>
              <strong>Ask the workspace</strong>
              <div class="cs-meta">Answers stay connected to saved sources.</div>
            </div>
            <label class="cs-sr-only" for="cs-ask-input">Ask a question about saved sources</label>
            <input class="cs-field" id="cs-ask-input" placeholder="Ask about saved sources">
            <button class="cs-button" id="cs-ask-submit-button" type="submit">Ask</button>
          </div>
          {f'<div class="cs-suggestion-row">{suggestions}</div>' if suggestions else ''}
          <div class="cs-status is-idle" id="cs-ask-status" data-state="idle" role="status" aria-live="polite" aria-atomic="true" hidden>No answer requested yet.</div>
        </form>
      </div>
      </section>
      {recent_questions}
      {recent}
    </div>
    {f'<aside class="cs-stack cs-home-activity">{activity}</aside>' if activity else ''}
  </div>
</section>
"""


def _answer_label(record: dict[str, Any]) -> tuple[str, str]:
    label = str(record.get("trust_label") or record.get("label") or "draft").lower()
    if label == "evidence_backed":
        return "Source-backed answer", "evidenceBacked"
    if label == "insufficient_evidence":
        return "Insufficient evidence", "insufficientEvidence"
    if label == "extractive_fallback":
        return "Extractive fallback", "underReview"
    return "Draft answer", "draft"


def _recent_questions_block(ctx: dict[str, Any]) -> str:
    answers = [record for record in ctx.get("answers", []) if record.get("answer_id")][:5]
    if not answers:
        return """
<section class="cs-panel flat cs-question-history" aria-labelledby="recent-questions-title">
  <div class="cs-panel-header"><div><h2 id="recent-questions-title">Recent questions</h2><p class="cs-muted">Questions and answers are saved here with their source context.</p></div></div>
  <div class="cs-empty">No saved answers yet. Ask a question after selecting one to five sources.</div>
</section>
"""
    rows = []
    for answer in answers:
        label, state = _answer_label(answer)
        rows.append(
            f"""
<a class="cs-question-history-row" href="{h(_detail_href('answers', answer.get('answer_id')))}">
  <span class="cs-question-history-mark">{icon("chat")}</span>
  <span class="cs-question-history-copy">
    <strong>{h(_truncate(str(answer.get("question") or "Saved question"), 120))}</strong>
    <span>{h(_truncate(str(answer.get("answer") or "No answer text was saved."), 180))}</span>
    <small>{h(_display_date(answer))} · Reopen answer and sources</small>
  </span>
  {_chip(label, state)}
</a>
"""
        )
    return f"""
<section class="cs-panel flat cs-question-history" aria-labelledby="recent-questions-title">
  <div class="cs-panel-header"><div><h2 id="recent-questions-title">Recent questions</h2><p class="cs-muted">Every Ask answer remains connected to its saved sources and audit trail.</p></div><a class="cs-meta" href="/audit">View all activity</a></div>
  <div class="cs-question-history-list">{"".join(rows)}</div>
</section>
"""


def _recent_items_block(ctx: dict[str, Any]) -> str:
    items: list[dict[str, Any]] = []
    for artifact in ctx["artifacts"][:3]:
        presentation = _artifact_state(ctx, artifact)
        items.append(
            {
                "kind": "Source",
                "title": _artifact_title(artifact),
                "detail": f"{_display_date(artifact)} - Original text preserved",
                "href": _detail_href("artifacts", artifact.get("artifact_id")),
                "label": presentation["label"],
                "state": presentation["state"],
            }
        )
    for brief in ctx["briefs"][:2]:
        label, state = _brief_label(brief)
        items.append(
            {
                "kind": "Brief",
                "title": _brief_title(brief),
                "detail": f"{_display_date(brief)} - Draft from visible sources",
                "href": _detail_href("briefs", brief.get("brief_id")),
                "label": label,
                "state": state,
            }
        )
    for claim in ctx["claims"][:2]:
        label, state = _claim_label(claim)
        product_kind = "Decision draft" if _is_decision_draft(claim) else "Claim"
        items.append(
            {
                "kind": product_kind,
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
  <span class="cs-home-item-icon">{icon(record_icon(str(item["kind"])))}</span>
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


def _recent_activity_block(ctx: dict[str, Any]) -> str:
    rows = []
    for event in ctx["audit"][:5]:
        event_type = str(event.get("event_type") or "")
        rows.append(
            f"""
<div class="cs-activity-row">
  <span class="cs-activity-icon">{icon(_audit_icon_asset(event))}</span>
  <div>
    <strong>{h(_plain_event(event_type))}</strong>
    <p class="cs-meta">{h(_audit_subject_label(event))} / {h(_display_date(event))}</p>
  </div>
</div>
"""
        )
    content = f'<div class="cs-activity-list">{"".join(rows)}</div>' if rows else '<div class="cs-empty">Activity appears after you save, search, draft, or review work.</div>'
    return f"""
<section class="cs-panel flat">
  <div class="cs-panel-header">
    <h2>Recent activity</h2>
    <a class="cs-meta" href="/audit">View audit</a>
  </div>
  {content}
</section>
"""


def _search_page(
    ctx: dict[str, Any],
    q: str,
    search_type: str = "all",
    outcome: dict[str, Any] | None = None,
) -> str:
    display_query = redact_text(q)
    success = bool(outcome and outcome.get("status") == "success")
    facets = outcome.get("facets", {}) if success and isinstance(outcome.get("facets"), dict) else {}
    counts_by_label = {
        "All": int(facets.get("all", 0) or 0),
        "Sources": int(facets.get("sources", 0) or 0),
        "Briefs": int(facets.get("briefs", 0) or 0),
        "Claims": int(facets.get("claims", 0) or 0),
        "Actions": int(facets.get("actions", 0) or 0),
    }
    selected_type_label = dict(PRODUCT_SEARCH_TYPES).get(search_type, "All")
    total = int(outcome.get("result_count", 0) or 0) if success else 0
    snapshot = outcome.get("search_snapshot", {}) if success and isinstance(outcome.get("search_snapshot"), dict) else {}
    snapshot_id = str(snapshot.get("search_snapshot_id") or "")
    audit_ref = str((outcome.get("audit_refs") or [""])[0]) if success else ""
    results = [
        _search_snapshot_result(row)
        for row in (outcome.get("results", []) if success else [])
        if isinstance(row, dict)
    ]
    rows = (
        "".join(results)
        if results
        else _search_empty(
            display_query,
            search_type=search_type,
            selected_type_label=selected_type_label,
            all_result_count=counts_by_label.get("All", 0),
        )
    )
    count_tabs = "".join(
        f'<a class="cs-search-tab{" is-active" if value == search_type else ""}" href="{h(_search_href(display_query, value, snapshot_id))}"'
        f'{" aria-current=\"page\"" if value == search_type else ""}>{h(label)} <strong>{h(counts_by_label.get(label, 0))}</strong></a>'
        for value, label in PRODUCT_SEARCH_TYPES
    )
    scope_label = _scope_label(ctx.get("scope"))
    heading = f'Results for “{display_query}”' if display_query.strip() else "Search saved work"
    guidance = (
        "Keyword matches across saved sources and derived work. Open the source before using a draft in a decision."
        if display_query.strip()
        else "Use the search field above to find saved sources, Briefs, decisions, and action drafts."
    )
    page = int(outcome.get("page", 1) or 1) if success else 1
    page_count = int(outcome.get("page_count", 1) or 1) if success else 1
    pager = _search_pager(display_query, search_type, snapshot_id, page, page_count)
    receipt = (
        f"""
  <details class="cs-disclosure cs-search-receipt" data-search-snapshot-id="{h(snapshot_id)}" data-audit-ref="{h(audit_ref)}">
    <summary><strong>Search receipt</strong><span>Scope, snapshot, and History reference</span></summary>
    <dl class="cs-detail-grid">
      <dt>Snapshot</dt><dd>{h(f'search_snapshot:{snapshot_id}')}</dd>
      <dt>History</dt><dd>{h(audit_ref)}</dd>
      <dt>Visible range</dt><dd>{h(str(outcome.get('page_start', 0)))}-{h(str(outcome.get('page_end', 0)))} of {h(str(total))}</dd>
      <dt>Workspace</dt><dd>{h(scope_label)}</dd>
    </dl>
  </details>
"""
        if success
        else ""
    )
    error_notice = (
        '<section class="cs-panel" data-product-state="degraded" role="alert"><h2>Search snapshot unavailable</h2><p>That saved search is unavailable in this workspace. Run the search again.</p></section>'
        if outcome and not success
        else ""
    )
    return f"""
<section class="cs-search-page" data-product-surface="search">
  <header class="cs-page-head">
    <div class="cs-kicker">Search</div>
    <h1>{h(heading)}</h1>
    <p>{h(guidance)}</p>
  </header>
  <nav class="cs-search-tabs" aria-label="Filter results by record type">{count_tabs}</nav>
  <div class="cs-result-list" aria-live="polite">
    <div class="cs-result-list-header">
      <strong>{h(str(total))} result{"s" if total != 1 else ""} · {h(selected_type_label)}</strong>
      <span>Workspace: {h(scope_label)}</span>
    </div>
    {rows}
  </div>
  {pager}
  {receipt}
  {error_notice}
</section>
"""


def _search_href(q: str, search_type: str, snapshot_id: str = "", page: int | None = None) -> str:
    values = [f"q={quote(q)}", f"type={quote(search_type)}"]
    if snapshot_id:
        values.append(f"snapshot={quote(snapshot_id)}")
    if page and page > 1:
        values.append(f"page={page}")
    return "/search?" + "&".join(values)


def _search_snapshot_result(row: dict[str, Any]) -> str:
    result_type = str(row.get("result_type") or "artifact")
    kind = {"artifact": "Source", "brief": "Brief", "claim": "Claim", "action": "Action"}.get(result_type, "Source")
    if result_type == "claim" and str(row.get("product_role") or "") == "decision_draft":
        kind = "Decision draft"
    record_id = str(row.get(f"{result_type}_id") or "")
    href_kind = {"artifact": "artifacts", "brief": "briefs", "claim": "claims", "action": "actions"}.get(result_type, "artifacts")
    if result_type == "artifact":
        presentation = artifact_presentation(
            {
                "original_storage_ref": row.get("original_storage_ref"),
                "checksum_sha256": "snapshot",
                "derived": {"status": row.get("derived_status"), "text_ref": row.get("derived_text_ref")},
            },
            text_available=row.get("derived_text_available") is True,
            original_available=row.get("original_available") is True,
        )
        label, state = str(presentation["label"]), str(presentation["state"])
    elif result_type == "brief":
        label, state = _brief_label(
            {
                "status": row.get("record_status"),
                "trust_label": row.get("trust_state"),
                "output_mode": row.get("output_mode"),
            }
        )
    elif result_type == "claim":
        status = str(row.get("record_status") or "").lower()
        label, state = ("Approved", "approved") if status == "approved" else ("Source support", "searchable") if len(row.get("evidence_refs", [])) > 1 else ("Draft", "draft")
    else:
        execution = str(row.get("execution_status") or "").lower()
        approval = str(row.get("approval_status") or "").lower()
        status = str(row.get("record_status") or "").lower()
        if execution == "executed":
            label, state = "Executed", "executed"
        elif execution in {"failed", "error"}:
            label, state = "Failed", "failed"
        elif status in {"blocked", "denied", "policy_blocked"}:
            label, state = "Policy blocked", "policyBlocked"
        elif approval == "approved":
            label, state = "Approved", "approved"
        elif approval in {"pending", "required", "not_approved"}:
            label, state = "Needs approval", "underReview"
        else:
            label, state = "Draft", "draft"
    modes = [str(value).replace("_", " ").title() for value in row.get("retrieval_modes", []) if isinstance(value, str)]
    match_label = "Keyword match" + (f" · {', '.join(modes)}" if modes else "")
    return _search_result_row(
        kind,
        str(row.get("title") or f"{kind} result"),
        str(row.get("snippet") or "Open the record to inspect this match."),
        _detail_href(href_kind, record_id),
        label,
        state,
        _display_date(row),
        result_ref=str(row.get("result_ref") or ""),
        match_label=match_label,
    )


def _search_pager(q: str, search_type: str, snapshot_id: str, page: int, page_count: int) -> str:
    if page_count <= 1:
        return ""
    previous = (
        f'<a class="cs-button secondary" rel="prev" href="{h(_search_href(q, search_type, snapshot_id, page - 1))}">Previous</a>'
        if page > 1
        else '<span class="cs-button secondary" aria-disabled="true">Previous</span>'
    )
    following = (
        f'<a class="cs-button secondary" rel="next" href="{h(_search_href(q, search_type, snapshot_id, page + 1))}">Next</a>'
        if page < page_count
        else '<span class="cs-button secondary" aria-disabled="true">Next</span>'
    )
    return f'<nav class="cs-pagination" aria-label="Search result pages">{previous}<span>Page {page} of {page_count}</span>{following}</nav>'


def _search_result_row(
    kind: str,
    title: str,
    detail: str,
    href: str,
    label: str,
    state: str,
    date: str,
    *,
    result_ref: str = "",
    match_label: str = "Keyword match",
) -> str:
    icon_class = {
        "Source": "is-source",
        "Brief": "is-brief",
        "Claim": "is-claim",
        "Decision draft": "is-claim",
        "Action": "is-action",
    }.get(kind, "is-source")
    return f"""
<article class="cs-result-row" data-result-ref="{h(result_ref)}">
  <span class="cs-result-icon {h(icon_class)}">{icon(record_icon(kind))}</span>
  <span class="cs-result-body">
    <span class="cs-result-meta"><span class="cs-result-type">{h(kind)}</span><span>{h(date)}</span><span>{h(match_label)}</span></span>
    <h3><a href="{h(href)}">{h(title)}</a></h3>
    <p>{h(_truncate(detail, 240))}</p>
  </span>
  <span class="cs-result-support">
    {_chip(label, state)}
    <span class="cs-result-actions">
      <a class="cs-button secondary" href="{h(href)}">{h(record_action(kind))}</a>
    </span>
  </span>
</article>
"""


def _search_empty(
    q: str,
    *,
    search_type: str = "all",
    selected_type_label: str = "All",
    all_result_count: int = 0,
) -> str:
    if q.strip():
        if search_type != "all":
            other_result_count = max(0, all_result_count)
            result_word = "result" if other_result_count == 1 else "results"
            verb = "exists" if other_result_count == 1 else "exist"
            return _empty_state(
                f"No {selected_type_label.lower()} match",
                f"No {selected_type_label.lower()} matched this keyword",
                (
                    f"{other_result_count} other local {result_word} {verb} for this keyword. "
                    "Show all result types or try a different keyword; no unsupported result is created."
                    if other_result_count
                    else f"No local {selected_type_label.lower()} matched this keyword. Try a different keyword or search all result types."
                ),
                "Show all result types",
                f"/search?q={quote(q)}&type=all",
                "Search another keyword",
                f"/search?type={quote(search_type)}",
                mark="?",
                receipts=[
                    ("Active filter", f"Only {selected_type_label.lower()} are shown."),
                    ("All local matches", f"{all_result_count} across visible result types."),
                    ("Decision safety", "No unsupported result is created."),
                ],
            )
        return _empty_state(
            "No match",
            "Try a broader search",
            "No saved source, brief, claim, or action draft matched that keyword. Shorter terms usually work better with the local search index.",
            "Search all sources",
            "/search",
            "Save a source",
            "/",
            mark="?",
            receipts=[
                ("Search match", "Only local keyword matches are shown."),
                ("Source path", "Save a source before broadening the query."),
                ("Decision safety", "No unsupported result is created."),
            ],
        )
    return _empty_state(
        "Search",
        "Search starts with saved work",
        "Enter a keyword to search saved sources, briefs, claims, and action drafts. If the workspace is empty, save a source from Home first.",
        "Save a source",
        "/",
        "Open artifacts",
        "/artifacts",
        mark="?",
        receipts=[
            ("Saved source", "Search begins after Home preserves input."),
            ("Result link", "Matches link back to a local record."),
            ("Follow-up", "Suggested queries stay inside workspace scope."),
        ],
    )


def _empty_state(
    kicker: str,
    title: str,
    body: str,
    primary_label: str,
    primary_href: str,
    secondary_label: str | None = None,
    secondary_href: str | None = None,
    *,
    mark: str = "+",
    steps: list[tuple[str, str]] | None = None,
    receipts: list[tuple[str, str]] | None = None,
) -> str:
    mark_icon = {
        "S": "document",
        "B": "brief",
        "C": "shield-check",
        "A": "action",
        "I": "review",
        "T": "history",
        "H": "history",
        "?": "search",
    }.get(mark[:1], "document")
    secondary = ""
    if secondary_label and secondary_href:
        secondary = f'<a class="cs-button secondary" href="{h(secondary_href)}">{h(secondary_label)}</a>'
    steps_html = ""
    if steps:
        step_rows = "".join(
            f"""
  <div class="cs-empty-step">
    <strong>{h(label)}</strong>
    <span class="cs-meta">{h(detail)}</span>
  </div>
"""
            for label, detail in steps
        )
        steps_html = f'<div class="cs-empty-steps" aria-label="Suggested start path">{step_rows}</div>'
    receipts_html = ""
    if receipts:
        receipt_rows = "".join(
            f"""
  <div class="cs-empty-receipt">
    <strong>{h(label)}</strong>
    <span>{h(detail)}</span>
  </div>
"""
            for label, detail in receipts
        )
        receipts_html = f"""
  <div class="cs-empty-briefing">
    <div>
      <h3>Startup path</h3>
      {steps_html or '<p class="cs-muted">Start from Home, save work, then open the generated records from the product lists.</p>'}
    </div>
    <div>
      <h3>What will appear</h3>
      <div class="cs-empty-receipts">{receipt_rows}</div>
    </div>
  </div>
"""
    return f"""
<article class="cs-empty-state">
  <div class="cs-empty-state-main">
    <span class="cs-empty-mark">{icon(mark_icon)}</span>
    <div class="cs-empty-copy">
      <div class="cs-kicker">{h(kicker)}</div>
      <h2>{h(title)}</h2>
      <p>{h(body)}</p>
    </div>
  </div>
  {receipts_html or steps_html}
  <div class="cs-empty-actions">
    <a class="cs-button" href="{h(primary_href)}">{h(primary_label)}</a>
    {secondary}
  </div>
</article>
"""


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


def _collection_row(
    icon_key: str,
    title: str,
    detail: str,
    href: str,
    meta: list[tuple[str, str]],
    chips: list[tuple[str, str]],
    *,
    footer: list[tuple[str, str]] | None = None,
    action_label: str = "Open",
) -> str:
    icon_name = {
        "S": "document",
        "B": "brief",
        "C": "shield-check",
        "A": "action",
    }.get(icon_key, "document")
    meta_row = "".join(f"<span>{h(label)}</span>" for label, _ in meta)
    chip_row = "".join(_chip(label, state) for label, state in chips)
    footer_row = ""
    if footer:
        footer_row = (
            '<span class="cs-collection-footrail">'
            + "".join(
                f"""
<span class="cs-collection-stage">
  <strong>{h(label)}</strong>
  <span>{h(value)}</span>
</span>
"""
                for label, value in footer
            )
            + "</span>"
        )
    return f"""
<a class="cs-collection-row" href="{h(href)}">
  <span class="cs-collection-icon">{icon(icon_name)}</span>
  <span class="cs-collection-body">
    <span class="cs-collection-meta">{meta_row}</span>
    <h3>{h(title)}</h3>
    <p>{h(_truncate(detail, 240))}</p>
    {footer_row}
  </span>
  <span class="cs-collection-actions">
    <span class="cs-row">{chip_row}</span>
    <span class="cs-collection-cta">{h(action_label)}</span>
  </span>
</a>
"""


def _artifact_list_page(ctx: dict[str, Any]) -> str:
    artifacts = ctx["artifacts"]
    rows = "".join(_artifact_collection_row(ctx, artifact) for artifact in artifacts) or _empty_state(
        "Day zero",
        "Start with a source",
        "Drop a note, paste text, or save a file from Home. Sources stay preserved before any brief, claim, or action uses them.",
        "Go to Home",
        "/",
        "Search workspace",
        "/search",
        mark="S",
        steps=[
            ("1. Save source", "Keep the original input intact."),
            ("2. Ask about it", "Draft a brief from saved work."),
            ("3. Review support", "Use sources before decisions."),
        ],
        receipts=[
            ("Original source", "Preserved text, date, and local scope."),
            ("Search match", "Source becomes available to keyword search."),
            ("Linked use", "Briefs and claims can reference this source."),
        ],
    )
    presentations = [_artifact_state(ctx, artifact) for artifact in artifacts]
    searchable_count = sum(1 for presentation in presentations if presentation["searchable"])
    preserved_count = sum(1 for presentation in presentations if presentation["saved"])
    pending_count = len(artifacts) - searchable_count
    linked_count = sum(
        1
        for record in [*ctx["briefs"], *ctx["claims"], *ctx["actions"]]
        for ref in _evidence_refs(record)
        if ref.startswith("artifact:")
    )
    return f"""
<section data-product-surface="artifacts">
  <div class="cs-page-head">
    <div class="cs-kicker">Artifacts</div>
    <h1>Saved sources</h1>
    <p>Every source remains preserved before any derived text, brief, claim, or action draft uses it.</p>
  </div>
  <div class="cs-collection-workbench">
    <div>
      {_collection_summary([("Saved sources", len(artifacts)), ("Searchable", searchable_count), ("Not searchable", pending_count)])}
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
          <dt>Preserved</dt><dd>{h(preserved_count)}</dd>
          <dt>Linked uses</dt><dd>{h(linked_count)}</dd>
          <dt>Trust</dt><dd>Untrusted until checked</dd>
        </dl>
      </section>
    </aside>
  </div>
</section>
"""


def _artifact_collection_row(ctx: dict[str, Any], artifact: dict[str, Any]) -> str:
    presentation = _artifact_state(ctx, artifact)
    chips = [("Saved", "saved")] if presentation["saved"] else [("Integrity issue", "failed")]
    presentation_chip = (str(presentation["label"]), str(presentation["state"]))
    if presentation["label"] != "Saved" and presentation_chip not in chips:
        chips.append(presentation_chip)
    return _collection_row(
        "S",
        _artifact_title(artifact),
        str(artifact.get("_preview") or presentation["explanation"]),
        _detail_href("artifacts", artifact.get("artifact_id")),
        [("Source", ""), (_display_date(artifact), "")],
        chips,
        footer=[
            ("Representation", str(presentation["label"])),
            ("Original", "Preserved" if presentation["saved"] else "Unavailable"),
        ],
    )


def _brief_list_page(ctx: dict[str, Any]) -> str:
    briefs = ctx["briefs"]
    with_sources = sum(1 for brief in briefs if _brief_source_count(brief))
    source_ref_count = sum(_brief_source_count(brief) for brief in briefs)
    rows = ""
    for brief in briefs:
        label, state = _brief_label(brief)
        source_count = _brief_source_count(brief)
        source_label = f"{source_count} source ref{'s' if source_count != 1 else ''}"
        rows += _collection_row(
            "B",
            _brief_title(brief),
            _brief_summary_text(brief),
            _detail_href("briefs", brief.get("brief_id")),
            [("Brief", ""), (_display_date(brief), ""), (source_label, "")],
            [(label, state), ("Open brief", "searchable")],
            footer=[
                ("Source refs", source_label),
                ("State", label),
                ("Next review step", "Read before claim or action"),
            ],
            action_label="Open brief",
        )
    rows = rows or _empty_state(
        "Day zero",
        "Create the first brief",
        "Save a source, then ask a question to draft a brief with visible source support. Briefs stay draft material until reviewed.",
        "Save a source",
        "/",
        "Open artifacts",
        "/artifacts",
        mark="B",
        steps=[
            ("1. Drop input", "Start from a real note or file."),
            ("2. Ask a question", "Use the workspace ask box."),
            ("3. Check sources", "Open the brief before use."),
        ],
        receipts=[
            ("Brief draft", "Answer stays draft until reviewed."),
            ("Source coverage", "Visible references before decision use."),
            ("Next use", "Claim or action only after support is checked."),
        ],
    )
    source_note = (
        "Open the visible sources before using any finding in a decision."
        if with_sources
        else "Briefs need saved sources before they can support a decision."
    )
    return f"""
<section data-product-surface="briefs">
  <div class="cs-page-head">
    <div class="cs-kicker">Briefs</div>
    <h1>Brief workspace</h1>
    <p>Review drafted answers, source coverage, and the next safe use before a brief becomes decision material.</p>
  </div>
  <div class="cs-collection-workbench">
    <div>
      {_collection_summary([("Briefs", len(briefs)), ("With sources", with_sources), ("Source refs", source_ref_count)])}
      <div class="cs-collection-list">{rows}</div>
    </div>
    <aside class="cs-stack">
      <section class="cs-panel flat">
        <div class="cs-panel-header"><h2>Source coverage</h2>{_chip(str(source_ref_count), "searchable")}</div>
        <p class="cs-muted">{h(source_note)}</p>
        <div class="cs-review-box">
          <a class="cs-button" href="/artifacts">Review sources</a>
          <a class="cs-button secondary" href="/search">Search workspace</a>
        </div>
      </section>
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Use next</h2>
        <p class="cs-muted">Move from a brief into a claim or action only after the source support is visible.</p>
        <div class="cs-review-box">
          <a class="cs-button secondary" href="/claims">Review claims</a>
          <a class="cs-button secondary" href="/actions">Review actions</a>
        </div>
      </section>
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Brief posture</h2>
        <dl class="cs-detail-grid">
          <dt>Drafts</dt><dd>{h(len(briefs))}</dd>
          <dt>With sources</dt><dd>{h(with_sources)}</dd>
          <dt>Use</dt><dd>Draft until checked</dd>
        </dl>
      </section>
    </aside>
  </div>
</section>
"""


def _claim_list_page(ctx: dict[str, Any]) -> str:
    claims = ctx["claims"]
    decision_draft_count = sum(1 for claim in claims if _is_decision_draft(claim))
    only_decision_drafts = bool(claims) and decision_draft_count == len(claims)
    collection_label = "Decision drafts" if only_decision_drafts else "Claims"
    collection_title = "Decision drafts under review" if only_decision_drafts else "Claims under review"
    supported_count = sum(
        1
        for claim in claims
        if _evidence_refs(claim)
        and not (
            isinstance(claim.get("evidence_integrity"), dict)
            and claim["evidence_integrity"].get("status") == "failed"
        )
    )
    semantic_review_needed_count = sum(
        1
        for claim in claims
        if _evidence_refs(claim)
        and not (
            isinstance(claim.get("evidence_integrity"), dict)
            and claim["evidence_integrity"].get("status") == "failed"
        )
        and not _claim_semantic_support_verified(claim)
        and str(claim.get("status") or "").lower() != "approved"
    )
    semantic_reviewed_count = sum(1 for claim in claims if _claim_semantic_support_verified(claim))
    approved_count = sum(1 for claim in claims if str(claim.get("status") or "").lower() == "approved")
    rows = ""
    for claim in claims:
        label, state = _claim_label(claim)
        decision_draft = _is_decision_draft(claim)
        product_kind = "Decision draft" if decision_draft else "Claim"
        source_count = len([ref for ref in _evidence_refs(claim) if ref.startswith("artifact:")])
        rows += _collection_row(
            "C",
            _claim_title(claim),
            "Review the preserved source links before using this draft." if decision_draft else "Review source support before approval.",
            _detail_href("claims", claim.get("claim_id")),
            [(product_kind, ""), (_display_date(claim), ""), (f"{source_count} source refs", "")],
            [(label, state), ("Review required", "underReview")],
            footer=[
                ("Evidence refs", f"{source_count} source refs"),
                ("Trust lane", label),
                ("Next review step", "Compare with the source Brief" if decision_draft else "Semantic review before owner approval"),
            ],
            action_label="Review decision draft" if decision_draft else "Review claim",
        )
    rows = rows or _empty_state(
        "Day zero",
        "No claims need review",
        "Claims appear after a brief finding is promoted or a statement is drafted with source support. Start with a brief before making a decision.",
        "Open briefs",
        "/briefs",
        "Check sources",
        "/artifacts",
        mark="C",
        steps=[
            ("1. Draft brief", "Summarize saved work first."),
            ("2. Choose finding", "Promote only useful statements."),
            ("3. Attach support", "Keep source links visible."),
        ],
        receipts=[
            ("Claim statement", "One decision-ready statement per row."),
            ("Trust lane", "Draft, source support, then separate semantic review before owner approval."),
            ("Evidence picker", "Support must stay visible before approval."),
        ],
    )
    collection_summary_items: list[tuple[str, int]] = [
        (collection_label, len(claims)),
        ("With sources", supported_count),
    ]
    if only_decision_drafts:
        collection_summary_items.append(("Draft", len(claims)))
    else:
        collection_summary_items.extend(
            [("Semantic review needed", semantic_review_needed_count), ("Approved", approved_count)]
        )
    collection_summary = _collection_summary(collection_summary_items)
    trust_or_boundary = (
        '<section class="cs-panel flat"><h2 class="cs-section-title">Decision boundary</h2><p class="cs-muted">A Decision draft preserves source links only. It cannot approve shared truth or authorize an action.</p></section>'
        if only_decision_drafts
        else f'<section class="cs-panel flat"><h2 class="cs-section-title">Trust ladder</h2>{_claim_trust_ladder(bool(supported_count), bool(semantic_reviewed_count), bool(approved_count))}</section>'
    )
    return f"""
<section data-product-surface="claims">
  <div class="cs-page-head">
    <div class="cs-kicker">{h(collection_label)}</div>
    <h1>{h(collection_title)}</h1>
    <p>{"Decision drafts preserve a sourced finding without approval, shared-truth, or action authority." if only_decision_drafts else "Trace each statement to saved sources, then review whether those sources actually support its meaning before owner approval."}</p>
  </div>
  <div class="cs-collection-workbench">
    <div>
      {collection_summary}
      <div class="cs-collection-list">{rows}</div>
    </div>
    <aside class="cs-stack">
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Review posture</h2>
        <p class="cs-muted">{"Decision drafts remain Draft and grant no approval or action authority." if only_decision_drafts else "Claims can show source support, but they stay Draft until statement-level semantic review is recorded separately from owner approval."}</p>
        <div class="cs-review-box">
          <a class="cs-button" href="/inbox">Open review inbox</a>
          <a class="cs-button secondary" href="/artifacts">Check sources</a>
        </div>
      </section>
      {trust_or_boundary}
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
        lifecycle = _action_lifecycle(action)
        stage = str(lifecycle["stage"])
        if stage == "pending":
            approval_count += 1
        dry_run = action.get("dry_run") if isinstance(action.get("dry_run"), dict) else {}
        result = lifecycle["result"]
        detail = str(
            result.get("message")
            if stage == "executed" and isinstance(result, dict) and result.get("message")
            else dry_run.get("goal") or "Preview before any external write."
        )
        impact = dry_run.get("expected_impact") if isinstance(dry_run.get("expected_impact"), dict) else {}
        risk = str(impact.get("risk") or action.get("risk") or "review").title()
        lifecycle_chip = ("Recorded result", "saved") if stage == "executed" else ("Dry-run first", "searchable")
        footer = (
            [
                ("Execution", "Recorded"),
                ("Result", str(result.get("status") or "available").title()),
                ("Approval", str(lifecycle["approval_status"]).replace("_", " ").title()),
            ]
            if stage == "executed"
            else [
                ("Dry-run", "Preview before send"),
                ("Risk", f"{risk} risk"),
                ("Next review step", "Check policy and approval"),
            ]
        )
        rows += _collection_row(
            "A",
            _action_title(action),
            detail,
            _detail_href("actions", action.get("action_id")),
            [("Action", ""), (_display_date(action), ""), (f"{risk} risk", "")],
            [(label, state), lifecycle_chip],
            footer=footer,
            action_label="Open result" if stage == "executed" else "Open preview",
        )
    rows = rows or _empty_state(
        "Day zero",
        "No action previews yet",
        "Action drafts appear after a supported claim or brief next step is turned into a reviewable preview. Nothing executes from this page.",
        "Open claims",
        "/claims",
        "Open briefs",
        "/briefs",
        mark="A",
        steps=[
            ("1. Pick supported work", "Use a claim or brief finding."),
            ("2. Preview impact", "Inspect proposed changes first."),
            ("3. Review before send", "Approval stays explicit."),
        ],
        receipts=[
            ("Dry-run preview", "Impact appears before any external step."),
            ("Policy check", "Risk and approval stay visible."),
            ("Audit link", "Execution remains traceable if approved."),
        ],
    )
    executed_count = sum(1 for action in actions if _action_lifecycle(action)["stage"] == "executed")
    return f"""
<section data-product-surface="actions">
  <div class="cs-page-head">
    <div class="cs-kicker">Actions</div>
    <h1>Action records</h1>
    <p>Every action starts with a preview; approved and executed lifecycle records remain visible here.</p>
  </div>
  <div class="cs-collection-workbench">
    <div>
      {_collection_summary([("Records", len(actions)), ("Need approval", approval_count), ("Executed", executed_count)])}
      <div class="cs-collection-list">{rows}</div>
    </div>
    <aside class="cs-stack">
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Dry-run posture</h2>
        <p class="cs-muted">Actions stay as previews until policy, approval, source support, and auditability are clear.</p>
        <div class="cs-review-box">
          <a class="cs-button" href="/inbox">Review approvals</a>
          <a class="cs-button secondary" href="/audit">Open History</a>
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


def _inbox_page(
    ctx: dict[str, Any],
    lane: str,
    lane_items: list[dict[str, Any]],
    visible_items: list[dict[str, Any]],
    page_info: dict[str, int],
) -> str:
    all_items = ctx["inbox"]
    active_label = dict(INBOX_LANES).get(lane, "Needs review")
    selected = ctx.get("selected_inbox_item") if isinstance(ctx.get("selected_inbox_item"), dict) else None
    selected_ref = str(selected.get("record_ref") or "") if selected else ""
    rows = "".join(
        _inbox_table_row(
            item,
            lane,
            str(item.get("record_ref") or "") == selected_ref,
            page=page_info["page"],
        )
        for item in visible_items
    )
    if not rows:
        rows = (
            f'<div class="cs-empty">No items are waiting in {h(active_label.lower())}.</div>'
            if all_items
            else _inbox_empty()
        )
    detail = _inbox_detail_panel(
        selected,
        ctx.get("selected_product_loop") if isinstance(ctx.get("selected_product_loop"), dict) else None,
        ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {},
    )
    counts = _inbox_counts(all_items)
    count_keys = {
        "needs-review": "needs_review",
        "evidence-gaps": "evidence_gaps",
        "approval-requests": "approval_requests",
        "policy-blocked": "policy_blocked",
        "failed-runs": "failed",
    }
    lane_tab_rows: list[str] = []
    for slug, lane_label in INBOX_LANES:
        active_class = " is-active" if slug == lane else ""
        current = ' aria-current="page"' if slug == lane else ""
        lane_tab_rows.append(
            f'<a class="cs-inbox-tab{active_class}" href="/inbox?lane={h(slug)}"{current}>'
            f'{h(lane_label)}<strong>{h(str(counts[count_keys[slug]]))}</strong></a>'
        )
    lane_tabs = "".join(lane_tab_rows)
    pager = _inbox_pager(lane, page_info["page"], page_info["page_count"])
    return f"""
<section data-product-surface="inbox" data-inbox-lane="{h(lane)}" data-selected-item="{h(selected_ref)}" data-inbox-page="{h(str(page_info['page']))}" data-inbox-total="{h(str(len(lane_items)))}">
  <div class="cs-page-head">
    <div class="cs-kicker">Review</div>
    <h1>Work that needs attention</h1>
    <p>Choose a lane, inspect one item, then continue in the source, Brief, Claim, or action record.</p>
  </div>
  <nav class="cs-inbox-tabs" aria-label="Review lanes">{lane_tabs}</nav>
  <div class="cs-inbox-workbench">
    <section aria-labelledby="cs-inbox-list-title">
      <div class="cs-inbox-list-heading">
        <div>
          <h2 id="cs-inbox-list-title">{h(active_label)}</h2>
          <p class="cs-muted">Showing {h(str(page_info['page_start']))}-{h(str(page_info['page_end']))} of {h(str(len(lane_items)))} open item{"s" if len(lane_items) != 1 else ""} in this lane.</p>
        </div>
      </div>
      <div class="cs-inbox-table" role="list" aria-label="{h(active_label)} items">
        {rows}
      </div>
      {pager}
    </section>
    {detail}
  </div>
</section>
"""


def _inbox_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "needs_review": sum(1 for item in items if item.get("queue") == "Needs review"),
        "approval_requests": sum(1 for item in items if item.get("queue") == "Approval requests"),
        "policy_blocked": sum(1 for item in items if item.get("queue") == "Policy blocked"),
        "failed": sum(1 for item in items if item.get("queue") == "Failed runs"),
        "evidence_gaps": sum(1 for item in items if int(item.get("evidence_gap_count", 0) or 0) > 0),
    }


def _inbox_table_row(item: dict[str, Any], lane: str, selected: bool = False, *, page: int = 1) -> str:
    selected_class = " is-selected" if selected else ""
    priority_state = "failed" if item.get("priority") == "High" else "underReview" if item.get("priority") == "Medium" else "draft"
    item_ref = str(item.get("record_ref") or "")
    page_query = f"&amp;page={page}" if page > 1 else ""
    selection_href = f"/inbox?lane={quote(lane, safe='')}{page_query}&amp;item={quote(item_ref, safe='')}#selected-work"
    current = ' aria-current="true"' if selected else ""
    return f"""
<a class="cs-inbox-row{selected_class}" href="{selection_href}" role="listitem"{current}>
  <span class="cs-inbox-icon">{icon(record_icon(str(item.get("kind") or "")))}</span>
  <span class="cs-inbox-item-title">
    <strong>{h(item["title"])}</strong>
    <span class="cs-meta">{h(item.get("type") or item["kind"])} · {h(item["date"])}</span>
    <span class="cs-muted">{h(item["detail"])}</span>
  </span>
  <span class="cs-inbox-row-state">{_chip(item.get("priority") or "Medium", priority_state)}{_chip(item["label"], item["state"])}</span>
</a>
"""


def _inbox_pager(lane: str, page: int, page_count: int) -> str:
    if page_count <= 1:
        return ""
    previous_href = f"/inbox?lane={quote(lane, safe='')}&amp;page={page - 1}"
    next_href = f"/inbox?lane={quote(lane, safe='')}&amp;page={page + 1}"
    previous = (
        f'<a class="cs-button secondary" rel="prev" href="{previous_href}">Previous</a>'
        if page > 1
        else '<span class="cs-button secondary" aria-disabled="true">Previous</span>'
    )
    following = (
        f'<a class="cs-button secondary" rel="next" href="{next_href}">Next</a>'
        if page < page_count
        else '<span class="cs-button secondary" aria-disabled="true">Next</span>'
    )
    return f'<nav class="cs-pagination" aria-label="Review item pages">{previous}<span>Page {page} of {page_count}</span>{following}</nav>'


def _inbox_detail_panel(
    item: dict[str, Any] | None,
    product_loop_result: dict[str, Any] | None = None,
    scope: dict[str, str] | None = None,
) -> str:
    if not item:
        return f"""
<aside class="cs-panel flat" id="selected-work">
  <h2 class="cs-section-title">Selected item</h2>
  {_empty_state(
        "Queue empty",
        "No selected work",
        "Save a source, draft a Claim, or preview an action to create reviewable work.",
        "Start from Home",
        "/",
        "Open Briefs",
        "/briefs",
        mark="I",
    )}
</aside>
"""
    reason = _inbox_waiting_reason(item)
    linked_sources = _inbox_linked_sources(item)
    journey_timeline = _inbox_journey_timeline(product_loop_result, scope or {})
    record_ref = str(item.get("record_ref") or "")
    return f"""
<aside class="cs-panel flat cs-inbox-detail" id="selected-work">
  <div class="cs-inbox-detail-title">
    <span class="cs-inbox-icon">{icon(record_icon(str(item.get("kind") or "")))}</span>
    <div>
      <div class="cs-kicker">Selected item</div>
      <h2>{h(item["title"])}</h2>
      <p class="cs-muted">{h(item["detail"])}</p>
    </div>
  </div>
  <div class="cs-row">{_chip(item.get("queue") or "Needs review", "underReview")}{_chip(item.get("priority") or "Medium", "underReview")}{_chip(item["label"], item["state"])}</div>
  <dl class="cs-detail-grid">
    <dt>Type</dt><dd>{h(item.get("type") or item["kind"])}</dd>
    <dt>Owner</dt><dd>{h(item.get("owner") or "Owner")}</dd>
    <dt>Updated</dt><dd>{h(item["date"])}</dd>
  </dl>
  <section class="cs-inbox-action-panel">
    <h3 class="cs-section-title">Continue review</h3>
    <div class="cs-inbox-actions">
      <a class="cs-button" href="{h(item["href"])}">{h(record_action(str(item.get("kind") or "")))}</a>
      <a class="cs-button secondary" href="/search?q={quote(item["title"])}">Find related sources</a>
      <a class="cs-button secondary" href="/audit?record={quote(record_ref, safe='')}">Open item history</a>
    </div>
  </section>
  <details class="cs-inbox-context">
    <summary><strong>Why this is here and linked sources</strong></summary>
    <p>{h(reason)}</p>
    <div class="cs-inbox-linked-list">{linked_sources}</div>
  </details>
  <details class="cs-inbox-context">
    <summary><strong>Related journey</strong></summary>
    {journey_timeline}
  </details>
</aside>
"""


def _inbox_journey_timeline(
    product_loop_result: dict[str, Any] | None,
    scope: dict[str, str],
) -> str:
    loop = product_loop_result.get("product_loop") if isinstance(product_loop_result, dict) else None
    validation = loop.get("loop_validation") if isinstance(loop, dict) else None
    stages = loop.get("stages") if isinstance(loop, dict) else None
    expected_stages = ["Inbox", "Brief", "Claim", "Memory/Wiki", "Action", "Learn"]
    loop_scope = loop.get("scope") if isinstance(loop, dict) else None
    selection = product_loop_result.get("selection") if isinstance(product_loop_result, dict) else None
    selected_ref = str(selection.get("selected_ref") or "") if isinstance(selection, dict) else ""
    stage_refs = {
        str(ref)
        for stage in stages or []
        if isinstance(stage, dict)
        for ref in stage.get("record_refs", [])
        if isinstance(ref, str) and ref
    }
    valid = (
        isinstance(loop, dict)
        and isinstance(validation, dict)
        and validation.get("status") == "validated"
        and isinstance(stages, list)
        and [stage.get("stage") for stage in stages if isinstance(stage, dict)] == expected_stages
        and loop_scope == scope
        and (not selected_ref or selected_ref in stage_refs)
    )
    if not valid:
        return f"""
<section class="cs-journey-timeline is-recovery" data-product-state="failed-with-recovery">
  <div class="cs-journey-header">
    <div>
      <div class="cs-kicker">Work journey</div>
      <h3>Journey details unavailable</h3>
      <p>CornerStone kept this item in the current workspace, but its linked work could not be shown safely.</p>
    </div>
    {_chip("Review safely", "underReview")}
  </div>
  {_inbox_journey_recovery_details()}
</section>
"""

    stage_html = "".join(_inbox_journey_stage(stage) for stage in stages if isinstance(stage, dict))
    return f"""
<section
  class="cs-journey-timeline"
  aria-labelledby="cs-journey-title"
  data-vs4-ops-inbox-journey-timeline="runtime-loop-view"
  data-vs4-loop-recovery-no-authority-expansion="true"
  data-vs4-loop-recovery-no-live-writeback="true"
>
  <div class="cs-journey-header">
    <div>
      <div class="cs-kicker">Work journey</div>
      <h3 id="cs-journey-title">Inbox to review outcome</h3>
      <p>See where this work came from and what still needs review.</p>
    </div>
    <span class="cs-chip cs-chip-searchable">Current saved journey</span>
  </div>
  <p class="cs-meta">Workspace: {h(_scope_label(scope))}</p>
  <ol class="cs-timeline cs-journey-stage-list">{stage_html}</ol>
  {_inbox_journey_recovery_details()}
</section>
"""


def _inbox_journey_stage(stage: dict[str, Any]) -> str:
    label = str(stage.get("stage") or "Stage")
    runtime_status = str(stage.get("status") or "not_requested")
    status_label, status_chip, status_state = _inbox_journey_status(stage)
    description = str(stage.get("description") or "Review this part of the journey before continuing.")
    record_refs = _string_refs(stage.get("record_refs"))
    evidence_refs = _string_refs(stage.get("evidence_refs"))
    audit_refs = [ref for ref in _string_refs(stage.get("audit_refs")) if ref.startswith("audit:")]
    primary_refs = " | ".join(record_refs)
    evidence_attr = " | ".join(evidence_refs)
    audit_attr = " | ".join(audit_refs)
    next_cues = {
        "Inbox": "Next: open the selected review item.",
        "Brief": "Next: check findings, gaps, and source support.",
        "Claim": "Next: review the candidate before any approval.",
        "Memory/Wiki": "Next: keep this knowledge candidate in review.",
        "Action": "Next: inspect the preview and approval boundary.",
        "Learn": "Next: review the outcome lesson before broader use.",
    }

    def _ref_detail(title: str, refs: list[str], empty: str) -> str:
        values = " ".join(f"<code>{h(ref)}</code>" for ref in refs) if refs else h(empty)
        return f"<dt>{h(title)}</dt><dd>{values}</dd>"

    return f"""
<li>
  <article
    class="cs-timeline-item cs-journey-stage is-{h(status_state)}"
    data-vs4-journey-stage="{h(label)}"
    data-vs4-journey-stage-status="{h(status_state)}"
    data-vs4-journey-runtime-status="{h(runtime_status)}"
    data-vs4-journey-description="{h(description)}"
    data-vs4-journey-stage-ref="{h(primary_refs)}"
    data-vs4-journey-evidence-refs="{h(evidence_attr)}"
    data-vs4-journey-audit-refs="{h(audit_attr)}"
  >
    <span class="cs-dot" aria-hidden="true"></span>
    <div class="cs-journey-stage-body">
      <div class="cs-journey-stage-heading">
        <strong>{h(label)}</strong>
        {_chip(status_label, status_chip)}
      </div>
      <p>{h(description)}</p>
      <span class="cs-meta">{h(next_cues.get(label, "Next: continue review."))}</span>
      <details class="cs-audit-detail">
        <summary>Evidence and activity refs</summary>
        <dl class="cs-journey-ref-grid">
          {_ref_detail("Work record", record_refs, "Not linked yet")}
          {_ref_detail("Evidence", evidence_refs, "No evidence ref available")}
          {_ref_detail("Activity", audit_refs, "No activity ref available")}
        </dl>
      </details>
    </div>
  </article>
</li>
"""


def _string_refs(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(ref) for ref in value if isinstance(ref, str) and ref))


def _inbox_journey_status(stage: dict[str, Any]) -> tuple[str, str, str]:
    status = str(stage.get("status") or "not_requested").lower().replace("-", "_")
    if any(marker in status for marker in ("blocked", "denied", "failed", "mismatch")):
        return "Blocked / recovery", "policyBlocked", "blocked"
    if status == "not_requested":
        return "Not linked yet", "underReview", "needs-review"
    if bool(stage.get("review_required")) or any(
        marker in status for marker in ("draft", "review", "pending", "required", "candidate")
    ):
        return "Needs review", "underReview", "needs-review"
    return "Ready", "saved", "ready"


def _inbox_journey_recovery_details() -> str:
    return """
<details class="cs-journey-recovery">
  <summary>Safe recovery behavior</summary>
  <div class="cs-journey-recovery-list">
    <div data-vs4-loop-recovery-state="missing-ref" data-vs4-loop-recovery-status="blocked-safe">
      <strong>Missing work item</strong>
      <span>CornerStone could not find one requested work item. Nothing new was approved or sent.</span>
    </div>
    <div data-vs4-loop-recovery-state="cross-scope" data-vs4-loop-recovery-status="blocked-safe">
      <strong>Outside workspace</strong>
      <span>Work from another workspace stays outside this journey. Workspace scope stayed unchanged.</span>
    </div>
    <div data-vs4-loop-recovery-state="lineage-mismatch" data-vs4-loop-recovery-status="blocked-safe">
      <strong>Different journey</strong>
      <span>CornerStone did not combine unrelated source-linked work. No new journey or activity record was created by the workflow; only this page view was added to History.</span>
    </div>
  </div>
</details>
"""


def _inbox_linked_sources(item: dict[str, Any]) -> str:
    title = item.get("title") or "Selected work"
    source_label = "Source support" if item.get("kind") == "Claim" else "Review sources"
    if item.get("kind") == "Action":
        source_label = "Dry-run impact"
    return f"""
<a class="cs-inbox-linked-row" href="/search?q={quote(title)}">
  <span class="cs-inbox-type-mark">{icon("search", class_name="cs-icon is-small")}</span>
  <span><strong>{h(source_label)}</strong><span class="cs-meta">Search matching local records before deciding.</span></span>
</a>
<a class="cs-inbox-linked-row" href="/audit">
  <span class="cs-inbox-type-mark">{icon("history", class_name="cs-icon is-small")}</span>
  <span><strong>History</strong><span class="cs-meta">Open the local event history for this work.</span></span>
</a>
"""


def _inbox_waiting_reason(item: dict[str, Any]) -> str:
    kind = item.get("kind") or item.get("type") or "Record"
    if kind == "Action":
        return "This action is waiting because the product only previews external impact until the owner opens the approval path."
    if kind == "Claim":
        return "This claim is waiting because it still needs source support before it can become a defensible decision."
    if kind == "Brief":
        return "This brief is waiting because sources and gaps should be checked before it is used for a decision."
    if kind == "Memory":
        return "This knowledge draft is waiting because saved learning stays draft-only until the owner reviews it."
    return "This item is waiting because it needs owner review before it moves further in the workflow."


def _inbox_empty() -> str:
    return _empty_state(
        "Day zero",
        "No work waiting",
        "When a brief, claim, or action preview needs review, it will appear here with a clear next step.",
        "Start from Home",
        "/",
        "Open History",
        "/audit",
        mark="I",
        steps=[
            ("1. Save source", "Create the first local record."),
            ("2. Draft work", "Briefs, claims, and previews can enter review."),
            ("3. Decide", "Open the item and inspect support."),
        ],
        receipts=[
            ("Review item", "Briefs, claims, and actions enter one queue."),
            ("Owner state", "Priority and risk are visible before action."),
            ("Next action", "Open the selected work with sources nearby."),
        ],
    )


def _audit_lifecycle_key(event_type: str) -> str:
    value = event_type.lower()
    if ".approved" in value or value.endswith(".approve") or "owner_approved" in value:
        return "approved"
    if any(marker in value for marker in (".denied", ".blocked", "policy_blocked", "failed")):
        return "blocked"
    if ".executed" in value or value.endswith(".execute"):
        return "executed"
    if any(marker in value for marker in (".created", ".create", ".ingested", ".ingest", ".draft", ".proposed")):
        return "created"
    if any(marker in value for marker in (".read", ".opened", ".viewed")):
        return "opened"
    return "other"


def _audit_record_ref(event: dict[str, Any]) -> str:
    subject = event.get("subject") if isinstance(event.get("subject"), dict) else {}
    subject_type = str(subject.get("type") or "").strip()
    subject_id = str(subject.get("id") or "").strip()
    return f"{subject_type}:{subject_id}" if subject_type and subject_id else ""


def _audit_icon_asset(event: dict[str, Any]) -> str:
    subject = event.get("subject") if isinstance(event.get("subject"), dict) else {}
    subject_type = str(subject.get("type") or "").lower()
    return {
        "artifact": "document",
        "brief": "brief",
        "claim": "shield-check",
        "action": "action",
        "conversation": "chat",
    }.get(subject_type, "history")


def _audit_page(
    ctx: dict[str, Any],
    record_filter: str = "",
    lifecycle_filter: str = "all",
    page: int = 1,
) -> str:
    events = ctx.get("audit_all") if isinstance(ctx.get("audit_all"), list) else ctx["audit"]
    integrity = ctx.get("audit_integrity") if isinstance(ctx.get("audit_integrity"), dict) else {}
    integrity_status = str(integrity.get("status") or "not_verified")
    if integrity_status == "success":
        integrity_label, integrity_state = "Hash chain verified", "searchable"
    elif integrity_status == "failed":
        integrity_label, integrity_state = "Integrity failed", "failed"
    else:
        integrity_label, integrity_state = "Integrity unavailable", "underReview"

    filtered = [
        event
        for event in events
        if (not record_filter or _audit_record_ref(event) == record_filter)
        and (lifecycle_filter == "all" or _audit_lifecycle_key(str(event.get("event_type") or "")) == lifecycle_filter)
    ]
    page_size = 40
    page_count = max(1, (len(filtered) + page_size - 1) // page_size)
    current_page = min(max(page, 1), page_count)
    offset = (current_page - 1) * page_size
    visible_events = filtered[offset : offset + page_size]

    record_labels: dict[str, str] = {}
    for event in events:
        record_ref = _audit_record_ref(event)
        if record_ref and record_ref not in record_labels:
            subject_id = record_ref.split(":", 1)[1]
            record_labels[record_ref] = f"{_audit_subject_label(event)} · {_short_ref(subject_id, 18)}"
    record_options = ['<option value="">All records</option>']
    if record_filter and record_filter not in record_labels:
        record_options.append(
            f'<option value="{h(record_filter)}" selected>'
            f'Unavailable record · {h(_short_ref(record_filter, 24))}</option>'
        )
    record_options.extend(
        f'<option value="{h(ref)}"{" selected" if ref == record_filter else ""}>{h(label)}</option>'
        for ref, label in sorted(record_labels.items(), key=lambda item: item[1].lower())
    )
    lifecycle_options = []
    for value, label in AUDIT_LIFECYCLES:
        selected = " selected" if value == lifecycle_filter else ""
        lifecycle_options.append(f'<option value="{h(value)}"{selected}>{h(label)}</option>')

    positions = {
        str(event.get("event_id") or id(event)): len(events) - index
        for index, event in enumerate(events)
    }
    if visible_events:
        row_html = "".join(
            f"""
<article class="cs-audit-row">
  <div class="cs-audit-row-main">
    <span class="cs-audit-icon">{icon(_audit_icon_asset(event))}</span>
    <div>
      <div class="cs-audit-row-top">
        <h2>{h(_plain_event(str(event.get("event_type") or "")))}</h2>
        <span class="cs-audit-row-position">{h(_audit_subject_label(event))}</span>
      </div>
      <div class="cs-audit-row-meta">
        <span>{h(_audit_family(str(event.get("event_type") or "")))}</span>
        <span>{h(_display_date(event))}</span>
        <span>{_breakable_ref(_audit_record_ref(event) or "No subject reference")}</span>
      </div>
    </div>
  </div>
  {_audit_detail(event, positions[str(event.get("event_id") or id(event))])}
</article>
"""
            for event in visible_events
        )
        rows = f'<div class="cs-audit-list">{row_html}</div>'
    elif events:
        rows = """
<div class="cs-empty">
  No history matches these filters. Clear the filters to return to the full workspace history.
</div>
"""
    else:
        rows = _empty_state(
            "History ready",
            "No activity recorded yet",
            "Save a source, ask a question, or record a decision to start the workspace history.",
            "Start from Home",
            "/",
            "Open saved sources",
            "/artifacts",
            mark="H",
        )

    def page_href(target_page: int) -> str:
        params: list[str] = []
        if record_filter:
            params.append(f"record={quote(record_filter, safe='')}")
        if lifecycle_filter != "all":
            params.append(f"lifecycle={quote(lifecycle_filter, safe='')}")
        if target_page > 1:
            params.append(f"page={target_page}")
        return "/audit" + (f"?{'&amp;'.join(params)}" if params else "")

    pagination = ""
    if page_count > 1:
        previous = (
            f'<a class="cs-button secondary" href="{page_href(current_page - 1)}">Previous</a>'
            if current_page > 1
            else ""
        )
        following = (
            f'<a class="cs-button secondary" href="{page_href(current_page + 1)}">Next</a>'
            if current_page < page_count
            else ""
        )
        pagination = f'<nav class="cs-audit-pagination" aria-label="History pages">{previous}<span>Page {current_page} of {page_count}</span>{following}</nav>'

    scope = ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {}
    workspace = str(scope.get("workspace_id") or "default")
    showing_start = offset + 1 if visible_events else 0
    showing_end = offset + len(visible_events)
    has_filters = bool(record_filter or lifecycle_filter != "all")
    clear_link = '<a class="cs-button ghost" href="/audit">Clear</a>' if has_filters else ""
    return f"""
<section data-product-surface="audit" data-audit-integrity-status="{h(integrity_status)}">
  <header class="cs-audit-hero">
    <div class="cs-brief-title">
      <div class="cs-kicker">Audit</div>
      <h1>History</h1>
      <p>See what happened in plain language. Event hashes, exact types, scope, and stored fields stay behind each raw-detail disclosure.</p>
      <div class="cs-brief-meta">
        <span>Workspace: {h(workspace)}</span>
        <span>{h(str(len(events)))} total events</span>
        <span>{h(integrity_label)} across the full local ledger</span>
      </div>
    </div>
    <div class="cs-audit-actions">
      {_chip(integrity_label, integrity_state)}
      <a class="cs-button secondary" href="/artifacts">Open saved sources</a>
    </div>
  </header>
  <form class="cs-audit-filters" action="/audit" method="get" aria-label="Filter history">
    <label>
      <span>Record</span>
      <select name="record">{''.join(record_options)}</select>
    </label>
    <label>
      <span>Lifecycle</span>
      <select name="lifecycle">{''.join(lifecycle_options)}</select>
    </label>
    <button class="cs-button" type="submit">Apply filters</button>
    {clear_link}
  </form>
  <div class="cs-audit-workbench">
    <section class="cs-panel flat cs-audit-list-panel">
      <div class="cs-panel-header">
        <div>
          <h2>Workspace history</h2>
          <p class="cs-muted">Showing {showing_start}-{showing_end} of {h(str(len(filtered)))} matching events, newest first.</p>
        </div>
      </div>
      {rows}
      {pagination}
    </section>
    <aside class="cs-stack">
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Ledger integrity</h2>
        <p class="cs-muted">{h(integrity_label)} applies to the full local ledger; active workspace rows are filtered below.</p>
        <dl class="cs-detail-grid">
          <dt>Scoped events</dt><dd>{h(str(len(events)))}</dd>
          <dt>Matching events</dt><dd>{h(str(len(filtered)))}</dd>
          <dt>Reading order</dt><dd>Newest first</dd>
          <dt>Raw detail</dt><dd>Closed by default</dd>
        </dl>
      </section>
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Continue</h2>
        <div class="cs-review-box">
          <a class="cs-button secondary" href="/artifacts">Open saved sources</a>
          <a class="cs-button ghost" href="/">Back to Home</a>
        </div>
      </section>
    </aside>
  </div>
</section>
"""


def _owner_review_page(ctx: dict[str, Any], readiness: dict[str, Any]) -> str:
    scope = ctx["scope"]
    connector_rows = _owner_connector_rows(ctx)
    activity_rows = _owner_connector_activity(ctx)
    gate = "true" if readiness.get("local_scenario_ready") is True else "false"
    runtime = "true" if readiness.get("vs0_runtime_ready") is True else "false"
    return f"""
<section data-product-surface="owner-review">
  <div class="cs-brief-hero is-stacked">
    <div class="cs-brief-title">
      <div class="cs-kicker">Owner area / Admin connectors</div>
      <h1>Connector governance console</h1>
      <p>Connector governance stays outside the daily workspace. Review source access, policy posture, namespace scope, and recent connector activity before enabling any external path.</p>
      <div class="cs-brief-meta">
        <span>Namespace: {h(scope.get("namespace_id") or "personal")}</span>
        <span>Workspace: {h(scope.get("workspace_id") or "default")}</span>
        <span>Owner: {h(scope.get("owner_id") or "local-user")}</span>
      </div>
    </div>
    <div class="cs-brief-actions">
      {_chip("Admin containment", "underReview")}
      {_chip("Local only", "searchable")}
      {_chip("External calls locked", "policyBlocked")}
    </div>
  </div>
  <div class="cs-owner-overview" aria-label="Admin containment">
    {_owner_metric("Connected source posture", f"{len(ctx['artifacts'])} saved", "Local artifacts and pasted sources only.")}
    {_owner_metric("Policy controls", "Dry-run first", "Actions require policy and approval before execution.")}
    {_owner_metric("Access roles", "Owner scoped", "Admin review is tied to the current local owner.")}
    {_owner_metric("Admin containment", "Review input", "Runtime and scenario status stay in the owner handoff.")}
  </div>
  <div class="cs-connector-grid">
    <div class="cs-owner-main-stack">
      <section class="cs-panel">
        <div class="cs-panel-header">
          <div>
            <h2>Connected source posture</h2>
            <p class="cs-muted">Each source shows whether it can read, write, or only simulate work in this local workspace.</p>
          </div>
          {_chip("Review before enablement", "underReview")}
        </div>
        <div class="cs-connector-table-head" aria-hidden="true">
          <span>Source</span>
          <span>Access</span>
          <span>Policy</span>
          <span>Activity / scope</span>
          <span>Status</span>
        </div>
        <div class="cs-connector-list" aria-label="Connected source posture rows">{connector_rows}</div>
      </section>
      <section class="cs-panel">
        <div class="cs-panel-header">
          <div>
            <h2>Namespace settings</h2>
            <p class="cs-muted">Scope is explicit so connector posture does not bleed across workspaces.</p>
          </div>
          {_chip("Owner scoped", "underReview")}
        </div>
        <div class="cs-owner-scope-table" aria-label="Namespace settings">
          {_owner_scope_row("Tenant", str(scope.get("tenant_id") or "local-dev"))}
          {_owner_scope_row("Namespace ID", str(scope.get("namespace_id") or "personal"))}
          {_owner_scope_row("Workspace", str(scope.get("workspace_id") or "default"))}
          {_owner_scope_row("Owner", str(scope.get("owner_id") or "local-user"))}
          {_owner_scope_row("Isolation", "Logical isolation, workspace scoped")}
          {_owner_scope_row("Retention", "Artifacts and audit history remain local review input")}
        </div>
      </section>
    </div>
    <aside class="cs-admin-stack">
      <section class="cs-panel">
        <div class="cs-panel-header">
          <div>
            <h2>Policy controls</h2>
            <p class="cs-muted">These controls describe the local review boundary; they do not enable live providers.</p>
          </div>
          {_chip("Contained", "underReview")}
        </div>
        <div class="cs-policy-list">{_owner_policy_rows()}</div>
      </section>
      <section class="cs-admin-note">
        <strong>Admin containment</strong>
        <span>Connector policy, role, and provider settings are intentionally kept behind the owner area. Daily users see only the resulting source, claim, action, inbox, and audit states.</span>
      </section>
      <section class="cs-panel">
        <div class="cs-panel-header"><h2>Access roles</h2>{_chip("Owner controlled", "underReview")}</div>
        <div class="cs-stat-list">
          {_owner_role_row("Owner", "Can inspect local gates and approve contained connector setup.", "1")}
          {_owner_role_row("Workspace user", "Can save sources, ask questions, and review drafts without connector administration.", "local")}
          {_owner_role_row("External provider", "No direct access from this local review page.", "0")}
        </div>
      </section>
      <section class="cs-panel">
        <div class="cs-panel-header">
          <div>
            <h2>Recent connector activity</h2>
            <p class="cs-muted">Admin review sees connector-adjacent activity without turning it into a normal user task.</p>
          </div>
          {_chip("Audit visible", "searchable")}
        </div>
        <div class="cs-timeline">{activity_rows}</div>
      </section>
      <section class="cs-panel">
        <div class="cs-panel-header">
          <div>
            <h2>Reference images</h2>
            <p class="cs-muted">Open the benchmark set used for UI direction. These images remain review input only.</p>
          </div>
          {_chip("Design input", "searchable")}
        </div>
        <a class="cs-button secondary" href="/review/reference-images">Open reference gallery</a>
      </section>
      {_owner_human_review_handoff(gate, runtime)}
    </aside>
  </div>
</section>
"""


def _owner_reference_images_page(root: Path) -> str:
    image_dir = root / "docs" / "design" / "reference-images"
    cards = "".join(_owner_reference_card(image_dir, filename, title, posture, detail) for filename, title, posture, detail in REFERENCE_IMAGE_ROWS)
    return f"""
<section
  data-product-surface="owner-review"
  data-vs4-reference-images-pass-evidence="false"
  data-vs4-reference-images-acceptance-evidence="false"
>
  <div class="cs-brief-hero is-stacked">
    <div class="cs-brief-title">
      <div class="cs-kicker">Owner area</div>
      <h1>Reference image gallery</h1>
      <p>Review the full visual benchmark set beside the current implementation. These files guide layout, hierarchy, and interaction direction; they do not prove scenario PASS or human acceptance.</p>
    </div>
    <div class="cs-brief-actions">
      {_chip("Review input only", "underReview")}
      {_chip(f"{len(REFERENCE_IMAGE_ROWS)} references", "searchable")}
      <a class="cs-button secondary" href="/review">Back to Owner</a>
    </div>
  </div>
  <section class="cs-admin-note">
    <strong>Implementation boundary</strong>
    <span>Use structure and hierarchy from these images. Do not copy fictional users, organizations, counts, vendors, or unavailable controls into the product.</span>
  </section>
  <div class="cs-reference-grid" aria-label="CornerStone UI reference images">
    {cards}
  </div>
</section>
"""


def _owner_reference_card(image_dir: Path, filename: str, title: str, posture: str, detail: str) -> str:
    file_path = image_dir / filename
    size_label = "missing"
    if file_path.exists():
        size_label = f"{round(file_path.stat().st_size / 1024)} KB"
    src = f"/review/reference-images/{quote(filename)}"
    return f"""
<article class="cs-reference-card" data-vs4-reference-image="{h(filename)}">
  <img src="{h(src)}" alt="{h(title)} reference screen" loading="lazy">
  <div class="cs-reference-body">
    <div class="cs-row">{_chip(posture, "underReview" if "Dormant" in posture or "Owner" in posture else "searchable")}{_chip(size_label, "draft")}</div>
    <h2>{h(title)}</h2>
    <p>{h(detail)}</p>
    <span class="cs-meta">{h(filename)}</span>
  </div>
</article>
"""


def _owner_human_review_handoff(gate: str, runtime: str) -> str:
    checkpoints = [
        ("drop-ask", "Drop and Ask"),
        ("evidence-backed-brief", "Evidence-backed Brief"),
        ("claim-candidate", "Claim candidate"),
        ("memory/wiki-candidate", "Memory/Wiki candidate"),
        ("action-card", "Action Card"),
        ("ops-inbox", "Ops Inbox"),
        ("evidence-audit", "Evidence and audit"),
        ("learn", "Learn review"),
        ("desktop", "Desktop capture"),
        ("mobile", "Mobile capture"),
        ("keyboard", "Keyboard pass"),
        ("unsafe-states", "Unsafe states"),
    ]
    checkpoint_rows = "".join(
        f'<div class="cs-stat-row" data-vs4-human-review-checkpoint="{h(key)}"><span class="cs-stat-icon">R</span><div><strong>{h(label)}</strong><div class="cs-meta">Review input only.</div></div>{_chip("Input", "searchable")}</div>'
        for key, label in checkpoints
    )
    artifacts = [
        "reports/human-gates/vs4/review-kit.json",
        "reports/human-gates/vs4/record-templates/VS4-H01.review-record.template.json",
        "reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json",
    ]
    artifact_rows = "".join(f'<li><code data-vs4-human-review-artifact>{h(path)}</code></li>' for path in artifacts)
    commands = [
        "cornerstone human-gate validate-record --scope vs4 --scenario VS4-H01",
        "cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json",
        "make verify-vs4-product-alpha-human-review-handoff",
    ]
    command_rows = "".join(f'<li><code data-vs4-human-review-command>{h(command)}</code></li>' for command in commands)
    return f"""
<section
  class="cs-panel"
  id="vs4-human-review-handoff"
  data-vs4-human-review-handoff="visible"
  data-vs4-human-review-status="human-required"
  data-vs4-human-review-input-only="true"
  data-vs4-human-ux-claimed="false"
  data-vs4-reference-images-acceptance-evidence="false"
  data-vs4-review-workspace="personal-project-default"
>
  <div class="cs-panel-header">
    <div>
      <h2>Owner review handoff</h2>
      <p class="cs-muted">Human review required. This material is review input only. No acceptance collected.</p>
    </div>
    {_chip("Human required", "underReview")}
  </div>
  <div class="cs-stat-list">{checkpoint_rows}</div>
  <aside class="cs-admin-note" data-vs4-human-review-package="review-input-only">
    <strong>Package is not acceptance.</strong>
    <span>VS4-H01 remains HUMAN_REQUIRED. Scope: local-user / personal / default. Acceptance claim: not collected. Reference images: design input only.</span>
    <span class="cs-meta">local_scenario_ready={h(gate)}; vs0_runtime_ready={h(runtime)}</span>
    <details data-vs4-human-review-detail="progressive">
      <summary>Review artifacts and commands</summary>
      <ul>{artifact_rows}</ul>
      <ul>{command_rows}</ul>
    </details>
  </aside>
</section>
"""


def _owner_metric(label: str, value: str, detail: str) -> str:
    return f"""
<div class="cs-owner-metric">
  <span class="cs-meta">{h(label)}</span>
  <strong>{h(value)}</strong>
  <span class="cs-muted">{h(detail)}</span>
</div>
"""


def _owner_connector_rows(ctx: dict[str, Any]) -> str:
    artifact_count = len(ctx["artifacts"])
    action_count = len(ctx["actions"])
    rows = [
        {
            "name": "Local source intake",
            "body": "Files and pasted text enter as preserved sources before any derived brief, claim, or action can rely on them.",
            "state": "Enabled",
            "chip": "saved",
            "mark": "L",
            "access": "Read local input",
            "policy": "Preserve original",
            "activity": f"{artifact_count} saved",
            "scope": "personal sources",
        },
        {
            "name": "Evidence and search index",
            "body": "Search and evidence bundles are scoped to the current local workspace and keep citations close to each result.",
            "state": "Enabled",
            "chip": "searchable",
            "mark": "E",
            "access": "Read indexed sources",
            "policy": "Scope matched",
            "activity": f"{len(ctx['briefs']) + len(ctx['claims'])} linked drafts",
            "scope": "workspace index",
        },
        {
            "name": "External writeback providers",
            "body": "Writeback-capable connectors stay locked behind dry-run previews, policy decisions, approval, and audit records.",
            "state": "Locked",
            "chip": "policyBlocked",
            "mark": "X",
            "access": "No live write",
            "policy": "Approval required",
            "activity": f"{action_count} previews",
            "scope": "locked providers",
        },
    ]
    return "".join(_owner_connector_card(row) for row in rows)


def _owner_connector_card(row: dict[str, str]) -> str:
    return f"""
<article class="cs-connector-card">
  <div class="cs-connector-source">
    <div class="cs-connector-title">
      <span class="cs-connector-icon" aria-hidden="true">{h(row["mark"])}</span>
      <h3>{h(row["name"])}</h3>
    </div>
    <p>{h(row["body"])}</p>
  </div>
  <div class="cs-connector-cell"><span class="cs-meta">Access</span><strong>{h(row["access"])}</strong></div>
  <div class="cs-connector-cell"><span class="cs-meta">Policy</span><strong>{h(row["policy"])}</strong></div>
  <div class="cs-connector-cell"><span class="cs-meta">Scope</span><strong>{h(row["scope"])}</strong><span class="cs-muted">{h(row["activity"])}</span></div>
  <div class="cs-row">{_chip(row["state"], row["chip"])}</div>
</article>
"""


def _owner_connector_activity(ctx: dict[str, Any]) -> str:
    events = ctx["audit"][:5]
    if not events:
        return '<div class="cs-empty">Connector activity appears here after sources, searches, drafts, or action previews are recorded.</div>'
    return "".join(
        f"""
<div class="cs-timeline-item">
  <span class="cs-dot"></span>
  <div>
    <strong>{h(_plain_event(str(event.get("event_type") or "")))}</strong>
    <div class="cs-meta">{h(_display_date(event))}</div>
  </div>
</div>
"""
        for event in events
    )


def _owner_policy_rows() -> str:
    rows = [
        ("Default egress", "External paths remain locked unless an owner-scoped policy allows them.", "Allowlist only", "underReview"),
        ("Approval rules", "Action previews require policy, owner approval, and an audit record before execution.", "Enforced", "evidenceBacked"),
        ("Sensitive data handling", "Saved sources remain local; derived drafts stay secondary until reviewed.", "Local review", "searchable"),
    ]
    return "".join(_owner_policy_row(label, detail, state, chip) for label, detail, state, chip in rows)


def _owner_policy_row(label: str, detail: str, state: str, chip: str) -> str:
    return f"""
<div class="cs-policy-row">
  <div>
    <strong>{h(label)}</strong>
    <p>{h(detail)}</p>
  </div>
  {_chip(state, chip)}
</div>
"""


def _owner_scope_row(label: str, value: str) -> str:
    return f"""
<div class="cs-owner-scope-row">
  <strong>{h(label)}</strong>
  <span class="cs-muted">{h(value)}</span>
</div>
"""


def _owner_role_row(label: str, detail: str, count: str) -> str:
    return f"""
<div class="cs-stat-row">
  <span class="cs-stat-icon">{h(label[:1])}</span>
  <div>
    <strong>{h(label)}</strong>
    <div class="cs-meta">{h(detail)}</div>
  </div>
  {_chip(count, "searchable")}
</div>
"""


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


def _artifact_detail(ctx: dict[str, Any], store: Any, artifact: dict[str, Any]) -> str:
    title = _artifact_title(artifact)
    presentation = _artifact_state(ctx, artifact)
    text, truncated = _safe_source_text(store, artifact, errors=ctx.get("load_errors"))
    if not text:
        text = "No readable text preview is available for this source."
    linked = _linked_records(ctx, artifact.get("artifact_id"))
    source = artifact.get("source") if isinstance(artifact.get("source"), dict) else {}
    fingerprint = _fingerprint(artifact.get("checksum_sha256") or artifact.get("original_storage_ref"))
    source_label = _plain_source(source)
    media_type = str(artifact.get("media_type") or "text/plain")
    scope = ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {}
    workspace = str(scope.get("workspace_id") or "default")
    keywords = _artifact_keywords(text, title)
    keyword_rows = "".join(
        f'<div class="cs-keyword-row"><span>{h(keyword)}</span>{_chip(str(count), "searchable")}</div>'
        for keyword, count in keywords
    )
    source_query = quote(title)
    artifact_id = str(artifact.get("artifact_id") or "")
    preview_note = (
        "Showing the first 50,000 characters. The saved original remains unchanged."
        if truncated
        else "Line breaks are preserved from the saved source text."
        if presentation["preview_supported"]
        else str(presentation["explanation"])
    )
    state_chip = "" if presentation["label"] == "Saved" else _chip(str(presentation["label"]), str(presentation["state"]))
    saved_chip = _chip("Saved", "saved") if presentation["saved"] else _chip("Integrity issue", "failed")
    search_action = (
        f'<a class="cs-button" href="/search?q={h(source_query)}">Search this source</a>'
        if presentation["searchable"]
        else ""
    )
    download_action = (
        f'<a class="cs-button secondary" href="/artifacts/{h(artifact_id)}/original">Download original</a>'
        if presentation["saved"]
        else ""
    )
    original_size = int(artifact.get("original_size_bytes") or 0)
    return f"""
<section class="cs-artifact-workbench" data-product-surface="artifact-detail" data-artifact-searchable="{str(bool(presentation['searchable'])).lower()}" data-derived-status="{h(presentation['derived_status'])}" aria-label="Source inspection workspace">
  <div class="cs-stack">
    <header class="cs-artifact-compact-hero">
      <div class="cs-artifact-title">
        <nav class="cs-artifact-breadcrumb" aria-label="Detail path">
          <span class="cs-meta">Detail path</span>
          <a href="/artifacts">Saved sources</a>
          <span aria-hidden="true">/</span>
          <span>{h(_truncate(title, 80))}</span>
        </nav>
        <div class="cs-artifact-title-row">
          <span class="cs-artifact-file-mark">{icon("document")}</span>
          <div>
            <h1>{h(title)}</h1>
            <div class="cs-row">{saved_chip}{state_chip}</div>
          </div>
        </div>
      </div>
      <div class="cs-artifact-actions">
        {search_action}
        {download_action}
        <a class="cs-button ghost" href="/artifacts">Back to saved sources</a>
      </div>
    </header>
    <section class="cs-artifact-viewer" aria-label="Original source document viewer">
      <div class="cs-artifact-toolbar">
        <div class="cs-artifact-toolbar-label">
          <strong>Source text</strong>
          <span class="cs-meta">{h(preview_note)}</span>
        </div>
        <span class="cs-artifact-page-count">{h(media_type)}</span>
      </div>
      <div class="cs-artifact-page-area">
        <article class="cs-document-page" aria-label="Saved source text preview" id="source-text">
          <div class="cs-source-text">{h(text)}</div>
        </article>
      </div>
    </section>
  </div>
  <aside class="cs-stack cs-artifact-rail">
    <section class="cs-artifact-side-card">
      <h2 class="cs-section-title">Source details</h2>
      <dl class="cs-detail-grid">
        <dt>Saved</dt><dd>{h(_display_date(artifact))}</dd>
        <dt>Source</dt><dd>{h(source_label)}</dd>
        <dt>File type</dt><dd>{h(media_type)}</dd>
        <dt>Original size</dt><dd>{h(original_size)} bytes</dd>
        <dt>Representation</dt><dd>{h(presentation["label"])}</dd>
        <dt>Workspace</dt><dd>{h(workspace)}</dd>
        <dt>Fingerprint</dt><dd>{h(fingerprint)}</dd>
      </dl>
      <p class="cs-muted">The fingerprint identifies the saved content. Derived work remains separate.</p>
      <p><strong>Current state:</strong> {h(presentation["explanation"])}</p>
      <p><strong>Next:</strong> {h(presentation["recovery"])}</p>
    </section>
    {linked}
    <details class="cs-artifact-side-card" id="keywords">
      <summary><strong>Frequent local terms</strong> <span class="cs-meta">{len(keywords)}</span></summary>
      <div class="cs-keyword-list">{keyword_rows or '<div class="cs-empty">No keyword preview is available.</div>'}</div>
    </details>
    <a class="cs-button secondary" href="/audit?record=artifact:{h(artifact_id)}">Open source history</a>
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
            rows.append(_generic_row("Decision draft" if _is_decision_draft(claim) else "Claim", _claim_title(claim), "Uses this source.", _detail_href("claims", claim.get("claim_id")), label, state, _display_date(claim)))
    if not rows:
        return '<section class="cs-artifact-side-card" id="linked-work"><h2 class="cs-section-title">Linked work</h2><div class="cs-empty">No Brief or Decision draft is linked to this source yet.</div></section>'
    return f'<section class="cs-artifact-side-card" id="linked-work"><h2 class="cs-section-title">Linked work</h2><div class="cs-list">{"".join(rows[:4])}</div></section>'


def _evidence_bundle_id(record: dict[str, Any]) -> str:
    bundle = record.get("evidence_bundle") if isinstance(record.get("evidence_bundle"), dict) else {}
    return str(bundle.get("evidence_bundle_id") or "")


def _related_brief_id(record: dict[str, Any]) -> str:
    related = record.get("related_brief") if isinstance(record.get("related_brief"), dict) else {}
    return str(related.get("brief_id") or "")


def _record_activity_panel(
    ctx: dict[str, Any],
    record_kind: str,
    record_id: str,
    audit_refs: list[str],
    *,
    product_label: str | None = None,
) -> str:
    events = []
    for event in ctx.get("audit", []):
        subject = event.get("subject") if isinstance(event.get("subject"), dict) else {}
        if str(subject.get("type") or "") == record_kind and str(subject.get("id") or "") == record_id:
            events.append(event)
    def event_label(event: dict[str, Any]) -> str:
        label = _plain_event(str(event.get("event_type") or "recorded"))
        return label.replace("Claim", product_label) if product_label else label

    rows = "".join(
        f'<div class="cs-stat-row"><span class="cs-stat-icon">{icon("history")}</span><div><strong>{h(event_label(event))}</strong><div class="cs-meta">{h(_display_date(event))}</div></div></div>'
        for event in events[:5]
    )
    refs = "".join(f"<li><code>{h(ref)}</code></li>" for ref in audit_refs[:5])
    if not rows:
        rows = '<div class="cs-empty">No record-specific history is visible yet.</div>'
    return f"""
<section class="cs-artifact-side-card" data-record-activity="{h(record_kind)}:{h(record_id)}">
  <h2 class="cs-section-title">History</h2>
  <div class="cs-stat-list">{rows}</div>
  <details class="cs-audit-detail">
    <summary>Audit references</summary>
    <ul>{refs or '<li>No audit reference recorded.</li>'}</ul>
  </details>
</section>
"""


def _brief_related_work(ctx: dict[str, Any], brief: dict[str, Any]) -> str:
    brief_id = str(brief.get("brief_id") or "")
    claims = [
        claim
        for claim in ctx.get("claims", [])
        if _related_brief_id(claim) == brief_id
    ]
    claim_ids = {str(claim.get("claim_id") or "") for claim in claims}
    actions = [action for action in ctx.get("actions", []) if str(action.get("source_claim_id") or "") in claim_ids]
    claim_rows = "".join(
        f'<a class="cs-list-row" href="{h(_detail_href("claims", claim.get("claim_id")))}"><span class="cs-meta">{"Decision draft" if _is_decision_draft(claim) else "Claim candidate"}</span><strong>{h(_claim_title(claim))}</strong></a>'
        for claim in claims[:4]
    ) or '<div class="cs-empty">No Decision draft has been saved from this Brief yet.</div>'
    action_rows = "".join(
        f'<a class="cs-list-row" href="{h(_detail_href("actions", action.get("action_id")))}"><span class="cs-meta">Action preview</span><strong>{h(_action_title(action))}</strong></a>'
        for action in actions[:4]
    )
    if not action_rows:
        action_rows = '<div class="cs-empty">No persisted action preview is linked to this Brief.</div>'
    return f"""
<section class="cs-panel" aria-label="Related Brief work">
  <div class="cs-panel-header"><div><h2>Related decisions and actions</h2><p class="cs-muted">Only persisted records are shown here. Draft suggestions remain in the next-steps section above.</p></div></div>
  <div class="cs-brief-note-grid">
    <div><h3>Decision drafts</h3>{claim_rows}</div>
    <div><h3>Action previews</h3>{action_rows}</div>
  </div>
</section>
"""


def _answer_detail(ctx: dict[str, Any], answer: dict[str, Any]) -> str:
    answer_id = str(answer.get("answer_id") or "")
    question = str(answer.get("question") or "Saved question")
    answer_text = str(answer.get("answer") or "No answer text was saved.")
    label, state = _answer_label(answer)
    source_items = _source_items(ctx, answer)
    source_list = _source_links_from_items(source_items)
    citation_refs = [str(ref) for ref in answer.get("citation_refs", []) if isinstance(ref, str)]
    citation_rows = []
    store = ctx.get("store")
    scope = ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {}
    artifact_by_id = {
        str(record.get("artifact_id") or ""): record
        for record in ctx.get("artifacts", [])
        if isinstance(record, dict)
    }
    for ref in citation_refs:
        if not ref.startswith("evidence_chunk:") or store is None:
            continue
        chunk = store.get_evidence_chunk(ref.split(":", 1)[1])
        if not isinstance(chunk, dict) or chunk.get("scope") != scope:
            continue
        artifact_id = str(chunk.get("artifact_id") or "")
        artifact = artifact_by_id.get(artifact_id)
        source_title = _artifact_title(artifact) if artifact else "Saved source"
        span = _format_span(chunk.get("span"))
        citation_rows.append(
            f"""
<article class="cs-answer-citation">
  <div class="cs-panel-header"><div><strong>{h(source_title)}</strong><p class="cs-meta">Exact supporting excerpt · span {h(span)}</p></div>{_chip("Source excerpt", "searchable")}</div>
  <blockquote>{h(str(chunk.get("text") or ""))}</blockquote>
  {f'<a class="cs-button secondary" href="/artifacts/{h(artifact_id)}">Open full source</a>' if artifact_id else ''}
</article>
"""
        )
    if not citation_rows:
        citation_rows.append(
            '<div class="cs-empty">No exact citation excerpt is available for this saved answer. Review the linked sources before relying on it.</div>'
        )
    audit_refs = [str(ref) for ref in answer.get("audit_refs", []) if isinstance(ref, str)]
    activity = _record_activity_panel(ctx, "conversation_answer", answer_id, audit_refs)
    presented_as_fact = answer.get("presented_as_fact") is True
    return f"""
<section class="cs-brief-workbench cs-answer-history-detail" data-product-surface="answer-history-detail" data-answer-id="{h(answer_id)}">
  <div class="cs-stack">
    <header class="cs-brief-titlebar">
      <nav class="cs-brief-breadcrumb" aria-label="Detail path"><a href="/">Home</a><span aria-hidden="true">/</span><span>Saved answer</span></nav>
      <div class="cs-brief-heading-row"><h1>{h(_truncate(question, 140))}</h1>{_chip(label, state)}</div>
      <p class="cs-muted">This question and answer were saved together. Reopen the supporting excerpts before using the answer in a decision.</p>
      <div class="cs-brief-meta"><span>{h(_display_date(answer))}</span><span>{h(str(len(source_items)))} linked source{"s" if len(source_items) != 1 else ""}</span><span>{h(str(len(citation_refs)))} citation ref{"s" if len(citation_refs) != 1 else ""}</span></div>
    </header>
    <section class="cs-brief-answer-panel" aria-labelledby="saved-answer-title">
      <div class="cs-brief-answer-head"><div><div class="cs-kicker">Question</div><p class="cs-brief-question">{h(question)}</p><h2 id="saved-answer-title">Saved answer</h2></div>{_chip(label, state)}</div>
      <p class="cs-brief-answer-text">{h(answer_text)}</p>
      <p class="cs-meta">Presented as fact: {"Yes — deterministic source checks passed" if presented_as_fact else "No — review or additional evidence is required"}</p>
    </section>
    <section class="cs-panel" aria-labelledby="answer-citations-title">
      <div class="cs-panel-header"><div><h2 id="answer-citations-title">Supporting excerpts</h2><p class="cs-muted">Exact retrieved text used by the saved answer.</p></div></div>
      <div class="cs-stack">{"".join(citation_rows)}</div>
    </section>
  </div>
  <aside class="cs-stack">
    <section class="cs-artifact-side-card"><div class="cs-panel-header"><h2>Sources used</h2>{_chip(str(len(source_items)), "searchable")}</div>{source_list}</section>
    <section class="cs-artifact-side-card"><h2 class="cs-section-title">Continue</h2><a class="cs-button secondary" href="/">Ask another question</a><a class="cs-button secondary" href="/audit?record=conversation_answer:{h(answer_id)}">Open answer history</a><p class="cs-muted">Reopening a saved answer does not regenerate it or change its trust state.</p></section>
    {activity}
  </aside>
</section>
"""


def _brief_detail(ctx: dict[str, Any], brief: dict[str, Any]) -> str:
    label, state = _brief_label(brief)
    summary = str(brief.get("summary") or "")
    key_points = [str(item) for item in brief.get("key_points", []) if isinstance(item, str)]
    findings = [str(item) for item in brief.get("findings", []) if isinstance(item, str)]
    conflicts = [str(item) for item in brief.get("conflicts_risks", []) if isinstance(item, str)]
    if not conflicts:
        conflicts = [str(item) for item in brief.get("contradictions", []) if isinstance(item, str)]
    gaps = [_plain_runtime_text(item) for item in brief.get("gaps", []) if isinstance(item, str)]
    gaps.extend(_plain_runtime_text(item) for item in brief.get("uncertainty", []) if isinstance(item, str))
    gaps = gaps or ["Check the linked sources before treating this as decision-ready."]
    source_items = _source_items(ctx, brief)
    source_list = _source_links_from_items(source_items)
    point_rows = _statement_rows(brief, key_points, source_items)
    finding_rows = _statement_rows(brief, findings, source_items, offset=len(key_points))
    conflict_rows = _statement_rows(brief, conflicts, source_items, offset=len(key_points) + len(findings))
    gap_rows = "".join(f"<li>{h(point)}</li>" for point in gaps[:8])
    next_steps = [_plain_runtime_text(item) for item in brief.get("recommended_next_steps", []) if isinstance(item, str)]
    next_rows = "".join(f"<li>{h(item)}</li>" for item in next_steps[:4]) or "<li>Review the visible sources before requesting review.</li>"
    provenance = _brief_provenance(brief)
    source_count = len(source_items)
    finding_count = len(key_points) + len(findings)
    brief_title = _brief_title(brief)
    label_state = _brief_label_state(brief)[0]
    citation_receipt = _brief_citation_receipt(brief, source_items)
    presented_as_fact = brief.get("presented_as_fact") is True
    if presented_as_fact and label_state == "Fact label earned":
        decision_snapshot = "Source-backed"
    elif label == "Keyword summary":
        decision_snapshot = "Keyword summary only"
    else:
        decision_snapshot = "Draft — source check needed"
    citation_ready = citation_receipt["citation_refs_count"] > 0 and citation_receipt["unresolved_citation_count"] == 0 and citation_receipt["citation_check_refs_count"] > 0
    check_chip = _chip("Citations checked" if citation_ready else "Source check needed", "searchable" if citation_ready else "underReview")
    check_note = (
        "Citation checks and visible source spans agree for the recorded references."
        if citation_ready
        else "Unsupported or unresolved citation work remains; use this as a draft until source spans are checked."
    )
    summary_text = summary or (key_points[0] if key_points else "No summary text was drafted yet. Use the findings and source snippets below before requesting review.")
    decision_question = str(brief.get("decision_question") or brief.get("evidence_bundle", {}).get("query") or "What matters for this decision?")
    load_bearing_rows = brief.get("load_bearing_statements") if isinstance(brief.get("load_bearing_statements"), list) else []
    bottom_line_refs = next(
        (
            [str(ref) for ref in row.get("citation_refs", []) if isinstance(ref, str)]
            for row in load_bearing_rows
            if isinstance(row, dict) and row.get("section") == "bottom_line"
        ),
        [],
    )
    bottom_line_citations = _citation_disclosure_for_refs(
        bottom_line_refs,
        source_items,
        brief,
        initially_open=True,
    )
    brief_id = str(brief.get("brief_id") or "")
    claim_statement = next((value for value in [summary_text, *key_points, *findings] if value.strip()), "")
    can_create_claim = bool(brief_id and _evidence_bundle_id(brief) and claim_statement)
    related_work = _brief_related_work(ctx, brief)
    activity = _record_activity_panel(
        ctx,
        "brief",
        brief_id,
        [str(ref) for ref in brief.get("audit_refs", []) if isinstance(ref, str)],
    )
    return f"""
<section
  class="cs-brief-workbench"
  data-product-surface="brief-detail"
  data-label-state="{h(label_state)}"
  data-presented-as-fact="{str(presented_as_fact).lower()}"
  data-citation-check-refs-count="{h(citation_receipt["citation_check_refs_count"])}"
  data-resolved-citation-count="{h(citation_receipt["resolved_citation_count"])}"
  data-unresolved-citation-count="{h(citation_receipt["unresolved_citation_count"])}"
  data-citation-ref-kind="{h(citation_receipt["citation_ref_kind"])}"
  aria-label="Brief reading workspace"
>
  <div class="cs-stack">
    <header class="cs-brief-titlebar">
      <div class="cs-brief-title">
        <nav class="cs-brief-breadcrumb" aria-label="Detail path">
          <span class="cs-meta">Detail path</span>
          <a href="/briefs">Brief workspace</a>
          <span aria-hidden="true">/</span>
          <span>{h(brief_title)}</span>
        </nav>
        <div class="cs-brief-heading-row">
          <h1>{h(brief_title)}</h1>
          {_chip(label, state)}
        </div>
        <p class="cs-muted">Read the bottom line, source support, conflicts, and missing evidence together before making a decision.</p>
        <div class="cs-brief-meta">
          <span>{h(_display_date(brief))}</span>
          <span>{h(str(source_count))} visible source{"s" if source_count != 1 else ""}</span>
          <span>{h(str(finding_count))} drafted finding{"s" if finding_count != 1 else ""}</span>
        </div>
      </div>
      <div class="cs-brief-actions">
        <a class="cs-button secondary" href="/briefs">Back to briefs</a>
        <a class="cs-button secondary" href="#citation-trail">Review sources</a>
      </div>
    </header>
    <section class="cs-brief-answer-panel" aria-labelledby="brief-answer-title">
      <div class="cs-brief-answer-head">
        <div>
          <div class="cs-kicker">Decision question</div>
          <p class="cs-brief-question">{h(decision_question)}</p>
          <h2 id="brief-answer-title">Bottom line</h2>
        </div>
        {check_chip}
      </div>
      <p class="cs-brief-answer-text">{h(summary_text)}</p>
      <div class="cs-citation-rail" aria-label="Citation disclosure for bottom line">{bottom_line_citations or _chip("Needs source check", "underReview")}</div>
      <div class="cs-brief-answer-meta" aria-label="Brief status">
        <span><strong>{h(decision_snapshot)}</strong> decision state</span>
        <span><strong>{h(source_count)}</strong> visible source{"s" if source_count != 1 else ""}</span>
        <span><strong>{h(finding_count)}</strong> drafted finding{"s" if finding_count != 1 else ""}</span>
      </div>
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>Key facts</h2>
          <p class="cs-muted">Each load-bearing statement stays close to the source that supports it.</p>
        </div>
        {_chip("Source coverage", "searchable")}
      </div>
      {point_rows}
      {f'<h2 class="cs-section-title">More findings</h2>{finding_rows}' if findings else ''}
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header"><div><h2>Conflicts and risks</h2><p class="cs-muted">Differences, dependencies, and unresolved risk that could change the decision.</p></div>{_chip("Decision risk", "underReview")}</div>
      {conflict_rows if conflicts else '<div class="cs-empty">No conflict or risk was identified in the selected sources.</div>'}
    </section>
    <div class="cs-brief-note-grid">
      <section class="cs-panel">
        <div class="cs-panel-header"><h2>Missing evidence</h2>{_chip("Needs source check", "underReview")}</div>
        <ul class="cs-brief-note-list">{gap_rows}</ul>
      </section>
      <section class="cs-panel">
        <div class="cs-panel-header"><h2>Recommended next step</h2>{_chip("Recommendation", "draft")}</div>
        <ul class="cs-brief-note-list">{next_rows}</ul>
      </section>
    </div>
    {related_work}
  </div>
  <aside class="cs-stack">
    <section class="cs-artifact-side-card" id="citation-trail">
      <div class="cs-panel-header"><h2>Sources used</h2>{_chip(str(source_count), "searchable")}</div>
      <p class="cs-muted">Open the exact source before promoting a finding.</p>
      {source_list}
    </section>
    <details class="cs-artifact-side-card cs-citation-checks">
      <summary><strong>Citation checks</strong></summary>
      <p>{h(check_note)}</p>
      <dl class="cs-detail-grid">
        <dt>Resolved</dt><dd>{h(citation_receipt["resolved_citation_count"])} / {h(citation_receipt["citation_refs_count"])}</dd>
        <dt>Check records</dt><dd>{h(citation_receipt["citation_check_refs_count"])}</dd>
        <dt>Reference kind</dt><dd>{h(citation_receipt["citation_ref_kind"])}</dd>
        <dt>Label state</dt><dd>{h(label_state)}</dd>
      </dl>
    </details>
    <details class="cs-artifact-side-card">
      <summary><strong>Provenance</strong></summary>
      {provenance}
    </details>
    <section class="cs-artifact-side-card">
      <h2 class="cs-section-title">Continue to a decision</h2>
      <div class="cs-review-box">
        {f'<button class="cs-button" type="button" id="cs-create-claim-button" data-brief-id="{h(brief_id)}" data-statement="{h(claim_statement)}">Save as Decision draft</button>' if can_create_claim else '<p class="cs-muted">A Decision draft can be saved after this Brief has a source-linked finding.</p>'}
        <a class="cs-button secondary" href="/audit?record=brief:{h(brief_id)}">Open Brief history</a>
      </div>
      <div id="cs-claim-create-status" class="cs-status is-idle" data-state="idle" role="status" aria-live="polite" aria-atomic="true" hidden>No Decision draft saved yet.</div>
      <p class="cs-muted">A Decision draft preserves the source links. It does not approve shared truth or authorize an action.</p>
    </section>
    {activity}
  </aside>
</section>
"""


def _claim_detail(ctx: dict[str, Any], claim: dict[str, Any]) -> str:
    label, state = _claim_label(claim)
    decision_draft = _is_decision_draft(claim)
    product_label = "Decision draft" if decision_draft else "Claim"
    product_collection_label = "Decisions" if decision_draft else "Claims"
    source_items = _source_items(ctx, claim)
    source_list = _source_links_from_items(source_items)
    authority = claim.get("authority") if isinstance(claim.get("authority"), dict) else {}
    evidence_integrity = claim.get("evidence_integrity") if isinstance(claim.get("evidence_integrity"), dict) else {}
    integrity_failed = evidence_integrity.get("status") == "failed"
    has_sources = bool(source_items)
    is_approved = str(claim.get("status") or "").lower() == "approved"
    evidence_backed_earned = _claim_evidence_backed_earned(claim)
    approval_eligible = not is_approved and authority.get("can_be_approved") is True
    rationale = _plain_runtime_text(claim.get("rationale") or "").strip() or f"No separate rationale was recorded for this {product_label}."
    claim_title = _claim_title(claim)
    claim_statement = str(claim.get("statement") or claim_title)
    source_label = f"{len(source_items)} source{'s' if len(source_items) != 1 else ''}"
    claim_id = str(claim.get("claim_id") or "")
    related_brief_id = _related_brief_id(claim)
    related_brief = (
        f'<a class="cs-button secondary" href="{h(_detail_href("briefs", related_brief_id))}">Open source Brief</a>'
        if related_brief_id
        else '<span class="cs-muted">No source Brief lineage is recorded.</span>'
    )
    gaps = [_plain_runtime_text(value) for value in claim.get("gaps", []) if isinstance(value, str)]
    gap_rows = "".join(f"<li>{h(value)}</li>" for value in gaps[:6]) or "<li>No separate gaps were recorded. Inspect the source support before approval.</li>"
    denial_events = []
    for event in ctx.get("audit", []):
        subject = event.get("subject") if isinstance(event.get("subject"), dict) else {}
        if str(event.get("event_type") or "") == "claim.approval.denied" and str(subject.get("id") or "") == claim_id:
            denial_events.append(event)
    denial_panel = ""
    if denial_events and not decision_draft:
        denied = denial_events[0]
        details = denied.get("details") if isinstance(denied.get("details"), dict) else {}
        audit_ref = f"audit:{denied.get('event_id')}" if denied.get("event_id") else "Audit reference unavailable"
        semantic_denial = str(details.get("reason") or "") == "semantic_support_review_required"
        denial_chip = "Semantic review required" if semantic_denial else "Evidence required"
        recovery = (
            "Review the exact statement against every cited span. Semantic support and owner approval remain separate decisions."
            if semantic_denial
            else "Return to the source Brief and create a Claim from a citation-checked finding."
        )
        denial_panel = f"""
<section class="cs-panel" data-claim-approval-denial="true">
  <div class="cs-panel-header"><div><h2>Approval blocked</h2><p class="cs-muted">The Claim remains a draft.</p></div>{_chip(denial_chip, "insufficientEvidence")}</div>
  <p><strong>Cause:</strong> {h(_plain_runtime_text(details.get("reason") or "Supporting source evidence is missing."))}</p>
  <p><strong>Recovery:</strong> {h(recovery)}</p>
  <details class="cs-audit-detail"><summary>Denial detail</summary><p><code>{h(audit_ref)}</code></p><p>{h(_plain_runtime_text(details.get("required") or "Supporting evidence is required before approval."))}</p></details>
</section>
"""
    activity = _record_activity_panel(
        ctx,
        "claim",
        claim_id,
        [str(ref) for ref in claim.get("audit_refs", []) if isinstance(ref, str)],
        product_label="Decision draft" if decision_draft else None,
    )
    approved_at = _display_date({"created_at": claim.get("approved_at")}) if claim.get("approved_at") else "Approval time not recorded"
    blocked_reason = _plain_runtime_text(
        authority.get("blocked_reason")
        or "The exact Claim statement needs current citation-checked source support before approval."
    )
    if decision_draft:
        approval_panel = f"""
<section class="cs-panel cs-decision-panel" data-decision-draft-boundary="true">
  <div class="cs-panel-header"><div><h2>Decision draft saved</h2><p class="cs-muted">This sourced finding is preserved for review. It grants no approval, shared-truth, or action authority.</p></div>{_chip("Draft", "draft")}</div>
  <div class="cs-review-box">{related_brief}</div>
</section>
"""
    elif is_approved:
        approval_panel = f"""
<section class="cs-panel cs-decision-panel" data-claim-approval-state="approved">
  <div class="cs-panel-header"><div><h2>Approval recorded</h2><p class="cs-muted">Recorded {h(approved_at)}. This does not authorize an external action.</p></div>{_chip("Approved", "approved")}</div>
  <a class="cs-button secondary" href="/audit?record=claim:{h(claim_id)}">Open Claim history</a>
</section>
"""
    elif approval_eligible:
        approval_panel = f"""
<section class="cs-panel cs-decision-panel" data-claim-approval-state="eligible">
  <div class="cs-panel-header"><div><h2>Ready for an owner decision</h2><p class="cs-muted">Current statement-level source support allows approval. Approval records shared truth; it does not start an action.</p></div>{_chip("Ready", "underReview")}</div>
  <button class="cs-button" type="button" id="cs-approve-claim-button" data-claim-id="{h(claim_id)}">Approve Claim</button>
  <div class="cs-status is-idle" id="cs-claim-approval-status" data-state="idle" role="status" aria-live="polite" aria-atomic="true" hidden></div>
  <dialog class="cs-confirm-dialog" id="cs-claim-approval-dialog" aria-labelledby="cs-claim-approval-title">
    <form method="dialog">
      <div class="cs-confirm-dialog-copy">
        <div class="cs-kicker">Confirm decision</div>
        <h2 id="cs-claim-approval-title">Approve this Claim?</h2>
        <p>{h(_truncate(claim_statement, 220))}</p>
        <p class="cs-muted">This records approval for shared truth. It does not execute or authorize an external action.</p>
      </div>
      <div class="cs-confirm-dialog-actions">
        <button class="cs-button secondary" value="cancel">Cancel</button>
        <button class="cs-button" value="confirm" id="cs-confirm-claim-approval">Approve Claim</button>
      </div>
    </form>
  </dialog>
</section>
"""
    else:
        approval_panel = f"""
<section class="cs-panel cs-decision-panel" data-claim-approval-state="blocked">
  <div class="cs-panel-header"><div><h2>Approval is not available</h2><p class="cs-muted">{h(blocked_reason)}</p></div>{_chip("Evidence integrity failed" if integrity_failed else "Semantic review required" if has_sources else "Source support needed", "failed" if integrity_failed else "insufficientEvidence")}</div>
  <div class="cs-review-box">{related_brief}</div>
</section>
"""
    return f"""
<section
  class="cs-grid-two cs-claim-workbench"
  data-product-surface="claim-detail"
  data-product-role="{h(str(claim.get('product_role') or 'claim_draft'))}"
  data-source-support-attached="{str(has_sources).lower()}"
  data-evidence-backed-earned="{str(evidence_backed_earned).lower()}"
  data-approval-eligible="{str(approval_eligible).lower()}"
>
  <div class="cs-stack">
    <header class="cs-claim-hero is-compact">
      <div class="cs-claim-titlebar">
        <div class="cs-brief-title">
          <nav class="cs-claim-breadcrumb" aria-label="Detail path">
            <span class="cs-meta">Detail path</span>
            <a href="/claims">{h(product_collection_label)}</a>
            <span aria-hidden="true">/</span>
            <span>{h(_truncate(claim_title, 90))}</span>
          </nav>
          <div class="cs-claim-heading-row">
            <h1>{h(claim_title)}</h1>
            {_chip(label, state)}
          </div>
          <div class="cs-brief-meta">
            <span>Created {h(_display_date(claim))}</span>
            <span>{h(source_label)} attached</span>
          </div>
        </div>
        <div class="cs-claim-actions" aria-label="{h(product_label)} actions">
          <a class="cs-button secondary" href="/claims">Back to {h(product_collection_label)}</a>
          <a class="cs-button secondary" href="/audit?record=claim:{h(claim_id)}">Open history</a>
        </div>
      </div>
    </header>
    <section class="cs-panel cs-claim-statement">
      <div class="cs-kicker">Decision statement</div>
      <h2>{h(claim_statement)}</h2>
      <div class="cs-claim-rationale">
        <span class="cs-meta">Rationale</span>
        <p>{h(rationale)}</p>
      </div>
    </section>
    {approval_panel}
    {denial_panel}
    <section class="cs-panel">
      <div class="cs-panel-header"><div><h2>Brief lineage and gaps</h2><p class="cs-muted">The source Brief stays separate from this {h(product_label)} record.</p></div></div>
      <div class="cs-review-box">{related_brief}</div>
      <ul class="cs-brief-note-list">{gap_rows}</ul>
    </section>
  </div>
  <aside class="cs-stack">
    <section class="cs-panel flat">
      <div class="cs-panel-header">
        <div>
          <h2>Supporting evidence</h2>
          <p class="cs-muted">These links open the visible local sources attached to this {h(product_label)}.</p>
        </div>
        {_chip(str(len(source_items)), "searchable")}
      </div>
      {source_list}
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">{h("Decision boundary" if decision_draft else "Authority")}</h2>
      <p class="cs-muted">{h("This Decision draft preserves a source-linked finding without granting approval or shared truth." if decision_draft else blocked_reason)}</p>
      <p class="cs-muted">{h("A Decision draft cannot authorize or start an external action." if decision_draft else "A Claim never starts an external action from this page.")}</p>
    </section>
    {activity}
  </aside>
</section>
"""


def _memory_detail(ctx: dict[str, Any], memory: dict[str, Any]) -> str:
    memory_id = str(memory.get("memory_id") or "")
    statement = str(memory.get("title") or memory.get("statement") or "Knowledge draft")
    status = str(memory.get("status") or "draft").lower()
    trust_state = str(memory.get("trust_state") or "draft").lower()
    canonicality = memory.get("canonicality") if isinstance(memory.get("canonicality"), dict) else {}
    usage = memory.get("usage_permissions") if isinstance(memory.get("usage_permissions"), dict) else {}
    freshness = memory.get("freshness") if isinstance(memory.get("freshness"), dict) else {}
    source = memory.get("source") if isinstance(memory.get("source"), dict) else {}
    scope = memory.get("scope") if isinstance(memory.get("scope"), dict) else {}
    owner_approved = status == "owner_approved" and canonicality.get("owner_approved") is True
    label = "Owner approved" if owner_approved else "Draft / Needs review"
    state = "approved" if owner_approved else "underReview"
    source_items = _source_items(ctx, memory)
    source_list = _source_links_from_items(source_items)
    evidence_refs = [str(ref) for ref in memory.get("evidence_refs", []) if isinstance(ref, str)]
    audit_refs = [str(ref) for ref in memory.get("audit_refs", []) if isinstance(ref, str)]
    influence_answers = usage.get("can_influence_answers") is True
    influence_actions = usage.get("can_influence_actions") is True
    influence_routing = usage.get("can_influence_routing") is True
    freshness_status = str(freshness.get("status") or "not reviewed").replace("_", " ").title()
    reviewed_at = str(freshness.get("last_reviewed_at") or "Not reviewed")
    source_bundle = str(source.get("evidence_bundle_id") or "Not recorded")
    audit = _record_activity_panel(ctx, "memory", memory_id, audit_refs)
    return f"""
<section
  class="cs-grid-two cs-memory-workbench"
  data-product-surface="memory-detail"
  data-owner-approved="{str(owner_approved).lower()}"
  data-can-influence-answers="{str(influence_answers).lower()}"
  data-can-influence-actions="{str(influence_actions).lower()}"
  data-can-influence-routing="{str(influence_routing).lower()}"
>
  <div class="cs-stack">
    <header class="cs-brief-titlebar">
      <div class="cs-brief-title">
        <nav class="cs-brief-breadcrumb" aria-label="Detail path">
          <span class="cs-meta">Detail path</span>
          <a href="/inbox">Ops Inbox</a>
          <span aria-hidden="true">/</span>
          <span>Memory candidate</span>
        </nav>
        <div class="cs-brief-heading-row"><h1>{h(_truncate(statement, 120))}</h1>{_chip(label, state)}</div>
        <p class="cs-muted">This record stays inspectable as a source-linked review item. Draft state does not grant answer, routing, or action authority.</p>
      </div>
      <div class="cs-brief-actions">
        <a class="cs-button secondary" href="/inbox">Back to inbox</a>
        <a class="cs-button secondary" href="/audit">Open History</a>
      </div>
    </header>
    <section class="cs-panel">
      <div class="cs-panel-header"><div><h2>Knowledge statement</h2><p class="cs-muted">Review the statement beside the evidence that produced it.</p></div>{_chip(label, state)}</div>
      <p>{h(statement)}</p>
      <dl class="cs-detail-grid">
        <dt>Lifecycle</dt><dd>{h(status.replace("_", " ").title())}</dd>
        <dt>Trust state</dt><dd>{h(trust_state.replace("_", " ").title())}</dd>
        <dt>Freshness</dt><dd>{h(freshness_status)}</dd>
        <dt>Last reviewed</dt><dd>{h(reviewed_at)}</dd>
        <dt>Owner</dt><dd>{h(str(scope.get("owner_id") or "local-user"))}</dd>
        <dt>Workspace</dt><dd>{h(str(scope.get("workspace_id") or "default"))}</dd>
      </dl>
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header"><div><h2>Authority boundary</h2><p class="cs-muted">Permissions are explicit; draft candidates are not hidden durable truth.</p></div>{_chip("Review only" if not owner_approved else "Owner approved", state)}</div>
      <dl class="cs-detail-grid">
        <dt>Influence answers</dt><dd>{"Allowed" if influence_answers else "Not allowed"}</dd>
        <dt>Influence routing</dt><dd>{"Allowed" if influence_routing else "Not allowed"}</dd>
        <dt>Influence actions</dt><dd>{"Allowed" if influence_actions else "Not allowed"}</dd>
        <dt>Canonical owner approval</dt><dd>{"Recorded" if owner_approved else "Not recorded"}</dd>
      </dl>
      <p class="cs-muted">Saving this draft as approved knowledge is intentionally unavailable in this active review slice.</p>
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header"><div><h2>Source and evidence</h2><p class="cs-muted">The candidate remains secondary to its saved sources.</p></div>{_chip(str(len(source_items)), "searchable")}</div>
      {source_list}
      <details class="cs-audit-detail"><summary>Supporting evidence details</summary><ul>{''.join(f'<li><code>{h(ref)}</code></li>' for ref in evidence_refs) or '<li>No evidence ref recorded.</li>'}</ul><p class="cs-meta">Supporting evidence record: {h(source_bundle)}</p></details>
    </section>
  </div>
  <aside class="cs-stack">
    <section class="cs-artifact-side-card">
      <h2 class="cs-section-title">Review controls</h2>
      <div class="cs-review-box">
        <a class="cs-button" href="/search?q={quote(statement)}">Review matching sources</a>
        <a class="cs-button secondary" href="/audit">Open History</a>
        <a class="cs-button secondary" href="/inbox">Continue in Ops Inbox</a>
      </div>
    </section>
    {audit}
  </aside>
</section>
"""


def _action_detail(ctx: dict[str, Any], action: dict[str, Any]) -> str:
    label, state = _action_label(action)
    lifecycle = _action_lifecycle(action)
    stage = str(lifecycle["stage"])
    approval = lifecycle["approval"]
    approval_status = str(lifecycle["approval_status"])
    approval_recorded = approval_status == "approved"
    execution = lifecycle["execution"]
    execution_result = lifecycle["result"]
    dry_run = action.get("dry_run") if isinstance(action.get("dry_run"), dict) else {}
    diff = dry_run.get("diff") if isinstance(dry_run.get("diff"), dict) else {}
    impact = dry_run.get("expected_impact") if isinstance(dry_run.get("expected_impact"), dict) else {}
    policy = action.get("policy_decision") if isinstance(action.get("policy_decision"), dict) else {}
    if not policy and isinstance(dry_run.get("policy_decision"), dict):
        policy = dry_run["policy_decision"]
    policy_decision_raw = str(policy.get("decision") or "").lower()
    policy_requires_approval = policy.get("approval_required") is True or policy_decision_raw in {
        "requires_approval",
        "require_approval",
        "escalate",
    }
    approval_record_requires = approval.get("required") is True
    approval_required = approval_record_requires or policy_requires_approval
    approval_allowed = (
        approval_record_requires
        and not approval_recorded
        and stage in {"draft", "pending"}
        and policy_decision_raw not in {"deny", "denied", "blocked", "policy_blocked", "escalate", "escalated"}
    )
    connector = action.get("connector_boundary") if isinstance(action.get("connector_boundary"), dict) else {}
    source_items = _source_items(ctx, action)
    source_list = _source_links_from_items(source_items)
    action_id = str(action.get("action_id") or "")
    action_title = _action_title(action)
    goal = _plain_runtime_text(dry_run.get("goal") or action.get("goal") or action_title)
    target = _plain_runtime_text(impact.get("target") or dry_run.get("target") or "Target not recorded")
    risk_label = str(impact.get("risk") or action.get("risk") or "review").replace("_", " ").title()
    decision_label = _plain_policy_decision(str(policy.get("decision") or "Not recorded"))
    policy_reason = _plain_runtime_text(policy.get("reason") or "No separate policy reason is recorded.")
    workspace_mode = str(policy.get("workspace_mode") or "local").replace("_", " ").title()
    execution_mode = (
        "Local / Mock result"
        if connector.get("mocked") is True and stage in {"executed", "failed"}
        else "Local / Mock / Draft"
        if connector.get("mocked") is True
        else f"{workspace_mode} / Governed"
    )
    planned_external_calls = int(impact.get("real_external_http_calls", 0) or 0)
    expected_connector_calls = int(impact.get("expected_connector_calls", 0) or 0)
    raw_observed_calls = execution_result.get("external_http_calls")
    try:
        observed_external_calls = (
            int(raw_observed_calls)
            if raw_observed_calls is not None and not isinstance(raw_observed_calls, bool)
            else None
        )
    except (TypeError, ValueError):
        observed_external_calls = None
    if observed_external_calls is not None and observed_external_calls < 0:
        observed_external_calls = None
    has_recorded_outcome = stage in {"executed", "failed", "blocked"}
    displayed_external_calls: int | str = (
        observed_external_calls
        if has_recorded_outcome and observed_external_calls is not None
        else "not-recorded"
        if has_recorded_outcome
        else planned_external_calls
    )
    source_claim_id = str(action.get("source_claim_id") or "")
    source_claim_link = (
        f'<a class="cs-button secondary" href="{h(_detail_href("claims", source_claim_id))}">Open supporting Claim</a>'
        if source_claim_id
        else '<p class="cs-muted">No supporting Claim lineage is recorded.</p>'
    )
    before = _plain_runtime_text(diff.get("before") or "No prior state was recorded.")
    after = _plain_runtime_text(diff.get("after") or "No proposed state was recorded.")
    result_message = _plain_runtime_text(
        execution_result.get("message")
        or execution.get("message")
        or "No execution message was recorded."
    )
    recovery_value = execution_result.get("recovery_path") or execution.get("recovery_path")
    if isinstance(recovery_value, list):
        recovery_value = " ".join(str(item) for item in recovery_value if isinstance(item, str))
    recovery = _plain_runtime_text(
        recovery_value
        or "Review the source, target, policy, and approval state before creating a new preview."
    )
    denial_events = []
    for event in ctx.get("audit", []):
        subject = event.get("subject") if isinstance(event.get("subject"), dict) else {}
        if str(event.get("event_type") or "") == "action.execution.denied" and str(subject.get("id") or "") == action_id:
            denial_events.append(event)
    denial_panel = ""
    if denial_events:
        denial = denial_events[0]
        denial_details = denial.get("details") if isinstance(denial.get("details"), dict) else {}
        resolution_value = denial_details.get("resolution_path")
        if isinstance(resolution_value, list):
            resolution_value = " ".join(str(item) for item in resolution_value if isinstance(item, str))
        denial_panel = f"""
<section class="cs-panel cs-action-outcome" data-action-execution-denial="true">
  <div class="cs-panel-header"><div><div class="cs-kicker">Safety boundary</div><h2>Execution blocked</h2></div>{_chip("Policy blocked", "policyBlocked")}</div>
  <p><strong>Cause:</strong> {h(_plain_runtime_text(denial_details.get("reason") or "The safety gate denied execution."))}</p>
  <p><strong>Recovery:</strong> {h(_plain_runtime_text(resolution_value or "Review approval, source evidence, and the local safety boundary before retrying."))}</p>
  <details class="cs-audit-detail">
    <summary>Technical denial detail</summary>
    <dl class="cs-detail-grid">
      <dt>Reason code</dt><dd>{h(str(denial_details.get("reason_code") or "Not recorded"))}</dd>
      <dt>Safety envelope</dt><dd>{h(str(denial_details.get("action_safety_envelope_id") or "Not recorded"))}</dd>
      <dt>External HTTP calls</dt><dd>{h(str(denial_details.get("external_http_calls", 0)))}</dd>
    </dl>
  </details>
</section>
"""
    if stage == "executed":
        lifecycle_panel = f"""
<section class="cs-panel cs-action-outcome" data-action-execution-result="true">
  <div class="cs-panel-header"><div><div class="cs-kicker">Recorded outcome</div><h2>Execution result</h2></div>{_chip(label, state)}</div>
  <p>{h(result_message)}</p>
  <dl class="cs-detail-grid">
    <dt>Execution state</dt><dd>{h(str(lifecycle["execution_status"]).replace("_", " ").title())}</dd>
    <dt>Result</dt><dd>{h(str(execution_result.get("status") or "recorded").replace("_", " ").title())}</dd>
    <dt>External HTTP calls</dt><dd>{h(str(displayed_external_calls))}</dd>
  </dl>
</section>
"""
    elif stage == "failed":
        lifecycle_panel = f"""
<section class="cs-panel cs-action-outcome" data-action-failure-recovery="true" data-product-state="failed-with-recovery">
  <div class="cs-panel-header"><div><div class="cs-kicker">Recorded outcome</div><h2>Action failed</h2></div>{_chip("Failed with recovery", "failed")}</div>
  <p><strong>Cause:</strong> {h(result_message)}</p>
  <p><strong>Recovery:</strong> {h(recovery)}</p>
  <dl class="cs-detail-grid"><dt>External HTTP calls</dt><dd>{h(str(displayed_external_calls))}</dd></dl>
</section>
"""
    elif stage == "blocked":
        execution_status_reason = ""
        if lifecycle["execution_status"] not in {"", "not_started"}:
            execution_status_reason = str(lifecycle["execution_status"]).replace("_", " ").capitalize()
        block_reason = _plain_runtime_text(
            policy.get("reason")
            or execution.get("reason")
            or execution.get("message")
            or execution_status_reason
            or "The recorded policy state blocks this action."
        )
        lifecycle_panel = f"""
<section class="cs-panel cs-action-outcome" data-action-policy-blocked="true" data-product-state="policy-blocked">
  <div class="cs-panel-header"><div><div class="cs-kicker">Recorded outcome</div><h2>Action blocked</h2></div>{_chip("Policy blocked", "policyBlocked")}</div>
  <p><strong>Cause:</strong> {h(block_reason)}</p>
  <p><strong>Recovery:</strong> {h(recovery)}</p>
  <dl class="cs-detail-grid"><dt>External HTTP calls</dt><dd>{h(str(displayed_external_calls))}</dd></dl>
</section>
"""
    else:
        lifecycle_panel = f"""
<section class="cs-panel cs-action-preview" data-action-preview="true">
  <div class="cs-panel-header"><div><div class="cs-kicker">Proposed change</div><h2>{h(goal)}</h2><p class="cs-muted">This is a dry-run preview. Nothing is executed from this page.</p></div>{_chip(label, state)}</div>
  <div class="cs-action-change-grid">
    <div class="cs-action-change"><span class="cs-meta">Before</span><p>{h(before)}</p></div>
    <div class="cs-action-change"><span class="cs-meta">After</span><p>{h(after)}</p></div>
  </div>
  <dl class="cs-detail-grid">
    <dt>Target</dt><dd>{h(target)}</dd>
    <dt>Risk</dt><dd>{h(risk_label)}</dd>
    <dt>Policy</dt><dd>{h(decision_label)}</dd>
  </dl>
</section>
"""
    scope = ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {}
    owner = str(scope.get("owner_id") or "local-user")
    if approval_recorded:
        approved_at = str(approval.get("approved_at") or "Approval time not recorded")
        approver = str(approval.get("approver") or "Approver not recorded")
        approval_panel = f"""
<section class="cs-panel cs-decision-panel" data-action-approval-state="approved">
  <div class="cs-panel-header"><div><h2>Approval recorded</h2><p class="cs-muted">Approved by {h(approver)} at {h(approved_at)}. Execution remains a separate step.</p></div>{_chip("Approved", "approved")}</div>
</section>
"""
    elif approval_allowed:
        approval_panel = f"""
<section class="cs-panel cs-decision-panel" data-action-approval-state="eligible">
  <div class="cs-panel-header"><div><h2>Approval required</h2><p class="cs-muted">Review the target, risk, source, and policy before recording approval.</p></div>{_chip("Needs approval", "underReview")}</div>
  <button class="cs-button" type="button" id="cs-approve-action-button" data-action-id="{h(action_id)}" data-approver="{h(owner)}">Approve action</button>
  <div class="cs-status is-idle" id="cs-action-approval-status" data-state="idle" role="status" aria-live="polite" aria-atomic="true" hidden></div>
  <dialog class="cs-confirm-dialog" id="cs-action-approval-dialog" aria-labelledby="cs-action-approval-title">
    <form method="dialog">
      <div class="cs-confirm-dialog-copy">
        <div class="cs-kicker">Confirm approval</div>
        <h2 id="cs-action-approval-title">Approve this action preview?</h2>
        <p><strong>Target:</strong> {h(target)}</p>
        <p><strong>Risk:</strong> {h(risk_label)}</p>
        <p class="cs-muted">This records approval only. It does not execute the action.</p>
      </div>
      <div class="cs-confirm-dialog-actions">
        <button class="cs-button secondary" value="cancel">Cancel</button>
        <button class="cs-button" value="confirm" id="cs-confirm-action-approval">Approve action</button>
      </div>
    </form>
  </dialog>
</section>
"""
    else:
        reason = (
            "Resolve the recorded blocked or failed state before creating a new action preview."
            if stage in {"blocked", "failed"}
            else "Policy requires approval, but this record has no matching approval contract. Create a fresh preview before proceeding."
            if approval_required and not approval_record_requires
            else "This action does not require a separate approval record."
            if not approval_required
            else "Approval is not available for this lifecycle state."
        )
        approval_panel = f"""
<section class="cs-panel cs-decision-panel" data-action-approval-state="unavailable">
  <div class="cs-panel-header"><div><h2>Approval</h2><p class="cs-muted">{h(reason)}</p></div>{_chip("Not available", "draft")}</div>
</section>
"""
    activity = _record_activity_panel(
        ctx,
        "action",
        action_id,
        [str(action.get("audit_ref"))] if action.get("audit_ref") else [],
    )
    return f"""
<section
  class="cs-grid-two cs-action-workbench"
  data-product-surface="action-detail"
  data-approval-required="{str(approval_required).lower()}"
  data-approval-eligible="{str(approval_allowed).lower()}"
  data-execution-mode="{h(execution_mode)}"
  data-real-external-http-calls="{h(displayed_external_calls)}"
  data-expected-connector-calls="{h(expected_connector_calls)}"
  data-product-state="{h(stage)}"
>
  <div class="cs-stack">
    <header class="cs-action-hero">
      <nav class="cs-action-breadcrumb" aria-label="Detail path">
        <span class="cs-meta">Detail path</span>
        <a href="/actions">Actions</a>
        <span aria-hidden="true">/</span>
        <span>{h(_truncate(action_title, 80))}</span>
      </nav>
      <div class="cs-action-titlebar">
        <div>
          <div class="cs-kicker">{"Action result" if stage == "executed" else "Failed action" if stage == "failed" else "Blocked action" if stage == "blocked" else "Action preview"}</div>
          <h1>{h(action_title)}</h1>
          <div class="cs-brief-meta">
            <span>{h(_display_date(action))}</span>
            <span>{h(risk_label)} risk</span>
            <span>{h(label)}</span>
          </div>
        </div>
        <div class="cs-action-actions">
          <a class="cs-button secondary" href="/actions">Back to Actions</a>
          <a class="cs-button secondary" href="/audit?record=action:{h(action_id)}">Open history</a>
        </div>
      </div>
    </header>
    {lifecycle_panel}
    {denial_panel}
    {approval_panel}
    <section class="cs-panel">
      <div class="cs-panel-header"><div><h2>Why this action</h2><p class="cs-muted">The proposal stays linked to the Claim and evidence that justify reviewing it.</p></div></div>
      <p>{h(policy_reason)}</p>
      <div class="cs-review-box">{source_claim_link}</div>
    </section>
  </div>
  <aside class="cs-stack cs-action-rail">
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Sources</h2>
      {source_list}
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Policy and boundary</h2>
      <dl class="cs-detail-grid">
        <dt>Decision</dt><dd>{h(decision_label)}</dd>
        <dt>Workspace mode</dt><dd>{h(workspace_mode)}</dd>
        <dt>Planned connector calls</dt><dd>{h(str(expected_connector_calls))}</dd>
        <dt>Planned external calls</dt><dd>{h(str(planned_external_calls))}</dd>
      </dl>
      <details class="cs-audit-detail">
        <summary>Technical boundary</summary>
        <dl class="cs-detail-grid">
          <dt>Connector mediated</dt><dd>{"Yes" if connector.get("direct_provider_access") is False else "Needs review"}</dd>
          <dt>Credentials exposed</dt><dd>{"No" if connector.get("credentials_exposed_to_agent") is False else "Needs review"}</dd>
          <dt>Mocked</dt><dd>{"Yes" if connector.get("mocked") is True else "No"}</dd>
          <dt>Observed external calls</dt><dd>{h(str(displayed_external_calls))}</dd>
        </dl>
      </details>
    </section>
    {activity}
  </aside>
</section>
"""


def _claim_trust_ladder(has_sources: bool, semantic_support_verified: bool, is_approved: bool) -> str:
    source_class = "is-active" if has_sources else "is-locked"
    semantic_class = "is-active" if semantic_support_verified else "is-locked"
    approved_class = "is-active" if is_approved else "is-locked"
    source_note = "Supporting source is attached." if has_sources else "Attach at least one source."
    semantic_note = "Statement-level semantic support is recorded." if semantic_support_verified else "Required before owner approval."
    approved_note = "Owner approval recorded." if is_approved else "Requires semantic review first."
    return f"""
<div class="cs-trust-ladder" aria-label="Claim trust ladder">
  <div class="cs-trust-step is-active">
    <strong>Draft</strong>
    <span class="cs-meta">Editable statement.</span>
  </div>
  <div class="cs-trust-step {source_class}">
    <strong>Source support</strong>
    <span class="cs-meta">{h(source_note)}</span>
  </div>
  <div class="cs-trust-step {semantic_class}">
    <strong>Semantic review</strong>
    <span class="cs-meta">{h(semantic_note)}</span>
  </div>
  <div class="cs-trust-step {approved_class}">
    <strong>Approved</strong>
    <span class="cs-meta">{h(approved_note)}</span>
  </div>
</div>
"""


def _source_links_from_items(items: list[dict[str, str]]) -> str:
    if not items:
        return '<div class="cs-empty">No linked source is visible in this workspace.</div>'
    rows = "".join(_source_card(item) for item in items[:8])
    return f'<div class="cs-list">{rows}</div>'


def _source_card(item: dict[str, str]) -> str:
    return f"""
<details class="cs-list-row cs-source-card">
  <summary>
    <span class="cs-meta">{h(item["label"])}</span>
    <span>{h(item["title"])}</span>
  </summary>
  <p><strong>Source snippet:</strong> {h(_truncate(item["snippet"], 260))}</p>
  <div class="cs-row">
    <a class="cs-button secondary" href="{h(item["href"])}">Inspect source</a>
    <a class="cs-button ghost" href="/audit">History</a>
    {_chip(item.get("state_label") or "Saved", item.get("state") or "saved")}
  </div>
  <div class="cs-provenance" aria-label="Full provenance">
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
    citation_rows = record.get("load_bearing_statements")
    if not isinstance(citation_rows, list):
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
        citation_cards = _citation_disclosure_for_refs(refs, source_items, record)
        resolved_count = sum(1 for ref in refs if _citation_item_for_ref(ref, source_items, record))
        unresolved_count = max(0, len(refs) - resolved_count)
        if refs and unresolved_count == 0:
            source_state = "searchable"
            source_label = "Source span linked"
        elif refs and resolved_count:
            source_state = "underReview"
            source_label = "Partial source support"
        elif refs:
            source_state = "underReview"
            source_label = "Unresolved source ref"
        else:
            source_state = "underReview"
            source_label = "Needs source check"
        rows.append(
            f"""
<li class="cs-finding">
  <div class="cs-finding-head">
    <span class="cs-finding-index">Finding {h(index + offset)}</span>
    {_chip(source_label, source_state)}
  </div>
  <div>{h(statement)}</div>
  <div class="cs-citation-rail" aria-label="Citation disclosure for finding {index + offset}">{citation_cards or _chip("Needs source check", "underReview")}</div>
</li>
"""
        )
    return f'<ol class="cs-finding-list">{"".join(rows)}</ol>'


def _citation_ref_kind(ref: str) -> str:
    if ref.startswith("evidence_chunk:"):
        return "evidence chunk"
    if ref.startswith("artifact:"):
        return "artifact"
    if ref.startswith("citation_check:"):
        return "citation check"
    return "unresolved"


def _format_span(span: Any) -> str:
    if isinstance(span, dict):
        start = span.get("char_start", span.get("start"))
        end = span.get("char_end", span.get("end"))
        if start is not None and end is not None:
            return f"{start}-{end}"
    if isinstance(span, (list, tuple)) and len(span) >= 2:
        return f"{span[0]}-{span[1]}"
    return "Recorded snippet"


def _citation_item_for_ref(ref: str, source_items: list[dict[str, str]], record: dict[str, Any]) -> dict[str, str] | None:
    if ref.startswith("artifact:"):
        item = next((item for item in source_items if item["ref"] == ref), None)
        return dict(item) if item else None
    if not ref.startswith("evidence_chunk:"):
        return None
    links = record.get("evidence_links")
    if not isinstance(links, list):
        return None
    for link in links:
        if not isinstance(link, dict):
            continue
        if str(link.get("evidence_chunk_ref") or "") != ref:
            continue
        artifact_ref = str(link.get("artifact_ref") or "")
        base = next((item for item in source_items if item["ref"] == artifact_ref), None)
        if not base:
            return None
        item = dict(base)
        item["ref"] = ref
        item["ref_kind"] = "Evidence chunk"
        item["artifact_ref"] = artifact_ref
        item["snippet"] = str(link.get("snippet") or base.get("snippet") or "")
        item["span"] = _format_span(link.get("span"))
        speaker_context = (
            link.get("speaker_context")
            if isinstance(link.get("speaker_context"), dict)
            else {}
        )
        item["speaker_label"] = str(speaker_context.get("label") or "")
        item["speaker_span"] = (
            _format_span(speaker_context.get("source_span"))
            if speaker_context
            else ""
        )
        return item
    return None


def _citation_disclosure_for_refs(
    refs: list[str],
    source_items: list[dict[str, str]],
    record: dict[str, Any] | None = None,
    *,
    initially_open: bool = False,
) -> str:
    record = record or {}
    brief_id = str(record.get("brief_id") or "")
    audit_refs = [str(ref) for ref in record.get("audit_refs", []) if isinstance(ref, str)]
    audit_ref = audit_refs[0] if audit_refs else "No creation audit reference recorded"
    items: list[dict[str, str]] = []
    unresolved: list[str] = []
    for ref in refs:
        item = _citation_item_for_ref(ref, source_items, record)
        if item and item not in items:
            items.append(item)
        elif not item:
            unresolved.append(ref)
    cards = []
    for index, item in enumerate(items[:3], start=1):
        open_attr = " open" if initially_open and index == 1 else ""
        ref_kind = item.get("ref_kind") or _citation_ref_kind(item.get("ref", ""))
        span = item.get("span") or "Whole source"
        speaker_context = (
            f'<p class="cs-citation-context"><strong>Speaker at span start:</strong> '
            f'{h(item["speaker_label"])} '
            f'<span class="cs-muted">(preceding source context {h(item.get("speaker_span") or "recorded")})</span></p>'
            if item.get("speaker_label")
            else ""
        )
        cards.append(
            f"""
<details class="cs-citation-card"{open_attr}>
  <summary>
    <span class="cs-citation-title">
      {_chip(item["label"], "searchable")}
      <strong>{h(item["title"])}</strong>
    </span>
    <span class="cs-citation-action">Inspect source</span>
  </summary>
  <div class="cs-citation-body">
    {speaker_context}
    <p class="cs-citation-snippet"><strong>Source snippet:</strong> {h(_truncate(item["snippet"], 260))}</p>
    <p class="cs-muted"><strong>Why this supports the finding:</strong> this recorded source span is the cited support for this draft finding; inspect it before use.</p>
    <div class="cs-citation-meta" aria-label="Full provenance">
      <div><span class="cs-meta">Citation ref</span><strong>{h(ref_kind)}</strong></div>
      <div><span class="cs-meta">Source span</span><strong>{h(span)}</strong></div>
      <div><span class="cs-meta">Saved</span><strong>{h(item["date"])}</strong></div>
      <div><span class="cs-meta">Fingerprint</span><strong>{h(item["fingerprint"])}</strong></div>
      <div><span class="cs-meta">Related object</span><strong>{h(f'Brief {brief_id}' if brief_id else 'Brief draft')}</strong></div>
      <div><span class="cs-meta">Audit reference</span><strong>{h(audit_ref)}</strong></div>
    </div>
    <div class="cs-citation-actions">
      <a class="cs-button secondary" href="{h(item["href"])}">Open source</a>
      <a class="cs-button ghost" href="/audit">History</a>
    </div>
  </div>
</details>
"""
        )
    for ref in unresolved[:3]:
        cards.append(
            f"""
<div class="cs-citation-card cs-citation-card-unresolved" data-citation-ref-kind="{h(_citation_ref_kind(ref))}">
  <div class="cs-citation-body">
    <div class="cs-panel-header"><h3>Unresolved citation ref</h3>{_chip("Needs source check", "underReview")}</div>
    <p class="cs-citation-snippet">A citation ref was recorded, but this page cannot match it to a visible source span yet.</p>
    <div class="cs-citation-meta" aria-label="Unresolved citation metadata">
      <div><span class="cs-meta">Citation ref kind</span><strong>{h(_citation_ref_kind(ref))}</strong></div>
      <div><span class="cs-meta">Review state</span><strong>Unsupported or unresolved</strong></div>
    </div>
  </div>
</div>
"""
        )
    return "".join(cards)


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
        presentation = _artifact_state(ctx, artifact)
        source_ref = storage_refs.get(ref) or artifact.get("checksum_sha256") or artifact.get("original_storage_ref")
        preview = str(artifact.get("_preview") or "") or _safe_preview(
            ctx.get("store"),
            artifact,
            260,
            ctx.get("load_errors") if isinstance(ctx.get("load_errors"), list) else None,
            "source preview",
        )
        display_artifact = artifact if artifact.get("_preview") else {**artifact, "_preview": preview}
        items.append(
            {
                "ref": ref,
                "label": f"Source {index}",
                "title": _artifact_title(display_artifact),
                "snippet": snippets.get(ref) or preview or "Open source.",
                "href": _detail_href("artifacts", artifact_id),
                "date": _display_date(artifact),
                "fingerprint": _fingerprint(source_ref),
                "state_label": str(presentation["label"]),
                "state": str(presentation["state"]),
            }
        )
    return items


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
    return _truncate(redact_text(source_ref), 96)


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
        "missing_evidence_bundle": "supporting evidence is missing",
        "attached Evidence Bundle": "attached supporting evidence",
        "this Evidence Bundle": "this supporting evidence",
        "linked Evidence Bundle": "linked supporting evidence",
        "an Evidence Bundle": "supporting evidence",
        "Evidence Bundle": "supporting evidence",
        "evidence bundle": "supporting evidence",
        "artifact references": "source links",
        "artifact reference": "source link",
        "owner_approval_required": "owner approval is required",
        "external_writeback": "provider write",
        "external writeback": "provider write",
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
    label_text = label.replace("-", " ")
    title = "We could not find that page" if label == "page" else f"This {label_text} is not available"
    body = (
        "The link may be old, or the page may not exist in this local product workspace. Search saved work or return to Home to continue."
        if label == "page"
        else f"The {label_text} may be outside this local workspace, hidden from the product area, or no longer saved. Search the workspace before starting over."
    )
    product_state = "not-found" if label == "page" else "permission-denied-or-not-found"
    boundary_note = (
        ""
        if label == "page"
        else "To protect workspace boundaries, CornerStone does not reveal whether unavailable work belongs to another owner."
    )
    return f"""
<section data-product-surface="not-found" data-product-state="{h(product_state)}">
  <div class="cs-grid-two">
    <div>
      {_empty_state(
        "Not found",
        title,
        body,
        "Search workspace",
        "/search",
        "Return home",
        "/",
        mark="?",
        steps=[
            ("1. Check search", "Look across saved sources, briefs, claims, and actions."),
            ("2. Open a list", "Use the sidebar if you know the work type."),
            ("3. Start again", "Drop or ask from Home if the work is not saved yet."),
        ],
    )}
      {f'<p class="cs-muted">{h(boundary_note)}</p>' if boundary_note else ''}
    </div>
    <aside class="cs-stack">
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Useful places</h2>
        <div class="cs-review-box">
          <a class="cs-button secondary" href="/artifacts">Saved sources</a>
          <a class="cs-button secondary" href="/briefs">Brief workspace</a>
          <a class="cs-button secondary" href="/inbox">Review inbox</a>
        </div>
      </section>
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Why this can happen</h2>
        <p class="cs-muted">CornerStone keeps each workspace local. A link from another workspace or an old run may not resolve here.</p>
      </section>
    </aside>
  </div>
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


def _home_script(scope: dict[str, Any]) -> str:
    safe_scope = {
        "tenant_id": str(scope.get("tenant_id") or "local-dev"),
        "owner_id": str(scope.get("owner_id") or "local-user"),
        "namespace_id": str(scope.get("namespace_id") or "personal"),
        "workspace_id": str(scope.get("workspace_id") or "default"),
    }
    return """
<script>
(function () {
  const scope = __SCOPE__;
  function scopedUrl(path) {
    const url = new URL(path, window.location.origin);
    Object.entries(scope).forEach(([key, value]) => url.searchParams.set(key, value));
    return url.pathname + url.search + url.hash;
  }
  function preserveScope() {
    document.querySelectorAll('a[href^="/"]').forEach(link => {
      link.setAttribute("href", scopedUrl(link.getAttribute("href")));
    });
    document.querySelectorAll('form[method="get"], form[method="GET"]').forEach(form => {
      Object.entries(scope).forEach(([key, value]) => {
        let input = form.querySelector('input[name="' + key + '"]');
        if (!input) {
          input = document.createElement("input");
          input.type = "hidden";
          input.name = key;
          form.appendChild(input);
        }
        input.value = value;
      });
    });
  }
  preserveScope();
  const dropForm = document.getElementById("cs-drop-form");
  const dropText = document.getElementById("cs-drop-text");
  const dropStatus = document.getElementById("cs-drop-status");
  const fileInput = document.getElementById("cs-file-input");
  const fileButton = document.getElementById("cs-file-button");
  const saveButton = document.getElementById("cs-save-source-button");
  const sourceChoiceList = document.getElementById("cs-source-choice-list");
  const sourceSelectionSummary = document.getElementById("cs-source-selection-summary");
  const askForm = document.getElementById("cs-ask-form");
  const askInput = document.getElementById("cs-ask-input");
  const askStatus = document.getElementById("cs-ask-status");
  const askButton = document.getElementById("cs-ask-submit-button");
  const claimButton = document.getElementById("cs-create-claim-button");
  const claimStatus = document.getElementById("cs-claim-create-status");
  const approveClaimButton = document.getElementById("cs-approve-claim-button");
  const approveClaimDialog = document.getElementById("cs-claim-approval-dialog");
  const confirmClaimApproval = document.getElementById("cs-confirm-claim-approval");
  const claimApprovalStatus = document.getElementById("cs-claim-approval-status");
  const approveActionButton = document.getElementById("cs-approve-action-button");
  const approveActionDialog = document.getElementById("cs-action-approval-dialog");
  const confirmActionApproval = document.getElementById("cs-confirm-action-approval");
  const actionApprovalStatus = document.getElementById("cs-action-approval-status");
  const maxDecisionSourceBytes = __MAX_DECISION_SOURCE_BYTES__;
  const maxDecisionTotalBytes = __MAX_DECISION_TOTAL_BYTES__;
  const maxDecisionSourceCount = 5;
  const currentBatchSourceIds = new Set();
  function setStatus(node, message, state) {
    if (!node) return;
    const status = state || "idle";
    node.hidden = false;
    node.textContent = message;
    node.dataset.state = status;
    node.classList.remove("is-idle", "is-loading", "is-success", "is-error");
    node.classList.add("is-" + status);
    node.setAttribute("aria-live", status === "error" ? "assertive" : "polite");
  }
  function setBusy(form, button, busy, loadingLabel, defaultLabel) {
    if (form) form.setAttribute("aria-busy", busy ? "true" : "false");
    if (!button) return;
    button.disabled = busy;
    button.textContent = busy ? loadingLabel : defaultLabel;
  }
  function decisionSourceInputs() {
    return Array.from(document.querySelectorAll("[data-decision-source]"));
  }
  function selectedSourceStats() {
    const selected = decisionSourceInputs().filter(node => node.checked && !node.disabled);
    return {
      selected,
      count: selected.length,
      totalBytes: selected.reduce((total, node) => total + Number(node.dataset.sourceSize || 0), 0)
    };
  }
  function formatKiB(bytes) {
    if (!bytes) return "0";
    return String(Math.ceil(bytes / 1024));
  }
  function updateSourceSelectionSummary() {
    if (!sourceSelectionSummary) return;
    const stats = selectedSourceStats();
    sourceSelectionSummary.textContent = stats.count + " of 5 sources selected · " + formatKiB(stats.totalBytes) + " of 512 KiB";
  }
  function bindSourceChoice(input) {
    if (!input || input.dataset.selectionBound === "true") return;
    input.dataset.selectionBound = "true";
    input.addEventListener("change", function () {
      const stats = selectedSourceStats();
      if (stats.count > maxDecisionSourceCount || stats.totalBytes > maxDecisionTotalBytes) {
        input.checked = false;
        setStatus(
          askStatus,
          stats.count > maxDecisionSourceCount
            ? "Select no more than five sources for this Brief."
            : "Selected sources exceed the 512 KiB Brief limit. Unselect a source before adding this one.",
          "error"
        );
      }
      updateSourceSelectionSummary();
    });
  }
  function beginCurrentSourceBatch() {
    currentBatchSourceIds.clear();
    decisionSourceInputs().forEach(node => {
      node.checked = false;
      node.dataset.currentBatch = "false";
    });
    updateSourceSelectionSummary();
  }
  function sourceTitle(artifact) {
    const source = artifact.source || {};
    if (source.filename) return source.filename;
    if (source.type === "user_paste") return "Pasted source";
    return source.ref || "Saved source";
  }
  function sourceBriefEligibility(artifact) {
    const size = Number(artifact.original_size_bytes || 0);
    const derived = artifact.derived || {};
    if (size > maxDecisionSourceBytes) {
      return {eligible: false, size, note: "Saved only · exceeds the 128 KiB per-source Brief limit"};
    }
    if (derived.status !== "ready" || !derived.text_ref) {
      return {eligible: false, size, note: "Saved only · no readable plain-text representation for this Brief"};
    }
    return {eligible: true, size, note: "Saved and available for this Brief"};
  }
  function addCurrentBatchSource(artifact) {
    if (!sourceChoiceList || !artifact || !artifact.artifact_id) return false;
    const id = String(artifact.artifact_id);
    const eligibility = sourceBriefEligibility(artifact);
    const newToCurrentBatch = !currentBatchSourceIds.has(id);
    currentBatchSourceIds.add(id);
    const empty = document.getElementById("cs-source-set-empty");
    if (empty) empty.remove();
    let input = decisionSourceInputs().find(node => node.value === id);
    let note = null;
    if (!input) {
      const label = document.createElement("label");
      label.className = "cs-source-choice";
      input = document.createElement("input");
      input.type = "checkbox";
      input.value = id;
      input.setAttribute("data-decision-source", "");
      const copy = document.createElement("span");
      const title = document.createElement("strong");
      note = document.createElement("small");
      note.setAttribute("data-source-note", "");
      title.textContent = sourceTitle(artifact);
      copy.append(title, note);
      label.append(input, copy);
      sourceChoiceList.append(label);
      bindSourceChoice(input);
    } else {
      note = input.closest("label") && input.closest("label").querySelector("[data-source-note]");
    }
    input.dataset.sourceSize = String(eligibility.size);
    input.dataset.currentBatch = "true";
    input.disabled = !eligibility.eligible;
    let selected = false;
    if (eligibility.eligible) {
      const stats = selectedSourceStats();
      if (stats.count < maxDecisionSourceCount && stats.totalBytes + eligibility.size <= maxDecisionTotalBytes) {
        input.checked = true;
        selected = true;
      }
    }
    if (note) {
      note.textContent = selected
        ? eligibility.note + " · selected"
        : eligibility.eligible
          ? "Saved · not selected because the current Brief reached its count or 512 KiB limit"
          : eligibility.note;
    }
    updateSourceSelectionSummary();
    return selected && newToCurrentBatch;
  }
  decisionSourceInputs().forEach(bindSourceChoice);
  updateSourceSelectionSummary();
  async function postJson(path, body) {
    const response = await fetch(path, {
      method: "POST",
      headers: {"content-type": "application/json", "accept": "application/json"},
      body: JSON.stringify(Object.assign({}, scope, body))
    });
    const payload = await response.json();
    if (!response.ok || payload.status === "failed" || payload.status === "denied") {
      const error = new Error((payload.errors && payload.errors[0] && payload.errors[0].message) || "Request failed.");
      error.payload = payload;
      throw error;
    }
    return payload;
  }
  async function postFile(file) {
    if (file.size > __MAX_UPLOAD_BYTES__) {
      throw new Error("File exceeds the __MAX_UPLOAD_MB__ MB local upload limit.");
    }
    const response = await fetch(scopedUrl("/artifacts/upload"), {
      method: "POST",
      headers: {
        "content-type": file.type || "application/octet-stream",
        "accept": "application/json",
        "x-cornerstone-filename": encodeURIComponent(file.name || "upload.bin")
      },
      body: file
    });
    const payload = await response.json();
    if (!response.ok || payload.status === "failed" || payload.status === "denied") {
      const error = new Error((payload.errors && payload.errors[0] && payload.errors[0].message) || "File upload failed.");
      error.payload = payload;
      throw error;
    }
    return payload;
  }
  async function saveText(text, sourceRef) {
    setBusy(dropForm, saveButton, true, "Saving", "Save source");
    setStatus(dropStatus, "Saving source...", "loading");
    try {
      const payload = {};
      payload.text = text;
      payload["source" + "_ref"] = sourceRef || "home-paste";
      const saved = await postJson("/artifacts", payload);
      const artifact = saved.artifact || {};
      beginCurrentSourceBatch();
      const selected = addCurrentBatchSource(artifact);
      dropText.value = "";
      setStatus(
        dropStatus,
        selected
          ? "Source saved and selected. Add more sources or ask the decision question below."
          : "Source saved, but it is outside the Brief text or size boundary. Add a supported source or choose another saved source below.",
        "success"
      );
    } catch (error) {
      setStatus(dropStatus, error.message, "error");
    } finally {
      setBusy(dropForm, saveButton, false, "Saving", "Save source");
    }
  }
  async function saveFiles(fileList) {
    const files = Array.from(fileList || []);
    if (files.length < 1) return;
    if (files.length > maxDecisionSourceCount) {
      setStatus(dropStatus, "Choose one to five files at a time.", "error");
      return;
    }
    setBusy(dropForm, saveButton, true, "Archiving", "Save source");
    if (fileButton) fileButton.disabled = true;
    beginCurrentSourceBatch();
    let savedCount = 0;
    let selectedCount = 0;
    try {
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index];
        setStatus(dropStatus, "Archiving " + (index + 1) + " of " + files.length + ": " + (file.name || "source") + "...", "loading");
        const saved = await postFile(file);
        const artifact = saved.artifact || {};
        savedCount += 1;
        if (addCurrentBatchSource(artifact)) selectedCount += 1;
      }
      setStatus(
        dropStatus,
        savedCount + " source" + (savedCount === 1 ? "" : "s") + " archived. " + selectedCount + " selected for this Brief. Add sources or ask the decision question below.",
        "success"
      );
    } catch (error) {
      setStatus(dropStatus, savedCount + " source" + (savedCount === 1 ? "" : "s") + " archived before the error. " + error.message, "error");
    } finally {
      setBusy(dropForm, saveButton, false, "Archiving", "Save source");
      if (fileButton) fileButton.disabled = false;
      if (fileInput) fileInput.value = "";
    }
  }
  if (dropForm && dropText) {
    dropForm.addEventListener("submit", function (event) {
      event.preventDefault();
      const text = dropText.value.trim();
      if (!text) {
        setStatus(dropStatus, "Paste text before saving.", "error");
        return;
      }
      saveText(text, "home-paste");
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
      const files = event.dataTransfer && event.dataTransfer.files;
      if (!files || !files.length) return;
      saveFiles(files);
    });
  }
  if (fileButton && fileInput) {
    fileButton.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", function () {
      const files = fileInput.files;
      if (!files || !files.length) return;
      saveFiles(files);
    });
  }
  document.querySelectorAll("[data-ask-suggestion]").forEach(button => {
    button.addEventListener("click", function () {
      if (askInput) askInput.value = button.getAttribute("data-ask-suggestion") || "";
      if (askInput) askInput.focus();
    });
  });
  if (claimButton) {
    claimButton.addEventListener("click", async function () {
      const briefId = claimButton.getAttribute("data-brief-id") || "";
      const statement = claimButton.getAttribute("data-statement") || "";
      if (!briefId || !statement) {
        setStatus(claimStatus, "This Brief needs a supported finding before a Decision draft can be saved.", "error");
        return;
      }
      claimButton.disabled = true;
      setStatus(claimStatus, "Saving Decision draft with its sources...", "loading");
      try {
        const created = await postJson("/claims", {brief_id: briefId, statement});
        const claim = created.claim || {};
        const claimId = claim["claim" + "_id"];
        if (!claimId) throw new Error("Decision draft was not saved.");
        setStatus(claimStatus, "Decision draft saved. Opening draft...", "success");
        window.location.href = scopedUrl("/claims/" + encodeURIComponent(claimId) + "?view=html");
      } catch (error) {
        setStatus(claimStatus, error.message, "error");
        claimButton.disabled = false;
      }
    });
  }
  if (approveClaimButton && approveClaimDialog) {
    approveClaimButton.addEventListener("click", function () {
      if (typeof approveClaimDialog.showModal === "function") approveClaimDialog.showModal();
      else approveClaimDialog.setAttribute("open", "");
    });
  }
  if (approveClaimButton && confirmClaimApproval) {
    confirmClaimApproval.addEventListener("click", async function (event) {
      event.preventDefault();
      const claimId = approveClaimButton.getAttribute("data-claim-id") || "";
      if (!claimId) {
        setStatus(claimApprovalStatus, "Claim approval is unavailable because the record ID is missing.", "error");
        return;
      }
      if (approveClaimDialog && typeof approveClaimDialog.close === "function") approveClaimDialog.close();
      approveClaimButton.disabled = true;
      confirmClaimApproval.disabled = true;
      setStatus(claimApprovalStatus, "Recording approval...", "loading");
      try {
        await postJson("/claims/" + encodeURIComponent(claimId) + "/approve", {});
        setStatus(claimApprovalStatus, "Approval recorded. Reloading Claim...", "success");
        window.location.href = scopedUrl("/claims/" + encodeURIComponent(claimId) + "?view=html");
      } catch (error) {
        setStatus(claimApprovalStatus, error.message, "error");
        approveClaimButton.disabled = false;
        confirmClaimApproval.disabled = false;
      }
    });
  }
  if (approveActionButton && approveActionDialog) {
    approveActionButton.addEventListener("click", function () {
      if (typeof approveActionDialog.showModal === "function") approveActionDialog.showModal();
      else approveActionDialog.setAttribute("open", "");
    });
  }
  if (approveActionButton && confirmActionApproval) {
    confirmActionApproval.addEventListener("click", async function (event) {
      event.preventDefault();
      const actionId = approveActionButton.getAttribute("data-action-id") || "";
      const approver = approveActionButton.getAttribute("data-approver") || scope.owner_id;
      if (!actionId) {
        setStatus(actionApprovalStatus, "Action approval is unavailable because the record ID is missing.", "error");
        return;
      }
      if (approveActionDialog && typeof approveActionDialog.close === "function") approveActionDialog.close();
      approveActionButton.disabled = true;
      confirmActionApproval.disabled = true;
      setStatus(actionApprovalStatus, "Recording approval...", "loading");
      try {
        await postJson("/actions/" + encodeURIComponent(actionId) + "/approve", {approver});
        setStatus(actionApprovalStatus, "Approval recorded. Reloading action...", "success");
        window.location.href = scopedUrl("/actions/" + encodeURIComponent(actionId) + "?view=html");
      } catch (error) {
        setStatus(actionApprovalStatus, error.message, "error");
        approveActionButton.disabled = false;
        confirmActionApproval.disabled = false;
      }
    });
  }
  if (askForm && askInput) {
    askForm.addEventListener("submit", async function (event) {
      event.preventDefault();
      const question = askInput.value.trim();
      if (!question) {
        setStatus(askStatus, "Enter a question first.", "error");
        return;
      }
      const sourceStats = selectedSourceStats();
      const sourceIds = sourceStats.selected.map(node => node.value).filter(Boolean);
      if (sourceIds.length < 1 || sourceIds.length > 5) {
        setStatus(askStatus, "Select one to five saved sources for this decision.", "error");
        return;
      }
      if (sourceStats.totalBytes > maxDecisionTotalBytes) {
        setStatus(askStatus, "Selected sources exceed the 512 KiB Brief limit.", "error");
        return;
      }
      setBusy(askForm, askButton, true, "Checking", "Ask");
      try {
        setStatus(askStatus, "Checking saved sources...", "loading");
        const started = await postJson("/conversations", {message: question});
        const conversation = started.conversation || {};
        const id = conversation["conversation" + "_id"];
        if (!id) throw new Error("Conversation was not saved.");
        const answered = await postJson("/conversations/" + encodeURIComponent(id) + "/answers", {question, artifact_ids: sourceIds});
        const answer = answered.answer || {};
        const sourceSnapshot = answered.search_snapshot || {};
        const text = answer.answer || "Draft answer saved. Open the linked sources before using it.";
        const safety = conversation.safety || {};
        if (safety.unsafe_instruction_detected === true) {
          setStatus(askStatus, "Draft answer saved. Brief preparation was blocked because the Ask text contained unsafe instructions.", "error");
          return;
        }
        if (Number(sourceSnapshot.result_count || 0) > 0 && sourceSnapshot.search_snapshot_id) {
          setStatus(askStatus, "Synthesizing a decision Brief from the selected sources...", "loading");
          const bundled = await postJson("/evidence-bundles", {search_snapshot_id: sourceSnapshot.search_snapshot_id});
          const bundle = bundled.evidence_bundle || {};
          if (!bundle.evidence_bundle_id) throw new Error("Supporting evidence was not saved.");
          const prepared = await postJson("/briefs", {evidence_bundle_id: bundle.evidence_bundle_id});
          const brief = prepared.brief || {};
          if (!brief.brief_id) throw new Error("Brief draft was not saved.");
          const briefMode = brief.output_mode === "ollama_generated" ? "Decision Brief ready." : "Extractive fallback prepared.";
          setStatus(askStatus, briefMode + " Opening source-linked draft...", "success");
          window.location.href = scopedUrl("/briefs/" + encodeURIComponent(brief.brief_id) + "?view=html");
          return;
        }
        setStatus(askStatus, "Draft answer: " + text + " No pre-existing source matched, so no Brief was created.", "success");
      } catch (error) {
        setStatus(askStatus, error.message, "error");
      } finally {
        setBusy(askForm, askButton, false, "Checking", "Ask");
      }
    });
  }
}());
</script>
""".replace("__SCOPE__", _script_json(safe_scope)).replace(
        "__MAX_UPLOAD_BYTES__", str(MAX_BROWSER_UPLOAD_BYTES)
    ).replace("__MAX_UPLOAD_MB__", str(MAX_BROWSER_UPLOAD_BYTES // (1024 * 1024))).replace(
        "__MAX_DECISION_SOURCE_BYTES__", str(VS5_DECISION_SOURCE_MAX_BYTES)
    ).replace("__MAX_DECISION_TOTAL_BYTES__", str(VS5_DECISION_TOTAL_SOURCE_MAX_BYTES))

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

PRODUCT_LIST_ROUTES = {"/", "/search", "/artifacts", "/briefs", "/claims", "/actions", "/inbox", "/audit"}
PRODUCT_SEARCH_TYPES = [
    ("all", "All"),
    ("sources", "Sources"),
    ("briefs", "Briefs"),
    ("claims", "Claims"),
    ("actions", "Actions"),
]
PRODUCT_DETAIL_ROUTES = {"artifacts", "briefs", "claims", "memories", "actions"}

REFERENCE_IMAGE_ROWS = [
    ("cornerstone-reference-01-vendor-detail.png", "Vendor object detail", "Dormant direction", "Entity/object explorer structure only; do not add this surface during the VS5 scope freeze."),
    ("cornerstone-reference-02-operations-inbox.png", "Operations inbox", "Active surface", "Lane tabs, triage table, selected preview, and next actions."),
    ("cornerstone-reference-03-admin-connectors.png", "Admin connectors", "Owner-only direction", "Connector governance stays contained in the owner area."),
    ("cornerstone-reference-04-search-results.png", "Search results", "Active surface", "Universal search, scoped filters, result receipts, and suggested follow-ups."),
    ("cornerstone-reference-05-claim-draft-supporting-evidence.png", "Claim draft", "Active surface", "Claim statement, trust ladder, supporting evidence, and review controls."),
    ("cornerstone-reference-06-artifact-viewer.png", "Artifact viewer", "Active surface", "Original artifact primary, source metadata, derived keywords, linked work, and provenance."),
    ("cornerstone-reference-07-home-upload-ask.png", "Home workspace", "Active surface", "Drop zone, ask box, recent items, knowledge states, and next steps."),
    ("cornerstone-reference-08-action-dry-run-approval.png", "Action dry-run", "Active surface", "Dry-run impact, proposed changes, policy decision, risk, approval, and auditability."),
]

INTERNAL_PRODUCT_VISIBILITIES = {"internal", "owner_only", "verification_only"}
INTERNAL_PRODUCT_SOURCE_TYPES = {"internal_fixture", "local_fixture", "scenario_fixture", "verification_fixture"}
PRODUCT_RECORD_ID_KEYS = ("artifact_id", "brief_id", "claim_id", "action_id", "memory_id")


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
    ctx = _build_context(store, scope)
    route = path if path in PRODUCT_LIST_ROUTES else "/"
    q = (query.get("q") or [""])[-1].strip()
    requested_search_type = (query.get("type") or ["all"])[-1].strip().lower()
    search_type = requested_search_type if requested_search_type in {value for value, _ in PRODUCT_SEARCH_TYPES} else "all"
    if route == "/":
        title = "Home"
        content = _home(ctx)
        active = "/"
    elif route == "/search":
        title = "Search"
        content = _search_page(ctx, q, search_type)
        active = "/search"
    elif route == "/artifacts":
        title = "Artifacts"
        content = _artifact_list_page(ctx)
        active = "/artifacts"
    elif route == "/briefs":
        title = "Briefs"
        content = _brief_list_page(ctx)
        active = "/briefs"
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
        _load_selected_product_loop(ctx)
        content = _inbox_page(ctx)
        active = "/inbox"
    else:
        title = "Audit"
        content = _audit_page(ctx)
        active = "/audit"
    return _page(root, title, active, content, ctx, q)


def render_owner_review_page(
    root: Path,
    store: Any,
    scope: dict[str, str],
    readiness: dict[str, Any],
) -> str:
    ctx = _build_context(store, scope)
    content = _owner_review_page(ctx, readiness)
    return _page(root, "Owner", "/review", content, ctx, "")


def render_owner_reference_images_page(
    root: Path,
    store: Any,
    scope: dict[str, str],
) -> str:
    ctx = _build_context(store, scope)
    content = _owner_reference_images_page(root)
    return _page(root, "Reference Images", "/review", content, ctx, "")


def render_product_not_found(
    root: Path,
    store: Any,
    scope: dict[str, str],
) -> str:
    ctx = _build_context(store, scope)
    return _page(root, "Page not found", "/", _not_found("page"), ctx, "")


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
        record = store.get_brief(record_id)
        if record and record.get("scope") != scope:
            record = None
        if record and _internal_product_record(record, ctx["internal_lineage_refs"]):
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
        if record and _internal_product_record(record, ctx["internal_lineage_refs"]):
            content = _internal_record_notice("claim")
            title = "Owner record"
            active = "/claims"
            return 200, _page(root, title, active, content, ctx, "")
        content = _claim_detail(ctx, record) if record else _not_found("claim")
        title = _truncate(str(record.get("statement") or "Claim"), 72) if record else "Claim not found"
        active = "/claims"
    elif kind == "memories":
        record = store.get_memory(record_id)
        if record and record.get("scope") != scope:
            record = None
        if record and _internal_product_record(record, ctx["internal_lineage_refs"]):
            content = _internal_record_notice("memory candidate")
            title = "Owner record"
            active = "/inbox"
            return 200, _page(root, title, active, content, ctx, "")
        content = _memory_detail(ctx, record) if record else _not_found("memory candidate")
        title = _truncate(str(record.get("title") or record.get("statement") or "Memory candidate"), 72) if record else "Memory candidate not found"
        active = "/inbox"
    elif kind == "actions":
        record = store.get_action(record_id)
        if record and record.get("scope") != scope:
            record = None
        if record and _internal_product_record(record, ctx["internal_lineage_refs"]):
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
    load_errors: list[str] = []
    artifacts = _recent(_safe_records(lambda: store._artifact_records(scope), load_errors, "saved sources"))
    briefs = _recent(_safe_records(lambda: store._brief_records(scope), load_errors, "briefs"))
    claims = _recent(_safe_records(lambda: store._claim_records(scope), load_errors, "claims"))
    actions = _recent(_safe_records(lambda: store._action_records(scope), load_errors, "action drafts"))
    memories = _recent(_safe_records(lambda: store._memory_records(scope), load_errors, "memory drafts"))
    audit = _recent(
        [
            event
            for event in _safe_records(lambda: store._all_audit_events(), load_errors, "activity receipts")
            if _same_scope(event.get("scope") if isinstance(event.get("scope"), dict) else event, scope)
        ],
        limit=80,
    )
    try:
        audit_integrity = store.verify_audit()
    except Exception:
        load_errors.append("audit integrity")
        audit_integrity = {"status": "not_verified", "event_count": 0, "errors": []}
    for artifact in artifacts:
        artifact["_preview"] = _safe_preview(store, artifact, 260, load_errors, "source preview")
    internal_lineage_refs, internal_record_objects = _internal_product_lineage(
        [artifacts, briefs, claims, actions, memories]
    )
    artifacts = [record for record in artifacts if id(record) not in internal_record_objects]
    briefs = [record for record in briefs if id(record) not in internal_record_objects]
    claims = [record for record in claims if id(record) not in internal_record_objects]
    actions = [record for record in actions if id(record) not in internal_record_objects]
    memories = [record for record in memories if id(record) not in internal_record_objects]
    inbox = _inbox_items(briefs, claims, actions, memories, scope=scope)
    return {
        "store": store,
        "scope": scope,
        "artifacts": artifacts,
        "briefs": briefs,
        "claims": claims,
        "actions": actions,
        "memories": memories,
        "audit": audit,
        "audit_integrity": audit_integrity,
        "internal_lineage_refs": internal_lineage_refs,
        "load_errors": list(dict.fromkeys(load_errors)),
        "suggestions": _suggestions(artifacts, briefs, claims),
        "inbox": inbox,
        "selected_product_loop": None,
    }


def _load_selected_product_loop(ctx: dict[str, Any]) -> None:
    inbox = ctx.get("inbox") if isinstance(ctx.get("inbox"), list) else []
    if not inbox:
        return
    selected = inbox[0]
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


def _same_scope(value: Any, scope: dict[str, str]) -> bool:
    return isinstance(value, dict) and all(value.get(key) == expected for key, expected in scope.items())


def _record_identity_refs(record: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in PRODUCT_RECORD_ID_KEYS:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            continue
        kind = key.removesuffix("_id")
        refs.update({value, f"{kind}:{value}"})
    return refs


def _record_lineage_refs(record: dict[str, Any]) -> set[str]:
    refs: set[str] = set()

    def collect(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            ref_type = value.get("type")
            ref_id = value.get("id")
            if isinstance(ref_type, str) and isinstance(ref_id, str) and ref_type and ref_id:
                refs.update({ref_id, f"{ref_type}:{ref_id}"})
            for child_key, child_value in value.items():
                collect(child_value, str(child_key))
            return
        if isinstance(value, list):
            for item in value:
                collect(item, key)
            return
        if not isinstance(value, str) or not value:
            return
        if key.endswith("_id") or key.endswith("_ref") or key.endswith("_refs"):
            refs.add(value)

    collect(record)
    return refs


def _internal_product_lineage(
    record_groups: list[list[dict[str, Any]]],
) -> tuple[set[str], set[int]]:
    internal_refs: set[str] = set()
    internal_record_objects: set[int] = set()
    pending = [record for records in record_groups for record in records]
    changed = True
    while changed:
        changed = False
        for record in pending:
            object_id = id(record)
            if object_id in internal_record_objects:
                continue
            if not _internal_product_record(record, internal_refs):
                continue
            internal_record_objects.add(object_id)
            internal_refs.update(_record_identity_refs(record))
            changed = True
    return internal_refs, internal_record_objects


def _internal_product_record(record: dict[str, Any], internal_refs: set[str] | None = None) -> bool:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    visibility_values = {
        str(record.get("visibility") or "").lower(),
        str(record.get("product_visibility") or "").lower(),
        str(metadata.get("visibility") or "").lower(),
        str(metadata.get("product_visibility") or "").lower(),
        str(provenance.get("visibility") or "").lower(),
    }
    source_type = str(source.get("type") or source.get("source_type") or "").lower()
    explicitly_internal = bool(visibility_values & INTERNAL_PRODUCT_VISIBILITIES) or source_type in INTERNAL_PRODUCT_SOURCE_TYPES
    return explicitly_internal or bool(internal_refs and _record_lineage_refs(record) & internal_refs)


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
        "action.execute": "Action executed",
        "action.execution.denied": "Action execution blocked",
        "action.preview.creation.denied": "Action preview boundary blocked",
        "conversation.start": "Ask started",
        "conversation.answer": "Draft answer saved",
        "product.mission_control.viewed": "Review queue opened",
        "product.loop.viewed": "Work journey opened",
        "mission.contract.created": "Decision path prepared",
        "mission.activated": "Decision path activated",
        "workspace.mode.set": "Workspace mode recorded",
    }
    if event_type in labels:
        return labels[event_type]
    return "Recorded activity"


def _audit_icon(event_type: str) -> str:
    if event_type.startswith("artifact."):
        return "S"
    if event_type.startswith("search.") or event_type.startswith("evidence_bundle."):
        return "E"
    if event_type.startswith("brief."):
        return "B"
    if event_type.startswith("claim."):
        return "C"
    if event_type.startswith("action."):
        return "A"
    if event_type.startswith("mission.") or event_type.startswith("workspace."):
        return "D"
    if event_type.startswith("conversation."):
        return "Q"
    return "L"


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


def _audit_receipt_card(label: str, value: int | str, detail: str) -> str:
    return f"""
<div class="cs-audit-receipt">
  <span class="cs-meta">{h(label)}</span>
  <strong>{h(str(value))}</strong>
  <span class="cs-meta">{h(detail)}</span>
</div>
"""


def _audit_lifecycle_card(title: str, count: int, detail: str, state: str) -> str:
    return f"""
<div class="cs-audit-lane">
  <div class="cs-audit-lane-head">
    <strong>{h(title)}</strong>
    <span class="cs-audit-lane-count">{h(str(count))}</span>
  </div>
  <p>{h(detail)}</p>
  {_chip("Present" if count else "Waiting", state)}
</div>
"""


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
    <div class="cs-audit-raw-item"><span class="cs-meta">Subject</span><strong>{h(subject_ref)}</strong></div>
    <div class="cs-audit-raw-item"><span class="cs-meta">Event type</span><strong>{h(str(event.get("event_type") or "recorded"))}</strong></div>
    <div class="cs-audit-raw-item"><span class="cs-meta">Event hash</span><strong>{h(event_hash)}</strong></div>
    <div class="cs-audit-raw-item"><span class="cs-meta">Previous hash</span><strong>{h(previous_hash)}</strong></div>
    <div class="cs-audit-raw-item"><span class="cs-meta">Detail fields</span><strong>{h(detail_keys)}</strong></div>
    <div class="cs-audit-raw-item"><span class="cs-meta">Scope</span><strong>{h(_scope_label(event_scope))}</strong></div>
  </div>
</details>
"""


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


def _brief_source_count(record: dict[str, Any]) -> int:
    refs = _evidence_refs(record)
    artifact_refs = {ref for ref in refs if ref.startswith("artifact:")}
    if artifact_refs:
        return len(artifact_refs)
    return len(set(refs))


def _brief_label_state(record: dict[str, Any]) -> tuple[str, str]:
    label_check = record.get("label_check") if isinstance(record.get("label_check"), dict) else {}
    if record.get("presented_as_fact") is True and label_check.get("earned_evidence_backed") is True:
        return "Fact label earned", "evidenceBacked"
    if label_check:
        return "Draft label", "underReview"
    return "Draft label", "draft"


def _brief_citation_refs(record: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("key_point_citations", "recommended_next_step_citations"):
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
    trust = str(record.get("trust_label") or record.get("status") or "").lower()
    return trust == "evidence_backed" and label_check.get("earned_evidence_backed") is True


def _action_lifecycle(record: dict[str, Any]) -> dict[str, Any]:
    approval = record.get("approval") if isinstance(record.get("approval"), dict) else {}
    execution = record.get("execution") if isinstance(record.get("execution"), dict) else {}
    result = execution.get("result") if isinstance(execution.get("result"), dict) else {}
    approval_status = str(approval.get("status") or "not_recorded").lower()
    execution_status = str(execution.get("status") or "not_started").lower()
    result_status = str(result.get("status") or "").lower()
    record_status = str(record.get("status") or "").lower()
    if execution_status == "executed":
        stage = "executed"
    elif execution_status in {"failed", "error"} or result_status in {"failed", "error"}:
        stage = "failed"
    elif (
        record_status in {"policy_blocked", "blocked", "denied"}
        or execution_status.startswith("blocked")
        or execution_status in {"denied", "policy_denied"}
        or result_status in {"blocked", "denied", "policy_denied"}
    ):
        stage = "blocked"
    elif approval_status == "approved" or execution_status == "ready_to_execute":
        stage = "approved"
    elif approval_status in {"pending", "not_approved", "required"}:
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
    *,
    scope: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    owner_label = str((scope or {}).get("owner_id") or "Owner")
    for brief in briefs[:2]:
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
    for claim in [claim for claim in claims if str(claim.get("status") or "").lower() != "approved"][:2]:
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
                "created_at": _record_time_key(claim),
                "queue": "Needs review",
                "priority": "High" if state == "draft" else "Medium",
                "owner": owner_label,
                "type": "Claim",
                "icon": "C",
                "record_kind": "claim",
                "record_id": str(claim.get("claim_id") or ""),
                "record_ref": f"claim:{claim.get('claim_id')}" if claim.get("claim_id") else "",
            }
        )
    pending_actions = [action for action in actions if _action_lifecycle(action)["stage"] != "executed"]
    for action in pending_actions[:2]:
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
            "Inspect the failed run and recovery receipt before retrying."
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
    for memory in draft_memories[:2]:
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
    return _recent(items, limit=8)


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
    aliases = {
        "--cs-color-evidence-600": "var(--cs-color-evidence-700)",
        "--cs-color-surface-primary": "var(--cs-color-background-surface)",
        "--cs-color-surface-subtle": "var(--cs-color-background-subtle)",
        "--cs-radius-xs": "var(--cs-radius-sm)",
        "--cs-shadow-sm": "var(--cs-shadow-card)",
        "--cs-state-draft-text": "var(--cs-state-draft-fg)",
        "--cs-state-evidenceBacked-text": "var(--cs-state-evidenceBacked-fg)",
        "--cs-state-searchable-text": "var(--cs-state-searchable-fg)",
        "--cs-state-underReview-text": "var(--cs-state-underReview-fg)",
        "--cs-typography-weight-bold": "var(--cs-typography-display-fontWeight)",
        "--cs-typography-weight-medium": "500",
        "--cs-typography-weight-semibold": "var(--cs-typography-label-fontWeight)",
    }
    variables.extend(aliases.items())
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
  background:
    linear-gradient(180deg, var(--cs-color-surface-primary), color-mix(in srgb, var(--cs-color-surface-subtle) 68%, var(--cs-color-surface-primary)));
  padding: var(--cs-space-6) var(--cs-space-4);
  display: flex;
  flex-direction: column;
  gap: var(--cs-space-5);
}}
.cs-brand {{
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-brand-mark {{
  width: 38px;
  height: 38px;
  border-radius: var(--cs-radius-md);
  background:
    linear-gradient(135deg, var(--cs-color-primary-600), var(--cs-color-primary-700));
  color: var(--cs-color-text-inverse);
  display: grid;
  place-items: center;
  font-weight: var(--cs-typography-weight-bold);
  box-shadow: 0 10px 24px rgba(37, 87, 209, .18);
}}
.cs-brand-name {{ font-weight: var(--cs-typography-weight-bold); font-size: var(--cs-typography-sectionTitle-fontSize); }}
.cs-brand-sub {{ color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-shell-note {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: var(--cs-typography-metadata-lineHeight);
  max-width: 18ch;
}}
.cs-nav {{ display: grid; gap: var(--cs-space-4); }}
.cs-nav-group {{ display: grid; gap: var(--cs-space-2); }}
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
  display: grid;
  grid-template-columns: 26px minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--cs-space-2);
  font-weight: var(--cs-typography-weight-medium);
  transition: background .18s ease, color .18s ease, box-shadow .18s ease;
}}
.cs-nav a:hover, .cs-nav a:focus-visible {{ background: var(--cs-color-surface-primary); outline: none; box-shadow: var(--cs-shadow-sm); }}
.cs-nav a[aria-current="page"] {{
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  box-shadow: inset 3px 0 0 var(--cs-color-primary-600);
}}
.cs-nav-mark {{
  width: 26px;
  height: 26px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: color-mix(in srgb, var(--cs-color-surface-primary) 72%, var(--cs-color-surface-subtle));
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-nav a[aria-current="page"] .cs-nav-mark {{
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-primary-700);
}}
.cs-nav-count {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  font-variant-numeric: tabular-nums;
}}
.cs-sidebar-status {{
  margin-top: auto;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-sidebar-status-row {{
  display: flex;
  justify-content: space-between;
  gap: var(--cs-space-3);
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-sidebar-status-row strong {{ color: var(--cs-color-text-primary); font-variant-numeric: tabular-nums; }}
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
.cs-command {{
  flex: 1 1 auto;
  max-width: 860px;
  min-width: 280px;
}}
.cs-search {{
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2) var(--cs-space-3);
  width: 100%;
  min-height: 48px;
}}
.cs-search span[aria-hidden="true"] {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
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
.cs-topbar-actions {{
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-icon-button {{
  width: 34px;
  height: 34px;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-text-secondary);
  display: grid;
  place-items: center;
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-avatar {{
  min-width: 36px;
  height: 36px;
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-surface-subtle);
  border: 1px solid var(--cs-color-border-default);
  display: grid;
  place-items: center;
  color: var(--cs-color-text-primary);
  font-weight: var(--cs-typography-weight-semibold);
}}
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
  padding: var(--cs-space-3) var(--cs-space-4) var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-home-canvas .cs-panel-header {{ margin-bottom: 0; flex-wrap: wrap; }}
.cs-home-canvas p {{ max-width: 62ch; }}
.cs-home-workspace {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-home-source-row {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  align-items: center;
  justify-content: center;
}}
.cs-home-paste-row {{
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: var(--cs-space-2);
  align-items: stretch;
}}
.cs-home-source-note {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: var(--cs-typography-metadata-lineHeight);
  text-align: center;
}}
.cs-drop {{
  min-height: 166px;
  border: 1px dashed var(--cs-color-border-strong);
  border-radius: var(--cs-radius-lg);
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--cs-color-primary-50) 34%, var(--cs-color-surface-primary)), var(--cs-color-surface-primary));
  display: grid;
  gap: var(--cs-space-3);
  padding: var(--cs-space-4);
  align-content: center;
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
  gap: var(--cs-space-2);
  place-items: center;
  text-align: center;
  padding: 0;
}}
.cs-drop-mark {{
  width: 46px;
  height: 46px;
  border-radius: var(--cs-radius-pill);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
  border: 1px solid var(--cs-color-primary-100);
  font-size: 18px;
}}
.cs-drop textarea.cs-drop-input {{
  min-height: 44px;
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
  padding: var(--cs-space-2) var(--cs-space-3);
  display: grid;
  grid-template-columns: auto auto minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--cs-space-3);
  box-shadow: var(--cs-shadow-sm);
}}
.cs-ask-mark {{
  width: 34px;
  height: 34px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-ask-bar .cs-field {{
  border: 0;
  padding: var(--cs-space-2);
  box-shadow: none;
}}
.cs-ask-bar .cs-field:focus {{ box-shadow: none; }}
.cs-suggestion-row {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--cs-space-2); }}
.cs-suggestion-row .cs-button {{ min-width: 0; justify-content: center; white-space: normal; }}
.cs-home-loop-inline {{
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: var(--cs-space-2);
  max-width: 460px;
}}
.cs-home-loop-inline span {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-text-secondary);
  padding: var(--cs-space-1) var(--cs-space-2);
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: var(--cs-typography-metadata-lineHeight);
  font-weight: var(--cs-typography-weight-medium);
  white-space: nowrap;
}}
.cs-home-loop-inline span:first-child {{
  border-color: var(--cs-color-primary-100);
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
}}
.cs-home-loop-inline strong {{
  color: var(--cs-color-text-primary);
}}
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
.cs-activity-list {{
  display: grid;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  overflow: hidden;
  background: var(--cs-color-surface-primary);
}}
.cs-activity-row {{
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
  min-height: 74px;
  padding: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-activity-row:last-child {{ border-bottom: 0; }}
.cs-activity-icon {{
  width: 30px;
  height: 30px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-activity-row strong, .cs-activity-row p {{ margin: 0; }}
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
  grid-template-columns: minmax(0, 1fr) minmax(300px, 340px);
  gap: var(--cs-space-5);
  align-items: start;
}}
.cs-search-main {{
  display: grid;
  gap: var(--cs-space-4);
  min-width: 0;
}}
.cs-search-canvas {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-search-command {{
  display: grid;
  gap: var(--cs-space-2);
  align-items: start;
}}
.cs-search-back {{
  color: var(--cs-color-primary-700);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-search-copy {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-search-titleline {{
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: var(--cs-space-3);
}}
.cs-search-copy h1 {{
  margin: 0;
  max-width: 34ch;
  font-size: 26px;
  line-height: 1.18;
  text-wrap: balance;
}}
.cs-search-mode {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-search-hero {{
  border: 1px solid var(--cs-color-border-focus);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  box-shadow: var(--cs-shadow-focus);
  padding: var(--cs-space-2);
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: var(--cs-space-2);
  align-items: center;
}}
.cs-search-lens {{
  min-width: 42px;
  min-height: 42px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-search-hero input {{
  min-height: 50px;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--cs-color-text-primary);
  padding: 0 var(--cs-space-2);
  font-size: 15px;
}}
.cs-search-submit {{
  min-width: 44px;
  min-height: 44px;
  justify-content: center;
}}
.cs-search-tabs, .cs-filter-row {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
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
.cs-search-filterbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
}}
.cs-search-context {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-search-context h2 {{
  margin: 0;
  font-size: var(--cs-typography-label-fontSize);
  line-height: var(--cs-typography-label-lineHeight);
  color: var(--cs-color-text-muted);
}}
.cs-result-list {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-result-list-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-3);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-result-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: 44px minmax(0, 1fr) minmax(150px, 210px);
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-result-row:hover {{ border-color: var(--cs-color-border-strong); box-shadow: var(--cs-shadow-sm); }}
.cs-result-icon {{
  width: 34px;
  min-height: 38px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-size: 11px;
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-result-icon.is-source {{ background: var(--cs-color-primary-50); color: var(--cs-color-primary-700); }}
.cs-result-icon.is-brief {{ background: var(--cs-state-underReview-bg); color: var(--cs-state-underReview-text); }}
.cs-result-icon.is-claim {{ background: var(--cs-state-searchable-bg); color: var(--cs-state-searchable-text); }}
.cs-result-icon.is-action {{ background: var(--cs-state-draft-bg); color: var(--cs-state-draft-text); }}
.cs-result-body {{ display: grid; gap: var(--cs-space-1); }}
.cs-result-body h3 {{ margin: 0; font-size: 16px; line-height: 1.35; }}
.cs-result-body h3 a {{ color: var(--cs-color-text-primary); }}
.cs-result-body h3 a:hover, .cs-result-body h3 a:focus-visible {{ color: var(--cs-color-primary-700); outline: none; }}
.cs-result-body p {{ margin: 0; color: var(--cs-color-text-secondary); max-width: 78ch; }}
.cs-result-meta {{ display: flex; flex-wrap: wrap; gap: var(--cs-space-2); color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-result-type {{
  color: var(--cs-color-text-secondary);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-result-support {{
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-result-support .cs-meta {{
  line-height: 1.35;
  text-align: right;
  max-width: 180px;
}}
.cs-result-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
}}
.cs-result-actions .cs-button {{ min-height: 34px; padding: var(--cs-space-1) var(--cs-space-3); }}
.cs-search-rail {{
  position: sticky;
  top: calc(var(--cs-space-4) + 72px);
}}
.cs-right-stat {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
  padding: var(--cs-space-2) 0;
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-right-stat:last-child {{ border-bottom: 0; }}
.cs-right-stat-label {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-right-stat-icon {{
  width: 24px;
  height: 24px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-size: 10px;
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-suggested-query {{
  display: grid;
  grid-template-columns: 22px minmax(0, 1fr);
  gap: var(--cs-space-2);
  align-items: start;
  color: var(--cs-color-text-secondary);
  padding: var(--cs-space-2) 0;
}}
.cs-suggested-query span:first-child {{
  width: 20px;
  height: 20px;
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  display: grid;
  place-items: center;
  font-weight: var(--cs-typography-weight-semibold);
  font-size: var(--cs-typography-metadata-fontSize);
}}
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
.cs-artifact-workbench {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 400px);
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-artifact-compact-hero {{
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
  padding-bottom: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-artifact-compact-hero > * {{
  min-width: 0;
}}
.cs-artifact-compact-hero .cs-artifact-title h1 {{
  max-width: 44ch;
  font-size: 27px;
  line-height: 1.12;
  text-wrap: balance;
  overflow-wrap: anywhere;
}}
.cs-artifact-compact-hero .cs-artifact-actions {{
  justify-content: flex-start;
  max-width: 100%;
}}
.cs-artifact-breadcrumb {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  align-items: center;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  min-width: 0;
}}
.cs-artifact-breadcrumb a {{
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-artifact-breadcrumb span:last-child {{
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.cs-artifact-title-row {{
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-artifact-file-mark {{
  width: 46px;
  height: 46px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-600);
  color: var(--cs-color-text-inverse);
  font-size: var(--cs-typography-label-fontSize);
  font-weight: var(--cs-typography-weight-bold);
}}
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
.cs-metadata-strip.is-artifact {{
  grid-template-columns: repeat(5, minmax(0, 1fr));
  border-bottom: 1px solid var(--cs-color-border-default);
  padding: var(--cs-space-3) 0;
  margin-bottom: 0;
}}
.cs-metadata-item strong {{
  word-break: break-word;
}}
.cs-artifact-inspection-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-4);
}}
.cs-artifact-inspection-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-artifact-inspection-card strong {{
  font-size: 18px;
  line-height: 1.25;
}}
.cs-artifact-viewer {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  overflow: hidden;
  box-shadow: var(--cs-shadow-sm);
  margin-top: var(--cs-space-2);
}}
.cs-artifact-toolbar {{
  min-height: 48px;
  border-bottom: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-primary);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-2);
  padding: var(--cs-space-2) var(--cs-space-3);
}}
.cs-artifact-toolbar-label {{
  display: grid;
  gap: 2px;
}}
.cs-artifact-toolgroup {{
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-artifact-tool {{
  min-width: 32px;
  height: 32px;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  display: grid;
  place-items: center;
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-artifact-tool.is-muted {{
  color: var(--cs-color-text-muted);
  background: var(--cs-color-surface-subtle);
}}
.cs-artifact-page-count {{
  min-width: 72px;
  height: 32px;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-document-frame {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-4);
}}
.cs-document-frame.has-rail {{
  border: 0;
  border-radius: 0;
  padding: 0;
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr);
  min-height: 600px;
}}
.cs-artifact-page-rail {{
  border-right: 1px solid var(--cs-color-border-default);
  background: color-mix(in srgb, var(--cs-color-surface-subtle) 74%, var(--cs-color-surface-primary));
  padding: var(--cs-space-3) var(--cs-space-2);
  display: grid;
  align-content: start;
  gap: var(--cs-space-2);
}}
.cs-artifact-page-rail-label {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-artifact-thumb {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  min-height: 92px;
  padding: var(--cs-space-2);
  display: grid;
  gap: var(--cs-space-1);
  color: var(--cs-color-text-muted);
  font-size: 10px;
}}
.cs-artifact-thumb.is-active {{
  border-color: var(--cs-color-primary-500);
  box-shadow: var(--cs-shadow-focus);
}}
.cs-artifact-thumb-line {{
  height: 5px;
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-border-default);
}}
.cs-artifact-thumb span {{
  text-align: center;
  margin-top: var(--cs-space-1);
}}
.cs-artifact-page-area {{
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--cs-color-surface-subtle) 72%, var(--cs-color-surface-primary)), var(--cs-color-surface-subtle));
  padding: var(--cs-space-4);
  overflow: auto;
}}
.cs-document-page {{
  max-width: 760px;
  min-height: 540px;
  margin: 0 auto;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  box-shadow: 0 18px 40px rgba(15, 23, 42, .08);
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
.cs-artifact-source-note {{
  display: flex;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
  align-items: center;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-document-page .cs-source-text {{
  border: 0;
  border-radius: 0;
  background: transparent;
  padding: 0;
  line-height: 1.75;
}}
.cs-artifact-rail {{
  position: sticky;
  top: calc(var(--cs-space-4) + 72px);
}}
.cs-artifact-rail-tabs {{
  display: flex;
  gap: var(--cs-space-5);
  border-bottom: 1px solid var(--cs-color-border-default);
  margin-bottom: var(--cs-space-4);
  overflow-x: auto;
}}
.cs-artifact-rail-tab {{
  padding: 0 0 var(--cs-space-3);
  color: var(--cs-color-text-secondary);
  font-weight: var(--cs-typography-weight-semibold);
  white-space: nowrap;
}}
.cs-artifact-rail-tab.is-active {{
  color: var(--cs-color-primary-700);
  border-bottom: 2px solid var(--cs-color-primary-600);
}}
.cs-artifact-panel-list {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-artifact-summary-lead {{
  border: 1px solid var(--cs-color-primary-100);
  border-radius: var(--cs-radius-md);
  background: color-mix(in srgb, var(--cs-color-primary-50) 42%, var(--cs-color-surface-primary));
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-artifact-summary-lead strong {{
  color: var(--cs-color-text-primary);
  line-height: 1.35;
}}
.cs-artifact-summary-lead p {{
  line-height: 1.65;
}}
.cs-artifact-side-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-artifact-side-card h2 {{
  margin: 0;
}}
.cs-artifact-side-card p {{
  margin: 0;
}}
.cs-artifact-side-card summary {{
  cursor: pointer;
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-artifact-side-card[open] summary {{
  margin-bottom: var(--cs-space-2);
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
.cs-inbox-lane-summary {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
  margin-bottom: var(--cs-space-3);
  color: var(--cs-color-text-secondary);
}}
.cs-inbox-summary-main {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-inbox-summary-main strong {{
  color: var(--cs-color-text-primary);
  font-size: var(--cs-typography-body-fontSize);
  font-variant-numeric: tabular-nums;
}}
.cs-inbox-summary-pills {{
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-inbox-summary-pill {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-surface-primary);
  padding: 4px var(--cs-space-2);
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-inbox-summary-pill.is-active {{
  border-color: var(--cs-color-border-focus);
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
}}
.cs-inbox-summary-pill strong {{
  color: var(--cs-color-text-primary);
  font-variant-numeric: tabular-nums;
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
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
  margin-bottom: var(--cs-space-4);
}}
.cs-inbox-toolbar .cs-filter-row {{ margin-top: 0; }}
.cs-inbox-filter-label {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  min-height: 34px;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: 0 var(--cs-space-3);
  color: var(--cs-color-text-primary);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-inbox-filter-label span {{
  color: var(--cs-color-primary-700);
}}
.cs-inbox-table {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  overflow: hidden;
}}
.cs-inbox-head, .cs-inbox-row {{
  display: grid;
  grid-template-columns: 28px minmax(210px, 1.45fr) minmax(82px, .55fr) minmax(78px, .55fr) minmax(88px, .55fr) minmax(76px, .5fr) minmax(106px, .68fr);
  gap: var(--cs-space-2);
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
.cs-inbox-select {{
  width: 16px;
  height: 16px;
  border: 1px solid var(--cs-color-border-strong);
  border-radius: var(--cs-radius-xs);
  display: grid;
  place-items: center;
  color: var(--cs-color-surface-primary);
  font-size: 11px;
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-inbox-row.is-selected .cs-inbox-select {{
  border-color: var(--cs-color-primary-600);
  background: var(--cs-color-primary-600);
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
.cs-inbox-type-cell {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-secondary);
}}
.cs-inbox-type-mark {{
  width: 20px;
  height: 20px;
  border-radius: var(--cs-radius-xs);
  display: grid;
  place-items: center;
  border: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-subtle);
  color: var(--cs-color-primary-700);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-inbox-owner {{
  display: inline-grid;
  grid-template-columns: 24px minmax(0, 1fr);
  gap: var(--cs-space-2);
  align-items: center;
  min-width: 0;
}}
.cs-inbox-owner-mark {{
  width: 22px;
  height: 22px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-surface-subtle);
  border: 1px solid var(--cs-color-border-default);
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-inbox-detail {{
  display: grid;
  gap: var(--cs-space-4);
}}
.cs-inbox-detail h2 {{ margin: 0; font-size: var(--cs-typography-sectionTitle-fontSize); }}
.cs-inbox-detail-title {{
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-inbox-close {{
  width: 28px;
  height: 28px;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  color: var(--cs-color-text-muted);
  background: var(--cs-color-surface-primary);
}}
.cs-inbox-action-panel {{
  border: 1px solid var(--cs-color-border-focus);
  border-radius: var(--cs-radius-md);
  background: linear-gradient(180deg, var(--cs-color-primary-50), var(--cs-color-surface-primary));
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-inbox-action-panel .cs-section-title {{ margin: 0; }}
.cs-inbox-preview-note {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-inbox-preview-note h3 {{
  margin: 0;
  font-size: var(--cs-typography-label-fontSize);
  line-height: var(--cs-typography-label-lineHeight);
}}
.cs-inbox-preview-note p {{ margin: 0; color: var(--cs-color-text-secondary); }}
.cs-journey-timeline {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-3);
  min-width: 0;
}}
.cs-journey-timeline.is-recovery {{
  border-color: var(--cs-state-underReview-border);
  background: var(--cs-state-underReview-bg);
}}
.cs-journey-header {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
  min-width: 0;
}}
.cs-journey-header h3 {{
  margin: 0;
  font-size: var(--cs-typography-sectionTitle-fontSize);
  line-height: var(--cs-typography-sectionTitle-lineHeight);
}}
.cs-journey-header p {{ margin: var(--cs-space-1) 0 0; color: var(--cs-color-text-secondary); }}
.cs-journey-stage-list {{
  list-style: none;
  margin: 0;
  padding: 0;
  min-width: 0;
}}
.cs-journey-stage-list > li {{ min-width: 0; }}
.cs-journey-stage {{
  border-left: 3px solid var(--cs-color-border-strong);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  min-width: 0;
}}
.cs-journey-stage.is-ready {{
  border-left-color: var(--cs-state-saved-border);
  background: var(--cs-state-saved-bg);
}}
.cs-journey-stage.is-needs-review {{
  border-left-color: var(--cs-state-underReview-border);
  background: var(--cs-state-underReview-bg);
}}
.cs-journey-stage.is-blocked {{
  border-left-color: var(--cs-state-policyBlocked-border);
  background: var(--cs-state-policyBlocked-bg);
}}
.cs-journey-stage.is-ready .cs-dot {{ background: var(--cs-state-saved-fg); }}
.cs-journey-stage.is-needs-review .cs-dot {{ background: var(--cs-state-underReview-fg); }}
.cs-journey-stage.is-blocked .cs-dot {{ background: var(--cs-state-policyBlocked-fg); }}
.cs-journey-stage-body {{ display: grid; gap: var(--cs-space-2); min-width: 0; }}
.cs-journey-stage-body p {{ margin: 0; color: var(--cs-color-text-secondary); overflow-wrap: anywhere; }}
.cs-journey-stage-heading {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-journey-ref-grid {{
  margin: var(--cs-space-2) 0 0;
  display: grid;
  grid-template-columns: minmax(88px, auto) minmax(0, 1fr);
  gap: var(--cs-space-1) var(--cs-space-2);
}}
.cs-journey-ref-grid dt {{ color: var(--cs-color-text-muted); }}
.cs-journey-ref-grid dd {{ margin: 0; min-width: 0; overflow-wrap: anywhere; word-break: break-word; }}
.cs-journey-ref-grid code {{ display: inline-block; max-width: 100%; overflow-wrap: anywhere; word-break: break-word; }}
.cs-journey-recovery {{ border-top: 1px solid var(--cs-color-border-default); padding-top: var(--cs-space-2); }}
.cs-journey-recovery summary {{
  cursor: pointer;
  min-height: 32px;
  padding: var(--cs-space-1) 0;
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-journey-recovery summary:focus-visible,
.cs-journey-stage summary:focus-visible {{
  outline: 2px solid var(--cs-color-border-focus);
  outline-offset: 2px;
}}
.cs-journey-recovery-list {{ display: grid; gap: var(--cs-space-2); margin-top: var(--cs-space-2); }}
.cs-journey-recovery-list > div {{
  border-left: 2px solid var(--cs-state-underReview-border);
  padding-left: var(--cs-space-2);
  display: grid;
  gap: var(--cs-space-1);
  color: var(--cs-color-text-secondary);
}}
.cs-journey-recovery-list strong {{ color: var(--cs-color-text-primary); }}
.cs-inbox-linked-list {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-inbox-linked-row {{
  display: grid;
  grid-template-columns: 24px minmax(0, 1fr);
  gap: var(--cs-space-2);
  align-items: start;
  color: var(--cs-color-text-secondary);
}}
.cs-inbox-linked-row strong {{
  display: block;
  color: var(--cs-color-text-primary);
}}
.cs-inbox-receipt-strip {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-2);
}}
.cs-inbox-receipt {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2);
  display: grid;
  gap: 2px;
}}
.cs-inbox-receipt strong {{ font-size: var(--cs-typography-metadata-fontSize); }}
.cs-inbox-receipt span {{ color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-inbox-actions {{ display: grid; gap: var(--cs-space-2); }}
.cs-inbox-actions .cs-button {{ justify-content: center; text-align: center; }}
.cs-inbox-foot {{
  padding: var(--cs-space-3) var(--cs-space-4);
  border-top: 1px solid var(--cs-color-border-default);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
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
  grid-template-columns: 38px minmax(0, 1fr) minmax(176px, auto);
  gap: var(--cs-space-4);
  align-items: start;
  transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}}
.cs-collection-row:hover {{
  border-color: var(--cs-color-border-strong);
  box-shadow: var(--cs-shadow-sm);
  transform: translateY(-1px);
}}
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
.cs-collection-body {{ display: grid; gap: var(--cs-space-2); }}
.cs-collection-body h3 {{ margin: 0; font-size: 16px; line-height: 1.35; }}
.cs-collection-body p {{ margin: 0; color: var(--cs-color-text-secondary); max-width: 82ch; }}
.cs-collection-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-collection-actions {{
  display: grid;
  gap: var(--cs-space-3);
  justify-items: end;
  align-content: start;
}}
.cs-collection-actions .cs-row {{ justify-content: flex-end; }}
.cs-collection-cta {{
  min-height: 32px;
  border-radius: var(--cs-radius-sm);
  border: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-subtle);
  padding: 6px 10px;
  color: var(--cs-color-text-primary);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-collection-footrail {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-2);
  margin-top: var(--cs-space-1);
}}
.cs-collection-stage {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-2);
  display: grid;
  gap: 2px;
  min-width: 0;
}}
.cs-collection-stage strong {{
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: var(--cs-typography-metadata-lineHeight);
  color: var(--cs-color-text-primary);
}}
.cs-collection-stage span {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  overflow-wrap: anywhere;
}}
.cs-queue-focus {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-2);
  margin-bottom: var(--cs-space-3);
}}
.cs-queue-focus-head {{
  display: flex;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
  align-items: center;
}}
.cs-queue-focus h2 {{
  margin: 0;
  font-size: 17px;
  line-height: 1.35;
}}
.cs-queue-focus p {{
  margin: 2px 0 0;
  color: var(--cs-color-text-secondary);
  max-width: 72ch;
}}
.cs-queue-lanes {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
  padding-top: var(--cs-space-2);
  border-top: 1px solid var(--cs-color-border-default);
}}
.cs-queue-lane {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-surface-subtle);
  padding: 4px var(--cs-space-2);
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-secondary);
}}
.cs-queue-lane strong {{
  color: var(--cs-color-text-primary);
  font-size: var(--cs-typography-body-fontSize);
  font-variant-numeric: tabular-nums;
}}
.cs-queue-lane span {{
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
.cs-empty-state {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background:
    linear-gradient(135deg, var(--cs-color-surface-primary), var(--cs-color-surface-subtle));
  padding: var(--cs-space-6);
  display: grid;
  gap: var(--cs-space-4);
  color: var(--cs-color-text-primary);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.68);
}}
.cs-empty-state-main {{
  display: grid;
  grid-template-columns: 44px minmax(0, 1fr);
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-empty-mark {{
  width: 44px;
  height: 44px;
  border-radius: var(--cs-radius-lg);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-empty-copy {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-empty-copy h2 {{
  margin: 0;
  font-size: 20px;
  line-height: 1.3;
  text-wrap: balance;
}}
.cs-empty-copy p {{
  margin: 0;
  color: var(--cs-color-text-secondary);
  max-width: 64ch;
  text-wrap: pretty;
}}
.cs-empty-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
}}
.cs-empty-steps {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-empty-step {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: color-mix(in srgb, var(--cs-color-surface-primary) 72%, var(--cs-color-surface-subtle));
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-empty-briefing {{
  border-top: 1px solid var(--cs-color-border-default);
  padding-top: var(--cs-space-4);
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(220px, 280px);
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-empty-briefing h3 {{
  margin: 0 0 var(--cs-space-2);
  font-size: var(--cs-typography-label-fontSize);
  line-height: var(--cs-typography-label-lineHeight);
}}
.cs-empty-receipts {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-empty-receipt {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2);
  display: grid;
  gap: 2px;
}}
.cs-empty-receipt strong {{
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: var(--cs-typography-metadata-lineHeight);
}}
.cs-empty-receipt span {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
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
.cs-detail-orientation {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  margin-bottom: var(--cs-space-4);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-4);
  align-items: center;
}}
.cs-detail-context {{
  display: grid;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-detail-path {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-detail-path a {{
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-detail-path span[aria-hidden="true"] {{ color: var(--cs-color-text-muted); }}
.cs-detail-summary {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-detail-summary-head {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--cs-space-2);
}}
.cs-detail-current {{
  color: var(--cs-color-text-primary);
  font-weight: var(--cs-typography-weight-semibold);
  overflow-wrap: anywhere;
}}
.cs-detail-summary p {{
  margin: 0;
  color: var(--cs-color-text-secondary);
  max-width: 68ch;
  text-wrap: pretty;
}}
.cs-detail-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  justify-content: flex-end;
}}
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
.cs-citation-rail {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-citation-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: color-mix(in srgb, var(--cs-color-surface-primary) 82%, var(--cs-color-primary-50));
  overflow: hidden;
}}
.cs-citation-card summary {{
  cursor: pointer;
  list-style: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-3);
  padding: var(--cs-space-3);
}}
.cs-citation-card summary::-webkit-details-marker {{ display: none; }}
.cs-citation-title {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-citation-title strong {{
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.cs-citation-action {{
  color: var(--cs-color-primary-700);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
  flex: 0 0 auto;
}}
.cs-citation-body {{
  border-top: 1px solid var(--cs-color-border-default);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-citation-snippet {{
  margin: 0;
  color: var(--cs-color-text-secondary);
  text-wrap: pretty;
}}
.cs-citation-meta {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--cs-space-2);
}}
.cs-citation-meta div {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2);
  min-width: 0;
}}
.cs-citation-meta strong {{
  display: block;
  overflow-wrap: anywhere;
}}
.cs-citation-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
}}
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
.cs-brief-workbench {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 380px);
  gap: var(--cs-space-6);
  align-items: start;
}}
.cs-brief-workbench > *, .cs-brief-workbench .cs-stack, .cs-brief-titlebar, .cs-brief-titlebar > *, .cs-brief-heading-row {{
  min-width: 0;
  max-width: 100%;
}}
.cs-brief-titlebar {{
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: var(--cs-space-2);
  align-items: start;
  padding-bottom: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
  margin-bottom: var(--cs-space-3);
}}
.cs-brief-titlebar .cs-brief-actions {{
  justify-content: flex-start;
}}
.cs-brief-breadcrumb {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  align-items: center;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  min-width: 0;
}}
.cs-brief-breadcrumb a {{
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-brief-breadcrumb span:last-child {{
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.cs-brief-heading-row {{
  display: flex;
  align-items: center;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
}}
.cs-brief-titlebar h1 {{
  margin: 0;
  flex: 1 1 24rem;
  min-width: 0;
  max-width: 58ch;
  font-size: 28px;
  line-height: 1.16;
  text-wrap: balance;
  overflow-wrap: anywhere;
}}
.cs-brief-titlebar p {{
  max-width: 68ch;
}}
.cs-brief-fact-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-2);
}}
.cs-brief-fact {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2);
  display: grid;
  gap: 2px;
}}
.cs-brief-fact strong {{ font-size: var(--cs-typography-body-fontSize); line-height: var(--cs-typography-body-lineHeight); }}
.cs-brief-receipt-panel {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--cs-color-primary-50) 50%, var(--cs-color-surface-primary)), var(--cs-color-surface-primary) 58%);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-3);
}}
.cs-brief-lead-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(260px, .75fr);
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-brief-answer-card, .cs-brief-receipt-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: color-mix(in srgb, var(--cs-color-surface-primary) 88%, white);
  padding: var(--cs-space-3);
  display: grid;
  align-content: start;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-brief-answer-card.is-primary {{
  min-height: 100%;
  border-color: var(--cs-color-primary-100);
  background: var(--cs-color-surface-primary);
}}
.cs-brief-answer-card p, .cs-brief-receipt-card p {{
  margin: 0;
  color: var(--cs-color-text-secondary);
  line-height: 1.55;
  text-wrap: pretty;
}}
.cs-brief-answer-card p {{
  color: var(--cs-color-text-primary);
  font-size: 16px;
  line-height: 1.65;
}}
.cs-brief-receipt-stack {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-brief-receipt-card strong {{
  color: var(--cs-color-text-primary);
  font-size: var(--cs-typography-body-fontSize);
  line-height: var(--cs-typography-body-lineHeight);
  overflow-wrap: anywhere;
}}
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
  border-radius: var(--cs-radius-pill);
  border: 2px solid var(--cs-color-border-strong);
  background: var(--cs-color-surface-primary);
}}
.cs-trust-step.is-active {{
  border-color: var(--cs-state-evidenceBacked-border);
  background: var(--cs-state-evidenceBacked-bg);
}}
.cs-trust-step.is-active strong::before {{ border-color: var(--cs-state-evidenceBacked-fg); background: var(--cs-state-evidenceBacked-fg); }}
.cs-trust-step.is-locked {{ opacity: .76; }}
.cs-claim-workbench {{
  grid-template-columns: minmax(0, 1fr) minmax(340px, 400px);
}}
.cs-claim-workbench, .cs-claim-workbench > *, .cs-claim-workbench .cs-stack, .cs-claim-hero, .cs-claim-hero > *, .cs-claim-titlebar, .cs-claim-titlebar > *, .cs-claim-heading-row {{
  min-width: 0;
  max-width: 100%;
}}
.cs-claim-hero, .cs-claim-titlebar, .cs-claim-titlebar > *, .cs-claim-actions {{
  width: 100%;
}}
.cs-claim-workbench .cs-stack, .cs-claim-hero, .cs-claim-titlebar, .cs-claim-titlebar .cs-brief-title {{
  grid-template-columns: minmax(0, 1fr);
}}
.cs-claim-hero {{
  display: grid;
  gap: var(--cs-space-4);
  margin-bottom: var(--cs-space-4);
}}
.cs-claim-hero.is-compact {{
  padding-bottom: var(--cs-space-4);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-claim-breadcrumb {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  min-width: 0;
}}
.cs-claim-breadcrumb a {{ color: var(--cs-color-primary-700); font-weight: var(--cs-typography-weight-semibold); }}
.cs-claim-breadcrumb span:last-child {{
  flex: 1 1 180px;
  min-width: 0;
  max-width: 100%;
  overflow-wrap: anywhere;
}}
.cs-claim-titlebar {{
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-claim-heading-row {{
  display: flex;
  align-items: center;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
}}
.cs-claim-titlebar h1 {{
  margin: 0;
  flex: 1 1 22rem;
  min-width: 0;
  max-width: 44ch;
  font-size: 28px;
  line-height: 1.16;
  text-wrap: balance;
  overflow-wrap: anywhere;
}}
.cs-claim-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  justify-content: flex-end;
  width: 100%;
  max-width: 100%;
  min-width: 0;
}}
.cs-button.is-disabled {{
  cursor: not-allowed;
  opacity: .68;
  background: var(--cs-color-surface-subtle);
  color: var(--cs-color-text-muted);
  border-color: var(--cs-color-border-default);
  box-shadow: none;
}}
.cs-button.is-disabled:hover {{ transform: none; box-shadow: none; }}
.cs-claim-progress {{
  position: relative;
  border: 0;
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3) var(--cs-space-4);
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-claim-progress::before {{
  content: "";
  position: absolute;
  left: var(--cs-space-8);
  right: var(--cs-space-8);
  top: 24px;
  border-top: 1px dashed var(--cs-color-border-strong);
}}
.cs-claim-progress-step {{
  position: relative;
  z-index: 1;
  display: grid;
  justify-items: center;
  gap: var(--cs-space-2);
  text-align: center;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-claim-dot {{
  width: 16px;
  height: 16px;
  border-radius: var(--cs-radius-pill);
  border: 2px solid var(--cs-color-border-strong);
  background: var(--cs-color-surface-primary);
}}
.cs-claim-progress-step.is-active {{ color: var(--cs-color-text-primary); font-weight: var(--cs-typography-weight-semibold); }}
.cs-claim-progress-step.is-active .cs-claim-dot {{
  border-color: var(--cs-color-primary-600);
  background: var(--cs-color-primary-600);
  box-shadow: 0 0 0 4px var(--cs-color-primary-100);
}}
.cs-claim-pathbar {{
  display: grid;
  grid-template-columns: minmax(170px, .32fr) minmax(0, 1fr);
  gap: var(--cs-space-4);
  align-items: center;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  margin-bottom: var(--cs-space-4);
}}
.cs-claim-pathbar-title {{
  display: grid;
  gap: var(--cs-space-1);
  min-width: 0;
}}
.cs-claim-pathbar-title strong {{
  color: var(--cs-color-text-primary);
  font-size: var(--cs-typography-body-fontSize);
  line-height: var(--cs-typography-body-lineHeight);
}}
.cs-claim-pathbar .cs-claim-progress {{
  background: transparent;
  border-radius: 0;
  padding: var(--cs-space-1) 0;
}}
.cs-claim-pathbar .cs-claim-progress::before {{
  left: 8%;
  right: 8%;
  top: 14px;
}}
.cs-claim-pathbar .cs-claim-progress-step {{
  gap: var(--cs-space-1);
}}
.cs-claim-pathbar .cs-claim-dot {{
  width: 14px;
  height: 14px;
}}
.cs-claim-tabs {{
  display: flex;
  gap: var(--cs-space-5);
  border-bottom: 1px solid var(--cs-color-border-default);
  margin-bottom: var(--cs-space-4);
  overflow-x: auto;
}}
.cs-claim-tab {{
  padding: 0 0 var(--cs-space-3);
  color: var(--cs-color-text-secondary);
  font-weight: var(--cs-typography-weight-semibold);
  white-space: nowrap;
}}
.cs-claim-tab.is-active {{
  color: var(--cs-color-primary-700);
  border-bottom: 2px solid var(--cs-color-primary-600);
}}
.cs-claim-form-card {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-claim-review-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-top: var(--cs-space-3);
}}
.cs-claim-review-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-claim-review-card strong {{
  font-size: var(--cs-typography-body-fontSize);
  line-height: var(--cs-typography-body-lineHeight);
}}
.cs-claim-field {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-claim-field.is-primary {{
  border-color: var(--cs-color-primary-100);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--cs-color-primary-500) 18%, transparent);
}}
.cs-claim-field-head {{
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-claim-text {{
  margin: 0;
  color: var(--cs-color-text-primary);
  font-size: 15px;
  line-height: 1.7;
}}
.cs-claim-text.is-statement {{
  font-size: 16px;
  line-height: 1.75;
}}
.cs-claim-field-foot {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  text-align: right;
}}
.cs-claim-taxonomy {{
  display: grid;
  grid-template-columns: minmax(180px, .42fr) minmax(0, 1fr);
  gap: var(--cs-space-3);
}}
.cs-claim-frameworks {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  min-height: 42px;
  padding: var(--cs-space-2) var(--cs-space-3);
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-claim-select, .cs-claim-tags {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  min-height: 42px;
  padding: var(--cs-space-2) var(--cs-space-3);
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-claim-footrail {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-claim-footrail div {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-claim-footrail strong {{
  color: var(--cs-color-text-primary);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-claim-save-note {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-state-evidenceBacked-text);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-medium);
}}
.cs-claim-save-note::before {{
  content: "";
  width: 8px;
  height: 8px;
  border-radius: var(--cs-radius-pill);
  background: var(--cs-state-evidenceBacked-fg);
}}
.cs-claim-control-list {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-claim-control-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-claim-control-mark {{
  width: 26px;
  height: 26px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-surface-subtle);
  color: var(--cs-color-text-secondary);
  border: 1px solid var(--cs-color-border-default);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-claim-control-row.is-ready .cs-claim-control-mark {{
  background: var(--cs-state-evidenceBacked-bg);
  color: var(--cs-state-evidenceBacked-fg);
  border-color: var(--cs-state-evidenceBacked-border);
}}
.cs-claim-control-row.is-review .cs-claim-control-mark {{
  background: var(--cs-state-underReview-bg);
  color: var(--cs-state-underReview-fg);
  border-color: var(--cs-state-underReview-border);
}}
.cs-claim-control-row strong, .cs-claim-control-row p {{ margin: 0; }}
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
.cs-evidence-toolbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
  margin-bottom: var(--cs-space-3);
}}
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
.cs-evidence-row.is-selected {{
  background: color-mix(in srgb, var(--cs-state-underReview-bg) 42%, var(--cs-color-surface-primary));
  border-color: var(--cs-state-underReview-border);
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
.cs-action-workbench {{
  grid-template-columns: minmax(0, 1fr) minmax(340px, 400px);
}}
.cs-action-hero {{
  display: grid;
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-3);
  padding-bottom: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-action-breadcrumb {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  align-items: center;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-action-breadcrumb a {{
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-action-titlebar {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-action-titlebar h1 {{
  margin: 0;
  max-width: 44ch;
  font-size: 28px;
  line-height: 1.14;
  text-wrap: balance;
}}
.cs-action-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  justify-content: flex-end;
}}
.cs-action-rail {{
  position: sticky;
  top: calc(var(--cs-space-4) + 72px);
}}
.cs-action-review-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin: var(--cs-space-3) 0 0;
}}
.cs-action-review-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-action-review-card strong {{ font-size: var(--cs-typography-body-fontSize); line-height: var(--cs-typography-body-lineHeight); }}
.cs-action-receipt-panel {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--cs-color-primary-50) 46%, var(--cs-color-surface-primary)), var(--cs-color-surface-primary) 62%);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-4);
}}
.cs-action-receipt-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-action-receipt-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: color-mix(in srgb, var(--cs-color-surface-primary) 90%, white);
  padding: var(--cs-space-3);
  display: grid;
  align-content: start;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-action-receipt-card strong {{
  color: var(--cs-color-text-primary);
  line-height: 1.4;
  overflow-wrap: anywhere;
}}
.cs-action-receipt-card p {{
  margin: 0;
  color: var(--cs-color-text-secondary);
  line-height: 1.55;
  text-wrap: pretty;
}}
.cs-action-mini-diff {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  overflow: hidden;
  display: grid;
}}
.cs-action-mini-diff div {{
  display: grid;
  gap: var(--cs-space-1);
  padding: var(--cs-space-2);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-action-mini-diff div:last-child {{ border-bottom: 0; }}
.cs-action-route-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-top: var(--cs-space-3);
}}
.cs-action-route-step {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-action-route-step.is-current {{
  border-color: var(--cs-color-border-focus);
  background: linear-gradient(180deg, var(--cs-color-primary-50), var(--cs-color-surface-primary));
  box-shadow: inset 3px 0 0 var(--cs-color-primary-600);
}}
.cs-action-route-top {{
  display: flex;
  gap: var(--cs-space-2);
  align-items: center;
}}
.cs-action-route-index {{
  width: 24px;
  height: 24px;
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  display: grid;
  place-items: center;
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-action-route-step p {{
  margin: 0;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: 1.35;
}}
.cs-owner-overview {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0;
  margin-bottom: var(--cs-space-5);
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  overflow: hidden;
}}
.cs-owner-tabs {{
  display: flex;
  gap: var(--cs-space-5);
  border-bottom: 1px solid var(--cs-color-border-default);
  margin: var(--cs-space-4) 0 var(--cs-space-5);
  overflow-x: auto;
}}
.cs-owner-tab {{
  padding: 0 0 var(--cs-space-3);
  color: var(--cs-color-text-secondary);
  font-weight: var(--cs-typography-weight-semibold);
  white-space: nowrap;
}}
.cs-owner-tab.is-active {{
  color: var(--cs-color-primary-700);
  border-bottom: 2px solid var(--cs-color-primary-600);
}}
.cs-owner-metric {{
  border-right: 1px solid var(--cs-color-border-default);
  background: linear-gradient(180deg, var(--cs-color-surface-primary), var(--cs-color-surface-subtle));
  padding: var(--cs-space-3) var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-owner-metric:last-child {{ border-right: 0; }}
.cs-owner-metric strong {{
  font-size: var(--cs-typography-sectionTitle-fontSize);
  line-height: var(--cs-typography-sectionTitle-lineHeight);
}}
.cs-reference-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--cs-space-4);
}}
.cs-reference-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  overflow: hidden;
  display: grid;
  min-width: 0;
}}
.cs-reference-card img {{
  width: 100%;
  aspect-ratio: 16 / 10;
  object-fit: cover;
  object-position: top left;
  border-bottom: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-subtle);
}}
.cs-reference-body {{
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-reference-body h2 {{
  margin: 0;
  font-size: var(--cs-typography-sectionTitle-fontSize);
  line-height: var(--cs-typography-sectionTitle-lineHeight);
}}
.cs-reference-body p {{
  margin: 0;
  color: var(--cs-color-text-secondary);
}}
.cs-connector-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-owner-main-stack, .cs-connector-list, .cs-admin-stack, .cs-policy-list {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-admin-stack {{
  position: sticky;
  top: calc(var(--cs-space-4) + 72px);
}}
.cs-connector-table-head {{
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) minmax(0, .7fr) minmax(0, .8fr) minmax(0, .9fr) auto;
  gap: var(--cs-space-3);
  padding: 0 var(--cs-space-3) var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-connector-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) minmax(0, .7fr) minmax(0, .8fr) minmax(0, .9fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-connector-card h3 {{ margin: 0 0 var(--cs-space-1); font-size: var(--cs-typography-body-fontSize); }}
.cs-connector-card p {{ margin: 0; color: var(--cs-color-text-secondary); font-size: var(--cs-typography-metadata-fontSize); line-height: 1.45; }}
.cs-connector-source {{
  display: grid;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-connector-title {{
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
}}
.cs-connector-icon {{
  width: 34px;
  height: 34px;
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  display: grid;
  place-items: center;
  font-weight: var(--cs-typography-weight-bold);
  font-size: var(--cs-typography-metadata-fontSize);
  flex: 0 0 auto;
}}
.cs-connector-cell {{
  display: grid;
  gap: var(--cs-space-1);
  min-width: 0;
}}
.cs-connector-cell span, .cs-connector-cell strong {{
  display: block;
  min-width: 0;
  overflow-wrap: anywhere;
}}
.cs-policy-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-policy-row strong, .cs-policy-row p {{
  margin: 0;
}}
.cs-policy-row p {{
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: 1.5;
}}
.cs-owner-scope-table {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  overflow: hidden;
}}
.cs-owner-scope-row {{
  display: grid;
  grid-template-columns: minmax(150px, .45fr) minmax(0, 1fr);
  gap: var(--cs-space-3);
  padding: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-owner-scope-row:last-child {{ border-bottom: 0; }}
.cs-owner-scope-row strong, .cs-owner-scope-row span {{ min-width: 0; word-break: break-word; }}
.cs-admin-note {{
  border: 1px solid var(--cs-state-underReview-border);
  border-radius: var(--cs-radius-md);
  background: var(--cs-state-underReview-bg);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-2);
}}
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
.cs-action-preview-frame {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-action-preview-meta {{
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
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
.cs-call-facts {{
  margin-top: var(--cs-space-3);
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-2);
}}
.cs-call-fact {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-2);
  display: grid;
  gap: 2px;
}}
.cs-call-fact strong {{ font-size: var(--cs-typography-metadata-fontSize); }}
.cs-call-fact span {{ color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
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
.cs-policy-checks {{
  display: grid;
  gap: var(--cs-space-2);
  margin-top: var(--cs-space-3);
}}
.cs-policy-check {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-policy-check-mark {{
  width: 24px;
  height: 24px;
  border-radius: var(--cs-radius-sm);
  background: var(--cs-state-underReview-bg);
  color: var(--cs-state-underReview-fg);
  border: 1px solid var(--cs-state-underReview-border);
  display: grid;
  place-items: center;
  font-weight: var(--cs-typography-weight-bold);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-timeline {{ display: grid; gap: var(--cs-space-3); }}
.cs-timeline-item {{ display: grid; grid-template-columns: 16px minmax(0, 1fr); gap: var(--cs-space-3); }}
.cs-dot {{ width: 10px; height: 10px; margin-top: 7px; border-radius: var(--cs-radius-pill); background: var(--cs-color-evidence-600); }}
.cs-audit-workbench {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 380px);
  gap: var(--cs-space-5);
  align-items: start;
}}
.cs-audit-hero {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background:
    linear-gradient(135deg, var(--cs-color-surface-primary), var(--cs-color-surface-subtle));
  padding: var(--cs-space-6);
  margin-bottom: var(--cs-space-4);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-5);
  align-items: end;
}}
.cs-audit-hero h1 {{
  margin: 0;
  font-size: 34px;
  line-height: 1.12;
  text-wrap: balance;
}}
.cs-audit-hero p {{
  max-width: 72ch;
  text-wrap: pretty;
}}
.cs-audit-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  justify-content: flex-end;
}}
.cs-audit-status-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-4);
}}
.cs-audit-overview {{
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(360px, .95fr);
  gap: var(--cs-space-4);
  margin-bottom: var(--cs-space-4);
  align-items: stretch;
}}
.cs-audit-latest {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--cs-color-primary-50) 44%, var(--cs-color-surface-primary)), var(--cs-color-surface-primary) 62%);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-3);
  min-width: 0;
}}
.cs-audit-latest h2 {{
  margin: 0;
  font-size: var(--cs-typography-sectionTitle-fontSize);
  line-height: var(--cs-typography-sectionTitle-lineHeight);
}}
.cs-audit-latest-title {{
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-audit-latest-title p {{
  margin: var(--cs-space-1) 0 0;
  color: var(--cs-color-text-secondary);
  text-wrap: pretty;
}}
.cs-audit-latest-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
}}
.cs-audit-overview-side {{
  display: grid;
  gap: var(--cs-space-3);
  min-width: 0;
}}
.cs-audit-summary {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-2);
}}
.cs-audit-receipt {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2);
  display: grid;
  gap: 2px;
}}
.cs-audit-receipt strong {{
  font-size: 18px;
  line-height: 1.15;
  font-variant-numeric: tabular-nums;
}}
.cs-audit-lifecycle {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-2);
}}
.cs-audit-lane {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: color-mix(in srgb, var(--cs-color-surface-primary) 78%, var(--cs-color-surface-subtle));
  padding: var(--cs-space-2);
  display: grid;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-audit-lane-head {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--cs-space-2);
}}
.cs-audit-lane-count {{
  font-variant-numeric: tabular-nums;
  font-weight: var(--cs-typography-weight-bold);
  color: var(--cs-color-primary-700);
}}
.cs-audit-lane p {{
  margin: 0;
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: var(--cs-typography-metadata-lineHeight);
}}
.cs-audit-list {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-audit-list-panel {{
  scroll-margin-top: 92px;
}}
.cs-audit-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-audit-row:hover {{
  border-color: var(--cs-color-border-strong);
  box-shadow: var(--cs-shadow-sm);
}}
.cs-audit-row-main {{
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-audit-row-top {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-3);
}}
.cs-audit-row-position {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}}
.cs-audit-icon {{
  width: 36px;
  height: 36px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-audit-row h2 {{
  margin: 0;
  font-size: var(--cs-typography-sectionTitle-fontSize);
  line-height: var(--cs-typography-sectionTitle-lineHeight);
}}
.cs-audit-row-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  margin-top: var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-audit-row-note {{
  margin: var(--cs-space-2) 0 0;
  max-width: 64ch;
  color: var(--cs-color-text-secondary);
}}
.cs-audit-side-list {{
  display: grid;
  gap: var(--cs-space-3);
  margin-top: var(--cs-space-3);
}}
.cs-audit-side-item {{
  border-left: 1px solid var(--cs-color-border-default);
  padding-left: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-audit-detail {{
  border-top: 1px solid var(--cs-color-border-default);
  padding-top: var(--cs-space-3);
}}
.cs-audit-detail summary {{
  cursor: pointer;
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-audit-raw-grid {{
  margin-top: var(--cs-space-3);
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-audit-raw-item {{
  border-left: 1px solid var(--cs-color-border-default);
  padding-left: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-audit-empty {{
  border: 1px dashed var(--cs-color-border-strong);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-6);
  display: grid;
  gap: var(--cs-space-4);
}}
.cs-audit-empty-steps {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-audit-empty-step {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-status {{
  min-height: 34px;
  border: 1px solid var(--cs-color-border-default);
  border-left-width: 3px;
  border-radius: var(--cs-radius-md);
  color: var(--cs-color-text-secondary);
  padding: var(--cs-space-2) var(--cs-space-3);
  background: var(--cs-color-surface-primary);
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
}}
.cs-status::before {{
  content: "";
  width: 8px;
  height: 8px;
  border-radius: var(--cs-radius-pill);
  background: currentColor;
  flex: 0 0 auto;
}}
.cs-status.is-idle {{
  border-left-color: var(--cs-color-primary-600);
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
}}
.cs-status.is-loading {{
  border-left-color: var(--cs-state-underReview-fg);
  background: var(--cs-state-underReview-bg);
  color: var(--cs-state-underReview-fg);
}}
.cs-status.is-success {{
  border-left-color: var(--cs-state-evidenceBacked-fg);
  background: var(--cs-state-evidenceBacked-bg);
  color: var(--cs-state-evidenceBacked-fg);
}}
.cs-status.is-error {{
  border-left-color: var(--cs-state-failed-fg);
  background: var(--cs-state-failed-bg);
  color: var(--cs-state-failed-fg);
}}
.cs-button:disabled {{
  cursor: progress;
  opacity: .72;
  transform: none;
}}
@media (max-width: 980px) {{
  .cs-shell {{ grid-template-columns: 1fr; padding-bottom: 68px; }}
  .cs-main {{ order: 1; display: flex; flex-direction: column; }}
  .cs-sidebar {{
    order: 3;
    position: static;
    height: 0;
    border-right: 0;
    border-bottom: 0;
    padding: 0;
    overflow: visible;
  }}
  .cs-sidebar > .cs-brand, .cs-sidebar > .cs-shell-note, .cs-nav > .cs-sidebar-status {{ display: none; }}
  .cs-nav {{
    position: fixed;
    inset: auto 0 0;
    z-index: 5;
    display: block;
    border-top: 1px solid var(--cs-color-border-default);
    background: color-mix(in srgb, var(--cs-color-surface-primary) 96%, transparent);
    backdrop-filter: blur(12px);
    padding: var(--cs-space-1) var(--cs-space-2);
  }}
  .cs-nav-group {{ grid-template-columns: repeat(5, minmax(0, 1fr)); gap: var(--cs-space-1); }}
  .cs-nav-label {{ display: none; }}
  .cs-nav a {{
    min-height: 58px;
    grid-template-columns: 1fr;
    justify-items: center;
    align-content: center;
    gap: 0;
    padding: var(--cs-space-1);
    text-align: center;
    font-size: var(--cs-typography-metadata-fontSize);
  }}
  .cs-nav a[aria-current="page"] {{ box-shadow: inset 0 3px 0 var(--cs-color-primary-600); }}
  .cs-nav-mark {{ width: 22px; height: 22px; }}
  .cs-nav-count {{ display: none; }}
  .cs-topbar {{ order: 1; position: static; padding: var(--cs-space-4); align-items: stretch; flex-direction: column; }}
  .cs-command {{
    max-width: none;
    min-width: 0;
  }}
  .cs-topbar-actions {{ justify-content: flex-start; }}
  .cs-search {{ max-width: none; flex-basis: auto; }}
  .cs-content {{ order: 2; padding: var(--cs-space-4); }}
  .cs-grid-hero, .cs-grid-two, .cs-module-grid, .cs-detail-orientation, .cs-brief-hero, .cs-brief-workbench, .cs-brief-titlebar, .cs-brief-lead-grid, .cs-search-workbench, .cs-search-command, .cs-artifact-hero, .cs-artifact-workbench, .cs-artifact-compact-hero, .cs-artifact-title-row, .cs-metadata-strip, .cs-metadata-strip.is-artifact, .cs-artifact-inspection-strip, .cs-inbox-workbench, .cs-inbox-lane-summary, .cs-inbox-receipt-strip, .cs-collection-workbench, .cs-collection-summary, .cs-collection-footrail, .cs-queue-lanes, .cs-empty-state-main, .cs-empty-steps, .cs-empty-briefing, .cs-brief-fact-strip, .cs-brief-note-grid, .cs-action-workbench, .cs-action-titlebar, .cs-action-review-strip, .cs-action-receipt-grid, .cs-action-route-strip, .cs-call-facts, .cs-audit-hero, .cs-audit-overview, .cs-audit-workbench, .cs-audit-status-strip, .cs-audit-summary, .cs-audit-lifecycle, .cs-audit-empty-steps, .cs-audit-raw-grid, .cs-owner-overview, .cs-reference-grid, .cs-connector-grid, .cs-connector-card, .cs-policy-row, .cs-owner-scope-row, .cs-claim-workbench, .cs-claim-titlebar, .cs-claim-pathbar, .cs-claim-progress, .cs-claim-review-strip, .cs-claim-taxonomy, .cs-claim-footrail {{ grid-template-columns: 1fr; }}
  .cs-owner-metric {{ border-right: 0; border-bottom: 1px solid var(--cs-color-border-default); }}
  .cs-owner-metric:last-child {{ border-bottom: 0; }}
  .cs-connector-table-head {{ display: none; }}
  .cs-page-head {{ margin-bottom: var(--cs-space-4); }}
  .cs-hero h1 {{ font-size: var(--cs-typography-pageTitle-fontSize); line-height: var(--cs-typography-pageTitle-lineHeight); }}
  .cs-home-intro {{ min-height: auto; }}
  .cs-home-canvas {{ padding: var(--cs-space-4); }}
  .cs-home-canvas > .cs-panel-header p {{ display: none; }}
  .cs-home-source-row, .cs-home-item, .cs-next-step, .cs-home-paste-row {{ grid-template-columns: 1fr; }}
  .cs-home-loop-inline {{ justify-content: flex-start; max-width: none; }}
  .cs-drop {{ min-height: auto; padding: var(--cs-space-3); }}
  .cs-drop-target {{ grid-template-columns: auto minmax(0, 1fr); place-items: center start; text-align: left; }}
  .cs-drop-target p {{ display: none; }}
  .cs-drop textarea.cs-drop-input {{ min-height: 72px; }}
  .cs-ask-bar {{ grid-template-columns: 1fr; }}
  .cs-suggestion-row {{ grid-template-columns: 1fr; }}
  .cs-inbox-lane-summary {{ align-items: flex-start; }}
  .cs-journey-ref-grid {{ grid-template-columns: 1fr; }}
  .cs-empty-actions {{ flex-direction: column; align-items: stretch; }}
  .cs-empty-actions .cs-button {{ justify-content: center; }}
  .cs-detail-actions {{ justify-content: flex-start; }}
  .cs-brief-actions {{ justify-content: flex-start; }}
  .cs-brief-titlebar h1 {{ font-size: 26px; }}
  .cs-brief-actions {{
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    justify-content: stretch;
  }}
  .cs-brief-fact-strip {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}
  .cs-brief-actions .cs-button {{
    justify-content: center;
    width: 100%;
    min-width: 0;
    max-width: 100%;
    white-space: normal;
    overflow-wrap: anywhere;
    text-align: center;
  }}
  .cs-claim-actions {{ justify-content: flex-start; }}
  .cs-action-actions {{ justify-content: flex-start; }}
  .cs-action-rail {{ position: static; }}
  .cs-audit-actions {{ justify-content: flex-start; }}
  .cs-admin-stack {{ position: static; }}
  .cs-claim-progress::before {{ display: none; }}
  .cs-trust-ladder, .cs-action-summary, .cs-citation-meta {{ grid-template-columns: 1fr; }}
  .cs-diff-line, .cs-call-row, .cs-result-row, .cs-inbox-head, .cs-inbox-row, .cs-collection-row, .cs-action-object-row, .cs-connector-card, .cs-claim-control-row {{ grid-template-columns: 1fr; }}
  .cs-collection-actions {{ justify-items: start; }}
  .cs-collection-actions .cs-row {{ justify-content: flex-start; }}
  .cs-inbox-head {{ display: none; }}
  .cs-artifact-compact-hero .cs-artifact-title h1 {{ font-size: 26px; }}
  .cs-artifact-compact-hero .cs-artifact-actions {{ padding-top: 0; }}
  .cs-artifact-actions {{ justify-content: flex-start; }}
  .cs-artifact-rail {{ position: static; }}
  .cs-artifact-toolbar {{ align-items: stretch; flex-direction: column; }}
  .cs-document-frame.has-rail {{ grid-template-columns: 1fr; min-height: auto; }}
  .cs-artifact-page-rail {{ display: none; }}
  .cs-artifact-page-area {{ padding: var(--cs-space-3); }}
  .cs-search-rail {{ position: static; }}
  .cs-search-titleline {{ align-items: start; flex-direction: column; }}
  .cs-search-mode {{ min-width: 0; width: 100%; }}
  .cs-claim-breadcrumb {{
    display: grid;
    grid-template-columns: auto auto minmax(0, 1fr);
    align-items: center;
  }}
  .cs-claim-breadcrumb span:last-child {{
    grid-column: 1 / -1;
    width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .cs-claim-actions {{
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    justify-content: stretch;
  }}
  .cs-claim-actions .cs-button {{
    justify-content: center;
    width: 100%;
    min-width: 0;
    max-width: 100%;
    white-space: normal;
    overflow-wrap: anywhere;
    text-align: center;
  }}
  .cs-claim-actions .is-disabled {{ grid-column: auto; }}
  .cs-result-support {{ justify-items: start; text-align: left; }}
  .cs-audit-row-main {{ grid-template-columns: auto minmax(0, 1fr); }}
  .cs-audit-row-main .cs-chip {{ justify-self: start; }}
  .cs-audit-row-top {{ align-items: flex-start; flex-direction: column; gap: var(--cs-space-1); }}
  .cs-document-page {{ min-height: auto; }}
  .cs-list-row {{ grid-template-columns: 1fr; }}
  .cs-detail-grid {{ grid-template-columns: 1fr; }}
}}
"""


def _css_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")


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


def _page(root: Path, title: str, active: str, content: str, ctx: dict[str, Any], q: str) -> str:
    css = _token_css(root)
    nav = _nav(active, ctx)
    topbar = _topbar(q, ctx)
    scope = ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {}
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
  <title>{h(title)} - CornerStone</title>
  <style>{css}</style>
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
  >
    <aside class="cs-sidebar" aria-label="CornerStone navigation">
      <div class="cs-brand">
        <div class="cs-brand-mark">CS</div>
        <div>
          <div class="cs-brand-name">CornerStone</div>
          <div class="cs-brand-sub">Evidence-first workspace</div>
        </div>
      </div>
      <div class="cs-shell-note">Drop, ask, decide, and audit with visible receipts.</div>
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
    counts = {
        "/": len(ctx["artifacts"]),
        "/search": len(ctx["artifacts"]) + len(ctx["briefs"]) + len(ctx["claims"]) + len(ctx["actions"]),
        "/artifacts": len(ctx["artifacts"]),
        "/claims": len(ctx["claims"]),
        "/actions": len(ctx["actions"]),
    }
    primary = [
        ("/", "Home"),
        ("/search", "Search"),
        ("/artifacts", "Artifacts"),
        ("/claims", "Claims"),
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


def _nav_link(href: str, label: str, active: str, count: int) -> str:
    current = ' aria-current="page"' if href == active else ""
    mark = label[:1].upper()
    return f'<a href="{h(href)}"{current}><span class="cs-nav-mark" aria-hidden="true">{h(mark)}</span><span>{h(label)}</span><span class="cs-nav-count">{h(str(count))}</span></a>'


def _sidebar_status(ctx: dict[str, Any]) -> str:
    review_count = len(ctx["inbox"])
    source_count = len(ctx["artifacts"])
    decision_count = len(ctx["claims"]) + len(ctx["actions"])
    scope_label = _scope_label(ctx.get("scope"), include_owner=True)
    return f"""
<section class="cs-sidebar-status" aria-label="Workspace posture">
  <div class="cs-sidebar-status-row"><span>Scope</span><strong>{h(scope_label)}</strong></div>
  <div class="cs-sidebar-status-row"><span>Sources</span><strong>{h(source_count)}</strong></div>
  <div class="cs-sidebar-status-row"><span>Decisions</span><strong>{h(decision_count)}</strong></div>
  <div class="cs-sidebar-status-row"><span>Review queue</span><strong>{h(review_count)}</strong></div>
</section>
"""


def _topbar(q: str, ctx: dict[str, Any]) -> str:
    count = len(ctx["artifacts"])
    label = f"{count} saved source" + ("" if count == 1 else "s")
    scope = ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {}
    owner = str(scope.get("owner_id") or "local-user")
    workspace_label = f"Workspace: {scope.get('workspace_id') or 'default'}"
    return f"""
<header class="cs-topbar">
  <div class="cs-command" aria-label="Global search">
    <form class="cs-search" action="/search" method="get">
      <span aria-hidden="true">Search</span>
      <input name="q" value="{h(q)}" aria-label="Search the active workspace" placeholder="Search across saved sources, claims, briefs, and action drafts">
      <button type="submit">Go</button>
    </form>
  </div>
  <div class="cs-topbar-actions" aria-label="Workspace status">
    {_chip(label, "saved")}
    {_chip(f"Owner: {owner}", "searchable")}
    {_chip(workspace_label, "searchable")}
    {_chip("Receipts required", "underReview")}
    <span class="cs-icon-button" aria-label="Help">?</span>
    <a class="cs-avatar" href="/review" aria-label="Open owner area for {h(owner)}">{h(_owner_initials(scope))}</a>
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
      <div class="cs-kicker">Local workspace · Local only</div>
      <h1>Drop anything, or ask what we know</h1>
      <p>Save messy input, find what is already in the workspace, and shape a brief only when the sources are visible.</p>
    </div>
    <section class="cs-panel cs-home-canvas" aria-labelledby="home-workbench-title">
      <div class="cs-panel-header">
        <div>
          <h2 id="home-workbench-title">Start with a source or a question</h2>
          <p class="cs-muted">CornerStone keeps the original source visible, then lets drafts point back to what supports them.</p>
        </div>
        <div class="cs-home-loop-inline" aria-label="Daily loop handoff">
          <span><strong>1</strong> Original source kept</span>
          <span><strong>2</strong> Draft from saved sources</span>
          <span><strong>3</strong> Receipts before decisions</span>
          <span><strong>4</strong> Work leaves a trail</span>
        </div>
        {_chip("Untrusted until checked", "underReview")}
      </div>
      <div class="cs-home-workspace">
        <form class="cs-drop" id="cs-drop-form">
          <div class="cs-drop-target">
            <div class="cs-drop-mark" aria-hidden="true">In</div>
            <div>
              <strong>Drag and drop files or paste notes here</strong>
              <p class="cs-muted">Local files and pasted text become saved sources with the original content preserved.</p>
            </div>
          </div>
          <div class="cs-home-source-row">
            <button class="cs-button secondary" type="button" id="cs-file-button">Browse files</button>
            <button class="cs-button" id="cs-save-source-button" type="submit">Save source</button>
            <input id="cs-file-input" type="file" hidden>
          </div>
          <div class="cs-home-paste-row" aria-label="Paste text source">
            <textarea class="cs-drop-input" id="cs-drop-text" placeholder="Paste notes, an email, a renewal clause, or any text source"></textarea>
            <div class="cs-home-source-note">Dropped files are read locally by the browser before saving. Paste is optional until you save.</div>
          </div>
          <div class="cs-status is-idle" id="cs-drop-status" data-state="idle" role="status" aria-live="polite">Ready for a source.</div>
        </form>
        <div class="cs-or-divider">or ask a question</div>
        <form class="cs-stack" id="cs-ask-form">
          <div class="cs-ask-bar" role="group" aria-label="Ask the workspace">
            <span class="cs-ask-mark" aria-hidden="true">?</span>
            <div>
              <strong>Ask the workspace</strong>
              <div class="cs-meta">Answers are drafts. Open sources before a decision.</div>
            </div>
            <input class="cs-field" id="cs-ask-input" placeholder="Ask about saved sources">
            <button class="cs-button" id="cs-ask-submit-button" type="submit">Ask</button>
          </div>
          <div class="cs-suggestion-row">{suggestions}</div>
          <div class="cs-status is-idle" id="cs-ask-status" data-state="idle" role="status" aria-live="polite">No answer requested yet.</div>
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
    <span class="cs-meta">{h(_scope_label(ctx.get("scope")))}</span>
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
<div class="cs-activity-row">
  <span class="cs-activity-icon" aria-hidden="true">{h(_audit_icon(event_type))}</span>
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


def _search_page(ctx: dict[str, Any], q: str, search_type: str = "all") -> str:
    counts = _search_counts(ctx, q)
    counts_by_label = dict(counts)
    selected_type_label = dict(PRODUCT_SEARCH_TYPES).get(search_type, "All")
    total = counts_by_label.get(selected_type_label, 0)
    results = _search_records(ctx, q, search_type)
    rows = (
        "".join(results)
        if results
        else _search_empty(
            q,
            search_type=search_type,
            selected_type_label=selected_type_label,
            all_result_count=counts_by_label.get("All", 0),
        )
    )
    count_tabs = "".join(
        f'<a class="cs-search-tab{" is-active" if value == search_type else ""}" href="/search?q={quote(q)}&amp;type={h(value)}"'
        f'{" aria-current=\"page\"" if value == search_type else ""}>{h(label)} <strong>{h(counts_by_label.get(label, 0))}</strong></a>'
        for value, label in PRODUCT_SEARCH_TYPES
    )
    right_rail = _search_right_rail(counts, q)
    scope_label = _scope_label(ctx.get("scope"))
    return f"""
<section data-product-surface="search">
  <div class="cs-search-workbench">
    <div class="cs-search-main">
      <section class="cs-search-canvas" aria-label="Search workspace">
        <div class="cs-search-command">
          <div class="cs-search-copy">
            <a class="cs-search-back" href="/">&#8592; Search</a>
            <div class="cs-search-titleline">
              <div>
                <div class="cs-kicker">Workspace search</div>
                <h1>Search the workspace</h1>
              </div>
              <span class="cs-filter-chip">Current search context</span>
            </div>
            <p class="cs-muted">Keyword search over saved sources and drafts. Open the receipts before using a result for a decision.</p>
          </div>
        </div>
        <form class="cs-search-hero" action="/search" method="get">
          <span class="cs-search-lens" aria-hidden="true">Search</span>
          <input name="q" value="{h(q)}" aria-label="Search saved workspace records" placeholder="Search saved sources, claims, action drafts, and briefs">
          <input type="hidden" name="type" value="{h(search_type)}">
          <button class="cs-button cs-search-submit" type="submit" aria-label="Run search">Go</button>
        </form>
        <div class="cs-search-tabs" aria-label="Filter results by record type">{count_tabs}</div>
        <div class="cs-search-filterbar">
          <div class="cs-filter-row" aria-label="Search result context">
            <span class="cs-filter-chip">Date range: all saved time</span>
            <span class="cs-filter-chip">Sources: all visible</span>
          </div>
          <span class="cs-filter-chip">Order: keyword match</span>
        </div>
        <div class="cs-search-mode" aria-label="Current search context">
          <div class="cs-filter-row">
            <span class="cs-filter-chip">Search mode: local keyword</span>
            <span class="cs-filter-chip">Scope: {h(scope_label)}</span>
            <span class="cs-filter-chip">Type: {h(selected_type_label)}</span>
            <span class="cs-filter-chip">Result receipt required</span>
          </div>
        </div>
      </section>
      <div class="cs-result-list">
        <div class="cs-result-list-header"><span>{total} results</span><span>Receipt-first results</span></div>
        {rows}
      </div>
    </div>
    {right_rail}
  </div>
</section>
"""


def _search_records(ctx: dict[str, Any], q: str, search_type: str = "all") -> list[str]:
    query = q.lower().strip()
    if not query:
        return []
    rows: list[tuple[int, str]] = []
    for artifact in ctx["artifacts"]:
        text = " ".join([_artifact_title(artifact), str(artifact.get("_preview") or "")]).lower()
        score = _score(text, query)
        if score > 0 and search_type in {"all", "sources"}:
            rows.append(
                (
                    score,
                    _search_result_row(
                        "Source",
                        "TXT",
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
        if score > 0 and search_type in {"all", "briefs"}:
            label, state = _brief_label(brief)
            rows.append((score, _search_result_row("Brief", "BRF", _brief_title(brief), str(brief.get("summary") or "Brief draft"), _detail_href("briefs", brief.get("brief_id")), label, state, _display_date(brief))))
    for claim in ctx["claims"]:
        text = _claim_title(claim).lower()
        score = _score(text, query)
        if score > 0 and search_type in {"all", "claims"}:
            label, state = _claim_label(claim)
            visible_source_count = len(_source_items(ctx, claim))
            detail = (
                f"Claim draft with {visible_source_count} visible supporting source{'s' if visible_source_count != 1 else ''}."
                if visible_source_count
                else "Claim draft; no visible supporting source is attached."
            )
            rows.append((score, _search_result_row("Claim", "CLM", _claim_title(claim), detail, _detail_href("claims", claim.get("claim_id")), label, state, _display_date(claim))))
    for action in ctx["actions"]:
        dry_run = action.get("dry_run") if isinstance(action.get("dry_run"), dict) else {}
        text = _action_search_text(action)
        score = _score(text, query)
        if score > 0 and search_type in {"all", "actions"}:
            label, state = _action_label(action)
            rows.append((score, _search_result_row("Action", "ACT", _action_title(action), str(dry_run.get("goal") or "Action draft"), _detail_href("actions", action.get("action_id")), label, state, _display_date(action))))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in rows[:20]]


def _action_search_text(action: dict[str, Any]) -> str:
    dry_run = action.get("dry_run") if isinstance(action.get("dry_run"), dict) else {}
    impact = dry_run.get("expected_impact") if isinstance(dry_run.get("expected_impact"), dict) else {}
    return " ".join(
        [
            _action_title(action),
            str(dry_run.get("goal") or ""),
            str(dry_run.get("target") or ""),
            str(impact.get("target") or ""),
        ]
    ).lower()


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
    action_count = sum(1 for action in ctx["actions"] if _score(_action_search_text(action), query) > 0)
    return [
        ("All", source_count + brief_count + claim_count + action_count),
        ("Sources", source_count),
        ("Briefs", brief_count),
        ("Claims", claim_count),
        ("Actions", action_count),
    ]


def _search_result_row(kind: str, icon: str, title: str, detail: str, href: str, label: str, state: str, date: str) -> str:
    icon_class = {
        "Source": "is-source",
        "Brief": "is-brief",
        "Claim": "is-claim",
        "Action": "is-action",
    }.get(kind, "is-source")
    return f"""
<article class="cs-result-row">
  <span class="cs-result-icon {h(icon_class)}" aria-hidden="true">{h(icon)}</span>
  <span class="cs-result-body">
    <span class="cs-result-meta"><span class="cs-result-type">{h(kind)}</span><span>{h(date)}</span><span>Keyword match</span></span>
    <h3><a href="{h(href)}">{h(title)}</a></h3>
    <p>{h(_truncate(detail, 240))}</p>
  </span>
  <span class="cs-result-support">
    {_chip(label, state)}
    <span class="cs-meta">Local record receipt</span>
    <span class="cs-result-actions">
      <a class="cs-button secondary" href="{h(href)}">Open receipt</a>
    </span>
  </span>
</article>
"""


def _search_right_rail(counts: list[tuple[str, int]], q: str) -> str:
    stat_rows = "".join(
        f'<div class="cs-right-stat"><span class="cs-right-stat-label"><span class="cs-right-stat-icon" aria-hidden="true">{h(label[:1])}</span><span>{h(label)}</span></span><strong>{h(count)}</strong></div>'
        for label, count in counts[1:]
    )
    suggestions = _search_followups(q, counts)
    suggestion_rows = "".join(
        f'<a class="cs-suggested-query" href="/search?q={quote(item)}"><span>Q</span><span>{h(item)}</span></a>'
        for item in suggestions
    )
    return f"""
<aside class="cs-stack cs-search-rail">
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
                ("Search receipt", "Only local keyword matches are shown."),
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
            ("Result receipt", "Matches link back to a local record."),
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
      <h3>First receipts</h3>
      <div class="cs-empty-receipts">{receipt_rows}</div>
    </div>
  </div>
"""
    return f"""
<article class="cs-empty-state">
  <div class="cs-empty-state-main">
    <span class="cs-empty-mark" aria-hidden="true">{h(mark[:2])}</span>
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


def _queue_focus(title: str, detail: str, lanes: list[tuple[str, int | str, str, str]]) -> str:
    total = sum(int(value) for _, value, _, _ in lanes if isinstance(value, int))
    lane_cards = "".join(
        f"""
<span class="cs-queue-lane">
  <span>{h(label)}</span>
  <strong>{h(value)}</strong>
  <span>{h(note)}</span>
</span>
"""
        for label, value, note, _ in lanes
    )
    lane_chips = "".join(_chip(label, state) for label, _, _, state in lanes)
    return f"""
<section class="cs-queue-focus" aria-label="Decision queue">
  <div class="cs-queue-focus-head">
    <div>
      <div class="cs-kicker">Decision queue</div>
      <h2>{h(title)}</h2>
      <p>{h(detail)}</p>
    </div>
    <div class="cs-row">{lane_chips}</div>
  </div>
  <div class="cs-queue-lanes" aria-label="Review lanes">
    <span class="cs-meta"><strong>{h(total)}</strong> visible queue item{"s" if total != 1 else ""}</span>
    <span class="cs-row">{lane_cards}</span>
  </div>
</section>
"""


def _collection_row(
    icon: str,
    title: str,
    detail: str,
    href: str,
    meta: list[tuple[str, str]],
    chips: list[tuple[str, str]],
    *,
    footer: list[tuple[str, str]] | None = None,
    action_label: str = "Open",
) -> str:
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
  <span class="cs-collection-icon" aria-hidden="true">{h(icon)}</span>
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
    ) or _empty_state(
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
            ("Search receipt", "Source becomes available to keyword search."),
            ("Linked use", "Briefs and claims can reference this source."),
        ],
    )
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
      {_collection_toolbar("Source register", len(artifacts), [f"Scope: {_scope_label(ctx.get('scope'))}", "Type: all sources", "Sort: newest first"])}
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
    briefs = ctx["briefs"]
    with_sources = sum(1 for brief in briefs if _brief_source_count(brief))
    source_ref_count = sum(_brief_source_count(brief) for brief in briefs)
    needs_sources = max(len(briefs) - with_sources, 0)
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
      {_queue_focus("Brief reading queue", "Review lanes keep drafted answers, source coverage, and next use visible before a brief becomes decision material.", [("Ready to read", with_sources, "Source links visible", "searchable"), ("Needs source check", needs_sources, "Do not use in decisions yet", "draft"), ("Can feed decision", with_sources, "Review before claim or action", "saved")])}
      {_collection_toolbar("Brief queue", len(briefs), [f"Scope: {_scope_label(ctx.get('scope'))}", "State: drafts and source-backed", "Sort: newest first"])}
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
    supported_count = sum(1 for claim in claims if _evidence_refs(claim))
    evidence_backed_count = sum(1 for claim in claims if _claim_evidence_backed_earned(claim))
    approved_count = sum(1 for claim in claims if str(claim.get("status") or "").lower() == "approved")
    needs_support = max(len(claims) - supported_count, 0)
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
            footer=[
                ("Evidence refs", f"{source_count} source refs"),
                ("Trust lane", label),
                ("Next review step", "Citation checks before evidence-backed"),
            ],
            action_label="Review claim",
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
            ("Trust lane", "Draft, source support, evidence-backed after checks, then approved."),
            ("Evidence picker", "Support must stay visible before approval."),
        ],
    )
    return f"""
<section data-product-surface="claims">
  <div class="cs-page-head">
    <div class="cs-kicker">Claims</div>
    <h1>Claims that need source support</h1>
    <p>Promote only statements that can be traced back to saved sources.</p>
  </div>
  <div class="cs-collection-workbench">
    <div>
      {_collection_summary([("Claims", len(claims)), ("With sources", supported_count), ("Evidence-backed", evidence_backed_count), ("Approved", approved_count)])}
      {_queue_focus("Claim review lanes", "Move statements from draft to source support, then to evidence-backed only after citation checks, then approved.", [("Draft lane", needs_support, "Needs source support", "draft"), ("Source-support lane", supported_count, "Support attached", "searchable"), ("Evidence-backed locked", evidence_backed_count, "Citation checks required", "underReview"), ("Approved lane", approved_count, "Decision-ready after review", "saved")])}
      {_collection_toolbar("Claim review queue", len(claims), [f"Scope: {_scope_label(ctx.get('scope'))}", "State: open and approved", "Sort: needs review first"])}
      <div class="cs-collection-list">{rows}</div>
    </div>
    <aside class="cs-stack">
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Review posture</h2>
        <p class="cs-muted">Claims can show source support before review. The evidence-backed label stays locked until citation checks earn it.</p>
        <div class="cs-review-box">
          <a class="cs-button" href="/inbox">Open review inbox</a>
          <a class="cs-button secondary" href="/artifacts">Check sources</a>
        </div>
      </section>
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Trust ladder</h2>
        {_claim_trust_ladder(bool(supported_count), bool(evidence_backed_count), bool(approved_count))}
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
    preview_count = max(len(actions) - executed_count, 0)
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
      {_queue_focus("Action approval lanes", "Dry-run, risk, policy, and approval stay in the queue before any external step is available.", [("Preview lane", preview_count, "Inspect impact first", "searchable"), ("Approval lane", approval_count, "Owner review required", "underReview"), ("Executed lane", executed_count, "Audit after send", "saved")])}
      {_collection_toolbar("Action preview queue", len(actions), [f"Scope: {_scope_label(ctx.get('scope'))}", "Mode: dry-run first", "Sort: approval risk first"])}
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
    detail = _inbox_detail_panel(
        items[0] if items else None,
        ctx.get("selected_product_loop") if isinstance(ctx.get("selected_product_loop"), dict) else None,
        ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {},
    )
    counts = _inbox_counts(items)
    lane_summary = _inbox_lane_summary(counts)
    item_range = f"1-{len(items)} of {len(items)} items" if items else "0 of 0 items"
    scope_label = _scope_label(ctx.get("scope"))
    scope = ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {}
    owner_label = str(scope.get("owner_id") or "local-user")
    recent_activity = _recent_activity_block(ctx)
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
        <span class="cs-inbox-tab">Evidence gaps {_chip(str(counts["evidence_gaps"]), "insufficientEvidence")}</span>
        <span class="cs-inbox-tab">Approval requests {_chip(str(counts["approval_requests"]), "draft")}</span>
        <span class="cs-inbox-tab">Policy blocked {_chip(str(counts["policy_blocked"]), "policyBlocked")}</span>
        <span class="cs-inbox-tab">Failed runs {_chip(str(counts["failed"]), "failed")}</span>
      </div>
      {lane_summary}
      <div class="cs-inbox-toolbar">
        <div class="cs-filter-row">
          <span class="cs-inbox-filter-label"><span>F</span> Filters</span>
          <span class="cs-filter-chip">Type: all visible</span>
          <span class="cs-filter-chip">Owner: {h(owner_label)}</span>
          <span class="cs-filter-chip">Scope: {h(scope_label)}</span>
          <span class="cs-filter-chip">Priority: open first</span>
          <span class="cs-filter-chip">Trust/risk: visible labels</span>
        </div>
        <span class="cs-meta">Showing {h(len(items))} open item{"s" if len(items) != 1 else ""}</span>
      </div>
      <div class="cs-inbox-table" role="list" aria-label="Operational inbox items">
        <div class="cs-inbox-head" aria-hidden="true">
          <span></span><span>Item</span><span>Type</span><span>Owner</span><span>Time</span><span>Priority</span><span>Trust / risk</span>
        </div>
        {rows}
        <div class="cs-inbox-foot">{h(item_range)}</div>
      </div>
    </div>
    {detail}
  </div>
  <div class="cs-inbox-activity" aria-label="Inbox recent activity">{recent_activity}</div>
</section>
"""


def _inbox_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "needs_review": sum(1 for item in items if item.get("queue") == "Needs review"),
        "approval_requests": sum(1 for item in items if item.get("queue") == "Approval requests"),
        "policy_blocked": sum(1 for item in items if item.get("queue") == "Policy blocked"),
        "failed": sum(1 for item in items if item.get("queue") == "Failed runs"),
        "evidence_gaps": sum(int(item.get("evidence_gap_count", 0) or 0) for item in items),
    }


def _inbox_lane_summary(counts: dict[str, int]) -> str:
    lanes = [
        ("Needs review", counts["needs_review"], True),
        ("Evidence gaps", counts["evidence_gaps"], False),
        ("Approval requests", counts["approval_requests"], False),
        ("Policy blocked", counts["policy_blocked"], False),
        ("Failed runs", counts["failed"], False),
    ]
    total = counts["needs_review"] + counts["approval_requests"] + counts["policy_blocked"] + counts["failed"]
    pills = "".join(
        f"""
<span class="cs-inbox-summary-pill{" is-active" if active else ""}">{h(label)} <strong>{h(count)}</strong></span>
"""
        for label, count, active in lanes
    )
    return f"""
<section class="cs-inbox-lane-summary" aria-label="Triage summary">
  <span class="cs-inbox-summary-main"><strong>{h(total)}</strong> open review items across one queue</span>
  <span class="cs-inbox-summary-pills">{pills}</span>
</section>
"""


def _inbox_table_row(item: dict[str, Any], selected: bool = False) -> str:
    selected_class = " is-selected" if selected else ""
    priority_state = "failed" if item.get("priority") == "High" else "underReview" if item.get("priority") == "Medium" else "draft"
    return f"""
<a class="cs-inbox-row{selected_class}" href="{h(item["href"])}" role="listitem">
  <span class="cs-inbox-select" aria-hidden="true"></span>
  <span class="cs-inbox-item-title">
    <span class="cs-inbox-icon" aria-hidden="true">{h(item.get("icon") or item["kind"][:1])}</span>
    <span>
      <strong>{h(item["title"])}</strong>
      <span class="cs-meta">{h(item["detail"])}</span>
    </span>
  </span>
  <span class="cs-inbox-type-cell"><span class="cs-inbox-type-mark" aria-hidden="true">{h(item.get("icon") or item["kind"][:1])}</span>{h(item.get("type") or item["kind"])}</span>
  <span class="cs-inbox-owner"><span class="cs-inbox-owner-mark" aria-hidden="true">O</span><span>{h(item.get("owner") or "Owner")}</span></span>
  <span class="cs-meta">{h(item["date"])}</span>
  <span>{_chip(item.get("priority") or "Medium", priority_state)}</span>
  <span>{_chip(item["label"], item["state"])}</span>
</a>
"""


def _inbox_detail_panel(
    item: dict[str, Any] | None,
    product_loop_result: dict[str, Any] | None = None,
    scope: dict[str, str] | None = None,
) -> str:
    if not item:
        return f"""
<aside class="cs-panel flat">
  <h2 class="cs-section-title">Selected item</h2>
  {_empty_state(
        "Queue empty",
        "No selected work",
        "Save a source, draft a claim, or preview an action to create reviewable work.",
        "Start from Home",
        "/",
        "Open briefs",
        "/briefs",
        mark="I",
    )}
</aside>
"""
    reason = _inbox_waiting_reason(item)
    linked_sources = _inbox_linked_sources(item)
    journey_timeline = _inbox_journey_timeline(product_loop_result, scope or {})
    return f"""
<aside class="cs-panel flat cs-inbox-detail">
  <div class="cs-inbox-detail-title">
    <span class="cs-inbox-icon" aria-hidden="true">{h(item.get("icon") or item["kind"][:1])}</span>
    <div>
      <div class="cs-kicker">Selected item</div>
      <h2>{h(item["title"])}</h2>
      <p class="cs-muted">{h(item["detail"])}</p>
    </div>
    <span class="cs-inbox-close" aria-hidden="true">x</span>
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
  {journey_timeline}
  <section class="cs-inbox-action-panel">
    <h2 class="cs-section-title">Next actions</h2>
    <div class="cs-inbox-actions">
      <a class="cs-button" href="{h(item["href"])}">Continue review</a>
      <a class="cs-button secondary" href="/search?q={quote(item["title"])}">Review sources</a>
      <a class="cs-button secondary" href="/audit">Open audit trail</a>
    </div>
  </section>
  <section class="cs-inbox-preview-note">
    <h3>Linked sources</h3>
    <div class="cs-inbox-linked-list">{linked_sources}</div>
  </section>
  <section class="cs-inbox-preview-note">
    <h3>Why this is here</h3>
    <p>{h(reason)}</p>
  </section>
  <section class="cs-inbox-preview-note">
    <h3>Safety state</h3>
    <p>Opening inbox work stays inside CornerStone. External writes still require the action approval path.</p>
  </section>
  <section>
    <h2 class="cs-section-title">Inbox receipt</h2>
    <div class="cs-inbox-receipt-strip">
      <div class="cs-inbox-receipt"><strong>Record</strong><span>{h(item.get("type") or item["kind"])}</span></div>
      <div class="cs-inbox-receipt"><strong>Evidence path</strong><span>Search sources</span></div>
      <div class="cs-inbox-receipt"><strong>Audit path</strong><span>Open trail</span></div>
    </div>
  </section>
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
      <span>CornerStone did not combine unrelated evidence-backed work. No new journey or activity record was created.</span>
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
  <span class="cs-inbox-type-mark" aria-hidden="true">S</span>
  <span><strong>{h(source_label)}</strong><span class="cs-meta">Search matching local records before deciding.</span></span>
</a>
<a class="cs-inbox-linked-row" href="/audit">
  <span class="cs-inbox-type-mark" aria-hidden="true">A</span>
  <span><strong>Audit trail</strong><span class="cs-meta">Open the local receipt history for this work.</span></span>
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
        "Open audit trail",
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


def _audit_page(ctx: dict[str, Any]) -> str:
    events = ctx["audit"]
    integrity = ctx.get("audit_integrity") if isinstance(ctx.get("audit_integrity"), dict) else {}
    integrity_status = str(integrity.get("status") or "not_verified")
    if integrity_status == "success":
        integrity_label = "Hash chain verified"
        integrity_state = "searchable"
    elif integrity_status == "failed":
        integrity_label = "Integrity failed"
        integrity_state = "failed"
    else:
        integrity_label = "Integrity unavailable"
        integrity_state = "underReview"
    event_count = len(events)
    visible_count = min(event_count, 40)
    scope = ctx.get("scope") if isinstance(ctx.get("scope"), dict) else {}
    workspace = str(scope.get("workspace_id") or "default")
    owner = str(scope.get("owner_id") or "local-user")
    source_count = sum(1 for event in events if _audit_family(str(event.get("event_type") or "")) == "Source")
    evidence_count = sum(1 for event in events if _audit_family(str(event.get("event_type") or "")) == "Evidence")
    decision_count = sum(1 for event in events if _audit_family(str(event.get("event_type") or "")) == "Decision")
    action_count = sum(1 for event in events if _audit_family(str(event.get("event_type") or "")) == "Action")
    latest_activity = _display_date(events[0]) if events else "No activity yet"
    first_activity = _display_date(events[-1]) if events else "No activity yet"
    latest_event = events[0] if events else None
    latest_title = _plain_event(str(latest_event.get("event_type") or "")) if latest_event else "No receipt yet"
    latest_family = _audit_family(str(latest_event.get("event_type") or "")) if latest_event else "Ledger"
    latest_note = (
        f"Readable receipt for {_audit_subject_label(latest_event)}. Raw event detail remains one step below the row."
        if latest_event
        else "Save a source or create work to start the local activity trail."
    )
    receipt_summary = f"""
<div class="cs-audit-summary" aria-label="Receipt summary">
  {_audit_receipt_card("Activity receipts", visible_count, f"{event_count} total scoped records")}
  {_audit_receipt_card("Source receipts", source_count, "Saved or opened sources")}
  {_audit_receipt_card("Evidence receipts", evidence_count, "Search, evidence, and brief work")}
  {_audit_receipt_card("Action receipts", action_count, "Preview, approval, or execution records")}
</div>
"""
    lifecycle = f"""
<section class="cs-audit-lifecycle" aria-label="Audit lifecycle">
  {_audit_lifecycle_card("Source saved", source_count, "Original inputs and source opens start the chain.", "searchable" if source_count else "draft")}
  {_audit_lifecycle_card("Supporting evidence prepared", evidence_count, "Search and brief receipts show source-linked work.", "evidenceBacked" if evidence_count else "draft")}
  {_audit_lifecycle_card("Decision recorded", decision_count, "Claims, workspace mode, and mission events explain decisions.", "underReview" if decision_count else "draft")}
  {_audit_lifecycle_card("Action proposed", action_count, "Action drafts stay inspectable before execution.", "underReview" if action_count else "draft")}
</section>
"""
    audit_overview = f"""
<section class="cs-audit-overview" aria-label="Audit status">
  <article class="cs-audit-latest">
    <div class="cs-audit-latest-title">
      <span class="cs-audit-icon" aria-hidden="true">{h(_audit_icon(str(latest_event.get("event_type") or "") if latest_event else ""))}</span>
      <div>
        <span class="cs-meta">Audit status</span>
        <h2>Latest readable receipt</h2>
        <p><strong>{h(latest_title)}</strong> · {h(latest_note)}</p>
      </div>
      {_chip(latest_family, "searchable")}
    </div>
    <div class="cs-brief-fact-strip">
      <div class="cs-brief-fact"><span class="cs-meta">Latest receipt</span><strong>{h(latest_activity)}</strong></div>
      <div class="cs-brief-fact"><span class="cs-meta">Scope</span><strong>{h(workspace)}</strong></div>
      <div class="cs-brief-fact"><span class="cs-meta">Chain status</span><strong>{h(integrity_label if events else "Ready")}</strong></div>
    </div>
    <div class="cs-audit-latest-actions">
      <a class="cs-button secondary" href="#activity-receipts">Read activity receipts</a>
      <a class="cs-button secondary" href="/artifacts">Open source register</a>
    </div>
  </article>
  <div class="cs-audit-overview-side">
    {receipt_summary}
    {lifecycle}
  </div>
</section>
"""
    if not ctx["audit"]:
        rows = _empty_state(
            "Audit ready",
            "No activity recorded yet",
            "Save a source, ask a question, draft a brief, or review a decision to start the local activity trail.",
            "Start from Home",
            "/",
            "Open artifacts",
            "/artifacts",
            mark="T",
            steps=[
                ("1. Save source", "Original input creates the first record."),
                ("2. Create work", "Searches, briefs, claims, and actions add readable events."),
                ("3. Inspect detail", "Raw event detail appears behind each row."),
            ],
            receipts=[
                ("Source receipt", "Original input starts the local ledger."),
                ("Decision receipt", "Claims and reviews explain why work moved."),
                ("Action receipt", "Previews and approvals stay inspectable."),
            ],
        )
    else:
        rows = "".join(
            f"""
<article class="cs-audit-row">
  <div class="cs-audit-row-main">
    <span class="cs-audit-icon" aria-hidden="true">{h(_audit_icon(str(event.get("event_type") or "")))}</span>
    <div>
      <div class="cs-audit-row-top">
        <h2>{h(_plain_event(str(event.get("event_type") or "")))}</h2>
        <span class="cs-audit-row-position">Receipt {index} of {visible_count}</span>
      </div>
      <p class="cs-audit-row-note">Readable receipt for {h(_audit_subject_label(event))}. Open raw event detail only when checking hashes, scope, or stored fields.</p>
      <div class="cs-audit-row-meta">
        <span>{h(_audit_family(str(event.get("event_type") or "")))}</span>
        <span>{h(_display_date(event))}</span>
        <span>Ledger position {index}</span>
        <span>{h(str(event.get("workspace_id") or workspace))}</span>
      </div>
    </div>
    {_chip(integrity_label, integrity_state)}
  </div>
  {_audit_detail(event, index)}
</article>
"""
            for index, event in enumerate(ctx["audit"][:40], start=1)
        )
        rows = f'<div class="cs-audit-list">{rows}</div>'
    return f"""
<section data-product-surface="audit" data-audit-integrity-status="{h(integrity_status)}">
  <header class="cs-audit-hero" aria-label="Audit receipt workspace">
    <div class="cs-brief-title">
      <div class="cs-kicker">Audit</div>
      <h1>Activity trail</h1>
      <p>Follow source, evidence, decision, and action receipts in one inspectable local ledger. The page reads like a product history first; raw event detail stays available for provenance checks.</p>
      <div class="cs-brief-meta">
        <span>Workspace: {h(workspace)}</span>
        <span>Latest: {h(latest_activity)}</span>
        <span>First receipt: {h(first_activity)}</span>
        <span>Reading order: newest first</span>
      </div>
    </div>
    <div class="cs-audit-actions">
      {_chip("Local ledger", "searchable")}
      {_chip("Readable receipts", "searchable")}
      {_chip(integrity_label, integrity_state)}
      <a class="cs-button secondary" href="/artifacts">Open source register</a>
      <a class="cs-button secondary" href="/">Back to Home</a>
    </div>
  </header>
  {audit_overview}
  <div class="cs-audit-workbench">
    <section class="cs-panel flat cs-audit-list-panel" id="activity-receipts">
      <div class="cs-panel-header">
        <div>
          <h2>Activity receipts</h2>
          <p class="cs-muted">Event stream. Showing {visible_count} of {event_count} scoped records as readable receipts.</p>
        </div>
        {_chip("Local ledger", "searchable")}
      </div>
      {rows}
    </section>
    <aside class="cs-stack">
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Audit posture</h2>
        <dl class="cs-detail-grid">
          <dt>Visible records</dt><dd>{visible_count}</dd>
          <dt>Decision receipts</dt><dd>{decision_count}</dd>
          <dt>Reading order</dt><dd>Newest first</dd>
          <dt>Disclosure</dt><dd>Raw event detail</dd>
          <dt>Scope</dt><dd>{h(workspace)}</dd>
        </dl>
      </section>
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Integrity chain</h2>
        <p class="cs-muted">Each row keeps its event hash, previous hash, subject, event type, and ledger position behind Raw event detail.</p>
        <div class="cs-audit-side-list" aria-label="Audit integrity checks">
          <div class="cs-audit-side-item"><strong>{h(integrity_label)}</strong><span class="cs-muted">Event hash and previous hash are verified before this page claims chain integrity.</span></div>
          <div class="cs-audit-side-item"><strong>Scoped ledger</strong><span class="cs-muted">Receipts are shown for the current local workspace.</span></div>
          <div class="cs-audit-side-item"><strong>Readable first</strong><span class="cs-muted">Plain labels stay above raw fields.</span></div>
        </div>
      </section>
      <section class="cs-panel flat">
        <h2 class="cs-section-title">Scope and recovery</h2>
        <p class="cs-muted">Use Audit to check what happened, then return to the source register or Home to continue the local workflow.</p>
        <div class="cs-empty-actions">
          <a class="cs-button secondary" href="/artifacts">Open source register</a>
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
  <nav class="cs-owner-tabs" aria-label="Owner review sections">
    <span class="cs-owner-tab is-active">Sources</span>
    <span class="cs-owner-tab">Policies</span>
    <span class="cs-owner-tab">Access roles</span>
    <span class="cs-owner-tab">Namespace</span>
  </nav>
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
          {_owner_scope_row("Retention", "Artifacts and audit receipts remain local review input")}
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


def _detail_orientation(
    *,
    parent_href: str,
    parent_label: str,
    current_label: str,
    summary: str,
    chip_label: str,
    chip_state: str,
    actions: list[tuple[str, str, str]] | None = None,
) -> str:
    action_links = "".join(
        f'<a class="cs-button {h(style)}" href="{h(href)}">{h(label)}</a>'
        for label, href, style in actions or []
    )
    return f"""
<header class="cs-detail-orientation">
  <div class="cs-detail-context">
    <nav class="cs-detail-path" aria-label="Detail path">
      <span class="cs-meta">Detail path</span>
      <a href="{h(parent_href)}">{h(parent_label)}</a>
      <span aria-hidden="true">/</span>
      <span>{h(current_label)}</span>
    </nav>
    <div class="cs-detail-summary">
      <div class="cs-detail-summary-head">
        <span class="cs-detail-current">{h(current_label)}</span>
        {_chip(chip_label, chip_state)}
      </div>
      <p>{h(summary)}</p>
    </div>
  </div>
  <div class="cs-detail-actions" aria-label="Detail page actions">
    {action_links}
  </div>
</header>
"""


def _artifact_detail(ctx: dict[str, Any], store: Any, artifact: dict[str, Any]) -> str:
    title = _artifact_title(artifact)
    text = _safe_preview(store, artifact, 5000, ctx.get("load_errors"), "source preview") or "No readable text preview is available for this source."
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
    summary = _truncate(text, 300)
    source_query = quote(title)
    linked_count = linked.count("cs-list-row")
    return f"""
<section class="cs-artifact-workbench" data-product-surface="artifact-detail" aria-label="Source inspection workspace">
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
          <span class="cs-artifact-file-mark" aria-hidden="true">TXT</span>
          <div>
            <h1>{h(title)}</h1>
            <div class="cs-row">{_chip("Saved", "saved")}{_chip("Searchable", "searchable")}{_chip("Untrusted until checked", "underReview")}</div>
          </div>
        </div>
      </div>
      <div class="cs-artifact-actions">
        <a class="cs-button" href="/search?q={h(source_query)}">Search this source</a>
        <a class="cs-button secondary" href="#linked-work">View linked work</a>
        <a class="cs-button ghost" href="/artifacts">Back to saved sources</a>
      </div>
    </header>
    <div class="cs-metadata-strip is-artifact" aria-label="Source metadata">
      <div class="cs-metadata-item"><span class="cs-meta">Saved source</span><strong>{h(source_label)}</strong></div>
      <div class="cs-metadata-item"><span class="cs-meta">Ingested</span><strong>{h(_display_date(artifact))}</strong></div>
      <div class="cs-metadata-item"><span class="cs-meta">File type</span><strong>{h(media_type)}</strong></div>
      <div class="cs-metadata-item"><span class="cs-meta">Workspace</span><strong>{h(workspace)}</strong></div>
      <div class="cs-metadata-item"><span class="cs-meta">Trust state</span><strong>Untrusted until checked</strong></div>
    </div>
    <section class="cs-artifact-viewer" aria-label="Original source document viewer">
      <div class="cs-artifact-toolbar">
        <div class="cs-artifact-toolgroup">
          <div class="cs-artifact-toolbar-label">
            <strong>Original source</strong>
            <span class="cs-meta">Plain text preview from the saved source</span>
          </div>
        </div>
        <div class="cs-artifact-toolgroup">
          <span class="cs-artifact-page-count">1 text source</span>
          <a class="cs-button ghost" href="#source-text">Source text</a>
        </div>
      </div>
      <div class="cs-document-frame has-rail">
        <nav class="cs-artifact-page-rail" aria-label="Source outline">
          <span class="cs-artifact-page-rail-label">Source outline</span>
          <a class="cs-artifact-thumb is-active" aria-current="page" href="#source-text">
            <span class="cs-artifact-thumb-line"></span>
            <span class="cs-artifact-thumb-line"></span>
            <span class="cs-artifact-thumb-line"></span>
            <span>Source text</span>
          </a>
          <a class="cs-artifact-thumb" href="#keywords">
            <span class="cs-artifact-thumb-line"></span>
            <span class="cs-artifact-thumb-line"></span>
            <span class="cs-artifact-thumb-line"></span>
            <span>Keywords</span>
          </a>
          <a class="cs-artifact-thumb" href="#linked-work">
            <span class="cs-artifact-thumb-line"></span>
            <span class="cs-artifact-thumb-line"></span>
            <span class="cs-artifact-thumb-line"></span>
            <span>Linked work</span>
          </a>
        </nav>
        <div class="cs-artifact-page-area">
          <article class="cs-document-page" aria-label="Original artifact preview" id="source-text">
            <header class="cs-document-heading">
              <span class="cs-meta">Original source preview · Plain text</span>
              <h3>{h(title)}</h3>
              <div class="cs-artifact-source-note">
                <span>{h(source_label)}</span>
                <span>/</span>
                <span>{h(_display_date(artifact))}</span>
                <span>/</span>
                <span>Original content primary</span>
              </div>
            </header>
            <div class="cs-source-text">{h(text)}</div>
          </article>
        </div>
      </div>
    </section>
  </div>
  <aside class="cs-stack cs-artifact-rail">
    <nav class="cs-artifact-rail-tabs" aria-label="Artifact detail tabs">
      <a class="cs-artifact-rail-tab is-active" href="#source-reading" aria-current="page">Details</a>
      <a class="cs-artifact-rail-tab" href="#keywords">Keywords ({len(keywords)})</a>
    </nav>
    <section class="cs-artifact-side-card" id="source-reading" aria-label="Source reading preview">
      <div class="cs-panel-header"><h2>Source reading preview</h2>{_chip("Original primary", "saved")}</div>
      <div class="cs-artifact-summary-lead">
        <span class="cs-meta">Original source excerpt</span>
        <strong>{h(_truncate(title, 110))}</strong>
        <p class="cs-muted">{h(summary)}</p>
      </div>
      <div class="cs-row">{_chip("Original content primary", "saved")}{_chip(f"{linked_count} linked drafts", "searchable")}</div>
    </section>
    <section class="cs-artifact-side-card" aria-label="Artifact inspection summary">
      <div class="cs-panel-header"><h2>Artifact inspection summary</h2>{_chip("Evidence links", "searchable")}</div>
      <div class="cs-artifact-panel-list">
        <div class="cs-artifact-inspection-card"><span class="cs-meta">Original preserved</span><strong>Yes</strong><span class="cs-muted">Derived drafts stay secondary.</span></div>
        <div class="cs-artifact-inspection-card"><span class="cs-meta">Preview mode</span><strong>Plain text preview</strong><span class="cs-muted">No simulated PDF controls.</span></div>
        <div class="cs-artifact-inspection-card"><span class="cs-meta">Linked drafts</span><strong>{linked_count}</strong><span class="cs-muted">Briefs or claims using this source.</span></div>
        <div class="cs-artifact-inspection-card"><span class="cs-meta">Fingerprint</span><strong>{h(fingerprint)}</strong><span class="cs-muted">Shown before reuse.</span></div>
      </div>
    </section>
    <section class="cs-artifact-side-card" id="keywords">
      <h2 class="cs-section-title">Source excerpt</h2>
      <p class="cs-muted">{h(summary)}</p>
    </section>
    <section class="cs-artifact-side-card">
      <div class="cs-panel-header"><h2>Frequent local terms</h2>{_chip(str(len(keywords)), "searchable")}</div>
      <div class="cs-keyword-list">{keyword_rows or '<div class="cs-empty">No keyword preview is available.</div>'}</div>
    </section>
    {linked}
    <section class="cs-artifact-side-card">
      <h2 class="cs-section-title">Source state</h2>
      <div class="cs-row">{_chip("Saved", "saved")}{_chip("Untrusted until checked", "underReview")}</div>
      <dl class="cs-detail-grid">
        <dt>Saved</dt><dd>{h(_display_date(artifact))}</dd>
        <dt>Source</dt><dd>{h(source_label)}</dd>
        <dt>Fingerprint</dt><dd>{h(fingerprint)}</dd>
      </dl>
      <p class="cs-muted">Keep this source visible before relying on derived drafts.</p>
    </section>
    <section class="cs-artifact-side-card">
      <h2 class="cs-section-title">Provenance</h2>
      <dl class="cs-detail-grid">
        <dt>Ingested from</dt><dd>{h(source_label)}</dd>
        <dt>Ingested</dt><dd>{h(_display_date(artifact))}</dd>
        <dt>Fingerprint</dt><dd>{h(fingerprint)}</dd>
      </dl>
      <a class="cs-meta" href="/audit">Open audit trail</a>
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
        return '<section class="cs-artifact-side-card" id="linked-work"><h2 class="cs-section-title">Linked work</h2><div class="cs-empty">No briefs or claims are linked to this source yet.</div></section>'
    return f'<section class="cs-artifact-side-card" id="linked-work"><h2 class="cs-section-title">Linked work</h2><div class="cs-list">{"".join(rows[:4])}</div></section>'


def _evidence_bundle_id(record: dict[str, Any]) -> str:
    bundle = record.get("evidence_bundle") if isinstance(record.get("evidence_bundle"), dict) else {}
    return str(bundle.get("evidence_bundle_id") or "")


def _related_brief_id(record: dict[str, Any]) -> str:
    related = record.get("related_brief") if isinstance(record.get("related_brief"), dict) else {}
    return str(related.get("brief_id") or "")


def _record_activity_panel(ctx: dict[str, Any], record_kind: str, record_id: str, audit_refs: list[str]) -> str:
    events = []
    for event in ctx.get("audit", []):
        subject = event.get("subject") if isinstance(event.get("subject"), dict) else {}
        if str(subject.get("type") or "") == record_kind and str(subject.get("id") or "") == record_id:
            events.append(event)
    rows = "".join(
        f'<div class="cs-stat-row"><span class="cs-stat-icon">A</span><div><strong>{h(_plain_event(str(event.get("event_type") or "recorded")))}</strong><div class="cs-meta">{h(_display_date(event))}</div></div>{_chip("Receipt", "searchable")}</div>'
        for event in events[:5]
    )
    refs = "".join(f"<li><code>{h(ref)}</code></li>" for ref in audit_refs[:5])
    if not rows:
        rows = '<div class="cs-empty">No record-specific activity receipt is visible yet.</div>'
    return f"""
<section class="cs-artifact-side-card" data-record-activity="{h(record_kind)}:{h(record_id)}">
  <h2 class="cs-section-title">Activity</h2>
  <div class="cs-stat-list">{rows}</div>
  <details class="cs-audit-detail">
    <summary>Audit refs</summary>
    <ul>{refs or '<li>No audit ref recorded.</li>'}</ul>
  </details>
</section>
"""


def _brief_related_work(ctx: dict[str, Any], brief: dict[str, Any]) -> str:
    brief_id = str(brief.get("brief_id") or "")
    bundle_id = _evidence_bundle_id(brief)
    claims = [
        claim
        for claim in ctx.get("claims", [])
        if _related_brief_id(claim) == brief_id
    ]
    claim_ids = {str(claim.get("claim_id") or "") for claim in claims}
    actions = [action for action in ctx.get("actions", []) if str(action.get("source_claim_id") or "") in claim_ids]
    memories = [
        memory
        for memory in ctx.get("memories", [])
        if bundle_id and any(str(ref) == f"evidence_bundle:{bundle_id}" for ref in memory.get("evidence_refs", []))
    ]
    claim_rows = "".join(
        f'<a class="cs-list-row" href="{h(_detail_href("claims", claim.get("claim_id")))}"><span class="cs-meta">Claim candidate</span><strong>{h(_claim_title(claim))}</strong></a>'
        for claim in claims[:4]
    ) or '<div class="cs-empty">No Claim candidate has been created from this Brief yet.</div>'
    memory_rows = "".join(
        f'<a class="cs-list-row" href="{h(_detail_href("memories", memory.get("memory_id")))}"><span class="cs-meta">Memory/Wiki candidate</span><strong>{h(_truncate(str(memory.get("title") or memory.get("statement") or "Knowledge draft"), 96))}</strong></a>'
        for memory in memories[:4]
    ) or '<div class="cs-empty">No Memory/Wiki candidate has been created. Saving a knowledge draft remains review-only.</div>'
    action_rows = "".join(
        f'<a class="cs-list-row" href="{h(_detail_href("actions", action.get("action_id")))}"><span class="cs-meta">Action preview</span><strong>{h(_action_title(action))}</strong></a>'
        for action in actions[:4]
    )
    if not action_rows:
        suggestions = [_plain_runtime_text(item) for item in brief.get("recommended_next_steps", []) if isinstance(item, str)]
        action_rows = "".join(f'<div class="cs-list-row"><span class="cs-meta">Suggested action</span><strong>{h(item)}</strong></div>' for item in suggestions[:3])
        action_rows = action_rows or '<div class="cs-empty">No suggested action is recorded.</div>'
    return f"""
<section class="cs-panel" aria-label="Related Brief work">
  <div class="cs-panel-header"><div><h2>Related work</h2><p class="cs-muted">Only persisted records are shown as candidates; suggestions remain clearly labelled.</p></div></div>
  <div class="cs-brief-note-grid">
    <div><h3>Claim candidates</h3>{claim_rows}</div>
    <div><h3>Memory/Wiki candidates</h3>{memory_rows}</div>
  </div>
  <div><h3>Suggested actions</h3>{action_rows}</div>
</section>
"""


def _brief_detail(ctx: dict[str, Any], brief: dict[str, Any]) -> str:
    label, state = _brief_label(brief)
    summary = str(brief.get("summary") or "")
    key_points = [str(item) for item in brief.get("key_points", []) if isinstance(item, str)]
    findings = [str(item) for item in brief.get("findings", []) if isinstance(item, str)]
    gaps = [_plain_runtime_text(item) for item in brief.get("gaps", []) if isinstance(item, str)]
    gaps.extend(_plain_runtime_text(item) for item in brief.get("uncertainty", []) if isinstance(item, str))
    gaps = gaps or ["Check the linked sources before treating this as decision-ready."]
    source_items = _source_items(ctx, brief)
    source_list = _source_links_from_items(source_items)
    point_rows = _statement_rows(brief, key_points, source_items)
    finding_rows = _statement_rows(brief, findings, source_items, offset=len(key_points))
    gap_rows = "".join(f"<li>{h(point)}</li>" for point in gaps[:8])
    next_steps = [_plain_runtime_text(item) for item in brief.get("recommended_next_steps", []) if isinstance(item, str)]
    next_rows = "".join(f"<li>{h(item)}</li>" for item in next_steps[:4]) or "<li>Review the visible sources before requesting review.</li>"
    provenance = _brief_provenance(brief)
    source_count = len(source_items)
    mode = _plain_output_mode(str(brief.get("output_mode") or "draft"))
    finding_count = len(key_points) + len(findings)
    brief_title = _brief_title(brief)
    label_state = _brief_label_state(brief)[0]
    citation_receipt = _brief_citation_receipt(brief, source_items)
    presented_as_fact = brief.get("presented_as_fact") is True
    citation_ready = citation_receipt["citation_refs_count"] > 0 and citation_receipt["unresolved_citation_count"] == 0 and citation_receipt["citation_check_refs_count"] > 0
    receipt_chip = _chip("Checked receipt" if citation_ready else "Source check", "searchable" if citation_ready else "underReview")
    receipt_note = (
        "Citation checks and visible source spans agree for the recorded references."
        if citation_ready
        else "Unsupported or unresolved citation work remains; use this as a draft until source spans are checked."
    )
    summary_text = summary or (key_points[0] if key_points else "No summary text was drafted yet. Use the findings and source snippets below before requesting review.")
    brief_id = str(brief.get("brief_id") or "")
    claim_statement = next((value for value in [*key_points, *findings, summary_text] if value.strip()), "")
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
        <p class="cs-muted">Read the answer, citation trail, and uncertainty together before using this brief for a decision.</p>
        <div class="cs-brief-meta">
          <span>{h(_display_date(brief))}</span>
          <span>{h(str(source_count))} visible source{"s" if source_count != 1 else ""}</span>
          <span>{h(str(finding_count))} drafted finding{"s" if finding_count != 1 else ""}</span>
        </div>
      </div>
      <div class="cs-brief-actions">
        <a class="cs-button secondary" href="/briefs">Back to briefs</a>
        <a class="cs-button secondary" href="#citation-trail">Review sources</a>
        <a class="cs-button secondary" href="/audit">Open audit trail</a>
      </div>
    </header>
    <section class="cs-brief-receipt-panel" aria-label="Receipt summary">
      <div class="cs-panel-header">
        <div>
          <span class="cs-meta">Receipt summary</span>
          <h2>Brief answer and receipt</h2>
          <p class="cs-muted">Read the answer, source support, and label state before promoting any finding.</p>
        </div>
        {receipt_chip}
      </div>
      <div class="cs-brief-lead-grid">
        <div class="cs-brief-answer-card is-primary">
          <span class="cs-meta">What we found</span>
          <p>{h(summary_text)}</p>
          <div class="cs-brief-fact-strip" aria-label="Brief status">
            <div class="cs-brief-fact"><span class="cs-meta">Decision snapshot</span><strong>Reviewed draft</strong></div>
            <div class="cs-brief-fact"><span class="cs-meta">Source coverage</span><strong>{h(source_count)} visible</strong></div>
            <div class="cs-brief-fact"><span class="cs-meta">Drafted findings</span><strong>{h(finding_count)}</strong></div>
          </div>
        </div>
        <div class="cs-brief-receipt-stack">
          <div class="cs-brief-receipt-card">
            <span class="cs-meta">Citation receipt</span>
            <strong>{h(citation_receipt["resolved_citation_count"])} / {h(citation_receipt["citation_refs_count"])} resolved</strong>
            <p>{h(receipt_note)}</p>
          </div>
          <div class="cs-brief-receipt-card">
            <span class="cs-meta">Label state</span>
            <strong>{h(label_state)}</strong>
            <p>{h(str(citation_receipt["citation_check_refs_count"]))} citation check ref{"s" if citation_receipt["citation_check_refs_count"] != 1 else ""}; {h(citation_receipt["citation_ref_kind"])} refs.</p>
          </div>
        </div>
      </div>
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
        <div class="cs-panel-header"><h2>What this brief cannot confirm</h2>{_chip("Needs source check", "underReview")}</div>
        <ul class="cs-brief-note-list">{gap_rows}</ul>
      </section>
      <section class="cs-panel">
        <div class="cs-panel-header"><h2>Suggested next steps</h2>{_chip("Draft only", "draft")}</div>
        <ul class="cs-brief-note-list">{next_rows}</ul>
      </section>
    </div>
    {related_work}
  </div>
  <aside class="cs-stack">
    <nav class="cs-artifact-rail-tabs" aria-label="Brief detail tabs">
      <span class="cs-artifact-rail-tab is-active">Sources</span>
      <span class="cs-artifact-rail-tab">Provenance</span>
    </nav>
    <section class="cs-artifact-side-card" id="citation-trail">
      <div class="cs-panel-header"><h2>Sources used</h2>{_chip(str(source_count), "searchable")}</div>
      <p class="cs-muted">Citation trail. Open a source before promoting a finding.</p>
      {source_list}
    </section>
    <section class="cs-artifact-side-card">
      <h2 class="cs-section-title">Provenance</h2>
      {provenance}
    </section>
    <section class="cs-artifact-side-card">
      <h2 class="cs-section-title">Use this brief</h2>
      <div class="cs-review-box">
        {f'<button class="cs-button" type="button" id="cs-create-claim-button" data-brief-id="{h(brief_id)}" data-statement="{h(claim_statement)}">Draft claim from finding</button>' if can_create_claim else '<span class="cs-button is-disabled" aria-disabled="true">Claim draft needs source evidence</span>'}
        <a class="cs-button secondary" href="/actions">Review action previews</a>
        <a class="cs-button secondary" href="/audit">Open audit trail</a>
      </div>
      <div id="cs-claim-create-status" class="cs-status is-idle" data-state="idle" role="status">No Claim candidate created yet.</div>
      <p class="cs-muted">Use this as a draft until each important statement is matched to a source span.</p>
    </section>
    {activity}
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
    evidence_backed_earned = _claim_evidence_backed_earned(claim)
    approval_note = (
        "Owner approval is recorded; autonomous action remains outside this Claim record."
        if is_approved
        else "Approval stays locked until review is recorded."
        if has_sources
        else "Approval stays locked until supporting evidence is attached."
    )
    rationale = _plain_runtime_text(claim.get("rationale") or "").strip() or "No separate rationale has been drafted yet. Use the source rail before asking for review."
    claim_title = _claim_title(claim)
    claim_statement = str(claim.get("statement") or claim_title)
    status_label = str(claim.get("status") or "draft").replace("_", " ").title()
    confidence_label = "Medium" if has_sources else "Needs evidence"
    source_label = f"{len(source_items)} source{'s' if len(source_items) != 1 else ''}"
    approved_stage = "is-active" if is_approved else ""
    source_stage = "is-active" if has_sources else ""
    evidence_stage = "is-active" if evidence_backed_earned else ""
    category = "Decision support"
    tags = [
        "Evidence review",
        "Owner approval",
        "Audit-ready",
    ]
    review_class = "is-ready" if has_sources else "is-review"
    claim_id = str(claim.get("claim_id") or "")
    related_brief_id = _related_brief_id(claim)
    related_brief = (
        f'<a class="cs-button secondary" href="{h(_detail_href("briefs", related_brief_id))}">Open source Brief</a>'
        if related_brief_id
        else '<span class="cs-muted">No source Brief lineage is recorded.</span>'
    )
    gaps = [_plain_runtime_text(value) for value in claim.get("gaps", []) if isinstance(value, str)]
    gap_rows = "".join(f"<li>{h(value)}</li>" for value in gaps[:6]) or "<li>No separate gaps were recorded; inspect source coverage before authority use.</li>"
    denial_events = []
    for event in ctx.get("audit", []):
        subject = event.get("subject") if isinstance(event.get("subject"), dict) else {}
        if str(event.get("event_type") or "") == "claim.approval.denied" and str(subject.get("id") or "") == claim_id:
            denial_events.append(event)
    denial_panel = ""
    if denial_events:
        denied = denial_events[0]
        details = denied.get("details") if isinstance(denied.get("details"), dict) else {}
        audit_ref = f"audit:{denied.get('event_id')}" if denied.get("event_id") else "Audit receipt unavailable"
        denial_panel = f"""
<section class="cs-panel" data-claim-approval-denial="true">
  <div class="cs-panel-header"><div><h2>Approval blocked</h2><p class="cs-muted">The Claim remains a draft.</p></div>{_chip("Evidence required", "insufficientEvidence")}</div>
  <p><strong>Cause:</strong> {h(_plain_runtime_text(details.get("reason") or "Supporting source evidence is missing."))}</p>
  <p><strong>Recovery:</strong> Attach supporting evidence with at least one saved source, then retry approval.</p>
  <details class="cs-audit-detail"><summary>Denial receipt</summary><p><code>{h(audit_ref)}</code></p><p>{h(_plain_runtime_text(details.get("required") or "Supporting evidence is required before approval."))}</p></details>
</section>
"""
    activity = _record_activity_panel(
        ctx,
        "claim",
        claim_id,
        [str(ref) for ref in claim.get("audit_refs", []) if isinstance(ref, str)],
    )
    lifecycle_name = "Approved claim" if is_approved else "Claim draft"
    workspace_title = "Approved claim workspace" if is_approved else "Claim draft workspace"
    workspace_note = (
        "Review the approved statement, its source support, and its audit receipt together."
        if is_approved
        else "Draft the claim beside its source support, then request review before a decision uses it."
    )
    heading_chips = _chip(label, state) + (_chip("Source support", "searchable") if has_sources else "")
    top_actions = (
        '<a class="cs-button" href="/audit">Open approval receipt</a><a class="cs-button secondary" href="/claims">Back to claims</a>'
        if is_approved
        else '<a class="cs-button secondary" href="/claims">Save draft</a><a class="cs-button" href="/inbox">Request review</a><span class="cs-button is-disabled" aria-disabled="true">Decision save locked</span><a class="cs-button secondary" href="/claims">Back to claims</a><a class="cs-button secondary" href="/inbox">Open inbox</a>'
    )
    review_controls = (
        '<a class="cs-button" href="/audit">Open approval receipt</a><a class="cs-button secondary" href="/claims">Back to claims</a>'
        if is_approved
        else '<a class="cs-button secondary" href="/claims">Save draft</a><a class="cs-button" href="/inbox">Request review</a><span class="cs-button is-disabled" aria-disabled="true">Decision save locked</span>'
    )
    return f"""
<section
  class="cs-grid-two cs-claim-workbench"
  data-product-surface="claim-detail"
  data-source-support-attached="{str(has_sources).lower()}"
  data-evidence-backed-earned="{str(evidence_backed_earned).lower()}"
>
  <div class="cs-stack">
    <div class="cs-claim-hero is-compact">
      <div class="cs-claim-titlebar">
        <div class="cs-brief-title">
          <nav class="cs-claim-breadcrumb" aria-label="Detail path">
            <span class="cs-meta">Detail path</span>
            <a href="/claims">Claims</a>
            <span aria-hidden="true">/</span>
            <span>{h(_truncate(claim_title, 90))}</span>
          </nav>
          <div class="cs-claim-heading-row">
            <h1>{h(claim_title)}</h1>
            {heading_chips}
          </div>
          <div class="cs-brief-meta">
            <span>{h(lifecycle_name)}</span>
            <span>Created {h(_display_date(claim))}</span>
            <span>{h(source_label)} attached</span>
          </div>
        </div>
        <div class="cs-claim-actions" aria-label="Claim review actions">
          {top_actions}
        </div>
      </div>
      <div class="cs-row">{_chip(label, state)}{_chip("Owner approval recorded", "approved") if is_approved else _chip("Review required before approval", "underReview")}</div>
    </div>
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>{h(workspace_title)}</h2>
          <p class="cs-muted">{h(workspace_note)}</p>
        </div>
      </div>
      <div class="cs-claim-pathbar" aria-label="Evidence-to-decision path">
        <div class="cs-claim-pathbar-title">
          <span class="cs-meta">Trust ladder</span>
          <strong>Evidence-to-decision path</strong>
          <span class="cs-muted">{"Owner approval is recorded; citation integrity and autonomous action remain separate gates." if is_approved else "Decision use stays locked until source and owner review are recorded."}</span>
        </div>
        <div class="cs-claim-progress" aria-label="Trust ladder">
          <div class="cs-claim-progress-step is-active">
            <span class="cs-claim-dot" aria-hidden="true"></span>
            <span>Draft</span>
          </div>
          <div class="cs-claim-progress-step {source_stage}">
            <span class="cs-claim-dot" aria-hidden="true"></span>
            <span>Source support</span>
          </div>
          <div class="cs-claim-progress-step {evidence_stage}">
            <span class="cs-claim-dot" aria-hidden="true"></span>
            <span>Evidence-backed locked</span>
          </div>
          <div class="cs-claim-progress-step {approved_stage}">
            <span class="cs-claim-dot" aria-hidden="true"></span>
            <span>Approved</span>
          </div>
        </div>
      </div>
      <div class="cs-claim-tabs" aria-label="Claim workspace overview">
        <span class="cs-claim-tab is-active">Claim</span>
        <span class="cs-claim-tab">Supporting evidence</span>
        <span class="cs-claim-tab">Counter evidence</span>
        <span class="cs-claim-tab">Impacted objects</span>
        <span class="cs-claim-tab">Discussion</span>
      </div>
      <div class="cs-claim-form-card">
        <div class="cs-claim-field is-primary">
          <div class="cs-claim-field-head">
            <div>
              <strong>Claim statement</strong>
              <p class="cs-muted">Provide a clear statement that can be traced to sources.</p>
            </div>
            {_chip(confidence_label, "underReview" if has_sources else "insufficientEvidence")}
          </div>
          <p class="cs-claim-text is-statement">{h(claim_statement)}</p>
          <div class="cs-claim-field-foot">{h(str(len(claim_statement)))} / 500</div>
        </div>
        <div class="cs-claim-field">
          <div class="cs-claim-field-head">
            <div>
              <strong>Rationale</strong>
              <p class="cs-muted">Why does this claim matter for the next decision?</p>
            </div>
          </div>
          <p class="cs-claim-text">{h(rationale)}</p>
          <div class="cs-claim-field-foot">{h(str(len(rationale)))} / 1000</div>
        </div>
        <div class="cs-claim-taxonomy">
          <div class="cs-claim-select">
            <span class="cs-meta">Claim category</span>
            <strong>{h(category)}</strong>
          </div>
          <div class="cs-claim-tags" aria-label="Claim tags">
            <span class="cs-meta">Tags</span>
            {"".join(_chip(tag, "draft") for tag in tags)}
          </div>
        </div>
        <div class="cs-claim-frameworks" aria-label="Related frameworks">
          <span class="cs-meta">Related frameworks</span>
          <strong>No framework selected</strong>
        </div>
        <span class="cs-claim-save-note">Saved locally</span>
      </div>
      <div class="cs-claim-review-strip" aria-label="Claim review summary">
        <div class="cs-claim-review-card"><span class="cs-meta">Claim state</span><strong>{h(label)}</strong><span class="cs-meta">{"Owner approval recorded" if is_approved else "Review before approval"}</span></div>
        <div class="cs-claim-review-card"><span class="cs-meta">Source support</span><strong>{h(source_label)}</strong><span class="cs-meta">Visible local sources</span></div>
        <div class="cs-claim-review-card"><span class="cs-meta">Evidence-backed gate</span><strong>{"Earned" if evidence_backed_earned else "Locked"}</strong><span class="cs-meta">Citation checks required</span></div>
        <div class="cs-claim-review-card"><span class="cs-meta">Decision gate</span><strong>{"Approved" if is_approved else "Locked"}</strong><span class="cs-meta">{"Owner approval recorded" if is_approved else "Owner review required"}</span></div>
      </div>
    </section>
    {denial_panel}
    <section class="cs-panel">
      <div class="cs-panel-header"><div><h2>Brief lineage and gaps</h2><p class="cs-muted">The source Brief stays separate from this Claim record.</p></div></div>
      <div class="cs-review-box">{related_brief}</div>
      <ul class="cs-brief-note-list">{gap_rows}</ul>
    </section>
    <section class="cs-claim-footrail" aria-label="Claim provenance">
      <div><span class="cs-meta">Record</span><strong>{"Approved local claim" if is_approved else "Local draft"}</strong></div>
      <div><span class="cs-meta">Source support</span><strong>{h(source_label)}</strong></div>
      <div><span class="cs-meta">Status</span><strong>{h(status_label)}</strong></div>
      <div><span class="cs-meta">Last activity</span><strong>{h(_display_date(claim))}</strong></div>
    </section>
  </div>
  <aside class="cs-stack">
    <section class="cs-panel flat">
      <div class="cs-panel-header">
        <div>
          <h2>Supporting evidence</h2>
          <p class="cs-muted">These links open the visible local sources attached to this Claim.</p>
        </div>
        {_chip(str(len(source_items)), "searchable")}
      </div>
      <div class="cs-evidence-toolbar" aria-label="Supporting source link order">
        <span class="cs-filter-chip">Visible sources</span>
        <span class="cs-filter-chip">Source order</span>
      </div>
      {source_list}
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Review controls</h2>
      <div class="cs-review-box">
        {review_controls}
        <p class="cs-muted">{h(approval_note)}</p>
      </div>
    </section>
    <section class="cs-panel flat">
      <div class="cs-panel-header">
        <div>
          <h2>Decision gate</h2>
          <p class="cs-muted">{"Owner approval is recorded; source support, citation integrity, audit, and action authority remain separately visible." if is_approved else "Source support, owner review, and audit records stay separate before decision use."}</p>
        </div>
      </div>
      <div class="cs-claim-control-list">
        <div class="cs-claim-control-row {review_class}">
          <span class="cs-claim-control-mark" aria-hidden="true">1</span>
          <div>
            <strong>Source support</strong>
            <p class="cs-muted">{h(source_label)} visible in this workspace.</p>
          </div>
        </div>
        <div class="cs-claim-control-row {"is-ready" if is_approved else "is-review"}">
          <span class="cs-claim-control-mark" aria-hidden="true">2</span>
          <div>
            <strong>Citation checks</strong>
            <p class="cs-muted">Evidence-backed stays locked until checks prove source coverage and citation integrity.</p>
          </div>
        </div>
        <div class="cs-claim-control-row is-review">
          <span class="cs-claim-control-mark" aria-hidden="true">3</span>
          <div>
            <strong>Owner review</strong>
            <p class="cs-muted">{"Owner approval is recorded for this Claim." if is_approved else "Review required before approval or shared truth."}</p>
          </div>
        </div>
        <div class="cs-claim-control-row">
          <span class="cs-claim-control-mark" aria-hidden="true">4</span>
          <div>
            <strong>No autonomous action</strong>
            <p class="cs-muted">This claim cannot trigger external work from this page.</p>
          </div>
        </div>
      </div>
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Authority</h2>
      <p class="cs-muted">{h(_plain_runtime_text(authority.get("blocked_reason") or "Owner approval is required before this claim becomes shared truth or drives autonomous action."))}</p>
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
        <a class="cs-button secondary" href="/audit">Open audit trail</a>
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
        <a class="cs-button secondary" href="/audit">Open audit trail</a>
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
    lifecycle_stage = str(lifecycle["stage"])
    is_executed = lifecycle_stage == "executed"
    is_failed = lifecycle_stage == "failed"
    is_blocked = lifecycle_stage == "blocked"
    is_approved_stage = lifecycle_stage == "approved"
    approval = lifecycle["approval"]
    approval_status_raw = str(lifecycle["approval_status"])
    approval_recorded = approval_status_raw == "approved"
    execution = lifecycle["execution"]
    execution_result = lifecycle["result"]
    execution_status_raw = str(lifecycle["execution_status"])
    has_execution_attempt = is_executed or is_failed
    has_recorded_outcome = has_execution_attempt or is_blocked
    dry_run = action.get("dry_run") if isinstance(action.get("dry_run"), dict) else {}
    diff = dry_run.get("diff") if isinstance(dry_run.get("diff"), dict) else {}
    impact = dry_run.get("expected_impact") if isinstance(dry_run.get("expected_impact"), dict) else {}
    source_items = _source_items(ctx, action)
    source_list = _source_links_from_items(source_items)
    policy = action.get("policy_decision") if isinstance(action.get("policy_decision"), dict) else {}
    if not policy and isinstance(dry_run.get("policy_decision"), dict):
        policy = dry_run["policy_decision"]
    decision_label = _plain_policy_decision(str(policy.get("decision") or ""))
    if is_blocked and not policy.get("decision"):
        decision_label = "Not recorded"
    workspace_mode = str(policy.get("workspace_mode") or "local").replace("_", " ").title()
    planned_real_external_calls = int(impact.get("real_external_http_calls", 0) or 0)
    expected_connector_calls = int(impact.get("expected_connector_calls", 0) or 0)
    raw_observed_external_calls = execution_result.get("external_http_calls")
    try:
        observed_external_calls = (
            int(raw_observed_external_calls)
            if raw_observed_external_calls is not None and not isinstance(raw_observed_external_calls, bool)
            else None
        )
    except (TypeError, ValueError):
        observed_external_calls = None
    if observed_external_calls is not None and observed_external_calls < 0:
        observed_external_calls = None
    displayed_external_calls: int | str = (
        observed_external_calls
        if has_recorded_outcome and observed_external_calls is not None
        else "not-recorded"
        if has_recorded_outcome
        else planned_real_external_calls
    )
    observed_call_note = (
        "External HTTP call count not recorded"
        if observed_external_calls is None
        else "No external HTTP calls recorded"
        if observed_external_calls == 0
        else f"{observed_external_calls} external HTTP call{'s' if observed_external_calls != 1 else ''} recorded"
    )
    call_label = (
        "Local mock execution recorded"
        if is_executed and action.get("connector_boundary", {}).get("mocked") is True
        else "Governed execution recorded"
        if is_executed
        else "Failed attempt recorded"
        if is_failed
        else "Blocked state recorded"
        if is_blocked
        else "Simulated in local mode"
        if planned_real_external_calls == 0
        else "Provider send planned"
    )
    connector = action.get("connector_boundary") if isinstance(action.get("connector_boundary"), dict) else {}
    execution_mode = (
        "Local / Mock / Draft"
        if connector.get("mocked") is True and not has_execution_attempt
        else "Local / Mock result"
        if connector.get("mocked") is True and has_execution_attempt
        else f"{workspace_mode} / Governed"
    )
    connector_label = "Connector-mediated path" if connector.get("direct_provider_access") is False else "Provider access needs review"
    approval_required = bool(approval.get("required") or "approval" in decision_label.lower())
    approval_label = "Approval recorded" if approval_recorded else "Approval required" if approval_required else "Approval not required"
    risk_label = str(impact.get("risk") or action.get("risk") or "review").title()
    target = str(impact.get("target") or "Local preview only.")
    goal = str(dry_run.get("goal") or action.get("goal") or _action_title(action))
    action_title = _action_title(action)
    approval_status = approval_status_raw.replace("_", " ").title()
    execution_status = execution_status_raw.replace("_", " ").title()
    result_status = str(execution_result.get("status") or "recorded").replace("_", " ").title()
    result_message = _plain_runtime_text(
        execution_result.get("message") or "The governed local execution result is recorded."
    )
    reason = _plain_runtime_text(
        approval.get("required_reason")
        or policy.get("reason")
        or "A reason is required before approval can move this preview toward execution."
    )
    policy_reason = _plain_runtime_text(
        policy.get("reason")
        or "This action is permitted only after review confirms the source, target, and risk."
    )
    raw_block_reason = policy.get("reason") or execution.get("reason") or execution.get("message")
    if not raw_block_reason and is_blocked and execution_status_raw not in {"", "not_started"}:
        raw_block_reason = execution_status_raw.replace("_", " ").capitalize()
    blocked_reason = _plain_runtime_text(raw_block_reason or "No block cause is recorded on this action.")
    raw_block_recovery = execution.get("recovery_path") or policy.get("resolution_path")
    if isinstance(raw_block_recovery, list):
        raw_block_recovery = " ".join(str(value) for value in raw_block_recovery if isinstance(value, str))
    blocked_recovery = _plain_runtime_text(
        raw_block_recovery
        or "Review the recorded state, source, target, and approval boundary before creating a new preview."
    )
    displayed_policy_reason = blocked_reason if is_blocked else policy_reason
    rail_reason_label = "Recorded cause" if is_blocked else "Required reason"
    rail_reason = blocked_reason if is_blocked else reason
    lifecycle_note = (
        f"A governed execution result is recorded. {observed_call_note}."
        if is_executed
        else f"The recorded attempt failed. Review the cause and recovery path before creating a new preview. {observed_call_note}."
        if is_failed
        else "This action is blocked. Resolve the recorded cause before creating a new preview."
        if is_blocked
        else "Approval is recorded; execution has not been recorded."
        if is_approved_stage
        else "Preview impact, policy, and approval history before any external step."
    )
    if is_executed:
        lifecycle_meta = "Executed result"
    elif is_failed:
        lifecycle_meta = "Failed with recovery"
    elif is_blocked:
        lifecycle_meta = "Policy blocked"
    elif is_approved_stage:
        lifecycle_meta = "Approved action"
    else:
        lifecycle_meta = "Dry-run first"
    external_meta = observed_call_note if has_recorded_outcome else "No external send yet"
    risk_chip = _chip(f"{risk_label} risk", "underReview")
    if is_executed:
        action_controls = f'{_chip(label, state)}{_chip(approval_label, "approved") if approval_recorded else ""}{risk_chip}'
    elif is_failed:
        action_controls = (
            f'{_chip("Failed with recovery", "failed")}{_chip(approval_label, "approved") if approval_recorded else ""}{risk_chip}'
            '<a class="cs-button" href="/inbox">Review recovery</a>'
        )
    elif is_blocked:
        action_controls = (
            f'{_chip("Policy blocked", "policyBlocked")}{risk_chip}'
            '<a class="cs-button" href="/audit">Review block receipt</a>'
        )
    elif is_approved_stage:
        action_controls = f'{_chip(label, state)}{_chip(approval_label, "approved")}{risk_chip}'
    else:
        action_controls = (
            f'{_chip("Preview (dry run)", "searchable")}'
            f'{_chip(approval_label, "underReview")}{risk_chip}'
            '<a class="cs-button" href="/inbox">Request approval</a>'
        )
    rail_control = (
        '<a class="cs-button" href="/inbox">Review recovery</a>'
        if is_failed
        else '<a class="cs-button" href="/audit">Review block receipt</a>'
        if is_blocked
        else '<a class="cs-button" href="/audit">Open approval receipt</a>'
        if approval_recorded
        else '<a class="cs-button" href="/inbox">Request approval</a>'
    )
    rail_note = (
        "Review the failed attempt before deciding whether a new preview is safe."
        if is_failed
        else "Resolve the policy block before creating a new preview."
        if is_blocked
        else "Approval and execution are recorded as separate lifecycle steps."
        if is_executed
        else "Approval is recorded; execution remains pending."
        if is_approved_stage
        else "Execution is not shown as the primary action until approval is satisfied."
    )
    if approval_recorded:
        approval_history = f"""
<div class="cs-stat-row">
  <span class="cs-stat-icon">A</span>
  <div>
    <strong>Approved by {h(str(approval.get("approver") or "owner"))}</strong>
    <div class="cs-meta">{h(str(approval.get("approved_at") or _display_date(action)))}</div>
  </div>
  {_chip("Recorded", "approved")}
</div>
"""
    else:
        approval_history = '<div class="cs-empty">No approvals have been recorded yet.</div>'
    execution_panel = ""
    if is_executed:
        execution_panel = f"""
<section class="cs-panel" data-action-execution-result="true">
  <div class="cs-panel-header">
    <div><h2>Execution result</h2><p class="cs-muted">Durable lifecycle state from this Action record.</p></div>
    {_chip(result_status, "executed")}
  </div>
  <p>{h(result_message)}</p>
  <dl class="cs-detail-grid">
    <dt>Execution</dt><dd>{h(execution_status)}</dd>
    <dt>Result</dt><dd>{h(result_status)}</dd>
    <dt>Boundary</dt><dd>{h(_plain_runtime_text(execution_result.get("side_effect_boundary") or "governed local record"))}</dd>
    <dt>External HTTP calls</dt><dd>{h(str(displayed_external_calls))}</dd>
  </dl>
</section>
"""
    raw_recovery = execution_result.get("recovery_path") or execution.get("recovery_path") or action.get("recovery_path")
    if isinstance(raw_recovery, list):
        raw_recovery = " ".join(str(value) for value in raw_recovery if isinstance(value, str))
    failure_reason = _plain_runtime_text(
        execution_result.get("message")
        or execution.get("message")
        or execution.get("reason")
        or "The recorded action attempt failed."
    )
    failure_recovery = _plain_runtime_text(
        raw_recovery
        or "Review the failure receipt and supporting sources in Inbox before creating a new action preview."
    )
    failure_panel = ""
    if is_failed:
        safety_note = (
            "External HTTP call count is not recorded; inspect the execution and audit receipts before retrying."
            if observed_external_calls is None
            else
            "No external HTTP call was recorded."
            if observed_external_calls == 0
            else f"{observed_external_calls} external HTTP call{'s were' if observed_external_calls != 1 else ' was'} recorded; review the audit receipt before retrying."
        )
        failure_panel = f"""
<section class="cs-panel" data-action-failure-recovery="true" data-product-state="failed-with-recovery">
  <div class="cs-panel-header">
    <div><h2>Action failed</h2><p class="cs-muted">The failed attempt remains a reviewable record; it is not presented as a new preview.</p></div>
    {_chip("Failed with recovery", "failed")}
  </div>
  <p><strong>Cause:</strong> {h(failure_reason)}</p>
  <p><strong>Recovery:</strong> {h(failure_recovery)}</p>
  <dl class="cs-detail-grid">
    <dt>Execution state</dt><dd>{h(execution_status)}</dd>
    <dt>Observed external calls</dt><dd>{h(str(displayed_external_calls))}</dd>
    <dt>What stayed safe</dt><dd>{h(safety_note)}</dd>
  </dl>
  <div class="cs-empty-actions">
    <a class="cs-button" href="/inbox">Open recovery queue</a>
    <a class="cs-button secondary" href="/audit">Open failure receipt</a>
  </div>
</section>
"""
    action_id = str(action.get("action_id") or "")
    source_claim_id = str(action.get("source_claim_id") or "")
    source_claim_link = (
        f'<a class="cs-button secondary" href="{h(_detail_href("claims", source_claim_id))}">Open supporting Claim</a>'
        if source_claim_id
        else '<span class="cs-muted">No supporting Claim lineage is recorded.</span>'
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
        resolution_values = denial_details.get("resolution_path") if isinstance(denial_details.get("resolution_path"), list) else []
        resolution = " ".join(str(value) for value in resolution_values if isinstance(value, str)) or "Review approval, source evidence, and the local safety boundary before retrying."
        denial_panel = f"""
<section class="cs-panel" data-action-execution-denial="true">
  <div class="cs-panel-header"><div><h2>Execution blocked</h2><p class="cs-muted">The Action remains non-executed and no external write is claimed.</p></div>{_chip("Policy blocked", "policyBlocked")}</div>
  <p><strong>Cause:</strong> {h(_plain_runtime_text(denial_details.get("reason") or "The safety gate denied execution."))}</p>
  <p><strong>Recovery:</strong> {h(_plain_runtime_text(resolution))}</p>
  <details class="cs-audit-detail"><summary>Safety receipt</summary><dl class="cs-detail-grid"><dt>Reason code</dt><dd>{h(str(denial_details.get("reason_code") or "Not recorded"))}</dd><dt>Safety envelope</dt><dd>{h(str(denial_details.get("action_safety_envelope_id") or "Not recorded"))}</dd><dt>External HTTP calls</dt><dd>{h(str(denial_details.get("external_http_calls", 0)))}</dd></dl></details>
</section>
"""
    blocked_state_panel = ""
    if is_blocked and not denial_events:
        blocked_state_panel = f"""
<section class="cs-panel" data-action-policy-blocked="true" data-product-state="policy-blocked">
  <div class="cs-panel-header">
    <div><h2>Action blocked</h2><p class="cs-muted">The recorded blocked state remains visible until its cause is resolved.</p></div>
    {_chip("Policy blocked", "policyBlocked")}
  </div>
  <p><strong>Cause:</strong> {h(blocked_reason)}</p>
  <p><strong>Recovery:</strong> {h(blocked_recovery)}</p>
  <div class="cs-empty-actions">
    <a class="cs-button" href="/inbox">Open review queue</a>
    <a class="cs-button secondary" href="/audit">Open block receipt</a>
  </div>
</section>
"""
    activity = _record_activity_panel(
        ctx,
        "action",
        action_id,
        [str(action.get("audit_ref"))] if action.get("audit_ref") else [],
    )
    summary_note = (
        "This durable result follows the original dry-run and approval record."
        if is_executed
        else "This failed attempt follows the original dry-run and recorded approval state."
        if is_failed
        else "This blocked record preserves the original preview and recorded cause."
        if is_blocked
        else "This is the proposed change, not an execution result."
    )
    impacted_label, impacted_state = (
        ("Execution recorded", "executed")
        if is_executed
        else ("Failed attempt recorded", "failed")
        if is_failed
        else ("Blocked", "policyBlocked")
        if is_blocked
        else ("Will be reviewed", "underReview")
    )
    proposed_change_note = (
        "The original before/after preview is retained with the execution result."
        if is_executed
        else "The original before/after preview is retained with the failed attempt."
        if is_failed
        else "The original preview is retained with the blocked state."
        if is_blocked
        else "Preview the before and after state before requesting approval."
    )
    external_calls_note = (
        f"The recorded result reports: {observed_call_note.lower()}."
        if has_execution_attempt
        else "This blocked state does not itself prove whether a provider call occurred."
        if is_blocked
        else "Provider writes remain simulated until approval is clear."
    )
    return f"""
<section
  class="cs-grid-two cs-action-workbench"
  data-product-surface="action-detail"
  data-approval-required="{str(approval_required).lower()}"
  data-execution-mode="{h(execution_mode)}"
  data-real-external-http-calls="{h(displayed_external_calls)}"
  data-expected-connector-calls="{h(expected_connector_calls)}"
  data-product-state="{h(lifecycle_stage)}"
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
          <div class="cs-kicker">{"Action result" if is_executed else "Action failed" if is_failed else "Action blocked" if is_blocked else "Approved action" if is_approved_stage else "Action preview"}</div>
          <h1>{h(action_title)}</h1>
          <div class="cs-brief-meta">
            <span>{h(lifecycle_meta)}</span>
            <span>{h(_display_date(action))}</span>
            <span>{h(external_meta)}</span>
          </div>
          <p class="cs-muted">{h(lifecycle_note)}</p>
        </div>
        <div class="cs-action-actions">
          <a class="cs-button secondary" href="/actions">Back to actions</a>
          <a class="cs-button secondary" href="/audit">Open audit trail</a>
        </div>
      </div>
      <div class="cs-brief-actions">
        {action_controls}
      </div>
    </header>
    {_action_approval_receipt(goal, diff, target, decision_label, approval_label, approval_status, risk_label, call_label, expected_connector_calls, planned_real_external_calls, displayed_policy_reason, lifecycle_stage=lifecycle_stage, approval_recorded=approval_recorded, observed_external_calls=observed_external_calls)}
    {execution_panel}
    {failure_panel}
    {denial_panel}
    {blocked_state_panel}
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>Summary</h2>
          <p class="cs-muted">{h(summary_note)}</p>
        </div>
        {_chip(label, state)}
      </div>
      <p>{h(goal)}</p>
      <div class="cs-action-summary">
        <div class="cs-action-metric"><span class="cs-meta">Trigger</span><span>Manual review</span></div>
        <div class="cs-action-metric"><span class="cs-meta">Target</span><span>{h(target)}</span></div>
        <div class="cs-action-metric"><span class="cs-meta">Policy</span><span>{h(decision_label)}</span></div>
      </div>
      <div class="cs-action-review-strip" aria-label="Action review status">
        <div class="cs-action-review-card"><span class="cs-meta">Risk level</span><strong>{h(risk_label)}</strong></div>
        <div class="cs-action-review-card"><span class="cs-meta">Approval</span><strong>{h(approval_label)}</strong></div>
        <div class="cs-action-review-card"><span class="cs-meta">External calls</span><strong>{h(call_label)}</strong></div>
        <div class="cs-action-review-card"><span class="cs-meta">Status</span><strong>{h(approval_status)}</strong></div>
      </div>
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header"><div><h2>Why this action</h2><p class="cs-muted">The proposal stays linked to the Claim and evidence that justify reviewing it.</p></div>{_chip("Source-linked", "searchable") if source_claim_id else _chip("Lineage missing", "insufficientEvidence")}</div>
      <p>{h(displayed_policy_reason)}</p>
      <div class="cs-review-box">{source_claim_link}</div>
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header"><h2>Impacted objects</h2>{_chip(impacted_label, impacted_state)}</div>
      <div class="cs-action-object-row">
        <span class="cs-action-object-icon" aria-hidden="true">A</span>
        <span>
          <strong>{h(target)}</strong>
          <span class="cs-meta">{h(connector_label)} / {h(call_label)}</span>
        </span>
        {_chip(impacted_label, impacted_state)}
      </div>
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header"><div><h2>Expected impact</h2><p class="cs-muted">This is the original dry-run expectation, not the observed execution receipt.</p></div>{_chip(execution_mode, state)}</div>
      <dl class="cs-detail-grid">
        <dt>Execution mode</dt><dd>{h(execution_mode)}</dd>
        <dt>Workspace mode</dt><dd>{h(workspace_mode)}</dd>
        <dt>Target</dt><dd>{h(target)}</dd>
        <dt>Planned connector calls</dt><dd>{h(expected_connector_calls)}</dd>
        <dt>Planned real external HTTP calls</dt><dd>{h(planned_real_external_calls)}</dd>
      </dl>
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>Proposed changes</h2>
          <p class="cs-muted">{h(proposed_change_note)}</p>
        </div>
        {_chip("Diff preview", "searchable")}
      </div>
      {_action_diff_view(diff, target, is_executed=has_execution_attempt, approval_recorded=approval_recorded)}
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header">
        <div>
          <h2>External calls</h2>
          <p class="cs-muted">{h(external_calls_note)}</p>
        </div>
        {_chip(call_label, state)}
      </div>
      {_action_external_calls(impact, connector_label, call_label, lifecycle_stage=lifecycle_stage, approval_recorded=approval_recorded, result=execution_result, observed_external_calls=observed_external_calls)}
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
        <span>{h(displayed_policy_reason)}</span>
      </div>
      {_action_policy_checks(bool(source_items), approval_label, call_label, lifecycle_stage=lifecycle_stage, approval_recorded=approval_recorded)}
      <details class="cs-audit-detail"><summary>Policy receipt</summary><p><code>{h(str(policy.get("id") or "No policy ref recorded"))}</code></p></details>
    </section>
    <section class="cs-panel">
      <div class="cs-panel-header"><div><h2>Safety check</h2><p class="cs-muted">Connector mediation, credentials, and live-call evidence remain explicit.</p></div>{_chip("Local mock" if connector.get("mocked") is True else "Review boundary", "searchable")}</div>
      <dl class="cs-detail-grid">
        <dt>Connector mediated</dt><dd>{"Yes" if connector.get("direct_provider_access") is False else "Needs review"}</dd>
        <dt>Credentials exposed</dt><dd>{"No" if connector.get("credentials_exposed_to_agent") is False else "Needs review"}</dd>
        <dt>Mocked</dt><dd>{"Yes" if connector.get("mocked") is True else "No"}</dd>
        <dt>Live writes observed</dt><dd>{h(displayed_external_calls)}</dd>
      </dl>
    </section>
    {_action_route_strip(approval_label, call_label, lifecycle_stage=lifecycle_stage, approval_recorded=approval_recorded)}
  </div>
  <aside class="cs-stack cs-action-rail">
    <section class="cs-panel flat">
      <h2 class="cs-section-title">{"Risk and lifecycle" if is_failed or is_blocked else "Risk and approval"}</h2>
      <dl class="cs-detail-grid">
        <dt>Risk level</dt><dd>{h(risk_label)}</dd>
        <dt>Approval</dt><dd>{h(approval_label)}</dd>
        <dt>Status</dt><dd>{h(approval_status)}</dd>
      </dl>
      <div class="cs-review-box">
        {rail_control}
        <div class="cs-approval-note">
          <strong>{h(rail_reason_label)}</strong>
          <span>{h(rail_reason)}</span>
        </div>
        <p class="cs-muted">{h(rail_note)}</p>
      </div>
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Approval history</h2>
      {approval_history}
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Sources</h2>
      {source_list}
    </section>
    <section class="cs-panel flat">
      <h2 class="cs-section-title">Auditability</h2>
      <p class="cs-muted">Dry-run, policy, approval, and execution records remain separately inspectable.</p>
    </section>
    {activity}
  </aside>
</section>
"""


def _action_approval_receipt(
    goal: str,
    diff: dict[str, Any],
    target: str,
    decision_label: str,
    approval_label: str,
    approval_status: str,
    risk_label: str,
    call_label: str,
    expected_connector_calls: int,
    planned_real_external_calls: int,
    policy_reason: str,
    *,
    lifecycle_stage: str = "draft",
    approval_recorded: bool = False,
    observed_external_calls: int | None = None,
) -> str:
    before = _plain_runtime_text(diff.get("before") or "No side effect applied.")
    after = _plain_runtime_text(diff.get("after") or "No provider send has been performed.")
    has_execution_attempt = lifecycle_stage in {"executed", "failed"}
    call_note = (
        "The execution receipt does not record an external HTTP call count."
        if has_execution_attempt and observed_external_calls is None
        else "The execution receipt records no external HTTP calls."
        if has_execution_attempt and observed_external_calls == 0
        else f"{observed_external_calls} external HTTP call{'s were' if observed_external_calls != 1 else ' was'} recorded; inspect the audit receipt before any retry."
        if has_execution_attempt
        else "The blocked record does not include an external HTTP call count."
        if lifecycle_stage == "blocked" and observed_external_calls is None
        else f"The blocked record reports {observed_external_calls} external HTTP call{'s' if observed_external_calls != 1 else ''}."
        if lifecycle_stage == "blocked"
        else "No provider send has run in this local workspace."
        if planned_real_external_calls == 0
        else "A provider send still requires the approval record and audit trail."
    )
    receipt_title = "Action lifecycle receipt" if lifecycle_stage in {"executed", "failed", "blocked"} else "Dry-run approval receipt"
    receipt_note = (
        "The original preview, policy, approval, and durable execution result remain visible together."
        if lifecycle_stage == "executed"
        else "The original preview, approval state, and failed execution receipt remain visible together."
        if lifecycle_stage == "failed"
        else "The original preview and recorded block remain visible together."
        if lifecycle_stage == "blocked"
        else "Preview only. Impact, proposed change, provider call plan, policy, and approval gate are visible before execution."
    )
    receipt_chip = (
        _chip("Executed", "executed")
        if lifecycle_stage == "executed"
        else _chip("Failed", "failed")
        if lifecycle_stage == "failed"
        else _chip("Blocked", "policyBlocked")
        if lifecycle_stage == "blocked"
        else _chip("Approved", "approved")
        if approval_recorded
        else _chip("Dry-run", "draft")
    )
    call_heading = "External call evidence" if has_execution_attempt else "External call plan"
    return f"""
<section class="cs-action-receipt-panel" aria-label="{h(receipt_title)}">
  <div class="cs-panel-header">
    <div>
      <h2>{h(receipt_title)}</h2>
      <p class="cs-muted">{h(receipt_note)}</p>
    </div>
    {receipt_chip}
  </div>
  <div class="cs-action-receipt-grid">
    <div class="cs-action-receipt-card">
      <span class="cs-meta">Proposed change</span>
      <strong>{h(goal)}</strong>
      <p>Target: {h(target)}</p>
    </div>
    <div class="cs-action-receipt-card">
      <span class="cs-meta">Proposed change preview</span>
      <div class="cs-action-mini-diff">
        <div><span class="cs-meta">Before</span><strong>{h(before)}</strong></div>
        <div><span class="cs-meta">After</span><strong>{h(after)}</strong></div>
      </div>
    </div>
    <div class="cs-action-receipt-card">
      <span class="cs-meta">{h(call_heading)}</span>
      <strong>{h(expected_connector_calls)} connector call{"s" if expected_connector_calls != 1 else ""}</strong>
      <p>{h(call_label)}. {h(call_note)}</p>
    </div>
    <div class="cs-action-receipt-card">
      <span class="cs-meta">Approval gate</span>
      <strong>{h(approval_label)}</strong>
      <p>{h(risk_label)} risk / {h(approval_status)}. Policy: {h(decision_label)}.</p>
    </div>
  </div>
  <p class="cs-muted">Policy reason: {h(policy_reason)}</p>
</section>
"""


def _action_route_strip(
    approval_label: str,
    call_label: str,
    *,
    lifecycle_stage: str = "draft",
    approval_recorded: bool = False,
) -> str:
    if lifecycle_stage == "executed":
        steps = [
            ("1", "Dry-run recorded", "The original proposed change remains inspectable.", "searchable", False),
            ("2", "Impact reviewed", "The affected object and before/after state remain visible.", "saved", False),
            ("3", approval_label, "Owner approval is recorded separately from execution.", "approved", False),
            ("4", "Execution and audit", f"{call_label}; the durable result is inspectable.", "executed", True),
        ]
    elif lifecycle_stage == "failed":
        steps = [
            ("1", "Dry-run retained", "The original proposed change remains inspectable.", "searchable", False),
            ("2", approval_label, "The approval record remains separate from execution." if approval_recorded else "No approval is recorded.", "approved" if approval_recorded else "underReview", False),
            ("3", "Failed attempt", f"{call_label}; review the execution receipt.", "failed", True),
            ("4", "Recovery and audit", "Inspect the failure and audit receipts before retrying.", "underReview", False),
        ]
    elif lifecycle_stage == "blocked":
        steps = [
            ("1", "Dry-run retained", "The original proposed change remains inspectable.", "searchable", False),
            ("2", approval_label, "Approval remains separately recorded." if approval_recorded else "No approval is recorded.", "approved" if approval_recorded else "underReview", False),
            ("3", "Blocked state", "Resolve the recorded cause before creating a new preview.", "policyBlocked", True),
            ("4", "Block receipt", "The block and recovery path remain inspectable.", "underReview", False),
        ]
    else:
        steps = [
            ("1", "Dry-run sequence", "Preview the proposed change before any provider write.", "searchable", True),
            ("2", "Impact review", "Check the affected object and the before/after state.", "underReview", False),
            (
                "3",
                approval_label,
                "Owner approval is recorded; execution is still pending." if approval_recorded else "Approval stays explicit before execution can appear.",
                "approved" if approval_recorded else "underReview",
                approval_recorded,
            ),
            ("4", "Audit trail", f"{call_label}; records stay inspectable.", "draft", False),
        ]
    cards = "".join(
        f"""
<div class="cs-action-route-step{" is-current" if active else ""}">
  <div class="cs-action-route-top">
    <span class="cs-action-route-index" aria-hidden="true">{h(index)}</span>
    <strong>{h(title)}</strong>
  </div>
  <p>{h(description)}</p>
  {_chip(title if index == "1" else index, state)}
</div>
"""
        for index, title, description, state, active in steps
    )
    return f"""
<section class="cs-action-route-strip" aria-label="{'Action lifecycle' if lifecycle_stage in {'executed', 'failed', 'blocked'} else 'Dry-run sequence'}">
  {cards}
</section>
"""


def _claim_trust_ladder(has_sources: bool, evidence_backed_earned: bool, is_approved: bool) -> str:
    source_class = "is-active" if has_sources else "is-locked"
    evidence_class = "is-active" if evidence_backed_earned else "is-locked"
    approved_class = "is-active" if is_approved else "is-locked"
    source_note = "Supporting source is attached." if has_sources else "Attach at least one source."
    evidence_note = "Citation checks earned this label." if evidence_backed_earned else "Locked until citation checks pass."
    approved_note = "Owner approval recorded." if is_approved else "Requires review first."
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


def _action_diff_view(
    diff: dict[str, Any],
    target: str,
    *,
    is_executed: bool = False,
    approval_recorded: bool = False,
) -> str:
    before = _plain_runtime_text(diff.get("before") or "No side effect applied.")
    after = _plain_runtime_text(diff.get("after") or "No external write has been performed.")
    return f"""
<div class="cs-action-preview-frame">
  <div class="cs-action-preview-meta">
    <span>Target: {h(target)}</span>
    <span>{"Original dry-run preview retained" if is_executed or approval_recorded else "Before approval: preview only"}</span>
  </div>
<div class="cs-diff-view" aria-label="Dry-run diff preview">
  <div class="cs-diff-line before"><span class="cs-meta">Before</span><span>{h(before)}</span></div>
  <div class="cs-diff-line after"><span class="cs-meta">After</span><span>{h(after)}</span></div>
</div>
  <p class="cs-meta">{"Execution is recorded separately from this original preview." if is_executed else "Preview shown. Exact downstream formatting may vary after approval."}</p>
</div>
"""


def _action_external_calls(
    impact: dict[str, Any],
    connector_label: str,
    call_label: str,
    *,
    lifecycle_stage: str = "draft",
    approval_recorded: bool = False,
    result: dict[str, Any] | None = None,
    observed_external_calls: int | None = None,
) -> str:
    expected = int(impact.get("expected_connector_calls", 0) or 0)
    target = str(impact.get("target") or "Local preview only.")
    result = result or {}
    if lifecycle_stage in {"executed", "failed"}:
        boundary = _plain_runtime_text(result.get("side_effect_boundary") or "Not recorded")
        observed = "Not recorded" if observed_external_calls is None else str(observed_external_calls)
        result_state = "executed" if lifecycle_stage == "executed" else "failed"
        result_label = "Execution recorded" if lifecycle_stage == "executed" else "Failed attempt recorded"
        return f"""
<div class="cs-call-row">
  <div>
    <strong>{h(call_label)}</strong>
    <p class="cs-muted">The execution receipt records the boundary and observed call count; missing telemetry is not treated as zero.</p>
  </div>
  {_chip(result_label, result_state)}
</div>
<div class="cs-call-facts" aria-label="Call result">
  <div class="cs-call-fact"><strong>Planned connector calls</strong><span>{h(expected)}</span></div>
  <div class="cs-call-fact"><strong>Recorded HTTP calls</strong><span>{h(observed)}</span></div>
  <div class="cs-call-fact"><strong>Boundary</strong><span>{h(boundary)}</span></div>
</div>
"""
    if lifecycle_stage == "blocked":
        observed = "Not recorded" if observed_external_calls is None else str(observed_external_calls)
        return f"""
<div class="cs-call-row">
  <div>
    <strong>{h(call_label)}</strong>
    <p class="cs-muted">The blocked state is recorded separately from any provider-call evidence.</p>
  </div>
  {_chip("Blocked", "policyBlocked")}
</div>
<div class="cs-call-facts" aria-label="Call result">
  <div class="cs-call-fact"><strong>Planned connector calls</strong><span>{h(expected)}</span></div>
  <div class="cs-call-fact"><strong>Recorded HTTP calls</strong><span>{h(observed)}</span></div>
  <div class="cs-call-fact"><strong>Boundary</strong><span>Inspect block receipt</span></div>
</div>
"""
    if approval_recorded:
        return f"""
<div class="cs-call-row">
  <div>
    <strong>{h(connector_label)}</strong>
    <p class="cs-muted">The connector plan is approved but no execution result is recorded yet for {h(target)}.</p>
  </div>
  {_chip("Execution pending", "underReview")}
</div>
<div class="cs-call-facts" aria-label="Call preview">
  <div class="cs-call-fact"><strong>Provider calls</strong><span>{h(expected)} approved</span></div>
  <div class="cs-call-fact"><strong>Target</strong><span>{h(target)}</span></div>
  <div class="cs-call-fact"><strong>Boundary</strong><span>Governed execution pending</span></div>
</div>
"""
    if expected <= 0:
        return f"""
<div class="cs-call-row">
  <div>
    <strong>No external connector call planned</strong>
    <p class="cs-muted">This preview records the policy and audit envelope without a provider send.</p>
  </div>
  {_chip(call_label, "draft")}
</div>
<div class="cs-call-facts" aria-label="Call preview">
  <div class="cs-call-fact"><strong>Provider calls</strong><span>0 planned</span></div>
  <div class="cs-call-fact"><strong>Target</strong><span>{h(target)}</span></div>
  <div class="cs-call-fact"><strong>Boundary</strong><span>{h(call_label)}</span></div>
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
<div class="cs-call-facts" aria-label="Call preview">
  <div class="cs-call-fact"><strong>Provider calls</strong><span>{h(expected)} planned</span></div>
  <div class="cs-call-fact"><strong>Target</strong><span>{h(target)}</span></div>
  <div class="cs-call-fact"><strong>Boundary</strong><span>{h(call_label)}</span></div>
</div>
"""


def _action_policy_checks(
    has_sources: bool,
    approval_label: str,
    call_label: str,
    *,
    lifecycle_stage: str = "draft",
    approval_recorded: bool = False,
) -> str:
    source_state = "Source visible" if has_sources else "Source not linked"
    source_note = "At least one local source is visible near this action." if has_sources else "Open sources before requesting approval."
    checks = [
        ("1", source_state, source_note, "searchable" if has_sources else "underReview"),
        (
            "2",
            approval_label,
            "Owner approval is recorded." if approval_recorded else "Request approval before this preview can move further.",
            "approved" if approval_recorded else "underReview",
        ),
        (
            "3",
            call_label,
            "The recorded execution remains inspectable through the governed Action path."
            if lifecycle_stage == "executed"
            else "The failed attempt and its call evidence remain inspectable."
            if lifecycle_stage == "failed"
            else "The blocked state remains inspectable before another preview."
            if lifecycle_stage == "blocked"
            else "External effects are still bounded by the action approval path.",
            "executed"
            if lifecycle_stage == "executed"
            else "failed"
            if lifecycle_stage == "failed"
            else "policyBlocked"
            if lifecycle_stage == "blocked"
            else "draft",
        ),
    ]
    rows = "".join(
        f"""
<div class="cs-policy-check">
  <span class="cs-policy-check-mark" aria-hidden="true">{h(index)}</span>
  <div>
    <strong>{h(title)}</strong>
    <p class="cs-muted">{h(note)}</p>
  </div>
  {_chip(title, state)}
</div>
"""
        for index, title, note, state in checks
    )
    return f"""
<div class="cs-policy-checks" aria-label="Policy checkpoints">
  {rows}
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
  <span class="cs-checkmark" aria-hidden="true">S</span>
  <span>
    <strong>{h(item["title"])}</strong>
    <span class="cs-meta">{h(item["label"])} / {h(item["date"])} / {h(item["fingerprint"])}</span>
    <span class="cs-muted">{h(_truncate(item["snippet"], 120))}</span>
  </span>
</a>
"""
        )
    return f'<div class="cs-evidence-picker" aria-label="Supporting source links">{"".join(rows)}</div>'


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
    <a class="cs-button ghost" href="/audit">Audit trail</a>
    {_chip("Searchable", "searchable")}
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
        start = span.get("start")
        end = span.get("end")
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
        return item
    return None


def _citation_disclosure_for_refs(refs: list[str], source_items: list[dict[str, str]], record: dict[str, Any] | None = None) -> str:
    record = record or {}
    brief_id = str(record.get("brief_id") or "")
    audit_refs = [str(ref) for ref in record.get("audit_refs", []) if isinstance(ref, str)]
    receipt_ref = audit_refs[0] if audit_refs else "No creation audit ref recorded"
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
        open_attr = " open" if index == 1 else ""
        ref_kind = item.get("ref_kind") or _citation_ref_kind(item.get("ref", ""))
        span = item.get("span") or "Whole source"
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
    <p class="cs-citation-snippet"><strong>Source snippet:</strong> {h(_truncate(item["snippet"], 260))}</p>
    <p class="cs-muted"><strong>Why this supports the finding:</strong> this recorded source span is the cited support for this draft finding; inspect it before use.</p>
    <div class="cs-citation-meta" aria-label="Full provenance">
      <div><span class="cs-meta">Citation ref</span><strong>{h(ref_kind)}</strong></div>
      <div><span class="cs-meta">Source span</span><strong>{h(span)}</strong></div>
      <div><span class="cs-meta">Saved</span><strong>{h(item["date"])}</strong></div>
      <div><span class="cs-meta">Fingerprint</span><strong>{h(item["fingerprint"])}</strong></div>
      <div><span class="cs-meta">Related object</span><strong>{h(f'Brief {brief_id}' if brief_id else 'Brief draft')}</strong></div>
      <div><span class="cs-meta">Audit receipt</span><strong>{h(receipt_ref)}</strong></div>
    </div>
    <div class="cs-citation-actions">
      <a class="cs-button secondary" href="{h(item["href"])}">Open source</a>
      <a class="cs-button ghost" href="/audit">Audit trail</a>
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
  const askForm = document.getElementById("cs-ask-form");
  const askInput = document.getElementById("cs-ask-input");
  const askStatus = document.getElementById("cs-ask-status");
  const askButton = document.getElementById("cs-ask-submit-button");
  const claimButton = document.getElementById("cs-create-claim-button");
  const claimStatus = document.getElementById("cs-claim-create-status");
  function setStatus(node, message, state) {
    if (!node) return;
    const status = state || "idle";
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
  async function saveText(text, sourceRef) {
    setBusy(dropForm, saveButton, true, "Saving", "Save source");
    setStatus(dropStatus, "Saving source...", "loading");
    try {
      const payload = {};
      payload.text = text;
      payload["source" + "_ref"] = sourceRef || "home-paste";
      const saved = await postJson("/artifacts", payload);
      const artifact = saved.artifact || {};
      const id = artifact["artifact" + "_id"];
      setStatus(dropStatus, "Saved. Opening source...", "success");
      if (id) window.location.href = scopedUrl("/artifacts/" + encodeURIComponent(id) + "?view=html");
    } catch (error) {
      setStatus(dropStatus, error.message, "error");
    } finally {
      setBusy(dropForm, saveButton, false, "Saving", "Save source");
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
      const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
      if (!file) return;
      file.text().then(text => saveText(text, file.name)).catch(error => setStatus(dropStatus, error.message, "error"));
    });
  }
  if (fileButton && fileInput) {
    fileButton.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", function () {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;
      file.text().then(text => saveText(text, file.name)).catch(error => setStatus(dropStatus, error.message, "error"));
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
        setStatus(claimStatus, "This Brief needs a supported finding before a Claim can be drafted.", "error");
        return;
      }
      claimButton.disabled = true;
      setStatus(claimStatus, "Drafting Claim candidate...", "loading");
      try {
        const created = await postJson("/claims", {brief_id: briefId, statement});
        const claim = created.claim || {};
        const claimId = claim["claim" + "_id"];
        if (!claimId) throw new Error("Claim candidate was not saved.");
        setStatus(claimStatus, "Claim candidate saved. Opening draft...", "success");
        window.location.href = scopedUrl("/claims/" + encodeURIComponent(claimId) + "?view=html");
      } catch (error) {
        setStatus(claimStatus, error.message, "error");
        claimButton.disabled = false;
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
      setBusy(askForm, askButton, true, "Checking", "Ask");
      try {
        setStatus(askStatus, "Checking saved sources...", "loading");
        const searched = await postJson("/search", {query: question, excluded_source_types: ["conversation_turn"]});
        const sourceSnapshot = searched.search_snapshot || {};
        const started = await postJson("/conversations", {message: question});
        const conversation = started.conversation || {};
        const id = conversation["conversation" + "_id"];
        if (!id) throw new Error("Conversation was not saved.");
        const answered = await postJson("/conversations/" + encodeURIComponent(id) + "/answers", {question});
        const answer = answered.answer || {};
        const text = answer.answer || "Draft answer saved. Open the linked sources before using it.";
        const safety = conversation.safety || {};
        if (safety.unsafe_instruction_detected === true) {
          setStatus(askStatus, "Draft answer saved. Brief preparation was blocked because the Ask text contained unsafe instructions.", "error");
          return;
        }
        if (Number(sourceSnapshot.result_count || 0) > 0 && sourceSnapshot.search_snapshot_id) {
          setStatus(askStatus, "Preparing a source-linked fallback Brief...", "loading");
          const bundled = await postJson("/evidence-bundles", {search_snapshot_id: sourceSnapshot.search_snapshot_id});
          const bundle = bundled.evidence_bundle || {};
          if (!bundle.evidence_bundle_id) throw new Error("Supporting evidence was not saved.");
          const prepared = await postJson("/briefs", {evidence_bundle_id: bundle.evidence_bundle_id});
          const brief = prepared.brief || {};
          if (!brief.brief_id) throw new Error("Brief draft was not saved.");
          setStatus(askStatus, "Keyword-summary Brief prepared. Opening draft...", "success");
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
""".replace("__SCOPE__", _script_json(safe_scope))

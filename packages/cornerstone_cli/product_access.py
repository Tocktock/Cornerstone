from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from cornerstone_cli.validators import redact_text


WORKSPACE_SEARCH_TYPES = ("artifact", "brief", "claim", "action")
EVIDENCE_SEARCH_TYPES = ("artifact", "ontology_object")
SEARCH_TYPE_ALIASES = {
    "all": None,
    "sources": "artifact",
    "artifacts": "artifact",
    "briefs": "brief",
    "claims": "claim",
    "actions": "action",
}


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    scope: dict[str, str]
    mode: str = "evidence"
    type_filter: str = "all"
    page: int = 1
    page_size: int = 20
    snapshot_id: str | None = None
    excluded_source_types: frozenset[str] = frozenset()
    included_artifact_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class RecordReadRequest:
    record_kind: str
    record_id: str
    scope: dict[str, str]
    reason: str


@dataclass(frozen=True, slots=True)
class CollectionViewRequest:
    surface: str
    scope: dict[str, str]
    filters: dict[str, str]


class ProductAccessApplication:
    """Deep product-access module shared by HTML, JSON, CLI, and Ask adapters."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def search(self, request: SearchRequest) -> dict[str, Any]:
        query = request.query.strip()
        if not query:
            return {"status": "query_required"}
        mode = request.mode.strip().lower()
        if mode not in {"workspace", "evidence"}:
            return {"status": "invalid_mode", "mode": mode}
        type_filter = request.type_filter.strip().lower()
        if type_filter not in SEARCH_TYPE_ALIASES:
            return {"status": "invalid_type", "type_filter": type_filter}
        if mode == "evidence" and type_filter not in {"all", "sources", "artifacts"}:
            return {
                "status": "invalid_type_for_mode",
                "mode": mode,
                "type_filter": type_filter,
            }
        result_types = set(WORKSPACE_SEARCH_TYPES if mode == "workspace" else EVIDENCE_SEARCH_TYPES)
        created = request.snapshot_id is None
        if request.snapshot_id:
            loaded = self.store.read_search_snapshot(
                request.snapshot_id,
                request.scope,
                reason=f"{mode}_search_page",
            )
            if loaded.get("status"):
                return loaded
            snapshot = loaded["snapshot"]
            audit_event = loaded["audit_event"]
            persisted_query = str(snapshot.get("query") or "")
            query_digest = str(snapshot.get("query_sha256") or "").strip().lower()
            if query_digest:
                # Current snapshots bind pagination to the exact raw query.  The
                # redacted display value is deliberately not an identity key:
                # two different secrets can both persist as ``[REDACTED]``.
                query_matches = query_digest == hashlib.sha256(query.encode("utf-8")).hexdigest()
            else:
                # Legacy snapshots predate query_sha256.  Reuse is safe only
                # when redaction did not alter the query, so equality remains
                # exact rather than collapsing secret-bearing inputs.
                redacted_query = redact_text(query)
                query_matches = (
                    query == redacted_query
                    and "[REDACTED]" not in persisted_query
                    and persisted_query == query
                )
            if not query_matches:
                return {"status": "snapshot_query_mismatch", "search_snapshot": snapshot}
            if str(snapshot.get("search_mode") or "evidence") != mode:
                return {"status": "snapshot_mode_mismatch", "search_snapshot": snapshot}
        else:
            result = self.store.search(
                query,
                **request.scope,
                excluded_source_types=set(request.excluded_source_types),
                included_artifact_ids=set(request.included_artifact_ids),
                result_types=result_types,
                search_mode=mode,
            )
            snapshot = result["snapshot"]
            audit_event = result["audit_event"]

        all_results = [row for row in snapshot.get("results", []) if isinstance(row, dict)]
        requested_type = SEARCH_TYPE_ALIASES[type_filter]
        projected = [row for row in all_results if requested_type is None or row.get("result_type") == requested_type]
        page_size = min(max(int(request.page_size or 20), 1), 100)
        page_count = max(1, (len(projected) + page_size - 1) // page_size)
        current_page = min(max(int(request.page or 1), 1), page_count)
        start = (current_page - 1) * page_size
        page_results = projected[start : start + page_size]
        facets = {
            "all": len(all_results),
            "sources": sum(1 for row in all_results if row.get("result_type") == "artifact"),
            "briefs": sum(1 for row in all_results if row.get("result_type") == "brief"),
            "claims": sum(1 for row in all_results if row.get("result_type") == "claim"),
            "actions": sum(1 for row in all_results if row.get("result_type") == "action"),
        }
        evidence_refs = [f"search_snapshot:{snapshot['search_snapshot_id']}"]
        for row in all_results:
            evidence_refs.extend(str(ref) for ref in row.get("evidence_refs", []) if isinstance(ref, str))
        return {
            "status": "success",
            "search_snapshot": snapshot,
            "results": page_results,
            "ordered_result_refs": [str(row.get("result_ref") or "") for row in projected],
            "facets": facets,
            "type_filter": type_filter,
            "result_count": len(projected),
            "page": current_page,
            "page_size": page_size,
            "page_count": page_count,
            "page_start": start + 1 if projected else 0,
            "page_end": min(start + page_size, len(projected)),
            "created_new_snapshot": created,
            "evidence_refs": list(dict.fromkeys(evidence_refs)),
            "audit_refs": [f"audit:{audit_event['event_id']}"],
            "audit_event": audit_event,
        }

    def read(self, request: RecordReadRequest) -> dict[str, Any]:
        return self.store.read_product_record(
            request.record_kind,
            request.record_id,
            request.scope,
            reason=request.reason,
        )

    def view_collection(self, request: CollectionViewRequest) -> dict[str, Any]:
        surface = request.surface.strip().lower().replace("/", "-").strip("-") or "home"
        event = self.store.append_audit(
            "product.collection.read",
            request.scope,
            {"type": "product_surface", "id": surface},
            {
                "surface": surface,
                "filters": {
                    str(key)[:64]: redact_text(str(value))[:256]
                    for key, value in sorted(request.filters.items())
                },
                "reason": "product_ui_collection",
            },
        )
        return {"status": "success", "surface": surface, "audit_event": event, "audit_refs": [f"audit:{event['event_id']}"]}

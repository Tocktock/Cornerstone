from __future__ import annotations

from typing import Any


INTERNAL_PRODUCT_VISIBILITIES = {"internal", "owner_only", "verification_only"}
INTERNAL_PRODUCT_SOURCE_TYPES = {
    "internal_fixture",
    "local_fixture",
    "scenario_fixture",
    "verification_fixture",
}
CONTEXT_ONLY_SOURCE_TYPES = {"conversation_turn"}
PRODUCT_RECORD_ID_KEYS = ("artifact_id", "brief_id", "claim_id", "action_id", "memory_id")


def record_identity_refs(record: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in PRODUCT_RECORD_ID_KEYS:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            continue
        kind = key.removesuffix("_id")
        refs.update({value, f"{kind}:{value}"})
    return refs


def record_lineage_refs(record: dict[str, Any]) -> set[str]:
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


def internal_product_record(record: dict[str, Any], internal_refs: set[str] | None = None) -> bool:
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
    return explicitly_internal or bool(internal_refs and record_lineage_refs(record) & internal_refs)


def internal_product_lineage(
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
            if object_id in internal_record_objects or not internal_product_record(record, internal_refs):
                continue
            internal_record_objects.add(object_id)
            internal_refs.update(record_identity_refs(record))
            changed = True
    return internal_refs, internal_record_objects


def context_only_artifact(record: dict[str, Any]) -> bool:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    source_type = str(source.get("type") or source.get("source_type") or "").lower()
    return source_type in CONTEXT_ONLY_SOURCE_TYPES

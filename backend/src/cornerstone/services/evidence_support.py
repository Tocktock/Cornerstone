from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol, TypeVar, cast

from cornerstone.schemas import CitationSupportRef, DecisionRecord, EvidenceFragment
from cornerstone.services.source_eligibility import (
    source_can_serve_captured_evidence,
    source_serving_limitations,
)
from cornerstone.store import NotFoundError



class SupportedEvidence(Protocol):
    fragment: EvidenceFragment
    supports: tuple[CitationSupportRef, ...]


SupportedEvidenceT = TypeVar("SupportedEvidenceT", bound=SupportedEvidence)


def dedupe_supported_evidence(items: list[SupportedEvidenceT]) -> list[SupportedEvidenceT]:
    """Merge duplicate evidence fragments while preserving all support references."""
    merged: dict[str, SupportedEvidenceT] = {}
    for item in items:
        existing = merged.get(item.fragment.id)
        if existing is None:
            merged[item.fragment.id] = item
            continue
        support_key = {(support.entity_type, support.entity_id, support.relationship) for support in existing.supports}
        supports = list(existing.supports)
        for support in item.supports:
            key = (support.entity_type, support.entity_id, support.relationship)
            if key not in support_key:
                supports.append(support)
                support_key.add(key)
        merged[item.fragment.id] = cast(SupportedEvidenceT, replace(existing, supports=tuple(supports)))
    return list(merged.values())


def decision_record_or_none(store: Any, decision_record_id: str) -> DecisionRecord | None:
    """Load a DecisionRecord when present without leaking store exceptions to service logic."""
    try:
        return cast(DecisionRecord, store.get_decision_record(decision_record_id))
    except NotFoundError:
        return None


def citation_validity_errors(store: Any, fragment: EvidenceFragment) -> list[str]:
    """Return deterministic citation/provenance validation errors for an evidence fragment."""
    errors: list[str] = []
    if not fragment.artifact_id:
        errors.append("missing_artifact_id")
    if not fragment.provenance.data_source_id:
        errors.append("missing_provenance_data_source_id")
    if not fragment.provenance.source_external_id:
        errors.append("missing_provenance_source_external_id")
    if not fragment.provenance.artifact_title:
        errors.append("missing_provenance_artifact_title")
    try:
        artifact = store.get_artifact(fragment.artifact_id)
    except NotFoundError:
        errors.append("missing_artifact")
        return errors
    if artifact.id != fragment.artifact_id:
        errors.append("artifact_mismatch")
    if artifact.datasource_id != fragment.provenance.data_source_id:
        errors.append("artifact_source_mismatch")
    try:
        store.get_data_source(artifact.datasource_id)
    except NotFoundError:
        errors.append("missing_data_source")
    return errors


def is_from_servable_source(store: Any, fragment: EvidenceFragment, *, production_mode: bool) -> bool:
    """Return whether a fragment belongs to a source that may serve captured evidence."""
    try:
        artifact = store.get_artifact(fragment.artifact_id)
        source = store.get_data_source(artifact.datasource_id)
    except NotFoundError:
        return False
    return source_can_serve_captured_evidence(source, production_mode=production_mode)


def source_limitations_for(store: Any, evidence: list[EvidenceFragment]) -> list[str]:
    """Collect source-level serving limitations once per source for a set of evidence."""
    limitations: list[str] = []
    seen_source_ids: set[str] = set()
    for fragment in evidence:
        try:
            artifact = store.get_artifact(fragment.artifact_id)
            if artifact.datasource_id in seen_source_ids:
                continue
            source = store.get_data_source(artifact.datasource_id)
        except NotFoundError:
            continue
        seen_source_ids.add(artifact.datasource_id)
        limitations.extend(source_serving_limitations(source))
    return limitations

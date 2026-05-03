from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from cornerstone.schemas import (
    Artifact,
    EvidenceFragment,
    EvidenceFragmentType,
    Provenance,
    QuoteRange,
)

_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+[.!?]?", re.MULTILINE)


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def extract_evidence_fragments(artifact: Artifact, content: str) -> list[EvidenceFragment]:
    fragments: list[EvidenceFragment] = []
    for text, start_offset, end_offset in _iter_fragments(content):
        fragments.append(
            EvidenceFragment(
                artifact_id=artifact.id,
                text=text,
                fragment_type=classify_fragment(text),
                provenance=Provenance(
                    data_source_id=artifact.datasource_id,
                    source_type=artifact.source_type,
                    source_external_id=artifact.source_external_id,
                    source_url=artifact.source_url,
                    artifact_title=artifact.title,
                    captured_at=artifact.captured_at,
                    source_updated_at=artifact.source_updated_at,
                    quote_range=QuoteRange(start_offset=start_offset, end_offset=end_offset),
                ),
                freshness_state=artifact.freshness_state,
            )
        )
    return fragments


def classify_fragment(text: str) -> EvidenceFragmentType:
    normalized = text.lower().strip()
    if normalized.endswith("?"):
        return EvidenceFragmentType.OPEN_QUESTION
    if any(token in normalized for token in ("decided", "decision", "we chose", "approved")):
        return EvidenceFragmentType.DECISION
    if any(token in normalized for token in ("must", "shall", "required", "requirement")):
        return EvidenceFragmentType.REQUIREMENT
    if any(token in normalized for token in ("policy", "rule", "governance")):
        return EvidenceFragmentType.POLICY
    if any(token in normalized for token in ("example", "for instance", "such as")):
        return EvidenceFragmentType.EXAMPLE
    if any(token in normalized for token in (" is ", " means ", " are ", " refers to ")):
        return EvidenceFragmentType.DEFINITION
    return EvidenceFragmentType.CLAIM


def _iter_fragments(content: str) -> Iterable[tuple[str, int, int]]:
    for match in _SENTENCE_PATTERN.finditer(content):
        text = match.group(0).strip()
        if not text:
            continue
        start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        end = start + len(text)
        yield text, start, end

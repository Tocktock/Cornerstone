from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from cornerstone.schemas import EvidenceCitation, FreshnessState, FreshnessSummary, TrustLabel


class _HasEvidence(Protocol):
    evidence: Sequence[EvidenceCitation]
    trust_label: TrustLabel


def freshness_policy_respected(
    *,
    freshness: FreshnessSummary,
    trust_label: TrustLabel,
    expected_freshness_state: FreshnessState | None = None,
) -> bool:
    """Shared freshness/trust-label rule for grounded-context and ontology-graph evals."""
    if expected_freshness_state is not None and freshness.state != expected_freshness_state:
        return False
    if freshness.stale_evidence_count > 0 and trust_label not in {TrustLabel.STALE, TrustLabel.CONFLICTED}:
        return False
    if freshness.state == FreshnessState.STALE and trust_label not in {TrustLabel.STALE, TrustLabel.CONFLICTED}:
        return False
    return not (freshness.state in {FreshnessState.UNKNOWN, FreshnessState.MIXED} and trust_label == TrustLabel.OFFICIAL)


def citation_validity_rate(response: _HasEvidence) -> float:
    """Shared citation-validity rate for response objects that expose evidence citations."""
    if not response.evidence:
        return 1.0 if response.trust_label == TrustLabel.UNSUPPORTED else 0.0
    valid = sum(1 for citation in response.evidence if citation.is_valid and not citation.validity_errors)
    return round(valid / len(response.evidence), 4)

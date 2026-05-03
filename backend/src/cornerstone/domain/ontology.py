from __future__ import annotations

from cornerstone.schemas import ConceptStatus, RelationStatus, TrustState

ONTOLOGY_GRAPH_MAX_DEPTH = 1
DEFAULT_ONTOLOGY_FOCUS_CONCEPT = "settlement"

CITABLE_TRUST_STATES = frozenset(
    {
        TrustState.UNREVIEWED,
        TrustState.REVIEWED,
        TrustState.CONFLICTED,
    }
)
TERMINAL_CONCEPT_STATUSES = frozenset({ConceptStatus.DEPRECATED, ConceptStatus.SUPERSEDED})
TERMINAL_RELATION_STATUSES = frozenset({RelationStatus.REJECTED})

SSOT_TRUST_BOUNDARY = (
    "Raw source data is source material, not the Single Source of Truth.",
    "Ontology extraction output is candidate-only.",
    "Only reviewed official Concepts and ConceptRelations form the ontology Single Source of Truth.",
)


def ensure_supported_graph_depth(depth: int, *, context: str = "ontology graph") -> None:
    """Enforce the release-wide ontology graph depth contract."""
    if depth < 0:
        raise ValueError(f"{context} depth must be 0 or {ONTOLOGY_GRAPH_MAX_DEPTH}.")
    if depth > ONTOLOGY_GRAPH_MAX_DEPTH:
        raise ValueError(f"{context} supports maximum depth {ONTOLOGY_GRAPH_MAX_DEPTH}.")

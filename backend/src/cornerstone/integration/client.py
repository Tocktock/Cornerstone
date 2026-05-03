from __future__ import annotations

from typing import Any

from cornerstone.schemas import (
    IntegrationOntologyResponse,
    IntegrationPackageManifest,
    OntologyGraphMode,
)
from cornerstone.services.ontology_graph import OntologyGraphService
from cornerstone.services.ontology_ssot_readiness import OntologySsotReadinessService


def build_integration_manifest() -> IntegrationPackageManifest:
    return IntegrationPackageManifest(
        job_to_be_done=(
            "Let another product or agent consume Cornerstone's official ontology graph, "
            "citations, and SSOT readiness without bypassing review gates."
        ),
        stable_endpoints=[
            "GET /v1/ontology/graph?concept={concept}&depth=1&mode=official",
            "GET /v1/ontology/ssot/readiness?focusConcept={concept}&depth=1&mode=official&includeGraph=true",
            "GET /v1/integration/package/manifest",
            "GET /v1/integration/ontology/{concept}",
        ],
        trust_boundary=(
            "Official Concepts and ConceptRelations are consumable; pending candidates are visible only as "
            "candidate summaries and never become truth without explicit review gates."
        ),
        non_chosen_reason=(
            "Frontend MVP is deferred because the immediate adoption risk is integration by existing tools and agents."
        ),
        quickstart=[
            "Start the backend API.",
            "Call GET /v1/integration/package/manifest to read the contract.",
            "Call GET /v1/integration/ontology/{concept} for graph, citations, trust state, and readiness.",
            "Treat reviewGateBypassAllowed=false as a hard boundary.",
        ],
    )


def build_integration_ontology_response(
    *,
    store: Any,
    concept: str,
    production_mode: bool,
) -> IntegrationOntologyResponse:
    graph = OntologyGraphService(store, production_mode=production_mode).graph(
        concept,
        depth=1,
        mode=OntologyGraphMode.OFFICIAL,
    )
    readiness = OntologySsotReadinessService(store, production_mode=production_mode).readiness(
        focus_concept=concept,
        depth=1,
        mode=OntologyGraphMode.OFFICIAL,
        include_graph=True,
    )
    unsupported_state = None
    if graph.focus_concept is None:
        unsupported_state = "unknown_concept"
    elif not graph.official_graph_available:
        unsupported_state = "not_ready_for_official_consumption"
    return IntegrationOntologyResponse(
        concept=concept,
        official_graph=graph,
        ssot_readiness=readiness,
        candidate_vs_official_boundary=(
            "The officialGraph field contains reviewed graph objects only. Candidate counts are summarized "
            "separately and cannot be consumed as official truth."
        ),
        evidence_citations=graph.evidence,
        trust_state=graph.trust_label,
        unsupported_state=unsupported_state,
        review_gate_bypass_allowed=False,
    )

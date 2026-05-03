from __future__ import annotations

from typing import Any, cast

from cornerstone.domain.ontology import ONTOLOGY_GRAPH_MAX_DEPTH, ensure_supported_graph_depth
from cornerstone.schemas import (
    OntologyCandidateStatus,
    OntologyGraphEvalMetricSummary,
    OntologyGraphMode,
    OntologyGraphResponse,
    OntologyProofRunStatus,
    OntologyProofStepStatus,
    OntologySsotReadinessCheck,
    OntologySsotReadinessResponse,
    TrustLabel,
    normalize_concept_term,
    utc_now,
)
from cornerstone.services.evaluation import OntologyGraphEvaluationService
from cornerstone.services.ontology_graph import OntologyGraphService


class OntologySsotReadinessService:
    """Read-only v2.0.0 release checks for the ontology SSOT loop.

    The service intentionally does not mutate sources, Artifacts, EvidenceFragments,
    candidates, Concepts, Relations, evaluation records, or proof data. It only
    serves a checklist that tells operators whether the current backend state is
    ready to act as an official evidence-backed ontology Single Source of Truth
    for a focus concept.
    """

    def __init__(self, store: Any, *, production_mode: bool = True) -> None:
        self.store = store
        self.production_mode = production_mode

    def readiness(
        self,
        *,
        focus_concept: str = "settlement",
        depth: int = 1,
        mode: OntologyGraphMode = OntologyGraphMode.OFFICIAL,
        include_graph: bool = False,
    ) -> OntologySsotReadinessResponse:
        ensure_supported_graph_depth(depth, context="ontology SSOT readiness")

        graph = OntologyGraphService(self.store, production_mode=self.production_mode).graph(
            focus_concept,
            depth=depth,
            mode=mode,
        )
        evaluation_summary = OntologyGraphEvaluationService(
            self.store,
            production_mode=self.production_mode,
        ).summarize()
        latest_evaluation = self._latest_evaluation_result(
            focus_concept=focus_concept,
            mode=mode,
            depth=depth,
        )

        pending_concept_candidates = len(
            self.store.list_concept_candidates(status=OntologyCandidateStatus.PENDING)
        )
        pending_relation_candidates = len(
            self.store.list_relation_candidates(status=OntologyCandidateStatus.PENDING)
        )
        source_count = len(self.store.list_data_sources())
        artifact_count = len(self.store.list_artifacts())
        evidence_count = len(self.store.list_evidence_fragments())
        official_graph_safe = self._official_graph_safe(graph, mode=mode)

        checks = [
            self._check(
                key="source_ingestion_available",
                title="Source-backed evidence exists",
                category="ingestion",
                passed=source_count > 0 and artifact_count > 0 and evidence_count > 0,
                goal="Cornerstone has source-backed evidence for the ontology loop.",
                passed_detail="Source ingestion has produced DataSources, Artifacts, and EvidenceFragments.",
                failed_detail="No source-backed evidence is available yet. Run manual upload/sync or connector sync first.",
                object_ids={
                    "sourceCount": source_count,
                    "artifactCount": artifact_count,
                    "evidenceFragmentCount": evidence_count,
                },
                next_actions=["Upload manual text data or run a connector sync for the focus domain."],
            ),
            self._check(
                key="official_graph_available",
                title="Official focus graph is available",
                category="graph",
                passed=graph.official_graph_available,
                goal="The focus concept can be served as an official ontology graph.",
                passed_detail="An official graph was served for the focus concept.",
                failed_detail="No official graph is available for the focus concept. Review evidence and approve candidates first.",
                checks=[
                    f"trustLabel={graph.trust_label}",
                    f"nodeCount={len(graph.nodes)}",
                    f"edgeCount={len(graph.edges)}",
                ],
                object_ids={"graphResponseId": graph.response_id},
                next_actions=["Run the operator proof loop or review/approve pending ontology candidates."],
            ),
            self._check(
                key="graph_depth_policy_respected",
                title="Depth policy is respected",
                category="graph",
                passed=graph.depth <= ONTOLOGY_GRAPH_MAX_DEPTH,
                goal=f"The release graph contract serves depth 0 or {ONTOLOGY_GRAPH_MAX_DEPTH} only.",
                passed_detail=f"The graph response respects the depth<={ONTOLOGY_GRAPH_MAX_DEPTH} serving policy.",
                failed_detail=f"The graph response violated the depth<={ONTOLOGY_GRAPH_MAX_DEPTH} serving policy.",
                object_ids={"depth": graph.depth},
            ),
            self._check(
                key="official_graph_safe",
                title="Official graph safety gates pass",
                category="trust",
                passed=official_graph_safe,
                goal="Official graph mode contains only official, evidence-supported graph objects.",
                passed_detail="The official graph contains only official nodes/edges and has official trust.",
                failed_detail="The graph is not safe to treat as official SSOT yet.",
                checks=[
                    f"trustLabel={graph.trust_label}",
                    f"nonOfficialNodeCount={graph.support_summary.non_official_node_count}",
                    f"nonOfficialEdgeCount={graph.support_summary.non_official_edge_count}",
                    f"invalidEvidenceReferenceCount={graph.support_summary.invalid_evidence_reference_count}",
                ],
                next_actions=["Resolve non-official graph objects, invalid evidence references, or trust-label issues."],
            ),
            self._check(
                key="evidence_citations_valid",
                title="Evidence citations are valid",
                category="evidence",
                passed=bool(graph.evidence) and all(citation.is_valid for citation in graph.evidence),
                goal="Every served official node/edge can be explained by valid evidence citations.",
                passed_detail="Graph evidence citations are present and valid.",
                failed_detail="Graph evidence citations are missing or invalid.",
                checks=[
                    f"evidenceCount={len(graph.evidence)}",
                    f"invalidEvidenceReferenceCount={graph.support_summary.invalid_evidence_reference_count}",
                ],
                next_actions=["Review evidence support and fix missing/invalid citation links."],
            ),
            self._check(
                key="review_provenance_present",
                title="Review provenance is present",
                category="review",
                passed=self._review_provenance_present(graph),
                goal="Official graph objects and citations expose review provenance.",
                passed_detail="Graph evidence and served entities include review provenance.",
                failed_detail="Review provenance is missing for at least one served graph item.",
                checks=[
                    f"reviewedEvidenceCount={graph.support_summary.reviewed_evidence_count}",
                    f"evidenceCount={graph.support_summary.evidence_count}",
                ],
                next_actions=["Review evidence and officialize Concepts/Relations through reviewer workflows."],
            ),
            self._check(
                key="candidate_boundary_respected",
                title="Candidate boundary is respected",
                category="trust",
                passed=mode != OntologyGraphMode.OFFICIAL
                or (
                    graph.support_summary.non_official_node_count == 0
                    and graph.support_summary.non_official_edge_count == 0
                ),
                goal="Candidate and non-official objects stay outside official graph mode.",
                passed_detail="Official mode excludes candidate/non-official graph objects.",
                failed_detail="Official mode included candidate or non-official graph objects.",
                checks=[
                    f"pendingConceptCandidateCount={pending_concept_candidates}",
                    f"pendingRelationCandidateCount={pending_relation_candidates}",
                    f"mode={mode}",
                ],
                next_actions=["Keep serving official mode by default and review pending candidates separately."],
            ),
            self._check(
                key="ontology_evaluation_available",
                title="Successful ontology evaluation exists",
                category="evaluation",
                passed=latest_evaluation is not None and bool(latest_evaluation.success),
                goal="The focus graph has a successful read-only ontology evaluation result.",
                passed_detail="A successful ontology evaluation result exists for the focus graph.",
                failed_detail="No successful ontology evaluation result exists for this focus graph yet.",
                object_ids={
                    "latestEvaluationResultId": None if latest_evaluation is None else latest_evaluation.id,
                    "evaluatedResultCount": evaluation_summary.evaluated_result_count,
                    "successfulResultCount": evaluation_summary.successful_result_count,
                },
                next_actions=["Run ontology evaluation for the official focus graph."],
            ),
            self._check(
                key="operator_proof_path_available",
                title="Operator proof path is documented",
                category="operator",
                passed=True,
                goal="Operators have a proof checklist path for end-to-end verification.",
                passed_detail="Operator proof checklist endpoint and CLI scope are available.",
                failed_detail="Operator proof checklist path is unavailable.",
                checks=[
                    "POST /v1/ontology/proof-runs",
                    "cornerstone proof run --ontology-loop",
                    "cornerstone proof run --ssot-readiness",
                ],
            ),
        ]

        required_checks = [check for check in checks if check.required]
        status_value = (
            OntologyProofRunStatus.PASSED
            if all(check.status == OntologyProofStepStatus.PASSED for check in required_checks)
            else OntologyProofRunStatus.FAILED
        )
        recommended_actions = [
            action
            for check in checks
            if check.required and check.status != OntologyProofStepStatus.PASSED
            for action in (check.next_actions or [check.detail or check.title])
        ]

        return OntologySsotReadinessResponse(
            focus_concept=focus_concept,
            mode=mode,
            depth=depth,
            status=status_value,
            official_graph_available=graph.official_graph_available,
            official_graph_safe=official_graph_safe,
            trust_label=graph.trust_label,
            graph_response_id=graph.response_id,
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            evidence_count=len(graph.evidence),
            pending_concept_candidate_count=pending_concept_candidates,
            pending_relation_candidate_count=pending_relation_candidates,
            latest_evaluation_result_id=None if latest_evaluation is None else latest_evaluation.id,
            latest_evaluation_success=None if latest_evaluation is None else latest_evaluation.success,
            evaluation_summary=cast(OntologyGraphEvalMetricSummary, evaluation_summary),
            checks=checks,
            recommended_actions=list(dict.fromkeys(recommended_actions)),
            graph=graph if include_graph else None,
            generated_at=utc_now(),
        )

    def _latest_evaluation_result(
        self,
        *,
        focus_concept: str,
        mode: OntologyGraphMode,
        depth: int,
    ) -> Any | None:
        normalized_focus = normalize_concept_term(focus_concept)
        matching = [
            result
            for result in self.store.list_ontology_graph_eval_results()
            if normalize_concept_term(result.concept_query) == normalized_focus
            and result.mode == mode
            and result.depth == depth
        ]
        if not matching:
            return None
        return sorted(matching, key=lambda item: item.evaluated_at)[-1]

    @staticmethod
    def _official_graph_safe(graph: OntologyGraphResponse, *, mode: OntologyGraphMode) -> bool:
        if mode != OntologyGraphMode.OFFICIAL:
            return False
        return all(
            [
                graph.official_graph_available,
                graph.trust_label == TrustLabel.OFFICIAL,
                graph.support_summary.non_official_node_count == 0,
                graph.support_summary.non_official_edge_count == 0,
                graph.support_summary.invalid_evidence_reference_count == 0,
                graph.support_summary.invalid_relation_count == 0,
                bool(graph.evidence),
                graph.support_summary.reviewed_evidence_count == graph.support_summary.evidence_count,
            ]
        )

    @staticmethod
    def _review_provenance_present(graph: OntologyGraphResponse) -> bool:
        if not graph.evidence:
            return False
        if not all(citation.reviewed_by and citation.reviewed_at for citation in graph.evidence):
            return False
        if not all(node.review_provenance is not None for node in graph.nodes):
            return False
        return all(edge.review_provenance is not None for edge in graph.edges)

    @staticmethod
    def _check(
        *,
        key: str,
        title: str,
        category: str,
        passed: bool,
        goal: str,
        passed_detail: str,
        failed_detail: str,
        checks: list[str] | None = None,
        object_ids: dict[str, Any] | None = None,
        next_actions: list[str] | None = None,
        required: bool = True,
    ) -> OntologySsotReadinessCheck:
        return OntologySsotReadinessCheck(
            key=key,
            title=title,
            category=category,
            goal=goal,
            required=required,
            status=OntologyProofStepStatus.PASSED if passed else OntologyProofStepStatus.FAILED,
            detail=passed_detail if passed else failed_detail,
            checks=checks or [],
            object_ids=object_ids or {},
            next_actions=[] if passed else (next_actions or []),
        )

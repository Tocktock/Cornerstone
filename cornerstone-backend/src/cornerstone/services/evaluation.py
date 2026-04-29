from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from cornerstone.schemas import (
    CreateGroundedContextEvalTaskRequest,
    FreshnessState,
    GroundedContextEvalMetricSummary,
    GroundedContextEvalResult,
    GroundedContextEvalRunResponse,
    GroundedContextEvalTask,
    GroundedContextResponse,
    RunGroundedContextEvalRequest,
    RunGroundedContextEvalTaskRequest,
    TrustLabel,
    utc_now,
)
from cornerstone.services.grounded_context import GroundedContextService
from cornerstone.store import NotFoundError


class GroundedContextEvaluationService:
    """Evaluate grounded context responses against PRD quality gates.

    A grounded context task succeeds only when answer content, evidence validity,
    provenance, trust-label behavior, freshness policy, and unsupported-official-claim
    prevention all pass.
    """

    def __init__(self, store: Any, *, production_mode: bool = True) -> None:
        self.store = store
        self.production_mode = production_mode

    def create_task(self, request: CreateGroundedContextEvalTaskRequest) -> GroundedContextEvalTask:
        now = utc_now()
        task = GroundedContextEvalTask(
            name=request.name,
            query=request.query,
            expected_answer_contains=request.expected_answer_contains,
            expected_trust_label=request.expected_trust_label,
            expected_freshness_state=request.expected_freshness_state,
            required_evidence_fragment_ids=request.required_evidence_fragment_ids,
            required_concept_ids=request.required_concept_ids,
            required_decision_record_ids=request.required_decision_record_ids,
            require_official_answer=request.require_official_answer,
            require_evidence=request.require_evidence,
            min_evidence_count=request.min_evidence_count,
            expected_clarification_reduced=request.expected_clarification_reduced,
            tags=request.tags,
            created_by=request.created_by,
            created_at=now,
            updated_at=now,
            metadata=request.metadata,
        )
        return cast(GroundedContextEvalTask, self.store.add_grounded_context_eval_task(task))

    def run_task(
        self,
        task_id: str,
        request: RunGroundedContextEvalTaskRequest | None = None,
    ) -> GroundedContextEvalResult:
        task = cast(GroundedContextEvalTask, self.store.get_grounded_context_eval_task(task_id))
        response = GroundedContextService(self.store, production_mode=self.production_mode).query(task.query)
        result = self.evaluate_response(task, response, request=request)
        return cast(GroundedContextEvalResult, self.store.add_grounded_context_eval_result(result))

    def run_tasks(self, request: RunGroundedContextEvalRequest | None = None) -> GroundedContextEvalRunResponse:
        request = request or RunGroundedContextEvalRequest()
        task_ids = set(request.task_ids)
        tasks = cast(list[GroundedContextEvalTask], self.store.list_grounded_context_eval_tasks())
        if task_ids:
            tasks = [task for task in tasks if task.id in task_ids]
            missing_ids = sorted(task_ids - {task.id for task in tasks})
            if missing_ids:
                raise NotFoundError(f"GroundedContextEvalTask not found: {missing_ids[0]}")
        results = [
            self.run_task(task.id, RunGroundedContextEvalTaskRequest(evaluated_by=request.evaluated_by))
            for task in tasks
        ]
        success_count = sum(1 for result in results if result.success)
        return GroundedContextEvalRunResponse(
            results=results,
            total_count=len(results),
            success_count=success_count,
            grounded_context_task_success_rate=_rate(success_count, len(results)),
        )

    def evaluate_response(
        self,
        task: GroundedContextEvalTask,
        response: GroundedContextResponse,
        *,
        request: RunGroundedContextEvalTaskRequest | None = None,
    ) -> GroundedContextEvalResult:
        failure_reasons: list[str] = []
        citation_validity_rate = _citation_validity_rate(response)

        answer_correct = _answer_matches(response.answer, task.expected_answer_contains)
        if not answer_correct:
            failure_reasons.append("answer_missing_expected_text")

        evidence_valid = _evidence_valid(task, response)
        if not evidence_valid:
            failure_reasons.append("evidence_invalid_or_missing")

        provenance_present = _provenance_present(task, response)
        if not provenance_present:
            failure_reasons.append("provenance_missing")

        trust_label_correct = _trust_label_correct(task, response)
        if not trust_label_correct:
            failure_reasons.append("trust_label_mismatch")

        freshness_policy_respected = _freshness_policy_respected(task, response)
        if not freshness_policy_respected:
            failure_reasons.append("freshness_policy_violation")

        unsupported_official_claim = _unsupported_official_claim(task, response)
        if unsupported_official_claim:
            failure_reasons.append("unsupported_official_claim")

        clarification_reduced = (
            request.clarification_reduced
            if request is not None and request.clarification_reduced is not None
            else task.expected_clarification_reduced
        )
        if task.expected_clarification_reduced is not None and clarification_reduced != task.expected_clarification_reduced:
            failure_reasons.append("clarification_reduction_mismatch")

        success = all(
            [
                answer_correct,
                evidence_valid,
                provenance_present,
                trust_label_correct,
                freshness_policy_respected,
                not unsupported_official_claim,
            ]
        )
        return GroundedContextEvalResult(
            task_id=task.id,
            response_id=response.response_id,
            query=task.query,
            answer=response.answer,
            trust_label=response.trust_label,
            response=response,
            answer_correct=answer_correct,
            evidence_valid=evidence_valid,
            provenance_present=provenance_present,
            trust_label_correct=trust_label_correct,
            freshness_policy_respected=freshness_policy_respected,
            unsupported_official_claim=unsupported_official_claim,
            citation_validity_rate=citation_validity_rate,
            clarification_reduced=clarification_reduced,
            success=success,
            failure_reasons=failure_reasons,
            evaluated_by=(request.evaluated_by if request is not None else "system"),
        )

    def summarize(self) -> GroundedContextEvalMetricSummary:
        tasks = cast(list[GroundedContextEvalTask], self.store.list_grounded_context_eval_tasks())
        results = cast(list[GroundedContextEvalResult], self.store.list_grounded_context_eval_results())
        return summarize_eval_results(total_tasks=len(tasks), results=results)


def summarize_eval_results(*, total_tasks: int, results: list[GroundedContextEvalResult]) -> GroundedContextEvalMetricSummary:
    total_results = len(results)
    successful = sum(1 for result in results if result.success)
    unsupported_expected_results = [result for result in results if result.response.trust_label == TrustLabel.UNSUPPORTED]
    unsupported_correct = [
        result for result in unsupported_expected_results if not result.unsupported_official_claim and result.trust_label == TrustLabel.UNSUPPORTED
    ]
    return GroundedContextEvalMetricSummary(
        total_task_count=total_tasks,
        evaluated_result_count=total_results,
        successful_result_count=successful,
        grounded_context_task_success_rate=_rate(successful, total_results),
        provenance_coverage_rate=_boolean_rate(result.provenance_present for result in results),
        citation_validity_rate=_average(result.citation_validity_rate for result in results),
        freshness_compliance_rate=_boolean_rate(result.freshness_policy_respected for result in results),
        trust_label_correctness_rate=_boolean_rate(result.trust_label_correct for result in results),
        unsupported_answer_correctness_rate=_rate(len(unsupported_correct), len(unsupported_expected_results)),
    )


def _answer_matches(answer: str, expected_answer_contains: list[str]) -> bool:
    if not expected_answer_contains:
        return bool(answer.strip())
    normalized = answer.lower()
    return all(fragment.lower() in normalized for fragment in expected_answer_contains)


def _evidence_valid(task: GroundedContextEvalTask, response: GroundedContextResponse) -> bool:
    citations = response.evidence
    if task.require_evidence and len(citations) < task.min_evidence_count:
        return False
    if len(citations) < task.min_evidence_count:
        return False
    required_evidence = set(task.required_evidence_fragment_ids)
    actual_evidence = {citation.evidence_fragment_id for citation in citations}
    if required_evidence and not required_evidence.issubset(actual_evidence):
        return False
    required_concepts = set(task.required_concept_ids)
    actual_concepts = {concept.id for concept in response.concepts}
    if required_concepts and not required_concepts.issubset(actual_concepts):
        return False
    required_decisions = set(task.required_decision_record_ids)
    actual_decisions = {decision.id for decision in response.decisions}
    if required_decisions and not required_decisions.issubset(actual_decisions):
        return False
    return all(citation.is_valid and not citation.validity_errors for citation in citations)


def _provenance_present(task: GroundedContextEvalTask, response: GroundedContextResponse) -> bool:
    if not response.evidence:
        return not task.require_evidence and task.min_evidence_count == 0
    for citation in response.evidence:
        if not citation.evidence_fragment_id:
            return False
        if not citation.artifact_id:
            return False
        if not citation.artifact_title:
            return False
        if citation.captured_at is None:
            return False
    return True


def _trust_label_correct(task: GroundedContextEvalTask, response: GroundedContextResponse) -> bool:
    if task.expected_trust_label is not None and response.trust_label != task.expected_trust_label:
        return False
    return not (task.require_official_answer and response.trust_label != TrustLabel.OFFICIAL)


def _freshness_policy_respected(task: GroundedContextEvalTask, response: GroundedContextResponse) -> bool:
    if task.expected_freshness_state is not None and response.freshness.state != task.expected_freshness_state:
        return False
    if response.freshness.stale_evidence_count > 0 and response.trust_label not in {TrustLabel.STALE, TrustLabel.CONFLICTED}:
        return False
    if response.freshness.state == FreshnessState.STALE and response.trust_label not in {TrustLabel.STALE, TrustLabel.CONFLICTED}:
        return False
    return not (response.freshness.state in {FreshnessState.UNKNOWN, FreshnessState.MIXED} and response.trust_label == TrustLabel.OFFICIAL)


def _unsupported_official_claim(task: GroundedContextEvalTask, response: GroundedContextResponse) -> bool:
    if response.trust_label == TrustLabel.OFFICIAL and not response.evidence:
        return True
    if response.trust_label == TrustLabel.OFFICIAL and not response.official_answer_available:
        return True
    if response.trust_label == TrustLabel.OFFICIAL and any(not citation.is_valid or citation.validity_errors for citation in response.evidence):
        return True
    return task.expected_trust_label == TrustLabel.UNSUPPORTED and response.trust_label == TrustLabel.OFFICIAL


def _citation_validity_rate(response: GroundedContextResponse) -> float:
    if not response.evidence:
        return 1.0 if response.trust_label == TrustLabel.UNSUPPORTED else 0.0
    valid = sum(1 for citation in response.evidence if citation.is_valid and not citation.validity_errors)
    return _rate(valid, len(response.evidence))


def _boolean_rate(values: Iterable[bool]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return _rate(sum(1 for item in items if item), len(items))


def _average(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(items) / len(items), 4)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)

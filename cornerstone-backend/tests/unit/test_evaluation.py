from __future__ import annotations

from cornerstone.schemas import (
    FreshnessState,
    FreshnessSummary,
    GroundedContextEvalResult,
    GroundedContextEvalTask,
    GroundedContextResponse,
    TrustLabel,
)
from cornerstone.services.evaluation import GroundedContextEvaluationService, summarize_eval_results
from cornerstone.store import InMemoryStore


def _response(*, trust_label: TrustLabel, answer: str = "There is no official or evidence-supported context.") -> GroundedContextResponse:
    return GroundedContextResponse(
        query="What is MissingThing?",
        answer=answer,
        trust_label=trust_label,
        evidence=[],
        freshness=FreshnessSummary(state=FreshnessState.UNKNOWN),
        limitations=["No matching Concept or EvidenceFragment was found."],
        official_answer_available=trust_label == TrustLabel.OFFICIAL,
    )


def test_eval_task_accepts_unsupported_without_evidence_requirement() -> None:
    task = GroundedContextEvalTask(
        name="Unsupported task",
        query="What is MissingThing?",
        expected_trust_label=TrustLabel.UNSUPPORTED,
        require_evidence=False,
        min_evidence_count=0,
        created_by="qa@example.com",
    )

    result = GroundedContextEvaluationService(InMemoryStore()).evaluate_response(task, _response(trust_label=TrustLabel.UNSUPPORTED))

    assert result.success is True
    assert result.evidence_valid is True
    assert result.provenance_present is True
    assert result.trust_label_correct is True
    assert result.unsupported_official_claim is False


def test_eval_task_fails_when_expected_trust_label_is_wrong() -> None:
    task = GroundedContextEvalTask(
        name="Unsupported task",
        query="What is Cornerstone?",
        expected_trust_label=TrustLabel.UNSUPPORTED,
        require_evidence=False,
        min_evidence_count=0,
        created_by="qa@example.com",
    )

    result = GroundedContextEvaluationService(InMemoryStore()).evaluate_response(
        task,
        _response(trust_label=TrustLabel.OFFICIAL, answer="Cornerstone: shared context layer."),
    )

    assert result.success is False
    assert result.trust_label_correct is False
    assert result.unsupported_official_claim is True
    assert "trust_label_mismatch" in result.failure_reasons
    assert "unsupported_official_claim" in result.failure_reasons


def test_eval_task_requires_expected_answer_text() -> None:
    task = GroundedContextEvalTask(
        name="Answer check",
        query="What is Cornerstone?",
        expected_answer_contains=["shared organizational context layer"],
        expected_trust_label=TrustLabel.UNSUPPORTED,
        require_evidence=False,
        min_evidence_count=0,
        created_by="qa@example.com",
    )

    result = GroundedContextEvaluationService(InMemoryStore()).evaluate_response(task, _response(trust_label=TrustLabel.UNSUPPORTED))

    assert result.success is False
    assert result.answer_correct is False
    assert "answer_missing_expected_text" in result.failure_reasons


def test_eval_summary_computes_grounded_context_task_success_rate() -> None:
    task = GroundedContextEvalTask(
        name="Task",
        query="What is MissingThing?",
        expected_trust_label=TrustLabel.UNSUPPORTED,
        require_evidence=False,
        min_evidence_count=0,
        created_by="qa@example.com",
    )
    service = GroundedContextEvaluationService(InMemoryStore())
    success = service.evaluate_response(task, _response(trust_label=TrustLabel.UNSUPPORTED))
    failure = service.evaluate_response(task, _response(trust_label=TrustLabel.OFFICIAL, answer="Official claim"))

    summary = summarize_eval_results(total_tasks=1, results=[success, failure])

    assert summary.total_task_count == 1
    assert summary.evaluated_result_count == 2
    assert summary.successful_result_count == 1
    assert summary.grounded_context_task_success_rate == 0.5
    assert summary.trust_label_correctness_rate == 0.5


def test_store_persists_eval_tasks_and_results_in_memory() -> None:
    store = InMemoryStore()
    task = store.add_grounded_context_eval_task(
        GroundedContextEvalTask(
            name="Unsupported task",
            query="What is MissingThing?",
            expected_trust_label=TrustLabel.UNSUPPORTED,
            require_evidence=False,
            min_evidence_count=0,
            created_by="qa@example.com",
        )
    )
    result = GroundedContextEvalResult(
        task_id=task.id,
        response_id="00000000-0000-0000-0000-000000000001",
        query=task.query,
        answer="Unsupported",
        trust_label=TrustLabel.UNSUPPORTED,
        response=_response(trust_label=TrustLabel.UNSUPPORTED),
        answer_correct=True,
        evidence_valid=True,
        provenance_present=True,
        trust_label_correct=True,
        freshness_policy_respected=True,
        unsupported_official_claim=False,
        citation_validity_rate=1.0,
        success=True,
        evaluated_by="qa@example.com",
    )
    saved = store.add_grounded_context_eval_result(result)

    assert store.get_grounded_context_eval_task(task.id).name == "Unsupported task"
    assert store.get_grounded_context_eval_result(saved.id).success is True
    assert len(store.list_grounded_context_eval_results(task_id=task.id)) == 1



def test_eval_task_rejects_non_unsupported_without_evidence_requirement() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="must require evidence"):
        GroundedContextEvalTask(
            name="Metric gaming task",
            query="What is Cornerstone?",
            expected_trust_label=TrustLabel.OFFICIAL,
            require_evidence=False,
            min_evidence_count=0,
            created_by="qa@example.com",
        )


def test_eval_task_rejects_vague_success_contract_in_model() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="explicit success condition"):
        GroundedContextEvalTask(
            name="Vague task",
            query="What is Cornerstone?",
            created_by="qa@example.com",
        )

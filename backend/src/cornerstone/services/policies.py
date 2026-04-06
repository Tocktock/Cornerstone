from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from cornerstone.domain.enums import (
    BaseRole,
    CuratedLifecycleState,
    DecisionLifecycleState,
    FreshnessState,
    ReviewAction,
    SupportVisibility,
    VerificationState,
)
from cornerstone.domain.models import (
    Actor,
    Concept,
    ConceptRelation,
    DecisionRecord,
    ReviewScopeGrant,
    SupportItem,
    VerificationPolicy,
)


class PolicyError(ValueError):
    pass


def derive_relation_review_domain(
    subject_concept: Concept,
    object_concept: Concept,
    *,
    workspace_wide: bool = False,
) -> str:
    if workspace_wide:
        return "workspace"
    if subject_concept.owning_domain == object_concept.owning_domain:
        return subject_concept.owning_domain
    return "workspace"


def can_review(
    session: Session,
    actor: Actor,
    resource: Concept | ConceptRelation | DecisionRecord,
    action: ReviewAction,
) -> bool:
    if actor.context_space_id != resource.context_space_id:
        return False
    if actor.base_role in {BaseRole.OWNER, BaseRole.ADMIN}:
        return True
    grants = session.query(ReviewScopeGrant).filter(ReviewScopeGrant.actor_id == actor.id).all()
    for grant in grants:
        if grant.context_space_id != resource.context_space_id:
            continue
        if grant.allowed_review_actions and action.value not in grant.allowed_review_actions:
            continue
        domains = set(grant.review_domains)
        if "workspace" in domains:
            return True
        if resource.review_domain in domains:
            return True
    return False


def support_visibility_for_consumer(
    policy: VerificationPolicy | None,
    support_items: list[SupportItem],
    visible_support_items: list[SupportItem],
) -> SupportVisibility:
    if not support_items:
        return SupportVisibility.INSUFFICIENT_SUPPORT
    minimum_total = policy.minimum_support_items if policy else 1
    minimum_visible = policy.minimum_visible_support_items_for_source_backed if policy else 1
    if len(visible_support_items) >= minimum_visible:
        return SupportVisibility.SOURCE_BACKED
    if len(support_items) >= minimum_total and len(visible_support_items) < len(support_items):
        return SupportVisibility.RESTRICTED_SUPPORT
    return SupportVisibility.INSUFFICIENT_SUPPORT


def verification_state_for_freshness(
    freshness_state: FreshnessState,
    current_state: VerificationState | None = None,
) -> VerificationState:
    if freshness_state is FreshnessState.DRIFT_DETECTED:
        return VerificationState.DRIFT_DETECTED
    if freshness_state is FreshnessState.STALE:
        return VerificationState.REVIEW_REQUIRED
    if freshness_state is FreshnessState.MONITORING:
        return VerificationState.MONITORING
    return current_state or VerificationState.VERIFIED


def worst_freshness_state(support_items: list[SupportItem]) -> FreshnessState:
    states = {item.freshness_state for item in support_items}
    if not states:
        return FreshnessState.UNKNOWN
    if FreshnessState.DRIFT_DETECTED in states:
        return FreshnessState.DRIFT_DETECTED
    if FreshnessState.STALE in states:
        return FreshnessState.STALE
    if FreshnessState.MONITORING in states:
        return FreshnessState.MONITORING
    if states == {FreshnessState.CURRENT}:
        return FreshnessState.CURRENT
    return FreshnessState.UNKNOWN


def require_review_permission(
    session: Session,
    actor: Actor,
    resource: Concept | ConceptRelation | DecisionRecord,
    action: ReviewAction,
) -> None:
    if not can_review(session, actor, resource, action):
        raise PolicyError(
            "Actor "
            f"{actor.display_name} does not hold review scope "
            f"for domain '{resource.review_domain}'."
        )


def ensure_support_sufficiency(
    resource: Concept | ConceptRelation | DecisionRecord,
    policy: VerificationPolicy | None,
) -> None:
    if isinstance(resource, (Concept, ConceptRelation)):
        support_items = [link.support_item for link in resource.support_links]
    else:
        support_items = [link.support_item for link in resource.support_links]
    minimum_total = policy.minimum_support_items if policy else 1
    if len(support_items) < minimum_total:
        raise PolicyError("Support is insufficient for review approval.")


def transition_after_review(
    resource: Concept | ConceptRelation | DecisionRecord,
    action: ReviewAction,
    *,
    policy: VerificationPolicy | None,
) -> None:
    if action in {
        ReviewAction.APPROVE,
        ReviewAction.OFFICIALIZE,
        ReviewAction.RESOLVE_REVIEW_REQUIRED,
    }:
        ensure_support_sufficiency(resource, policy)
        if isinstance(resource, (Concept, ConceptRelation)):
            resource.lifecycle_state = CuratedLifecycleState.OFFICIAL
            resource.verification_state = verification_state_for_freshness(
                worst_freshness_state([link.support_item for link in resource.support_links]),
                VerificationState.VERIFIED,
            )
        else:
            resource.lifecycle_state = DecisionLifecycleState.ACCEPTED
            resource.verification_state = verification_state_for_freshness(
                worst_freshness_state([link.support_item for link in resource.support_links]),
                VerificationState.VERIFIED,
            )
        return

    if action is ReviewAction.REJECT:
        if isinstance(resource, DecisionRecord):
            resource.lifecycle_state = DecisionLifecycleState.REJECTED
        else:
            resource.lifecycle_state = CuratedLifecycleState.ARCHIVED
        resource.verification_state = VerificationState.REVIEW_REQUIRED
        return

    if action is ReviewAction.SUPERSEDE:
        if isinstance(resource, DecisionRecord):
            resource.lifecycle_state = DecisionLifecycleState.SUPERSEDED
            resource.verification_state = VerificationState.MONITORING
        else:
            resource.lifecycle_state = CuratedLifecycleState.DEPRECATED
            resource.verification_state = VerificationState.MONITORING
        return

    if action is ReviewAction.MARK_FOR_REVALIDATION:
        resource.verification_state = VerificationState.REVIEW_REQUIRED
        return

    raise PolicyError(f"Unsupported review action: {action.value}")


def member_can_see_restricted_outputs(policy: VerificationPolicy | None) -> bool:
    return bool(policy.allow_member_restricted_support_publication) if policy else True


def stamp_sync_freshness(
    source_updated_at: datetime | None,
    now: datetime,
    *,
    stale_after_hours: int,
    drift_after_hours: int,
) -> FreshnessState:
    if source_updated_at is None:
        return FreshnessState.UNKNOWN
    age_hours = max((now - source_updated_at).total_seconds(), 0) / 3600
    if age_hours >= drift_after_hours:
        return FreshnessState.DRIFT_DETECTED
    if age_hours >= stale_after_hours:
        return FreshnessState.STALE
    if age_hours >= stale_after_hours / 2:
        return FreshnessState.MONITORING
    return FreshnessState.CURRENT

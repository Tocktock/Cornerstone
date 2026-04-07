from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from cornerstone.clock import utcnow
from cornerstone.config import Settings
from cornerstone.domain.enums import (
    ActorKind,
    BaseRole,
    Capability,
    ConsumerScope,
    ContextSpaceKind,
    DecisionLifecycleState,
    FreshnessState,
    OriginDisclosureLevel,
    ReviewAction,
    SharedSelectionKind,
    SourceConnectionState,
    SyncMode,
    VerificationState,
    VisibilityClass,
)
from cornerstone.domain.models import (
    Actor,
    Base,
    ConnectorScopeGrant,
    ContextSpace,
    ReviewScopeGrant,
    SourceConnection,
    SupportItem,
    VerificationPolicy,
)
from cornerstone.domain.schemas import (
    DraftConceptCreate,
    DraftDecisionCreate,
    DraftRelationCreate,
    PromotionRequest,
)
from cornerstone.services.creation import (
    create_concept,
    create_decision,
    create_promoted_support,
    create_relation,
)
from cornerstone.services.normalization import stable_id
from cornerstone.services.review import review_concept, review_decision, review_relation
from cornerstone.services.sync import run_sync


def initialize_database(engine, *, reset: bool = False) -> None:
    if reset:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def seed_demo(session: Session, settings: Settings) -> None:
    workspace = session.scalar(
        select(ContextSpace).where(ContextSpace.slug == settings.default_workspace_slug).limit(1)
    )
    if workspace is not None:
        return

    now = utcnow(settings)
    workspace = ContextSpace(
        id=stable_id("ctx", settings.default_workspace_slug),
        kind=ContextSpaceKind.WORKSPACE,
        name=settings.default_workspace_name,
        slug=settings.default_workspace_slug,
        membership_boundary="workspace:cornerstone",
        default_visibility_class=VisibilityClass.MEMBER_VISIBLE,
        visibility_defaults={"member": VisibilityClass.MEMBER_VISIBLE.value},
        is_default=True,
    )
    personal_context = ContextSpace(
        id=stable_id("ctx", settings.default_personal_slug),
        kind=ContextSpaceKind.PERSONAL,
        name=settings.default_personal_name,
        slug=settings.default_personal_slug,
        membership_boundary="principal:tars",
        default_visibility_class=VisibilityClass.EVIDENCE_ONLY,
        visibility_defaults={"personal_default": VisibilityClass.EVIDENCE_ONLY.value},
        is_default=False,
    )
    session.add_all([workspace, personal_context])
    session.flush()

    policy = VerificationPolicy(
        id=stable_id("policy", workspace.id),
        context_space_id=workspace.id,
        label="Default P0 policy",
        version="p0",
        minimum_support_items=1,
        minimum_durable_support_items=1,
        minimum_visible_support_items_for_source_backed=1,
        allow_restricted_support_for_officialization=True,
        allow_member_restricted_support_publication=True,
        freshness_target_hours=settings.freshness_target_hours,
        continuous_revalidation_enabled=True,
        allow_accepted_decision_lineage_as_support=True,
    )
    session.add(policy)

    actors = _seed_actors(session, workspace)
    session.flush()

    member_actor = actors["member"]
    reviewer_actor = actors["reviewer"]
    admin_actor = actors["admin"]

    member_visible_connection = _seed_connection(
        session,
        workspace,
        source_label="Workspace handbook",
        boundary_locator=settings.workspace_source_root,
        template_key="member-visible",
        visibility_class=VisibilityClass.MEMBER_VISIBLE,
    )
    evidence_only_connection = _seed_connection(
        session,
        workspace,
        source_label="Review-only notebook",
        boundary_locator=str(
            (Path(settings.fixture_root) / "minimal" / "workspace" / "evidence-only").resolve()
        ),
        template_key="evidence-only",
        visibility_class=VisibilityClass.EVIDENCE_ONLY,
    )
    personal_connection = _seed_connection(
        session,
        personal_context,
        source_label="Personal analyst notebook",
        boundary_locator=settings.personal_source_root,
        template_key="personal-private",
        visibility_class=VisibilityClass.EVIDENCE_ONLY,
    )
    stale_connection = _seed_connection(
        session,
        workspace,
        source_label="Scheduled exports",
        boundary_locator=str(
            (Path(settings.fixture_root) / "minimal" / "workspace" / "stale").resolve()
        ),
        template_key="stale-snapshot",
        visibility_class=VisibilityClass.MEMBER_VISIBLE,
    )
    degraded_connection = _seed_connection(
        session,
        workspace,
        source_label="Broken shared drive mirror",
        boundary_locator=str(
            (Path(settings.fixture_root) / "missing" / "does-not-exist").resolve()
        ),
        template_key="degraded",
        visibility_class=VisibilityClass.EVIDENCE_ONLY,
    )
    paused_connection = _seed_connection(
        session,
        workspace,
        source_label="Paused import",
        boundary_locator=str(
            (Path(settings.fixture_root) / "minimal" / "workspace" / "paused").resolve()
        ),
        template_key="paused",
        visibility_class=VisibilityClass.MEMBER_VISIBLE,
    )
    removed_connection = _seed_connection(
        session,
        workspace,
        source_label="Removed legacy source",
        boundary_locator=str(
            (Path(settings.fixture_root) / "minimal" / "workspace" / "removed").resolve()
        ),
        template_key="removed",
        visibility_class=VisibilityClass.MEMBER_VISIBLE,
    )
    session.flush()

    run_sync(session, member_visible_connection, settings)
    run_sync(session, evidence_only_connection, settings)
    run_sync(session, personal_connection, settings)

    stale_connection.source_connection_state = SourceConnectionState.ACTIVE
    stale_connection.freshness_state = FreshnessState.STALE
    stale_connection.last_attempted_sync_at = now - timedelta(hours=55)
    stale_connection.last_successful_sync_at = now - timedelta(hours=50)
    stale_connection.last_error = None

    run_sync(session, degraded_connection, settings)

    paused_connection.source_connection_state = SourceConnectionState.PAUSED
    paused_connection.freshness_state = FreshnessState.MONITORING
    paused_connection.last_attempted_sync_at = now - timedelta(hours=6)
    paused_connection.last_successful_sync_at = now - timedelta(hours=6)
    paused_connection.last_error = "Paused by operator for fixture isolation."

    removed_connection.source_connection_state = SourceConnectionState.REMOVED
    removed_connection.freshness_state = FreshnessState.UNKNOWN
    removed_connection.last_attempted_sync_at = now - timedelta(days=14)
    removed_connection.last_successful_sync_at = now - timedelta(days=14)
    removed_connection.last_error = "Removed from workspace scope."
    removed_connection.removed_at = now - timedelta(days=13)

    session.flush()

    support = _support_lookup(session)
    promoted_support = create_promoted_support(
        session,
        PromotionRequest(
            personal_support_item_id=support["executive-call.md"],
            workspace_context_id=workspace.id,
            shared_selection_kind=SharedSelectionKind.SUMMARY_CLAIM,
            shared_payload=(
                "A redacted executive insight recommends "
                "risk-based escalation for VIP incidents."
            ),
            visibility_class=VisibilityClass.MEMBER_VISIBLE,
            origin_disclosure_level=OriginDisclosureLevel.REDACTED_ORIGIN,
        ),
        promoter=member_actor,
        settings=settings,
    )

    ops_playbook = create_concept(
        session,
        DraftConceptCreate(
            context_space_id=workspace.id,
            canonical_name="Ops Playbook",
            definition="The shared operating handbook for incident response.",
            owning_domain="sales_ops",
            support_item_ids=[support["ops-playbook.md"]],
        ),
    )
    partner_sla = create_concept(
        session,
        DraftConceptCreate(
            context_space_id=workspace.id,
            canonical_name="Partner SLA",
            definition="The published partner commitment used during issue triage.",
            owning_domain="sales_ops",
            support_item_ids=[support["partner-sla.md"]],
        ),
    )
    platform_breaker = create_concept(
        session,
        DraftConceptCreate(
            context_space_id=workspace.id,
            canonical_name="Platform Circuit Breaker",
            definition="The platform safety control that limits cascading failures.",
            owning_domain="platform",
            support_item_ids=[support["platform-circuit-breaker.md"]],
        ),
    )
    restricted_concept = create_concept(
        session,
        DraftConceptCreate(
            context_space_id=workspace.id,
            canonical_name="Private Escalation Trigger",
            definition="A sensitive review-only signal that can justify escalation.",
            owning_domain="sales_ops",
            support_item_ids=[support["private-escalation-note.md"]],
        ),
    )
    promoted_concept = create_concept(
        session,
        DraftConceptCreate(
            context_space_id=workspace.id,
            canonical_name="VIP Escalation Insight",
            definition="A workspace-visible snapshot distilled from personal context.",
            owning_domain="sales_ops",
            support_item_ids=[promoted_support.id],
        ),
    )
    session.flush()

    review_concept(session, ops_playbook, admin_actor, ReviewAction.OFFICIALIZE, policy=policy)
    review_concept(session, partner_sla, reviewer_actor, ReviewAction.OFFICIALIZE, policy=policy)
    review_concept(session, platform_breaker, admin_actor, ReviewAction.OFFICIALIZE, policy=policy)
    review_concept(
        session, restricted_concept, admin_actor, ReviewAction.OFFICIALIZE, policy=policy
    )
    review_concept(session, promoted_concept, admin_actor, ReviewAction.OFFICIALIZE, policy=policy)

    official_relation = create_relation(
        session,
        DraftRelationCreate(
            context_space_id=workspace.id,
            subject_concept_id=ops_playbook.id,
            predicate="depends_on",
            object_concept_id=partner_sla.id,
            description="The playbook references the partner SLA during response.",
            support_item_ids=[support["ops-playbook.md"]],
        ),
    )
    review_relation(
        session, official_relation, reviewer_actor, ReviewAction.OFFICIALIZE, policy=policy
    )

    cross_domain_relation = create_relation(
        session,
        DraftRelationCreate(
            context_space_id=workspace.id,
            subject_concept_id=partner_sla.id,
            predicate="depends_on",
            object_concept_id=platform_breaker.id,
            description="Partner response depends on the platform safety mechanism.",
            support_item_ids=[support["platform-circuit-breaker.md"]],
        ),
    )
    cross_domain_relation.verification_state = VerificationState.REVIEW_REQUIRED

    legacy_decision = create_decision(
        session,
        DraftDecisionCreate(
            context_space_id=workspace.id,
            title="Legacy Escalation Routing",
            decision_statement="Escalate incidents only after partner SLA breach confirmation.",
            problem_statement="The old policy favored SLA confirmation before escalation.",
            rationale="This kept the process simple but delayed some escalations.",
            constraints=["Historic partner workflow"],
            impact_summary="Created slow paths for high-value incidents.",
            owning_domain="sales_ops",
            support_item_ids=[support["legacy-routing.md"]],
            linked_concept_ids=[ops_playbook.id, partner_sla.id],
        ),
    )
    review_decision(
        session, legacy_decision, reviewer_actor, ReviewAction.OFFICIALIZE, policy=policy
    )

    modern_decision = create_decision(
        session,
        DraftDecisionCreate(
            context_space_id=workspace.id,
            title="Risk-Based Escalation Routing",
            decision_statement=(
                "Escalate VIP incidents when risk signals or private "
                "review inputs justify urgency."
            ),
            problem_statement=(
                "The workspace needed faster escalation for "
                "high-value incidents."
            ),
            rationale=(
                "Visible playbook guidance plus promoted insight "
                "improve speed without exposing private origins."
            ),
            constraints=["Review private inputs through promoted support only"],
            impact_summary=(
                "Makes high-value incident routing faster while "
                "preserving disclosure controls."
            ),
            owning_domain="sales_ops",
            support_item_ids=[support["ops-playbook.md"], promoted_support.id],
            linked_concept_ids=[ops_playbook.id, promoted_concept.id, restricted_concept.id],
            linked_relation_ids=[official_relation.id],
            supersedes_decision_id=legacy_decision.id,
        ),
    )
    review_decision(session, modern_decision, admin_actor, ReviewAction.OFFICIALIZE, policy=policy)
    legacy_decision.lifecycle_state = DecisionLifecycleState.SUPERSEDED
    legacy_decision.supersedes_decision_id = None

    session.commit()


def _seed_actors(session: Session, workspace: ContextSpace) -> dict[str, Actor]:
    actor_specs = {
        "member": Actor(
            id=stable_id("actor", workspace.id, "member"),
            context_space_id=workspace.id,
            principal_key="principal:tars",
            actor_kind=ActorKind.HUMAN,
            display_name="Member",
            base_role=BaseRole.MEMBER,
            auth_token="demo-member-token",
            scoped_capabilities=[{"capability": Capability.OPERATE.value, "scope": "workspace"}],
            preferred_consumer_scope=ConsumerScope.MEMBER,
        ),
        "operator": Actor(
            id=stable_id("actor", workspace.id, "operator"),
            context_space_id=workspace.id,
            principal_key="principal:operator",
            actor_kind=ActorKind.HUMAN,
            display_name="Operator",
            base_role=BaseRole.MEMBER,
            auth_token="demo-operator-token",
            scoped_capabilities=[
                {"capability": Capability.OPERATE.value, "scope": "workspace"},
                {"capability": Capability.MANAGE_CONNECTORS.value, "scope": "workspace"},
            ],
            preferred_consumer_scope=ConsumerScope.REVIEW,
        ),
        "reviewer": Actor(
            id=stable_id("actor", workspace.id, "reviewer"),
            context_space_id=workspace.id,
            principal_key="principal:reviewer",
            actor_kind=ActorKind.HUMAN,
            display_name="Domain Reviewer",
            base_role=BaseRole.MEMBER,
            auth_token="demo-reviewer-token",
            scoped_capabilities=[{"capability": Capability.REVIEW.value, "scope": "sales_ops"}],
            preferred_consumer_scope=ConsumerScope.REVIEW,
        ),
        "admin": Actor(
            id=stable_id("actor", workspace.id, "admin"),
            context_space_id=workspace.id,
            principal_key="principal:admin",
            actor_kind=ActorKind.HUMAN,
            display_name="Workspace Admin",
            base_role=BaseRole.ADMIN,
            auth_token="demo-admin-token",
            scoped_capabilities=[{"capability": Capability.REVIEW.value, "scope": "workspace"}],
            preferred_consumer_scope=ConsumerScope.ADMIN,
        ),
    }
    session.add_all(actor_specs.values())
    session.flush()
    session.add(
        ReviewScopeGrant(
            id=stable_id("grant", workspace.id, "reviewer"),
            context_space_id=workspace.id,
            actor_id=actor_specs["reviewer"].id,
            review_domains=["sales_ops"],
            allowed_review_actions=[action.value for action in ReviewAction],
            target_object_kinds=["concept", "relation", "decision"],
        )
    )
    session.add(
        ConnectorScopeGrant(
            id=stable_id("connector", workspace.id, "operator"),
            context_space_id=workspace.id,
            actor_id=actor_specs["operator"].id,
            allowed_connector_actions=["sync", "pause", "resume"],
        )
    )
    return actor_specs


def _seed_connection(
    session: Session,
    context_space: ContextSpace,
    *,
    source_label: str,
    boundary_locator: str,
    template_key: str,
    visibility_class: VisibilityClass,
) -> SourceConnection:
    connection = SourceConnection(
        id=stable_id("source", context_space.id, source_label, boundary_locator),
        context_space_id=context_space.id,
        provider="filesystem",
        source_label=source_label,
        source_boundary_locator=boundary_locator,
        template_key=template_key,
        visibility_class=visibility_class,
        sync_mode=SyncMode.POLLING,
        sync_interval_seconds=300,
        source_connection_state=SourceConnectionState.PENDING_SETUP,
        effective_sync_policy={"mode": "fixture", "template_key": template_key},
    )
    session.add(connection)
    return connection


def _support_lookup(session: Session) -> dict[str, str]:
    support_items = session.scalars(select(SupportItem).order_by(SupportItem.id.asc())).all()
    lookup: dict[str, str] = {}
    for support_item in support_items:
        if support_item.artifact is None:
            continue
        lookup[support_item.artifact.external_id] = support_item.id
    return lookup

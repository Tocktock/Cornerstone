from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from cornerstone.config import Settings
from cornerstone.domain.enums import ActorType, SyncMode
from cornerstone.domain.models import Actor, Base, ContextSpace, SourceConnection
from cornerstone.services.sync import run_sync


def initialize_database(engine) -> None:
    Base.metadata.create_all(engine)


def seed_demo(session: Session, settings: Settings) -> None:
    context_space = session.scalar(
        select(ContextSpace).where(ContextSpace.namespace == settings.default_context_space_namespace)
    )
    if context_space is None:
        context_space = ContextSpace(
            name=settings.default_context_space_name,
            namespace=settings.default_context_space_namespace,
            review_policy={"require_grounding_for_official": True},
            visibility_policy={"default": "internal"},
        )
        session.add(context_space)
        session.flush()

    existing_actor_names = {
        actor.display_name
        for actor in session.scalars(select(Actor).where(Actor.context_space_id == context_space.id)).all()
    }
    default_actors = [
        ("Review Admin", ActorType.HUMAN, ["admin", "reviewer"]),
        ("Knowledge Operator", ActorType.HUMAN, ["operator"]),
        ("AI Worker", ActorType.AI, ["worker"]),
    ]
    for display_name, actor_type, roles in default_actors:
        if display_name not in existing_actor_names:
            session.add(
                Actor(
                    context_space_id=context_space.id,
                    actor_type=actor_type,
                    display_name=display_name,
                    roles=roles,
                    external_identities={},
                )
            )
    session.flush()

    source_path = Path(settings.source_root)
    connection = session.scalar(
        select(SourceConnection).where(
            SourceConnection.context_space_id == context_space.id,
            SourceConnection.provider == "filesystem",
            SourceConnection.external_scope == str(source_path),
        )
    )
    if connection is None:
        connection = SourceConnection(
            context_space_id=context_space.id,
            provider="filesystem",
            external_scope=str(source_path),
            sync_mode=SyncMode.POLLING,
            sync_interval_seconds=settings.default_sync_interval_seconds,
            settings={"recursive": True},
        )
        session.add(connection)
        session.flush()

    run_sync(session, connection)
    session.commit()

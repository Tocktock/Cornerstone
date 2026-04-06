from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from cornerstone.domain.enums import BaseRole, ConsumerScope
from cornerstone.domain.models import Actor, ReviewScopeGrant

bearer_scheme = HTTPBearer(auto_error=False)


def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.db.session_factory() as session:
        yield session


def get_settings(request: Request):
    return request.app.state.settings


def get_current_actor(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Actor:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token."
        )
    actor = db.scalar(select(Actor).where(Actor.auth_token == credentials.credentials).limit(1))
    if actor is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown actor token.")
    return actor


def get_consumer_scope(
    requested_scope: ConsumerScope | None = Query(default=None),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
) -> ConsumerScope:
    if requested_scope is None:
        return actor.preferred_consumer_scope
    if requested_scope is ConsumerScope.MEMBER:
        return ConsumerScope.MEMBER
    if actor.base_role is BaseRole.ADMIN:
        return requested_scope
    if requested_scope is ConsumerScope.REVIEW:
        has_review_grant = db.scalar(
            select(ReviewScopeGrant).where(ReviewScopeGrant.actor_id == actor.id).limit(1)
        )
        if has_review_grant:
            return ConsumerScope.REVIEW
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Requested consumer scope is not allowed."
    )

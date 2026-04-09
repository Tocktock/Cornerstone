from __future__ import annotations

from typing import Any

import mcp.types as mcp_types
from fastapi import HTTPException
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from cornerstone.api.deps import resolve_actor_from_bearer_token, resolve_consumer_scope
from cornerstone.config import Settings
from cornerstone.database import Database
from cornerstone.domain.enums import ConsumerScope, ResourceKind
from cornerstone.domain.models import Actor, Concept, ConceptRelation, ContextSpace, DecisionRecord
from cornerstone.services.answering import answer_query, search_context
from cornerstone.services.catalog import get_policy, resource_visible_to_consumer
from cornerstone.services.read_surfaces import (
    context_space_or_404,
    graph_slice_response,
    provenance_response,
)
from cornerstone.services.serialization import (
    concept_envelope,
    decision_envelope,
    relation_envelope,
)


class BearerActorMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, database: Database):
        super().__init__(app)
        self.database = database

    async def dispatch(self, request: Request, call_next):
        with self.database.session_factory() as db:
            try:
                actor = resolve_actor_from_bearer_token(_bearer_token(request), db)
            except HTTPException as exc:
                return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        request.state.actor = actor
        return await call_next(request)


def create_mcp_app(settings: Settings, database: Database):
    server = FastMCP(
        name="Cornerstone MCP",
        instructions=(
            "Read-only model-facing transport for Cornerstone's canonical serving contract. "
            "Use the returned structured JSON as the authoritative result."
        ),
        streamable_http_path="/mcp",
    )
    _restrict_to_tools_only(server)

    @server.tool(name="search_context", structured_output=True)
    def search_context_tool(
        query: str,
        consumer_scope: ConsumerScope | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        with database.session_factory() as db:
            _, context_space, scope = _resolve_request_context(ctx, consumer_scope, db)
            return search_context(db, settings, context_space, scope, query).model_dump(mode="json")

    @server.tool(name="get_answer", structured_output=True)
    def get_answer_tool(
        query: str,
        consumer_scope: ConsumerScope | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        with database.session_factory() as db:
            _, context_space, scope = _resolve_request_context(ctx, consumer_scope, db)
            return answer_query(db, settings, context_space, scope, query).model_dump(mode="json")

    @server.tool(name="get_concept", structured_output=True)
    def get_concept_tool(
        resource_id: str,
        consumer_scope: ConsumerScope | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        with database.session_factory() as db:
            _, context_space, scope = _resolve_request_context(ctx, consumer_scope, db)
            concept = db.get(Concept, resource_id)
            if concept is None:
                raise ToolError("Concept not found.")
            policy = get_policy(db, context_space.id)
            if concept.context_space_id != context_space.id or not resource_visible_to_consumer(
                concept, scope, policy
            ):
                raise ToolError("Concept not found.")
            return concept_envelope(settings, context_space, policy, concept, scope).model_dump(
                mode="json"
            )

    @server.tool(name="get_relation", structured_output=True)
    def get_relation_tool(
        resource_id: str,
        consumer_scope: ConsumerScope | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        with database.session_factory() as db:
            _, context_space, scope = _resolve_request_context(ctx, consumer_scope, db)
            relation = db.get(ConceptRelation, resource_id)
            if relation is None:
                raise ToolError("Relation not found.")
            policy = get_policy(db, context_space.id)
            if relation.context_space_id != context_space.id or not resource_visible_to_consumer(
                relation, scope, policy
            ):
                raise ToolError("Relation not found.")
            return relation_envelope(settings, context_space, policy, relation, scope).model_dump(
                mode="json"
            )

    @server.tool(name="get_decision", structured_output=True)
    def get_decision_tool(
        resource_id: str,
        consumer_scope: ConsumerScope | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        with database.session_factory() as db:
            _, context_space, scope = _resolve_request_context(ctx, consumer_scope, db)
            decision = db.get(DecisionRecord, resource_id)
            if decision is None:
                raise ToolError("Decision not found.")
            policy = get_policy(db, context_space.id)
            if decision.context_space_id != context_space.id or not resource_visible_to_consumer(
                decision, scope, policy
            ):
                raise ToolError("Decision not found.")
            return decision_envelope(
                db, settings, context_space, policy, decision, scope
            ).model_dump(mode="json")

    @server.tool(name="get_graph_slice", structured_output=True)
    def get_graph_slice_tool(
        root: str | None = None,
        consumer_scope: ConsumerScope | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        with database.session_factory() as db:
            _, context_space, scope = _resolve_request_context(ctx, consumer_scope, db)
            return graph_slice_response(
                db, settings, context_space, scope, root=root
            ).model_dump(mode="json")

    @server.tool(name="follow_provenance", structured_output=True)
    def follow_provenance_tool(
        resource_kind: ResourceKind,
        resource_id: str,
        consumer_scope: ConsumerScope | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        with database.session_factory() as db:
            _, _, scope = _resolve_request_context(ctx, consumer_scope, db)
            try:
                envelope = provenance_response(
                    db,
                    settings,
                    consumer_scope=scope,
                    resource_kind=resource_kind,
                    resource_id=resource_id,
                )
            except HTTPException as exc:
                raise ToolError(_http_exception_detail(exc)) from exc
            return envelope.model_dump(mode="json")

    app = server.streamable_http_app()
    app.add_middleware(BearerActorMiddleware, database=database)
    return app


def _restrict_to_tools_only(server: FastMCP) -> None:
    # FastMCP registers prompt/resource handlers by default; remove them so the
    # advertised capability surface matches Cornerstone's read-only tool contract.
    for request_type in (
        mcp_types.ListPromptsRequest,
        mcp_types.GetPromptRequest,
        mcp_types.ListResourcesRequest,
        mcp_types.ReadResourceRequest,
        mcp_types.ListResourceTemplatesRequest,
    ):
        server._mcp_server.request_handlers.pop(request_type, None)


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("Authorization")
    if header is None:
        return None
    scheme, _, credentials = header.partition(" ")
    if scheme.lower() != "bearer" or not credentials:
        return None
    return credentials.strip()


def _resolve_request_context(
    ctx: Context | None,
    requested_scope: ConsumerScope | None,
    db: Session,
) -> tuple[Actor, ContextSpace, ConsumerScope]:
    actor = _actor_from_context(ctx)
    scope = _resolve_scope_with_session(actor, requested_scope, db)
    context_space = context_space_or_404(db, actor.context_space_id)
    return actor, context_space, scope


def _resolve_scope_with_session(
    actor: Actor, requested_scope: ConsumerScope | None, db: Session
) -> ConsumerScope:
    try:
        refreshed_actor = db.scalar(select(Actor).where(Actor.id == actor.id).limit(1))
        if refreshed_actor is None:
            raise ToolError("Actor context is unavailable.")
        return resolve_consumer_scope(requested_scope, refreshed_actor, db)
    except HTTPException as exc:
        raise ToolError(_http_exception_detail(exc)) from exc


def _actor_from_context(ctx: Context | None) -> Actor:
    request = _request_from_context(ctx)
    actor = getattr(request.state, "actor", None)
    if actor is None:
        raise ToolError("Actor context is unavailable.")
    return actor


def _request_from_context(ctx: Context | None) -> Request:
    if ctx is None or ctx.request_context is None or ctx.request_context.request is None:
        raise ToolError("HTTP request context is unavailable.")
    request = ctx.request_context.request
    if not isinstance(request, Request):
        raise ToolError("HTTP request context is unavailable.")
    return request


def _http_exception_detail(exc: HTTPException) -> str:
    if isinstance(exc.detail, str):
        return exc.detail
    return "Request failed."

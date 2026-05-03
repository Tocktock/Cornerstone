from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI

from cornerstone import __version__
from cornerstone.api.routes import (
    artifacts,
    audit,
    concept_relations,
    concepts,
    connector_catalog,
    connectors,
    context,
    decision_records,
    evaluations,
    evidence,
    health,
    integration,
    manual_sources,
    ontology,
    sources,
    sync_runtime,
)
from cornerstone.config import Settings, get_settings
from cornerstone.observability import RequestLoggingMiddleware, configure_logging, log_event
from cornerstone.store import InMemoryStore


def create_app(
    store: Any | None = None,
    *,
    settings: Settings | None = None,
    validate_runtime_config: bool | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    should_validate_runtime_config = store is None if validate_runtime_config is None else validate_runtime_config
    if should_validate_runtime_config:
        settings.assert_runtime_config_safe()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="Cornerstone Backend API",
        description="Evidence-backed organizational context layer backend.",
        version=__version__,
    )
    if store is not None:
        app.state.store = store
        persistence_backend = store.__class__.__name__
    elif settings.persistence_backend == "postgres":
        from cornerstone.persistence.database import create_persistent_store

        app.state.store = create_persistent_store(settings)
        persistence_backend = "postgres"
    else:
        app.state.store = InMemoryStore()
        persistence_backend = "memory"
    log_event(
        "app.persistence_configured",
        persistenceBackend=persistence_backend,
        appEnv=settings.app_env,
        productionMode=settings.production_mode,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(health.router)
    app.include_router(connector_catalog.router, prefix=settings.api_prefix)
    app.include_router(connectors.router, prefix=settings.api_prefix)
    app.include_router(sources.router, prefix=settings.api_prefix)
    app.include_router(manual_sources.router, prefix=settings.api_prefix)
    app.include_router(sync_runtime.router, prefix=settings.api_prefix)
    app.include_router(artifacts.router, prefix=settings.api_prefix)
    app.include_router(evidence.router, prefix=settings.api_prefix)
    app.include_router(concepts.router, prefix=settings.api_prefix)
    app.include_router(concept_relations.router, prefix=settings.api_prefix)
    app.include_router(ontology.router, prefix=settings.api_prefix)
    app.include_router(integration.router, prefix=settings.api_prefix)
    app.include_router(decision_records.router, prefix=settings.api_prefix)
    app.include_router(context.router, prefix=settings.api_prefix)
    app.include_router(evaluations.router, prefix=settings.api_prefix)
    app.include_router(audit.router, prefix=settings.api_prefix)
    return app


# Test collection can import create_app without constructing the process-global ASGI app.
# Production and normal ASGI imports still expose `app`.
if os.getenv("CORNERSTONE_SKIP_GLOBAL_APP") == "1":
    app = FastAPI(
        title="Cornerstone Backend API",
        description="Evidence-backed organizational context layer backend.",
        version=__version__,
    )
else:
    app = create_app()

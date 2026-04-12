from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cornerstone.api.router import router
from cornerstone.config import Settings
from cornerstone.database import Database
from cornerstone.mcp import create_mcp_app
from cornerstone.services.bootstrap import ensure_minimal_bootstrap, initialize_database, seed_demo


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings()
    database = Database(resolved_settings.database_url)
    mcp_app = create_mcp_app(resolved_settings, database)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.db = database
        initialize_database(database.engine, reset=resolved_settings.reset_database_on_start)
        with database.session_factory() as session:
            ensure_minimal_bootstrap(session, resolved_settings)
            if resolved_settings.seed_demo_content:
                seed_demo(session, resolved_settings)
            else:
                session.commit()
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    app = FastAPI(title=resolved_settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix=resolved_settings.api_prefix)
    app.router.add_route("/mcp", mcp_app, methods=["GET", "POST", "DELETE"])

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "name": resolved_settings.app_name,
            "message": "Cornerstone backend is running.",
            "docs": "/docs",
            "api": resolved_settings.api_prefix,
            "mcp": "/mcp",
        }

    return app

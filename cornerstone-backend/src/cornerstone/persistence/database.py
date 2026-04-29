from __future__ import annotations

from cornerstone.config import Settings
from cornerstone.persistence.extensions import parse_required_extensions, verify_required_extensions
from cornerstone.persistence.store import SqlAlchemyStore, create_sqlalchemy_engine


def create_persistent_store(settings: Settings) -> SqlAlchemyStore:
    engine = create_sqlalchemy_engine(settings.database_url, echo=settings.database_echo)
    if settings.persistence_backend == "postgres" and settings.verify_postgres_extensions_on_startup:
        verify_required_extensions(engine, parse_required_extensions(settings.postgres_required_extensions_raw))
    return SqlAlchemyStore(engine)

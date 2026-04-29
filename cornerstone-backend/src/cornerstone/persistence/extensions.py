from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import Engine, text

REQUIRED_POSTGRES_EXTENSIONS: tuple[str, ...] = ("pgcrypto", "citext", "vector")


class DatabaseExtensionError(RuntimeError):
    """Raised when the configured PostgreSQL database is missing required extensions."""


def parse_required_extensions(raw: str | Iterable[str] | None = None) -> tuple[str, ...]:
    if raw is None:
        return REQUIRED_POSTGRES_EXTENSIONS
    values = raw.split(",") if isinstance(raw, str) else list(raw)
    normalized = tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
    return normalized or REQUIRED_POSTGRES_EXTENSIONS


def ensure_postgres_engine(engine: Engine) -> None:
    if engine.dialect.name != "postgresql":
        raise DatabaseExtensionError(
            "PostgreSQL persistence requires a postgresql+psycopg DATABASE_URL. "
            f"Configured dialect: {engine.dialect.name!r}."
        )


def verify_required_extensions(
    engine: Engine,
    required_extensions: Iterable[str] = REQUIRED_POSTGRES_EXTENSIONS,
) -> set[str]:
    """Verify required extensions are installed in the current PostgreSQL database.

    Migrations create extensions. This runtime check fails fast if an operator points the API at
    a database where migrations/extensions were not applied or pgvector is unavailable.
    """

    ensure_postgres_engine(engine)
    required = set(parse_required_extensions(required_extensions))
    with engine.connect() as connection:
        installed = set(
            connection.execute(text("select extname from pg_extension")).scalars().all()
        )
    missing = required - installed
    if missing:
        raise DatabaseExtensionError(
            "Missing required PostgreSQL extensions: "
            f"{', '.join(sorted(missing))}. Run Alembic migrations first."
        )
    return installed


def extension_create_statements(
    required_extensions: Iterable[str] = REQUIRED_POSTGRES_EXTENSIONS,
) -> list[str]:
    return [f'CREATE EXTENSION IF NOT EXISTS "{extension}";' for extension in required_extensions]

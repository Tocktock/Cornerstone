from __future__ import annotations

import os

import psycopg
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url

from cornerstone.app import create_app
from cornerstone.config import Settings

DEFAULT_TEST_DATABASE_URL = (
    "postgresql+psycopg://cornerstone:cornerstone@localhost:55432/cornerstone_test"
)


def _psycopg_dsn(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://")


@pytest.fixture(scope="session")
def test_database_url() -> str:
    database_url = os.getenv("CORNERSTONE_TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL)
    try:
        _ensure_database(database_url)
    except psycopg.Error as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres test database is unavailable: {exc}")
    return database_url


def _ensure_database(database_url: str) -> None:
    target_url = make_url(database_url)
    admin_url = target_url.set(database="postgres")
    dsn = _psycopg_dsn(admin_url.render_as_string(hide_password=False))
    database_name = target_url.database
    if database_name is None:
        raise RuntimeError("Test database URL must include a database name.")
    with psycopg.connect(dsn, autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
        if cursor.fetchone() is None:
            cursor.execute(f'CREATE DATABASE "{database_name}"')


@pytest.fixture()
def client(test_database_url: str) -> TestClient:
    settings = Settings(
        database_url=test_database_url,
        auto_seed_demo=True,
        reset_database_on_start=True,
        fixed_now="2026-04-06T09:00:00+09:00",
        cors_origins=["http://localhost:4173"],
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def tokens(client: TestClient) -> dict[str, str]:
    bootstrap = client.get("/api/v1/bootstrap")
    bootstrap.raise_for_status()
    return {actor["display_name"]: actor["token"] for actor in bootstrap.json()["actors"]}


@pytest.fixture()
def headers(tokens: dict[str, str]):
    return {
        "member": {"Authorization": f"Bearer {tokens['Member']}"},
        "reviewer": {"Authorization": f"Bearer {tokens['Domain Reviewer']}"},
        "admin": {"Authorization": f"Bearer {tokens['Workspace Admin']}"},
        "operator": {"Authorization": f"Bearer {tokens['Operator']}"},
    }

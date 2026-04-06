from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


class Database:
    def __init__(self, database_url: str):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        if database_url.startswith("sqlite:///"):
            db_path = Path(database_url.removeprefix("sqlite:///"))
            if db_path.parent and str(db_path.parent) != ".":
                db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            database_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            class_=Session,
        )

    def session(self) -> Iterator[Session]:
        with self.session_factory() as session:
            yield session

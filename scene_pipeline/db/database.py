from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from scene_pipeline.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    from scene_pipeline.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_level3_columns()


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _ensure_level3_columns() -> None:
    """Best-effort additive migration for local DBs created before Level 3."""

    inspector = inspect(engine)
    if "scene_records" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("scene_records")}
    statements = []
    if "quality_score" not in columns:
        statements.append("ALTER TABLE scene_records ADD COLUMN quality_score FLOAT")
    if "quality_payload" not in columns:
        statements.append("ALTER TABLE scene_records ADD COLUMN quality_payload JSON")
    if not statements:
        return
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

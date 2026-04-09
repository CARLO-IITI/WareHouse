"""Database connection, session management, and schema initialization."""

from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "warehouse.db",
)


def _enable_wal_mode(dbapi_conn, connection_record):
    """Enable WAL mode for better concurrent read performance."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


def create_db_engine(db_path: str | None = None):
    """Create a SQLAlchemy engine with optimized SQLite settings."""
    path = db_path or DEFAULT_DB_PATH
    engine = create_engine(
        f"sqlite:///{path}",
        pool_size=5,
        pool_pre_ping=True,
        echo=False,
    )
    event.listen(engine, "connect", _enable_wal_mode)
    return engine


def init_db(engine) -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)


def drop_db(engine) -> None:
    """Drop all tables (used for re-initialization)."""
    Base.metadata.drop_all(engine)


def get_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session(session_factory: sessionmaker):
    """Context manager yielding a transactional session."""
    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_table_counts(session: Session) -> dict[str, int]:
    """Return row counts for all tables."""
    tables = ["zones", "racks", "slots", "items", "order_history"]
    counts = {}
    for table in tables:
        result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
        counts[table] = result.scalar()
    return counts

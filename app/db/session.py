"""Database engine and session factory.

Creates the SQLAlchemy `Engine` from the configured database URL and a
`SessionLocal` factory used to obtain transactional sessions for each
request or unit of work. Also exposes `init_db()` for one-shot table
creation during local development.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.models import Base

engine = create_engine(settings.database_url, echo=False)

SessionLocal = sessionmaker(engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create all tables declared on the ORM `Base` metadata.

    **Alembic owns the schema** — run `alembic upgrade head` instead
    (compose and the Fly release command both do). This helper remains
    only for throwaway databases in tests and scratch work, where paying
    for a migration run buys nothing.

    Idempotent, but not a migration tool: it does not detect or apply
    column additions, type changes, or any schema diff, and it does not
    stamp an Alembic revision — a database created this way looks
    un-migrated to Alembic.

    Ensures the pgvector `vector` extension exists first, since the
    `chunks.embedding` column depends on it.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
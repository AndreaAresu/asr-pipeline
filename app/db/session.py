"""Database engine and session factory.

Creates the SQLAlchemy `Engine` from the configured database URL and a
`SessionLocal` factory used to obtain transactional sessions for each
request or unit of work. Also exposes `init_db()` for one-shot table
creation during local development.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.models import Base


engine = create_engine(settings.database_url, echo=False)

SessionLocal = sessionmaker(engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create all tables declared on the ORM `Base` metadata.

    Idempotent: tables that already exist are left untouched. This is
    suitable for the first run and for local development, but it is not a
    migration tool — it does not detect or apply column additions, type
    changes, index drops, or any schema diff.
    """
    Base.metadata.create_all(bind=engine)
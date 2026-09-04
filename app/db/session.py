"""Database engine and session factory.

Creates the SQLAlchemy `Engine` from the configured database URL and a
`SessionLocal` factory used to obtain transactional sessions for each
request or unit of work. **Alembic owns the schema**: nothing here creates
tables, run `alembic upgrade head` (the compose `api` service does, before
the server binds).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, echo=False)

SessionLocal = sessionmaker(engine, autocommit=False, autoflush=False)


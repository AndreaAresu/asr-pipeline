"""Alembic environment.

The database URL is taken from the application `Settings` rather than
from alembic.ini, so migrations always target whatever the app itself
would connect to — including in containers, where the URL is assembled
from injected environment variables. No credentials live in alembic.ini.

`app.db.models` is imported for its side effect of registering every
model on `Base.metadata`; without it autogenerate sees an empty schema
and cheerfully proposes dropping all your tables.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

from app.config import settings
from app.db.models import Base
import app.db.models  # noqa: F401  (registers the models on Base.metadata)

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit migration SQL without connecting to a database."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Catch column type changes, which Alembic ignores by default.
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

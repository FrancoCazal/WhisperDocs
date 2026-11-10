from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app import models  # noqa: F401  (ensure models are imported for autogenerate)
from app.config import settings
from app.db import Base

config = context.config

# Use psycopg (v3) sync driver for migrations. asyncpg + Windows ProactorLoop
# has known reset issues during the Postgres startup handshake. Migrations
# don't need async — use the sync path and keep runtime on asyncpg.
sync_url = settings.psycopg_database_url.replace("?ssl=disable", "")
if not sync_url.startswith("postgresql+psycopg://"):
    sync_url = sync_url.replace("postgresql://", "postgresql+psycopg://", 1)

config.set_main_option("sqlalchemy.url", sync_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# The checkpointer's tables: LangGraph creates and migrates them
# (app/agent/checkpointer.py), so autogenerate and `alembic check` leave them
# alone.
LANGGRAPH_TABLES = frozenset(
    {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
)


def include_name(name, type_, parent_names) -> bool:
    if type_ == "table":
        return name not in LANGGRAPH_TABLES
    return parent_names.get("table_name") not in LANGGRAPH_TABLES


def run_migrations_offline() -> None:
    context.configure(
        url=sync_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, include_name=include_name
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

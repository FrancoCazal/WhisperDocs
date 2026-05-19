# Alembic migrations

Alembic is the official database migration tool of the SQLAlchemy ecosystem. It lets you version schema changes over time, apply and revert them in a controlled fashion, and keep multiple environments in sync (development, staging, production). It's the equivalent of Django migrations but decoupled from any specific web framework.

## Why migrations exist

Without migrations, evolving a schema in production means running SQL manually, which creates problems:

- There is no historical record of which changes were applied.
- Different environments end up out of sync.
- Rollbacks are manual and error-prone.
- New developer onboarding requires reproducing ad-hoc SQL.

With migrations, every schema change is versioned as a Python script, lives in the repository, and is applied identically in every environment. A new developer clones the repo, runs `alembic upgrade head`, and ends up with the exact schema that production has.

## Initial setup

Inside a project that uses SQLAlchemy:

```bash
alembic init alembic
```

This creates:

- `alembic/` — directory with the migration environment.
- `alembic/env.py` — environment configuration (connection, metadata).
- `alembic/versions/` — directory where migration scripts live.
- `alembic/script.py.mako` — template for how new scripts are generated.
- `alembic.ini` — runner configuration.

`env.py` needs to know two things: how to connect to the DB, and where SQLAlchemy's metadata lives (the `Base.metadata` with the declared models). A modern async `env.py` looks like this:

```python
import asyncio
from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config
from app.config import settings
from app.db import Base
from app import models  # import to register all models

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()

async def run_migrations_online():
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()

asyncio.run(run_migrations_online())
```

The `run_sync` pattern is key: Alembic is written in sync, and `run_sync` allows it to use an async connection underneath without changing its internal API.

## Autogenerate

The most-used command:

```bash
alembic revision --autogenerate -m "add users table"
```

This generates a script in `alembic/versions/<hash>_add_users_table.py` with the operations needed to take the current schema to the state defined by the models.

Autogenerate compares:

- **The current DB state** (introspected via connection).
- **The Python models' state** (via `target_metadata`).

And emits operations: `create_table`, `add_column`, `drop_column`, `create_index`, etc.

Known limitations of autogenerate:

- **It does not detect renames**: if you rename a column, it sees it as drop + add — data is lost. You have to edit the script manually.
- **It does not always detect server default changes**.
- **Type changes can be ambiguous** (e.g. String(100) to String(200)): sometimes ignored.
- **It does not detect data migrations**: only schema changes.
- **Custom types (pgvector, etc.) may require manual tweaks**.

That's why **you should always read and edit the generated script** before applying it. Autogenerate is an assistant, not a substitute for human judgment.

## Writing migrations by hand

Sometimes it's clearer to write the migration directly. A typical migration:

```python
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email", "users", ["email"])

def downgrade() -> None:
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
```

The most-used operations:

- `op.create_table()`, `op.drop_table()`
- `op.add_column()`, `op.drop_column()`, `op.alter_column()`
- `op.create_index()`, `op.drop_index()`
- `op.create_foreign_key()`, `op.drop_constraint()`
- `op.execute("RAW SQL")` for things not covered (creating extensions, functions, etc.)

## Apply and revert

```bash
alembic upgrade head           # apply all pending migrations
alembic upgrade +1             # apply only one
alembic upgrade <revision>     # apply up to a specific revision

alembic downgrade -1           # revert one migration
alembic downgrade base         # revert all

alembic current                # show the current revision
alembic history --verbose      # show full history
```

Every migration defines an `upgrade()` and a `downgrade()`. `upgrade` runs when moving forward; `downgrade` runs when reverting. A well-written downgrade is symmetric to the upgrade.

## Data migrations

Sometimes the change is not only to the schema but to the data: normalize a column, populate a table from another, transform formats.

```python
def upgrade():
    # Schema change first
    op.add_column("users", sa.Column("full_name", sa.String(200)))

    # Data migration
    op.execute("""
        UPDATE users
        SET full_name = first_name || ' ' || last_name
    """)

    # Cleanup
    op.drop_column("users", "first_name")
    op.drop_column("users", "last_name")
```

For complex transformations you can use the `connection` from the context:

```python
def upgrade():
    connection = op.get_bind()
    result = connection.execute(sa.text("SELECT id, data FROM old_table"))
    for row in result:
        transformed = process(row.data)
        connection.execute(
            sa.text("INSERT INTO new_table (id, data) VALUES (:id, :data)"),
            {"id": row.id, "data": transformed},
        )
```

Data migrations are riskier than schema migrations. In production with large tables, consider running them in background or during maintenance windows.

## Production: critical considerations

### Non-blocking migrations

Operations like `ADD COLUMN NOT NULL` on large tables can lock the table for minutes. Safe patterns:

1. `ADD COLUMN` without NOT NULL.
2. Backfill with small batched updates.
3. `SET NOT NULL` once every row has a value.

```python
def upgrade():
    op.add_column("users", sa.Column("status", sa.String(20), server_default="active"))
    op.alter_column("users", "status", server_default=None)
    # Now everyone has the "active" default, you can set NOT NULL
    op.alter_column("users", "status", nullable=False)
```

### Concurrent indexes

Creating indexes on large tables blocks writes. PostgreSQL provides `CONCURRENTLY`:

```python
def upgrade():
    op.execute("CREATE INDEX CONCURRENTLY idx_users_email ON users(email)")

def downgrade():
    op.execute("DROP INDEX CONCURRENTLY idx_users_email")
```

Watch out: `CONCURRENTLY` cannot run inside a transaction. You'll need to configure Alembic so that migration is not wrapped in a transaction.

### Applying migrations on deploy

Common strategies:

- **At container startup**: the service command runs `alembic upgrade head` before the server. Simple, works well in development and simple deploys. There's a risk if multiple replicas start at the same time — several may try to run migrations simultaneously.
- **As a separate job**: a dedicated pod/job runs migrations before the rolling update of the API. More robust for production.
- **Manually**: in critical production setups, sometimes you prefer running migrations by hand with review windows.

For small projects, the "at startup" pattern is acceptable. For serious production with replicas, the dedicated job is the standard.

### Backups before migrating

Always. Every migration that touches production should run on top of a recent snapshot. Migrations are code; code has bugs. A snapshot is the insurance policy.

## Branches and merging migrations

Alembic supports branches when two developers create migrations in parallel from the same parent.

```
alembic heads
# shows whether there are multiple heads (divergence)

alembic merge -m "merge divergent heads" <rev1> <rev2>
# creates a revision that unifies both
```

In large teams, avoiding merges is better: coordinate who creates migrations and rebase before the PR.

## Typical errors

### "Target database is not up to date"

A migration was attempted in an environment that is not at the expected version. `alembic current` shows where it is; `alembic upgrade head` syncs it.

### "Can't locate revision identified by '<hash>'"

The migration script was deleted, or the `alembic_version` table points to a non-existent revision. Recover the script from git or manually edit `alembic_version`.

### Autogenerate does not detect changes

Typical causes:

- The new models are not imported in `env.py` (or in `app/models/__init__.py`).
- `target_metadata` points to a different `Base` than the one the models use.
- There are multiple `Base` classes in the project and only one was passed to `target_metadata`.

### Migration ran partially and failed

PostgreSQL supports transactional DDL, so by default a failed migration reverts everything. But some operations (like `CREATE INDEX CONCURRENTLY`) escape the transaction. In those cases, manual cleanup may be necessary.

## Alembic and async

Alembic is internally sync. The trick with async engines is to use `connection.run_sync(...)` as in the `env.py` example above. Migration generation runs against a sync connection wrapped over the async one — autogenerate requires no conceptual changes.

## Trade-offs vs declarative schema alone

| Aspect | Declarative only (create_all) | With Alembic |
|--------|-------------------------------|--------------|
| Setup | Zero config | Requires initialization |
| Schema evolution | Manual, ad-hoc | Versioned, reproducible |
| Multi-environment | Hard to keep in sync | Trivial with `upgrade head` |
| Change history | Only in code git | Explicit in `alembic/versions/` |
| Rollbacks | Manual | `downgrade -1` |
| Data migrations | Not supported | First-class |

For prototypes and tests, `Base.metadata.create_all()` is convenient. For any project that has production or staging, Alembic is non-negotiable.

## When not to use Alembic

- Projects using a framework with its own migration system (Django migrations, for example).
- One-shot scripts with no persistent state.
- Throwaway proofs of concept.

For any serious SQLAlchemy backend, Alembic is the standard tool and well worth the initial setup cost.

## References

- Alembic docs: https://alembic.sqlalchemy.org/
- Autogenerate: https://alembic.sqlalchemy.org/en/latest/autogenerate.html
- Async setup: https://alembic.sqlalchemy.org/en/latest/cookbook.html#using-asyncio-with-alembic
- PostgreSQL DDL transactions: https://www.postgresql.org/docs/current/sql-begin.html

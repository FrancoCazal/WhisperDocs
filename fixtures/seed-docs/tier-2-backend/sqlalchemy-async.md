# SQLAlchemy async

SQLAlchemy is the most-used ORM in the Python ecosystem. From version 1.4 onward it includes native async support, and starting with 2.0 the async API is first-class, with types and patterns consistent with the sync one. Using SQLAlchemy async correctly requires understanding several concepts that don't apply in sync mode: the driver, the engine, the session, and in particular the loading of relationships.

## Async architecture

Three main pieces:

1. **Async driver**: the DB connector must support async. For PostgreSQL, `asyncpg` is the de facto standard; `psycopg` in its async mode from version 3 onward also exists.
2. **AsyncEngine**: created with `create_async_engine()`. Wraps the async driver with the connection pool.
3. **AsyncSession**: the session you use in your code. The async equivalent of the sync `Session`, with `await`-able methods.

Typical setup:

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
```

The `expire_on_commit=False` is important: by default, SQLAlchemy expires objects after a commit, which triggers an automatic reload when their attributes are accessed. In async code, that reload would be synchronous (inside an attribute access), triggering an error. Disabling it avoids the problem and is the recommended pattern for async.

## Modern declarative base

In SQLAlchemy 2.0, the modern declarative pattern uses `DeclarativeBase` and `Mapped[...]` with `mapped_column`:

```python
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    orders: Mapped[list["Order"]] = relationship(back_populates="user")

class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    total: Mapped[float] = mapped_column(nullable=False)

    user: Mapped["User"] = relationship(back_populates="orders")
```

Python types map to SQL automatically: `Mapped[int]`, `Mapped[str]`, `Mapped[datetime | None]`. It's more concise and gives better type-checker support than the old `Column(...)` style.

## Modern queries

In 2.0 you use `select()` instead of `session.query(...)`. The API is uniform across sync and async.

```python
from sqlalchemy import select

async def list_users(db: AsyncSession) -> list[User]:
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    return list(result.scalars().all())

async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()
```

To fetch a row by primary key, `db.get(Model, pk)` is simpler:

```python
user = await db.get(User, user_id)
```

## Relationships and loading

The most subtle problem in SQLAlchemy async is relationship loading. In sync mode, lazy loading works magically: `user.orders` fires a query when accessed. In async, that pattern breaks because attribute access cannot be async.

```python
user = await db.get(User, 1)
orders = user.orders  # ERROR in async: MissingGreenlet
```

The classic error is `sqlalchemy.exc.MissingGreenlet`. Solutions:

### selectinload — eager loading with an extra query

```python
from sqlalchemy.orm import selectinload

result = await db.execute(
    select(User).options(selectinload(User.orders)).where(User.id == 1)
)
user = result.scalar_one()
orders = user.orders  # already loaded, no extra query
```

It fires a separate query that pulls all orders for the selected users with `WHERE user_id IN (...)`. It's the recommended default strategy for async because it's predictable and efficient.

### joinedload — eager loading with a JOIN

```python
from sqlalchemy.orm import joinedload

result = await db.execute(
    select(User).options(joinedload(User.orders)).where(User.id == 1)
)
user = result.unique().scalar_one()
```

It uses a JOIN to bring everything in a single query. More efficient if you're loading few objects with a many-to-one relationship, but it can generate a lot of traffic if it's one-to-many with many children (cartesian explosion).

### Lazy with `await`-able

Since SQLAlchemy 2.0 there's `AsyncAttrs` which allows explicit lazy loading:

```python
from sqlalchemy.ext.asyncio import AsyncAttrs

class Base(DeclarativeBase, AsyncAttrs):
    pass

user = await db.get(User, 1)
orders = await user.awaitable_attrs.orders  # triggers async lazy load
```

Useful as a fallback, but it's better to plan the load with `selectinload` from the initial query.

## N+1 problem

The classic problem: load a list of entities and then access a relationship on each one, firing one query per element.

```python
users = (await db.execute(select(User))).scalars().all()
for user in users:
    print(user.orders)  # N extra queries
```

Detecting N+1: enable `echo=True` on the engine and check the SQL logs. If N SELECTs appear after an initial one, you have N+1.

Solution: `selectinload` or `joinedload` in the initial query. In development it's useful to run tests with `echo=True` to detect regressions.

## Transactions

SQLAlchemy async uses implicit transactions: when you open a session, the first query starts a transaction. It's closed with `commit()` or `rollback()`.

```python
async with SessionLocal() as db:
    user = User(email="a@b.com")
    db.add(user)
    await db.flush()  # sends INSERT but does not commit
    order = Order(user_id=user.id, total=100)
    db.add(order)
    await db.commit()
```

`flush()` sends pending changes to the driver but keeps the transaction open. Necessary when you need the DB-generated `id` (e.g. to use as an FK) before the final commit. `commit()` confirms the transaction; `rollback()` discards it.

Nested transactions with `begin_nested()` use savepoints:

```python
async with db.begin():
    user = User(...)
    db.add(user)
    try:
        async with db.begin_nested():
            risky_operation(user)
    except Exception:
        # savepoint rollback, but the outer transaction continues
        pass
    await db.commit()
```

## Dependency injection with FastAPI

The canonical pattern uses an async generator as a dependency:

```python
from collections.abc import AsyncIterator
from fastapi import Depends

async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session

@app.get("/users/{id}")
async def get_user(id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, id)
    return user
```

The session is opened at the start of the request and closed automatically at the end, with implicit rollback if there was an exception and manual commit where appropriate.

## Testing

For async tests with a real DB, use a test DB (don't mock). Useful patterns:

- **DB per test**: create and drop the DB for each test. Slow but clean.
- **Transaction rollback**: wrap each test in a transaction that's rolled back at the end. Fast and isolated, but complicates tests that need multiple transactions.
- **Explicit cleanup**: truncate tables after each test. Fast and simple, with the risk of shared fixtures.

```python
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

@pytest.fixture
async def db():
    engine = create_async_engine("postgresql+asyncpg://test:test@localhost/test_db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
```

## Common errors

### MissingGreenlet

Cause: accessing a relationship or lazy-loaded attribute outside an async context.
Solution: use `selectinload` or `AsyncAttrs.awaitable_attrs`.

### Session used after commit with expire_on_commit=True

Cause: after commit, attributes expired and the first access triggers a sync reload.
Solution: set `expire_on_commit=False` on the sessionmaker.

### Connections exhausted

Cause: forgetting to close sessions or using `AsyncSession` outside a context manager.
Solution: always use `async with SessionLocal() as session:` and monitor the pool with `engine.pool.status()`.

### Session shared between concurrent coroutines

Cause: passing the same session to multiple `asyncio.gather()` tasks.
Solution: each coroutine opens its own session. Sessions are not coroutine-safe.

## Trade-offs with sync mode

| Dimension | Sync | Async |
|-----------|------|-------|
| Simplicity | Very simple | Requires understanding eager loading |
| Performance for I/O-heavy apps | Limited | Much better |
| Library ecosystem | Mature | Less complete (missing adapters for some sync libraries) |
| Debugging | Clear stack traces | More complex |
| Tooling (alembic, etc.) | Full support | Growing support, not yet full |

Alembic, for example, runs in sync by default even in async projects — it's configured to use an async engine internally but migration generation is synchronous. It's not a problem, but it surprises you the first time.

## When not to use SQLAlchemy async

- Projects where most of the code is synchronous (scripts, CLIs, batch ETLs): sync mode is enough.
- Teams without prior experience: the extra learning curve isn't always worth it if queries aren't the bottleneck.
- Projects that depend on libraries that don't have an async version (some auth adapters, certain legacy integrations).

For modern FastAPI APIs with high concurrency and intensive I/O, SQLAlchemy async is almost always the right choice, especially when combined with PostgreSQL + asyncpg.

## References

- SQLAlchemy 2.0 docs: https://docs.sqlalchemy.org/en/20/
- Async ORM tutorial: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- asyncpg: https://github.com/MagicStack/asyncpg
- Unified tutorial (sync + async): https://docs.sqlalchemy.org/en/20/tutorial/index.html

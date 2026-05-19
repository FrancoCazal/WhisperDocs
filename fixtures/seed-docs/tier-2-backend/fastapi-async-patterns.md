# FastAPI async patterns

FastAPI is a modern Python web framework built on top of Starlette and Pydantic, with native async support. Unlike Flask or Django (in their classic WSGI mode), FastAPI is designed from the ground up to leverage the `asyncio` event loop. This makes it especially well-suited for I/O-heavy workloads: external API calls, async database queries, large file processing, LLM response streaming.

## When async matters

Async is **not automatically faster** than synchronous code. It makes sense when the handler spends most of its time waiting — DB I/O, HTTP calls to other services, large file reads, LLM generation. In those cases, the event loop can serve other requests while one is waiting on a response from an external service.

For CPU-bound code (heavy computation, image processing in pure Python, intensive parsing), async **does not help** and in some cases makes things worse because it blocks the entire event loop. For that you have to use `run_in_executor` or move the task to an external worker like Celery.

Rule of thumb:

- Handler does I/O → define it as `async def` and use async clients (`httpx.AsyncClient`, `asyncpg`, etc.)
- Handler does heavy CPU work → define it as a regular `def`; FastAPI runs it in a thread pool.
- Handler does heavy CPU work and *needs* to be async → use `asyncio.to_thread()` or an executor.

## Async handlers

An async handler is structurally identical to a sync one — only the declaration changes:

```python
from fastapi import FastAPI
import httpx

app = FastAPI()

@app.get("/weather/{city}")
async def get_weather(city: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
    return response.json()
```

During the `await`, the event loop is free to serve other requests. In a synchronous handler, the worker is blocked until the external API responds.

## Dependency injection

FastAPI's dependency system is one of its strongest differentiators. It lets you inject resources (DB sessions, HTTP clients, authenticated users) declaratively, with native async support.

```python
from collections.abc import AsyncIterator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

SessionLocal = async_sessionmaker(engine)

async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session

@app.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    return user
```

Key points:

- A dependency can be an async generator (`yield`), equivalent to a context manager. The code after the `yield` runs at the end of the request, typically for cleanup.
- Dependencies are resolved in parallel where possible, respecting topological order.
- They are testable: `app.dependency_overrides[get_db] = lambda: fake_db` swaps out the dependency in tests.

Dependencies can be chained:

```python
async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    user_id = decode_token(token)
    return await db.get(User, user_id)

@app.get("/me")
async def read_me(current: User = Depends(get_current_user)):
    return current
```

## Background tasks

For operations that don't need to block the response (sending a confirmation email, logging to an external service, invalidating cache), FastAPI offers `BackgroundTasks`:

```python
from fastapi import BackgroundTasks

async def send_email_async(to: str, subject: str, body: str) -> None:
    async with httpx.AsyncClient() as client:
        await client.post("https://email-service/send", json={...})

@app.post("/signup")
async def signup(user: UserCreate, background: BackgroundTasks):
    new_user = await create_user_in_db(user)
    background.add_task(send_email_async, user.email, "Welcome", "...")
    return {"status": "ok"}
```

The task is executed after the response is returned to the client, in the same process. This is fine for short, failure-tolerant tasks. For long tasks, with retries, with persistence, you have to use Celery or arq — not BackgroundTasks.

## Streaming responses

When the response is large or generated progressively — LLM streaming, a huge file, server-sent events — `StreamingResponse` lets you send chunks without loading everything in memory:

```python
from fastapi.responses import StreamingResponse

async def llm_stream(prompt: str):
    async with openai.AsyncClient().chat.completions.stream(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    ) as stream:
        async for event in stream:
            if event.type == "content.delta":
                yield event.delta

@app.post("/chat")
async def chat(prompt: str):
    return StreamingResponse(llm_stream(prompt), media_type="text/event-stream")
```

The client receives tokens as the LLM generates them, dramatically improving perceived latency.

## Async middleware

Middlewares intercept every request/response. They are useful for logging, metrics, global auth, rate limiting. FastAPI supports async middlewares natively:

```python
import time
from fastapi import Request

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = str(int((time.perf_counter() - start) * 1000))
    return response
```

Middlewares run in reverse order of registration (the last one registered runs first). The outermost middleware is usually tracing/logging; the innermost is authorization.

## Concurrency with asyncio.gather

When a handler needs to make multiple independent calls, running them in parallel with `asyncio.gather` significantly reduces latency:

```python
import asyncio

@app.get("/dashboard/{user_id}")
async def dashboard(user_id: int, db: AsyncSession = Depends(get_db)):
    user_task = get_user(user_id, db)
    orders_task = get_recent_orders(user_id, db)
    recs_task = get_recommendations(user_id)

    user, orders, recs = await asyncio.gather(user_task, orders_task, recs_task)
    return {"user": user, "orders": orders, "recommendations": recs}
```

If each call takes 300ms, the sequential synchronous handler takes 900ms. With `gather`, it takes ~300ms (the slowest one). It's free at the code level and the improvement is huge for handlers with multiple external dependencies.

Caveat: if the tasks share the same DB session and run concurrent queries, you can get errors due to concurrent use of a single connection. SQLAlchemy async sessions are neither thread-safe nor coroutine-safe. The solution is to open independent sessions or run the operations that touch the same DB sequentially.

## Typical errors in async

### Blocking code in an async handler

The most common mistake is calling synchronous libraries inside an `async def` handler. Typical example:

```python
@app.get("/users/{id}")
async def get_user(id: int):
    user = requests.get(f"...")  # BLOCKS the entire event loop
    return user.json()
```

This blocks the whole event loop during the HTTP call. Every other request is stuck. The solution is to use `httpx.AsyncClient` or, if the dependency has no async version:

```python
user = await asyncio.to_thread(requests.get, f"...")
```

This runs the blocking call in a thread pool, freeing the event loop.

### Improperly closed async generators

Async generators with `yield` require the context to be properly closed. If you use `async for` and break early, you must ensure resources are released. The `async with` pattern inside the generator helps guarantee that.

### Event loop conflicts in tests

When testing async handlers with pytest, you need to use `pytest-asyncio` (or `anyio`). Without configuring it, async tests fail silently or don't run.

```python
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

With `asyncio_mode = "auto"`, any `async def test_...` function is detected and run automatically.

## Testing async endpoints

FastAPI recommends `httpx.AsyncClient` combined with `AsyncClient(transport=ASGITransport(app))` or `TestClient` for tests.

```python
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
```

FastAPI's synchronous `TestClient` also works and is simpler, but it doesn't let you test the actual event loop execution.

## Trade-offs of using async

| Scenario | Sync | Async |
|----------|------|-------|
| APIs with heavy external I/O | Slow | Very fast |
| APIs with heavy CPU work | Fine | Harmful (blocks event loop) |
| Existing code written with sync libraries | Zero change | Requires rewrite or threadpool |
| Debugging stack traces | Straightforward | More complex (traces span coroutines) |
| Team without async experience | OK | Learning curve |
| Scaling to thousands of concurrent connections | Limited | High |

The cost of writing async well is non-trivial. For small projects with little external I/O, the sync framework is more than enough. For systems with many external calls, LLMs, streaming, async is transformative.

## When not to use FastAPI async

- Server-rendered sites with classic HTML templates and little external I/O: Django or Flask give better productivity.
- Teams without async experience: the learning curve is real, the bugs are subtle (deadlocks, forgotten coroutines, shared sessions).
- Projects where most code is CPU-bound: async doesn't help and can hurt if misused.

For modern APIs with LLM integrations, async databases, many external services, FastAPI async is almost always the right choice.

## References

- FastAPI docs: https://fastapi.tiangolo.com/
- Async tutorial: https://fastapi.tiangolo.com/async/
- Starlette (ASGI base): https://www.starlette.io/
- Official asyncio: https://docs.python.org/3/library/asyncio.html

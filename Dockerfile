# Stages (audit E18):
#   build    the venv, from uv.lock, with the compilers it needs
#   dev      the venv with the dev extra (pytest, ruff, mypy) and the code:
#            what compose builds, so the tests run in the container
#   runtime  the last, so the default: the venv and the code, no compilers
#            and no dev tools
# Both run as a non-root user. A production image: `docker build .`. The
# reranker's extra: `--build-arg EXTRAS="--extra rerank"`.

# ---------- build ----------
# A literal base image, which Dependabot can update.
FROM python:3.12-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /usr/local/bin/uv

WORKDIR /code
COPY pyproject.toml uv.lock ./

# Exactly the versions in uv.lock. The app runs from /code (uvicorn, alembic
# and `python -m` put it on sys.path), so the project itself is not installed.
ARG EXTRAS=""
RUN uv sync --frozen --no-install-project ${EXTRAS}
# The tokenizer's file, fetched now so a container never needs to.
RUN /opt/venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

FROM build AS build-dev
# A build argument ends with its stage: declared again, or the sync below
# would drop the extras the build stage installed.
ARG EXTRAS=""
RUN uv sync --frozen --no-install-project ${EXTRAS} --extra dev

# ---------- what both images share ----------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

# uid 1000: on a Linux host, compose's bind mounts usually belong to it, so
# the API can write eval/results there.
RUN useradd --create-home --uid 1000 app

COPY --from=build /opt/tiktoken /opt/tiktoken
WORKDIR /code
EXPOSE 8000

# The probe: no key, the database only. A degraded API answers 503, which
# fails it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- dev ----------
FROM base AS dev
COPY --from=build-dev /opt/venv /opt/venv
COPY --chown=app:app . .
USER app

# ---------- runtime ----------
FROM base AS runtime
COPY --from=build /opt/venv /opt/venv
COPY --chown=app:app . .
USER app

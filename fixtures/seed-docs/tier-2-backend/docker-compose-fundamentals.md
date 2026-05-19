# Docker Compose fundamentals

Docker Compose is a tool for defining and running multi-container applications with a single YAML file. It solves the problem of orchestrating several services (API, database, cache, worker, etc.) in a reproducible development environment, and it is the standard way to distribute modern backend projects.

## The compose.yml file

A Compose file describes services, networks, and volumes. The modern version (Compose V2) does not require the obsolete `version:` field from V1.

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://user:pass@db:5432/mydb
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

With that and `docker compose up`, Compose creates the network, brings up the DB, waits for the healthcheck to pass, and only then starts the API.

## Services

Each service defines how a container is built/run:

- `image:` — use an existing image from a registry.
- `build:` — build from a local Dockerfile. Can be a string with the context path or an object with options (`context`, `dockerfile`, `args`).
- `command:` — override the Dockerfile's CMD.
- `entrypoint:` — override the ENTRYPOINT.
- `environment:` — environment variables. Can be defined inline or with `env_file:`.
- `ports:` — port mapping `host:container`.
- `volumes:` — disk mounts.
- `depends_on:` — startup dependencies.
- `restart:` — restart policy (`no`, `on-failure`, `always`, `unless-stopped`).

## Networks

By default, Compose creates a network named `<project>_default` where all services can discover each other by name. The `api` service can reach `db` using the hostname `db` in its DATABASE_URL.

Defining explicit networks allows for isolation or connecting services to multiple networks:

```yaml
services:
  api:
    networks:
      - frontend
      - backend
  db:
    networks:
      - backend

networks:
  frontend:
  backend:
    internal: true  # no internet egress
```

## Volumes

Two main types:

### Named volumes

Managed by Docker, they persist between `docker compose down` and `up`. They're the choice for DB data, uploads, cache.

```yaml
volumes:
  pgdata:

services:
  db:
    volumes:
      - pgdata:/var/lib/postgresql/data
```

### Bind mounts

Mount a host directory inside the container. Useful for development with hot-reload: code on the host is reflected in the container without rebuilding.

```yaml
services:
  api:
    volumes:
      - ./app:/code/app
      - ./alembic:/code/alembic
```

Bind mounts are fragile across operating systems (Linux vs Windows permissions, performance on macOS with Docker Desktop). For production, prefer named volumes or images with code copied in.

### tmpfs

In-RAM filesystem, useful for temporary files that can be lost on shutdown.

```yaml
services:
  api:
    tmpfs:
      - /tmp
```

## Health checks

Defining healthchecks lets Compose know when a service is ready to receive traffic, and lets `depends_on` wait for that healthy state before starting dependents.

```yaml
services:
  db:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 10s

  api:
    depends_on:
      db:
        condition: service_healthy
```

Without a healthcheck, `depends_on` only waits for the container to start — not for the process inside to be ready. Postgres takes 2-5 seconds to accept connections after start. Without a healthcheck, the API tries to connect immediately and fails.

Valid conditions in `depends_on`:

- `service_started` (default): the container started.
- `service_healthy`: the healthcheck passed.
- `service_completed_successfully`: the service ran and exited with code 0 (useful for one-shot migrations).

## Environment variables and .env

The `.env` file in the same directory as `compose.yml` is read automatically. Its variables can be used as `${VAR}` in the YAML and are also available to services that include them in `environment`.

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-defaultuser}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?password required}
```

Syntax:

- `${VAR}` — use the variable.
- `${VAR:-default}` — use the default if not set or empty.
- `${VAR:?error message}` — fail with an error if not set.
- `${VAR:+value}` — use `value` only if the variable is set.

The `.env` file is never committed to git. You do commit `.env.example` with documented variables but without sensitive values.

## Profiles

Profiles let you define optional services that only start when explicitly specified.

```yaml
services:
  api:
    # always active
  pgadmin:
    image: dpage/pgadmin4
    profiles:
      - tools
```

```bash
docker compose up           # starts only api
docker compose --profile tools up  # starts api + pgadmin
```

Useful for separating development tools, specific workers, or alternative configurations.

## Override files

Compose automatically combines `compose.yml` with `compose.override.yml` if it exists. A common pattern: `compose.yml` contains the base config, and the override adds dev-only adjustments (bind mounts, exposed ports, commands with reload).

```yaml
# compose.yml
services:
  api:
    build: .
    command: uvicorn app.main:app --host 0.0.0.0

# compose.override.yml
services:
  api:
    volumes:
      - ./app:/code/app
    command: uvicorn app.main:app --host 0.0.0.0 --reload
```

In production you use `docker compose -f compose.yml -f compose.prod.yml up` to combine explicitly, without the dev override.

## Build with arguments and multi-stage

For complex builds, you can pass arguments and target specific stages of a multi-stage Dockerfile:

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
      args:
        PYTHON_VERSION: 3.12
```

Multi-stage lets you have a slim production image (no dev dependencies, no build tools) separate from a development image that includes pytest, ruff, etc.

## Useful commands

```bash
docker compose up                  # bring everything up in the foreground
docker compose up -d               # in the background
docker compose up --build          # rebuild images
docker compose down                # bring everything down (keeps volumes)
docker compose down -v             # bring everything down and delete volumes
docker compose logs -f api         # logs for one service with follow
docker compose exec api bash       # shell inside the container
docker compose ps                  # container status
docker compose restart api         # restart a service
docker compose config              # show the resolved config
```

## Best practices

### Split compose.yml by environment

Have `compose.yml` as the base and additional files `compose.dev.yml`, `compose.prod.yml`. The automatic override is used for development; production is specified with `-f`.

### Do not use `latest` in production

Always pin explicit image versions (`postgres:16.2`, not `postgres:latest`). `latest` changes without warning and breaks deploys.

### Volumes for persistent data

Any state you don't want to lose on `docker compose down` should live in a named volume. Particularly DB data, uploads, persisted logs.

### Limit resources

In production and for load testing, define limits:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 512M
```

### Consistent service names

Use short, descriptive names: `api`, `db`, `redis`, `worker`, `nginx`. Services are referenced by name from other services, so long names make the configuration unreadable.

### .dockerignore and build cache

Without `.dockerignore`, Docker copies the whole directory to the build context, including `.venv`, `node_modules`, `.git`. This breaks layer cache and makes builds extremely slow.

```
.git
.venv
__pycache__
*.pyc
node_modules
.env
```

## Frequent troubleshooting

### "Cannot connect to the Docker daemon"

Docker is not running. On Docker Desktop, verify it's active. On Linux, `sudo systemctl start docker`.

### Container dies in a loop with "exited (1)"

The main process failed at startup. `docker compose logs <service>` shows the error. Typical causes: missing environment variables, port already in use, dependency not ready.

### Changes are not reflected with hot reload

The bind mount may not be active, or the server inside the container is not detecting changes. Verify that `--reload` is in the command and that the bind mount points to the right directory.

### "Port is already allocated"

Another process (another container, another local service) is using that port on the host. `docker compose down` first, or change the port mapping.

### "Phantom" volumes

Named volumes persist until explicitly deleted. If the DB changed schema and the old volume holds incompatible data, `docker compose down -v` deletes everything.

## When not to use Compose

- **Production at scale**: Compose is designed for a single host. For clusters, use Kubernetes, ECS, or Docker Swarm.
- **Single-container applications**: if there are no auxiliary services, a `Dockerfile` + `docker run` is enough.
- **Complex CI/CD pipelines**: some tools (GitHub Actions, GitLab CI) have their own primitives for services that are more integrated.

For local development, demos, single-node staging, and open-source project setup, Compose remains the most pragmatic tool.

## References

- Docker Compose docs: https://docs.docker.com/compose/
- Compose file reference: https://docs.docker.com/compose/compose-file/
- Compose samples: https://github.com/docker/awesome-compose

# Local Development with Docker

This guide explains how the platform team runs the service stack on a developer
laptop. Everything below assumes Docker Desktop 4.30 or a compatible engine.

## Building the image

The application image is described by the `Dockerfile` in the repository root.
It uses a two stage build: the first stage compiles the wheels, the second stage
copies only the resulting virtual environment into a slim runtime image. Rebuild
it after changing dependencies:

```bash
docker build -t platform/api:dev .
```

Layer caching is aggressive. If a dependency upgrade does not seem to take
effect, rebuild with `--no-cache` before opening a bug report.

## Running the stack

The full stack (API, worker, PostgreSQL, Redis) is described in
`docker-compose.yml`. Start everything in the background with:

```bash
docker compose up -d
```

Older installations that still ship the standalone binary should use
`docker-compose up -d` instead; both forms are supported by our Makefile
targets. Stop the stack with `docker compose down`. Adding `-v` also removes
the named volumes, which is the fastest way to reset a corrupted local
database.

## Talking to a remote engine

By default the CLI talks to the local engine socket. To target a remote builder
export the DOCKER_HOST environment variable before running any command:

```bash
export DOCKER_HOST=ssh://builder.internal:22
```

DOCKER_HOST accepts `unix://`, `tcp://` and `ssh://` URLs. When it is set, bind
mounts refer to paths on the remote machine, not on your laptop, which is the
most common source of confusing "file not found" errors.

## Resource limits

Containers inherit the memory ceiling of the virtual machine that hosts the
engine. When the API container is terminated with exit code 137 the kernel
out-of-memory killer stopped it: the process asked for more memory than the
container was allowed to use. Raise the limit in the `deploy.resources` section
of the compose file, or give the whole virtual machine more RAM in Docker
Desktop settings. Long running test suites are the usual culprit because they
keep fixtures alive between cases.

## Logs and troubleshooting

Follow the logs of a single service with `docker compose logs -f api`. A
container that restarts in a loop usually fails during configuration parsing;
the first fifty lines of its log almost always contain the reason. Use
`docker compose exec api sh` to open a shell inside a running container and
inspect the effective environment.

## Environment files

Compose reads variables from a `.env` file next to `docker-compose.yml`. The
file is not committed; a template with every supported key and a short comment
lives in `.env.example`. Values defined in the shell win over values from the
file, which is what makes a one-off override such as
`LOG_LEVEL=debug docker compose up api` work without editing anything.

Secrets are never stored in the compose file. The API container receives its
credential from the surrounding shell, so the variable has to be exported in
the session that starts the stack. A container that starts but immediately
answers `401` to every request is almost always missing that export.

## Networking between containers

Compose creates one user defined bridge network per project. Inside that
network services address each other by service name, so the API reaches the
database at `db:5432` and Redis at `redis:6379`. `localhost` inside a container
refers to that container, not to your laptop - this is the single most common
mistake when a connection string is copied from a local run into the compose
file.

Only ports listed under `ports:` are published to the host. Everything else
stays reachable from sibling containers only, which is the behaviour you want
for the database in a development stack.

## Volumes and file permissions

Source code is bind mounted into the API container so that the reloader picks
up edits without a rebuild. Dependencies, on the other hand, live in a named
volume; that is why a fresh checkout needs one `docker compose build` before
the first `up`. On Linux the container user and the host user may have
different ids, which shows up as files created inside the container being owned
by root on the host. Set `user: "${UID}:${GID}"` for the affected service to
avoid it.

Named volumes survive `docker compose down` and are removed by
`docker compose down -v`. Anonymous volumes accumulate silently; reclaim the
space with `docker volume prune` when the disk fills up.

## Health checks and start-up order

`depends_on` only controls start order, not readiness. Without a health check
the API starts while PostgreSQL is still initialising and crashes on its first
query. Declare a health check on the database service and make the API depend
on it with `condition: service_healthy`; the retry loop in the entrypoint then
becomes a safety net rather than the primary mechanism.

## Cleaning up disk space

Build caches, dangling images and stopped containers grow without bound on a
machine used for daily development. `docker system df` shows what is using the
space and `docker system prune -a` reclaims it. Be aware that the second
command also deletes images that are not referenced by a running container, so
the next build downloads the base layers again.

## Continuous integration

CI builds the same Dockerfile with BuildKit and pushes the result to the
internal registry, tagged with the commit sha. The pipeline never runs
`docker compose up`; it starts the services it needs as separate containers on
a shared network so that each job can be scheduled independently. If a test
passes locally but fails in CI, compare the effective environment first - the
image is identical, the variables usually are not.

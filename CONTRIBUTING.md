# Contributing

Thanks for taking a look. This is a personal project, but issues and pull
requests are welcome.

## Getting set up

```bash
uv sync                       # install dependencies (Python 3.13)
cp .env.example .env          # fill in POSTGRES_* at minimum
docker compose up -d          # Postgres + Redis
uv run alembic upgrade head   # create the schema
```

Then, in two terminals:

```bash
uv run fastapi dev app/main.py   # API on :8000
uv run python -m app.workers.run # worker — jobs stay 'queued' without it
```

`ffmpeg` must be on your PATH: faster-whisper shells out to it, and the
rate limiter uses `ffprobe` to read upload durations.

## Before you open a PR

```bash
uv run ruff check .    # lint — CI runs this
uv run pytest          # unit tests — CI runs this
```

If your change touches the schema, the Dockerfile, or the queue wiring,
also run the end-to-end check, which builds the real image and transcribes
real audio:

```bash
bash scripts/smoke_test.sh
```

## Conventions

- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `chore:`, `docs:`, `test:`. One logical change each.
- **Schema changes** go through Alembic. Autogenerate the migration, then
  read it — it does not detect `CREATE EXTENSION`, and it emits pgvector
  column types without importing the module:
  ```bash
  uv run alembic revision --autogenerate -m "what changed"
  ```
- **Layering**: `app/core/` holds business logic and must not import
  FastAPI. `app/api/` is the HTTP layer and stays thin. This is what lets
  chunking, embedding and summarization be tested without a server.
- **Sessions** are managed explicitly — open `SessionLocal()`, close it in
  a `finally`. There is no request-scoped session; follow the surrounding
  pattern when adding routes.
- **Docstrings** explain *why*, not *what*. The signature already says
  what.

## Things worth knowing

- The API and the worker are separate processes and each loads its own
  copy of the models. That is expected, not a bug.
- On macOS the worker uses RQ's non-forking `SimpleWorker`: the default
  forking worker aborts once CTranslate2 has been loaded, because the
  libraries it initialises are not safe to use after `fork()`.
- Tests must run with no Postgres, no Redis and no model downloads.
  Anything needing real services belongs in `scripts/smoke_test.sh`.

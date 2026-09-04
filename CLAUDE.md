# CLAUDE.md

Working notes for coding agents (and anyone editing this repo) about the
invariants that are easy to break without noticing. It is not an
introduction: [README.md](README.md) explains what the project is,
[CONTRIBUTING.md](CONTRIBUTING.md) has the conventions, and
[docs/design.md](docs/design.md), [docs/performance.md](docs/performance.md)
and [docs/deploy.md](docs/deploy.md) hold the reference material. What
follows is only what is not written down in those.

## Commands

Dependencies are managed with `uv` (see `uv.lock`; `.python-version` pins
3.13). Prefix Python invocations with `uv run` to use the project venv.

```bash
uv sync                                   # install/refresh dependencies
docker compose up -d --build              # full stack: api, worker, postgres, redis
docker compose up -d postgres redis       # just the datastores, for host-run dev
uv run alembic upgrade head               # apply migrations
uv run fastapi dev app/main.py            # the API (dev, hot reload) on :8000
uv run python -m app.workers.run          # the queue worker (REQUIRED for jobs to progress)
uv run python scripts/create_api_key.py <name> [daily_minute_quota]
bash scripts/dump_seed.sh [--check]       # regenerate data/seed/nasa_corpus.sql

uv run ruff check .                       # lint (CI runs this)
uv run pytest                             # unit tests, no services needed (CI runs this)
bash scripts/smoke_test.sh <audio>        # end-to-end: build, transcribe, search, summarize
bash scripts/benchmark.sh <audio>         # P50/P95 latency table, needs jq
bash scripts/search_eval.sh               # retrieval quality on a query set
```

The API and the worker are **two separate processes**: an upload sits in
`queued` forever unless the worker (`app/workers/run.py`) subscribed to the
`transcribe` queue is running. Under compose the API listens on **8080**; run
directly on the host it listens on **8000**. The entrypoint is `app/main.py`.

## Environment

Config is loaded via `pydantic-settings` from `.env` (see `.env.example`).
Either set `DATABASE_URL` (a full URL; a bare `postgres://` or `postgresql://`
prefix is rewritten to `postgresql+psycopg://`), **or** all of
`POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB`. A model validator
fails at startup if neither is complete. `GROQ_API_KEY` is optional:
transcription and search work without it, and `/summarize` returns 503.

## Where the request flow lives

1. `app/api/transcribe.py`: `POST /transcribe` validates the extension (by
   filename, not content), spools the upload into `settings.temp_audio_dir`,
   reads its duration with `ffprobe`, enforces the caller's rolling-24h
   audio-minute quota, writes a `Job` row, and enqueues `transcribe_job`.
2. `app/workers/transcribe_worker.py`: drives the `Job` through `queued`,
   `processing`, `done` or `failed`, runs the model, persists a `Transcript`,
   then chunks, embeds and stores `Chunk` rows **in the same transaction**
   that marks the job `done`, and deletes the spooled file in a `finally`.
3. `app/api/jobs.py`: status, result, and the caller's recent jobs. All three
   are scoped to the key that submitted the job.
4. `app/api/search.py`: embeds the query **in the API process** and ranks
   chunks by pgvector cosine distance, ordering and limiting inside Postgres.
5. `app/api/summarize.py`: cache-first, so only a miss calls the LLM.

## Invariants that are easy to break

- **Model loading is expensive and cached per-process.** `ASRModel`
  (`app/core/asr.py`) and the sentence-transformers model
  (`app/core/embeddings.py`) are lazy module-level singletons. Never
  instantiate them per request.
- **The API never imports the worker, and `app/core/__init__.py` stays
  empty.** `app/api/transcribe.py` enqueues by string
  (`TRANSCRIBE_JOB = "app.workers.transcribe_worker.transcribe_job"`), and the
  package `__init__` re-exports nothing. Both look like things to tidy up, and
  either one undone pulls faster-whisper and ctranslate2 back into the API
  process: importing anything under `app.core` used to cost 2.6s and load
  torch. `tests/test_app.py` resolves the string and asserts it is callable,
  which is the import-time check that string reference gives up.
- **Job ownership is checked before anything else**, including the
  `status != 'done'` check, and a denial is **404, never 403**, with a body
  byte-for-byte identical to a job that does not exist.
  `tests/test_job_ownership.py` pins both. The reasoning is in
  [docs/design.md](docs/design.md); the thing to know here is that
  `_owned_job()` in `app/api/jobs.py` is the only correct way to load a job
  in a handler.
- **Auth is per-route, not middleware.** Protection comes from
  `Depends(get_api_key)` on each handler, so a new route is public until you
  add it. There is no allowlist to update and no blanket default to rely on.
  `/`, `/health` and `/metrics` are open precisely because they declare no
  such dependency. `tests/test_app.py` keeps a table of protected endpoints
  that must 401 without a key; add new routes to it.
- **Nothing deletes an `ApiKey` row on your behalf.** `Job.api_key_hash` is a
  plain column with no foreign key, so a deleted key leaves its jobs owned by
  a hash that matches nobody: they answer 404 to every caller and vanish from
  `GET /jobs`, while their transcripts and chunks stay reachable through
  `/search`, which does not filter by key.
- **Timeouts scale with audio duration.** `job_timeout_for()` gives 5x the
  recording length with a floor of 10 minutes, and the stale-job reaper
  mirrors that budget per job. A fixed timeout kills long recordings mid-job.
  That bug shipped once; do not reintroduce it.
- **A killed work-horse is handled by a worker-level hook.**
  `Worker(work_horse_killed_handler=...)` in `app/workers/run.py`, *not* the
  job's `on_failure` callback: RQ runs failure callbacks inside the
  work-horse, which is the process that died. The handler must never raise,
  because RQ passes `ret_val=None` on some paths.
- **Long transcripts are thinned before summarizing**, not truncated
  (`thin_segments`). Keeping the first N characters would summarize the
  opening and report it as the whole thing. The response carries
  `transcript_thinned`.
- **LLM timestamps are defended twice**: `[Ns]` markers injected into the
  prompt so the model quotes real times, and `_sanitise_sections` clamping
  whatever comes back to the audio duration.
- **Metrics come from Postgres, not counters.** The worker produces the job
  numbers in a different process, so in-process counters on the API would
  read zero. `app/core/metrics.py` queries at scrape time instead.
- **Blocking work runs in a threadpool.** ASR, embedding, `ffprobe` and Groq
  are all blocking, and every async handler that calls them wraps them.
- **The browser console is one static HTML file.** `app/web/index.html`,
  returned by the `/` route in `app/main.py` as a `FileResponse`. Keep it
  self-contained: inline the CSS and JS, and add UI features by editing that
  one file. It calls the same origin that served it, so there is no host to
  configure, and its fetch helper attaches `X-API-Key` from a field the user
  fills in, so nothing is persisted server-side.
- **The demo seed is generated, not hand-written.**
  `data/seed/nasa_corpus.sql` is `scripts/dump_seed.sh` output, and both of
  its filters matter. Only jobs matching `audio_filename LIKE 'nasa-%'` are
  included, because the dev database accumulates rows from every test and a
  plain `pg_dump -t jobs -t transcripts -t chunks` would publish them,
  silently breaking the claim that `scripts/search_eval.md` Run 2 describes
  the deployed corpus. And `api_key_hash` is rewritten to the literal `seed`,
  because the real column is the SHA-256 of a working key and must not reach
  a public repo. Rows are ordered so the output is reproducible, and
  `--check` asserts that against the committed file.
- **That `seed` literal is now load-bearing**, and no longer only a
  precaution. `GET /transcripts` and the `list_transcripts` MCP tool show
  exactly the jobs carrying it (`CURATED_CORPUS_MARKER` in
  `app/core/retrieval.py`), which is how a stranger's upload stays out of a
  listing while remaining findable by `/search`. Two consequences: changing
  the literal in `scripts/dump_seed.sh` silently empties the listing, and a
  development database that transcribed the NASA episodes itself lists
  nothing, because its rows carry a real key hash. Only a database restored
  from the dump has a curated corpus.

## The test trap that only fails in CI

**Tests that build a `Settings` must neutralise the ambient environment.**
`Settings(_env_file=None, ...)` stops the developer's `.env` from being read
but *not* real environment variables, and pydantic-settings ranks those
**above** the keyword arguments the test passes. CI exports `DATABASE_URL` so
that importing `app.config` succeeds, which is enough to break assertions
silently: the URL assembled from `POSTGRES_*` parts comes back as the CI URL,
and a deliberately incomplete configuration looks valid and never raises.
`tests/test_config.py` clears the six database names in an autouse fixture;
extend that list when adding a setting. A config test that passes locally and
fails only in CI is almost always this.

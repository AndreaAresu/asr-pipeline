# Deploying to Fly.io

Two Fly apps sharing one Dockerfile — `asr-api-andrea` and
`asr-worker-andrea` — plus managed Postgres and Upstash Redis. They are
separate apps so they scale separately: the API is small and always-on,
the worker is the expensive half, and a worker crash does not take the
API down with it.

## Read this first: the audio spool blocks the two-app split

`POST /transcribe` writes the upload to `TEMP_AUDIO_DIR` and passes the
**path** to the worker through the queue. That is fine when both processes
see the same filesystem — one machine, or compose with a shared volume.

On Fly the API and the worker are separate machines with separate disks.
The worker will receive a path that does not exist on it, and every job
will fail with `FileNotFoundError`. **Fly volumes cannot be shared between
apps**, so a bigger volume does not fix this.

Three ways out, in the order I would try them:

1. **Object storage (recommended).** The API uploads to S3-compatible
   storage — [Tigris](https://fly.io/docs/tigris/) is built into Fly and
   has a free tier — and enqueues the object key. The worker downloads,
   transcribes, deletes. Costs one dependency (`boto3`) and one set of
   credentials, and is how this is normally solved.
2. **Send the bytes through the queue.** Enqueue the audio itself rather
   than a path. No new services, but Redis payloads are capped in
   practice: Upstash's free tier is 256MB total and a 30-minute 16kHz WAV
   is ~57MB. Viable only for short clips.
3. **One app, both processes.** Run the API and worker as two processes on
   the same machine so they share a disk. Simplest to deploy, but it
   throws away independent scaling — which is the reason for splitting
   them in the first place.

Until one of these is done, deploy is single-app only. Everything below is
correct regardless of which you pick.

## 1. Managed Postgres and Redis

```bash
flyctl postgres create --name asr-pg-andrea --region fra \
  --vm-size shared-cpu-1x --volume-size 3 --initial-cluster-size 1
```

Save the connection string — it is shown **once**.

For Redis, create a database in an EU region at
[console.upstash.com](https://console.upstash.com) and copy the `rediss://`
(TLS) URL.

## 2. Create the apps

```bash
flyctl apps create asr-api-andrea
flyctl apps create asr-worker-andrea
```

The configs are already in the repo, so `flyctl launch` is not needed —
it would only overwrite them.

## 3. Set secrets

Never put these in `fly.toml`; secrets are injected as environment
variables at runtime.

Fly hands out `postgres://…`. The app rewrites that prefix to
`postgresql+psycopg://` on its own (`app/config.py`), so either form
works — but being explicit costs nothing:

```bash
flyctl secrets set --app asr-api-andrea \
  DATABASE_URL="postgresql+psycopg://…flycast:5432/asr" \
  REDIS_URL="rediss://…" \
  GROQ_API_KEY="…"

flyctl secrets set --app asr-worker-andrea \
  DATABASE_URL="postgresql+psycopg://…flycast:5432/asr" \
  REDIS_URL="rediss://…"
```

The worker needs no Groq key — it never summarizes.

## 4. Volume for the model cache

Without it every worker restart re-downloads Whisper (~400MB) and MiniLM
(~22MB), which on a small VM is minutes of cold start.

```bash
flyctl volumes create whisper_cache --app asr-worker-andrea \
  --size 3 --region fra
```

`fly.worker.toml` already mounts it at `/cache/huggingface`, which is
where `HF_HOME` points.

## 5. Deploy

```bash
flyctl deploy -c fly.api.toml       # runs `alembic upgrade head` first
flyctl deploy -c fly.worker.toml
```

The API's `release_command` applies migrations on a temporary machine
with the new image *before* any instance is promoted. A failing migration
aborts the deploy instead of half-breaking production. Only the API app
does this — two apps racing on the same DDL is how a schema gets
corrupted.

## 6. Verify

```bash
flyctl logs --app asr-api-andrea
flyctl logs --app asr-worker-andrea

URL=https://asr-api-andrea.fly.dev
curl $URL/health

flyctl ssh console --app asr-api-andrea -C "python -m scripts.create_api_key prod"

curl -X POST $URL/transcribe -H "X-API-Key: $PROD_KEY" \
  -F "audio=@data/samples/sample.wav"
```

Then poll `/jobs/{id}` until `done`. That is the deploy actually working —
`/health` returning 200 only proves the API booted.

## Troubleshooting

**Jobs stay `queued`.** The worker is not consuming. Check
`flyctl logs --app asr-worker-andrea`, and confirm `REDIS_URL` is
identical in both apps — a job enqueued to one Redis is invisible to a
worker watching another.

**Jobs fail with `FileNotFoundError`.** The spool problem above. The
worker is looking for a file that only exists on the API machine.

**Build runs out of memory.** Fly's remote builder is small. Build
locally and push the image instead: `flyctl deploy --local-only`.

**Image is too large and the deploy times out.** The image should be
~2.5GB. If it is ~11GB, the CUDA build of torch got in: `pyproject.toml`
pins the CPU-only wheel index for Linux, so check that `uv.lock` is in
sync (`uv lock`) and that the build is not using a stale cache. Model
weights must never be baked in — they belong on the volume.

**Cold start on the first request.** `min_machines_running = 0` lets the
API suspend when idle, which keeps a demo inside the free allowance.
Raise it to 1 if the latency is not acceptable.

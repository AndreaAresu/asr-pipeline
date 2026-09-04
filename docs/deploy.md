# Deploying

The deployment is **one VPS running the `compose.yaml` in this repo**, with
Caddy on the host terminating TLS in front of it.

Live at:

| | |
|---|---|
| API + browser console | https://api.159-195-250-205.sslip.io |
| Streamlit demo | https://demo.159-195-250-205.sslip.io |

## Why one host, and not API and worker on separate machines

Not a shortcut — the split does not work, and this is the reason:

`POST /transcribe` writes the upload to `TEMP_AUDIO_DIR` and passes the
**path** to the worker through the queue. That is fine when both processes
see the same filesystem — one machine, or compose with a shared volume. Put
them on separate machines and the worker receives a path that does not exist
on it, and every job fails with `FileNotFoundError`. A bigger disk does not
fix it; the two disks are different disks.

Three ways out, if the split ever becomes worth it:

1. **Object storage (the real fix).** The API uploads to S3-compatible
   storage and enqueues the object key. The worker downloads, transcribes,
   deletes. Costs one dependency (`boto3`) and one set of credentials, and
   is how this is normally solved. **Not implemented.**
2. **Send the bytes through the queue.** Enqueue the audio rather than a
   path. No new services, but Redis payloads are capped in practice and a
   30-minute 16kHz WAV is ~57MB. Viable only for short clips.
3. **Both processes on one machine.** Which is what this deployment does.

One host makes the problem not exist, at the cost of independent scaling —
a trade that is obviously right at one worker and one queue.

## The machine

netcup VPS Lite 2: 4 vCore, 8 GB, 157 GB disk, Debian 13, €8,11/month.
2 GB of RAM for the worker is the floor (see the memory note in the README);
8 GB leaves room for Postgres and the model cache alongside it.

Access is by key only, no passwords. `asr@` owns `/opt/asr-pipeline`; `root@`
is for Caddy and systemd.

## Layout on the server

| What | Where |
|---|---|
| Repo | `/opt/asr-pipeline`, branch `interview-ready` |
| Production config | `/opt/asr-pipeline/.env` — not in git, never committed |
| Demo API key | `/opt/asr-pipeline/.demo-key`, quota 15 audio-minutes / 24h |
| Caddy | `/etc/caddy/Caddyfile`, a copy of `deploy/Caddyfile` |
| Seed corpus | `data/seed/nasa_corpus.sql`, arrives by `git pull` |

`compose.yaml` binds every port to `127.0.0.1`, so Postgres, Redis, the API
(8080) and the demo (8501) are unreachable from outside. Caddy is the only
way in, which is also why `/metrics` can stay open to the machine itself and
answer 404 through the proxy.

## First-time setup

1. Docker, Docker Compose and Caddy installed; a non-root user owning the
   checkout.
2. `git clone` into `/opt/asr-pipeline`, checkout the branch.
3. Write `.env` from `.env.example`. On a public deployment the values that
   differ from local:

   ```
   WHISPER_MODEL=base.en      # small.en is ~2x slower; a visitor is waiting
   MAX_UPLOAD_MB=10           # local default is 500
   MAX_AUDIO_SECONDS=90       # local default is 0, meaning no limit
   DEMO_API_KEY=<the demo key>
   ```

   Both caps are read by **two** services: the API enforces them and the
   demo states them in its UI before a visitor picks a file. One source, so
   the page cannot advertise a limit the API does not apply.
4. `docker compose up -d --build`.
5. Create the demo key and keep it:
   `docker compose exec api python scripts/create_api_key.py demo 15 | tee .demo-key`,
   then put it in `.env` as `DEMO_API_KEY` and `docker compose up -d demo`.
6. Restore the corpus, after the API has applied its migrations:
   `docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" < data/seed/nasa_corpus.sql`
7. `sudo cp deploy/Caddyfile /etc/caddy/Caddyfile && sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy`.
   Both hostnames are `<ip-with-dashes>.sslip.io`, which resolves to the
   address encoded in it — a real public DNS name with no domain to buy, and
   one Let's Encrypt will issue for. Certificates are fetched on first
   request and renewed by Caddy.

## Updating a running deployment

```bash
ssh asr@<host>
cd /opt/asr-pipeline
git pull --ff-only
docker compose up -d --build
docker compose ps                 # every service healthy?
```

If `deploy/Caddyfile` changed, repeat step 7 as root.

## Verify — in this order

`/health` returning 200 only proves the API booted. The order matters
because each step depends on the one before it:

```bash
curl https://api.<host>/health              # 200
curl -o /dev/null -w '%{http_code}\n' \
     https://api.<host>/metrics             # 404 — blocked at the proxy
curl https://demo.<host>/_stcore/health     # 200
```

Then, through the browser, on the demo:

1. Upload a short clip and watch it reach **done**. That is the queue, the
   worker, the model and the database all working; nothing else proves it.
2. Search the corpus for something it covers, and something it does not —
   the second should say so rather than return three irrelevant passages.
3. Summarize an episode twice: the first call takes seconds, the second is
   a cache hit and returns in milliseconds.
4. Upload something over 90 seconds. It must be refused with
   *"audio is Ns long; the limit is 90s"*, visible in the page.

## Troubleshooting

**Jobs stay `queued`.** The worker is not consuming. `docker compose logs
worker`, and check `REDIS_URL` — a job enqueued to one Redis is invisible to
a worker watching another.

**`docker compose exec -T` swallows the rest of your script.** It consumes
stdin, so inside a script piped over `ssh` it eats everything after it and
the output truncates with no error. Redirect its stdin explicitly: `<
/dev/null` when it needs no input, or `< file.sql` when it does. Do not
write `<<'SQL' ... SQL < /dev/null` — the later redirect wins and psql reads
nothing.

**Summarization returns 500 with `model_not_found`.** Groq retires models
without notice; `llama-3.3-70b-versatile` was the default here until it
started answering 404. Check what is servable with
`curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $KEY"`
and set `SUMMARIZE_MODEL`.

**The demo says `ASR_API_KEY is not configured`.** `DEMO_API_KEY` is missing
from `.env`, or the `demo` service was started before it was added. compose
passes it through; there is no copy inside the image.

**Image size.** Measured at 5.05 GB, dominated by one 3.27 GB dependency
layer: a 1.7 GB venv plus 1.4 GB of `uv` download cache baked in beside it.
If it is ~11 GB instead, the CUDA build of torch got in — `pyproject.toml`
pins the CPU-only wheel index for Linux, so check `uv.lock` is in sync and
the build is not using a stale cache. Model weights must never be baked in;
they live on the `hf_cache` volume.

**Everything on the machine is fine and the site is not.** Caddy runs on the
host, not in compose, so `docker compose down` cannot take TLS with it —
but it also means `systemctl status caddy` is a separate thing to check.

## Appendix: the Fly configs

`fly.api.toml` and `fly.worker.toml` are still in the repo. They describe
the two-app deployment this project started with, which the spool problem
above makes unworkable without object storage. They are kept as a record of
that design, not as an alternative you can run today.

# Design notes

The API reference, the choices behind the stack, and the full list of things
this deliberately does not do.

## API

All endpoints except `/`, `/health` and `/metrics` require an `X-API-Key`
header. Interactive docs are at `/docs`.

| Endpoint | Request | Response |
|---|---|---|
| `POST /transcribe` | multipart `audio` file | `202` · `{job_id, status}` |
| `GET /jobs` | — | recent jobs for this key, newest first |
| `GET /jobs/{id}` | — | `{id, status, error_message, duration}` |
| `GET /jobs/{id}/result` | — | `{transcript_id, full_text, language, segments}` |
| `POST /search` | `{query, top_k, transcript_id?}` | `{query, hits: [{transcript_id, start_sec, end_sec, text, score}]}` |
| `POST /summarize/{transcript_id}` | — | `{transcript_id, cached, model, sections, meta}` |
| `GET /health` | — | `{status: "ok"}` |
| `GET /metrics` | — | Prometheus text format |

`status` moves `queued → processing → done | failed`.

Every response carries an `X-Request-Id`, and the same id appears in every
log line the request produced **including the worker's**, so one id
reconstructs the whole journey across two processes.

### What ownership does and does not cover

A job is readable only by the key that submitted it. Reading someone else's
job returns **404, not 403** — with a body identical to a job that does not
exist, so the endpoint cannot be used to probe which ids are real. The check
runs before the `status` check too, or a foreign job would leak its status
through a `400`.

`/search` and `/summarize` check no such thing: any valid key can search the
whole index and summarize any `transcript_id`. That is deliberate — it is
what lets a visitor query a demo corpus their own key owns none of — but it
is a property to know before pointing a second tenant at this.

## Stack and trade-offs

| Choice | Instead of | Why |
|---|---|---|
| **faster-whisper** | openai-whisper, whisper.cpp | ~4x faster on CPU via CTranslate2, clean Python API, native word timestamps |
| **pgvector** | Qdrant, Pinecone, Weaviate | One datastore instead of two. Job state and vectors live in the same database and the same transaction, so they cannot disagree |
| **all-MiniLM-L6-v2** | all-mpnet-base-v2 | 384 dims, 22 MB, fast enough on CPU. Adequate is not an opinion here: 15/15 correct episode routing, graded by hand in [`../scripts/search_eval.md`](../scripts/search_eval.md) |
| **RQ** | Celery, Arq | The job model here is "run one function, retry never". Celery's broker abstractions and routing buy nothing at this size |
| **Groq, model from config** | GPT-4o-mini | Free tier good enough for a demo, sub-second latency. The model is `SUMMARIZE_MODEL`, not a constant, because hosted providers retire models: this was built against Llama 3.3 70B, which Groq withdrew mid-project. It runs `openai/gpt-oss-120b` today |
| **Time-based chunking** | fixed character counts | Audio has a time axis; a hit is only useful if it says *when*. ~45 s windows with 10 s overlap so a thought is not cut in half |
| **Quota in audio minutes** | requests per hour | A 2-hour upload costs 240x what a 30-second one does. Counting requests would price them identically |
| **API keys** | JWT / OAuth | There are no user sessions. A hashed key is the right size of solution |
| **One image, two processes** | separate API and worker images | They share all the code and nearly all the dependencies; the command picks the process. Two images would double build time and let the halves drift |

## Limitations

Known and deliberate.

- **Rate limiting is post-upload.** The quota check needs the file's
  duration, read with `ffprobe` after the bytes have landed. A caller over
  quota is rejected, but only after paying the upload. The size cap is the
  exception and runs *while* the bytes arrive — buffering an arbitrary body
  first is how you end up holding it in memory.
- **The public demo is capped hard, in config rather than code.**
  `MAX_UPLOAD_MB=10` and `MAX_AUDIO_SECONDS=90` on the deployment. Both
  default permissive (500 MB, no duration limit) so indexing a long
  recording locally still works, and `0` disables either check. A rejected
  upload gets `413` naming both the measured value and the limit.
- **The demo runs on one shared API key.** Everyone using the public link
  submits under it, against one daily audio-minute quota, so one visitor can
  exhaust it for the rest until the 24-hour window rolls.
- **The audio spool is a shared filesystem.** The API writes the upload where
  the worker reads it. That works under compose via a shared volume; across
  separate hosts it needs object storage, which is not implemented. It is
  the reason the deployment is a single host — see [`deploy.md`](deploy.md).
- **English only.** `small.en` indexed the corpus, `base.en` runs the
  deployment. A multilingual model means changing `WHISPER_MODEL` and
  re-testing chunk quality; nothing else assumes English.
- **CPU only.** No GPU path. 0.47x real time with `small.en` on Apple
  Silicon, 0.4–0.7x with `base.en` on the VPS.
- **No streaming.** Transcription is batch; there is no partial-result API.
- **No diarization.** No speaker labels.
- **Sequential vector scan.** Exact and fast to ~10k chunks. An IVFFlat index
  needs data present at creation to train its clusters, so it is a
  post-backfill step, noted in `app/db/models.py`.
- **Groq free tier rate-limits** at roughly 30 requests/minute. Fine for a
  demo, not for load testing.
- **Long transcripts are thinned before summarizing**, not truncated. Past
  ~10k tokens the transcript sent to the LLM keeps every Nth segment,
  because a multi-hour recording exceeds the free tier's token budget
  outright. Coverage of the full running time is preserved and the response
  reports `transcript_thinned`, but the reading is coarser: a 1-hour episode
  keeps 692 of 1,384 segments.
- **A killed worker is only detected where there is a work-horse.** On Linux
  an OOM-killed job is marked `failed` within seconds. Under macOS's
  non-forking `SimpleWorker` the job dies with the process, and the startup
  reaper is the only backstop.

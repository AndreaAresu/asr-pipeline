# Performance

All numbers from `scripts/benchmark.sh` against the full compose stack.
Numbers without a machine attached are noise, so each table names one.

## The deployed VPS

netcup VPS Lite 2, 4 vCore, 8 GB, CPU only, `base.en`, the smaller model
the public demo runs, because a visitor is waiting.

| Operation | P50 | P95 | Samples |
|---|---|---|---|
| `POST /transcribe` to `done`, 30s audio | 19.7 s | 19.8 s | 3 |
| `POST /transcribe` to `done`, 85s audio | 36.2 s | 36.3 s | 3 |
| `POST /search`, top_k=5 | 62 ms | 296 ms | 9 |
| `POST /summarize`, cache miss | 1.6 s | n/a | 1 |
| `POST /summarize`, cache hit | 37 ms | 38 ms | 3 |

Two numbers that are easy to guess wrong:

- **Cold start costs 4.5 s, not 30.** The first job after a worker restart
  took 24.1 s against 19.6 s warm. Model weights live on the `hf_cache`
  volume and are not re-downloaded.
- **Peak worker memory was 1.30 GB**, 410 MB at rest.

A visitor uploading at the 90-second cap waits about **37 s warm, 42 s
cold**. That wait, not spare CPU, is what the cap is set against.

## Apple Silicon

Darwin arm64, Docker Desktop, CPU only, `small.en`, the larger model, used
to index the demo corpus.

| Operation | P50 | P95 | Samples |
|---|---|---|---|
| `POST /transcribe` to `done`, 30s audio | 16.1 s | 30.8 s | 3 |
| `POST /search`, top_k=5 | 58 ms | 236 ms | 9 |

Measured separately on one **60-minute** recording, the case that exercises
every limit at once (a podcast episode used for load characterisation only;
it is not part of the indexed corpus):

| | |
|---|---|
| Transcription, end to end | **28 min** (~0.47x real time) |
| Peak worker memory | **1.42 GB** |
| Output | 10,789 words, 1,384 segments, 102 chunks |
| Chunk coverage | full 3,600 s, mean chunk 46.4 s |
| `POST /summarize`, cache miss | 4.1 s, 6,659 input tokens |
| `POST /summarize`, cache hit | **62 ms** |

## Memory: the number that decides your VM size

A 1-hour recording needs **~1.5 GB** in the worker, and Whisper allocates it
in the first minute. In a Docker VM capped at 3.8 GB and shared with
Postgres, the forking worker was OOM-killed 78 seconds in, before
transcribing anything. Size the worker at **2 GB minimum**; the API needs a
fraction of that.

## Reading these honestly

Transcription runs at roughly **0.4-0.7x real time** on the deployed CPU
with `base.en`: a clip costs less wall time than its own duration, but fixed
overheads dominate short files, which is why 30 s costs 19.7 s and 85 s only
36.2 s. On Apple Silicon with `small.en` it is 0.47x.

The spread between P50 and P95 is queue wait, not model speed: with one
worker, a job that arrives while another is running waits for it.

Search is fast because the corpus is small and the scan is sequential. That
number grows with the index until an IVFFlat index is worth building.

## Re-run it

```bash
ASR_API_KEY=<key> bash scripts/benchmark.sh /path/to/your-recording.mp3
```

Needs `jq` on the machine you run it from, which is not in the image.

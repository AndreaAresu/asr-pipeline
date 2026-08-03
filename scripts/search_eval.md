# Retrieval quality notes

Running log of `POST /search` sanity checks. The point is to catch bad
retrieval early — a broken normalisation or a badly sized chunk window is
cheap to fix now and expensive to discover during a demo.

## How to run

```bash
docker compose up -d
uv run uvicorn app.main:app --port 8000     # terminal 1
uv run python -m app.workers.run            # terminal 2

# index something, then:
ASR_API_KEY=<key> bash scripts/search_eval.sh scripts/eval_queries.txt 3
```

## What to look for

The absolute score matters less than the **separation between query
classes**. With normalised embeddings and cosine distance:

| Query class        | Healthy score band | Reading |
|--------------------|--------------------|---------|
| specific           | > 0.45             | named entities and technical terms in the audio |
| generic            | 0.20 – 0.45        | themes covered without being named |
| out of distribution| < 0.10             | topics absent from the audio |

Failure signatures:

- **Out-of-distribution queries scoring as high as specific ones** — the
  embeddings carry no signal. Check `normalize_embeddings=True` and that
  the query is embedded with the same model used at index time.
- **All scores clustered near 1.0** — you are likely reading raw cosine
  *distance* as similarity, or ranking on the wrong operator (`<->`
  euclidean instead of `<=>` cosine).
- **Top-3 consistently irrelevant on a large corpus** — chunk window is
  wrong. Try `target_duration` 30 or 60 in `app/core/chunking.py` and
  re-index.

---

## Run 1 — 2026-08-03 — 30s sample, single chunk

Corpus: `data/samples/sample.wav`, a 30s Lex Fridman excerpt about Obama.
One chunk indexed (the audio is shorter than the 45s chunk window), so
this run validates **scoring**, not **ranking** — with one candidate,
ordering is trivially correct.

| Query | Class | Top-1 score |
|---|---|---|
| first black president | specific | **0.570** |
| what did Obama achieve in 2008 | specific | **0.503** |
| what makes someone a historic figure | generic | 0.368 |
| change in American politics | generic | 0.242 |
| how to bake sourdough bread | out of dist. | **-0.031** |
| kubernetes pod autoscaling | out of dist. | **-0.040** |

Reading: separation is clean and lands in the expected bands — roughly
0.5 for specific, 0.25–0.37 for generic, and *negative* for both
off-topic queries. The embedding pipeline and the cosine ranking are
behaving correctly.

Caveat, stated plainly: a single-chunk corpus cannot demonstrate ranking
quality. Run 2 below is the one that actually validates retrieval.

## Run 2 — TODO — 15-20 min audio, multi-chunk

Not yet run: needs a longer English recording on a clear topic
(`data/samples/`), transcribed end to end (~5-10 min of CPU on this
machine), then the same query set adapted to its content.

What to record here once run:

- number of chunks indexed, and chunk duration min/max
  (`uv run python scripts/chunk_preview.py`)
- per-query top-3 with scores, and a one-line verdict each
  ("top-1 pertinent", "top-1 off but top-3 has a good hit", "top-3 all
  irrelevant")
- whether the *rank order* is defensible, not just the scores

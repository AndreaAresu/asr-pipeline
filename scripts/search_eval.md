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

| Query class        | Healthy top-1 band | Reading |
|--------------------|--------------------|---------|
| specific           | > 0.50             | named entities and technical terms in the audio |
| generic            | > 0.45             | themes covered without being named |
| out of distribution| < 0.25             | topics absent from the audio |

**These bands move with corpus size, and Run 2 is where that stopped being
theoretical.** On the single-chunk corpus of Run 1, generic queries landed at
0.24-0.37 and out-of-distribution queries scored *negative*. On the 109-chunk
corpus of Run 2 the same kind of queries scored 0.51-0.62 and 0.10-0.19. More
candidates means a better best match, for every query class. Read the gap
between classes, never the absolute number against a band from another run.
The demo turns that gap into a cutoff — see *Choosing a relevance cutoff* at
the end, which measures where it actually sits rather than reading it off
these bands.

With more than one recording indexed there is a second, sharper check that
does not depend on score at all: **a specific query must retrieve from the
right recording.** Wrong-episode hits at a healthy score are the failure a
score band cannot see.

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

## Run 2 — 2026-09-03 — 3 NASA episodes, 109 chunks

Corpus: three episodes of NASA's *Houston We Have a Podcast*, public domain,
66 min 54 s of audio, transcribed with `small.en` and indexed as **109
chunks**. The topics are deliberately disjoint, which is what makes the
routing check below possible:

| Transcript | Episode | Topic | Chunks |
|---|---|---|---|
| `2db117b7` | Gateway: Together to the Moon | lunar station engineering, ESA modules | 40 |
| `3069178a` | Astronaut and Microbiologist | microbiology, DNA sequencing, ISS science | 38 |
| `387b1417` | Apollo 11 to Now | spaceflight history, Artemis, policy | 31 |

Chunk duration min/max: 44.2/49.2 s, 11.9/72.0 s, 38.6/55.3 s respectively.
The wide spread on `3069178a` is not a chunker bug: that episode contains a
27.3 s Whisper segment, and the chunker cannot split below a segment
boundary, so the 45 s target gives way. Worth watching as a ranking bias —
see the reading below.

### Results, top-3 per query

| # | Query | Class | Top-1 | Episodes in top-3 | Verdict |
|---|---|---|---|---|---|
| 1 | DNA sequencing on the space station | specific | **0.684** | Micro ×3 | top-1 pertinent — the passage where Rubins is called the first person to sequence DNA in space |
| 2 | the Lunar Link communication module | specific | **0.552** | Gateway ×3 | top-3 all pertinent; rank order defensible but not optimal — the *definition* of Lunar Link is rank 2, its design discussion is rank 1 |
| 3 | Apollo 11 landing in July 1969 | specific | **0.538** | Apollo ×3 | top-1 pertinent — "July 20, 1969" verbatim |
| 4 | European Space Agency contribution to Gateway | specific | **0.658** | Gateway ×3 | top-1 pertinent |
| 5 | studying viruses and outbreaks in Africa | specific | **0.624** | Micro ×3 | top-1 pertinent — monkeypox in Congo |
| 6 | why go back to the moon | generic | 0.618 | Apollo ×3 | pertinent, but see the reading: Gateway covers this too and never appears |
| 7 | what training an astronaut goes through | generic | 0.514 | Gateway ×2, Micro ×1 | pertinent and correctly mixed — both episodes discuss training |
| 8 | doing science in microgravity | generic | 0.574 | Micro ×3 | pertinent |
| 9 | how to bake sourdough bread | out of dist. | **0.105** | Apollo, Gateway, Micro | correctly near-zero; the three hits are one per episode, i.e. noise |
| 10 | kubernetes pod autoscaling | out of dist. | **0.188** | Micro, Gateway, Apollo | correctly low, but see the reading |

### Reading

**Routing is perfect on specific queries: 15 of 15 top-3 hits landed in the
right episode.** This is the check Run 1 structurally could not perform, and
it is the one that matters for a multi-recording index. No specific query
pulled a plausible-looking passage from the wrong recording.

**Rank order is defensible.** Query 2 is the only one where a human would
reorder the top 3, and even there all three hits are about Lunar Link — the
definitional passage sits second behind a passage about servicing it. Every
other query puts the passage a human would pick first in position 1.

**Score separation holds, but not where Run 1 put it.** Specific and generic
now overlap almost completely (0.54-0.68 against 0.51-0.62): with a corpus
this size, a broad theme finds a genuinely good passage, so a high score no
longer distinguishes a named entity from a theme. What survives is the gap to
out-of-distribution: worst in-domain top-1 is 0.514, best out-of-distribution
top-1 is 0.188, a factor of 2.7. The embedding pipeline and the cosine
ranking are behaving correctly; the table above has been rewritten to match.

**Out-of-distribution scores rose from negative to positive**, which is
expected and not a regression: 109 candidate chunks contain a better nearest
neighbour for *any* query than 1 chunk did. `kubernetes pod autoscaling` at
0.188 is the reminder that this floor keeps rising with the index, and that a
score threshold hard-coded today will drift.

**No sign of a long-chunk ranking bias.** The 72 s chunk on `3069178a` never
appears in any top-3, and the winning chunks cluster around the 45 s target.
The uneven chunking on that episode did not distort retrieval on this query
set — worth re-checking on a corpus where the variance is wider.

**One caveat about reading the harness output**: `search_eval.sh` truncates
each hit to ~96 characters. On query 1 the truncated preview of the top hit
looks generic while the full chunk is squarely on topic. Judge a suspicious
rank by the full chunk text, not the preview.

### Not validated here

- **Recall.** This measures what comes back at the top, not what was missed.
  Query 6 is the visible symptom: Gateway discusses returning to the Moon at
  length and never enters the top 3.
- **Scale.** 109 chunks is a sequential scan. Ranking behaviour under an
  IVFFlat index, which is approximate, is a different experiment.

## Choosing a relevance cutoff — 2026-09-04, same 109-chunk corpus

Run 2 showed out-of-distribution queries scoring 0.10-0.19 while in-domain
queries scored 0.51-0.68, which invites a threshold. But Run 2 only recorded
**top-1** per query, and a threshold does not filter queries — it filters
individual hits. So the top-5 of every query was measured, plus three
out-of-distribution queries Run 2 did not use.

| Class | Queries | Hits | Score range |
|---|---|---|---|
| in-domain (specific + generic) | 8 | 40 | 0.181 – 0.684 |
| out of distribution | 5 | 25 | 0.077 – 0.224 |

**The pooled ranges overlap, and that is the finding.** One in-domain query —
*studying viruses and outbreaks in Africa* — runs 0.624, 0.539, 0.372 and then
falls off a cliff to 0.189, 0.181. The corpus holds about three passages on
that subject; ranks 4 and 5 are the tail, and they score *below* the best
out-of-distribution hit (0.224, *best pizza in naples*). A good query has a
noise tail too.

**But no hit lands between 0.224 and 0.372.** Every out-of-distribution hit is
under the first number; every hit a human would keep is over the second. The
band is empty, so the cutoff goes in the middle of it:

    MIN_RELEVANT_SCORE = 0.30

Equidistant from both ways of being wrong: 0.076 above the highest observed
noise, 0.072 below the weakest hit worth showing. Run 2's suggestion of 0.35
would have left only 0.022 of margin on the in-domain side.

**Why the margin above matters.** Adding three out-of-distribution queries not
in Run 2 pushed the ceiling from 0.188 to 0.224. Sample more nonsense and the
maximum keeps creeping; a cutoff sitting just above the observed maximum is a
cutoff waiting to be crossed.

**What would move this number.** Corpus size, first: at one chunk (Run 1) these
queries scored *negative*, at 109 chunks they score 0.10-0.22. Also the
embedding model, the chunk length, and the domain — a corpus covering more
subjects makes a stray query likelier to find something genuinely close. Treat
0.30 as measured on this index, not as a property of cosine similarity.

**Where it is applied.** In `demo/app.py`, not in `/search`. The API stays a
pure ranking engine: the number expires with the corpus, so it belongs next to
the sentence explaining it rather than in a parameter a client would hardcode.
Hits under it are put behind a "show the nearest matches anyway" expander
rather than dropped, because *retrieval always returns the nearest neighbours*
is the thing worth understanding, not the thing to hide.

The cutoff is **not** applied to searching one's own uploaded transcript. It
was calibrated against 45s chunks in a 109-chunk index; a 35s upload is a
single chunk holding everything, and well-aimed queries against one measured
0.309-0.325 — right on top of the cutoff, because the passage is diluted
rather than irrelevant. A short upload would look broken.

# asr-pipeline

**Upload a recording, get a searchable transcript.** A FastAPI service takes
audio, returns a job id immediately, and transcribes it out of band with
Whisper. On the way out every transcript is chunked into overlapping time
windows, embedded, and stored in Postgres, so the whole archive answers
questions in plain language and any recording can be summarized into
timestamped sections.

**[Try the live demo](https://demo.159-195-250-205.sslip.io)**: no signup, no
key. Search 67 minutes of indexed NASA podcasts, or upload your own clip and
watch it go through. Also live: the [API and its browser
console](https://api.159-195-250-205.sslip.io), with [OpenAPI
docs](https://api.159-195-250-205.sslip.io/docs), and an
[**MCP endpoint**](#ask-a-model-instead-of-a-search-box) that hands the same
retrieval to an assistant as three tools.

![Semantic search over the indexed corpus](docs/img/search.jpg)

> The query above never says "radiation", "CO2" or "spacesuit". It matches on
> meaning, and every hit carries the timestamp where it was said.

![Uploading a clip and watching it transcribe](docs/img/upload-flow.gif)

## How it works

Transcription is minutes of CPU, far past any sensible HTTP timeout, so the
request that starts it does not wait for it.

```mermaid
flowchart LR
    Client["Client<br/>curl, console at /, Streamlit demo"]

    subgraph API["FastAPI process"]
        direction TB
        T["POST /transcribe"]
        J["GET /jobs/{id}"]
        S["POST /search<br/>embeds the query"]
        Z["POST /summarize/{id}"]
    end

    Spool[("Shared spool<br/>temp_audio_dir")]
    Redis[("Redis<br/>RQ queue")]

    subgraph Worker["RQ worker process"]
        direction TB
        W1["Whisper<br/>transcribe"]
        W2["chunk + embed<br/>MiniLM"]
    end

    PG[("Postgres + pgvector<br/>jobs, transcripts,<br/>chunks, summaries")]
    Groq["Groq<br/>openai/gpt-oss-120b"]

    Client -->|"audio upload"| T
    T -->|"spool the file"| Spool
    T -->|"job: queued"| PG
    T -->|"enqueue id + path"| Redis
    Redis -->|"dequeue"| W1
    Spool -->|"read, then delete"| W1
    W1 --> W2
    W2 -->|"transcript, vectors, job: done<br/>one transaction"| PG
    Client -->|"poll"| J
    J --> PG
    Client --> S
    S -->|"cosine distance top-k"| PG
    Client --> Z
    Z -->|"cache miss"| Groq
    Z -->|"cache read/write"| PG
```

The API validates the upload, spools it to a directory the worker can also
read, writes a `Job` row, pushes the id onto Redis and returns `202`. A
separate worker process runs Whisper, then **in the same transaction that
marks the job `done`** it writes the transcript, splits it into 45-second
windows with 10 seconds of overlap, embeds them and stores the vectors. A
transcript is therefore never visible without its index. The worker deletes
the spooled file whether the job succeeded or not.

Search embeds the query with the same model, in the API process, and ranks by
cosine distance inside Postgres. Summaries are generated once per transcript
and cached, since the input never changes.

That shared spool is the one piece of coupling in the diagram, and it is why
the API and the worker run on the same host. See
[`docs/deploy.md`](docs/deploy.md).

## Retrieval quality

Semantic search is the part of a project like this most easily claimed and
least often measured. [`scripts/search_eval.md`](scripts/search_eval.md)
carries two dated runs against a fixed 10-query set, graded by hand: five
queries name entities the audio contains, three name broad themes it covers
without saying so, and two are about subjects it has never heard of.

Run 2, over the 109 chunks of the seeded corpus:

| | |
|---|---|
| Episode routing | **15 of 15** top-3 hits from the five specific queries landed in the right recording |
| Rank order | 9 of 10 queries put the passage a human would pick first in position 1 |
| In-domain vs out-of-distribution | worst in-domain top-1 **0.514**, best out-of-distribution **0.188**, a factor of 2.7 |

Routing is the check that needs more than one recording, and the one that
matters for a multi-recording index. The question is not "did it find
something plausible" but "did it find it in the document that actually holds
the answer".

The finding worth more than the scores: **Run 1's score bands did not survive
the corpus growing.** At one chunk, specific and generic queries separated
cleanly. At 109 they overlap almost completely, because a broad theme now
finds a genuinely good passage. Out-of-distribution scores rose from negative
to 0.10-0.19 for the same reason, since more candidates mean a better nearest
neighbour for any query at all. A relevance threshold hard-coded today drifts
as the index grows, so the demo's 0.30 cutoff is recorded together with the
measurement it came from.

![A cached summary with timestamped sections](docs/img/summary.jpg)

## The corpus

Three episodes of NASA's *Houston We Have a Podcast*, 66 min 54 s of audio,
one `small.en` pass, 109 chunks. It serves both the evaluation and the demo,
so the two cannot drift apart.

- [Gateway: Together to the Moon](https://www.nasa.gov/podcasts/houston-we-have-a-podcast/gateway-together-to-the-moon/)
- [Astronaut and Microbiologist](https://www.nasa.gov/podcasts/houston-we-have-a-podcast/astronaut-and-microbiologist/)
- [Apollo 11 to Now](https://www.nasa.gov/podcasts/houston-we-have-a-podcast/apollo-11-to-now/)

Public domain as works of a US government agency. NASA asks to be credited,
which is what this section does; its insignia is not public domain and is not
used here. The indexed rows are committed as `data/seed/nasa_corpus.sql` and
regenerated by `scripts/dump_seed.sh`.

## Ask a model instead of a search box

The retrieval also speaks **MCP**, over Streamable HTTP at
`https://api.159-195-250-205.sslip.io/mcp`, so an assistant can query the
corpus itself: decide what to search for, widen a hit when it needs more, and
answer with the recording and the timestamp it read.

```json
{
  "mcpServers": {
    "asr-pipeline": {
      "type": "http",
      "url": "https://api.159-195-250-205.sslip.io/mcp",
      "headers": { "X-API-Key": "04l-JpnSG5IDxvMVgnGjnjOS-7_G-biptXKDy1sJLqE" }
    }
  }
}
```

That key is real, public on purpose, and quota-limited to 10 audio minutes a
day. It is separate from the one behind the Streamlit demo so it can be
revoked on its own; if it ever is, run the stack locally (see
[Run it](#run-it)), mint your own with
`docker compose exec api python -m scripts.create_api_key mcp-demo 10`, and
point the config at `http://localhost:8080/mcp`.

For Claude Code that block goes in `.mcp.json`, or in one command:

```bash
claude mcp add --transport http asr-pipeline \
  https://api.159-195-250-205.sslip.io/mcp \
  -H "X-API-Key: 04l-JpnSG5IDxvMVgnGjnjOS-7_G-biptXKDy1sJLqE"
```

`Authorization: Bearer <key>` is accepted as well, for clients that can only
compose a Bearer header. A key in the query string is not, because the MCP
specification forbids it.

**Clients that work:** anything that can send a header to a remote server, so
Claude Code, Cursor, VS Code / Copilot, Windsurf and Codex CLI.
**Clients that do not:** Claude Desktop, claude.ai and ChatGPT, whose custom
connectors offer OAuth or no authentication and nothing in between. That is a
limitation of this server and not a fault of yours: running an OAuth
authorization server to hand out one demo key is not a trade this project
makes.

### The three tools

| Tool | Answers |
|---|---|
| `list_transcripts` | *what is in this corpus?* Filename, duration, language, passage count |
| `search_transcripts` | *where is X discussed?* Passages ranked by cosine similarity, each with its filename and mm:ss span |
| `fetch_transcript_window` | *what exactly is said around that point?* The text over a time interval, up to five minutes |

Three, not one, because a single search tool is a `POST` with a JSON schema
around it: the model would get five passages identified by a UUID, with no way
to know what the index holds and no way to widen a hit into something
quotable.

### A real conversation

Claude Code, connected with the block above and nothing else, asked one
question. Verbatim exchange, translated from Italian; the quoted passages are
the tool output word for word.

> **"What is in this corpus, and where does it talk about DNA sequencing?"**
>
> *Called asr-pipeline 3 times*

Three calls, chosen by the model with no instruction beyond the question -
it announced it would list the corpus and search it in parallel. What came
back:

> **What is in the corpus.** Three public-domain episodes of NASA's *Houston
> We Have a Podcast*, in English, 109 indexed passages: `nasa-apollo11.mp3`
> (19:00, 31 passages), `nasa-gateway.mp3` (24:21, 40), and
> `nasa-microbiologist.mp3` (23:31, 38).
>
> Note: the search index also contains recordings uploaded by visitors to the
> public demo, but the curated corpus is only this.
>
> **Where DNA sequencing is discussed.** All of it in
> `nasa-microbiologist.mp3`, in one continuous block between ~9:40 and
> ~12:10.
>
> **9:41-10:26** - the first DNA sequencing in space:
> *"And you got to do some of the first things, like sequencing DNA. That was
> a big one because you were the first person to sequence DNA. [...] This was
> a technology that we were trying to prove if it would work in microgravity.
> We didn't know."*
>
> **10:16-11:03** - the result, over two billion base pairs:
> *"We were able to demonstrate that we could do it. [...] How many billions
> of base pairs can we get to? I think we sequenced a little bit over two
> billion base pairs."*
>
> **10:51-12:13** - what comes next, multiplexing with molecular barcodes.
>
> Related context, not sequencing proper: **4:13-5:06**, fieldwork in Africa
> on monkeypox and Ebola (thematic relevance only, score just above the 0.30
> noise floor).

Three things in that answer came from the tool descriptions rather than from
the model's own knowledge, and they are the reason this is not a `POST` with
a JSON schema around it. It knew the corpus was a corpus and said what was in
it. It warned, unprompted, that search reaches further than the listing. And
it flagged the weak hit **as** weak instead of presenting it beside the good
ones, because the hit arrived marked rather than filtered out.

### The uncomfortable parts, stated

- **Search covers the whole index, including clips uploaded by visitors to
  the public demo.** `list_transcripts` shows only the curated corpus, so a
  hit whose filename is not in the listing came from a stranger. Both tool
  descriptions say so, because the description is the documentation a model
  actually reads.
- **A weak hit is marked, not hidden.** Retrieval always returns the nearest
  passages however far away. Below the measured 0.30 noise floor a hit comes
  back with `below_threshold: true` rather than being dropped, so a model can
  say "the closest passage was this, and it does not answer you" instead of
  concluding the corpus is silent.
- **Passages, not documents.** A hit is one ~45 s chunk, which is why
  `fetch_transcript_window` exists.
- **Caps.** Ten passages per search, five minutes per window. Over the window
  cap the call is refused with a message rather than trimmed in silence.
- **Nothing counts a search, and that is a decision, not an oversight.** The
  only quota in this system is in audio minutes and applies to `/transcribe`,
  so it bounds nothing the three tools do. Measured on the deployed box
  before publishing the key: a search costs **25 ms P50** in-process, of
  which **15 ms is embedding the query and 4.5 ms is the pgvector scan**, and
  during a live transcription on the same 4-core machine it degrades to
  ~279 ms P50 with an 783 ms tail (measured over the internet, against 239 ms
  idle). At that price a looping agent is a nuisance, not a threat, and the
  key can be revoked in one statement. Capping `top_k` would be the wrong
  guard anyway, since the scan is the cheap half; if one is ever needed it
  has to count *queries* per key. Details in
  [`docs/performance.md`](docs/performance.md).

## Run it

```bash
cp .env.example .env            # fill in POSTGRES_USER / PASSWORD / DB
docker compose up -d --build    # API, worker, Postgres+pgvector, Redis
docker compose exec api python -m scripts.create_api_key dev
```

The last command prints an API key once. Only its SHA-256 is stored, so copy
it now. Then open **http://localhost:8080/** for the built-in console, or:

```bash
curl -X POST http://localhost:8080/transcribe \
  -H "X-API-Key: $KEY" -F "audio=@/path/to/recording.mp3"   # -> {"job_id": "..."}
curl http://localhost:8080/jobs/$JOB_ID -H "X-API-Key: $KEY"          # poll
curl http://localhost:8080/jobs/$JOB_ID/result -H "X-API-Key: $KEY"   # transcript
```

Audio is not committed, since `data/samples/` is gitignored, so bring your
own: `.wav`, `.mp3`, `.m4a`, `.flac` or `.mp4`, checked by filename.

`bash scripts/smoke_test.sh /path/to/recording.mp3` runs the whole thing:
build, transcribe, search, summarize, teardown. It needs `jq`.

![The console the API serves at /](docs/img/console.jpg)

<details>
<summary><b>Two UIs and two ports, both on purpose</b></summary>

<br>

`app/web/index.html` is the console above, served by the API at `/`. It is one
static file with no build step and no second process, and it is what you get
by running the stack. `demo/app.py` is the Streamlit app behind the public
link, with its own compose service and its own image, written for a visitor
who is not going to read an API reference.

Port **8080** is compose. Running the API directly on the host with
`uv run fastapi dev app/main.py` listens on **8000**. Both appear in this
repo and neither is wrong.

</details>

<details>
<summary><b>Developing without Docker</b></summary>

<br>

```bash
uv sync
docker compose up -d postgres redis
uv run alembic upgrade head
uv run fastapi dev app/main.py            # terminal 1, serves on :8000
uv run python -m app.workers.run          # terminal 2, required

uv run ruff check .                       # lint
uv run pytest                             # unit tests, no services needed
```

The worker is a **separate process**. Without it, uploads sit in `queued`
forever. See [CONTRIBUTING.md](CONTRIBUTING.md) for the gotchas worth knowing:
macOS fork behaviour, Alembic and pgvector, layering rules.

</details>

## Numbers

Measured on the deployed VPS, 4 vCore, CPU only, `base.en`:

| | |
|---|---|
| 30 s of audio to `done` | **19.7 s** P50 |
| At the 90 s cap, what a visitor waits | **~37 s** warm, 42 s cold |
| `POST /search`, top_k=5 | **62 ms** P50 |
| `POST /summarize`, cache hit | **37 ms** |
| Peak worker memory | **1.30 GB**, so size the worker at 2 GB |
| Container image | **3.17 GB** |

Full tables, both machines, and the 60-minute run are in
[`docs/performance.md`](docs/performance.md).

## What it does not do

CPU only, English only, no diarization, no streaming, no partial results. The
vector scan is sequential: exact, and fast to roughly 10k chunks. Rate
limiting is enforced after the upload lands, because the quota is counted in
audio minutes and that needs `ffprobe`. `/search` and `/summarize` check no
ownership, so any valid key reads the whole index.

The complete list with the reasoning is in
[`docs/design.md`](docs/design.md), which also holds the API reference and
the stack trade-offs.

## Deployment

One VPS running this repo's `compose.yaml`, with Caddy on the host
terminating TLS, and the API and the demo on separate `sslip.io` hostnames so
no domain is needed. The runbook, including why the API and the worker are
*not* on separate machines, is in [`docs/deploy.md`](docs/deploy.md).

## License

MIT, see [LICENSE](LICENSE). Built with `uv` on Python 3.13.

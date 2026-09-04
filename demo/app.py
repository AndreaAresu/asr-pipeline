"""Streamlit demo for the ASR pipeline.

A thin client over the API: it holds no logic of its own, just calls
`/search` and `/summarize` and renders what comes back. It lives outside
`app/` and has its own requirements.txt and Dockerfile because the two
images should stay apart — the API image carries torch, this one must not,
and this one carries Streamlit, which the worker must not.

Runs as the `demo` service in compose, on the same host as the API, not on
Streamlit Community Cloud: a community-cloud app sleeps when idle and
greets the first visitor after a quiet period with a "wake this app up"
button, which is the wrong first impression for a public demo.

Note this is the *second* UI. `app/web/index.html`, served by the API at
`/`, is the console that needs no extra process; this one is the public
face and reaches the API over the compose network.

Configuration comes from the environment (compose passes it in):

    ASR_API_URL   base URL of the API — http://api:8080 inside compose
    ASR_API_KEY   a demo key with a deliberately low daily quota

    MAX_UPLOAD_MB, MAX_AUDIO_SECONDS
                  the API's own upload caps, passed in so the UI can state
                  them before a visitor picks a file. compose feeds both
                  services from the same .env values, so the numbers shown
                  here are the numbers actually enforced — a UI that
                  hardcoded its own would drift the moment either changed.
"""

import os
import time

import requests
import streamlit as st

API_URL = os.environ.get("ASR_API_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("ASR_API_KEY", "")
TIMEOUT = 60

# 0 means "no limit" on both, matching app/config.py.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "0"))
MAX_AUDIO_SECONDS = int(os.environ.get("MAX_AUDIO_SECONDS", "0"))

# Extensions app/api/transcribe.py accepts. It checks the filename, not the
# content, so an accurate list here is what stops a visitor from waiting on
# an upload the API will reject on arrival.
ACCEPTED_TYPES = ["wav", "mp3", "m4a", "flac", "mp4"]

# How long to keep polling a job before giving up on the UI side. The job
# itself is not cancelled — it stays in the queue and finishes — so this
# only bounds how long the page waits.
POLL_TIMEOUT_SEC = 600
POLL_INTERVAL_SEC = 2

# The indexed corpus: three public-domain episodes of NASA's "Houston We
# Have a Podcast", restored from data/seed/nasa_corpus.sql. Ids come from
# the environment so a deployment with a different corpus can point these
# elsewhere without touching the code.
EPISODES = {
    "NASA — Gateway: Together to the Moon": os.environ.get("DEMO_TRANSCRIPT_GATEWAY", ""),
    "NASA — Astronaut and Microbiologist": os.environ.get("DEMO_TRANSCRIPT_MICROBIOLOGY", ""),
    "NASA — Apollo 11 to Now": os.environ.get("DEMO_TRANSCRIPT_APOLLO", ""),
}

st.set_page_config(page_title="ASR Pipeline Demo", layout="wide", page_icon="🎙️")


def api_post(path: str, **kwargs) -> tuple[dict | None, str | None]:
    """POST to the API, returning `(payload, error_message)`.

    Network and HTTP failures are turned into a message for the UI rather
    than an exception, so a cold backend or an exhausted quota shows up as
    readable text instead of a Streamlit traceback.
    """
    if not API_KEY:
        return None, "ASR_API_KEY is not configured for this demo."
    try:
        response = requests.post(
            f"{API_URL}{path}",
            headers={"X-API-Key": API_KEY},
            timeout=TIMEOUT,
            **kwargs,
        )
    except requests.RequestException as e:
        return None, f"Could not reach the API: {e}"

    return _interpret(response)


def api_get(path: str, **kwargs) -> tuple[dict | None, str | None]:
    """GET from the API, with the same error handling as `api_post`.

    Exists for job polling, which is the one read the demo does.
    """
    if not API_KEY:
        return None, "ASR_API_KEY is not configured for this demo."
    try:
        response = requests.get(
            f"{API_URL}{path}",
            headers={"X-API-Key": API_KEY},
            timeout=TIMEOUT,
            **kwargs,
        )
    except requests.RequestException as e:
        return None, f"Could not reach the API: {e}"

    return _interpret(response)


def _interpret(response: requests.Response) -> tuple[dict | None, str | None]:
    """Turn a response into `(payload, error_message)`."""
    if response.status_code == 413:
        # The API's own message names the limit and the value that broke it
        # ("audio is 180s long; the limit is 90s"), which is more useful
        # than anything this side could reconstruct.
        return None, _detail(response, "The upload is over one of the demo's limits.")
    if response.status_code == 429:
        return None, (
            "The demo key's shared daily quota is exhausted — everyone using "
            "this page draws on the same one. It refills on a rolling 24h "
            "window, so try again later."
        )
    if response.status_code == 503:
        return None, "Summarization is not configured on this deployment."
    if not response.ok:
        return None, f"API returned {response.status_code}: {response.text[:300]}"
    return response.json(), None


def _detail(response: requests.Response, fallback: str) -> str:
    """Pull FastAPI's `detail` out of an error body, if there is one."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        return fallback
    return detail if isinstance(detail, str) else fallback


def fmt_time(seconds: float) -> str:
    """Render seconds as mm:ss."""
    return f"{int(seconds) // 60:d}:{int(seconds) % 60:02d}"


def render_summary(payload: dict) -> None:
    """Render a /summarize response: provenance caption, then sections.

    Shared by the Summarize tab and the Transcribe tab's follow-up button,
    which summarize different transcripts through the same response shape.
    """
    meta = payload["meta"]
    st.caption(
        f"{payload['model']} · {'cached' if payload['cached'] else 'freshly generated'} · "
        f"{meta['input_tokens']} in / {meta['output_tokens']} out tokens"
    )
    for section in payload["sections"]:
        header = (
            f"{section['title']}  ·  "
            f"{fmt_time(section['start_sec'])}–{fmt_time(section['end_sec'])}"
        )
        with st.expander(header, expanded=True):
            for point in section["key_points"]:
                st.markdown(f"- {point}")


def describe_limits() -> str:
    """Sentence naming the caps in force, for display before file choice."""
    parts = []
    if MAX_AUDIO_SECONDS:
        parts.append(f"**{MAX_AUDIO_SECONDS} seconds** of audio")
    if MAX_UPLOAD_MB:
        parts.append(f"**{MAX_UPLOAD_MB} MB**")
    if not parts:
        return "No upload limits are configured on this deployment."
    return "Maximum " + " and ".join(parts) + " per file."


st.title("🎙️ ASR Pipeline")
st.markdown(
    "Whisper transcription as an async service, with semantic search and "
    "LLM summarization over the result. Audio is transcribed by a queue "
    "worker, chunked into overlapping time windows, embedded, and indexed "
    "in Postgres with pgvector."
)
st.caption(
    "Running on a single small VPS, one worker, one queue. Nothing suspends "
    "between visits, but the first transcription after a restart also pays "
    "for loading the model, and two visitors at once wait in line."
)

transcribe_tab, search_tab, summarize_tab = st.tabs(["Transcribe", "Search", "Summarize"])

def poll_until_finished(job_id: str, status_box) -> tuple[dict | None, str | None]:
    """Poll `/jobs/{id}` until it leaves the queue, narrating as it goes.

    `queued` and `processing` are shown rather than hidden behind a bare
    spinner: with one worker, "queued" means someone else's file is ahead
    of you, and that is information a visitor can act on. A spinner that
    says nothing for two minutes reads as a hang.
    """
    deadline = time.monotonic() + POLL_TIMEOUT_SEC
    seen = None
    while time.monotonic() < deadline:
        job, error = api_get(f"/jobs/{job_id}")
        if error:
            return None, error

        if job["status"] != seen:
            seen = job["status"]
            if seen == "queued":
                status_box.update(label="Queued — waiting for the worker to pick it up")
            elif seen == "processing":
                status_box.update(label="Transcribing…")

        if job["status"] == "done":
            return job, None
        if job["status"] == "failed":
            return None, job.get("error_message") or "The job failed."

        time.sleep(POLL_INTERVAL_SEC)

    return None, (
        f"Still running after {POLL_TIMEOUT_SEC // 60} minutes. The job was not "
        "cancelled — reload and it may have finished."
    )


with transcribe_tab:
    st.subheader("Transcribe your own audio")
    st.write(
        "Upload a file and watch it go through the queue. This is the whole "
        "pipeline: the API accepts the upload and returns a job id "
        "immediately, a worker transcribes it out of band, and the "
        "transcript is chunked, embedded and indexed on the way out."
    )

    # Stated before the file picker, deliberately. A limit discovered after
    # an upload is a failure; announced before it is a design decision, and
    # the visitor gets to pick a file that fits.
    st.info(
        f"{describe_limits()}  \n"
        "The daily quota is shared by everyone using this page, and there is "
        "a single worker — if someone else is mid-transcription, yours waits "
        "in line behind theirs."
    )

    uploaded = st.file_uploader(
        "Audio or video file",
        type=ACCEPTED_TYPES,
        help="Transcribed by Whisper on CPU. Short clips come back fastest.",
    )

    oversize = (
        uploaded is not None
        and MAX_UPLOAD_MB
        and uploaded.size > MAX_UPLOAD_MB * 1024 * 1024
    )
    if oversize:
        # Caught here as well as by the API's 413: there is no reason to
        # push megabytes over the wire only to be told they were too many.
        st.error(
            f"That file is {uploaded.size / 1024 / 1024:.1f} MB, over the "
            f"{MAX_UPLOAD_MB} MB limit. Pick a shorter clip."
        )

    if st.button("Transcribe", type="primary", disabled=uploaded is None or bool(oversize)):
        # st.status renders as a collapsed expander, so anything written
        # inside it is one click away from being read. The status box gets
        # the progress label and nothing else; the failure message — which
        # is the one thing a rejected visitor must see, since it names the
        # limit that was hit — is rendered after the block, in the open.
        error = None
        with st.status("Uploading…") as status_box:
            payload, error = api_post(
                "/transcribe",
                files={"audio": (uploaded.name, uploaded.getvalue())},
            )
            if error:
                status_box.update(label="Rejected", state="error")
            else:
                job_id = payload["job_id"]
                status_box.update(label=f"Accepted as job {job_id} — HTTP 202")
                job, error = poll_until_finished(job_id, status_box)

                if not error:
                    result, error = api_get(f"/jobs/{job_id}/result")

                if error:
                    status_box.update(label="Failed", state="error")
                else:
                    status_box.update(label="Done", state="complete")
                    # Kept in session state so the transcript survives the
                    # reruns that every later widget triggers.
                    st.session_state["own"] = {
                        "transcript_id": result["transcript_id"],
                        "full_text": result["full_text"],
                        "filename": uploaded.name,
                        "duration": job.get("duration"),
                    }
                    st.session_state.pop("own_summary", None)

        if error:
            st.error(error)

    own = st.session_state.get("own")
    if own:
        st.divider()
        duration = f" · {fmt_time(own['duration'])}" if own.get("duration") else ""
        st.markdown(f"**{own['filename']}**{duration}")
        st.text_area("Transcript", own["full_text"], height=200)
        st.caption(f"transcript_id `{own['transcript_id']}` — indexed and searchable.")

        own_query = st.text_input(
            "Search inside this transcript",
            placeholder="ask about something that was said in your file",
            key="own_query",
        )
        if own_query:
            # transcript_id scopes the search to this file alone, so hits
            # come from the visitor's own audio and not from the NASA corpus.
            with st.spinner("Searching your transcript…"):
                payload, error = api_post(
                    "/search",
                    json={"query": own_query, "top_k": 3, "transcript_id": own["transcript_id"]},
                )
            if error:
                st.error(error)
            elif not payload["hits"]:
                st.warning("No hits — the file may be too short to have produced a chunk.")
            else:
                for hit in payload["hits"]:
                    with st.container(border=True):
                        st.caption(
                            f"{fmt_time(hit['start_sec'])}–{fmt_time(hit['end_sec'])} "
                            f"· score {hit['score']:.3f}"
                        )
                        st.write(hit["text"])

        if st.button("Summarize this transcript"):
            with st.spinner("Summarizing…"):
                payload, error = api_post(f"/summarize/{own['transcript_id']}")
            # Stored, not rendered here: a button is True for one rerun only,
            # so rendering inline would make the summary disappear as soon as
            # the visitor touched the search box above it.
            st.session_state["own_summary"] = {
                "transcript_id": own["transcript_id"],
                "payload": payload,
                "error": error,
            }

        summary = st.session_state.get("own_summary")
        if summary and summary["transcript_id"] == own["transcript_id"]:
            if summary["error"]:
                st.error(summary["error"])
            else:
                render_summary(summary["payload"])

with search_tab:
    st.subheader("Semantic search")
    st.write(
        "Matches meaning, not keywords — the query does not have to use the "
        "words that were actually spoken."
    )

    query = st.text_input("Query", placeholder="e.g. what makes intelligence hard to define")
    top_k = st.slider("Results", min_value=1, max_value=10, value=5)

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching…"):
            payload, error = api_post("/search", json={"query": query, "top_k": top_k})

        if error:
            st.error(error)
        elif not payload["hits"]:
            st.warning("No hits — nothing is indexed on this deployment yet.")
        else:
            for hit in payload["hits"]:
                with st.container(border=True):
                    score_col, text_col = st.columns([1, 6])
                    with score_col:
                        st.metric("Score", f"{hit['score']:.3f}")
                        st.caption(f"{fmt_time(hit['start_sec'])}–{fmt_time(hit['end_sec'])}")
                    with text_col:
                        st.write(hit["text"])

with summarize_tab:
    st.subheader("Structured summary")
    st.write(
        "The transcript is sent to an LLM with inline timestamp markers, so "
        "section times are quoted from the audio rather than invented. The "
        "model that answered is named under each result — the deployment "
        "picks it, because hosted models get retired. Results are cached: "
        "the first call takes seconds, every later one is a database lookup."
    )

    labelled = {name: tid for name, tid in EPISODES.items() if tid}
    if not labelled:
        st.info("No demo transcripts are configured. Set DEMO_TRANSCRIPT_* in the app secrets.")
    else:
        choice = st.selectbox("Episode", list(labelled))

        if st.button("Summarize", type="primary"):
            with st.spinner("Summarizing…"):
                payload, error = api_post(f"/summarize/{labelled[choice]}")

            if error:
                st.error(error)
            else:
                render_summary(payload)

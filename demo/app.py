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
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("ASR_API_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("ASR_API_KEY", "")
TIMEOUT = 60

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

    if response.status_code == 429:
        return None, "The demo key's daily quota is exhausted. Try again tomorrow."
    if response.status_code == 503:
        return None, "Summarization is not configured on this deployment."
    if not response.ok:
        return None, f"API returned {response.status_code}: {response.text[:300]}"
    return response.json(), None


def fmt_time(seconds: float) -> str:
    """Render seconds as mm:ss."""
    return f"{int(seconds) // 60:d}:{int(seconds) % 60:02d}"


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

search_tab, summarize_tab = st.tabs(["Search", "Summarize"])

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

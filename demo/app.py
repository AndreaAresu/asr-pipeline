"""Streamlit demo for the ASR pipeline.

A thin client over the public API: it holds no logic of its own, just
calls `/search` and `/summarize` and renders what comes back. Deployed
separately on Streamlit Community Cloud, which is why it lives outside
`app/` and has its own requirements.txt — the API image should not carry
Streamlit.

Configuration comes from the environment (Streamlit Cloud calls these
"secrets"):

    ASR_API_URL   base URL of the API
    ASR_API_KEY   a demo key with a deliberately low daily quota
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("ASR_API_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("ASR_API_KEY", "")
TIMEOUT = 60

# Transcripts indexed on the demo database. Replace the ids with the ones
# printed by /jobs/{id}/result after indexing your own episodes.
EPISODES = {
    "Lex Fridman — AI / ML": os.environ.get("DEMO_TRANSCRIPT_AI", ""),
    "Lex Fridman — Math / Physics": os.environ.get("DEMO_TRANSCRIPT_SCIENCE", ""),
    "Lex Fridman — History / Politics": os.environ.get("DEMO_TRANSCRIPT_HISTORY", ""),
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
    "The backend suspends when idle, so the first request after a quiet "
    "period can take ~30s to wake it."
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
        "The transcript is sent to Llama 3.3 70B with inline timestamp "
        "markers, so section times are quoted from the audio rather than "
        "invented. Results are cached: the first call takes seconds, every "
        "later one is a database lookup."
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

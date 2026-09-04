# The Streamlit demo, as its own image.
#
# Build context is ./demo (see compose.yaml), not the repo root. That is the
# whole reason this file lives here: the root .dockerignore excludes demo/,
# and a build rooted at demo/ never consults it. Nothing to un-exclude, and
# the API build context stays free of Streamlit.
#
# Two images, not one, on purpose: the API image carries torch and
# faster-whisper and weighs gigabytes. Merging them would put Streamlit in
# the worker and torch in the demo, and every rebuild of either would pay
# for both.
#
# Measured: 547MB, of which 285MB is pyarrow + pandas + numpy. The demo
# renders no dataframe and wants none of them, but Streamlit hard-depends
# on all three, so they are the price of Streamlit rather than something to
# trim. Still an order of magnitude under the API image, which is the point.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# pip, not uv: two direct dependencies pinned in a requirements.txt, no lock
# file and no project to install. uv would buy nothing here.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

EXPOSE 8501

# headless: skip the "enter your email" prompt Streamlit shows on first run,
# which would block the container at startup.
# address 0.0.0.0: bind inside the container so compose can publish it. The
# port is still bound to 127.0.0.1 on the host — Caddy is the only way in.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

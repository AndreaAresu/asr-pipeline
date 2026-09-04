# Single image for both the API and the worker: they share all the code
# and nearly all the dependencies, and the process is chosen by the
# command (see compose.yaml / fly.*.toml). Two images would double build
# time and let the two halves drift apart.
#
# Python 3.13 to match .python-version and requires-python in pyproject.
FROM python:3.13-slim

# uv resolves from uv.lock, so the image gets the same versions as local
# development rather than whatever pip resolves on build day.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Redirect the HuggingFace cache to a mount point: Whisper (~400MB) and
    # MiniLM (~22MB) download on first use, and without a persistent volume
    # here every container restart re-downloads them.
    HF_HOME=/cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# ffmpeg: faster-whisper shells out to it to decode audio, and so does the
# ffprobe duration check in the rate limiter.
# libgomp1: OpenMP runtime required by CTranslate2.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than
# the source, so editing a route does not reinstall torch.
COPY pyproject.toml uv.lock README.md ./
# `uv cache clean` runs in the same RUN as the sync, not a later one: a
# later layer cannot shrink an earlier one, so the cache would stay in the
# image regardless. UV_LINK_MODE=copy means the venv holds its own copies
# of every package, so deleting the download cache leaves it intact, and
# it is not small: 1.4 GB of it beside a 1.7 GB venv, most of this image.
RUN uv sync --frozen --no-dev --no-install-project \
    && uv cache clean

COPY alembic.ini ./
COPY alembic/ ./alembic/
COPY app/ ./app/
COPY scripts/ ./scripts/
RUN uv sync --frozen --no-dev

EXPOSE 8080

# Overridden by the worker service, which runs app.workers.run instead.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

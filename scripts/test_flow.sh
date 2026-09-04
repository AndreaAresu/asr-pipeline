#!/usr/bin/env bash
#
# Minimal upload -> poll -> print flow, for a quick manual check against a
# locally running API and worker. `scripts/smoke_test.sh` is the thorough
# one; this is what you run while iterating.
#
# Usage:
#   ASR_API_KEY=<key> bash scripts/test_flow.sh [audio_file]
#
# Env:
#   ASR_API_URL  API base URL (default http://localhost:8000, the port
#                uvicorn uses on the host; compose publishes 8080)

set -euo pipefail

API_URL="${ASR_API_URL:-http://localhost:8000}"
AUDIO="${1:-data/samples/sample.wav}"

if [[ -z "${ASR_API_KEY:-}" ]]; then
  echo "ASR_API_KEY is not set, create one with:" >&2
  echo "  uv run python scripts/create_api_key.py dev" >&2
  exit 1
fi

JOB=$(curl -fsS -X POST "$API_URL/transcribe" \
  -H "X-API-Key: $ASR_API_KEY" \
  -F "audio=@$AUDIO" | jq -r .job_id)
echo "Job: $JOB"

while true; do
  STATUS=$(curl -fsS "$API_URL/jobs/$JOB" -H "X-API-Key: $ASR_API_KEY" | jq -r .status)
  echo "Status: $STATUS"
  [ "$STATUS" = "done" ] && break
  [ "$STATUS" = "failed" ] && {
    curl -fsS "$API_URL/jobs/$JOB" -H "X-API-Key: $ASR_API_KEY" | jq -r .error_message >&2
    exit 1
  }
  sleep 1
done

curl -fsS "$API_URL/jobs/$JOB/result" -H "X-API-Key: $ASR_API_KEY" | jq -r .full_text

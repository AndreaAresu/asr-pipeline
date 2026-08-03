#!/usr/bin/env bash
#
# Latency benchmark for the three user-facing operations.
#
#   ASR_API_KEY=<key> bash scripts/benchmark.sh [audio_file …]
#
# Each measurement is repeated REPEATS times and reported as P50 and P95,
# because a single run of a queue-backed system tells you very little: the
# first call pays for model loading, and the worker's queue wait varies.
#
# Env:
#   ASR_API_URL  API base URL (default http://localhost:8080)
#   ASR_API_KEY  required
#   REPEATS      samples per measurement (default 3)
#
# Paste the resulting table into the README's Performance section, and say
# what hardware produced it — numbers without a machine attached are noise.

set -euo pipefail

cd "$(dirname "$0")/.."

API_URL="${ASR_API_URL:-http://localhost:8080}"
REPEATS="${REPEATS:-3}"
AUDIO_FILES=("$@")
[[ ${#AUDIO_FILES[@]} -eq 0 ]] && AUDIO_FILES=("data/samples/sample.wav")

if [[ -z "${ASR_API_KEY:-}" ]]; then
  echo "ASR_API_KEY is not set" >&2
  exit 1
fi

now() { python3 -c 'import time; print(time.time())'; }

# P50/P95 over the samples on stdin. With few samples P95 is effectively
# the worst observed run — which is the honest reading, not a smoothed one.
percentiles() {
  python3 -c '
import sys
xs = sorted(float(line) for line in sys.stdin if line.strip())
def pct(p):
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
print(f"{pct(0.50):.3f} {pct(0.95):.3f} {len(xs)}")
'
}

row() { printf "| %-34s | %8s | %8s | %7s |\n" "$1" "$2" "$3" "$4"; }

echo "api=$API_URL  repeats=$REPEATS"
echo
row "operation" "P50 (s)" "P95 (s)" "samples"
printf "|%s|%s|%s|%s|\n" "------------------------------------" "----------" "----------" "---------"

# --- transcribe: upload → status done ---------------------------------
LAST_TRANSCRIPT=""
for audio in "${AUDIO_FILES[@]}"; do
  [[ -f "$audio" ]] || { echo "missing: $audio" >&2; continue; }
  seconds=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$audio" 2>/dev/null || echo 0)
  samples=""

  for _ in $(seq 1 "$REPEATS"); do
    start=$(now)
    job=$(curl -fsS -X POST "$API_URL/transcribe" -H "X-API-Key: $ASR_API_KEY" -F "audio=@$audio" | jq -r .job_id)
    while true; do
      status=$(curl -fsS "$API_URL/jobs/$job" | jq -r .status)
      [[ "$status" == "done" ]] && break
      [[ "$status" == "failed" ]] && { echo "job failed for $audio" >&2; exit 1; }
      sleep 1
    done
    samples+="$(python3 -c "print($(now) - $start)")"$'\n'
    LAST_TRANSCRIPT=$(curl -fsS "$API_URL/jobs/$job/result" | jq -r .transcript_id)
  done

  read -r p50 p95 n <<<"$(printf '%s' "$samples" | percentiles)"
  row "transcribe $(printf '%.0f' "$seconds")s audio" "$p50" "$p95" "$n"
done

# --- search ------------------------------------------------------------
samples=""
for _ in $(seq 1 "$((REPEATS * 3))"); do
  start=$(now)
  jq -n '{query: "what is being discussed here", top_k: 5}' \
    | curl -fsS -X POST "$API_URL/search" -H "X-API-Key: $ASR_API_KEY" \
        -H 'Content-Type: application/json' -d @- > /dev/null
  samples+="$(python3 -c "print($(now) - $start)")"$'\n'
done
read -r p50 p95 n <<<"$(printf '%s' "$samples" | percentiles)"
row "search (top_k=5)" "$p50" "$p95" "$n"

# --- summarize: cache miss, then cache hit -----------------------------
if [[ -n "$LAST_TRANSCRIPT" ]]; then
  start=$(now)
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API_URL/summarize/$LAST_TRANSCRIPT" -H "X-API-Key: $ASR_API_KEY")
  miss=$(python3 -c "print($(now) - $start)")

  if [[ "$code" == "200" ]]; then
    row "summarize (cache miss)" "$(printf '%.3f' "$miss")" "-" "1"
    samples=""
    for _ in $(seq 1 "$REPEATS"); do
      start=$(now)
      curl -fsS -X POST "$API_URL/summarize/$LAST_TRANSCRIPT" -H "X-API-Key: $ASR_API_KEY" > /dev/null
      samples+="$(python3 -c "print($(now) - $start)")"$'\n'
    done
    read -r p50 p95 n <<<"$(printf '%s' "$samples" | percentiles)"
    row "summarize (cache hit)" "$p50" "$p95" "$n"
  else
    echo
    echo "summarize skipped: API returned $code (GROQ_API_KEY not set?)"
  fi
fi

echo
echo "Machine: $(uname -sm)"

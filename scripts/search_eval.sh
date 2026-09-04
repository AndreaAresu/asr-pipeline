#!/usr/bin/env bash
#
# Retrieval quality harness for POST /search.
#
# Runs a list of queries against the API and prints, for each, the top-k
# hits with score, source transcript and time range, enough to eyeball
# whether retrieval is sane before trusting it in a demo.
#
# The transcript column is the one that catches a subtle failure: with more
# than one recording indexed, a query can score plausibly while pulling from
# the wrong episode entirely.
#
# Usage:
#   ASR_API_KEY=<key> bash scripts/search_eval.sh [queries_file] [top_k]
#
# queries_file: one query per line, '#' comments and blank lines skipped.
#               Defaults to scripts/eval_queries.txt.
# Env:
#   ASR_API_URL  API base URL (default http://localhost:8000)
#   ASR_API_KEY  required
#
# Record your reading of the output in scripts/search_eval.md.

set -euo pipefail

API_URL="${ASR_API_URL:-http://localhost:8000}"
QUERIES_FILE="${1:-$(dirname "$0")/eval_queries.txt}"
TOP_K="${2:-3}"

if [[ -z "${ASR_API_KEY:-}" ]]; then
  echo "ASR_API_KEY is not set" >&2
  exit 1
fi
if [[ ! -f "$QUERIES_FILE" ]]; then
  echo "queries file not found: $QUERIES_FILE" >&2
  exit 1
fi

echo "api=$API_URL  top_k=$TOP_K  queries=$QUERIES_FILE"
echo

while IFS= read -r query || [[ -n "$query" ]]; do
  [[ -z "${query// }" || "$query" == \#* ]] && continue

  echo "== $query"
  jq -n --arg q "$query" --argjson k "$TOP_K" '{query: $q, top_k: $k}' \
    | curl -sS -X POST "$API_URL/search" \
        -H "X-API-Key: $ASR_API_KEY" \
        -H 'Content-Type: application/json' \
        -d @- \
    | jq -r '
        if (.hits | length) == 0 then "   (no hits, nothing indexed?)"
        else .hits[]
          | "   \(.score * 1000 | round / 1000)  \(.transcript_id[0:8])  [\(.start_sec | floor)s-\(.end_sec | floor)s]  \(.text[0:96])..."
        end'
  echo
done < "$QUERIES_FILE"

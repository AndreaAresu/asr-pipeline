#!/usr/bin/env bash
#
# End-to-end smoke test against the containerised stack.
#
#   bash scripts/smoke_test.sh [audio_file]
#
# Brings the whole stack up from a clean build, drives every endpoint in
# the order a real client would, and tears down. Run it after any change
# that touches the schema, the Dockerfile or the queue wiring, those are
# the failures unit tests do not catch.
#
# Env:
#   KEEP_UP=1   leave the stack running afterwards (for debugging)
#   API_URL     override the API base URL (default http://localhost:8080)

set -euo pipefail

cd "$(dirname "$0")/.."

AUDIO="${1:-data/samples/sample.wav}"
API_URL="${API_URL:-http://localhost:8080}"
HEALTH_RETRIES=60
JOB_RETRIES=120

pass() { echo "  ok   $*"; }
fail() { echo "  FAIL $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

cleanup() {
  if [[ "${KEEP_UP:-0}" == "1" ]]; then
    echo; echo "==> KEEP_UP=1, leaving the stack running"
    return
  fi
  echo; echo "==> tearing down"
  docker compose down
}
trap cleanup EXIT

[[ -f "$AUDIO" ]] || fail "audio file not found: $AUDIO"
[[ -f .env ]] || fail ".env not found, copy .env.example and fill it in"

step "building and starting the stack"
docker compose up -d --build

step "waiting for the API to become healthy"
for i in $(seq 1 $HEALTH_RETRIES); do
  if curl -fsS "$API_URL/health" >/dev/null 2>&1; then
    pass "/health responded after ${i}s"
    break
  fi
  [[ $i -eq $HEALTH_RETRIES ]] && { docker compose logs --tail 50 api; fail "API never became healthy"; }
  sleep 1
done

step "auth is enforced"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API_URL/transcribe" -F "audio=@$AUDIO")
[[ "$code" == "401" ]] || fail "expected 401 without an API key, got $code"
pass "unauthenticated /transcribe returns 401"

step "creating an API key"
# The key is printed once, on the last line of the script's output.
KEY=$(docker compose exec -T api python -m scripts.create_api_key smoke | tail -1 | tr -d '[:space:]')
[[ -n "$KEY" ]] || fail "could not create an API key"
pass "created key ${KEY:0:8}..."

step "uploading $AUDIO"
JOB=$(curl -fsS -X POST "$API_URL/transcribe" -H "X-API-Key: $KEY" -F "audio=@$AUDIO" | jq -r .job_id)
[[ "$JOB" != "null" && -n "$JOB" ]] || fail "no job_id returned"
pass "job $JOB accepted"

step "polling until the job finishes"
for i in $(seq 1 $JOB_RETRIES); do
  STATUS=$(curl -fsS "$API_URL/jobs/$JOB" -H "X-API-Key: $KEY" | jq -r .status)
  case "$STATUS" in
    done)   pass "job completed after ~$((i * 2))s"; break ;;
    failed) curl -s "$API_URL/jobs/$JOB" -H "X-API-Key: $KEY" | jq .; docker compose logs --tail 50 worker; fail "job failed" ;;
  esac
  [[ $i -eq $JOB_RETRIES ]] && { docker compose logs --tail 50 worker; fail "job still $STATUS after $((JOB_RETRIES * 2))s"; }
  sleep 2
done

step "fetching the transcript"
RESULT=$(curl -fsS "$API_URL/jobs/$JOB/result" -H "X-API-Key: $KEY")
TRANSCRIPT_ID=$(echo "$RESULT" | jq -r .transcript_id)
WORDS=$(echo "$RESULT" | jq -r '.full_text | split(" ") | length')
[[ "$TRANSCRIPT_ID" != "null" ]] || fail "no transcript_id in the result"
[[ "$WORDS" -gt 5 ]] || fail "transcript looks empty ($WORDS words)"
pass "transcript $TRANSCRIPT_ID, $WORDS words"

step "searching"
HITS=$(jq -n '{query: "what is being discussed", top_k: 3}' \
  | curl -fsS -X POST "$API_URL/search" -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -d @- \
  | jq '.hits | length')
[[ "$HITS" -gt 0 ]] || fail "search returned no hits, chunks were not indexed"
pass "search returned $HITS hit(s)"

step "reading a transcript window"
WINDOW=$(curl -fsS "$API_URL/transcripts/$TRANSCRIPT_ID/window?start_sec=0&end_sec=30" -H "X-API-Key: $KEY")
CHARS=$(echo "$WINDOW" | jq -r '.text | length')
[[ "$CHARS" -gt 0 ]] || fail "the first 30s of the transcript came back empty"
pass "window 0-30s returned $CHARS characters"

# Over the cap it must refuse and say so, rather than trim in silence.
code=$(curl -s -o /dev/null -w '%{http_code}' \
  "$API_URL/transcripts/$TRANSCRIPT_ID/window?start_sec=0&end_sec=100000" -H "X-API-Key: $KEY")
[[ "$code" == "400" ]] || fail "an oversized window returned $code, expected 400"
pass "an oversized window is refused"

step "listing the curated corpus"
# Only jobs marked as the curated corpus are listed, so on a stack that has
# not restored data/seed/nasa_corpus.sql this is legitimately empty: what is
# under test here is that the endpoint answers with a list at all.
LISTING=$(curl -fsS "$API_URL/transcripts" -H "X-API-Key: $KEY")
echo "$LISTING" | jq -e 'type == "array"' >/dev/null || fail "/transcripts did not return a list"
LISTED=$(echo "$LISTING" | jq 'length')
# The file just uploaded is owned by this run's key, not by the curated
# marker, so it must NOT appear here: that is the property under test.
echo "$LISTING" | jq -e --arg id "$TRANSCRIPT_ID" 'map(.transcript_id) | index($id) | not' >/dev/null \
  || fail "the freshly uploaded transcript appeared in the curated listing"
pass "listing returned $LISTED curated transcript(s), and not this run's upload"

step "calling the MCP endpoint"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API_URL/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' -d '{}')
[[ "$code" == "401" ]] || fail "/mcp without a key returned $code, expected 401"
pass "/mcp refuses an unauthenticated caller"

# One self-contained POST: the protocol envelope goes in params._meta.
TOOLS=$(curl -fsS -X POST "$API_URL/mcp" \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}' \
  | sed -n 's/^data: //p' | jq '.result.tools | length')
[[ "$TOOLS" == "3" ]] || fail "/mcp advertised $TOOLS tools, expected 3"
pass "/mcp advertised $TOOLS tools"

MCP_HITS=$(curl -fsS -X POST "$API_URL/mcp" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}},"name":"search_transcripts","arguments":{"query":"what is being discussed","top_k":3}}}' \
  | sed -n 's/^data: //p' | jq '.result.structuredContent.result | length')
[[ "$MCP_HITS" -gt 0 ]] || fail "search_transcripts returned no hits over MCP"
pass "search_transcripts returned $MCP_HITS hit(s) over MCP, authenticated with a Bearer token"

step "summarizing"
code=$(curl -s -o /tmp/asr_smoke_summary.json -w '%{http_code}' \
  -X POST "$API_URL/summarize/$TRANSCRIPT_ID" -H "X-API-Key: $KEY")
case "$code" in
  200)
    SECTIONS=$(jq '.sections | length' /tmp/asr_smoke_summary.json)
    pass "summary generated with $SECTIONS section(s)"
    # A cached read must not call the LLM again.
    CACHED=$(curl -fsS -X POST "$API_URL/summarize/$TRANSCRIPT_ID" -H "X-API-Key: $KEY" | jq -r .cached)
    [[ "$CACHED" == "true" ]] || fail "second /summarize call was not served from cache"
    pass "second call served from cache"
    ;;
  503)
    echo "  skip summarize: GROQ_API_KEY is not set (endpoint correctly returned 503)"
    ;;
  *)
    cat /tmp/asr_smoke_summary.json >&2
    fail "unexpected status $code from /summarize"
    ;;
esac

echo
echo "==> SMOKE TEST PASSED"

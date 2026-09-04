#!/usr/bin/env bash
#
# Regenerate data/seed/nasa_corpus.sql from the local database.
#
#   bash scripts/dump_seed.sh              # write the seed file
#   bash scripts/dump_seed.sh --check      # regenerate and diff, changing nothing
#
# The seed is a *filtered* dump, and both filters are load-bearing:
#
#   1. Only jobs whose audio_filename matches SEED_PATTERN (default 'nasa-%').
#      The dev database accumulates rows — every smoke test, every trial of
#      the upload tab. A plain `pg_dump -t jobs -t transcripts -t chunks`
#      would quietly publish them, and the corpus would stop matching the
#      one scripts/search_eval.md reports Run 2 against.
#   2. api_key_hash is replaced with the literal 'seed'. The real column
#      holds the SHA-256 of a working API key; it does not belong in a
#      public repo, and 'seed' is a value nothing hashes to, so the seeded
#      jobs are owned by nobody and answer 404 on /jobs to every caller.
#      /search and /summarize do not filter by key, which is what keeps the
#      corpus reachable through exactly the two endpoints the demo uses.
#
# Everything else is ordered deterministically (by job creation time, then
# chunk index) so that re-running this on an unchanged database reproduces
# the committed file byte for byte. `--check` is that assertion; run it
# before trusting a diff you did not expect.
#
# Env:
#   SEED_PATTERN  LIKE pattern for audio_filename (default 'nasa-%')
#   POSTGRES_USER, POSTGRES_DB  read from .env when unset

set -euo pipefail

cd "$(dirname "$0")/.."

OUT="data/seed/nasa_corpus.sql"
PATTERN="${SEED_PATTERN:-nasa-%}"
CHECK=false
[[ "${1:-}" == "--check" ]] && CHECK=true

if [[ -f .env ]]; then
  PGUSER_DEFAULT=$(grep -E '^POSTGRES_USER=' .env | cut -d= -f2- || true)
  PGDB_DEFAULT=$(grep -E '^POSTGRES_DB=' .env | cut -d= -f2- || true)
fi
PGUSER="${POSTGRES_USER:-${PGUSER_DEFAULT:-}}"
PGDB="${POSTGRES_DB:-${PGDB_DEFAULT:-}}"

if [[ -z "$PGUSER" || -z "$PGDB" ]]; then
  echo "POSTGRES_USER / POSTGRES_DB are not set and could not be read from .env" >&2
  exit 1
fi

# `docker compose exec -T` consumes stdin, and this script is often piped or
# run under `bash -s`: without the redirect it eats the rest of the file and
# the output truncates with no error.
psql_q() {
  docker compose exec -T postgres psql -U "$PGUSER" -d "$PGDB" \
    -v ON_ERROR_STOP=1 -qAt -c "SET TIME ZONE 'UTC'; $1" < /dev/null
}

# --- what is about to be dumped, before anything is written ----------------

SELECTED=$(psql_q "SELECT audio_filename || '  ' || status FROM jobs
                   WHERE audio_filename LIKE '$PATTERN' ORDER BY created_at;")
if [[ -z "$SELECTED" ]]; then
  echo "no job matches audio_filename LIKE '$PATTERN'" >&2
  exit 1
fi
if grep -qv ' done$' <<<"$SELECTED"; then
  echo "refusing to dump: a matching job is not 'done' —" >&2
  echo "$SELECTED" >&2
  exit 1
fi

N_CHUNKS=$(psql_q "SELECT count(*) FROM chunks c JOIN transcripts t ON t.id = c.transcript_id
                   JOIN jobs j ON j.id = t.job_id WHERE j.audio_filename LIKE '$PATTERN';")
echo "dumping $(wc -l <<<"$SELECTED" | tr -d ' ') jobs, $N_CHUNKS chunks:" >&2
sed 's/^/  /' <<<"$SELECTED" >&2

FIRST_TRANSCRIPT=$(psql_q "SELECT t.id FROM transcripts t JOIN jobs j ON j.id = t.job_id
                           WHERE j.audio_filename LIKE '$PATTERN'
                           ORDER BY j.created_at LIMIT 1;")

# --- the file ---------------------------------------------------------------
#
# The header travels with the data because it is the only place a reader
# meets the corpus: where it came from, how to load it, and why the owner
# column says 'seed'. Edit it here, not in the generated .sql.

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

cat > "$TMP" <<HEADER
-- Demo corpus: three public-domain NASA podcast episodes, transcribed with
-- small.en and indexed as $N_CHUNKS chunks. This is the data the public demo
-- searches and summarizes, and the data scripts/search_eval.md reports Run 2
-- against — one transcription pass serving both, so the published evaluation
-- describes exactly what a reader can query.
--
-- Regenerate with scripts/dump_seed.sh; do not hand-edit, and do not replace
-- it with a plain pg_dump, which has no way to leave the dev database's other
-- rows behind.
--
--   docker compose exec -T postgres psql -U "\$POSTGRES_USER" \\
--     -d "\$POSTGRES_DB" < data/seed/nasa_corpus.sql
--
-- Restoring it needs the schema to exist, so run it after the API has applied
-- its migrations. .dockerignore excludes data/, so this file reaches a server
-- through git rather than through the image.
--
-- api_key_hash is the literal 'seed' rather than a real key hash: nothing
-- hashes to it, so these jobs are owned by nobody and answer 404 on /jobs to
-- every caller. That is deliberate. /search and /summarize do not filter by
-- key, so the corpus stays reachable through exactly the two endpoints the
-- demo uses, and through no others.
--
-- Sources, all public domain as works of a US government agency (NASA asks
-- only to be credited; its insignia is not public domain and is not used):
--   Gateway: Together to the Moon
--     https://www.nasa.gov/podcasts/houston-we-have-a-podcast/gateway-together-to-the-moon/
--   Astronaut and Microbiologist
--     https://www.nasa.gov/podcasts/houston-we-have-a-podcast/astronaut-and-microbiologist/
--   Apollo 11 to Now
--     https://www.nasa.gov/podcasts/houston-we-have-a-podcast/apollo-11-to-now/

BEGIN;

DO \$\$
BEGIN
  IF EXISTS (SELECT 1 FROM transcripts WHERE id = '$FIRST_TRANSCRIPT') THEN
    RAISE EXCEPTION 'seed already applied: the NASA corpus is present in this database';
  END IF;
END \$\$;

COPY jobs (id, api_key_hash, audio_filename, duration, status, error_message, created_at, started_at, finished_at) FROM stdin;
HEADER

psql_q "COPY (
  SELECT id, 'seed', audio_filename, duration, status, error_message,
         created_at, started_at, finished_at
  FROM jobs WHERE audio_filename LIKE '$PATTERN' ORDER BY created_at
) TO STDOUT;" >> "$TMP"

cat >> "$TMP" <<'MID'
\.

COPY transcripts (id, job_id, full_text, language, word_timestamps) FROM stdin;
MID

psql_q "COPY (
  SELECT t.id, t.job_id, t.full_text, t.language, t.word_timestamps
  FROM transcripts t JOIN jobs j ON j.id = t.job_id
  WHERE j.audio_filename LIKE '$PATTERN' ORDER BY j.created_at
) TO STDOUT;" >> "$TMP"

cat >> "$TMP" <<'MID'
\.

COPY chunks (id, transcript_id, chunk_index, start_sec, end_sec, text, embedding) FROM stdin;
MID

psql_q "COPY (
  SELECT c.id, c.transcript_id, c.chunk_index, c.start_sec, c.end_sec, c.text, c.embedding
  FROM chunks c JOIN transcripts t ON t.id = c.transcript_id
  JOIN jobs j ON j.id = t.job_id
  WHERE j.audio_filename LIKE '$PATTERN'
  ORDER BY j.created_at, c.chunk_index
) TO STDOUT;" >> "$TMP"

cat >> "$TMP" <<'TAIL'
\.

COMMIT;
TAIL

if $CHECK; then
  if diff -q "$OUT" "$TMP" > /dev/null; then
    echo "unchanged: $OUT matches this database" >&2
  else
    echo "DIFFERS from $OUT:" >&2
    diff "$OUT" "$TMP" | head -40 >&2
    exit 1
  fi
else
  mv "$TMP" "$OUT"
  trap - EXIT
  echo "wrote $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)" >&2
fi

"""Create a new API key.

Usage:
    uv run python scripts/create_api_key.py <name> [daily_minute_quota]

Generates a cryptographically secure random key, stores only its SHA-256
hash in the database (alongside the given name), and prints the key in
clear text exactly once. The plaintext is never persisted, so copy it now:
it cannot be recovered later.

The quota is audio *minutes* per rolling 24h window, not a request count
(see `app.core.rate_limit`). The default suits casual use; indexing long
recordings needs it raised explicitly, and a public demo key wants it
lowered:

    create_api_key.py ingest 600    # enough for several hours of audio
    create_api_key.py demo 10       # public key, deliberately small
"""

import hashlib
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.models import ApiKey
from app.db.session import SessionLocal

DEFAULT_QUOTA_MINUTES = 60


def make_key() -> tuple[str, str]:
    """Return a `(key, key_hash)` pair for a fresh API key."""
    key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, key_hash


def main() -> None:
    if not 2 <= len(sys.argv) <= 3:
        print("usage: create_api_key.py <name> [daily_minute_quota]", file=sys.stderr)
        sys.exit(1)

    name = sys.argv[1]
    quota = DEFAULT_QUOTA_MINUTES
    if len(sys.argv) == 3:
        try:
            quota = int(sys.argv[2])
        except ValueError:
            print(f"quota must be a whole number of minutes, got {sys.argv[2]!r}", file=sys.stderr)
            sys.exit(1)
        if quota <= 0:
            print("quota must be greater than zero", file=sys.stderr)
            sys.exit(1)

    key, key_hash = make_key()

    session = SessionLocal()
    try:
        session.add(ApiKey(key_hash=key_hash, name=name, daily_minute_quota=quota))
        session.commit()
    finally:
        session.close()

    print(f"API key created for {name!r} with a quota of {quota} audio minutes per day.")
    print("Store it now, this is the only time it will be shown:")
    print()
    print(f"    {key}")


if __name__ == "__main__":
    main()

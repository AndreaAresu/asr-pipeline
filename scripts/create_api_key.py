"""Create a new API key.

Usage:
    uv run python scripts/create_api_key.py <name>

Generates a cryptographically secure random key, stores only its SHA-256
hash in the database (alongside the given name), and prints the key in
clear text exactly once. The plaintext is never persisted, so copy it now
— it cannot be recovered later.
"""

import hashlib
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.models import ApiKey
from app.db.session import SessionLocal


def make_key() -> tuple[str, str]:
    """Return a `(key, key_hash)` pair for a fresh API key."""
    key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, key_hash


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: create_api_key.py <name>", file=sys.stderr)
        sys.exit(1)

    name = sys.argv[1]
    key, key_hash = make_key()

    session = SessionLocal()
    try:
        session.add(ApiKey(key_hash=key_hash, name=name))
        session.commit()
    finally:
        session.close()

    print(f"API key created for {name!r}.")
    print("Store it now — this is the only time it will be shown:")
    print()
    print(f"    {key}")


if __name__ == "__main__":
    main()

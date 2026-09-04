"""API-key authentication.

Only the SHA-256 hash of a key is ever stored, so a presented key is
hashed and looked up by `key_hash`.

Three pieces, in order of how much they know about HTTP: `resolve_api_key`
is the lookup and nothing else; `credential_from_headers` reads a key out
of either header form the service accepts; `get_api_key` is the FastAPI
dependency that turns a failed lookup into a 401. The MCP route is not a
FastAPI route and cannot use the dependency, so it composes the first two
itself (`app/mcp/asgi.py`) - which is why the lookup does not raise
`HTTPException` on its own.
"""

import hashlib
from collections.abc import Mapping

from fastapi import Header, HTTPException

from app.db.models import ApiKey
from app.db.session import SessionLocal


def resolve_api_key(presented: str | None) -> ApiKey | None:
    """Return the stored row for a presented key, or None if there is none.

    Args:
        presented: The plaintext key a caller sent, in whatever way their
            protocol carries it.

    Returns:
        The matching `ApiKey` row, or None when the key is absent, empty
        or unknown. The caller decides what a None means: a 401 over HTTP,
        a tool error over MCP.
    """
    if not presented:
        return None

    key_hash = hashlib.sha256(presented.encode()).hexdigest()

    db = SessionLocal()
    try:
        return db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
    finally:
        db.close()


def credential_from_headers(headers: Mapping[str, str]) -> str | None:
    """Read a key from `Authorization: Bearer` or from `X-API-Key`.

    Both are accepted on `/mcp`. The MCP specification describes only the
    Bearer form for HTTP, but authorization there is OPTIONAL and no other
    scheme is forbidden: `X-API-Key` is undescribed rather than
    non-conformant, and it is what the rest of this API already uses.
    Accepting both costs these few lines and covers the clients that can
    only compose a Bearer header.

    A key in the query string is deliberately *not* read: the
    specification forbids it, and URLs end up in proxy and browser logs.

    Args:
        headers: Request headers, keyed in lower case or by a
            case-insensitive mapping.

    Returns:
        The presented key, or None if neither header carries one.
    """
    authorization = headers.get("authorization") or ""
    if authorization.lower().startswith("bearer "):
        bearer = authorization[len("bearer ") :].strip()
        if bearer:
            return bearer
    return headers.get("x-api-key") or None


def get_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> ApiKey:
    """Resolve and validate the caller's API key.

    Args:
        x_api_key: Value of the `X-API-Key` request header.

    Returns:
        The matching `ApiKey` row.

    Raises:
        HTTPException: 401 if the header is missing or does not match any
            stored key.
    """
    if not x_api_key:
        raise HTTPException(401, "missing API key")

    api_key = resolve_api_key(x_api_key)
    if api_key is None:
        raise HTTPException(401, "invalid API key")

    return api_key

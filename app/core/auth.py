"""API-key authentication for the HTTP layer.

Exposes `get_api_key`, a FastAPI dependency that resolves the caller's
`X-API-Key` header to an `ApiKey` row. Only the SHA-256 hash of the key
is ever stored, so the header is hashed and looked up by `key_hash`.
"""

import hashlib

from fastapi import Header, HTTPException

from app.db.models import ApiKey
from app.db.session import SessionLocal


def get_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> ApiKey:
    """Resolve and validate the caller's API key.

    Args:
        x_api_key: Value of the `X-API-Key` request header.

    Returns:
        The matching `ApiKey` row.

    Raises:
        HTTPException: 401 if the header does not match any stored key.
    """
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()

    db = SessionLocal()
    try:
        api_key = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
    finally:
        db.close()

    if api_key is None:
        raise HTTPException(401, "invalid API key")

    return api_key

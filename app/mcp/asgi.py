"""The gate in front of the MCP route.

`Depends(get_api_key)` is FastAPI machinery that runs inside an `APIRoute`,
and the MCP endpoint is not one: Streamable HTTP writes its response
through the raw ASGI `send`, so it is a Starlette route with an ASGI app
behind it and no dependency ever runs. This wrapper is that dependency's
replacement, at the same height (one route, not a global middleware), and
it answers 401 *before* the MCP handshake, so an anonymous caller does not
even learn which tools exist.

It also stamps the route label the metrics middleware reads. Both are
about the route rather than about MCP, which is why they live here and not
in `server.py`.
"""

import json

from starlette.types import Receive, Scope, Send

from app.core.auth import credential_from_headers, resolve_api_key

MCP_PATH = "/mcp"

_UNAUTHORIZED = json.dumps({"detail": "missing or invalid API key"}).encode()


class _McpRoute:
    """Carries `path` so the metrics middleware can label MCP requests.

    `scope["route"]` is written by FastAPI, and only for its own
    `APIRoute`s; Starlette's `Route` and `Mount` never set it. Without
    this, every MCP call would be counted under `path="unmatched"`,
    together with the genuine 404s - and "how often did a model call the
    tools" is the most useful number this feature produces.
    """

    path = MCP_PATH


class ApiKeyGate:
    """Reject unauthenticated callers with 401 before the MCP handshake.

    An MCP client knows nothing about this API's dependencies, but it does
    know a 401: it is a transport error, reported as such instead of being
    retried as a protocol failure. The challenge header follows the shape
    the SDK itself uses, minus the `resource_metadata` pointer, which
    would send the client into an OAuth discovery that does not exist
    here.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope["route"] = _McpRoute()

        headers = {name.decode(): value.decode() for name, value in scope.get("headers", [])}
        if resolve_api_key(credential_from_headers(headers)) is None:
            await self._unauthorized(send)
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _unauthorized(send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_UNAUTHORIZED)).encode()),
                    (b"www-authenticate", b'Bearer realm="asr-pipeline"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _UNAUTHORIZED})

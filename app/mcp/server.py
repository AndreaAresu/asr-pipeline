"""The MCP server: the retrieval, exposed to a model instead of a person.

Three tools, because one would not be enough to be useful. A lone search
tool is a POST with a JSON schema around it: the model gets five passages
identified by a UUID, cannot know what the index holds, and cannot widen a
hit into text worth quoting. `list_transcripts` answers "what is in this
corpus?", `search_transcripts` answers "where is X discussed?", and
`fetch_transcript_window` answers "what exactly is said around that
point?".

Everything these tools do goes through `app.core.retrieval`, the same
functions the HTTP routes call, so the two front ends cannot answer the
same question differently. What is written here and nowhere else are the
**descriptions**: they are the documentation a model actually reads, and
they are where the awkward facts have to be stated - that results are
passages and not whole transcripts, that search covers the entire index
while the listing shows only the curated corpus, and what the score means.

Authentication happens twice, for two different purposes: `ApiKeyGate`
(`app/mcp/asgi.py`) refuses an unauthenticated caller before the MCP
handshake, and each tool resolves the key again to log which one is
calling. Headers are client-supplied input, never an identity assertion,
so the row is looked up rather than trusted.
"""

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.core.auth import credential_from_headers, resolve_api_key
from app.core.logging import logger
from app.core.retrieval import (
    TranscriptSummary,
    list_curated_transcripts,
)
from app.db.models import ApiKey
from app.db.session import SessionLocal

# What the curated corpus actually is, stated once and quoted into the tool
# descriptions. A model that does not know what the corpus covers cannot
# tell "this question is out of scope" from "retrieval failed".
CORPUS = (
    "The curated corpus is three public-domain episodes of NASA's "
    "'Houston We Have a Podcast' (Apollo 11, the Gateway lunar station, and a NASA "
    "microbiologist), about 19-24 minutes each, in English, indexed as 109 passages."
)

# Chunking targets 45-second windows with 10 seconds of overlap; measured on
# the deployed corpus the passages run 12-72 seconds, median 47.
PASSAGE_LENGTH = "about 45 seconds of speech"

mcp = MCPServer(
    "asr-pipeline",
    version="0.1.0",
    instructions=(
        "Semantic search over audio transcripts produced by this pipeline. "
        f"{CORPUS} Start with list_transcripts to see what is available, use "
        "search_transcripts to find where something is discussed, and "
        "fetch_transcript_window to read the surrounding text before quoting it. "
        "Note the asymmetry: search covers the whole index, including recordings "
        "uploaded by visitors to the public demo, while list_transcripts shows only "
        "the curated corpus."
    ),
)


def _calling_key(headers: dict[str, str]) -> ApiKey:
    """Resolve the key behind a tool call, for logging and future quota.

    The gate has already refused anonymous callers, so a failure here means
    the key was revoked between the handshake and the call.
    """
    api_key = resolve_api_key(credential_from_headers(headers))
    if api_key is None:
        raise ToolError("missing or invalid API key")
    return api_key


def _headers(ctx: Context) -> dict[str, str]:
    """Copy the request headers out of the context as a plain dict."""
    return {name.lower(): value for name, value in (ctx.headers or {}).items()}


LIST_DESCRIPTION = (
    "List the transcripts in the curated corpus: what this index can be asked about. "
    f"{CORPUS} "
    "Each entry gives the transcript_id to pass to the other tools, the source audio "
    "filename to cite, the duration in seconds and as mm:ss, the language, and how many "
    "indexed passages it holds. "
    "This lists ONLY the curated corpus, while search_transcripts searches the whole "
    "index, which also contains recordings uploaded by visitors to the public demo: a "
    "search hit whose filename is not listed here came from a visitor, not from the corpus."
)


@mcp.tool(description=LIST_DESCRIPTION)
async def list_transcripts(ctx: Context) -> list[TranscriptSummary]:
    """List the curated corpus. See `LIST_DESCRIPTION` for the model-facing text."""
    return await run_in_threadpool(_list_transcripts, _headers(ctx))


def _list_transcripts(headers: dict[str, str]) -> list[TranscriptSummary]:
    """The blocking half of `list_transcripts`, run off the event loop."""
    api_key = _calling_key(headers)

    db = SessionLocal()
    try:
        transcripts = list_curated_transcripts(db)
    finally:
        db.close()

    logger.info("mcp.transcripts.listed", api_key_hash=api_key.key_hash, transcripts=len(transcripts))
    return transcripts


def _transport_security() -> TransportSecuritySettings:
    """Host allowlist for the anti DNS-rebinding check.

    Left alone, the SDK allows only localhost, and Caddy does not rewrite
    the `Host` header: behind the reverse proxy the app would see the
    public name and answer **421 to every request**, after passing every
    local test. Set `MCP_ALLOWED_HOSTS` on a deployment that sits behind a
    proxy; when it is empty the check is turned off explicitly rather than
    left to fail in a way nobody can read.
    """
    allowed = [host.strip() for host in settings.mcp_allowed_hosts.split(",") if host.strip()]
    if not allowed:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return TransportSecuritySettings(
        allowed_hosts=[form for host in allowed for form in (host, f"{host}:*")],
        allowed_origins=[],
    )


# Built at import time on purpose: `mcp.session_manager` does not exist
# until `streamable_http_app()` has been called, and `app/main.py` needs it
# in its lifespan.
#
# `stateless_http=True`: on the current protocol a request is self-contained
# and there is no session to keep, and none of these three tools uses what a
# session would buy (sampling, elicitation, resumability). For a client still
# speaking an older protocol it means a restart of this process is invisible
# instead of invalidating their session mid-conversation.
mcp_asgi_app = mcp.streamable_http_app(
    streamable_http_path="/mcp",
    stateless_http=True,
    transport_security=_transport_security(),
)

"""Tests for the MCP route: the gate in front of it, and the tools behind it.

The route is not a FastAPI route, so none of what protects the rest of the
API applies to it by construction: `Depends(get_api_key)` never runs, and
`scope["route"]` - which the metrics middleware labels on - is never set.
Both are covered here, along with the two mounting mistakes that would
only show up somewhere else: a 307 on `/mcp`, and API 404s turning into
plain text.

Nothing here needs Postgres: the key lookup is replaced, and `tools/list`
never reaches the database. Calling a tool for real belongs in
`scripts/smoke_test.sh`.
"""

import json

import pytest
from fastapi.testclient import TestClient
from mcp_types import (
    CLIENT_CAPABILITIES_META_KEY,
    LATEST_PROTOCOL_VERSION,
    PROTOCOL_VERSION_META_KEY,
)

import app.mcp.asgi as asgi
import app.mcp.server as server
from app.db.models import ApiKey
from app.main import app
from app.mcp.server import mcp

DEMO_KEY = "an-issued-key"
DEMO_ROW = ApiKey(key_hash="hash-mcp", name="mcp-demo", daily_minute_quota=10)

# A modern MCP request is one self-contained POST, and it has to carry the
# protocol envelope in `params._meta` or the server answers -32602.
ENVELOPE = {
    "_meta": {
        PROTOCOL_VERSION_META_KEY: LATEST_PROTOCOL_VERSION,
        CLIENT_CAPABILITIES_META_KEY: {},
    }
}
TOOLS_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": ENVELOPE}
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module", autouse=True)
def an_issued_key():
    """Stand in for the api_keys table: one key exists, everything else does not."""

    def lookup(presented):
        return DEMO_ROW if presented == DEMO_KEY else None

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(asgi, "resolve_api_key", lookup)
        patch.setattr(server, "resolve_api_key", lookup)
        yield


@pytest.fixture(scope="module")
def client():
    """A client whose lifespan has run, so the session manager is alive.

    `TestClient(app)` on its own never runs the lifespan, and an MCP
    message would then die on "Task group is not initialized". Module
    scoped because the session manager refuses to be started twice in a
    process, which is also true of the real server.
    """
    with TestClient(app) as started:
        yield started


def test_a_caller_without_a_key_is_refused_before_the_handshake():
    """Anonymous callers must not even learn which tools exist."""
    response = TestClient(app).post("/mcp", headers=MCP_HEADERS, json=TOOLS_LIST)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer realm="asr-pipeline"'
    assert json.loads(response.content) == {"detail": "missing or invalid API key"}


def test_an_unknown_key_is_refused():
    response = TestClient(app).post("/mcp", headers={**MCP_HEADERS, "X-API-Key": "nope"}, json=TOOLS_LIST)

    assert response.status_code == 401


def test_a_key_in_the_query_string_is_not_a_credential():
    """The MCP specification forbids it, and URLs end up in logs."""
    response = TestClient(app).post(f"/mcp?api_key={DEMO_KEY}", headers=MCP_HEADERS, json=TOOLS_LIST)

    assert response.status_code == 401


@pytest.mark.parametrize(
    "credential",
    [
        pytest.param({"X-API-Key": DEMO_KEY}, id="x-api-key"),
        pytest.param({"Authorization": f"Bearer {DEMO_KEY}"}, id="bearer"),
    ],
)
def test_both_accepted_header_forms_get_through(client, credential):
    """Bearer is what the specification describes; X-API-Key is what the rest of this API uses."""
    response = client.post("/mcp", headers={**MCP_HEADERS, **credential}, json=TOOLS_LIST)

    assert response.status_code == 200
    assert "list_transcripts" in response.text


def test_an_mcp_call_is_counted_under_its_own_route(client):
    """Otherwise it lands in `unmatched`, together with the genuine 404s.

    `scope["route"]` is FastAPI's doing and only for its own routes, so the
    gate has to stamp it. This is the regression that would go unnoticed
    longest: the endpoint keeps working, the metric quietly stops meaning
    anything.
    """
    client.post("/mcp", headers={**MCP_HEADERS, "X-API-Key": DEMO_KEY}, json=TOOLS_LIST)

    metrics = client.get("/metrics").text

    assert 'path="/mcp"' in metrics


def test_the_route_answers_on_the_exact_path(client):
    """No 307 to `/mcp/`: a client that does not follow redirects on POST would stop there."""
    response = client.post("/mcp", headers={**MCP_HEADERS, "X-API-Key": DEMO_KEY}, json=TOOLS_LIST)

    assert response.status_code == 200
    assert not response.history


def test_an_unrouted_path_still_answers_json(client):
    """This is how you notice someone has gone back to mounting on the root."""
    response = client.get("/definitely-not-a-route")

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": "Not Found"}


@pytest.mark.anyio
async def test_the_tools_are_registered_under_the_expected_names():
    """The equivalent, for tools, of resolving the worker's job string.

    A renamed tool or a broken schema fails nowhere in this repo: it fails
    inside somebody else's client, at the far end of a demo.
    """
    assert {tool.name for tool in await mcp.list_tools()} == {"list_transcripts", "search_transcripts"}


@pytest.mark.anyio
async def test_the_listing_tool_takes_no_arguments():
    (tool,) = [tool for tool in await mcp.list_tools() if tool.name == "list_transcripts"]

    assert tool.input_schema["properties"] == {}


@pytest.mark.anyio
async def test_the_descriptions_carry_the_facts_a_model_cannot_infer():
    """The tool description is the only documentation the model reads.

    It has to say what the corpus holds, and that listing and searching do
    not cover the same rows - otherwise the model presents a stranger's
    upload as part of the corpus.
    """
    (tool,) = [tool for tool in await mcp.list_tools() if tool.name == "list_transcripts"]

    assert "NASA" in tool.description
    assert "ONLY the curated corpus" in tool.description
    assert "search_transcripts searches the whole index" in tool.description


async def a_tool(name: str):
    (tool,) = [tool for tool in await mcp.list_tools() if tool.name == name]
    return tool


@pytest.mark.anyio
async def test_the_search_tool_caps_how_much_it_can_be_asked_for():
    """A client truncating an oversized tool response is an illegible failure."""
    schema = (await a_tool("search_transcripts")).input_schema

    assert schema["properties"]["top_k"]["maximum"] == server.MAX_TOOL_HITS
    assert schema["required"] == ["query"]


@pytest.mark.anyio
async def test_the_search_description_says_what_a_result_actually_is():
    """Passages, the whole index, and a weak hit marked rather than dropped."""
    description = (await a_tool("search_transcripts")).description

    assert "PASSAGES" in description
    assert "WHOLE index" in description
    assert "below_threshold" in description


@pytest.mark.anyio
async def test_a_hit_is_described_well_enough_to_be_quoted():
    """The output schema is the other half of what the model reads."""
    schema = (await a_tool("search_transcripts")).output_schema

    hit = schema["$defs"]["SearchHit"]["properties"]
    assert {"audio_filename", "start", "end", "below_threshold"} <= set(hit)

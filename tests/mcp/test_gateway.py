import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.mcp.registry import MCPRegistry
from app.mcp.gateway import MCPGateway


@pytest.mark.asyncio
async def test_invoke_no_replicas():
    registry = MCPRegistry()
    gateway = MCPGateway(registry)

    with pytest.raises(RuntimeError, match="No healthy replicas"):
        await gateway.invoke("nonexistent", {})


@pytest.mark.asyncio
async def test_invoke_failover_to_next_replica():
    registry = MCPRegistry()
    registry.register("search", "replica_1", "http://localhost:9001")
    registry.register("search", "replica_2", "http://localhost:9002")
    gateway = MCPGateway(registry)

    # Mock the client to fail for replica_1, succeed for replica_2
    mock_resp_success = MagicMock()
    mock_resp_success.json.return_value = {"result": "from_replica_2"}
    mock_resp_success.raise_for_status = MagicMock()

    mock_resp_fail = MagicMock()
    mock_resp_fail.raise_for_status.side_effect = Exception("connection refused")

    call_count = 0

    async def mock_post(url, json=None):
        nonlocal call_count
        call_count += 1
        if "9001" in url:
            return mock_resp_fail
        return mock_resp_success

    mock_client = AsyncMock()
    mock_client.post = mock_post

    gateway._clients["http://localhost:9001"] = mock_client
    gateway._clients["http://localhost:9002"] = mock_client

    result = await gateway.invoke("search", {"query": "test"})
    assert result["result"] == "from_replica_2"


@pytest.mark.asyncio
async def test_invoke_all_replicas_fail():
    registry = MCPRegistry()
    registry.register("search", "replica_1", "http://localhost:9001")
    registry.register("search", "replica_2", "http://localhost:9002")
    gateway = MCPGateway(registry)

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("connection refused")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    gateway._clients["http://localhost:9001"] = mock_client
    gateway._clients["http://localhost:9002"] = mock_client

    with pytest.raises(RuntimeError, match="All replicas"):
        await gateway.invoke("search", {"query": "test"})


@pytest.mark.asyncio
async def test_invoke_first_replica_succeeds():
    registry = MCPRegistry()
    registry.register("search", "replica_1", "http://localhost:9001")
    gateway = MCPGateway(registry)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": "success"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    gateway._clients["http://localhost:9001"] = mock_client

    result = await gateway.invoke("search", {"query": "test"})
    assert result["result"] == "success"

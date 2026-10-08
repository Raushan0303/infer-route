"""MCP gateway: real MCP protocol (official SDK) + per-replica failover.

Replicas are in-process MCP servers built with the SDK's low-level Server;
the gateway talks to them with the SDK Client, i.e. real initialize,
tools/list and tools/call JSON-RPC messages.
"""
import mcp_types as types
import pytest
from mcp import Client
from mcp.server.lowlevel import Server

from app.mcp.gateway import MCPGateway, MCPToolError
from app.mcp.registry import MCPRegistry


def search_server(tag: str) -> Server:
    async def on_list_tools(ctx, params):
        return types.ListToolsResult(tools=[types.Tool(
            name="search", description="search docs",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})])

    async def on_call_tool(ctx, params):
        if not params.arguments.get("query"):
            return types.CallToolResult(is_error=True, content=[types.TextContent(text="query must not be empty")])
        return types.CallToolResult(content=[types.TextContent(text=tag)],
                                    structured_content={"result": f"{tag}:{params.arguments['query']}"})

    return Server(f"search-{tag}", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


def factory(servers: dict, calls: list):
    def make(endpoint):
        calls.append(endpoint)
        target = servers[endpoint]
        if isinstance(target, Exception):
            raise target
        return Client(target)
    return make


@pytest.mark.asyncio
async def test_invoke_no_replicas():
    gateway = MCPGateway(MCPRegistry())
    with pytest.raises(RuntimeError, match="No healthy replicas"):
        await gateway.invoke("nonexistent", {})


@pytest.mark.asyncio
async def test_discover_registers_tools_from_tools_list():
    registry = MCPRegistry()
    gateway = MCPGateway(registry, factory({"mcp://a": search_server("a")}, []))
    assert await gateway.discover("mcp://a", "replica_1") == ["search"]
    assert registry.list_tools() == ["search"]


@pytest.mark.asyncio
async def test_invoke_first_replica_succeeds():
    registry = MCPRegistry()
    registry.register("search", "replica_1", "mcp://a")
    gateway = MCPGateway(registry, factory({"mcp://a": search_server("a")}, []))
    assert await gateway.invoke("search", {"query": "x"}) == {"result": "a:x"}


@pytest.mark.asyncio
async def test_invoke_failover_to_next_replica():
    registry = MCPRegistry()
    registry.register("search", "replica_1", "mcp://down")
    registry.register("search", "replica_2", "mcp://b")
    calls = []
    gateway = MCPGateway(registry, factory(
        {"mcp://down": ConnectionError("connection refused"), "mcp://b": search_server("b")}, calls))
    assert await gateway.invoke("search", {"query": "x"}) == {"result": "b:x"}
    assert calls == ["mcp://down", "mcp://b"]


@pytest.mark.asyncio
async def test_invoke_all_replicas_fail():
    registry = MCPRegistry()
    registry.register("search", "replica_1", "mcp://d1")
    registry.register("search", "replica_2", "mcp://d2")
    gateway = MCPGateway(registry, factory(
        {"mcp://d1": ConnectionError("refused"), "mcp://d2": ConnectionError("refused")}, []))
    with pytest.raises(RuntimeError, match="All replicas"):
        await gateway.invoke("search", {"query": "x"})


@pytest.mark.asyncio
async def test_tool_error_does_not_fail_over_or_trip_the_breaker():
    registry = MCPRegistry()
    registry.register("search", "replica_1", "mcp://a")
    registry.register("search", "replica_2", "mcp://b")
    calls = []
    gateway = MCPGateway(registry, factory({"mcp://a": search_server("a"), "mcp://b": search_server("b")}, calls))
    with pytest.raises(MCPToolError, match="must not be empty"):
        await gateway.invoke("search", {"query": ""})
    assert calls == ["mcp://a"]  # bad arguments are not the replica's fault
    assert len(registry.get_healthy("search")) == 2

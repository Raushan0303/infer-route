import pytest
from app.mcp.registry import MCPRegistry


def test_register_and_select():
    registry = MCPRegistry()
    registry.register("search", "replica_1", "http://localhost:9001")
    registry.register("search", "replica_2", "http://localhost:9002")

    replica = registry.select("search")
    assert replica is not None
    assert replica.tool_name == "search"


def test_round_robin_selection():
    registry = MCPRegistry()
    registry.register("search", "replica_1", "http://localhost:9001")
    registry.register("search", "replica_2", "http://localhost:9002")

    selections = [registry.select("search").replica_id for _ in range(4)]
    # Round-robin: should alternate
    assert selections[0] != selections[1]
    assert selections[0] == selections[2]


def test_mark_unhealthy_skipped():
    registry = MCPRegistry()
    registry.register("search", "replica_1", "http://localhost:9001")
    registry.register("search", "replica_2", "http://localhost:9002")

    registry.mark_unhealthy("search", "replica_1")

    healthy = registry.get_healthy("search")
    assert len(healthy) == 1
    assert healthy[0].replica_id == "replica_2"


def test_mark_healthy_re_added():
    registry = MCPRegistry()
    registry.register("search", "replica_1", "http://localhost:9001")
    registry.mark_unhealthy("search", "replica_1")
    assert len(registry.get_healthy("search")) == 0

    registry.mark_healthy("search", "replica_1")
    assert len(registry.get_healthy("search")) == 1


def test_no_healthy_replicas():
    registry = MCPRegistry()
    assert registry.select("nonexistent") is None


def test_get_status():
    registry = MCPRegistry()
    registry.register("search", "replica_1", "http://localhost:9001")
    registry.register("calc", "replica_1", "http://localhost:9003")

    status = registry.get_status()
    assert len(status) == 2
    tools = [s["tool_name"] for s in status]
    assert "search" in tools
    assert "calc" in tools


def test_list_tools():
    registry = MCPRegistry()
    registry.register("search", "r1", "http://localhost:9001")
    registry.register("calc", "r1", "http://localhost:9003")

    tools = registry.list_tools()
    assert set(tools) == {"search", "calc"}

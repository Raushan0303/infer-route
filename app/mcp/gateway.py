"""MCP gateway: load-balanced, circuit-broken tools/call to MCP server replicas.

Speaks the real Model Context Protocol through the official SDK client
(JSON-RPC over Streamable HTTP): replicas are MCP servers, discovery is
`tools/list`, invocation is `tools/call`. Per-replica circuit breakers and
failover are InferRoute's addition on top of the protocol.

Error semantics:
  * transport / server failure  → breaker records a failure, fail over
  * tool error (is_error=True)  → MCPToolError to the caller, no failover
    and no breaker penalty: the replica is healthy, the arguments were bad
"""
import logging
from collections.abc import Callable

from mcp import Client

from app.core.config import settings
from app.mcp.registry import MCPRegistry
from app.observability.metrics import mcp_tool_invoke_total
from app.resilience.circuit_breaker import CircuitBreaker

logger = logging.getLogger("inferroute")


class MCPToolError(Exception):
    """The tool ran and reported an error (CallToolResult.is_error)."""


def _payload(result) -> dict:
    if result.structured_content is not None:
        return result.structured_content
    return {"content": [getattr(c, "text", None) for c in result.content]}


class MCPGateway:

    def __init__(self, registry: MCPRegistry, client_factory: Callable[[str], Client] = Client):
        self._registry = registry
        self._breakers: dict[str, CircuitBreaker] = {}
        # endpoint → MCP client (a URL means Streamable HTTP). Injectable so
        # tests can connect to in-process MCP servers.
        self._client_factory = client_factory

    def _breaker_key(self, tool_name: str, replica_id: str) -> str:
        return f"{tool_name}:{replica_id}"

    def _get_breaker(self, tool_name: str, replica_id: str) -> CircuitBreaker:
        key = self._breaker_key(tool_name, replica_id)
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(
                provider_id=key,
                failure_threshold=settings.mcp_circuit_breaker_threshold,
                cooldown_seconds=settings.mcp_circuit_breaker_cooldown,
            )
        return self._breakers[key]

    async def discover(self, endpoint: str, replica_id: str) -> list[str]:
        """tools/list on an MCP server; register it as a replica of each tool."""
        async with self._client_factory(endpoint) as client:
            result = await client.list_tools()
        names = [t.name for t in result.tools]
        for name in names:
            self._registry.register(name, replica_id, endpoint)
        logger.info("MCP discover endpoint=%s replica=%s tools=%s", endpoint, replica_id, names)
        return names

    async def invoke(self, tool_name: str, arguments: dict) -> dict:
        tried = []
        last_error = None

        replicas = self._registry.get_healthy(tool_name)
        if not replicas:
            mcp_tool_invoke_total.labels(tool_name=tool_name, status="no_replicas").inc()
            raise RuntimeError(f"No healthy replicas for tool '{tool_name}'")

        for replica in replicas:
            breaker = self._get_breaker(tool_name, replica.replica_id)
            if not await breaker.can_execute():
                logger.debug("Skipping replica %s — breaker open", replica.replica_id)
                continue

            tried.append(replica)
            self._registry.increment_in_flight(tool_name, replica.replica_id)
            try:
                async with self._client_factory(replica.endpoint) as client:
                    result = await client.call_tool(tool_name, arguments)
            except Exception as e:
                last_error = e
                await breaker.record_failure()
                self._registry.mark_unhealthy(tool_name, replica.replica_id)
                mcp_tool_invoke_total.labels(tool_name=tool_name, status="error").inc()
                logger.warning("MCP replica %s failed: %s. Trying next.", replica.replica_id, e)
                continue
            finally:
                self._registry.decrement_in_flight(tool_name, replica.replica_id)

            await breaker.record_success()
            if result.is_error:
                mcp_tool_invoke_total.labels(tool_name=tool_name, status="tool_error").inc()
                raise MCPToolError(" ".join(getattr(c, "text", "") for c in result.content))
            mcp_tool_invoke_total.labels(tool_name=tool_name, status="success").inc()
            return _payload(result)

        raise RuntimeError(
            f"All replicas for tool '{tool_name}' failed. Tried: {[r.replica_id for r in tried]}. Last error: {last_error}"
        )

    async def close(self):
        pass

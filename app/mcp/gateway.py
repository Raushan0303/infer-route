import asyncio
import logging
import httpx
from app.mcp.registry import MCPRegistry
from app.resilience.circuit_breaker import CircuitBreaker
from app.core.config import settings
from app.observability.metrics import mcp_tool_invoke_total

logger = logging.getLogger("inferroute")


class MCPGateway:

    def __init__(self, registry: MCPRegistry):
        self._registry = registry
        self._breakers: dict[str, CircuitBreaker] = {}
        self._clients: dict[str, httpx.AsyncClient] = {}

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

    def _get_client(self, endpoint: str) -> httpx.AsyncClient:
        if endpoint not in self._clients:
            self._clients[endpoint] = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
            )
        return self._clients[endpoint]

    async def invoke(self, tool_name: str, payload: dict) -> dict:
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
                client = self._get_client(replica.endpoint)
                resp = await client.post(
                    f"{replica.endpoint}/invoke",
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()

                await breaker.record_success()
                self._registry.decrement_in_flight(tool_name, replica.replica_id)
                mcp_tool_invoke_total.labels(tool_name=tool_name, status="success").inc()
                return result

            except Exception as e:
                last_error = e
                await breaker.record_failure()
                self._registry.decrement_in_flight(tool_name, replica.replica_id)
                self._registry.mark_unhealthy(tool_name, replica.replica_id)
                mcp_tool_invoke_total.labels(tool_name=tool_name, status="error").inc()
                logger.warning("MCP replica %s failed: %s. Trying next.", replica.replica_id, e)
                continue

        raise RuntimeError(
            f"All replicas for tool '{tool_name}' failed. Tried: {[r.replica_id for r in tried]}. Last error: {last_error}"
        )

    async def close(self):
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

import logging
from app.gateway.models import InferRouteRequest, InferRouteResponse
from app.routing.provider_registry import ProviderRegistry
from app.routing.config_registry import ConfigRegistry, RoutingConfig
from app.routing.dynamic_router import DynamicRouter
from app.observability.metrics import provider_in_flight
from app.resilience.circuit_breaker import CircuitBreaker
from app.resilience.retry_budget import RetryBudget
from app.resilience.failover import FailoverChain
from app.core.config import settings

logger = logging.getLogger("inferroute")


class RoutingService:

    def __init__(self, registry: ProviderRegistry, config_registry: ConfigRegistry = None):
        self._registry = registry
        self._config_registry = config_registry
        self._dynamic_router = DynamicRouter(registry)
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._retry_budget = RetryBudget(max_concurrent=settings.retry_budget_max_concurrent)
        self._failover_chain = FailoverChain(self._circuit_breakers, self._retry_budget)

    def register_circuit_breaker(self, provider_id: str, breaker: CircuitBreaker):
        self._circuit_breakers[provider_id] = breaker

    async def route(self, request: InferRouteRequest, extra_headers: dict = None) -> InferRouteResponse:
        # Extract tenant_id and config_id from headers
        tenant_id = (extra_headers or {}).get("x-tenant-id", "default")
        config_id = (extra_headers or {}).get("x-config-id")

        # Mode 1: Config-based routing (Portkey-style, dynamic scoring)
        if config_id and self._config_registry:
            config = await self._config_registry.get(tenant_id, config_id)
            if config:
                ordered = self._dynamic_router.select(config, circuit_breakers=self._circuit_breakers)
                if ordered:
                    logger.debug("Config-based routing: tenant=%s config=%s → %s (scored %d providers)",
                                 tenant_id, config_id, ordered[0].provider_id, len(ordered))
                    return await self._execute_with_failover(ordered, request, extra_headers)
                logger.warning("Config %s matched no providers, falling back to strategy", config_id)

        # Mode 2: Strategy-based routing (existing behavior — heuristic/auto)
        healthy = self._registry.get_healthy()
        if not healthy:
            raise RuntimeError("No healthy providers available")

        selected = await self._registry.select(request=request)
        if selected is None:
            selected = healthy[0]

        ordered = [selected] + [p for p in healthy if p.provider_id != selected.provider_id]
        return await self._execute_with_failover(ordered, request, extra_headers)

    async def _execute_with_failover(self, ordered, request, extra_headers):
        """Execute request with failover chain across ordered providers."""
        async def action(provider):
            self._registry.increment_in_flight(provider.provider_id)
            provider_in_flight.labels(provider=provider.provider_id).inc()
            try:
                response = await provider.adapter.complete(request, extra_headers=extra_headers)
                self._registry.update_latency(provider.provider_id, response.latency_ms)
                return response
            finally:
                self._registry.decrement_in_flight(provider.provider_id)
                provider_in_flight.labels(provider=provider.provider_id).dec()

        return await self._failover_chain.execute(ordered, action)

    async def stream_route(self, request: InferRouteRequest, extra_headers: dict = None):
        """Route a streaming request. Yields SSE chunks from the selected provider."""
        tenant_id = (extra_headers or {}).get("x-tenant-id", "default")
        config_id = (extra_headers or {}).get("x-config-id")

        selected = None

        # Mode 1: Config-based routing
        if config_id and self._config_registry:
            config = await self._config_registry.get(tenant_id, config_id)
            if config:
                ordered = self._dynamic_router.select(config, circuit_breakers=self._circuit_breakers)
                if ordered:
                    selected = ordered[0]

        # Mode 2: Strategy-based routing (fallback)
        if selected is None:
            healthy = self._registry.get_healthy()
            if not healthy:
                raise RuntimeError("No healthy providers available")
            selected = await self._registry.select(request=request)
            if selected is None:
                selected = healthy[0]

        self._registry.increment_in_flight(selected.provider_id)
        provider_in_flight.labels(provider=selected.provider_id).inc()
        try:
            async for chunk in selected.adapter.stream_complete(request, extra_headers=extra_headers):
                yield chunk, selected.provider_id
        finally:
            self._registry.decrement_in_flight(selected.provider_id)
            provider_in_flight.labels(provider=selected.provider_id).dec()

    def get_provider_status(self) -> list[dict]:
        statuses = self._registry.get_provider_status()
        for s in statuses:
            breaker = self._circuit_breakers.get(s["provider_id"])
            if breaker:
                s["circuit_breaker"] = breaker.state.name
            else:
                s["circuit_breaker"] = "CLOSED"
        return statuses

    async def close(self):
        await self._registry.close()

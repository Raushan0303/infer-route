import logging
from app.routing.strategies import (
    ProviderHealth,
    LoadBalancingStrategy,
    create_strategy,
)
from app.core.config import settings
from app.core.constants import STATUS_HEALTHY

logger = logging.getLogger("inferroute")


class ProviderRegistry:

    def __init__(self, strategy_name: str = None, strategy: LoadBalancingStrategy = None):
        self._providers: dict[str, ProviderHealth] = {}
        if strategy is not None:
            self._strategy = strategy
        else:
            self._strategy = create_strategy(
                strategy_name or settings.default_strategy
            )

    def register(
        self,
        provider_id: str,
        adapter: object,
        base_url: str,
        weight: float = 1.0,
        cost_per_1k_tokens: float = 0.0,
        tier: str = "standard",
    ):
        self._providers[provider_id] = ProviderHealth(
            provider_id=provider_id,
            adapter=adapter,
            base_url=base_url,
            weight=weight,
            cost_per_1k_tokens=cost_per_1k_tokens,
            tier=tier,
        )
        logger.info(
            "Registered provider: %s (weight=%.2f, tier=%s, cost=$%.4f/1k)",
            provider_id, weight, tier, cost_per_1k_tokens,
        )

    def get_healthy(self) -> list[ProviderHealth]:
        return [p for p in self._providers.values() if p.status == STATUS_HEALTHY]

    def get_all(self) -> list[ProviderHealth]:
        return list(self._providers.values())

    async def select(self, request: object = None) -> ProviderHealth | None:
        return await self._strategy.select(self.get_all(), request=request)

    def mark_unhealthy(self, provider_id: str):
        if provider_id in self._providers:
            old_status = self._providers[provider_id].status
            self._providers[provider_id].status = "UNHEALTHY"
            if old_status != "UNHEALTHY":
                logger.warning("Provider %s marked UNHEALTHY", provider_id)

    def mark_healthy(self, provider_id: str):
        if provider_id in self._providers:
            old_status = self._providers[provider_id].status
            self._providers[provider_id].status = STATUS_HEALTHY
            if old_status != STATUS_HEALTHY:
                logger.info("Provider %s marked HEALTHY (recovered)", provider_id)

    def increment_in_flight(self, provider_id: str):
        if provider_id in self._providers:
            self._providers[provider_id].in_flight += 1

    def decrement_in_flight(self, provider_id: str):
        if provider_id in self._providers and self._providers[provider_id].in_flight > 0:
            self._providers[provider_id].in_flight -= 1

    def update_latency(self, provider_id: str, latency_ms: float):
        if provider_id in self._providers:
            provider = self._providers[provider_id]
            if hasattr(self._strategy, "update_latency"):
                self._strategy.update_latency(provider, latency_ms)

    def get_provider_status(self) -> list[dict]:
        return [
            {
                "provider_id": p.provider_id,
                "status": p.status,
                "in_flight": p.in_flight,
                "ewma_latency_ms": round(p.ewma_latency, 1),
                "last_latency_ms": round(p.last_latency_ms, 1),
                "weight": p.weight,
            }
            for p in self._providers.values()
        ]

    async def close(self):
        for provider in self._providers.values():
            await provider.adapter.close()

    @property
    def strategy_name(self) -> str:
        return settings.default_strategy

    @property
    def strategy_instance(self) -> LoadBalancingStrategy:
        return self._strategy

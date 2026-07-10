import pytest
from collections import Counter
from unittest.mock import AsyncMock, MagicMock
from app.routing.provider_registry import ProviderRegistry
from app.routing.routing_service import RoutingService
from app.gateway.models import InferRouteRequest, InferRouteMessage


@pytest.mark.asyncio
async def test_round_robin_distribution_within_5_percent():
    adapters = []
    for i in range(3):
        adapter = AsyncMock()
        adapter.complete.return_value = MagicMock(
            content="ok",
            model="test",
            id="test",
            usage=MagicMock(input_tokens=1, output_tokens=1),
            provider=f"provider_{i}",
            latency_ms=10.0,
        )
        adapters.append(adapter)

    registry = ProviderRegistry(strategy_name="round_robin")
    for i, adapter in enumerate(adapters):
        registry.register(f"provider_{i}", adapter, f"http://p{i}", weight=1.0)

    service = RoutingService(registry)

    req = InferRouteRequest(
        messages=[InferRouteMessage(role="user", content="test")],
        model="test-model",
    )

    counts = Counter()
    for _ in range(1000):
        resp = await service.route(req)
        counts[resp.provider] += 1

    for provider_id, count in counts.items():
        assert 283 <= count <= 383, f"{provider_id} got {count}, expected ~333"

    await service.close()

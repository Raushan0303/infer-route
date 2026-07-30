import pytest
from unittest.mock import AsyncMock

from app.routing.provider_registry import ProviderRegistry


@pytest.mark.asyncio
async def test_unhealthy_provider_removed_from_rotation():
    mock_adapter_1 = AsyncMock()
    mock_adapter_1.health_check.return_value = True

    mock_adapter_2 = AsyncMock()
    mock_adapter_2.health_check.return_value = False

    registry = ProviderRegistry(strategy_name="round_robin")
    registry.register("provider_a", mock_adapter_1, "http://a", weight=1.0)
    registry.register("provider_b", mock_adapter_2, "http://b", weight=1.0)

    for provider in registry.get_all():
        healthy = await provider.adapter.health_check()
        if healthy:
            registry.mark_healthy(provider.provider_id)
        else:
            registry.mark_unhealthy(provider.provider_id)

    statuses = {p.provider_id: p.status for p in registry.get_all()}
    assert statuses["provider_b"] == "UNHEALTHY"
    assert statuses["provider_a"] == "HEALTHY"

    selected = await registry.select()
    assert selected.provider_id == "provider_a"


@pytest.mark.asyncio
async def test_recovered_provider_re_added_to_rotation():
    mock_adapter = AsyncMock()

    registry = ProviderRegistry(strategy_name="round_robin")
    registry.register("provider_a", mock_adapter, "http://a", weight=1.0)

    registry.mark_unhealthy("provider_a")
    assert await registry.select() is None

    mock_adapter.health_check.return_value = True
    registry.mark_healthy("provider_a")
    selected = await registry.select()
    assert selected is not None
    assert selected.provider_id == "provider_a"

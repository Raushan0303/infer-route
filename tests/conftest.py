import pytest
from unittest.mock import AsyncMock, MagicMock
from app.routing.strategies import ProviderHealth


@pytest.fixture
def mock_adapter():
    adapter = AsyncMock()
    adapter.complete.return_value = MagicMock(
        content="test response",
        model="test-model",
        usage=MagicMock(input_tokens=10, output_tokens=5),
        provider="test",
        latency_ms=100.0,
        id="test-id",
    )
    adapter.health_check.return_value = True
    return adapter


@pytest.fixture
def healthy_providers(mock_adapter):
    return [
        ProviderHealth(
            provider_id="anthropic", adapter=mock_adapter, base_url="",
            ewma_latency=850, in_flight=5, weight=0.5,
        ),
        ProviderHealth(
            provider_id="openai", adapter=mock_adapter, base_url="",
            ewma_latency=1500, in_flight=3, weight=0.3,
        ),
        ProviderHealth(
            provider_id="vllm", adapter=mock_adapter, base_url="",
            ewma_latency=2100, in_flight=8, weight=0.2,
        ),
    ]

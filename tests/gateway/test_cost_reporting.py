"""cost_usd on /v1/chat/completions, through the real route.

The cache pipeline returns source = "exact" | "semantic" | "coalesced" |
"upstream". Only "upstream" is a billed provider call.
"""
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.main import app
from app.gateway.models import InferRouteResponse, InferRouteUsage
from app.routing.provider_registry import ProviderRegistry


def _registry():
    registry = ProviderRegistry(strategy_name="round_robin")
    # gpt-4o-mini list price: $0.15 / 1M input, $0.60 / 1M output
    registry.register("openai", adapter=MagicMock(), base_url="",
                      cost_per_1m_input=0.15, cost_per_1m_output=0.60)
    return registry


def _response():
    return InferRouteResponse(content="ok", model="gpt-4o-mini", provider="openai",
                              usage=InferRouteUsage(input_tokens=1_000_000, output_tokens=1_000_000))


async def _post(source):
    pipeline = MagicMock()
    pipeline.process = AsyncMock(return_value=(_response(), source))
    for attr in ("rate_limiter", "ab_router", "shadow_traffic", "tracing_ctx", "quota_enforcer", "db"):
        setattr(app.state, attr, None)
    app.state.cache_pipeline = pipeline
    app.state.registry = _registry()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.asyncio
async def test_upstream_call_is_billed_at_list_price():
    body = await _post("upstream")
    # 1M input * $0.15 + 1M output * $0.60 = $0.75
    assert body["cost_usd"] == pytest.approx(0.75)
    assert body["x_routing"]["cache_status"] == "upstream"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["exact", "semantic", "coalesced"])
async def test_cache_hits_and_coalesced_calls_cost_nothing(source):
    body = await _post(source)
    assert body["cost_usd"] == 0.0

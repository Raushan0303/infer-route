"""A/B experiments must change which provider serves the request.

Before the fix, /v1/chat/completions assigned a variant and recorded an
outcome for it, but the request was routed by the normal strategy — the
variant's provider was never used. These tests drive the real route, the
real CachePipeline and the real RoutingService with two fake providers.
"""
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.cache.coalescer import RequestCoalescer
from app.cache.pipeline import CachePipeline
from app.experiments.ab_router import ABRouter
from app.gateway.models import InferRouteResponse, InferRouteUsage
from app.main import app
from app.routing.provider_registry import ProviderRegistry
from app.routing.routing_service import RoutingService


def _adapter(provider_id):
    a = MagicMock()
    a.complete = AsyncMock(return_value=InferRouteResponse(
        content=f"from {provider_id}", model=f"{provider_id}-model", provider=provider_id,
        usage=InferRouteUsage(input_tokens=10, output_tokens=10), latency_ms=5.0))
    return a


class MissCache:
    async def get(self, *a, **k):
        return None

    async def set(self, *a, **k):
        return None


def _setup(with_cache: bool):
    registry = ProviderRegistry(strategy_name="round_robin")
    registry.register("control", adapter=_adapter("control"), base_url="")
    registry.register("challenger", adapter=_adapter("challenger"), base_url="")
    routing = RoutingService(registry)
    ab = ABRouter()
    ab._provider_a, ab._provider_b, ab._split = "control", "challenger", 0.5
    for attr in ("rate_limiter", "shadow_traffic", "tracing_ctx", "quota_enforcer", "db"):
        setattr(app.state, attr, None)
    app.state.registry = registry
    app.state.routing_service = routing
    app.state.ab_router = ab
    app.state.cache_pipeline = (
        CachePipeline(MissCache(), MissCache(), RequestCoalescer(), routing) if with_cache else None
    )
    return ab


async def _chat(user):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", headers={"Authorization": "Bearer tenant-x"},
                         json={"model": "auto", "user": user,
                               "messages": [{"role": "user", "content": f"hello {uuid.uuid4()}"}]})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_cache", [True, False])
async def test_each_request_is_served_by_its_variants_provider(with_cache):
    ab = _setup(with_cache)
    served = {"A": set(), "B": set()}
    for i in range(40):
        user = f"user-{i}"
        variant = ab.assign_variant("tenant-x", user)
        body = await _chat(user)
        served[variant].add(body["x_routing"]["provider"])
    assert served == {"A": {"control"}, "B": {"challenger"}}, f"variant -> providers that served it: {served}"
    assert body["x_routing"]["ab_variant"] == variant


@pytest.mark.asyncio
async def test_variant_falls_back_when_its_provider_is_unhealthy():
    ab = _setup(with_cache=False)
    app.state.registry.mark_unhealthy("challenger")
    user = next(f"u{i}" for i in range(100) if ab.assign_variant("tenant-x", f"u{i}") == "B")
    body = await _chat(user)
    assert body["x_routing"]["provider"] == "control"


@pytest.mark.asyncio
async def test_promote_swaps_providers_and_logs_the_swap():
    ab = _setup(with_cache=False)
    app.state.redis = None
    for i in range(20):
        await _chat(f"user-{i}")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        refused = await c.post("/v1/experiments/promote", json={"min_samples": 1000})
        assert refused.status_code == 409  # not enough evidence yet
        r = await c.post("/v1/experiments/promote",
                         json={"reason": "challenger cheaper, same latency", "min_samples": 5})
        assert r.status_code == 200, r.text
        swaps = (await c.get("/v1/experiments/swaps")).json()["swaps"]
    assert len(swaps) == 1
    assert swaps[0]["from_provider"] == "control" and swaps[0]["to_provider"] == "challenger"
    assert swaps[0]["stats_at_swap"]["A"]["served_by"] == {"control": swaps[0]["stats_at_swap"]["A"]["count"]}
    assert ab.get_provider_for_variant("A") == "challenger"

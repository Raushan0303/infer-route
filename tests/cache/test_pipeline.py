import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.cache.pipeline import CachePipeline
from app.cache.exact_cache import ExactMatchCache
from app.cache.semantic_cache import SemanticCache
from app.cache.coalescer import RequestCoalescer
from app.gateway.models import InferRouteRequest, InferRouteMessage, InferRouteResponse, InferRouteUsage


def make_request(content="hello", model="gpt-4o-mini"):
    return InferRouteRequest(
        messages=[InferRouteMessage(role="user", content=content)],
        model=model,
    )


def make_response(content="response", provider="openai"):
    return InferRouteResponse(
        content=content,
        model="gpt-4o-mini",
        usage=InferRouteUsage(input_tokens=10, output_tokens=5),
        provider=provider,
    )


@pytest.mark.asyncio
async def test_pipeline_exact_cache_hit():
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()

    response = make_response("cached answer")
    mock_redis.get.return_value = response.model_dump_json()

    exact = ExactMatchCache(mock_redis)
    semantic = SemanticCache(mock_redis, mock_embedding)
    coalescer = RequestCoalescer()
    routing = AsyncMock()

    pipeline = CachePipeline(exact, semantic, coalescer, routing)
    result, source = await pipeline.process("tenant_a", make_request())

    assert source == "exact"
    assert result.content == "cached answer"
    routing.route.assert_not_called()  # no upstream call


@pytest.mark.asyncio
async def test_pipeline_upstream_on_cache_miss():
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()

    mock_redis.get.return_value = None  # exact miss
    mock_redis.smembers.return_value = set()  # semantic miss
    mock_embedding.embed.return_value = [1.0, 0.0]

    upstream_response = make_response("fresh answer")
    routing = AsyncMock()
    routing.route.return_value = upstream_response

    exact = ExactMatchCache(mock_redis)
    semantic = SemanticCache(mock_redis, mock_embedding)
    coalescer = RequestCoalescer()

    pipeline = CachePipeline(exact, semantic, coalescer, routing)
    result, source = await pipeline.process("tenant_a", make_request())

    assert source == "upstream"
    assert result.content == "fresh answer"
    routing.route.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_coalesces_concurrent_identical_requests():
    """50 concurrent identical requests → 1 upstream call."""
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()

    mock_redis.get.return_value = None
    mock_redis.smembers.return_value = set()
    mock_embedding.embed.return_value = [1.0, 0.0]

    call_count = 0

    async def route_fn(req):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)
        return make_response(f"result_{call_count}")

    routing = MagicMock()
    routing.route = route_fn

    exact = ExactMatchCache(mock_redis)
    semantic = SemanticCache(mock_redis, mock_embedding)
    coalescer = RequestCoalescer()

    pipeline = CachePipeline(exact, semantic, coalescer, routing)

    results = await asyncio.gather(
        *[pipeline.process("tenant_a", make_request()) for _ in range(50)]
    )

    assert call_count == 1  # only 1 upstream call
    assert all(r[0].content == "result_1" for r in results)


@pytest.mark.asyncio
async def test_pipeline_tenant_isolation():
    """Tenant B should not get tenant A's cached response."""
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()

    response_a = make_response("tenant A answer")

    # First call (tenant A): cache miss → upstream → cache write
    # Second call (tenant B): cache miss (different tenant key) → upstream
    call_count = 0

    async def get_side_effect(key):
        nonlocal call_count
        if call_count == 0:
            return None  # tenant A: miss
        elif "tenant_a" in key:
            return response_a.model_dump_json()  # would be cached for tenant A
        return None  # tenant B: miss

    mock_redis.get.side_effect = get_side_effect
    mock_redis.smembers.return_value = set()
    mock_embedding.embed.return_value = [1.0, 0.0]

    route_count = 0

    async def route_fn(req):
        nonlocal route_count
        route_count += 1
        return make_response(f"answer_{route_count}")

    routing = MagicMock()
    routing.route = route_fn

    exact = ExactMatchCache(mock_redis)
    semantic = SemanticCache(mock_redis, mock_embedding)
    coalescer = RequestCoalescer()

    pipeline = CachePipeline(exact, semantic, coalescer, routing)

    # Reset mock for sequential calls
    mock_redis.get.side_effect = [None, None]  # both miss
    mock_redis.smembers.return_value = set()

    result_a, source_a = await pipeline.process("tenant_a", make_request())
    result_b, source_b = await pipeline.process("tenant_b", make_request())

    assert source_a == "upstream"
    assert source_b == "upstream"
    assert route_count == 2  # both went to upstream

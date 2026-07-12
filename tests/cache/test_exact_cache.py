import json
import pytest
from unittest.mock import AsyncMock
from app.cache.exact_cache import ExactMatchCache, canonical_json, request_hash, apply_ttl_jitter
from app.gateway.models import InferRouteRequest, InferRouteMessage, InferRouteResponse, InferRouteUsage


def test_canonical_json_key_ordering():
    """Different key ordering should produce the same canonical JSON."""
    req1 = InferRouteRequest(
        messages=[InferRouteMessage(role="user", content="hi")],
        model="gpt-4o-mini",
        temperature=0.7,
    )
    req2 = InferRouteRequest(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[InferRouteMessage(role="user", content="hi")],
    )
    assert canonical_json(req1) == canonical_json(req2)
    assert request_hash(req1) == request_hash(req2)


def test_different_requests_different_hashes():
    req1 = InferRouteRequest(
        messages=[InferRouteMessage(role="user", content="hello")],
        model="gpt-4o-mini",
    )
    req2 = InferRouteRequest(
        messages=[InferRouteMessage(role="user", content="hi")],
        model="gpt-4o-mini",
    )
    assert request_hash(req1) != request_hash(req2)


def test_apply_ttl_jitter_within_bounds():
    ttl = 3600
    jitter = 0.1
    for _ in range(100):
        result = apply_ttl_jitter(ttl, jitter)
        assert 3240 <= result <= 3960  # ±10% of 3600


@pytest.mark.asyncio
async def test_exact_cache_hit():
    mock_redis = AsyncMock()
    response = InferRouteResponse(
        content="cached response",
        model="gpt-4o-mini",
        usage=InferRouteUsage(input_tokens=10, output_tokens=5),
        provider="openai",
    )
    mock_redis.get.return_value = response.model_dump_json()

    cache = ExactMatchCache(mock_redis)
    req = InferRouteRequest(
        messages=[InferRouteMessage(role="user", content="hi")],
        model="gpt-4o-mini",
    )
    result = await cache.get("tenant_a", req)
    assert result is not None
    assert result.content == "cached response"


@pytest.mark.asyncio
async def test_exact_cache_miss():
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    cache = ExactMatchCache(mock_redis)
    req = InferRouteRequest(
        messages=[InferRouteMessage(role="user", content="hi")],
        model="gpt-4o-mini",
    )
    result = await cache.get("tenant_a", req)
    assert result is None


@pytest.mark.asyncio
async def test_exact_cache_redis_error_returns_none():
    mock_redis = AsyncMock()
    mock_redis.get.side_effect = Exception("Redis connection refused")

    cache = ExactMatchCache(mock_redis)
    req = InferRouteRequest(
        messages=[InferRouteMessage(role="user", content="hi")],
        model="gpt-4o-mini",
    )
    result = await cache.get("tenant_a", req)
    assert result is None  # fails gracefully


@pytest.mark.asyncio
async def test_exact_cache_set():
    mock_redis = AsyncMock()
    cache = ExactMatchCache(mock_redis)
    req = InferRouteRequest(
        messages=[InferRouteMessage(role="user", content="hi")],
        model="gpt-4o-mini",
    )
    response = InferRouteResponse(
        content="response",
        model="gpt-4o-mini",
        usage=InferRouteUsage(input_tokens=10, output_tokens=5),
        provider="openai",
    )
    await cache.set("tenant_a", req, response)
    mock_redis.setex.assert_called_once()

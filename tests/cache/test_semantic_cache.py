import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock
from app.cache.semantic_cache import SemanticCache, cosine_similarity, xfetch_expired
from app.gateway.models import InferRouteRequest, InferRouteMessage, InferRouteResponse, InferRouteUsage


def test_cosine_similarity_identical_vectors():
    a = [1.0, 2.0, 3.0]
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    a = [0.0, 0.0]
    b = [1.0, 2.0]
    assert cosine_similarity(a, b) == 0.0


def test_xfetch_expired_old_entry():
    assert xfetch_expired(time.time() - 7200, 3600) is True  # 2 hours old, TTL 1 hour


def test_xfetch_not_expired_fresh_entry():
    assert xfetch_expired(time.time(), 3600) is False  # just created


@pytest.mark.asyncio
async def test_semantic_cache_hit_above_threshold():
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()

    query_embedding = [1.0, 0.0, 0.0]
    cached_embedding = [0.99, 0.01, 0.0]  # very similar

    mock_embedding.embed.return_value = query_embedding

    entry = {
        "embedding": cached_embedding,
        "response": {
            "content": "cached answer",
            "model": "gpt-4o-mini",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "provider": "openai",
        },
        "created_at": time.time(),
        "prompt": "hello world",
    }

    mock_redis.smembers.return_value = {"entry1"}
    mock_redis.mget.return_value = [json.dumps(entry)]

    cache = SemanticCache(mock_redis, mock_embedding)
    req = InferRouteRequest(
        messages=[InferRouteMessage(role="user", content="hello world")],
        model="gpt-4o-mini",
    )
    result = await cache.get("tenant_a", req)
    assert result is not None
    assert result.content == "cached answer"


@pytest.mark.asyncio
async def test_semantic_cache_miss_below_threshold():
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()

    query_embedding = [1.0, 0.0, 0.0]
    cached_embedding = [0.0, 1.0, 0.0]  # orthogonal — completely different

    mock_embedding.embed.return_value = query_embedding

    entry = {
        "embedding": cached_embedding,
        "response": {
            "content": "cached answer",
            "model": "gpt-4o-mini",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "provider": "openai",
        },
        "created_at": time.time(),
        "prompt": "completely different prompt",
    }

    mock_redis.smembers.return_value = {"entry1"}
    mock_redis.mget.return_value = [json.dumps(entry)]

    cache = SemanticCache(mock_redis, mock_embedding)
    req = InferRouteRequest(
        messages=[InferRouteMessage(role="user", content="hello world")],
        model="gpt-4o-mini",
    )
    result = await cache.get("tenant_a", req)
    assert result is None  # below threshold — no false positive


@pytest.mark.asyncio
async def test_semantic_cache_tenant_isolation():
    """Tenant B should not see tenant A's cached entries."""
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()

    mock_embedding.embed.return_value = [1.0, 0.0, 0.0]

    # Tenant B's index is empty
    mock_redis.smembers.return_value = set()

    cache = SemanticCache(mock_redis, mock_embedding)
    req = InferRouteRequest(
        messages=[InferRouteMessage(role="user", content="hello world")],
        model="gpt-4o-mini",
    )
    result = await cache.get("tenant_b", req)
    assert result is None  # tenant B gets no hits from tenant A's cache


@pytest.mark.asyncio
async def test_semantic_cache_redis_error_returns_none():
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()
    mock_embedding.embed.side_effect = Exception("Embedding service down")

    cache = SemanticCache(mock_redis, mock_embedding)
    req = InferRouteRequest(
        messages=[InferRouteMessage(role="user", content="hi")],
        model="gpt-4o-mini",
    )
    result = await cache.get("tenant_a", req)
    assert result is None  # fails gracefully

"""Tests for RAG-augmented chat completions.

Verifies:
1. Chat without namespace → no RAG, same as before
2. Chat with namespace → RAG retrieval, augmented prompt, x_rag in response
3. Chat with namespace + cache hit → no RAG retrieval needed, x_rag shows cache_hit
4. Chat with namespace that has no documents → graceful fallback, no x_rag
5. Cache key isolation: same prompt with different namespaces → separate cache entries
6. Transform callback augments prompt correctly
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.gateway.models import InferRouteRequest, InferRouteMessage, InferRouteResponse, InferRouteUsage
from app.cache.pipeline import CachePipeline
from app.cache.exact_cache import ExactMatchCache
from app.cache.semantic_cache import SemanticCache
from app.cache.coalescer import RequestCoalescer
from app.gateway.routes import _build_rag_transform


def make_request(content="hello", model="gpt-4o-mini", namespace=None):
    return InferRouteRequest(
        messages=[InferRouteMessage(role="user", content=content)],
        model=model,
        namespace=namespace,
    )


def make_response(content="response", provider="openai"):
    return InferRouteResponse(
        content=content,
        model="gpt-4o-mini",
        usage=InferRouteUsage(input_tokens=10, output_tokens=5),
        provider=provider,
    )


# --- Transform callback tests ---

@pytest.mark.asyncio
async def test_rag_transform_augments_prompt():
    """Transform callback should prepend context to the last user message."""
    rag_service = AsyncMock()
    rag_service.query.return_value = [
        {"doc_id": "doc1", "content": "Refunds within 30 days.", "score": 0.95, "source": "dense"},
        {"doc_id": "doc2", "content": "Shipping is free.", "score": 0.80, "source": "bm25"},
    ]

    transform, rag_metadata = _build_rag_transform(rag_service, "tenant_a", "company-docs")

    original = make_request("What is our refund policy?", namespace="company-docs")
    augmented = await transform(original)

    # Original should be unchanged (cache keying depends on it)
    assert original.messages[0].content == "What is our refund policy?"

    # Augmented should have context prepended
    assert "Context:" in augmented.messages[0].content
    assert "Refunds within 30 days." in augmented.messages[0].content
    assert "Shipping is free." in augmented.messages[0].content
    assert "Question: What is our refund policy?" in augmented.messages[0].content

    # Metadata should be populated
    assert rag_metadata["retrieved"] == 2
    assert rag_metadata["namespace"] == "company-docs"
    assert len(rag_metadata["sources"]) == 2
    assert rag_metadata["sources"][0]["doc_id"] == "doc1"
    assert rag_metadata["context_tokens"] > 0


@pytest.mark.asyncio
async def test_rag_transform_no_results_returns_original():
    """If RAG returns no documents, transform should return original request unchanged."""
    rag_service = AsyncMock()
    rag_service.query.return_value = []

    transform, rag_metadata = _build_rag_transform(rag_service, "tenant_a", "empty-ns")

    original = make_request("What is our refund policy?", namespace="empty-ns")
    augmented = await transform(original)

    assert augmented is original  # same object, no augmentation
    assert rag_metadata["retrieved"] == 0


@pytest.mark.asyncio
async def test_rag_transform_no_user_message_returns_original():
    """If there's no user message, transform should return original request."""
    rag_service = AsyncMock()
    rag_service.query.return_value = [
        {"doc_id": "doc1", "content": "Some content", "score": 0.9, "source": "dense"},
    ]

    transform, rag_metadata = _build_rag_transform(rag_service, "tenant_a", "company-docs")

    original = InferRouteRequest(
        messages=[InferRouteMessage(role="system", content="You are a helpful assistant")],
        model="gpt-4o-mini",
        namespace="company-docs",
    )
    augmented = await transform(original)

    assert augmented is original
    assert rag_metadata["retrieved"] == 0


# --- Cache pipeline with transform ---

@pytest.mark.asyncio
async def test_pipeline_applies_transform_on_cache_miss():
    """On cache miss, the transform should be applied before routing."""
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()

    mock_redis.get.return_value = None  # exact miss
    mock_redis.smembers.return_value = set()  # semantic miss
    mock_embedding.embed.return_value = [1.0, 0.0]

    upstream_response = make_response("grounded answer")
    routing = AsyncMock()
    routing.route.return_value = upstream_response

    exact = ExactMatchCache(mock_redis)
    semantic = SemanticCache(mock_redis, mock_embedding)
    coalescer = RequestCoalescer()

    pipeline = CachePipeline(exact, semantic, coalescer, routing)

    # Track what request gets routed
    routed_request = None

    async def route_fn(req):
        nonlocal routed_request
        routed_request = req
        return upstream_response

    routing.route = route_fn

    # Build a simple transform that adds context
    async def transform(req):
        return InferRouteRequest(
            messages=[InferRouteMessage(role="user", content=f"Context: docs\n\n{req.messages[0].content}")],
            model=req.model,
            namespace=req.namespace,
        )

    result, source = await pipeline.process("tenant_a", make_request("hello", namespace="test-ns"), transform=transform)

    assert source == "upstream"
    assert result.content == "grounded answer"
    # The routed request should be augmented
    assert "Context: docs" in routed_request.messages[0].content


@pytest.mark.asyncio
async def test_pipeline_no_transform_when_cache_hit():
    """On cache hit, transform should NOT be called — cached response returned directly."""
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()

    response = make_response("cached answer")
    mock_redis.get.return_value = response.model_dump_json()

    exact = ExactMatchCache(mock_redis)
    semantic = SemanticCache(mock_redis, mock_embedding)
    coalescer = RequestCoalescer()
    routing = AsyncMock()

    pipeline = CachePipeline(exact, semantic, coalescer, routing)

    transform_called = False

    async def transform(req):
        nonlocal transform_called
        transform_called = True
        return req

    result, source = await pipeline.process("tenant_a", make_request("hello"), transform=transform)

    assert source == "exact"
    assert result.content == "cached answer"
    assert not transform_called  # transform should not run on cache hit
    routing.route.assert_not_called()


# --- Cache key isolation with namespace ---

@pytest.mark.asyncio
async def test_exact_cache_namespace_isolation():
    """Same prompt with different namespaces should produce different cache keys."""
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()

    mock_redis.get.return_value = None  # always miss
    mock_redis.smembers.return_value = set()
    mock_embedding.embed.return_value = [1.0, 0.0]

    routing = AsyncMock()
    routing.route.return_value = make_response("answer")

    exact = ExactMatchCache(mock_redis)
    semantic = SemanticCache(mock_redis, mock_embedding)
    coalescer = RequestCoalescer()

    pipeline = CachePipeline(exact, semantic, coalescer, routing)

    req_no_ns = make_request("What is our refund policy?")
    req_with_ns = make_request("What is our refund policy?", namespace="company-docs")

    await pipeline.process("tenant_a", req_no_ns)
    await pipeline.process("tenant_a", req_with_ns)

    # Two different cache keys should have been written
    setex_calls = mock_redis.setex.call_args_list
    assert len(setex_calls) >= 2
    # The keys should be different
    keys = [call.args[0] for call in setex_calls]
    assert len(set(keys)) >= 2  # at least 2 unique keys


@pytest.mark.asyncio
async def test_semantic_cache_namespace_isolation():
    """Semantic cache index keys should include namespace."""
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()

    mock_redis.get.return_value = None
    mock_embedding.embed.return_value = [1.0, 0.0]

    exact = ExactMatchCache(mock_redis)
    semantic = SemanticCache(mock_redis, mock_embedding)

    req_no_ns = make_request("What is our refund policy?")
    req_with_ns = make_request("What is our refund policy?", namespace="company-docs")

    await semantic.set("tenant_a", req_no_ns, make_response("answer 1"))
    await semantic.set("tenant_a", req_with_ns, make_response("answer 2"))

    # Check that smembers was called with different index keys
    # The index keys should differ because namespace is included
    sadd_calls = mock_redis.sadd.call_args_list
    index_keys = [call.args[0] for call in sadd_calls]
    assert len(set(index_keys)) == 2  # two different index keys
    assert any("company-docs" in k for k in index_keys)
    assert any("_default" in k for k in index_keys)


# --- Backward compatibility ---

@pytest.mark.asyncio
async def test_pipeline_without_transform_works_as_before():
    """Pipeline without transform argument should work exactly as before."""
    mock_redis = AsyncMock()
    mock_embedding = AsyncMock()

    mock_redis.get.return_value = None
    mock_redis.smembers.return_value = set()
    mock_embedding.embed.return_value = [1.0, 0.0]

    upstream_response = make_response("fresh answer")
    routing = AsyncMock()
    routing.route.return_value = upstream_response

    exact = ExactMatchCache(mock_redis)
    semantic = SemanticCache(mock_redis, mock_embedding)
    coalescer = RequestCoalescer()

    pipeline = CachePipeline(exact, semantic, coalescer, routing)

    # No transform argument — should work as before
    result, source = await pipeline.process("tenant_a", make_request("hello"))

    assert source == "upstream"
    assert result.content == "fresh answer"
    routing.route.assert_called_once()

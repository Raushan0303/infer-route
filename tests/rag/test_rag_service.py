import pytest
from unittest.mock import AsyncMock
from app.rag.rag_service import RAGService
from app.rag.store import VectorStore, Document
from app.rag.retriever import HybridRetriever


@pytest.mark.asyncio
async def test_rag_query_returns_results():
    store = VectorStore()
    docs = [
        Document(id="d1", content="machine learning", embedding=[1.0, 0.0]),
        Document(id="d2", content="deep learning", embedding=[0.8, 0.2]),
    ]
    await store.upsert("tenant_a", "ns1", docs)

    retriever = HybridRetriever(store)
    service = RAGService(store, retriever, redis=None)

    results = await service.query("tenant_a", "ns1", "machine learning")
    # Without embedding service, only BM25 runs
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_rag_upsert():
    store = VectorStore()
    retriever = HybridRetriever(store)
    service = RAGService(store, retriever, redis=None)

    documents = [
        {"id": "d1", "content": "hello world"},
        {"id": "d2", "content": "foo bar"},
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    count = await service.upsert("tenant_a", "ns1", documents, embeddings)
    assert count == 2


@pytest.mark.asyncio
async def test_rag_query_with_redis_cache():
    mock_redis = AsyncMock()
    mock_redis.get.return_value = '[{"doc_id": "cached", "content": "cached", "score": 1.0, "source": "dense"}]'

    store = VectorStore()
    retriever = HybridRetriever(store)
    service = RAGService(store, retriever, redis=mock_redis)

    results = await service.query("tenant_a", "ns1", "test")
    assert len(results) == 1
    assert results[0]["doc_id"] == "cached"

import pytest
from app.rag.store import VectorStore, Document
from app.rag.retriever import HybridRetriever


@pytest.mark.asyncio
async def test_hybrid_retrieval_fuses_dense_and_bm25():
    store = VectorStore()
    docs = [
        Document(id="d1", content="machine learning models", embedding=[1.0, 0.0, 0.0]),
        Document(id="d2", content="deep learning networks", embedding=[0.8, 0.2, 0.0]),
        Document(id="d3", content="neural network architecture", embedding=[0.1, 0.9, 0.0]),
        Document(id="d4", content="machine learning is powerful", embedding=[0.9, 0.1, 0.0]),
    ]
    await store.upsert("tenant_a", "ns1", docs)

    retriever = HybridRetriever(store)
    results = await retriever.retrieve("tenant_a", "ns1", "machine learning", query_embedding=[1.0, 0.0, 0.0])

    assert len(results) > 0
    # d1 and d4 should rank high (both match "machine learning" in BM25 and are close in dense)
    doc_ids = [r.doc_id for r in results]
    assert "d1" in doc_ids
    assert "d4" in doc_ids


@pytest.mark.asyncio
async def test_rrf_combines_rankings():
    retriever = HybridRetriever(VectorStore())

    dense = [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)]
    bm25 = [("d2", 5.0), ("d4", 3.0), ("d1", 2.0)]

    fused = retriever._reciprocal_rank_fusion(dense, bm25)

    # d1 appears in both (rank 0 in dense, rank 2 in bm25) → high RRF score
    # d2 appears in both (rank 1 in dense, rank 0 in bm25) → high RRF score
    fused_ids = [f[0] for f in fused]
    assert "d1" in fused_ids
    assert "d2" in fused_ids
    assert "d4" in fused_ids  # only in bm25 but still included


@pytest.mark.asyncio
async def test_tenant_isolation_in_retrieval():
    store = VectorStore()
    docs_a = [Document(id="d1", content="tenant a secret", embedding=[1.0, 0.0])]
    await store.upsert("tenant_a", "ns1", docs_a)

    retriever = HybridRetriever(store)
    results = await retriever.retrieve("tenant_b", "ns1", "tenant a secret", query_embedding=[1.0, 0.0])
    assert len(results) == 0  # tenant B cannot access tenant A's data

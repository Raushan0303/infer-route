import pytest
from app.rag.store import VectorStore, Document


@pytest.mark.asyncio
async def test_upsert_and_query_dense():
    store = VectorStore()
    docs = [
        Document(id="d1", content="machine learning models", embedding=[1.0, 0.0, 0.0]),
        Document(id="d2", content="deep learning networks", embedding=[0.0, 1.0, 0.0]),
        Document(id="d3", content="neural networks architecture", embedding=[0.9, 0.1, 0.0]),
    ]
    count = await store.upsert("tenant_a", "ns1", docs)
    assert count == 3

    results = await store.query_dense("tenant_a", "ns1", [1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0][0] == "d1"  # closest to query


@pytest.mark.asyncio
async def test_upsert_and_query_bm25():
    store = VectorStore()
    docs = [
        Document(id="d1", content="machine learning is fun"),
        Document(id="d2", content="deep learning requires gpus"),
        Document(id="d3", content="machine learning models are powerful"),
    ]
    await store.upsert("tenant_a", "ns1", docs)

    results = await store.query_bm25("tenant_a", "ns1", "machine learning", top_k=3)
    assert len(results) >= 1
    # d1 and d3 both contain "machine learning"
    doc_ids = [r[0] for r in results]
    assert "d1" in doc_ids
    assert "d3" in doc_ids


@pytest.mark.asyncio
async def test_tenant_isolation():
    store = VectorStore()
    docs_a = [Document(id="d1", content="tenant a data", embedding=[1.0, 0.0])]
    docs_b = [Document(id="d2", content="tenant b data", embedding=[0.0, 1.0])]
    await store.upsert("tenant_a", "ns1", docs_a)
    await store.upsert("tenant_b", "ns1", docs_b)

    # Tenant B cannot query tenant A's data
    results = await store.query_dense("tenant_b", "ns1", [1.0, 0.0])
    doc_ids = [r[0] for r in results]
    assert "d1" not in doc_ids  # tenant A's doc not accessible

    # Tenant A can query their own
    results = await store.query_dense("tenant_a", "ns1", [1.0, 0.0])
    assert len(results) == 1
    assert results[0][0] == "d1"


@pytest.mark.asyncio
async def test_get_document():
    store = VectorStore()
    doc = Document(id="d1", content="hello world", embedding=[1.0, 0.0])
    await store.upsert("tenant_a", "ns1", [doc])

    result = await store.get_document("tenant_a", "ns1", "d1")
    assert result is not None
    assert result.content == "hello world"

    # Wrong tenant
    result = await store.get_document("tenant_b", "ns1", "d1")
    assert result is None


@pytest.mark.asyncio
async def test_list_namespaces():
    store = VectorStore()
    await store.upsert("tenant_a", "ns1", [Document(id="d1", content="a")])
    await store.upsert("tenant_a", "ns2", [Document(id="d2", content="b")])
    await store.upsert("tenant_b", "ns1", [Document(id="d3", content="c")])

    ns_a = await store.list_namespaces("tenant_a")
    assert set(ns_a) == {"ns1", "ns2"}

    ns_b = await store.list_namespaces("tenant_b")
    assert ns_b == ["ns1"]

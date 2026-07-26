import json
import logging
import time
from app.core.config import settings
from app.core.constants import RAG_KEY_PREFIX
from app.rag.store import VectorStore, Document
from app.rag.retriever import HybridRetriever, RetrievalResult
from app.observability.metrics import rag_query_total, rag_query_duration

logger = logging.getLogger("inferroute")


class RAGService:

    def __init__(self, store: VectorStore, retriever: HybridRetriever, redis=None):
        self._store = store
        self._retriever = retriever
        self._redis = redis
        self._cache_ttl = settings.rag_cache_ttl

    def _cache_key(self, tenant_id: str, namespace: str, query: str) -> str:
        import hashlib
        h = hashlib.sha256(f"{tenant_id}:{namespace}:{query}".encode()).hexdigest()
        return f"{RAG_KEY_PREFIX}:{tenant_id}:{h}"

    async def query(
        self,
        tenant_id: str,
        namespace: str,
        query: str,
        top_k: int = None,
    ) -> list[dict]:
        start = time.monotonic()
        rag_query_total.labels(tenant_id=tenant_id, namespace=namespace).inc()

        cache_key = self._cache_key(tenant_id, namespace, query)
        if self._redis:
            try:
                cached = await self._redis.get(cache_key)
                if cached:
                    rag_query_duration.observe(time.monotonic() - start)
                    return json.loads(cached)
            except Exception as e:
                logger.warning("RAG cache read error: %s", e)

        results = await self._retriever.retrieve(tenant_id, namespace, query, top_k=top_k)

        enriched = []
        for result in results:
            doc = await self._store.get_document(tenant_id, namespace, result.doc_id)
            enriched.append({
                "doc_id": result.doc_id,
                "content": doc.content if doc else "",
                "score": result.score,
                "source": result.source,
                "metadata": doc.metadata if doc else {},
            })

        if self._redis:
            try:
                await self._redis.setex(cache_key, self._cache_ttl, json.dumps(enriched))
            except Exception as e:
                logger.warning("RAG cache write error: %s", e)

        rag_query_duration.observe(time.monotonic() - start)
        return enriched

    async def upsert(
        self,
        tenant_id: str,
        namespace: str,
        documents: list[dict],
        embeddings: list[list[float]] = None,
    ) -> int:
        docs = []
        for i, d in enumerate(documents):
            doc = Document(
                id=d["id"],
                content=d["content"],
                embedding=embeddings[i] if embeddings else [],
                namespace=namespace,
                metadata=d.get("metadata", {}),
            )
            docs.append(doc)

        return await self._store.upsert(tenant_id, namespace, docs)

import logging
from dataclasses import dataclass
from app.core.config import settings
from app.rag.store import VectorStore

logger = logging.getLogger("inferroute")


@dataclass
class RetrievalResult:
    doc_id: str
    content: str
    score: float
    source: str  # "dense", "bm25", "fused"


class HybridRetriever:

    def __init__(self, store: VectorStore, embedding_service=None):
        self._store = store
        self._embedding_service = embedding_service
        self._rrf_k = 60  # RRF constant

    async def retrieve(
        self,
        tenant_id: str,
        namespace: str,
        query: str,
        query_embedding: list[float] = None,
        top_k: int = None,
    ) -> list[RetrievalResult]:
        top_k = top_k or settings.rag_top_k

        if query_embedding is None and self._embedding_service:
            query_embedding = await self._embedding_service.embed(query)

        dense_results = []
        if query_embedding:
            dense_results = await self._store.query_dense(tenant_id, namespace, query_embedding, top_k)

        bm25_results = await self._store.query_bm25(tenant_id, namespace, query, top_k)

        fused = self._reciprocal_rank_fusion(dense_results, bm25_results)

        reranked = self._rerank(query, fused, settings.rag_rerank_top_k)

        return reranked

    def _reciprocal_rank_fusion(
        self,
        dense: list[tuple[str, float]],
        bm25: list[tuple[str, float]],
    ) -> list[tuple[str, float, str]]:
        scores = {}

        for rank, (doc_id, _) in enumerate(dense):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)

        for rank, (doc_id, _) in enumerate(bm25):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)

        source_map = {}
        for doc_id, _ in dense:
            source_map[doc_id] = "dense"
        for doc_id, _ in bm25:
            if doc_id not in source_map:
                source_map[doc_id] = "bm25"
            else:
                source_map[doc_id] = "fused"

        fused = [(doc_id, score, source_map.get(doc_id, "fused")) for doc_id, score in scores.items()]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    def _rerank(
        self,
        query: str,
        fused: list[tuple[str, float, str]],
        top_k: int,
    ) -> list[RetrievalResult]:
        query_tokens = set(query.lower().split())

        reranked = []
        for doc_id, rrf_score, source in fused[:top_k]:
            reranked.append((doc_id, rrf_score, source))

        reranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score, source in reranked:
            results.append(RetrievalResult(
                doc_id=doc_id,
                content="",
                score=score,
                source=source,
            ))

        return results

import logging
import json
from dataclasses import dataclass, field
from app.core.config import settings
from app.core.database import db

logger = logging.getLogger("inferroute")


@dataclass
class Document:
    id: str
    content: str
    embedding: list[float] = field(default_factory=list)
    namespace: str = ""
    metadata: dict = field(default_factory=dict)


class VectorStore:

    def __init__(self):
        self._store: dict[str, dict[str, Document]] = {}
        self._bm25_index: dict[str, list[dict]] = {}
        self._db = db

    def _ns_key(self, tenant_id: str, namespace: str) -> str:
        return f"{tenant_id}:{namespace}"

    async def upsert(
        self,
        tenant_id: str,
        namespace: str,
        documents: list[Document],
    ) -> int:
        ns_key = self._ns_key(tenant_id, namespace)
        if ns_key not in self._store:
            self._store[ns_key] = {}
            self._bm25_index[ns_key] = []

        count = 0
        for doc in documents:
            doc.namespace = namespace
            self._store[ns_key][doc.id] = doc

            tokens = doc.content.lower().split()
            self._bm25_index[ns_key].append({
                "id": doc.id,
                "tokens": set(tokens),
                "token_list": tokens,
                "content": doc.content,
            })
            count += 1

        # Persist to Postgres
        try:
            pool = await self._db.get_pool()
            async with pool.acquire() as conn:
                for doc in documents:
                    embedding_str = f"[{','.join(str(x) for x in doc.embedding)}]" if doc.embedding else None
                    await conn.execute(
                        """INSERT INTO rag_documents (id, tenant_id, namespace, content, embedding, metadata)
                        VALUES ($1, $2, $3, $4, $5::vector, $6)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata""",
                        doc.id, tenant_id, namespace, doc.content, embedding_str,
                        json.dumps(doc.metadata),
                    )
        except Exception as e:
            logger.warning("Failed to persist RAG documents to Postgres (in-memory only): %s", e)

        logger.info("Upserted %d documents into namespace '%s'", count, ns_key)
        return count

    async def query_dense(
        self,
        tenant_id: str,
        namespace: str,
        query_embedding: list[float],
        top_k: int = None,
    ) -> list[tuple[str, float]]:
        # Try Postgres + pgvector first
        try:
            pool = await self._db.get_pool()
            async with pool.acquire() as conn:
                embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
                rows = await conn.fetch(
                    """SELECT id, 1 - (embedding <=> $1::vector) as similarity
                    FROM rag_documents
                    WHERE tenant_id = $2 AND namespace = $3 AND embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $4""",
                    embedding_str, tenant_id, namespace, top_k or settings.rag_top_k,
                )
                if rows:
                    return [(row["id"], float(row["similarity"])) for row in rows]
        except Exception as e:
            logger.debug("pgvector query failed, falling back to in-memory: %s", e)

        # Fallback to in-memory
        ns_key = self._ns_key(tenant_id, namespace)
        if ns_key not in self._store:
            return []

        top_k = top_k or settings.rag_top_k
        scores = []

        for doc_id, doc in self._store[ns_key].items():
            if not doc.embedding:
                continue
            score = self._cosine_similarity(query_embedding, doc.embedding)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def query_bm25(
        self,
        tenant_id: str,
        namespace: str,
        query: str,
        top_k: int = None,
    ) -> list[tuple[str, float]]:
        ns_key = self._ns_key(tenant_id, namespace)
        if ns_key not in self._bm25_index:
            return []

        top_k = top_k or settings.rag_top_k
        query_tokens = set(query.lower().split())

        if not self._bm25_index[ns_key]:
            return []

        avg_doc_len = sum(len(d["token_list"]) for d in self._bm25_index[ns_key]) / len(self._bm25_index[ns_key])
        k1 = 1.5
        b = 0.75

        doc_freq = {}
        for d in self._bm25_index[ns_key]:
            for token in d["tokens"]:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        N = len(self._bm25_index[ns_key])
        scores = []

        for d in self._bm25_index[ns_key]:
            score = 0.0
            doc_len = len(d["token_list"])
            for token in query_tokens:
                if token not in d["tokens"]:
                    continue
                df = doc_freq.get(token, 0)
                idf = max(0, ((N - df + 0.5) / (df + 0.5) + 1))
                tf = d["token_list"].count(token)
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
            if score > 0:
                scores.append((d["id"], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def get_document(self, tenant_id: str, namespace: str, doc_id: str) -> Document | None:
        # Try Postgres first
        try:
            pool = await self._db.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, content, namespace, metadata FROM rag_documents WHERE id = $1 AND tenant_id = $2 AND namespace = $3",
                    doc_id, tenant_id, namespace,
                )
                if row:
                    return Document(
                        id=row["id"],
                        content=row["content"],
                        namespace=row["namespace"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    )
        except Exception as e:
            logger.debug("Postgres get_document failed, falling back to in-memory: %s", e)

        ns_key = self._ns_key(tenant_id, namespace)
        return self._store.get(ns_key, {}).get(doc_id)

    async def list_namespaces(self, tenant_id: str) -> list[str]:
        # Try Postgres first
        try:
            pool = await self._db.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT DISTINCT namespace FROM rag_documents WHERE tenant_id = $1",
                    tenant_id,
                )
                if rows:
                    return [row["namespace"] for row in rows]
        except Exception as e:
            logger.debug("Postgres list_namespaces failed, falling back to in-memory: %s", e)

        return [
            k.split(":", 1)[1]
            for k in self._store
            if k.startswith(f"{tenant_id}:")
        ]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

import json
import time
import uuid
import math
import random
import logging
from app.gateway.models import InferRouteRequest, InferRouteResponse
from app.core.config import settings
from app.core.constants import CACHE_KEY_PREFIX_SEMANTIC

logger = logging.getLogger("inferroute")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def xfetch_expired(created_at: float, ttl: int) -> bool:
    age = time.time() - created_at
    if age >= ttl:
        return True
    beta = 1.0
    probability = 1 - math.exp(-(max(age, 0) + beta) / ttl)
    return random.random() < probability


def extract_prompt_text(request: InferRouteRequest) -> str:
    return " ".join(m.content for m in request.messages)


class SemanticCache:

    def __init__(self, redis, embedding_service):
        self._redis = redis
        self._embedding = embedding_service
        self._threshold = settings.cache_similarity_threshold
        self._ttl = settings.cache_ttl_seconds

    def _index_key(self, tenant_id: str, namespace: str = None) -> str:
        ns = namespace or "_default"
        return f"{CACHE_KEY_PREFIX_SEMANTIC}:index:{tenant_id}:{ns}"

    def _entry_key(self, tenant_id: str, entry_id: str, namespace: str = None) -> str:
        ns = namespace or "_default"
        return f"{CACHE_KEY_PREFIX_SEMANTIC}:{tenant_id}:{ns}:{entry_id}"

    async def get(self, tenant_id: str, request: InferRouteRequest) -> InferRouteResponse | None:
        try:
            prompt_text = extract_prompt_text(request)
            query_embedding = await self._embedding.embed(prompt_text)

            namespace = request.namespace
            index_key = self._index_key(tenant_id, namespace)
            entry_ids = await self._redis.smembers(index_key)
            if not entry_ids:
                return None

            best_similarity = 0.0
            best_response = None

            for entry_id in entry_ids:
                entry_key = self._entry_key(tenant_id, entry_id, namespace)
                raw = await self._redis.get(entry_key)
                if raw is None:
                    await self._redis.srem(index_key, entry_id)
                    continue

                entry = json.loads(raw)
                if xfetch_expired(entry["created_at"], self._ttl):
                    await self._redis.delete(entry_key)
                    await self._redis.srem(index_key, entry_id)
                    continue

                sim = cosine_similarity(query_embedding, entry["embedding"])
                if sim > best_similarity:
                    best_similarity = sim
                    best_response = entry

            if best_response is not None and best_similarity >= self._threshold:
                logger.debug(
                    "Semantic cache hit: similarity=%.4f threshold=%.2f",
                    best_similarity, self._threshold,
                )
                return InferRouteResponse(**best_response["response"])
            return None
        except Exception as e:
            logger.debug("Semantic cache miss (error): %s", e)
            return None

    async def set(self, tenant_id: str, request: InferRouteRequest, response: InferRouteResponse):
        try:
            prompt_text = extract_prompt_text(request)
            embedding = await self._embedding.embed(prompt_text)

            namespace = request.namespace
            entry_id = str(uuid.uuid4())
            entry_key = self._entry_key(tenant_id, entry_id, namespace)
            index_key = self._index_key(tenant_id, namespace)

            entry = {
                "embedding": embedding,
                "response": response.model_dump(),
                "created_at": time.time(),
                "prompt": prompt_text,
            }

            ttl_with_jitter = int(self._ttl * (1 + random.uniform(-0.1, 0.1)))
            await self._redis.setex(entry_key, max(1, ttl_with_jitter), json.dumps(entry))
            await self._redis.sadd(index_key, entry_id)
            await self._redis.expire(index_key, max(1, ttl_with_jitter + 60))
        except Exception as e:
            logger.debug("Semantic cache write failed: %s", e)

    async def close(self):
        pass

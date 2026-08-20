import json
import hashlib
import random
import logging
from app.gateway.models import InferRouteRequest, InferRouteResponse
from app.core.config import settings
from app.core.constants import CACHE_KEY_PREFIX_EXACT

logger = logging.getLogger("inferroute")


def canonical_json(request: InferRouteRequest) -> str:
    return json.dumps(
        {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stop": request.stop,
            "namespace": request.namespace,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def request_hash(request: InferRouteRequest) -> str:
    return hashlib.sha256(canonical_json(request).encode()).hexdigest()


def apply_ttl_jitter(ttl: int, jitter: float) -> int:
    delta = ttl * jitter
    return max(1, int(ttl + random.uniform(-delta, delta)))


class ExactMatchCache:

    def __init__(self, redis):
        self._redis = redis
        self._ttl = settings.cache_ttl_seconds
        self._jitter = settings.cache_ttl_jitter

    def _key(self, tenant_id: str, request: InferRouteRequest) -> str:
        return f"{CACHE_KEY_PREFIX_EXACT}:{tenant_id}:{request_hash(request)}"

    async def get(self, tenant_id: str, request: InferRouteRequest) -> InferRouteResponse | None:
        try:
            key = self._key(tenant_id, request)
            data = await self._redis.get(key)
            if data is None:
                return None
            return InferRouteResponse(**json.loads(data))
        except Exception as e:
            logger.debug("Exact cache miss (Redis error): %s", e)
            return None

    async def set(self, tenant_id: str, request: InferRouteRequest, response: InferRouteResponse):
        try:
            key = self._key(tenant_id, request)
            ttl = apply_ttl_jitter(self._ttl, self._jitter)
            await self._redis.setex(key, ttl, response.model_dump_json())
        except Exception as e:
            logger.debug("Exact cache write failed (Redis error): %s", e)

    async def close(self):
        pass

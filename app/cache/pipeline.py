import logging
from app.gateway.models import InferRouteRequest, InferRouteResponse
from app.cache.exact_cache import ExactMatchCache, request_hash
from app.cache.semantic_cache import SemanticCache
from app.cache.coalescer import RequestCoalescer
from app.observability.metrics import (
    cache_hit_total,
    cache_miss_total,
    coalesced_requests_total,
)

logger = logging.getLogger("inferroute")


class CachePipeline:

    def __init__(
        self,
        exact_cache: ExactMatchCache,
        semantic_cache: SemanticCache,
        coalescer: RequestCoalescer,
        routing_service,
    ):
        self._exact = exact_cache
        self._semantic = semantic_cache
        self._coalescer = coalescer
        self._routing = routing_service

    async def process(
        self,
        tenant_id: str,
        request: InferRouteRequest,
        transform: callable = None,
    ) -> tuple[InferRouteResponse, str]:
        # 1. Try exact-match cache
        cached = await self._exact.get(tenant_id, request)
        if cached is not None:
            cache_hit_total.labels(cache_type="exact").inc()
            logger.debug("Exact cache hit for tenant=%s", tenant_id)
            return cached, "exact"

        # 2. Try semantic cache
        cached = await self._semantic.get(tenant_id, request)
        if cached is not None:
            cache_hit_total.labels(cache_type="semantic").inc()
            logger.debug("Semantic cache hit for tenant=%s", tenant_id)
            return cached, "semantic"

        # 3. Coalesce + call upstream
        coalesce_key = f"{tenant_id}:{request_hash(request)}"

        in_flight_before = self._coalescer.in_flight_count

        async def call_upstream() -> InferRouteResponse:
            # Apply transform (e.g. RAG augmentation) before routing.
            # Cache keys are derived from the original request, so the
            # transform only affects the upstream LLM call, not caching.
            route_request = request
            if transform is not None:
                route_request = await transform(request)
            response = await self._routing.route(route_request)
            # Write to both caches on success (keyed by original request)
            await self._exact.set(tenant_id, request, response)
            await self._semantic.set(tenant_id, request, response)
            return response

        response = await self._coalescer.coalesce(coalesce_key, call_upstream)

        if self._coalescer.in_flight_count < in_flight_before:
            # This request was coalesced (the key already existed when we tried)
            coalesced_requests_total.inc()
            return response, "coalesced"

        cache_miss_total.inc()
        return response, "upstream"

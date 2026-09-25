import time
import uuid
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from app.gateway.models import (
    InferRouteRequest,
    InferRouteMessage,
    OpenAIChatCompletionResponse,
)
from app.observability.metrics import (
    request_total,
    request_duration,
    rag_augmented_total,
    rag_context_tokens,
)
from app.observability.tracing import TracingContext
from app.core.database import db
from app.core.config import settings
from app.routing.cost import compute_call_cost

logger = logging.getLogger("inferroute")
router = APIRouter()


def _get_tracing_ctx(request: Request):
    return getattr(request.app.state, "tracing_ctx", None)


def _get_trace_span(request: Request):
    return getattr(request.state, "trace_span", None)


def _record_llm_trace(request: Request, response, provider: str, cache_hit: bool = False,
                      routing_decision: str = "", complexity: str = "", rag_metadata: dict = None):
    """Record LLM-specific attributes on the current trace span."""
    tracing_ctx = _get_tracing_ctx(request)
    span = _get_trace_span(request)
    if tracing_ctx is None or span is None:
        return

    prompt_text = " ".join(m.content for m in getattr(request.state, "internal_request", None).messages if hasattr(m, "content")) if hasattr(request.state, "internal_request") else ""

    attributes = dict(
        prompt=prompt_text[:2000],
        completion=response.content[:2000] if hasattr(response, "content") else "",
        model=response.model if hasattr(response, "model") else "",
        provider=provider,
        input_tokens=response.usage.input_tokens if hasattr(response, "usage") else 0,
        output_tokens=response.usage.output_tokens if hasattr(response, "usage") else 0,
        cache_hit=cache_hit,
        routing_decision=routing_decision,
        complexity=complexity,
        tenant_id=getattr(request.state, "tenant_id", "anonymous"),
    )

    if rag_metadata:
        attributes["rag_namespace"] = rag_metadata.get("namespace", "")
        attributes["rag_retrieved"] = rag_metadata.get("retrieved", 0)
        attributes["rag_context_tokens"] = rag_metadata.get("context_tokens", 0)

    tracing_ctx.set_llm_attributes(span, **attributes)


def _build_trace_headers(request: Request) -> dict | None:
    """Build traceparent header for outgoing provider calls (W3C propagation)."""
    span = _get_trace_span(request)
    if span is None:
        return None
    return {"traceparent": f"00-{span.trace_id}-{span.span_id}-01"}


async def _enforce_quota(request: Request, tenant_id: str, tokens: int):
    """Check and consume token quota. Raises HTTPException if exceeded."""
    quota_enforcer = getattr(request.app.state, "quota_enforcer", None)
    if quota_enforcer is None:
        return None
    result = await quota_enforcer.check_and_consume(tenant_id, tokens)
    if not result.allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "daily_remaining": result.daily_remaining,
                "monthly_remaining": result.monthly_remaining,
                "reset_at": result.reset_at,
            },
            headers={"Retry-After": str(result.reset_at)},
        )
    return result


async def _log_usage(tenant_id: str, provider: str, model: str,
                     input_tokens: int, output_tokens: int,
                     cache_hit: bool = False, routing_decision: str = "",
                     trace_id: str = "", cost_usd: float = 0.0):
    """Log token usage to Postgres for billing/analytics."""
    try:
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO usage_log
                (tenant_id, provider, model, input_tokens, output_tokens, total_tokens, cost_usd, cache_hit, routing_decision, trace_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                tenant_id, provider, model, input_tokens, output_tokens,
                input_tokens + output_tokens, cost_usd, cache_hit, routing_decision, trace_id,
            )
    except Exception as e:
        logger.debug("Usage log failed (non-fatal): %s", e)


def _build_rag_transform(rag_service, tenant_id: str, namespace: str, top_k: int = 3):
    """Build a transform callback that augments the prompt with RAG context.

    Returns (transform_callback, rag_metadata_holder).
    rag_metadata_holder is a dict that will be populated with retrieval info
    after the transform runs (on cache miss).
    """
    rag_metadata = {"namespace": namespace, "retrieved": 0, "sources": [], "context_tokens": 0, "cache_hit": False}

    async def transform(original_request: InferRouteRequest) -> InferRouteRequest:
        # Extract the last user message for RAG query
        user_message = ""
        for msg in reversed(original_request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        if not user_message:
            return original_request

        # Retrieve relevant documents
        results = await rag_service.query(tenant_id, namespace, user_message, top_k=top_k)
        if not results:
            return original_request

        # Build context block
        context_parts = []
        sources = []
        for r in results:
            context_parts.append(r["content"])
            sources.append({
                "doc_id": r["doc_id"],
                "score": r["score"],
                "source": r["source"],
            })

        context = "\n\n".join(context_parts)
        context_tokens = len(context) // 4  # rough estimate

        # Update metadata
        rag_metadata["retrieved"] = len(results)
        rag_metadata["sources"] = sources
        rag_metadata["context_tokens"] = context_tokens

        # Track metrics
        rag_augmented_total.labels(tenant_id=tenant_id, namespace=namespace).inc()
        rag_context_tokens.labels(namespace=namespace).observe(context_tokens)

        # Create augmented request (copy, don't mutate original — original is used for cache keying)
        augmented_messages = []
        for i, msg in enumerate(original_request.messages):
            if msg.role == "user" and i == len(original_request.messages) - 1:
                augmented_content = f"Context:\n{context}\n\nQuestion: {msg.content}"
                augmented_messages.append(InferRouteMessage(role=msg.role, content=augmented_content))
            else:
                augmented_messages.append(InferRouteMessage(role=msg.role, content=msg.content))

        return InferRouteRequest(
            messages=augmented_messages,
            model=original_request.model,
            max_tokens=original_request.max_tokens,
            temperature=original_request.temperature,
            stream=original_request.stream,
            stop=original_request.stop,
            namespace=original_request.namespace,
        )

    return transform, rag_metadata


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    try:
        internal_request = InferRouteRequest(
            messages=[InferRouteMessage(**m) for m in body["messages"]],
            model=body["model"],
            max_tokens=body.get("max_tokens"),
            temperature=body.get("temperature", 1.0),
            stream=body.get("stream", False),
            stop=body.get("stop"),
            namespace=body.get("namespace"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    request.state.internal_request = internal_request
    tenant_id = getattr(request.state, "tenant_id", "anonymous")
    trace_headers = _build_trace_headers(request)

    # Build extra headers for routing (trace + config-based routing)
    extra_headers = trace_headers or {}
    config_id = request.headers.get("x-config-id")
    if config_id:
        extra_headers["x-config-id"] = config_id
    extra_headers["x-tenant-id"] = tenant_id

    # RAG augmentation setup (only if namespace provided and RAG service available)
    rag_transform = None
    rag_metadata = None
    namespace = internal_request.namespace
    if namespace:
        rag_service = getattr(request.app.state, "rag_service", None)
        if rag_service is not None:
            rag_transform, rag_metadata = _build_rag_transform(rag_service, tenant_id, namespace)

    # Streaming path
    if internal_request.stream:
        routing_service = request.app.state.routing_service
        if routing_service is None:
            raise HTTPException(status_code=503, detail="No providers available")

        async def sse_generator():
            provider_id = "unknown"
            try:
                async for chunk, provider_id in routing_service.stream_route(internal_request, extra_headers=extra_headers):
                    yield f"{chunk}\n\n"
            except Exception as e:
                request_total.labels(provider=provider_id, status="error", tenant_id=tenant_id).inc()
                yield f"data: {{\"error\": \"{e}\"}}\n\n"
            request_total.labels(provider=provider_id, status="success", tenant_id=tenant_id).inc()

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Trace-Id": getattr(request.state, "trace_id", ""),
            },
        )

    # A/B routing
    ab_router = getattr(request.app.state, "ab_router", None)
    variant = None
    if ab_router is not None:
        variant = ab_router.assign_variant(tenant_id, body.get("user", ""))

    # Classify complexity for routing metadata
    complexity = "medium"
    tier = "standard"
    try:
        registry = getattr(request.app.state, "registry", None)
        if registry and hasattr(registry.strategy_instance, "classify_complexity"):
            complexity, _ = await registry.strategy_instance.classify_complexity(internal_request)
            tier_map = {"simple": "cheap", "medium": "standard", "complex": "premium"}
            tier = tier_map.get(complexity, "standard")
    except Exception:
        pass

    # Try cache pipeline first
    cache_pipeline = getattr(request.app.state, "cache_pipeline", None)
    if cache_pipeline is not None:
        start = time.monotonic()
        try:
            response, source = await cache_pipeline.process(
                tenant_id, internal_request, transform=rag_transform,
            )
        except Exception as e:
            request_total.labels(
                provider="unknown", status="error", tenant_id=tenant_id,
            ).inc()
            raise HTTPException(status_code=502, detail=f"Provider error: {e}")

        duration = time.monotonic() - start
        request_total.labels(
            provider=response.provider, status="success", tenant_id=tenant_id,
        ).inc()
        request_duration.labels(provider=response.provider).observe(duration)

        # Record LLM trace
        _record_llm_trace(
            request, response,
            provider=response.provider,
            cache_hit=(source != "miss"),
            routing_decision=f"cache:{source}",
            rag_metadata=rag_metadata if rag_metadata and rag_metadata["retrieved"] > 0 else None,
        )

        # Enforce quota + log usage
        total_tokens = response.usage.input_tokens + response.usage.output_tokens
        await _enforce_quota(request, tenant_id, total_tokens)

        # Compute cost (cache hits cost $0 — the original call already paid)
        cost_usd = 0.0
        if source == "miss":
            registry = getattr(request.app.state, "registry", None)
            if registry is not None:
                provider_health = next(
                    (p for p in registry.get_all() if p.provider_id == response.provider),
                    None,
                )
                if provider_health is not None:
                    cost_usd = compute_call_cost(
                        provider_health.cost_per_1k_input,
                        provider_health.cost_per_1k_output,
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                    )

        await _log_usage(
            tenant_id, response.provider, response.model,
            response.usage.input_tokens, response.usage.output_tokens,
            cache_hit=(source != "miss"), routing_decision=f"cache:{source}",
            trace_id=getattr(request.state, "trace_id", ""),
            cost_usd=cost_usd,
        )

        # Record A/B outcome
        if ab_router is not None and variant:
            ab_router.record_outcome(
                variant,
                cost=response.usage.input_tokens + response.usage.output_tokens,
                latency_ms=duration * 1000,
            )

        # Shadow traffic (fire-and-forget, non-blocking)
        shadow_mgr = getattr(request.app.state, "shadow_traffic", None)
        if shadow_mgr is not None:
            registry = getattr(request.app.state, "registry", None)
            if registry is not None:
                target_provider = getattr(shadow_mgr, "_target_provider", "vllm")
                for p in registry.get_all():
                    if p.provider_id == target_provider:
                        await shadow_mgr.maybe_shadow(internal_request, p.adapter)
                        break

        # Build response (include x_rag only when RAG was used)
        response_obj = OpenAIChatCompletionResponse(
            id=response.id or f"chatcmpl-{response.provider}",
            model=response.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.content},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )
        if rag_metadata and rag_metadata["retrieved"] > 0:
            rag_metadata["cache_hit"] = source != "miss" and source != "upstream"
            response_obj = response_obj.model_dump()
            response_obj["x_rag"] = rag_metadata
        # Add routing metadata for observability
        if isinstance(response_obj, dict):
            response_obj["cost_usd"] = cost_usd
            response_obj["x_routing"] = {
                "provider": response.provider,
                "model": response.model,
                "complexity": complexity,
                "tier": tier,
                "cache_status": source if source != "miss" else "miss",
                "strategy": getattr(request.app.state, "strategy_name", "intelligence_aware"),
                "latency_ms": round(duration * 1000, 1),
            }
        return response_obj

    # Fallback: no cache pipeline, route directly
    routing_service = request.app.state.routing_service
    if routing_service is None:
        raise HTTPException(status_code=503, detail="No providers available")

    # Apply RAG augmentation if needed (no cache in this path)
    route_request = internal_request
    if rag_transform is not None:
        route_request = await rag_transform(internal_request)

    start = time.monotonic()
    try:
        response = await routing_service.route(route_request, extra_headers=extra_headers)
    except Exception as e:
        request_total.labels(
            provider="unknown", status="error", tenant_id=tenant_id,
        ).inc()
        raise HTTPException(status_code=502, detail=f"Provider error: {e}")

    duration = time.monotonic() - start
    request_total.labels(
        provider=response.provider, status="success", tenant_id=tenant_id,
    ).inc()
    request_duration.labels(provider=response.provider).observe(duration)

    # Record LLM trace
    _record_llm_trace(
        request, response,
        provider=response.provider,
        cache_hit=False,
        routing_decision="direct",
        rag_metadata=rag_metadata if rag_metadata and rag_metadata["retrieved"] > 0 else None,
    )

    # Enforce quota + log usage
    total_tokens = response.usage.input_tokens + response.usage.output_tokens
    await _enforce_quota(request, tenant_id, total_tokens)

    # Compute cost
    cost_usd = 0.0
    registry = getattr(request.app.state, "registry", None)
    if registry is not None:
        provider_health = next(
            (p for p in registry.get_all() if p.provider_id == response.provider),
            None,
        )
        if provider_health is not None:
            cost_usd = compute_call_cost(
                provider_health.cost_per_1k_input,
                provider_health.cost_per_1k_output,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

    await _log_usage(
        tenant_id, response.provider, response.model,
        response.usage.input_tokens, response.usage.output_tokens,
        cache_hit=False, routing_decision="direct",
        trace_id=getattr(request.state, "trace_id", ""),
        cost_usd=cost_usd,
    )

    response_obj = OpenAIChatCompletionResponse(
        id=response.id or f"chatcmpl-{response.provider}",
        model=response.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.content},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        },
    )
    if rag_metadata and rag_metadata["retrieved"] > 0:
        rag_metadata["cache_hit"] = False
        response_obj = response_obj.model_dump()
        response_obj["x_rag"] = rag_metadata
    # Add routing metadata for observability
    if isinstance(response_obj, dict):
        response_obj["cost_usd"] = cost_usd
        response_obj["x_routing"] = {
            "provider": response.provider,
            "model": response.model,
            "complexity": complexity,
            "tier": tier,
            "cache_status": "miss",
            "strategy": getattr(request.app.state, "strategy_name", "intelligence_aware"),
            "latency_ms": round(duration * 1000, 1),
        }
    return response_obj


@router.post("/v1/feedback")
async def submit_feedback(request: Request):
    body = await request.json()

    trace_id = body.get("trace_id")
    thumbs = body.get("thumbs")
    comment = body.get("comment", "")

    if not trace_id or thumbs not in ("up", "down"):
        raise HTTPException(status_code=400, detail="trace_id and thumbs (up/down) required")

    feedback_store = getattr(request.app.state, "feedback_store", None)
    if feedback_store is None:
        raise HTTPException(status_code=503, detail="Feedback storage not available")

    await feedback_store.record(trace_id=trace_id, thumbs=thumbs, comment=comment)
    return {"status": "recorded", "trace_id": trace_id, "thumbs": thumbs}


@router.get("/v1/feedback/stats")
async def feedback_stats(request: Request):
    feedback_store = getattr(request.app.state, "feedback_store", None)
    if feedback_store is None:
        return {"total": 0}
    return await feedback_store.get_stats()


@router.get("/v1/experiments/stats")
async def experiment_stats(request: Request):
    ab_router = getattr(request.app.state, "ab_router", None)
    if ab_router is None:
        return {"experiments": "disabled"}
    return ab_router.get_stats()


# --- RAG Endpoints ---

@router.post("/v1/embeddings")
async def create_embeddings(request: Request):
    body = await request.json()
    text = body.get("input")
    if not text:
        raise HTTPException(status_code=400, detail="input required")

    rag_embedding = getattr(request.app.state, "rag_embedding", None)
    if rag_embedding is None:
        raise HTTPException(status_code=503, detail="Embedding service not available")

    if isinstance(text, str):
        embedding = await rag_embedding.embed(text)
        return {"data": [{"embedding": embedding, "index": 0}], "model": rag_embedding._model}
    else:
        embeddings = await rag_embedding.embed_batch(text)
        return {"data": [{"embedding": e, "index": i} for i, e in enumerate(embeddings)], "model": rag_embedding._model}


@router.post("/v1/rag/query")
async def rag_query(request: Request):
    body = await request.json()
    tenant_id = getattr(request.state, "tenant_id", "anonymous")
    namespace = body.get("namespace", "default")
    query = body.get("query")
    top_k = body.get("top_k")

    if not query:
        raise HTTPException(status_code=400, detail="query required")

    rag_service = getattr(request.app.state, "rag_service", None)
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not available")

    results = await rag_service.query(tenant_id, namespace, query, top_k=top_k)
    return {"results": results, "namespace": namespace, "count": len(results)}


@router.post("/v1/rag/upsert")
async def rag_upsert(request: Request):
    body = await request.json()
    tenant_id = getattr(request.state, "tenant_id", "anonymous")
    namespace = body.get("namespace", "default")
    documents = body.get("documents", [])

    if not documents:
        raise HTTPException(status_code=400, detail="documents required")

    rag_service = getattr(request.app.state, "rag_service", None)
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not available")

    rag_embedding = getattr(request.app.state, "rag_embedding", None)
    embeddings = None
    if rag_embedding:
        texts = [d["content"] for d in documents]
        embeddings = await rag_embedding.embed_batch(texts)

    count = await rag_service.upsert(tenant_id, namespace, documents, embeddings)
    return {"status": "upserted", "count": count, "namespace": namespace}


# --- MCP Endpoints ---

@router.post("/v1/mcp/tools/{tool_name}/invoke")
async def mcp_invoke(tool_name: str, request: Request):
    body = await request.json()
    payload = body.get("payload", body)

    mcp_gateway = getattr(request.app.state, "mcp_gateway", None)
    if mcp_gateway is None:
        raise HTTPException(status_code=503, detail="MCP gateway not available")

    try:
        result = await mcp_gateway.invoke(tool_name, payload)
        return {"status": "ok", "tool": tool_name, "result": result}
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/v1/mcp/tools/{tool_name}/register")
async def mcp_register(tool_name: str, request: Request):
    body = await request.json()
    replica_id = body.get("replica_id")
    endpoint = body.get("endpoint")

    if not replica_id or not endpoint:
        raise HTTPException(status_code=400, detail="replica_id and endpoint required")

    mcp_registry = getattr(request.app.state, "mcp_registry", None)
    if mcp_registry is None:
        raise HTTPException(status_code=503, detail="MCP registry not available")

    mcp_registry.register(tool_name, replica_id, endpoint)
    return {"status": "registered", "tool": tool_name, "replica": replica_id}


@router.get("/v1/mcp/tools")
async def mcp_list_tools(request: Request):
    mcp_registry = getattr(request.app.state, "mcp_registry", None)
    if mcp_registry is None:
        return {"tools": []}
    return {"tools": mcp_registry.list_tools(), "replicas": mcp_registry.get_status()}


# --- Enhanced Health ---

@router.get("/v1/health")
async def health(request: Request):
    routing_service = request.app.state.routing_service
    if routing_service is None:
        return {"status": "starting", "providers": []}

    providers = routing_service.get_provider_status()

    cache_status = "disabled"
    if hasattr(request.app.state, "cache_pipeline") and request.app.state.cache_pipeline:
        cache_status = "enabled"

    mcp_status = []
    mcp_registry = getattr(request.app.state, "mcp_registry", None)
    if mcp_registry:
        mcp_status = mcp_registry.get_status()

    return {
        "status": "ok",
        "service": "inferroute",
        "providers": providers,
        "cache": cache_status,
        "mcp_replicas": mcp_status,
        "rate_limiting": "enabled" if getattr(request.app.state, "rate_limiter", None) else "disabled",
        "experiments": "enabled" if getattr(request.app.state, "ab_router", None) else "disabled",
        "rag": "enabled" if getattr(request.app.state, "rag_service", None) else "disabled",
        "tracing": "enabled" if getattr(request.app.state, "tracing", None) else "disabled",
    }


# --- Usage & Quota ---

@router.get("/v1/usage")
async def get_usage(request: Request):
    """Per-tenant token usage and quota status."""
    tenant_id = getattr(request.state, "tenant_id", "anonymous")
    quota_enforcer = getattr(request.app.state, "quota_enforcer", None)

    # Get current quota from Redis
    daily_used = 0
    monthly_used = 0
    daily_limit = settings.quota_daily_token_limit
    monthly_limit = settings.quota_monthly_token_limit
    reset_at = 0

    if quota_enforcer:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        month = datetime.now(timezone.utc).strftime("%Y%m")
        daily_key = f"quota:daily:{tenant_id}:{today}"
        monthly_key = f"quota:monthly:{tenant_id}:{month}"

        redis = getattr(request.app.state, "redis", None)
        if redis:
            try:
                pipe = redis.pipeline()
                await pipe.get(daily_key)
                await pipe.get(monthly_key)
                results = await pipe.execute()
                daily_used = int(results[0] or 0)
                monthly_used = int(results[1] or 0)
            except Exception:
                pass

        reset_at = int(time.time()) + quota_enforcer._seconds_until_midnight_utc()

    # Get usage breakdown from Postgres
    by_provider = {}
    by_model = {}
    total_requests = 0

    try:
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT provider, model,
                    SUM(input_tokens) as input_tokens,
                    SUM(output_tokens) as output_tokens,
                    SUM(total_tokens) as total_tokens,
                    COUNT(*) as request_count
                FROM usage_log
                WHERE tenant_id = $1 AND created_at >= date_trunc('day', NOW())
                GROUP BY provider, model""",
                tenant_id,
            )
            for row in rows:
                p = row["provider"]
                m = row["model"]
                by_provider[p] = by_provider.get(p, 0) + row["total_tokens"]
                by_model[m] = by_model.get(m, 0) + row["total_tokens"]
                total_requests += row["request_count"]
    except Exception as e:
        logger.debug("Usage query from Postgres failed: %s", e)

    return {
        "tenant_id": tenant_id,
        "daily": {
            "used": daily_used,
            "limit": daily_limit,
            "remaining": max(0, daily_limit - daily_used),
        },
        "monthly": {
            "used": monthly_used,
            "limit": monthly_limit,
            "remaining": max(0, monthly_limit - monthly_used),
        },
        "reset_at": reset_at,
        "today_requests": total_requests,
        "by_provider": by_provider,
        "by_model": by_model,
    }


# --- Routing Configs (Portkey-style per-tenant config management) ---


@router.get("/v1/routing/configs")
async def list_routing_configs(request: Request):
    """List all routing configs for the current tenant."""
    config_registry = getattr(request.app.state, "config_registry", None)
    if config_registry is None:
        raise HTTPException(status_code=503, detail="Config registry not initialized")
    tenant_id = getattr(request.state, "tenant_id", "default")
    configs = await config_registry.list_configs(tenant_id)
    return {"tenant_id": tenant_id, "configs": configs}


@router.post("/v1/routing/configs")
async def create_routing_config(request: Request):
    """Create or update a routing config.

    Body:
    {
        "config_id": "cfg_decide",
        "name": "Decision node — premium models",
        "targets": [
            {"provider": "openai", "model": "gpt-4o", "tier": "premium", "weight": 1.0},
            {"provider": "anthropic", "model": "claude-sonnet", "tier": "premium", "weight": 1.0}
        ],
        "fallback_targets": [
            {"provider": "groq", "model": "llama-3.3-70b", "tier": "standard", "weight": 1.0}
        ],
        "latency_budget_ms": null,
        "cost_ceiling": null,
        "retry_attempts": 3
    }
    """
    config_registry = getattr(request.app.state, "config_registry", None)
    if config_registry is None:
        raise HTTPException(status_code=503, detail="Config registry not initialized")

    tenant_id = getattr(request.state, "tenant_id", "default")
    body = await request.json()

    from app.routing.config_registry import RoutingConfig, RoutingTarget

    config = RoutingConfig(
        config_id=body["config_id"],
        tenant_id=tenant_id,
        name=body.get("name", ""),
        targets=[RoutingTarget(**t) for t in body.get("targets", [])],
        fallback_targets=[RoutingTarget(**t) for t in body.get("fallback_targets", [])],
        latency_budget_ms=body.get("latency_budget_ms"),
        cost_ceiling=body.get("cost_ceiling"),
        retry_attempts=body.get("retry_attempts", 3),
    )

    success = await config_registry.create(config)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save config")

    return {"status": "created", "config_id": config.config_id, "tenant_id": tenant_id}


@router.get("/v1/routing/configs/{config_id}")
async def get_routing_config(config_id: str, request: Request):
    """Get a specific routing config."""
    config_registry = getattr(request.app.state, "config_registry", None)
    if config_registry is None:
        raise HTTPException(status_code=503, detail="Config registry not initialized")
    tenant_id = getattr(request.state, "tenant_id", "default")
    config = await config_registry.get(tenant_id, config_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Config not found")
    return config.to_json()


@router.delete("/v1/routing/configs/{config_id}")
async def delete_routing_config(config_id: str, request: Request):
    """Delete a routing config."""
    config_registry = getattr(request.app.state, "config_registry", None)
    if config_registry is None:
        raise HTTPException(status_code=503, detail="Config registry not initialized")
    tenant_id = getattr(request.state, "tenant_id", "default")
    success = await config_registry.delete(tenant_id, config_id)
    if not success:
        raise HTTPException(status_code=404, detail="Config not found")
    return {"status": "deleted", "config_id": config_id}


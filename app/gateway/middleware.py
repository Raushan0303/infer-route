import time
import logging
from fastapi import Request, Response
from starlette.responses import JSONResponse
from app.observability.metrics import rate_limit_rejected_total
from app.observability.tracing import TracingContext

logger = logging.getLogger("inferroute")


def resolve_tenant(request: Request) -> str:
    """Set request.state.api_key / tenant_id from the Authorization header (idempotent)."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id is not None:
        return tenant_id
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        request.state.api_key = auth[7:]
        request.state.tenant_id = auth[7:]
    else:
        request.state.api_key = None
        request.state.tenant_id = "anonymous"
    return request.state.tenant_id


async def auth_middleware(request: Request, call_next):
    resolve_tenant(request)
    response = await call_next(request)
    return response


async def rate_limit_middleware(request: Request, call_next):
    if request.method == "GET" and request.url.path == "/v1/health":
        return await call_next(request)

    rate_limiter = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter is None:
        return await call_next(request)

    # Resolve here too, so the bucket key never depends on middleware order.
    tenant_id = resolve_tenant(request)
    result = await rate_limiter.check(tenant_id)

    if not result.allowed:
        rate_limit_rejected_total.labels(tenant_id=tenant_id).inc()
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded", "retry_after": result.retry_after},
            headers={
                "X-RateLimit-Remaining": str(result.remaining),
                "Retry-After": str(result.retry_after),
            },
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(result.remaining)
    return response


async def tracing_middleware(request: Request, call_next):
    """Extract W3C traceparent from incoming headers and create root span."""
    tracing_ctx = getattr(request.app.state, "tracing_ctx", None)
    if tracing_ctx is None:
        return await call_next(request)

    # Extract W3C traceparent from AgentMesh/upstream
    headers = dict(request.headers)
    traceparent = TracingContext.extract_traceparent(headers)

    trace_id = traceparent["trace_id"] if traceparent else None
    parent_id = traceparent["parent_id"] if traceparent else ""

    # Start root span for this request
    span = tracing_ctx.start_span(
        name=f"{request.method} {request.url.path}",
        trace_id=trace_id,
        parent_id=parent_id,
        attributes={
            "http.method": request.method,
            "http.url": str(request.url),
            "tenant_id": getattr(request.state, "tenant_id", "anonymous"),
        },
    )

    # Store on request state for routes to use
    request.state.trace_span = span
    request.state.trace_id = span.trace_id

    try:
        response = await call_next(request)

        # Record response status
        tracing_ctx.set_llm_attributes(span, attributes={**span.attributes, "http.status_code": response.status_code})

        # Add trace IDs to response headers for client correlation
        response.headers["X-Trace-Id"] = span.trace_id
        response.headers["X-Span-Id"] = span.span_id

        return response
    finally:
        tracing_ctx.end_span(span)


async def metrics_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start

    logger.info(
        "request method=%s path=%s status=%d duration=%.3fs",
        request.method, request.url.path, response.status_code, duration,
    )

    return response

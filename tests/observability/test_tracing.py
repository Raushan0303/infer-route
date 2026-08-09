import pytest
import time
from app.observability.tracing import TracingContext
from app.observability.tracing_backend import NoopBackend, LLMTraceSpan


def test_extract_traceparent_valid():
    headers = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
    result = TracingContext.extract_traceparent(headers)
    assert result is not None
    assert result["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
    assert result["parent_id"] == "b7ad6b7169203331"


def test_extract_traceparent_invalid():
    assert TracingContext.extract_traceparent({}) is None
    assert TracingContext.extract_traceparent({"traceparent": "invalid"}) is None


def test_extract_traceparent_case_insensitive():
    headers = {"Traceparent": "00-abc123-def456-01"}
    result = TracingContext.extract_traceparent(headers)
    assert result is not None
    assert result["trace_id"] == "abc123"


def test_inject_traceparent():
    headers = {}
    TracingContext.inject_traceparent(headers, "trace123", "span456")
    assert "traceparent" in headers
    assert "trace123" in headers["traceparent"]
    assert "span456" in headers["traceparent"]


def test_start_and_end_span():
    ctx = TracingContext(backend=NoopBackend())
    span = ctx.start_span("test_span", attributes={"provider": "openai"})
    assert span.name == "test_span"
    assert span.attributes["provider"] == "openai"
    assert span.end_time == 0.0

    ctx.end_span(span)
    assert span.end_time > span.start_time
    assert span.latency_ms > 0


def test_set_llm_attributes():
    ctx = TracingContext(backend=NoopBackend())
    span = ctx.start_span("llm_call")
    ctx.set_llm_attributes(
        span,
        prompt="hello",
        completion="hi there",
        model="gpt-4o",
        provider="openai",
        input_tokens=5,
        output_tokens=3,
        cache_hit=False,
    )
    assert span.prompt == "hello"
    assert span.completion == "hi there"
    assert span.model == "gpt-4o"
    assert span.provider == "openai"
    assert span.input_tokens == 5
    assert span.output_tokens == 3
    assert span.cache_hit is False
    ctx.end_span(span)


def test_add_event():
    ctx = TracingContext(backend=NoopBackend())
    span = ctx.start_span("test_span")
    ctx.add_event(span, "cache_hit", {"key": "abc123"})
    # NoopBackend doesn't store events, but should not error
    ctx.end_span(span)


def test_traceparent_propagation():
    """Test that trace_id from incoming traceparent is used for the span."""
    ctx = TracingContext(backend=NoopBackend())
    headers = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
    traceparent = TracingContext.extract_traceparent(headers)

    span = ctx.start_span(
        "llm_request",
        trace_id=traceparent["trace_id"],
        parent_id=traceparent["parent_id"],
    )
    assert span.trace_id == "0af7651916cd43dd8448eb211c80319c"
    assert span.parent_id == "b7ad6b7169203331"
    ctx.end_span(span)


def test_noop_backend():
    backend = NoopBackend()
    span = backend.start_span("test")
    assert span.name == "test"
    backend.set_llm_attributes(span, provider="openai", model="gpt-4o")
    assert span.provider == "openai"
    backend.end_span(span)
    assert span.latency_ms > 0
    backend.flush()  # should not error


def test_clear():
    ctx = TracingContext(backend=NoopBackend())
    ctx.start_span("span1")
    ctx.clear()
    assert len(ctx._active_spans) == 0

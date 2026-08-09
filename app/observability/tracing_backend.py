import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace, propagate
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

logger = logging.getLogger("inferroute")


@dataclass
class LLMTraceSpan:
    """Represents a single LLM trace span with all relevant metadata."""
    trace_id: str
    span_id: str
    parent_id: str = ""
    name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    # LLM-specific fields
    prompt: str = ""
    completion: str = ""
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    routing_decision: str = ""
    complexity: str = ""
    cache_hit: bool = False
    tenant_id: str = ""
    latency_ms: float = 0.0


class TracingBackend(ABC):
    """Abstract tracing backend — pluggable for Langfuse, LangSmith, or others."""

    @abstractmethod
    def start_span(self, name: str, trace_id: str = None, parent_id: str = "",
                   attributes: dict = None) -> LLMTraceSpan:
        ...

    @abstractmethod
    def end_span(self, span: LLMTraceSpan):
        ...

    @abstractmethod
    def add_event(self, span: LLMTraceSpan, name: str, attributes: dict = None):
        ...

    @abstractmethod
    def set_llm_attributes(self, span: LLMTraceSpan, **kwargs):
        ...

    @abstractmethod
    def flush(self):
        ...


class LangfuseBackend(TracingBackend):
    """Langfuse tracing backend — self-hosted or cloud. Uses Langfuse v4 API."""

    def __init__(self, public_key: str, secret_key: str, host: str):
        from langfuse import Langfuse

        self._langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        self._observations: dict[str, Any] = {}
        logger.info("Langfuse tracing backend initialized: host=%s", host)

    def start_span(self, name: str, trace_id: str = None, parent_id: str = "",
                   attributes: dict = None) -> LLMTraceSpan:
        span_id = uuid.uuid4().hex[:16]

        # Validate trace_id — W3C requires 32-char lowercase hex
        # Langfuse will crash on non-hex trace IDs
        valid_trace_id = None
        if trace_id:
            try:
                int(trace_id, 16)
                if len(trace_id) == 32:
                    valid_trace_id = trace_id
                else:
                    valid_trace_id = trace_id.zfill(32).lower()[-32:]
            except ValueError:
                valid_trace_id = uuid.uuid4().hex
                logger.warning("Invalid trace_id '%s' — generated new one", trace_id)

        trace_context = None
        if valid_trace_id:
            trace_context = {"trace_id": valid_trace_id}
            if parent_id:
                trace_context["parent_span_id"] = parent_id

        lf_obs = self._langfuse.start_observation(
            name=name,
            as_type="span",
            trace_context=trace_context,
        )

        span = LLMTraceSpan(
            trace_id=lf_obs.trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=name,
            start_time=time.time(),
            attributes=attributes or {},
        )
        self._observations[span_id] = lf_obs
        return span

    def end_span(self, span: LLMTraceSpan):
        span.end_time = time.time()
        span.latency_ms = (span.end_time - span.start_time) * 1000

        lf_obs = self._observations.get(span.span_id)
        if lf_obs:
            lf_obs.update(
                input=span.prompt[:2000] if span.prompt else None,
                output=span.completion[:2000] if span.completion else None,
                model=span.model or None,
                metadata={
                    "provider": span.provider,
                    "input_tokens": span.input_tokens,
                    "output_tokens": span.output_tokens,
                    "cost": span.cost,
                    "routing_decision": span.routing_decision,
                    "complexity": span.complexity,
                    "cache_hit": span.cache_hit,
                    "tenant_id": span.tenant_id,
                    "latency_ms": span.latency_ms,
                    **span.attributes,
                },
                usage_details={
                    "input": span.input_tokens,
                    "output": span.output_tokens,
                } if span.input_tokens or span.output_tokens else None,
            )
            lf_obs.end()
            del self._observations[span.span_id]

    def add_event(self, span: LLMTraceSpan, name: str, attributes: dict = None):
        lf_obs = self._observations.get(span.span_id)
        if lf_obs:
            lf_obs.create_event(name=name, metadata=attributes or {})

    def set_llm_attributes(self, span: LLMTraceSpan, **kwargs):
        for key, value in kwargs.items():
            if hasattr(span, key):
                setattr(span, key, value)
            else:
                span.attributes[key] = value

    def flush(self):
        self._langfuse.flush()


class LangSmithBackend(TracingBackend):
    """LangSmith tracing backend — hosted by LangChain."""

    def __init__(self, api_key: str, project: str, endpoint: str):
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        logger.info("LangSmith tracing backend initialized: project=%s", project)
        self._spans: dict[str, LLMTraceSpan] = {}

    def start_span(self, name: str, trace_id: str = None, parent_id: str = "",
                   attributes: dict = None) -> LLMTraceSpan:
        span_id = uuid.uuid4().hex[:16]
        trace_id = trace_id or uuid.uuid4().hex

        span = LLMTraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=name,
            start_time=time.time(),
            attributes=attributes or {},
        )
        self._spans[span_id] = span
        return span

    def end_span(self, span: LLMTraceSpan):
        span.end_time = time.time()
        span.latency_ms = (span.end_time - span.start_time) * 1000
        logger.debug(
            "LangSmith span ended: trace=%s name=%s duration=%.1fms provider=%s",
            span.trace_id, span.name, span.latency_ms, span.provider,
        )
        self._spans.pop(span.span_id, None)

    def add_event(self, span: LLMTraceSpan, name: str, attributes: dict = None):
        span.events.append({
            "name": name,
            "attributes": attributes or {},
            "timestamp": time.time(),
        })

    def set_llm_attributes(self, span: LLMTraceSpan, **kwargs):
        for key, value in kwargs.items():
            if hasattr(span, key):
                setattr(span, key, value)
            else:
                span.attributes[key] = value

    def flush(self):
        pass


class NoopBackend(TracingBackend):
    """No-op backend when tracing is disabled."""

    def start_span(self, name: str, trace_id: str = None, parent_id: str = "",
                   attributes: dict = None) -> LLMTraceSpan:
        return LLMTraceSpan(
            trace_id=trace_id or uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent_id,
            name=name,
            start_time=time.time(),
            attributes=attributes or {},
        )

    def end_span(self, span: LLMTraceSpan):
        span.end_time = time.time()
        span.latency_ms = (span.end_time - span.start_time) * 1000

    def add_event(self, span: LLMTraceSpan, name: str, attributes: dict = None):
        pass

    def set_llm_attributes(self, span: LLMTraceSpan, **kwargs):
        for key, value in kwargs.items():
            if hasattr(span, key):
                setattr(span, key, value)

    def flush(self):
        pass


def create_tracing_backend(service: str = None, **kwargs) -> TracingBackend:
    """Factory to create the configured tracing backend."""
    from app.core.config import settings

    service = service or settings.tracing_service

    if not settings.tracing_enabled:
        logger.info("Tracing disabled, using NoopBackend")
        return NoopBackend()

    if service == "langfuse":
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.warning("Langfuse keys not configured, falling back to NoopBackend")
            return NoopBackend()
        return LangfuseBackend(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )

    elif service == "langsmith":
        if not settings.langsmith_api_key:
            logger.warning("LangSmith API key not configured, falling back to NoopBackend")
            return NoopBackend()
        return LangSmithBackend(
            api_key=settings.langsmith_api_key,
            project=settings.langsmith_project,
            endpoint=settings.langsmith_endpoint,
        )

    else:
        logger.warning("Unknown tracing service '%s', using NoopBackend", service)
        return NoopBackend()

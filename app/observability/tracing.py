import uuid
import logging
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace, propagate
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from app.core.config import settings
from app.observability.tracing_backend import (
    TracingBackend,
    LLMTraceSpan,
    create_tracing_backend,
)

logger = logging.getLogger("inferroute")


class TracingContext:
    """Manages trace spans with W3C traceparent propagation, OTLP export to Jaeger,
    and pluggable LLM-specific backend (Langfuse/LangSmith)."""

    def __init__(self, backend: TracingBackend = None):
        self._active_spans: dict[str, LLMTraceSpan] = {}
        self._otel_spans: dict[str, Any] = {}

        # Set up OpenTelemetry tracer with OTLP exporter for Jaeger
        # Must be done BEFORE Langfuse initializes (it sets its own provider)
        resource = Resource.create({
            "service.name": "infer-route",
            "service.version": "0.1.0",
        })
        provider = TracerProvider(resource=resource)

        # Add OTLP HTTP exporter → Jaeger
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=settings.otlp_exporter_endpoint,
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info("OTLP exporter configured: %s", settings.otlp_exporter_endpoint)
        except ImportError:
            logger.warning("opentelemetry-exporter-otlp-proto-http not installed — Jaeger export disabled")
        except Exception as e:
            logger.warning("Failed to configure OTLP exporter: %s", e)

        try:
            trace.set_tracer_provider(provider)
        except Exception:
            # Provider already set (e.g. by Langfuse) — use existing
            logger.debug("TracerProvider already set, using existing one")

        self._tracer = trace.get_tracer("infer-route")
        self._propagator = TraceContextTextMapPropagator()

        # Now initialize the LLM backend (Langfuse/LangSmith)
        self._backend = backend or create_tracing_backend()

    @staticmethod
    def extract_traceparent(headers: dict) -> dict | None:
        """Extract W3C traceparent from incoming HTTP headers."""
        traceparent = headers.get("traceparent") or headers.get("Traceparent")
        if not traceparent:
            return None
        parts = traceparent.strip().split("-")
        if len(parts) < 4:
            return None
        return {
            "version": parts[0],
            "trace_id": parts[1],
            "parent_id": parts[2],
            "trace_flags": parts[3],
        }

    @staticmethod
    def inject_traceparent(headers: dict, trace_id: str, span_id: str):
        """Inject W3C traceparent into outgoing HTTP headers."""
        headers["traceparent"] = f"00-{trace_id}-{span_id}-01"

    def start_span(
        self,
        name: str,
        trace_id: str = None,
        parent_id: str = "",
        attributes: dict = None,
    ) -> LLMTraceSpan:
        # Create the LLM backend span (Langfuse/LangSmith)
        span = self._backend.start_span(
            name=name,
            trace_id=trace_id,
            parent_id=parent_id,
            attributes=attributes,
        )

        # Create a real OpenTelemetry span for Jaeger export
        # Build context from incoming traceparent if available
        otel_span = None
        try:
            if trace_id:
                # W3C spec: parent span ID of all zeros is invalid
                # If parent_id is all zeros or empty, treat as root span with the given trace_id
                valid_parent = parent_id and parent_id != "0" * 16 and all(c in "0123456789abcdef" for c in parent_id)
                if valid_parent:
                    carrier = {"traceparent": f"00-{trace_id}-{parent_id}-01"}
                    ctx = self._propagator.extract(carrier)
                    otel_span = self._tracer.start_span(name, context=ctx, kind=SpanKind.SERVER)
                else:
                    # Root span — OTel will generate a new trace ID
                    # We'll override it with the incoming trace_id after creation
                    otel_span = self._tracer.start_span(name, kind=SpanKind.SERVER)
            else:
                otel_span = self._tracer.start_span(name, kind=SpanKind.SERVER)

            if attributes:
                for key, value in attributes.items():
                    otel_span.set_attribute(key, value)

            # Use OTel-generated span_id, but KEEP the incoming trace_id
            # OTel may generate a new trace_id when parent is invalid (all zeros)
            # We must preserve the trace_id from the incoming traceparent for correlation
            otel_ctx = otel_span.get_span_context()
            span.span_id = f"{otel_ctx.span_id:016x}"

            # If OTel generated a different trace_id (happens when parent_id was invalid),
            # override the OTel span's trace context to use the incoming trace_id
            otel_trace_id = f"{otel_ctx.trace_id:032x}"
            if trace_id and otel_trace_id != trace_id:
                from opentelemetry.trace import SpanContext, TraceFlags
                new_ctx = SpanContext(
                    trace_id=int(trace_id, 16),
                    span_id=otel_ctx.span_id,
                    is_remote=otel_ctx.is_remote,
                    trace_flags=otel_ctx.trace_flags,
                    trace_state=otel_ctx.trace_state,
                )
                otel_span._span_context = new_ctx
                logger.debug("Overrode OTel trace_id: %s → %s", otel_trace_id, trace_id)
        except Exception as e:
            logger.debug("OTel span creation failed (non-fatal): %s", e)

        self._otel_spans[span.span_id] = otel_span
        self._active_spans[span.span_id] = span
        logger.debug(
            "Span started: trace=%s span=%s name=%s",
            span.trace_id, span.span_id, name,
        )
        return span

    def end_span(self, span: LLMTraceSpan):
        self._backend.end_span(span)

        # End the OTel span for Jaeger
        otel_span = self._otel_spans.pop(span.span_id, None)
        if otel_span:
            try:
                otel_span.set_attribute("latency_ms", span.latency_ms)
                otel_span.set_attribute("provider", span.provider or "")
                otel_span.set_attribute("model", span.model or "")
                otel_span.set_attribute("input_tokens", span.input_tokens)
                otel_span.set_attribute("output_tokens", span.output_tokens)
                otel_span.set_attribute("cache_hit", span.cache_hit)
                otel_span.set_attribute("routing_decision", span.routing_decision or "")
                otel_span.set_attribute("complexity", span.complexity or "")
                otel_span.set_status(Status(StatusCode.OK))
                otel_span.end()
            except Exception as e:
                logger.debug("OTel span end failed (non-fatal): %s", e)

        self._active_spans.pop(span.span_id, None)
        logger.debug(
            "Span ended: trace=%s name=%s duration=%.1fms",
            span.trace_id, span.name, span.latency_ms,
        )

    def add_event(self, span: LLMTraceSpan, name: str, attributes: dict = None):
        self._backend.add_event(span, name, attributes)
        otel_span = self._otel_spans.get(span.span_id)
        if otel_span:
            try:
                otel_span.add_event(name, attributes or {})
            except Exception:
                pass

    def set_llm_attributes(self, span: LLMTraceSpan, **kwargs):
        self._backend.set_llm_attributes(span, **kwargs)
        otel_span = self._otel_spans.get(span.span_id)
        if otel_span:
            try:
                for key, value in kwargs.items():
                    if value is not None:
                        otel_span.set_attribute(f"llm.{key}", str(value))
            except Exception:
                pass

    def get_backend(self) -> TracingBackend:
        return self._backend

    def flush(self):
        self._backend.flush()

    def clear(self):
        self._active_spans.clear()
        self._otel_spans.clear()
